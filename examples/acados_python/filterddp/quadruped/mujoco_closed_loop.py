#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

"""
Closed-loop evaluation in MuJoCo of the Quadruped-PyMPC controller (installed, commit 6dcbb35) with its
NMPC solver swapped for FILTERDDP or acados SQP through pympc_adapter.py. The loop is that of
Quadruped-PyMPC simulation/simulation.py (gym-quadruped QuadrupedEnv, physics at 500 Hz, MPC at 100 Hz,
whole-body interface, torque limits), without rendering, with a fixed forward velocity command and a
random external wrench on the base as in Turrisi et al., IROS 2024: every `wrench_period` seconds
after `wrench_start`, force and torque components are redrawn uniformly in [-A, A] (N, N m).

An episode fails when gym-quadruped terminates it (ground contact of some non-foot bodies, or the robot
leaves the terrain), or when roll or pitch exceeds 1 rad or the base drops below 0.12 m
(closed_loop_core.TILT_LIMIT, HEIGHT_LIMIT; fall_cause 1, 2, 3); gym-quadruped alone lets a robot on
its back flail for seconds. Tracking cost per episode: time mean of |v_xy - v_ref|^2 + (w_z - w_z,ref)^2 +
(z - z_ref)^2 + roll^2 + pitch^2 over the part that was run.

Controllers:  STOCK                   Quadruped-PyMPC's own solver (SQP, 1 iteration, LINEAR_LS, ERK)
              <SOLVER>:<max_iter>     FILTERDDP_GN or SQP_GN from quadruped_ocp.py, capped at max_iter
                                      (<= 300, the build cap); the result is used whatever the status
              <SOLVER>:<max_iter>:<variant>
                                      FILTERDDP_GN with option flags, e.g. vg+ws (quadruped_ocp.FILTERDDP_FLAGS:
                                      vg value-gradient stationarity, ws shifted-policy warm start)

Every episode runs in a fresh worker process. Results go to <output_dir>/<tag>.txt, one line per
episode; --log_episodes k also writes every MPC call of the first k episodes to <output_dir>/calls/.

    mujoco_closed_loop.py --controllers STOCK SQP_GN:3 FILTERDDP_GN:15 --episodes 25 --wrench 10
                          [--seconds 10] [--speed 0.5] [--scene flat] [--workers 8] [--log_episodes 1]
                          [--tag name]
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from closed_loop_core import BUILD_MAX_ITER, TOL, parse_controller, run_episode, setup  # noqa: E402


def run_task(task):
    """One episode in a fresh worker process (the pool replaces workers after every task), so that no
    solver, controller or simulator state carries over between episodes."""
    spec, ep, args = task
    ctx = setup(spec, args, env_lock_path=os.path.join(HERE, '.quadruped_env.lock'))
    record = ep < args.log_episodes
    result, failed = run_episode(ctx, ep, args, record=record)
    obj = ctx.obj
    if record and obj.instances:
        os.makedirs(os.path.join(HERE, args.output_dir, 'calls'), exist_ok=True)
        path = os.path.join(HERE, args.output_dir, 'calls', f"{args.tag}_{spec.replace(':', '_')}_ep{ep}.npz")
        np.savez_compressed(path, **{k: np.array([d[k] for d in obj.instances]) for k in obj.instances[0]},
                            failed=failed, t_end=result['t_end'], seed=args.seed + ep, wrench=args.wrench)
    ctx.env.close()
    return [result]


FIELDS = ['controller', 'episode', 'success', 't_end', 'cost', 'rms_v', 'rms_wz', 'rms_z', 'rms_rp', 'n_calls',
          'status0', 'status2', 'status_other', 'iters_mean', 'iters_max', 'ms_mean', 'ms_p99', 'ms_max',
          'over_budget', 'wall_s', 'max_tilt', 'min_z', 'fall_cause']


def acados_revision():
    try:
        return subprocess.run(['git', '-C', os.environ.get('ACADOS_SOURCE_DIR', '.'), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown revision'


def fmt(v):
    return f'{v:.5g}' if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controllers', nargs='+', default=['STOCK', 'SQP_GN:1', 'FILTERDDP_GN:5'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--speed', type=float, default=0.5, help='forward velocity command (m/s)')
    parser.add_argument('--friction', type=float, default=0.8)
    parser.add_argument('--scene', default='flat')
    parser.add_argument('--gait', default='trot')
    parser.add_argument('--wrench', type=float, default=5.0, help='disturbance amplitude A (N and N m)')
    parser.add_argument('--wrench_period', type=float, default=2.0)
    parser.add_argument('--wrench_start', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--log_episodes', type=int, default=0, help='record every MPC call of the first k episodes')
    parser.add_argument('--tag', default='closed_loop')
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()

    sys.path.insert(0, HERE)
    os.chdir(HERE)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    # build Quadruped-PyMPC's own solver once, before the workers load it
    import quadruped_pympc.config as cfg
    from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal
    Acados_NMPC_Nominal()
    # load (and if the stored hash differs, rebuild) our solvers once here, not concurrently in the workers
    from acados_template import AcadosOcpSolver
    from quadruped_ocp import FILTERDDP_FLAGS, FILTERDDP_OPTIONS, setup_ocp
    for name in sorted({parse_controller(s)[0] for s in args.controllers} - {'STOCK'}):
        AcadosOcpSolver(setup_ocp(name, TOL, BUILD_MAX_ITER), verbose=False, generate=False, build=False)

    tasks = []
    for spec in args.controllers:
        name, cap, _ = parse_controller(spec)
        assert name == 'STOCK' or cap <= BUILD_MAX_ITER, f'cap {cap} exceeds the build cap {BUILD_MAX_ITER}'
        tasks += [(spec, ep, args) for ep in range(args.episodes)]

    t0 = time.time()
    rows = []
    with mp.get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_task, tasks):
            for r in res:
                rows.append(r)
                print(' '.join(fmt(r.get(k, '')) for k in FIELDS), flush=True)
    rows.sort(key=lambda r: (args.controllers.index(r['controller']), r['episode']))

    os.makedirs(os.path.join(HERE, args.output_dir), exist_ok=True)
    out = os.path.join(HERE, args.output_dir, f'{args.tag}.txt')
    with open(out, 'w') as f:
        f.write(f'# robot {cfg.robot} scene {args.scene} gait {args.gait} speed {args.speed} friction {args.friction} '
                f'wrench +-{args.wrench} every {args.wrench_period}s from {args.wrench_start}s, {args.seconds}s episodes, '
                f'seed {args.seed}, {args.workers} parallel workers (timings are indicative), '
                f'acados {acados_revision()}, FILTERDDP options {FILTERDDP_OPTIONS}, '
                f'variant flags {FILTERDDP_FLAGS}\n')
        f.write(' '.join(FIELDS) + '\n')
        for r in rows:
            f.write(' '.join(fmt(r.get(k, '')) for k in FIELDS) + '\n')

    print(f'-- {args.tag}: {len(rows)} episodes in {time.time() - t0:.0f} s, written to {out}')
    print('controller success cost_mean(success) rms_v rms_z iters_mean iters_max capped_calls other_status ms_mean ms_p99 over_10ms')
    for spec in args.controllers:
        sel = [r for r in rows if r['controller'] == spec]
        ok = [r for r in sel if r['success']]
        calls = sum(r['n_calls'] for r in sel)
        cm = np.mean([r['cost'] for r in ok]) if ok else float('nan')
        print(f"{spec:16s} {len(ok)}/{len(sel)} {cm:.4f} {np.mean([r['rms_v'] for r in sel]):.3f} "
              f"{np.mean([r['rms_z'] for r in sel]):.4f} {np.average([r['iters_mean'] for r in sel], weights=[r['n_calls'] for r in sel]):.2f} "
              f"{max(r['iters_max'] for r in sel)} {sum(r['status2'] for r in sel)}/{calls} {sum(r['status_other'] for r in sel)} "
              f"{np.average([r['ms_mean'] for r in sel], weights=[r['n_calls'] for r in sel]):.3f} "
              f"{max(r['ms_p99'] for r in sel):.3f} {sum(r['over_budget'] for r in sel)}")


if __name__ == '__main__':
    main()

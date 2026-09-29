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
Minimal example: build the Quadruped-PyMPC centroidal NMPC as an acados OCP (quadruped_model.py,
quadruped_ocp.py) and solve one MPC call recorded in closed loop (example_calls.npz) to convergence
with FILTERDDP_GN and with Quadruped-PyMPC's own solver settings (SQP_GN), from the same initial guess.

    python solve_quadruped_ocp.py [--call 0] [--tol 1e-6]
"""

import argparse
import time

import numpy as np
from acados_template import AcadosOcpSolver

from quadruped_model import GRAVITY, N_HORIZON, P_MASS, P_STANCE, friction_cone_bounds
from quadruped_ocp import apply_filterddp_options, set_call, setup_ocp

N = N_HORIZON
LEGS = ('FL', 'FR', 'RL', 'RR')


def solve(name, x0, P, tol):
    # generate = build = False: acados reuses a previous build if the OCP is unchanged, else builds
    solver = AcadosOcpSolver(setup_ocp(name, tol=tol), verbose=False, generate=False, build=False)
    if name == 'FILTERDDP_GN':
        apply_filterddp_options(solver)
    lh, uh = friction_cone_bounds(P[0, P_MASS]*GRAVITY)      # pyramid cone, 0 <= f_z <= m g
    for k in range(N):
        solver.constraints_set(k, 'lh', lh)
        solver.constraints_set(k, 'uh', uh)
    set_call(solver, name, x0, P)
    for k in range(N + 1):                                   # initial guess: hold the state, zero inputs
        solver.set(k, 'x', x0)
    t0 = time.perf_counter()
    status = solver.solve()
    ms = 1e3*(time.perf_counter() - t0)
    U = np.array([solver.get(k, 'u') for k in range(N)])
    return dict(status=status, iters=int(solver.get_stats('nlp_iter')), ms=ms, cost=solver.get_cost(), U=U)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--call', type=int, default=0, help='which of the recorded calls (0 to 9)')
    parser.add_argument('--tol', type=float, default=1e-6)
    args = parser.parse_args()

    calls = np.load('example_calls.npz')
    x0, P = calls['x0'][args.call], calls['P'][args.call]
    print(f"{calls['note']}\ncall {args.call}: t = {calls['t'][args.call]:.2f} s, "
          f"stance legs {[leg for leg, c in zip(LEGS, P[0, P_STANCE]) if c]}\n")

    results = {name: solve(name, x0, P, args.tol) for name in ('FILTERDDP_GN', 'SQP_GN')}
    print(f"{'solver':14s}{'status':>7s}{'iters':>7s}{'ms':>9s}{'objective':>16s}")
    for name, r in results.items():
        print(f"{name:14s}{r['status']:7d}{r['iters']:7d}{r['ms']:9.1f}{r['cost']:16.8e}")
    a, b = results['FILTERDDP_GN'], results['SQP_GN']
    print('\nfirst-stage ground reaction forces f_z (N), FILTERDDP_GN / SQP_GN:')
    for i, leg in enumerate(LEGS):
        print(f"  {leg}: {a['U'][0, 14 + 3*i]:8.2f} / {b['U'][0, 14 + 3*i]:8.2f}")
    print(f"max difference over all stages and inputs: {np.max(np.abs(a['U'] - b['U'])):.2e}")


if __name__ == '__main__':
    main()

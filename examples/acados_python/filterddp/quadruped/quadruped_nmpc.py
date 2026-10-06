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
Quadruped-PyMPC's nominal centroidal NMPC with FILTERDDP as its solver, in closed loop in MuJoCo: a Unitree
Go2 trots forward on flat ground (gym-quadruped). The OCP is Quadruped-PyMPC's own (single rigid body with
point feet, LINEAR_LS tracking cost, ERK dynamics, friction cone, N = 12 stages of 20 ms); only its solver
options change, from SQP (one Gauss-Newton iteration per call) to FILTERDDP with the Gauss-Newton Hessian,
capped at --max_iter iterations and warm started from the shifted policy of the previous solve, rolled out
in closed loop from the measured state (AcadosOcpSolver.warm_start_from_policy). The MPC
runs at 100 Hz, the physics at 500 Hz, and the whole-body layer (stance torques from the planned forces,
swing trajectories, friction compensation) is Quadruped-PyMPC's.

    quadruped_nmpc.py [--solver filterddp|sqp] [--max_iter 5] [--no_warm_start] [--wrench 5] [--render]

--wrench A pushes the base with a random wrench redrawn every 2 s from t = 1 s, each force and torque
component uniform in [-A, A] (N, N m), after the disturbance test of Turrisi et al. (IROS 2024).
--solver sqp keeps Quadruped-PyMPC's solver for comparison. --render opens MuJoCo's viewer with
Quadruped-PyMPC's overlays.
"""

import argparse
import copy
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6
TAU_LIMIT = 0.9          # fraction of the actuator torque range, as in Quadruped-PyMPC's simulation
RENDER_FREQ = 30.0       # Hz, wall clock
WRENCH_START, WRENCH_PERIOD = 1.0, 2.0     # s
TILT_LIMIT = 1.0         # rad, roll or pitch: a fall
HEIGHT_LIMIT = 0.12      # m, base height: a fall
LEGS = ('FL', 'FR', 'RL', 'RR')


def use_filterddp(max_iter):
    """Make Quadruped-PyMPC's NMPC build its OCP for FILTERDDP. The generated code goes to this directory."""
    import quadruped_pympc.config as config
    import quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal as nominal
    AcadosOcpSolver = nominal.AcadosOcpSolver

    def build(ocp, json_file=None, **kwargs):
        opts = ocp.solver_options
        opts.nlp_solver_type = 'FILTERDDP'
        opts.nlp_solver_max_iter = max_iter
        opts.qp_solver_cond_N = config.mpc_params['horizon']
        opts.levenberg_marquardt = 0.0      # an SQP setting FILTERDDP does not use
        opts.nlp_solver_tol_stat = opts.nlp_solver_tol_eq = opts.nlp_solver_tol_ineq = opts.nlp_solver_tol_comp = TOL
        ocp.code_export_directory = os.path.join(HERE, 'c_generated_code')
        return AcadosOcpSolver(ocp, json_file=os.path.join(HERE, 'quadruped_filterddp_ocp.json'), **kwargs)

    nominal.AcadosOcpSolver = build


class SolveLog:
    """Stands in for the controller's AcadosOcpSolver and records status, iterations and time of each solve. With
    warm_start, each solve starts from the previous solve's policy rolled out from the initial state the controller
    sets (lbx at stage 0)."""

    def __init__(self, solver, warm_start=False):
        self.solver = solver
        self.warm_start = warm_start
        self.x0 = None
        self.calls = []

    def set(self, stage, field, value):
        if stage == 0 and field == 'lbx':
            self.x0 = np.array(value, dtype=float).flatten()
        return self.solver.set(stage, field, value)

    def constraints_set(self, stage, field, value, *args, **kwargs):
        if stage == 0 and field == 'lbx':
            self.x0 = np.array(value, dtype=float).flatten()
        return self.solver.constraints_set(stage, field, value, *args, **kwargs)

    def solve(self):
        if self.warm_start and self.x0 is not None:
            self.solver.warm_start_from_policy(self.x0)   # no policy before the first solve or after a failed one
        status = self.solver.solve()
        self.calls.append((status, self.solver.get_stats('nlp_iter'), 1e3*self.solver.get_stats('time_tot')))
        return status

    def __getattr__(self, name):
        return getattr(self.solver, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--solver', default='filterddp', choices=['filterddp', 'sqp'])
    parser.add_argument('--max_iter', type=int, default=5, help='FILTERDDP iterations per MPC call')
    parser.add_argument('--no_warm_start', dest='warm_start', action='store_false')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--speed', type=float, default=0.5, help='forward velocity command (m/s)')
    parser.add_argument('--wrench', type=float, default=0.0, help='push amplitude A (N, N m), 0 = no pushes')
    parser.add_argument('--seed', type=int, default=0, help='seed of the pushes')
    parser.add_argument('--render', action='store_true', help="open MuJoCo's viewer (needs a display)")
    args = parser.parse_args()

    import mujoco
    import quadruped_pympc.config as cfg
    from gym_quadruped.quadruped_env import QuadrupedEnv
    from gym_quadruped.utils.mujoco.visual import render_vector
    from gym_quadruped.utils.quadruped_utils import LegsAttr
    from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
    if args.solver == 'filterddp':
        use_filterddp(args.max_iter)
    from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

    sim_dt = cfg.simulation_params['dt']
    env = QuadrupedEnv(robot=cfg.robot, scene='flat', sim_dt=sim_dt, ref_base_lin_vel=args.speed, ref_base_ang_vel=0.0,
                       ground_friction_coeff=0.8, base_vel_command_type='forward', state_obs_names=tuple())
    env.mjModel.opt.gravity[2] = -cfg.gravity_constant
    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))
    env.reset(random=False)
    wrapper = QuadrupedPyMPC_Wrapper(initial_feet_pos=env.feet_pos, legs_order=LEGS, feet_geom_id=env._feet_geom_id,
                                     quadrupedpympc_observables_names=('ref_feet_pos', 'nmpc_footholds', 'swing_time',
                                                                       'lift_off_positions'))
    controller = wrapper.srbd_controller_interface.controller
    log = SolveLog(controller.acados_ocp_solver, args.solver == 'filterddp' and args.warm_start)
    controller.acados_ocp_solver = log
    tau_limits = {leg: TAU_LIMIT*env.mjModel.actuator_ctrlrange[env.legs_tau_idx[leg]] for leg in LEGS}
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in LEGS])
    if args.render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    last_render = time.time()
    feet_traj_geom_ids, feet_grf_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    base_body = int(env.mjModel.jnt_bodyid[np.where(env.mjModel.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]])
    rng = np.random.default_rng(args.seed)
    next_push = WRENCH_START

    # Quadruped-PyMPC's simulation loop (simulation/simulation.py), with the pushes
    errors, t_fall = [], None
    for step in range(int(round(args.seconds/sim_dt))):
        if args.wrench > 0 and step*sim_dt >= next_push - 1e-12:
            env.mjData.xfrc_applied[base_body] = rng.uniform(-args.wrench, args.wrench, 6)
            next_push += WRENCH_PERIOD
        base_lin_vel = env.base_lin_vel(frame='world')
        base_ang_vel = env.base_ang_vel(frame='base')
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
        tau = wrapper.compute_actions(
            copy.deepcopy(env.com), copy.deepcopy(env.base_pos), base_lin_vel, env.base_ori_euler_xyz, base_ang_vel,
            env.feet_pos(frame='world'), env.hip_positions(frame='world'),
            LegsAttr(**{leg: env.legs_qvel_idx[leg] for leg in LEGS}), None, LEGS, sim_dt, ref_base_lin_vel,
            ref_base_ang_vel, env.step_num, env.mjData.qpos, env.mjData.qvel,
            env.feet_jacobians(frame='world', return_rot_jac=False),
            env.feet_jacobians_dot(frame='world', return_rot_jac=False), env.feet_vel(frame='world'),
            env.legs_qfrc_passive, env.legs_qfrc_bias, env.legs_mass_matrix, env.legs_qpos_idx, env.legs_qvel_idx,
            tau, env.get_base_inertia().flatten() if cfg.simulation_params['use_inertia_recomputation']
            else cfg.inertia.flatten(), env.mjData.contact)
        action = np.zeros(env.mjModel.nu)
        for leg in LEGS:
            tau[leg] = np.clip(tau[leg], tau_limits[leg][:, 0], tau_limits[leg][:, 1])
            action[env.legs_tau_idx[leg]] = tau[leg]
        _, _, terminated, _, _ = env.step(action=action)
        errors.append(np.concatenate((np.asarray(base_lin_vel[:2]) - np.asarray(ref_base_lin_vel[:2]),
                                      [base_ang_vel[2] - ref_base_ang_vel[2]])))
        if args.render and time.time() - last_render > 1.0/RENDER_FREQ:
            # Quadruped-PyMPC's overlays: planned swing trajectories (red), footholds (green), contact forces
            obs, stc = wrapper.get_obs(), wrapper.wb_interface.stc
            feet_traj_geom_ids = plot_swing_mujoco(
                viewer=env.viewer, swing_traj_controller=stc, swing_period=stc.swing_period,
                swing_time=LegsAttr(*obs['swing_time']), lift_off_positions=obs['lift_off_positions'],
                nmpc_footholds=obs['nmpc_footholds'], ref_feet_pos=obs['ref_feet_pos'],
                early_stance_detector=wrapper.wb_interface.esd, geom_ids=feet_traj_geom_ids)
            _, _, feet_grf = env.feet_contact_state(ground_reaction_forces=True)
            feet_pos = env.feet_pos(frame='world')
            for leg in LEGS:
                feet_grf_geom_ids[leg] = render_vector(
                    env.viewer, vector=feet_grf[leg], pos=feet_pos[leg], scale=np.linalg.norm(feet_grf[leg])*0.005,
                    color=np.array([0, 1, 0, 0.5]), geom_id=feet_grf_geom_ids[leg])
            env.render()
            last_render = time.time()
        if terminated or np.max(np.abs(env.base_ori_euler_xyz[:2])) > TILT_LIMIT or env.base_pos[2] < HEIGHT_LIMIT:
            t_fall = (step + 1)*sim_dt
            break
    env.close()

    e = np.array(errors)
    calls = np.array(log.calls)
    status, iters, ms = calls[:, 0].astype(int), calls[:, 1], calls[:, 2]
    name = f'FILTERDDP, {args.max_iter} iterations' + (', warm started' if args.warm_start else '') \
        if args.solver == 'filterddp' else 'SQP, 1 iteration (Quadruped-PyMPC)'
    print(f'{name}: {"fell at %.2f s" % t_fall if t_fall else "walked %.1f s" % args.seconds}, '
          f'RMS velocity error {np.sqrt(np.mean(np.sum(e[:, :2]**2, axis=1))):.3f} m/s, '
          f'RMS yaw-rate error {np.sqrt(np.mean(e[:, 2]**2)):.3f} rad/s')
    print(f'{len(calls)} MPC calls: {np.sum(status == 0)} converged, {np.sum(status == 2)} at the iteration cap, '
          f'{np.sum((status != 0) & (status != 2))} other; iterations mean {iters.mean():.2f}; '
          f'solve time mean {ms.mean():.2f} ms, p99 {np.percentile(ms, 99):.2f} ms, max {ms.max():.2f} ms')


if __name__ == '__main__':
    main()

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
One closed-loop episode of Quadruped-PyMPC in MuJoCo with a chosen NMPC solver, shared by
mujoco_closed_loop.py (evaluation) and visualize_episode.py (videos). See mujoco_closed_loop.py for
the protocol. `run_episode` calls `on_step(ctx, step, t, wrench, terminated)` after every physics step.
"""

import copy
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_MAX_ITER = 300
TOL = 1e-6
# Fall test in addition to gym-quadruped's termination, which only reacts to ground contact of some
# bodies: a robot on its back can flail for seconds without terminating.
TILT_LIMIT = 1.0        # rad, roll or pitch
HEIGHT_LIMIT = 0.12     # m, base height (nominal 0.28)
FALL_CAUSES = {0: 'none', 1: 'termination', 2: 'tilt', 3: 'height'}


def parse_controller(spec):
    """'STOCK' or '<SOLVER>:<max_iter>[:<variant>]', variant flags from quadruped_ocp.FILTERDDP_FLAGS."""
    if spec == 'STOCK':
        return 'STOCK', None, None
    name, cap, *variant = spec.split(':')
    return name, int(cap), (variant[0] if variant else None)


def patch_stock_solver_reuse():
    """Make Quadruped-PyMPC's controller load its prebuilt solver instead of regenerating it (the
    stored hash is still checked by acados)."""
    import quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal as nm
    original = nm.AcadosOcpSolver
    if getattr(original, '_reuse_patched', False):
        return

    def reuse(ocp, json_file=None, **kwargs):
        kwargs.setdefault('generate', False)
        kwargs.setdefault('build', False)
        kwargs['verbose'] = False
        return original(ocp, json_file=json_file, **kwargs)
    reuse._reuse_patched = True
    nm.AcadosOcpSolver = reuse


def make_solver_object(name, cap, variant, controller, cfg, record):
    from pympc_adapter import SolverAdapter, StockRecorder
    if name == 'STOCK':
        return StockRecorder(controller.acados_ocp_solver, record_instances=record)
    from acados_template import AcadosOcpSolver
    from quadruped_ocp import apply_filterddp_options, setup_ocp
    # generate = build = False: acados checks the stored hash and rebuilds only if the OCP changed
    solver = AcadosOcpSolver(setup_ocp(name, TOL, BUILD_MAX_ITER), verbose=False, generate=False, build=False)
    if name.startswith('FILTERDDP'):
        apply_filterddp_options(solver, variant)
    else:
        assert variant is None, f'variant {variant} is only defined for FILTERDDP'
    return SolverAdapter(name, solver, f_max=cfg.mass*cfg.gravity_constant, max_iter=cap, record_instances=record)


def install(obj, controller):
    """compute_control's reset() replaces the solver object with a fresh stock solver; re-attach."""
    from pympc_adapter import StockRecorder
    if isinstance(obj, StockRecorder):
        obj.solver = controller.acados_ocp_solver
    else:
        obj.reset()
    controller.acados_ocp_solver = obj


def setup(spec, args, env_lock_path=None):
    os.chdir(HERE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import mujoco
    from gym_quadruped.quadruped_env import QuadrupedEnv
    from gym_quadruped.utils.quadruped_utils import LegsAttr
    import quadruped_pympc.config as cfg
    cfg.simulation_params['scene'] = args.scene
    cfg.simulation_params['gait'] = args.gait
    patch_stock_solver_reuse()
    from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

    name, cap, variant = parse_controller(spec)
    sim_dt = cfg.simulation_params['dt']

    def make_env():
        return QuadrupedEnv(robot=cfg.robot, scene=args.scene, sim_dt=sim_dt, ref_base_lin_vel=args.speed,
                            ref_base_ang_vel=0.0, ground_friction_coeff=args.friction,
                            base_vel_command_type='forward', state_obs_names=tuple())
    if env_lock_path is None:
        env = make_env()
    else:
        # QuadrupedEnv writes <robot>-<scene>.xml into the gym_quadruped package and reads it back, so
        # concurrent constructions in several processes race on that file
        import fcntl
        with open(env_lock_path, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            env = make_env()
            fcntl.flock(lock, fcntl.LOCK_UN)
    env.mjModel.opt.gravity[2] = -cfg.gravity_constant
    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))
    env.reset(random=False)
    free_joint = int(np.where(env.mjModel.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0])
    legs_order = ['FL', 'FR', 'RL', 'RR']
    wrapper = QuadrupedPyMPC_Wrapper(initial_feet_pos=env.feet_pos, legs_order=tuple(legs_order),
                                     feet_geom_id=env._feet_geom_id,
                                     quadrupedpympc_observables_names=('ref_base_height', 'nmpc_GRFs', 'nmpc_footholds'))
    controller = wrapper.srbd_controller_interface.controller
    return SimpleNamespace(
        spec=spec, cfg=cfg, env=env, wrapper=wrapper, controller=controller, legs_order=legs_order,
        obj=make_solver_object(name, cap, variant, controller, cfg, record=False), sim_dt=sim_dt,
        base_body=int(env.mjModel.jnt_bodyid[free_joint]), LegsAttr=LegsAttr,
        tau_limits=LegsAttr(*[env.mjModel.actuator_ctrlrange[env.legs_tau_idx[leg]]*0.9 for leg in legs_order]))


def run_episode(ctx, ep, args, on_step=None, record=False):
    env, wrapper, cfg, obj, LegsAttr = ctx.env, ctx.wrapper, ctx.cfg, ctx.obj, ctx.LegsAttr
    from pympc_adapter import summarize_calls
    rng = np.random.default_rng(args.seed + ep)
    env.reset(random=False)
    wrapper.reset(initial_feet_pos=env.feet_pos(frame='world'))
    obj.record_instances = record
    obj.calls, obj.instances = [], []
    install(obj, ctx.controller)
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    wrench = np.zeros(6)
    next_draw = args.wrench_start
    cost_terms = []
    failed, t_end, cause = False, args.seconds, 0
    max_tilt, min_z = 0.0, np.inf
    n_steps = int(round(args.seconds/ctx.sim_dt))
    wall0 = time.perf_counter()
    for step in range(n_steps):
        t = step*ctx.sim_dt
        if args.wrench > 0 and t >= next_draw - 1e-12:
            wrench = rng.uniform(-args.wrench, args.wrench, 6)
            next_draw += args.wrench_period
        env.mjData.xfrc_applied[ctx.base_body] = wrench

        feet_pos = env.feet_pos(frame='world')
        feet_vel = env.feet_vel(frame='world')
        hip_pos = env.hip_positions(frame='world')
        base_lin_vel = env.base_lin_vel(frame='world')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        base_pos = copy.deepcopy(env.base_pos)
        com_pos = copy.deepcopy(env.com)
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
        if cfg.simulation_params['use_inertia_recomputation']:
            inertia = env.get_base_inertia().flatten()
        else:
            inertia = cfg.inertia.flatten()
        legs_qvel_idx = env.legs_qvel_idx
        legs_qpos_idx = env.legs_qpos_idx
        joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR, RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)
        tau = wrapper.compute_actions(
            com_pos, base_pos, base_lin_vel, base_ori_euler_xyz, base_ang_vel, feet_pos, hip_pos, joints_pos,
            None, ctx.legs_order, ctx.sim_dt, ref_base_lin_vel, ref_base_ang_vel, env.step_num, env.mjData.qpos,
            env.mjData.qvel, env.feet_jacobians(frame='world', return_rot_jac=False),
            env.feet_jacobians_dot(frame='world', return_rot_jac=False), feet_vel, env.legs_qfrc_passive,
            env.legs_qfrc_bias, env.legs_mass_matrix, legs_qpos_idx, legs_qvel_idx, tau, inertia,
            env.mjData.contact)
        for leg in ctx.legs_order:
            tau[leg] = np.clip(tau[leg], ctx.tau_limits[leg][:, 0], ctx.tau_limits[leg][:, 1])
        action = np.zeros(env.mjModel.nu)
        for leg in ctx.legs_order:
            action[env.legs_tau_idx[leg]] = tau[leg]
        _, _, is_terminated, _, _ = env.step(action=action)

        z_ref = wrapper.get_obs()['ref_base_height']
        dv = np.asarray(base_lin_vel[:2]) - np.asarray(ref_base_lin_vel[:2])
        dwz = float(base_ang_vel[2] - np.asarray(ref_base_ang_vel).flatten()[-1])
        cost_terms.append((dv @ dv, dwz**2, float(base_pos[2] - z_ref)**2,
                           base_ori_euler_xyz[0]**2 + base_ori_euler_xyz[1]**2))
        rpy, z = env.base_ori_euler_xyz, float(env.base_pos[2])
        tilt = max(abs(float(rpy[0])), abs(float(rpy[1])))
        max_tilt, min_z = max(max_tilt, tilt), min(min_z, z)
        cause = 1 if is_terminated else 2 if tilt > TILT_LIMIT else 3 if z < HEIGHT_LIMIT else 0
        if on_step is not None:
            on_step(ctx, step, t + ctx.sim_dt, wrench, cause != 0)
        if cause:
            failed, t_end = True, (step + 1)*ctx.sim_dt
            break
    c = np.array(cost_terms)
    result = dict(controller=ctx.spec, episode=ep, success=int(not failed), t_end=t_end,
                  cost=float(c.sum(axis=1).mean()), rms_v=float(np.sqrt(c[:, 0].mean())),
                  rms_wz=float(np.sqrt(c[:, 1].mean())), rms_z=float(np.sqrt(c[:, 2].mean())),
                  rms_rp=float(np.sqrt(c[:, 3].mean())), wall_s=time.perf_counter() - wall0,
                  max_tilt=max_tilt, min_z=min_z, fall_cause=cause,
                  **summarize_calls(obj.calls))
    obj.record_instances = False
    return result, failed

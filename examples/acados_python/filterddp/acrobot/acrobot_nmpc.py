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
Closed-loop NMPC of the acrobot swing-up with FILTERDDP. The plant is integrated at 1 kHz with Gaussian
torque noise. The controller is delayed by one OCP step: at every 50 ms step it measures x(t_k), applies
the affine policy of the active solution, predicts x(t_{k+1}) with the OCP model and solves the OCP from
the prediction, warm started from the shifted policy of the previous solve rolled out from the prediction
(warm_start_from_policy); the new solution becomes
active at t_{k+1}, so each solve has one step (50 ms) of computation time. Between OCP steps the policy
u = u0 + k0 + K0 (x - x0(t)) is applied at 100 Hz, with x0(t) the planned initial state propagated with
the nominal model under u0.

The initial swing-up solve from rest runs to convergence before the loop starts. The warm-started solves
are capped at --max_iter iterations so that each one finishes within the 50 ms step: the default of 20
iterations takes at most about 20 ms on a desktop CPU (about 1 ms per iteration in the worst case).

    acrobot_nmpc.py [--control_limits] [--noise 0.5] [--max_iter 20] [--plot] [--save acrobot_nmpc.gif]
"""

import argparse
import sys
import time

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from acrobot_model import DEFAULT_PARAMETERS, DT, NX, acrobot_explicit, acrobot_rk4, export_acrobot_rk4_model

N_HORIZON = 100
H_SIM = 1e-3
T_SIM = 8.0
POLICY_RATE = 100.0
CONTROL_LIMIT = 4.0
TOL = 1e-5
MAX_ITER_INITIAL = 1000
X_TARGET = np.array([np.pi, 0.0, 0.0, 0.0])


def setup_ocp(control_limit: float | None) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = export_acrobot_rk4_model()
    ocp.parameter_values = DEFAULT_PARAMETERS
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.constraints.x0 = np.zeros(NX)
    if control_limit is not None:
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([-control_limit])
        ocp.constraints.ubu = np.array([control_limit])

    opts = ocp.solver_options
    opts.N_horizon = N_HORIZON
    opts.tf = N_HORIZON*DT
    opts.integrator_type = 'DISCRETE'
    opts.nlp_solver_type = 'FILTERDDP'
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    opts.qp_solver_cond_N = N_HORIZON
    opts.hessian_approx = 'EXACT'
    opts.regularize_method = 'NO_REGULARIZE'
    opts.globalization = 'FIXED_STEP'
    # the memory is sized for the initial solve; the warm-started solves lower the cap at runtime
    opts.nlp_solver_max_iter = MAX_ITER_INITIAL
    opts.nlp_solver_tol_stat = TOL
    opts.nlp_solver_tol_eq = TOL
    opts.nlp_solver_tol_ineq = TOL
    opts.nlp_solver_tol_comp = TOL
    opts.print_level = 0
    opts.cost_scaling = np.ones(N_HORIZON + 1)  # the stage cost carries its time step
    return ocp


def plant_step(p):
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', 1)
    k1 = acrobot_explicit(p, x, u)
    k2 = acrobot_explicit(p, x + 0.5*H_SIM*k1, u)
    k3 = acrobot_explicit(p, x + 0.5*H_SIM*k2, u)
    k4 = acrobot_explicit(p, x + H_SIM*k3, u)
    return ca.Function('plant', [x, u], [x + (H_SIM/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)])


def ocp_step(p):
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', 1)
    return ca.Function('model', [x, u], [acrobot_rk4(p, x, u)])


class Policy:
    """Affine feedback policy of stage 0 of the current solution."""

    def __init__(self, solver, limit):
        self.u0 = solver.get(0, 'u')
        self.x0 = solver.get(0, 'x')
        self.K = solver.get_from_qp_in(0, 'K')
        self.k = solver.get_from_qp_in(0, 'k').reshape(-1)
        self.limit = limit

    def clip(self, u):
        return np.clip(u, -self.limit, self.limit) if self.limit is not None else u

    def __call__(self, x, x_ref):
        return self.clip(self.u0 + self.k + self.K @ (x - x_ref))


def solve(solver, x0, warm=False):
    """With warm, the initial guess is the previous solve's policy, shifted and rolled out from x0 (if the previous
    solve left one: it converged, reached the iteration cap or timed out)."""
    solver.set(0, 'lbx', x0)
    solver.set(0, 'ubx', x0)
    t0 = time.perf_counter()
    if warm:
        solver.warm_start_from_policy(x0)
    status = solver.solve()
    return status, solver.get_stats('nlp_iter'), 1e3*(time.perf_counter() - t0)


def visualize(t, x_log, u_log, solves, limit, save=None):
    """Animation of the acrobot next to its states, torque and per-solve iterations and computation time."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    l1, l2 = DEFAULT_PARAMETERS[2], DEFAULT_PARAMETERS[6]
    q1, q2 = x_log[:, 0], x_log[:, 1]
    elbow = np.stack([l1*np.sin(q1), -l1*np.cos(q1)], axis=1)
    tip = elbow + np.stack([l2*np.sin(q1 + q2), -l2*np.cos(q1 + q2)], axis=1)

    fig = plt.figure(figsize=(13, 7), layout='constrained')
    grid = fig.add_gridspec(4, 2, width_ratios=[1, 1.4])
    ax_anim = fig.add_subplot(grid[:, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    reach = 1.1*(l1 + l2)
    ax_anim.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect='equal', title='acrobot')
    ax_anim.axhline(0.0, color='0.85', lw=0.8)
    ax_anim.plot(0.0, l1 + l2, 'x', color='tab:green', ms=10, label='target')
    links, = ax_anim.plot([], [], 'o-', lw=4, ms=8, color='tab:blue')
    trace, = ax_anim.plot([], [], '-', lw=0.8, color='tab:orange', alpha=0.6)
    clock = ax_anim.text(0.02, 0.97, '', transform=ax_anim.transAxes, va='top')
    ax_anim.legend(loc='lower right')

    axes[0].plot(t, q1, label='$q_1$')
    axes[0].plot(t, q2, label='$q_2$')
    axes[0].axhline(np.pi, ls='--', color='0.5', lw=0.8)
    axes[0].axhline(0.0, ls='--', color='0.5', lw=0.8)
    axes[0].set_ylabel('angle [rad]')
    axes[0].legend(loc='right')
    axes[1].plot(t, x_log[:, 2], label=r'$\dot q_1$')
    axes[1].plot(t, x_log[:, 3], label=r'$\dot q_2$')
    axes[1].set_ylabel('velocity [rad/s]')
    axes[1].legend(loc='right')
    axes[2].plot(t, u_log, color='tab:red', lw=0.8)
    if limit is not None:
        for bound in (-limit, limit):
            axes[2].axhline(bound, ls='--', color='0.5', lw=0.8)
    axes[2].set_ylabel('torque [Nm]')
    t_solve, iters, ms = solves[:, 0], solves[:, 1], solves[:, 2]
    axes[3].bar(t_solve, ms, width=0.8*DT, align='edge', color='tab:purple', label='solve time')
    axes[3].axhline(1e3*DT, ls='--', color='tab:red', lw=0.8, label='budget')
    axes[3].set_ylabel('solve time [ms]')
    axes[3].set_xlabel('time [s]')
    axes[3].legend(loc='upper right')
    ax_it = axes[3].twinx()
    ax_it.plot(t_solve + 0.4*DT, iters, '.', color='k', ms=3)
    ax_it.set_ylabel('iterations')
    cursors = [ax.axvline(0.0, color='k', lw=0.8) for ax in axes]

    stride = int(round(0.02/H_SIM))  # 50 frames per second
    frames = range(0, len(t), stride)

    def update(i):
        links.set_data([0.0, elbow[i, 0], tip[i, 0]], [0.0, elbow[i, 1], tip[i, 1]])
        trace.set_data(tip[max(0, i - 500):i + 1, 0], tip[max(0, i - 500):i + 1, 1])
        clock.set_text(f't = {t[i]:.2f} s')
        for cursor in cursors:
            cursor.set_xdata([t[i], t[i]])
        return links, trace, clock, *cursors

    anim = FuncAnimation(fig, update, frames=frames, interval=1e3*stride*H_SIM, blit=True)
    if save:
        anim.save(save, fps=round(1.0/(stride*H_SIM)))
        print(f'saved the animation to {save}')
    else:
        plt.show()
    return anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_limits', action='store_true', help=f'limit the torque to +-{CONTROL_LIMIT} Nm')
    parser.add_argument('--noise', type=float, default=0.5, help='standard deviation of the torque noise in Nm')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=20,
                        help='iteration cap of the warm-started solves, keeping each within the 50 ms OCP step')
    parser.add_argument('--plot', action='store_true', help='show an animation with the closed-loop trajectories')
    parser.add_argument('--save', default=None, help='save the animation to this file (.gif or .mp4) instead')
    args = parser.parse_args()
    limit = CONTROL_LIMIT if args.control_limits else None
    if not 1 <= args.max_iter <= MAX_ITER_INITIAL:
        parser.error(f'--max_iter must be in [1, {MAX_ITER_INITIAL}]')

    ocp = setup_ocp(limit)
    solver = AcadosOcpSolver(ocp, json_file='acrobot_nmpc_ocp.json', verbose=False)
    plant = plant_step(DEFAULT_PARAMETERS)
    model = ocp_step(DEFAULT_PARAMETERS)
    rng = np.random.default_rng(args.seed)

    # initial solve to optimality from rest, then capped warm starts from the shifted policy
    x = np.zeros(NX)
    for stage in range(N_HORIZON + 1):
        solver.set(stage, 'x', np.zeros(NX))
        if stage < N_HORIZON:
            solver.set(stage, 'u', np.zeros(1))
    status, iters, ms = solve(solver, x)
    print(f'initial solve: status {status}, {iters} iterations, {ms:.1f} ms')
    solver.options_set('max_iter', args.max_iter)
    active = Policy(solver, limit)

    n_step = int(round(DT/H_SIM))
    n_policy = int(round(1.0/POLICY_RATE/H_SIM))
    solves, x_log, u_log = [], [], []
    pending = None
    for k in range(int(round(T_SIM/H_SIM))):
        if k % n_step == 0:
            if pending is not None:
                active = pending
            x_ref = active.x0.copy()
            u = active(x, x_ref)
            status, iters, ms = solve(solver, np.asarray(model(x, u)).reshape(NX), warm=True)
            solves.append((k*H_SIM, iters, ms, status))
            pending = Policy(solver, limit)
        elif k % n_policy == 0:
            u = active(x, x_ref)
        x = np.asarray(plant(x, u + args.noise*rng.standard_normal(1))).reshape(NX)
        x_ref = np.asarray(plant(x_ref, active.clip(active.u0))).reshape(NX)
        x_log.append(x)
        u_log.append(float(u[0]))

    x_log, u_log, solves = np.array(x_log), np.array(u_log), np.array(solves)
    dx = x_log[-int(round(1.0/H_SIM)):] - X_TARGET
    dx[:, :2] = np.mod(dx[:, :2] + np.pi, 2*np.pi) - np.pi
    error = np.linalg.norm(dx, axis=1).max()
    iterations, ms, statuses = solves[:, 1], solves[:, 2], solves[:, 3]
    print(f'{len(solves)} warm-started solves (at most {args.max_iter} iterations): {np.sum(statuses == 0)} converged, '
          f'iterations mean {iterations.mean():.1f} max {iterations.max():.0f}')
    print(f'solve time median {np.median(ms):.2f} ms, max {ms.max():.2f} ms, '
          f'{np.sum(ms > 1e3*DT)} over the {1e3*DT:.0f} ms step')
    print(f'peak torque {np.abs(u_log).max():.2f} Nm, max deviation from the upright over the last second: {error:.3f}')
    if args.plot or args.save:
        t = H_SIM*np.arange(1, len(x_log) + 1)
        visualize(t, x_log, u_log, solves, limit, save=args.save)
    return 0 if error < 0.2 else 1


if __name__ == '__main__':
    sys.exit(main())

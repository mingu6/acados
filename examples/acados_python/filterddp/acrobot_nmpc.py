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
torque noise. At every 50 ms OCP step the controller measures x(t_k), applies the affine policy of the
active solution, predicts x(t_{k+1}) with the OCP model and solves from the prediction, warm started
from the shifted policy of the previous solve; the new solution becomes active at t_{k+1}. Between OCP
steps the policy u = u0 + k0 + K0 (x - x0(t)) is applied at 100 Hz, with x0(t) the planned initial
state propagated with the nominal model under u0.

    acrobot_nmpc.py [--control_limits] [--noise 0.5]
"""

import argparse
import sys

import casadi as ca
import numpy as np
from acados_template import AcadosOcpSolver

from acrobot_filterddp import initialize, setup_rk4
from acrobot_model import DEFAULT_PARAMETERS, DT, NX, acrobot_explicit, acrobot_rk4

H_SIM = 1e-3
T_SIM = 8.0
POLICY_RATE = 100.0
CONTROL_LIMIT = 4.0
X_TARGET = np.array([np.pi, 0.0, 0.0, 0.0])


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


def solve(solver, x0):
    solver.set(0, 'lbx', x0)
    solver.set(0, 'ubx', x0)
    status = solver.solve()
    return status, solver.get_stats('nlp_iter')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_limits', action='store_true', help=f'limit the torque to +-{CONTROL_LIMIT} Nm')
    parser.add_argument('--noise', type=float, default=0.5, help='standard deviation of the torque noise in Nm')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    limit = CONTROL_LIMIT if args.control_limits else None

    ocp = setup_rk4(limit, 1e-5, 1000, 0)
    solver = AcadosOcpSolver(ocp, json_file='acrobot_nmpc_ocp.json', verbose=False)
    plant = plant_step(DEFAULT_PARAMETERS)
    model = ocp_step(DEFAULT_PARAMETERS)
    rng = np.random.default_rng(args.seed)

    # initial solve to optimality from rest, then warm starts from the shifted policy
    x = np.zeros(NX)
    initialize(solver, 1, DEFAULT_PARAMETERS, np.zeros(1))
    status, iters = solve(solver, x)
    print(f'initial solve: status {status}, {iters} iterations')
    solver.options_set('filterddp_warm_start', 1)
    active = Policy(solver, limit)

    n_step = int(round(DT/H_SIM))
    n_policy = int(round(1.0/POLICY_RATE/H_SIM))
    statuses, iterations, x_log = [], [], []
    u_peak = 0.0
    pending = None
    for k in range(int(round(T_SIM/H_SIM))):
        if k % n_step == 0:
            if pending is not None:
                active = pending
            x_ref = active.x0.copy()
            u = active(x, x_ref)
            status, iters = solve(solver, np.asarray(model(x, u)).reshape(NX))
            statuses.append(status)
            iterations.append(iters)
            pending = Policy(solver, limit)
        elif k % n_policy == 0:
            u = active(x, x_ref)
        u_peak = max(u_peak, float(np.abs(u).max()))
        x = np.asarray(plant(x, u + args.noise*rng.standard_normal(1))).reshape(NX)
        x_ref = np.asarray(plant(x_ref, active.clip(active.u0))).reshape(NX)
        x_log.append(x)

    dx = np.array(x_log[-int(round(1.0/H_SIM)):]) - X_TARGET
    dx[:, :2] = np.mod(dx[:, :2] + np.pi, 2*np.pi) - np.pi
    error = np.linalg.norm(dx, axis=1).max()
    statuses, iterations = np.array(statuses), np.array(iterations)
    print(f'{len(statuses)} warm-started solves: {np.sum(statuses == 0)} converged, '
          f'iterations mean {iterations.mean():.1f} max {iterations.max()}')
    print(f'peak torque {u_peak:.2f} Nm, max deviation from the upright over the last second: {error:.3f}')
    return 0 if error < 0.2 else 1


if __name__ == '__main__':
    sys.exit(main())

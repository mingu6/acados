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
FILTERDDP with soft constraints on the unicycle of test_filterddp.py, against the exact-Hessian SQP solver
on the same soft formulation: softened nonlinear inequality, state bound, control bound and nonlinear
equality, with L1, L2 and mixed penalties, and a constraint that is infeasible as a hard constraint.
Compares the solution, the slacks, the multipliers and the exported residuals, and checks that the
slacks are active where the penalty is weak.
"""

import sys

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosOcp, AcadosOcpSolver

from test_filterddp import N, X0, check, setup, solve, trajectory


def penalties(ocp: AcadosOcp, zl, zu, Zl, Zu, ns_0: int) -> None:
    ocp.cost.zl, ocp.cost.zu = np.asarray(zl, float), np.asarray(zu, float)
    ocp.cost.Zl, ocp.cost.Zu = np.asarray(Zl, float), np.asarray(Zu, float)
    # the initial stage has no soft state bounds, its soft rows come first in the path layout [bu; g; h]
    ocp.cost.zl_0, ocp.cost.zu_0 = ocp.cost.zl[:ns_0], ocp.cost.zu[:ns_0]
    ocp.cost.Zl_0, ocp.cost.Zu_0 = ocp.cost.Zl[:ns_0], ocp.cost.Zu[:ns_0]


def soft_obstacle(ocp: AcadosOcp) -> None:
    """weak mixed penalty on the obstacle distance: the optimal path cuts through the obstacle"""
    ocp.constraints.idxsh = np.array([0])
    ocp.constraints.idxsh_0 = np.array([0])
    penalties(ocp, [0.1], [0.1], [1.0], [1.0], 1)


def soft_state_bound(ocp: AcadosOcp) -> None:
    """L2 penalty on the velocity bound, active in the hard problem"""
    ocp.constraints.idxsbx = np.array([0])
    penalties(ocp, [0.0], [0.0], [2.0], [2.0], 0)


def soft_control_bound(ocp: AcadosOcp) -> None:
    """L1 penalty on the acceleration bound, active in the hard problem"""
    ocp.constraints.idxsbu = np.array([0])
    penalties(ocp, [0.5], [0.5], [0.0], [0.0], 1)


def soft_equality(ocp: AcadosOcp) -> None:
    """mixed penalty on the equality u2 = u0 u1, an inequality with the slacks as width"""
    ocp.constraints.idxsh = np.array([1])
    ocp.constraints.idxsh_0 = np.array([1])
    penalties(ocp, [1.0], [1.0], [10.0], [10.0], 1)


def soft_unreachable(ocp: AcadosOcp) -> None:
    """x0 + x1 >= 10 cannot be reached in the horizon: infeasible as a hard constraint, L1 penalty"""
    x = ocp.model.x
    ocp.model.con_h_expr = ca.vertcat(ocp.model.con_h_expr, x[0] + x[1])
    ocp.model.con_h_expr_0 = ocp.model.con_h_expr
    ocp.constraints.lh = np.append(ocp.constraints.lh, 10.0)
    ocp.constraints.uh = np.append(ocp.constraints.uh, ACADOS_INFTY)
    ocp.constraints.lh_0, ocp.constraints.uh_0 = ocp.constraints.lh, ocp.constraints.uh
    ocp.constraints.idxsh = np.array([2])
    ocp.constraints.idxsh_0 = np.array([2])
    penalties(ocp, [0.05], [0.05], [0.0], [0.0], 1)


def soft_all(ocp: AcadosOcp) -> None:
    """every constraint soft with slack lower bounds, also the control bound with a lower slack bound"""
    ocp.constraints.idxsbu = np.array([0, 1])
    ocp.constraints.idxsbx = np.array([0])
    ocp.constraints.idxsh = np.array([0, 1])
    ocp.constraints.idxsh_0 = np.array([0, 1])
    ocp.constraints.lsh = np.array([0.0, 0.01])
    ocp.constraints.ush = np.array([0.0, 0.01])
    ocp.constraints.lsh_0, ocp.constraints.ush_0 = ocp.constraints.lsh, ocp.constraints.ush
    # slack layout [sbu (2); sbx (1); sh (2)] on the path, [sbu (2); sh (2)] at stage 0
    penalties(ocp, [1.0, 1.0, 0.0, 0.1, 1.0], [1.0, 1.0, 0.0, 0.1, 1.0], [0.0, 0.0, 2.0, 1.0, 10.0], [0.0, 0.0, 2.0, 1.0, 10.0], 0)
    ocp.cost.zl_0, ocp.cost.zu_0 = ocp.cost.zl[[0, 1, 3, 4]], ocp.cost.zu[[0, 1, 3, 4]]
    ocp.cost.Zl_0, ocp.cost.Zu_0 = ocp.cost.Zl[[0, 1, 3, 4]], ocp.cost.Zu[[0, 1, 3, 4]]


# name: (formulation, slacks active, unique slack bound multipliers). The soft equality has a penalty strong
# enough for its slacks to vanish; a soft equality at its bound with both slacks zero has two active rows and
# two active slack bounds, whose multipliers satisfy lam_l + xi_l = z_l, lam_u + xi_u = z_u and
# lam_u - lam_l = nu only, so the split between the rows and the slack bounds is not unique.
FORMULATIONS = {
    'soft obstacle': (soft_obstacle, True, True),
    'soft state bound': (soft_state_bound, True, True),
    'soft control bound': (soft_control_bound, True, True),
    'soft equality': (soft_equality, False, False),
    'soft unreachable': (soft_unreachable, True, True),
    'soft all': (soft_all, True, False),
}


def slacks(solver: AcadosOcpSolver) -> np.ndarray:
    return np.concatenate([np.concatenate([solver.get(k, 'sl'), solver.get(k, 'su')]) for k in range(N)])


def multipliers(solver: AcadosOcpSolver) -> tuple:
    """lam_upper - lam_lower of the constraint rows, and the multipliers of the slack bounds, which follow the
    rows in the layout [lower; upper; lower slack bounds; upper slack bounds]"""
    net, bounds = [], []
    for k in range(N):
        lam = solver.get(k, 'lam')
        ns = solver.get(k, 'sl').size
        ni0 = lam.size//2 - ns
        net.append(lam[ni0:2*ni0] - lam[:ni0])
        bounds.append(lam[2*ni0:])
    return np.concatenate(net), np.concatenate(bounds)


def main():
    ok = True
    for name, (formulation, active, unique_slack_multipliers) in FORMULATIONS.items():
        tag = name.replace(' ', '_')
        solvers = {}
        for solver_type in ('FILTERDDP', 'SQP'):
            ocp = setup(solver_type)
            formulation(ocp)
            ocp.model.name = f'filterddp_unicycle_{tag}'
            ocp.code_gen_options.code_export_directory = f'c_generated_code_soft_{tag}_{solver_type.lower()}'
            solvers[solver_type] = AcadosOcpSolver(ocp, json_file=f'test_soft_{tag}_{solver_type.lower()}_ocp.json', verbose=False)
        ddp, sqp = solvers['FILTERDDP'], solvers['SQP']
        print(f'\n{name}')
        for solver_type, solver in solvers.items():
            status = solve(solver, X0)
            print(f'{solver_type}: status {status}, {solver.get_stats("nlp_iter")} iterations, cost {solver.get_cost():.10f}')
            ok &= status == 0
        # with an L2 penalty only, a slack and its bound multiplier vanish together at an inactive row, and the
        # interior point solutions sit sqrt(mu) from the bound
        ok &= check('x', trajectory(ddp, 'x', range(N + 1)), trajectory(sqp, 'x', range(N + 1)), 1e-5)
        ok &= check('u', trajectory(ddp, 'u', range(N)), trajectory(sqp, 'u', range(N)), 1e-5)
        ok &= check('slacks', slacks(ddp), slacks(sqp), 1e-4)
        ok &= check('pi', trajectory(ddp, 'pi', range(N)), trajectory(sqp, 'pi', range(N)), 1e-5)
        ok &= check('lam (upper - lower)', multipliers(ddp)[0], multipliers(sqp)[0], 1e-5)
        if unique_slack_multipliers:
            ok &= check('slack bound multipliers', multipliers(ddp)[1], multipliers(sqp)[1], 1e-4)
        ok &= check('residuals (stat, eq, ineq, comp)', ddp.get_residuals(recompute=True), np.zeros(4), 1e-6)
        largest = np.max(slacks(sqp))
        print(f'     largest slack {largest:.3e}, expected {"active" if active else "zero"}')
        ok &= largest > 1e-3 if active else largest < 1e-6

    # warm start from the shifted policy with soft rows, against a cold solve from the same initial state.
    # The shift needs the same constraint rows at every stage, so the state bound of the path stages, which
    # the initial stage does not have, is removed.
    print('\nwarm start, soft obstacle without the state bound')
    ocp = setup('FILTERDDP')
    soft_obstacle(ocp)
    ocp.constraints.idxbx = np.array([], dtype=int)
    ocp.constraints.lbx = np.array([])
    ocp.constraints.ubx = np.array([])
    ocp.model.name = 'filterddp_unicycle_soft_warm'
    ocp.code_gen_options.code_export_directory = 'c_generated_code_soft_warm'
    ddp = AcadosOcpSolver(ocp, json_file='test_soft_warm_ocp.json', verbose=False)
    # the next state along the solution from X0, where the shifted policy is nearly exact
    solve(ddp, X0)
    x1 = ddp.get(1, 'x')
    cold = solve(ddp, x1)
    cold_x, cold_u, cold_s = trajectory(ddp, 'x', range(N + 1)), trajectory(ddp, 'u', range(N)), slacks(ddp)
    cold_iterations = ddp.get_stats('nlp_iter')
    solve(ddp, X0)
    ddp.options_set('filterddp_warm_start', 1)
    ddp.constraints_set(0, 'lbx', x1)
    ddp.constraints_set(0, 'ubx', x1)
    warm = ddp.solve()
    print(f'cold: status {cold}, {cold_iterations} iterations; warm: status {warm}, {ddp.get_stats("nlp_iter")} iterations')
    ok &= cold == 0 and warm == 0
    ok &= check('x warm against cold', trajectory(ddp, 'x', range(N + 1)), cold_x, 1e-5)
    ok &= check('u warm against cold', trajectory(ddp, 'u', range(N)), cold_u, 1e-5)
    ok &= check('slacks warm against cold', slacks(ddp), cold_s, 1e-4)
    ok &= check('residuals (stat, eq, ineq, comp)', ddp.get_residuals(recompute=True), np.zeros(4), 1e-6)

    print('\nall checks passed' if ok else '\nsome checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

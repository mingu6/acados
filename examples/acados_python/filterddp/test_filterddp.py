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
Consistency test of FILTERDDP against the exact-Hessian SQP solver on a unicycle with an active control
bound, an active state bound on stages 1..N-1, a nonlinear obstacle inequality and a nonlinear equality.
Checks the solution, all multipliers, the residuals, the value function gradient and the solution
sensitivities (against finite differences) and the feedback gain of the affine policy.
"""

import sys

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver

N = 20
DT = 0.1
NX = 4
NU = 3
X0 = np.array([0.0, 0.0, 0.0, 0.0])
X_TARGET = np.array([2.0, 1.0, 0.0, 0.0])
OBSTACLE = (1.0, 0.4, 0.3)
V_MAX = 0.75  # not a multiple of DT times the acceleration limit, which would make the active set degenerate


def export_model() -> AcadosModel:
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)

    def f(x, u):
        return ca.vertcat(x[3]*ca.cos(x[2]), x[3]*ca.sin(x[2]), u[1], u[0])

    k1 = f(x, u)
    k2 = f(x + 0.5*DT*k1, u)
    k3 = f(x + 0.5*DT*k2, u)
    k4 = f(x + DT*k3, u)

    model = AcadosModel()
    model.name = 'filterddp_unicycle'
    model.x = x
    model.u = u
    model.disc_dyn_expr = x + (DT/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    dx = x - X_TARGET
    model.cost_expr_ext_cost = DT*(ca.sumsqr(dx) + 0.1*ca.sumsqr(u))
    model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    model.cost_expr_ext_cost_e = 10.0*ca.sumsqr(dx)
    # obstacle distance and the equality u2 = u0*u1 on a third control
    h = ca.vertcat((x[0] - OBSTACLE[0])**2 + (x[1] - OBSTACLE[1])**2, u[2] - u[0]*u[1])
    model.con_h_expr = h
    model.con_h_expr_0 = h
    return model


def setup(nlp_solver_type: str) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = export_model()
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_0 = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.constraints.x0 = X0
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-1.0, -1.0])
    ocp.constraints.ubu = np.array([1.0, 1.0])
    ocp.constraints.idxbx = np.array([3])
    ocp.constraints.lbx = np.array([-V_MAX])
    ocp.constraints.ubx = np.array([V_MAX])
    ocp.constraints.lh = np.array([OBSTACLE[2]**2, 0.0])
    ocp.constraints.uh = np.array([ACADOS_INFTY, 0.0])
    ocp.constraints.lh_0 = ocp.constraints.lh
    ocp.constraints.uh_0 = ocp.constraints.uh

    opts = ocp.solver_options
    opts.N_horizon = N
    opts.tf = N*DT
    opts.integrator_type = 'DISCRETE'
    opts.nlp_solver_type = nlp_solver_type
    opts.hessian_approx = 'EXACT'
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    opts.qp_solver_cond_N = N
    opts.qp_solver_ric_alg = 0  # classic Riccati recursion for the solution sensitivities
    opts.nlp_solver_max_iter = 300
    opts.nlp_solver_tol_stat = 1e-7
    opts.nlp_solver_tol_eq = 1e-7
    opts.nlp_solver_tol_ineq = 1e-7
    opts.nlp_solver_tol_comp = 1e-7
    opts.cost_scaling = np.ones(N + 1)
    if nlp_solver_type == 'FILTERDDP':
        opts.regularize_method = 'NO_REGULARIZE'
        opts.globalization = 'FIXED_STEP'
    else:
        opts.regularize_method = 'MIRROR'
        opts.globalization = 'MERIT_BACKTRACKING'
    ocp.code_export_directory = f'c_generated_code_test_{nlp_solver_type.lower()}'
    return ocp


def solve(solver: AcadosOcpSolver, x0: np.ndarray) -> int:
    for stage in range(N):
        solver.set(stage, 'x', x0)
        solver.set(stage, 'u', np.zeros(NU))
    solver.set(N, 'x', x0)
    solver.constraints_set(0, 'lbx', x0)
    solver.constraints_set(0, 'ubx', x0)
    return solver.solve()


def trajectory(solver: AcadosOcpSolver, field: str, stages) -> np.ndarray:
    return np.concatenate([solver.get(stage, field) for stage in stages])


def net_multipliers(solver: AcadosOcpSolver, stages) -> np.ndarray:
    """lam_upper - lam_lower per row; unlike the split over the two sides of an equality it is unique."""
    lam = [solver.get(stage, 'lam') for stage in stages]
    return np.concatenate([l[l.size//2:] - l[:l.size//2] for l in lam])


def check(name: str, value, reference, tol: float) -> bool:
    error = np.max(np.abs(np.asarray(value) - np.asarray(reference)))
    scale = max(1.0, np.max(np.abs(reference)))
    ok = error <= tol*scale
    print(f'{"ok  " if ok else "FAIL"} {name:40s} max error {error:.2e} (tol {tol*scale:.1e})')
    return ok


def main():
    solvers = {t: AcadosOcpSolver(setup(t), json_file=f'test_{t.lower()}_ocp.json', verbose=False)
               for t in ('FILTERDDP', 'SQP')}
    ddp, sqp = solvers['FILTERDDP'], solvers['SQP']
    ok = True

    for name, solver in solvers.items():
        status = solve(solver, X0)
        print(f'{name}: status {status}, {solver.get_stats("nlp_iter")} iterations, cost {solver.get_cost():.10f}')
        ok &= status == 0

    # solution and multipliers: at the tolerance 1e-7 the interior point solution sits about mu/z inside
    # the active bounds, so the controls agree to about 1e-5
    ok &= check('x', trajectory(ddp, 'x', range(N + 1)), trajectory(sqp, 'x', range(N + 1)), 1e-6)
    ok &= check('u', trajectory(ddp, 'u', range(N)), trajectory(sqp, 'u', range(N)), 1e-5)
    ok &= check('pi', trajectory(ddp, 'pi', range(N)), trajectory(sqp, 'pi', range(N)), 1e-5)
    ok &= check('lam (upper - lower)', net_multipliers(ddp, range(N)), net_multipliers(sqp, range(N)), 1e-5)
    active = np.sum(trajectory(sqp, 'lam', range(N)) > 1e-4)
    print(f'     {active} active inequality multipliers')
    ok &= active > 0

    # residuals of the exported primal-dual solution
    residuals = ddp.get_residuals(recompute=True)
    ok &= check('residuals (stat, eq, ineq, comp)', residuals, np.zeros(4), 1e-6)

    # value function gradient w.r.t. the initial state
    ok &= check('value gradient', ddp.eval_and_get_optimal_value_gradient('initial_state'),
                sqp.eval_and_get_optimal_value_gradient('initial_state'), 1e-5)

    # solution sensitivities w.r.t. the initial state against central finite differences
    x0 = X0 + np.array([0.0, 0.0, 0.1, 0.1])
    solve(ddp, x0)
    ddp.setup_qp_matrices_and_factorize()
    sens = ddp.eval_solution_sensitivity(list(range(N)), 'initial_state', return_sens_x=False)['sens_u']
    K0 = ddp.get_from_qp_in(0, 'K')
    eps = 1e-4
    fd = np.zeros((N, NU, NX))
    for j in range(NX):
        u_plus, u_minus = [], []
        for sign, store in ((1.0, u_plus), (-1.0, u_minus)):
            dx0 = np.zeros(NX)
            dx0[j] = sign*eps
            solve(ddp, x0 + dx0)
            store.append(trajectory(ddp, 'u', range(N)).reshape(N, NU))
        fd[:, :, j] = (u_plus[0] - u_minus[0])/(2*eps)
    ok &= check('sensitivity du/dx0 (finite differences)', np.asarray(sens), fd, 1e-2)
    ok &= check('feedback gain K0 = sensitivity du0/dx0', K0, np.asarray(sens)[0], 1e-6)

    print('all checks passed' if ok else 'some checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

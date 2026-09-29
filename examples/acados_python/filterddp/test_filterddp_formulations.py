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
FILTERDDP with the cost modules, integrators and Hessian approximations other than EXTERNAL cost, DISCRETE
dynamics and EXACT Hessian, on the unicycle of test_filterddp.py. Each formulation represents the same OCP,
so its solution and multipliers are compared against the exact-Hessian SQP solution of the reference
formulation: DISCRETE RK4 dynamics, which ERK with four stages and one step reproduces, or IRK for IRK.
"""

import sys

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from test_filterddp import (DT, N, NU, NX, X0, X_TARGET, check, dynamics, net_multipliers, setup, solve,
                            trajectory)

# the cost of test_filterddp.py as 0.5 |y - yref|_W^2 with y = [x; u] and y_e = x
W = 2.0*DT*np.diag(np.concatenate([np.ones(NX), 0.1*np.ones(NU)]))
W_E = 20.0*np.eye(NX)
YREF = np.concatenate([X_TARGET, np.zeros(NU)])


def clear_external_cost(ocp: AcadosOcp) -> None:
    ocp.model.cost_expr_ext_cost = None
    ocp.model.cost_expr_ext_cost_0 = None
    ocp.model.cost_expr_ext_cost_e = None


def least_squares_weights(ocp: AcadosOcp) -> None:
    ocp.cost.W_0 = W
    ocp.cost.W = W
    ocp.cost.W_e = W_E
    ocp.cost.yref_0 = YREF
    ocp.cost.yref = YREF
    ocp.cost.yref_e = X_TARGET


def linear_ls(ocp: AcadosOcp) -> None:
    clear_external_cost(ocp)
    ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = 'LINEAR_LS'
    Vx = np.vstack([np.eye(NX), np.zeros((NU, NX))])
    Vu = np.vstack([np.zeros((NX, NU)), np.eye(NU)])
    ocp.cost.Vx_0, ocp.cost.Vu_0 = Vx, Vu
    ocp.cost.Vx, ocp.cost.Vu = Vx, Vu
    ocp.cost.Vx_e = np.eye(NX)
    least_squares_weights(ocp)


def nonlinear_ls(ocp: AcadosOcp) -> None:
    clear_external_cost(ocp)
    ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = 'NONLINEAR_LS'
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.model.cost_y_expr_0 = y
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = ocp.model.x
    least_squares_weights(ocp)


def convex_over_nonlinear(ocp: AcadosOcp) -> None:
    clear_external_cost(ocp)
    ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'
    r = ca.SX.sym('r', NX + NU)
    r_e = ca.SX.sym('r_e', NX)
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    for suffix, y_expr, r_in, weight in (('_0', y, r, W), ('', y, r, W), ('_e', ocp.model.x, r_e, W_E)):
        setattr(ocp.model, f'cost_y_expr{suffix}', y_expr)
        setattr(ocp.model, f'cost_r_in_psi_expr{suffix}', r_in)
        setattr(ocp.model, f'cost_psi_expr{suffix}', 0.5*r_in.T @ weight @ r_in)
    ocp.cost.yref_0 = YREF
    ocp.cost.yref = YREF
    ocp.cost.yref_e = X_TARGET
    # the outer function Hessian of the cost, exact Hessians of dynamics and constraints
    ocp.solver_options.exact_hess_cost = False


def erk(ocp: AcadosOcp) -> None:
    ocp.model.xdot = ca.SX.sym('xdot', NX)
    ocp.model.f_expl_expr = dynamics(ocp.model.x, ocp.model.u)
    ocp.model.disc_dyn_expr = None
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1


def irk(ocp: AcadosOcp) -> None:
    ocp.model.xdot = ca.SX.sym('xdot', NX)
    ocp.model.f_impl_expr = ocp.model.xdot - dynamics(ocp.model.x, ocp.model.u)
    ocp.model.disc_dyn_expr = None
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.sim_method_newton_tol = 1e-12
    ocp.solver_options.sim_method_newton_iter = 20


def gauss_newton(ocp: AcadosOcp) -> None:
    nonlinear_ls(ocp)
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'


# name: (formulation, reference)
FORMULATIONS = {
    'LINEAR_LS': (linear_ls, 'DISCRETE'),
    'NONLINEAR_LS': (nonlinear_ls, 'DISCRETE'),
    'CONVEX_OVER_NONLINEAR': (convex_over_nonlinear, 'DISCRETE'),
    'ERK': (erk, 'DISCRETE'),
    'IRK': (irk, 'IRK'),
    'GAUSS_NEWTON': (gauss_newton, 'DISCRETE'),
}


def create_solver(nlp_solver_type: str, formulation, tag: str) -> AcadosOcpSolver:
    ocp = setup(nlp_solver_type)
    if formulation is not None:
        formulation(ocp)
    ocp.model.name = f'filterddp_unicycle_{tag.lower()}'
    ocp.code_gen_options.code_export_directory = f'c_generated_code_formulations_{tag.lower()}'
    return AcadosOcpSolver(ocp, json_file=f'test_formulations_{tag.lower()}_ocp.json', verbose=False)


def main():
    ok = True
    references = {}
    for name, formulation in (('DISCRETE', None), ('IRK', irk)):
        references[name] = create_solver('SQP', formulation, f'sqp_{name}')
        status = solve(references[name], X0)
        print(f'reference SQP {name}: status {status}, {references[name].get_stats("nlp_iter")} iterations')
        ok &= status == 0

    for name, (formulation, reference_name) in FORMULATIONS.items():
        ddp = create_solver('FILTERDDP', formulation, f'ddp_{name}')
        sqp = references[reference_name]
        status = solve(ddp, X0)
        print(f'\nFILTERDDP {name}: status {status}, {ddp.get_stats("nlp_iter")} iterations, '
              f'cost {ddp.get_cost():.10f} (SQP {reference_name} {sqp.get_cost():.10f})')
        ok &= status == 0
        ok &= check('x', trajectory(ddp, 'x', range(N + 1)), trajectory(sqp, 'x', range(N + 1)), 1e-6)
        ok &= check('u', trajectory(ddp, 'u', range(N)), trajectory(sqp, 'u', range(N)), 1e-5)
        ok &= check('pi', trajectory(ddp, 'pi', range(N)), trajectory(sqp, 'pi', range(N)), 1e-5)
        ok &= check('lam (upper - lower)', net_multipliers(ddp, range(N)), net_multipliers(sqp, range(N)), 1e-5)
        ok &= check('residuals (stat, eq, ineq, comp)', ddp.get_residuals(recompute=True), np.zeros(4), 1e-6)

    print('\nall checks passed' if ok else '\nsome checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

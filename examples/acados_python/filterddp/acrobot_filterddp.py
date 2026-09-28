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

import argparse
import sys
import time

import numpy as np
from acados_template import ACADOS_INFTY, AcadosOcp, AcadosOcpSolver

from acrobot_model import (DEFAULT_PARAMETERS, DT, NX, export_acrobot_contact_model,
                           export_acrobot_rk4_model, load_parameter_sets)

N_HORIZON = 100


def common_options(ocp: AcadosOcp, tol: float, max_iter: int, print_level: int):
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON*DT
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'FILTERDDP'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = N_HORIZON
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'
    ocp.solver_options.globalization = 'FIXED_STEP'
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.solver_options.nlp_solver_tol_stat = tol
    ocp.solver_options.nlp_solver_tol_eq = tol
    ocp.solver_options.nlp_solver_tol_ineq = tol
    ocp.solver_options.nlp_solver_tol_comp = tol
    ocp.solver_options.print_level = print_level
    ocp.solver_options.cost_scaling = np.ones(N_HORIZON + 1)


def setup_rk4(control_limit: float | None, tol: float, max_iter: int, print_level: int) -> AcadosOcp:
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
    common_options(ocp, tol, max_iter, print_level)
    return ocp


def setup_contact(tol: float, max_iter: int, print_level: int) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = export_acrobot_contact_model()
    ocp.parameter_values = DEFAULT_PARAMETERS
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.idxbu = np.array([0, 3, 4, 5, 6])
    ocp.constraints.lbu = np.array([-8.0, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([8.0, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.lh = np.zeros(6)
    ocp.constraints.uh = np.array([0.0, 0.0, 0.0, 0.0, ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.lh_0 = ocp.constraints.lh
    ocp.constraints.uh_0 = ocp.constraints.uh
    common_options(ocp, tol, max_iter, print_level)
    return ocp


def initialize(solver: AcadosOcpSolver, nu: int, parameters: np.ndarray, u_init: np.ndarray):
    for stage in range(N_HORIZON):
        solver.set(stage, 'x', np.zeros(NX))
        solver.set(stage, 'u', u_init)
        solver.set(stage, 'p', parameters)
    solver.set(N_HORIZON, 'x', np.zeros(NX))
    solver.set(N_HORIZON, 'p', parameters)


def solve(solver: AcadosOcpSolver, nu: int, parameters: np.ndarray, u_init: np.ndarray):
    initialize(solver, nu, parameters, u_init)
    t0 = time.perf_counter()
    status = solver.solve()
    wall = time.perf_counter() - t0
    return status, solver.get_stats('nlp_iter'), solver.get_cost(), 1e3*wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', choices=['rk4', 'rk4_limits', 'contact'])
    parser.add_argument('--params', default=None, help='FilterDDP.jl params file with one parameter set per line')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--print_level', type=int, default=0)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()

    if args.problem == 'rk4':
        ocp = setup_rk4(None, args.tol, args.max_iter, args.print_level)
        u_init = np.zeros(1)
    elif args.problem == 'rk4_limits':
        ocp = setup_rk4(8.0, args.tol, args.max_iter, args.print_level)
        u_init = np.zeros(1)
    else:
        ocp = setup_contact(args.tol, args.max_iter, args.print_level)
        u_init = np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01])
    nu = ocp.model.u.rows()

    solver = AcadosOcpSolver(ocp, json_file=f'acrobot_{args.problem}_ocp.json', verbose=False)

    if args.params is not None:
        parameter_sets = load_parameter_sets(args.params)[:args.seeds]
    else:
        parameter_sets = DEFAULT_PARAMETERS[None, :]

    n_success = 0
    for seed, parameters in enumerate(parameter_sets, start=1):
        solve(solver, nu, parameters, u_init)
        status, iters, cost, wall = solve(solver, nu, parameters, u_init)
        n_success += status == 0
        print(f'seed={seed:3d} status={status} iterations={iters:4d} objective={cost:.8e} time={wall:.2f} ms')
        if args.print_level == 0 and seed == 1:
            solver.print_statistics()

    print(f'Solved {n_success}/{len(parameter_sets)} problems')
    return 0 if n_success == len(parameter_sets) else 1


if __name__ == '__main__':
    sys.exit(main())

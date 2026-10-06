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
Multi-phase OCPs with FILTERDDP against the exact-Hessian SQP solver, on the formulations of the acados
multi-phase examples:
- mocp_transition_example: a double integrator (nx = 2), a transition stage without controls (nu = 0) that
  drops the velocity, and a single integrator (nx = 1);
- time_varying/piecewiese_polynomial_control_example: piecewise polynomial controls of degree 0 and 4 on the
  pendulum (nu = 1, then nu = 5);
- multiphase_nonlinear_constraints: two phases of the same dimensions, a nonlinear constraint in the second one.
Checks the solution, the multipliers, the residuals and the value gradient. Then in closed loop, solves warm
started with warm_start_from_policy against cold-started solves: the transition example, whose shift crosses the
change of dimensions, and the nonlinear constraints example with the constraint in phase 1 or in phase 0, whose
shift crosses a change of the constraint rows; with both initializations of the multipliers
(filterddp_bound_mult_init_method).

The tolerance is 1e-8, and 1e-10 for the comparison on the transition example, whose small acceleration
cost determines the controls only weakly: at 1e-8 the controls of the two solvers differ by about 1e-4.
"""

import importlib.util
import os
import sys

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosMultiphaseOcp, AcadosOcp, AcadosOcpSolver

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
TOL = 1e-8


def load(name: str, path: str):
    """Import a module of another example directory, whose own imports are resolved in that directory."""
    directory = os.path.join(EXAMPLES, os.path.dirname(path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_options(mocp: AcadosMultiphaseOcp, nlp_solver_type: str, tol: float = TOL):
    opts = mocp.solver_options
    opts.nlp_solver_type = nlp_solver_type
    opts.hessian_approx = 'EXACT'
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    opts.nlp_solver_max_iter = 300
    opts.nlp_solver_tol_stat = tol
    opts.nlp_solver_tol_eq = tol
    opts.nlp_solver_tol_ineq = tol
    opts.nlp_solver_tol_comp = tol
    if nlp_solver_type == 'FILTERDDP':
        opts.regularize_method = 'NO_REGULARIZE'
        opts.globalization = 'FIXED_STEP'
    else:
        opts.regularize_method = 'MIRROR'
        opts.globalization = 'MERIT_BACKTRACKING'


def transition_mocp(nlp_solver_type: str, warm_start: bool = False, tol: float = 1e-10) -> AcadosMultiphaseOcp:
    """mocp_transition_example with N_list = [10, 1, 15] and T_1 = 0.4."""
    ex = load('mocp_transition_main', 'mocp_transition_example/main.py')
    N_list = [10, 1, 15]
    t_horizon_1 = 0.4*ex.T_HORIZON
    mocp = AcadosMultiphaseOcp(N_list=N_list)
    mocp.set_phase(ex.formulate_double_integrator_ocp(), 0)
    transition = AcadosOcp()
    transition.model = ex.get_transition_model()
    transition.cost.cost_type = 'NONLINEAR_LS'
    transition.model.cost_y_expr = transition.model.x
    transition.cost.W = np.diag([ex.L2_COST_P, 1e-1*ex.L2_COST_V])
    transition.cost.yref = np.array([0., 0.])
    mocp.set_phase(transition, 1)
    mocp.set_phase(ex.formulate_single_integrator_ocp(), 2)
    mocp.mocp_opts.integrator_type = ['IRK', 'DISCRETE', 'IRK']
    # the transition stage has the time step 1 so that its stage cost is not scaled
    mocp.solver_options.tf = ex.T_HORIZON + 1.0
    t_horizon_2 = ex.T_HORIZON - t_horizon_1
    mocp.solver_options.time_steps = np.array(N_list[0]*[t_horizon_1/N_list[0]] + [1.0] + N_list[2]*[t_horizon_2/N_list[2]])
    set_options(mocp, nlp_solver_type, tol)
    mocp.name = f'mp_transition_{nlp_solver_type.lower()}{"_ws" if warm_start else ""}_{-np.log10(tol):.0f}'
    return mocp


def polynomial_mocp(nlp_solver_type: str) -> AcadosMultiphaseOcp:
    """
    main_mocp of the piecewise polynomial control example: degrees [0, 4] on N_list = [1, 5]. FILTERDDP needs
    cost_discretization EULER, with which the cost sees the polynomial only at the start of its interval and
    leaves the higher coefficients undetermined; a small cost on the coefficients makes the OCP well posed. The
    pendulum starts at 0.4 rad from the upright, as the swing-up from below on these six stages is not
    solved reliably by either solver.
    """
    ex = load('piecewise_polynomial', 'time_varying/piecewiese_polynomial_control_example.py')
    N_list = [1, 5]
    mocp = AcadosMultiphaseOcp(N_list)
    for i, degree in enumerate([0, 4]):
        ocp, _ = ex.create_ocp_formulation_without_opts('NONLINEAR_LS', degree)
        nu = ocp.model.u.rows()
        ny = ocp.cost.W.shape[0]
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.cost_y_expr, ocp.model.u)
        ocp.cost.W = np.block([[ocp.cost.W, np.zeros((ny, nu))], [np.zeros((nu, ny)), 1e-3*np.eye(nu)]])
        ocp.cost.yref = np.concatenate([ocp.cost.yref, np.zeros(nu)])
        ocp.constraints.x0 = np.array([0.0, 0.4, 0.0, 0.0])
        mocp.set_phase(ocp, i)
    mocp.solver_options.integrator_type = 'IRK'
    mocp.solver_options.cost_discretization = 'EULER'  # FILTERDDP evaluates the cost separately from the dynamics
    T_horizon = 1.0
    dt_short = 0.02
    n_long = sum(N_list) - 1
    mocp.solver_options.time_steps = np.array([dt_short] + n_long*[(T_horizon - dt_short)/n_long])
    mocp.solver_options.tf = T_horizon
    mocp.solver_options.sim_method_num_stages = np.array([2] + n_long*[4])
    set_options(mocp, nlp_solver_type)
    mocp.name = f'mp_polynomial_{nlp_solver_type.lower()}'
    return mocp


def nonlinear_constraints_mocp(nlp_solver_type: str, warm_start: bool = False,
                               constrained_phase: int = 1) -> AcadosMultiphaseOcp:
    """
    create_mocp of the multi-phase nonlinear constraints example with the velocity constraint of phase 1 hard
    and one-sided: the finite placeholder lower bound -1e9 of the example, needed for qpOASES only, keeps both
    solvers from converging to tight tolerances. With constrained_phase 0, phase 0 has a velocity constraint,
    v >= -0.1, and phase 1 none.
    """
    ex = load('mp_nonlinear_constraints', 'multiphase_nonlinear_constraints/create_mocp.py')
    mocp = ex.create_mocp(soften_h=False, qp_solver='PARTIAL_CONDENSING_HPIPM')
    if constrained_phase == 0:
        mocp.model[0].con_h_expr = mocp.model[0].x[1:]
        mocp.constraints[0].lh = np.array([-0.1])
        mocp.constraints[0].uh = np.array([ACADOS_INFTY])
        mocp.model[1].con_h_expr = []
        mocp.constraints[1].lh = np.array([])
        mocp.constraints[1].uh = np.array([])
    else:
        mocp.constraints[1].lh = -ACADOS_INFTY*np.ones(1)
    set_options(mocp, nlp_solver_type)
    mocp.name = (f'mp_nonlinear_constraints_{nlp_solver_type.lower()}{"_ws" if warm_start else ""}'
                 f'{"_phase0" if constrained_phase == 0 else ""}')
    return mocp


def create_solver(mocp: AcadosMultiphaseOcp) -> AcadosOcpSolver:
    mocp.code_gen_options.code_export_directory = f'c_generated_code_{mocp.name}'
    mocp.code_gen_options.json_file = f'{mocp.name}.json'
    return AcadosOcpSolver(mocp, verbose=False)


def trajectory(solver: AcadosOcpSolver, field: str, stages) -> np.ndarray:
    return np.concatenate([solver.get(stage, field) for stage in stages])


def net_multipliers(solver: AcadosOcpSolver, stages) -> np.ndarray:
    """lam_upper - lam_lower per row; unlike the split over the two sides of an equality it is unique."""
    lam = [solver.get(stage, 'lam') for stage in stages]
    return np.concatenate([l[l.size//2:] - l[:l.size//2] for l in lam])


def check(name: str, value, reference, tol: float) -> bool:
    error = np.max(np.abs(np.asarray(value) - np.asarray(reference))) if np.size(reference) > 0 else 0.0
    scale = max(1.0, np.max(np.abs(reference))) if np.size(reference) > 0 else 1.0
    ok = error <= tol*scale
    print(f'{"ok  " if ok else "FAIL"} {name:40s} max error {error:.2e} (tol {tol*scale:.1e})')
    return ok


def compare_with_sqp(label: str, build) -> bool:
    print(f'\n{label}')
    solvers = {t: create_solver(build(t)) for t in ('FILTERDDP', 'SQP')}
    ddp, sqp = solvers['FILTERDDP'], solvers['SQP']
    N = ddp.N
    ok = True
    for name, solver in solvers.items():
        status = solver.solve()
        print(f'{name}: status {status}, {solver.get_stats("nlp_iter")} iterations, cost {solver.get_cost():.10f}')
        ok &= status == 0
    ok &= check('x', trajectory(ddp, 'x', range(N + 1)), trajectory(sqp, 'x', range(N + 1)), 1e-6)
    ok &= check('u', trajectory(ddp, 'u', range(N)), trajectory(sqp, 'u', range(N)), 1e-6)
    ok &= check('pi', trajectory(ddp, 'pi', range(N)), trajectory(sqp, 'pi', range(N)), 1e-6)
    ok &= check('lam (upper - lower)', net_multipliers(ddp, range(N)), net_multipliers(sqp, range(N)), 1e-6)
    ok &= check('cost', ddp.get_cost(), sqp.get_cost(), 1e-8)
    ok &= check('residuals (stat, eq, ineq, comp)', ddp.get_residuals(recompute=True), np.zeros(4), 1e-6)
    ok &= check('value gradient', ddp.eval_and_get_optimal_value_gradient('initial_state'),
                sqp.eval_and_get_optimal_value_gradient('initial_state'), 1e-6)
    # the feedback gains of the affine policy have the dimensions of their stage
    for stage in range(N):
        K = ddp.get_from_qp_in(stage, 'K')
        k = ddp.get_from_qp_in(stage, 'k')
        nu, nx = ddp.get(stage, 'u').size, ddp.get(stage, 'x').size
        if K.shape != (nu, nx) or k.shape != (nu, 1) or not np.all(np.isfinite(K)):
            print(f'FAIL K, k at stage {stage}: shapes {K.shape}, {k.shape}, expected ({nu}, {nx}), ({nu}, 1)')
            ok = False
    return ok


def closed_loop(label: str, build, x0: np.ndarray, u_tol: float, n_steps: int = 30,
                bounded_iterations: bool = False) -> bool:
    """
    Receding horizon from the predicted next state, solves warm started with warm_start_from_policy against
    cold-started solves (from the previous iterate, constant multipliers), for both bound_mult_init_method values.
    warm_start_from_policy must take the policy of every solve, and every warm-started solve must converge to the
    solution of the cold-started solve. With bounded_iterations, no warm-started solve may take more iterations than
    the first, cold, solve.
    """
    mocps = {'warm': build('FILTERDDP', warm_start=True), 'cold': build('FILTERDDP')}
    ok = True
    for k, method in enumerate(('constant', 'mu_based')):
        print(f'\n{label} in closed loop: warm-started ({method}) against cold-started solves')
        # fresh solvers for each method, the code generated and built once
        solvers = {name: create_solver(mocp) if k == 0 else
                   AcadosOcpSolver(mocp, json_file=mocp.code_gen_options.json_file, generate=False, build=False,
                                   verbose=False)
                   for name, mocp in mocps.items()}
        warm, cold = solvers['warm'], solvers['cold']
        warm.options_set('filterddp_bound_mult_init_method', method)
        N = warm.N
        x = x0.copy()
        iters = {'warm': [], 'cold': []}
        error = 0.0
        for step in range(n_steps):
            for name, solver in (('warm', warm), ('cold', cold)):
                solver.set(0, 'lbx', x)
                solver.set(0, 'ubx', x)
                if name == 'warm' and step > 0 and solver.warm_start_from_policy(x) != 0:
                    print(f'FAIL warm_start_from_policy at step {step}')
                    ok = False
                status = solver.solve()
                iters[name].append(solver.get_stats('nlp_iter'))
                if status != 0:
                    print(f'FAIL {name} solve at step {step}: status {status}')
                    ok = False
            error = max(error, np.max(np.abs(trajectory(warm, 'u', range(N)) - trajectory(cold, 'u', range(N)))))
            x = warm.get(1, 'x')
        for name in ('warm', 'cold'):
            print(f'     {name}: iterations first {iters[name][0]}, then mean {np.mean(iters[name][1:]):.1f}, '
                  f'max {np.max(iters[name][1:])}')
        ok &= check('u warm against cold, all steps', error, 0.0, u_tol)
        if bounded_iterations:
            ok &= np.max(iters['warm'][1:]) <= iters['warm'][0]
    return ok


def main():
    ok = True
    ok &= compare_with_sqp('transition example: nx 2, transition stage nu 0, nx 1', transition_mocp)
    ok &= compare_with_sqp('piecewise polynomial controls: nu 1, nu 5', polynomial_mocp)
    ok &= compare_with_sqp('phase dependent nonlinear constraint', nonlinear_constraints_mocp)
    # the shift crosses the transition stage, where the stage before it and the transition stage keep their own
    # rules; controls of magnitude up to 50, weakly determined at the tolerance 1e-8 (see above)
    ex = load('mocp_transition_main', 'mocp_transition_example/main.py')
    ok &= closed_loop('transition example', lambda t, warm_start=False: transition_mocp(t, warm_start, tol=TOL),
                      ex.X0, u_tol=1e-4, bounded_iterations=True)
    # phases of the same dimensions with different constraint rows: the last stage of phase 0 takes the control
    # rule of the first stage of phase 1, without the velocity row of phase 0 or with the velocity row of phase 1
    # that it does not have; the slacks start from the rows' values, the multipliers of mu_based by row index
    x0 = np.array([1.0, 0.25])  # the initial state of create_mocp
    ok &= closed_loop('nonlinear constraint in phase 1', nonlinear_constraints_mocp, x0, u_tol=1e-6, n_steps=20)
    ok &= closed_loop('nonlinear constraint in phase 0',
                      lambda t, warm_start=False: nonlinear_constraints_mocp(t, warm_start, constrained_phase=0), x0,
                      u_tol=1e-6, n_steps=20)
    print('\nall checks passed' if ok else '\nsome checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

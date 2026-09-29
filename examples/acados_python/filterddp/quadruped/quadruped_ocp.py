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
The Quadruped-PyMPC nominal centroidal NMPC (quadruped_model.py) as an acados OCP, for two solvers on
the same discrete NLP (N = 12, dt = 0.02 s, RK4, friction cones):

  FILTERDDP_GN   FILTERDDP with the dynamics Hessian dropped (exact_hess_dyn = 0); the cost is
                 quadratic and the cone linear, so this is Gauss-Newton. EXTERNAL cost.
  SQP_GN         Quadruped-PyMPC's own solver settings: SQP, LINEAR_LS cost, Gauss-Newton,
                 Levenberg-Marquardt 1e-3, full steps, HPIPM balance. Upstream runs 1 iteration.

The iteration cap is set per solve at run time; max_iter here is the build cap (FILTERDDP sizes its
memory from it).
"""

import numpy as np
from acados_template import AcadosOcp

from quadruped_model import DT, GRAVITY, N_HORIZON, NP, NP_MODEL, NU, NX, P_YREF, export_model, friction_cone_bounds, weights

SOLVERS = ('FILTERDDP_GN', 'SQP_GN')
N = N_HORIZON
GO2_MASS = 15.206            # kg, Quadruped-PyMPC config for go2; the closed loop resets the cone bound from it


def setup_ocp(solver_name, tol=1e-6, max_iter=300, print_level=0, f_max=GO2_MASS*GRAVITY):
    if solver_name not in SOLVERS:
        raise ValueError(f'{solver_name}: expected one of {SOLVERS}')
    linear_ls = solver_name == 'SQP_GN'
    ocp = AcadosOcp()
    ocp.model = export_model(f'quadruped_{solver_name.lower()}', 'LINEAR_LS' if linear_ls else 'EXTERNAL')
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N*DT
    ocp.parameter_values = np.zeros(NP)

    if linear_ls:
        Q, R = weights()
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.block([[Q, np.zeros((NX, NU))], [np.zeros((NU, NX)), R]])
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.vstack((np.eye(NX), np.zeros((NU, NX))))
        ocp.cost.Vu = np.vstack((np.zeros((NX, NU)), np.eye(NU)))
        ocp.cost.Vx_e = np.eye(NX)
        ocp.cost.yref = np.zeros(NX + NU)
        ocp.cost.yref_e = np.zeros(NX)
        ocp.solver_options.cost_scaling = np.concatenate((DT*np.ones(N), [1.0]))
    else:
        # dt is inside the EXTERNAL stage cost
        ocp.cost.cost_type = ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.solver_options.cost_scaling = np.ones(N + 1)

    lh, uh = friction_cone_bounds(f_max)
    ocp.constraints.lh, ocp.constraints.uh = lh, uh
    ocp.constraints.lh_0, ocp.constraints.uh_0 = lh, uh
    ocp.constraints.x0 = np.zeros(NX)

    opts = ocp.solver_options
    opts.integrator_type = 'DISCRETE'
    opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    opts.qp_solver_cond_N = N
    opts.nlp_solver_max_iter = max_iter
    opts.nlp_solver_tol_stat = opts.nlp_solver_tol_eq = opts.nlp_solver_tol_ineq = opts.nlp_solver_tol_comp = tol
    opts.print_level = print_level
    if linear_ls:
        opts.nlp_solver_type = 'SQP'
        opts.hessian_approx = 'GAUSS_NEWTON'
        opts.levenberg_marquardt = 1e-3
        opts.globalization = 'FIXED_STEP'
        opts.hpipm_mode = 'BALANCE'
        opts.qp_solver_iter_max = 100
    else:
        opts.nlp_solver_type = 'FILTERDDP'
        opts.hessian_approx = 'EXACT'
        opts.exact_hess_dyn = 0
        opts.regularize_method = 'NO_REGULARIZE'
        opts.globalization = 'FIXED_STEP'
    ocp.code_gen_options.code_export_directory = f'c_generated_code_quadruped_{solver_name.lower()}'
    ocp.code_gen_options.json_file = f'quadruped_{solver_name.lower()}_ocp.json'
    return ocp


def set_call(solver, solver_name, x0, P):
    """Initial state and the (N+1) x 83 stage parameters of one MPC call; SQP_GN also takes the stage
    references as LINEAR_LS yref."""
    solver.set(0, 'lbx', x0)
    solver.set(0, 'ubx', x0)
    for k in range(N + 1):
        solver.set(k, 'p', P[k])
    if solver_name == 'SQP_GN':
        for k in range(N):
            solver.cost_set(k, 'yref', P[k, P_YREF])
        solver.cost_set(N, 'yref', P[N, NP_MODEL:NP_MODEL + NX])


def set_int_option(solver, field, value):
    """Set an integer NLP solver option at run time through the C interface (the filterddp_* options
    are not all exposed by AcadosOcpSolver.options_set)."""
    from ctypes import byref, c_char_p, c_int, c_void_p
    lib = solver._AcadosOcpSolver__acados_lib
    lib.ocp_nlp_solver_opts_set.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]
    lib.ocp_nlp_solver_opts_set.restype = None
    lib.ocp_nlp_solver_opts_set(solver.nlp_config, solver.nlp_opts, field.encode('utf-8'), byref(c_int(value)))


# FILTERDDP options, set explicitly rather than left to the library defaults: costate in the
# stationarity residual, value Hessian symmetrised after each stage, every solve initialized from the
# previous iterate. A variant is a '+'-separated list of flags, e.g. 'vg+ws':
#   vg  value gradient instead of the costate in the stationarity residual (the default of filterddp-nmpc)
#   ws  warm start from the previous solve's affine policy shifted by one stage, with its slacks,
#       multipliers and barrier parameter (filterddp_warm_start)
FILTERDDP_OPTIONS = {'filterddp_value_gradient_stationarity': 0, 'filterddp_symmetric_value_hessian': 1,
                     'filterddp_warm_start': 0}
FILTERDDP_FLAGS = {'vg': {'filterddp_value_gradient_stationarity': 1}, 'ws': {'filterddp_warm_start': 1}}


def filterddp_options(variant=None):
    opts = dict(FILTERDDP_OPTIONS)
    for flag in (variant.split('+') if variant else []):
        if flag not in FILTERDDP_FLAGS:
            raise ValueError(f'unknown FILTERDDP variant flag {flag!r}, expected one of {sorted(FILTERDDP_FLAGS)}')
        opts.update(FILTERDDP_FLAGS[flag])
    return opts


def apply_filterddp_options(solver, variant=None):
    for field, value in filterddp_options(variant).items():
        set_int_option(solver, field, value)

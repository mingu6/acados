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
FILTERDDP options on the unicycle of test_filterddp.py: the separate termination tolerances, the FILTERDDP
options through code generation and AcadosOcpSolver.options_set, and the rejection of options of the
SQP-type solvers that FILTERDDP does not use.
"""

import sys

import numpy as np
from acados_template import AcadosOcpOptions, AcadosOcpSolver

from test_filterddp import N, X0, check, setup, solve, trajectory

FILTERDDP_OPTIONS = [name for name in dir(AcadosOcpOptions) if name.startswith('filterddp_')]


def final_errors(solver: AcadosOcpSolver) -> dict:
    stat = solver.get_stats('statistics')
    return dict(du_inf=stat[1][-1], pr_inf=stat[2][-1], cs_inf=stat[3][-1])


def main():
    ok = True

    # default options, tolerance 1e-7 on all four errors
    ocp = setup('FILTERDDP')
    ocp.make_consistent()  # a second call in the solver creation must not reject the defaults it filled in
    default = AcadosOcpSolver(ocp, json_file='test_options_default_ocp.json', verbose=False)
    status = solve(default, X0)
    iter_default = default.get_stats('nlp_iter')
    print(f'default options: status {status}, {iter_default} iterations, final errors {final_errors(default)}')
    ok &= status == 0

    # through code generation: a complementarity tolerance below the one the default options end at, and other
    # FILTERDDP parameters
    ocp = setup('FILTERDDP')
    opts = ocp.solver_options
    opts.nlp_solver_tol_comp = 1e-8
    opts.filterddp_mu_init = 0.1
    opts.filterddp_tau_min = 0.995
    opts.filterddp_kappa_eps = 5.0
    opts.filterddp_symmetric_value_hessian = False
    ocp.model.name = 'filterddp_unicycle_options'
    ocp.code_gen_options.code_export_directory = 'c_generated_code_test_options'
    tuned = AcadosOcpSolver(ocp, json_file='test_options_tuned_ocp.json', verbose=False)
    status = solve(tuned, X0)
    errors = final_errors(tuned)
    print(f'tol_comp 1e-8, mu_init 0.1, tau_min 0.995, kappa_eps 5: status {status}, '
          f'{tuned.get_stats("nlp_iter")} iterations, final errors {errors}')
    ok &= status == 0
    ok &= errors['cs_inf'] < 1e-8 and errors['du_inf'] < 1e-7 and errors['pr_inf'] < 1e-7
    ok &= tuned.get_stats('nlp_iter') != iter_default
    ok &= check('x against default options', trajectory(tuned, 'x', range(N + 1)), trajectory(default, 'x', range(N + 1)), 1e-6)
    ok &= check('u against default options', trajectory(tuned, 'u', range(N)), trajectory(default, 'u', range(N)), 1e-5)

    # at run time: every FILTERDDP option, set to its default, leaves the solve unchanged
    defaults = AcadosOcpOptions()
    for name in FILTERDDP_OPTIONS:
        value = getattr(defaults, name)
        default.options_set(name, int(value) if isinstance(value, bool) else value)
    status = solve(default, X0)
    print(f'options_set of {len(FILTERDDP_OPTIONS)} options to their defaults: status {status}, '
          f'{default.get_stats("nlp_iter")} iterations')
    ok &= status == 0 and default.get_stats('nlp_iter') == iter_default
    default.options_set('filterddp_mu_init', 0.01)
    status = solve(default, X0)
    print(f'options_set filterddp_mu_init 0.01: status {status}, {default.get_stats("nlp_iter")} iterations')
    ok &= status == 0 and default.get_stats('nlp_iter') != iter_default

    # options FILTERDDP does not use are rejected
    for name, value in (('qp_solver_iter_max', 100), ('globalization_alpha_min', 0.5), ('levenberg_marquardt', 1e-3),
                        ('tau_min', 1e-6), ('globalization_fixed_step_length', 0.5)):
        ocp = setup('FILTERDDP')
        setattr(ocp.solver_options, name, value)
        try:
            ocp.make_consistent()
            print(f'FAIL {name} = {value} was accepted')
            ok = False
        except NotImplementedError as error:
            print(f'ok   {name} = {value} rejected: {error}')

    print('all checks passed' if ok else 'some checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

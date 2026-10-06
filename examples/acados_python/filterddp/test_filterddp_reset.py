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
reset() of FILTERDDP on the unicycle of test_filterddp.py: after a solve that ends with NaN, reset() returns the
solver to the state of a new one, so that the next solve is bit for bit the first solve of a new solver, and a
warm-started solve after reset() starts cold instead of from the policy of the solves before.
"""

import sys

import numpy as np
from acados_template import AcadosOcpSolver

from test_filterddp import N, NU, NX, X0, setup, solve
from test_filterddp_timeout import identical, report, result

ACADOS_SUCCESS = 0
ACADOS_NAN_DETECTED = 1
JSON_FILE = 'test_filterddp_reset_ocp.json'


def new_solver(ocp, build: bool = False) -> AcadosOcpSolver:
    """a new solver; all but the first on the code generated and built for the first"""
    return AcadosOcpSolver(ocp, json_file=JSON_FILE, build=build, generate=build, verbose=False)


def main():
    ok = True

    ocp = setup('FILTERDDP')
    ocp.code_gen_options.code_export_directory = 'c_generated_code_test_filterddp_reset'
    reference_solver = new_solver(ocp, build=True)
    solver = new_solver(ocp)

    # the first solve of a new solver
    solve(reference_solver, X0)
    reference = result(reference_solver)
    ok &= report('reference converges', reference['status'] == ACADOS_SUCCESS)

    # a solve from a NaN initial guess of the controls ends with NaN in the iterate and the workspace
    solve(solver, X0)
    for stage in range(N):
        solver.set(stage, 'u', np.full(NU, np.nan))
    status = solver.solve()
    ok &= report('NaN initial guess: status ACADOS_NAN_DETECTED', status == ACADOS_NAN_DETECTED)
    solver.reset()
    solve(solver, X0)
    ok &= report('after reset(): the first solve of a new solver, bit for bit', identical(result(solver), reference))

    # the converged solves of both solvers leave a policy; a warm-started solve shifts it, unless reset() came between
    x1 = reference['x'][NX:2*NX]
    for s in (reference_solver, solver):
        s.options_set('filterddp_warm_start', 1)
    solve(reference_solver, x1)
    ok &= report('warm start without reset(): shifts the policy',
                 reference_solver.get_stats('filterddp_warm_started') == 1)
    solver.reset()
    solve(solver, x1)
    cold_solver = new_solver(ocp)
    cold_solver.options_set('filterddp_warm_start', 1)
    solve(cold_solver, x1)
    ok &= report('warm start after reset(): cold, the first solve of a new solver, bit for bit',
                 solver.get_stats('filterddp_warm_started') == 0 and identical(result(solver), result(cold_solver)))

    print('all checks passed' if ok else 'some checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

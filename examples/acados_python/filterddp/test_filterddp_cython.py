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
warm_start_from_policy through the Cython wrapper (AcadosOcpSolverCython) on the unicycle of test_filterddp.py:
a receding horizon from the predicted next state, warm started with both bound_mult_init_method values, gives bit
for bit the solves of the ctypes wrapper (AcadosOcpSolver), with the same status codes: the policy is taken, a
second call without a solve in between finds none (ACADOS_READY), and neither does a call after reset().
"""

import sys

import numpy as np
from acados_template import AcadosOcpSolver

from test_filterddp import N, X0, setup, solve, trajectory

ACADOS_SUCCESS = 0
ACADOS_READY = 5


def result(solver) -> dict:
    return {'iter': solver.get_stats('sqp_iter'), 'x': trajectory(solver, 'x', range(N + 1)),
            'u': trajectory(solver, 'u', range(N)), 'pi': trajectory(solver, 'pi', range(N)),
            'lam': trajectory(solver, 'lam', range(N))}


def report(name: str, ok: bool) -> bool:
    print(f'{"ok  " if ok else "FAIL"} {name}')
    return ok


def main():
    ok = True
    solvers = {}
    for wrapper in ('ctypes', 'cython'):
        ocp = setup('FILTERDDP')
        ocp.code_gen_options.code_export_directory = f'c_generated_code_test_filterddp_{wrapper}'
        ocp.code_gen_options.json_file = f'test_filterddp_{wrapper}_ocp.json'
        if wrapper == 'ctypes':
            solvers[wrapper] = AcadosOcpSolver(ocp, json_file=ocp.code_gen_options.json_file, verbose=False)
        else:
            solvers[wrapper] = AcadosOcpSolver.create_cython_solver(ocp, verbose=False)

    for method in ('constant', 'mu_based'):
        codes, results = {}, {}
        for wrapper, s in solvers.items():
            s.reset()
            s.options_set('filterddp_bound_mult_init_method', 'constant')
            solve(s, X0)
            s.options_set('filterddp_bound_mult_init_method', method)
            codes[wrapper], results[wrapper] = [], []
            x = s.get(1, 'x')
            for _ in range(10):
                s.set(0, 'lbx', x)
                s.set(0, 'ubx', x)
                codes[wrapper].append(s.warm_start_from_policy(x))
                codes[wrapper].append(s.warm_start_from_policy(x))
                codes[wrapper].append(s.solve())
                results[wrapper].append(result(s))
                x = s.get(1, 'x')
            s.reset()
            codes[wrapper].append(s.warm_start_from_policy(x))
        expected = [ACADOS_SUCCESS, ACADOS_READY, ACADOS_SUCCESS]*10 + [ACADOS_READY]
        ok &= report(f'{method}: status codes of both wrappers (taken, refused once taken, solved; refused after '
                     'reset())', codes['ctypes'] == codes['cython'] == expected)
        same = all(all(np.array_equal(a[f], b[f]) for f in a) for a, b in zip(results['ctypes'], results['cython']))
        ok &= report(f'{method}: ctypes and Cython warm-started solves identical, bit for bit', same)

    print('all checks passed' if ok else 'some checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

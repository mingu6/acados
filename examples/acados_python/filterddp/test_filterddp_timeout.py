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
Timeout of FILTERDDP (timeout_max_time, timeout_heuristic, set at code generation) on the unicycle of
test_filterddp.py: an unreached timeout gives the same result as no timeout; a timeout returns status
ACADOS_TIMEOUT with the iterate of a solve capped at as many iterations (an accepted iterate, strictly inside
the control bounds, with positive inequality multipliers); the warm start after a timeout (warm_start_from_policy)
takes the update rules of the returned iterate; the heuristics run. The part-way timeouts are fractions of the
measured solve time.
"""

import sys

import numpy as np
from acados_template import AcadosOcpSolver

from test_filterddp import N, NU, X0, setup, solve, trajectory

ACADOS_SUCCESS = 0
ACADOS_TIMEOUT = 7


def result(solver: AcadosOcpSolver) -> dict:
    return {'status': solver.status, 'iter': solver.get_stats('nlp_iter'),
            'x': trajectory(solver, 'x', range(N + 1)), 'u': trajectory(solver, 'u', range(N)),
            'pi': trajectory(solver, 'pi', range(N)), 'lam': trajectory(solver, 'lam', range(N)),
            'K': np.concatenate([solver.get_from_qp_in(i, 'K').ravel() for i in range(N)]),
            'k': np.concatenate([solver.get_from_qp_in(i, 'k').ravel() for i in range(N)])}


def identical(a: dict, b: dict, fields=('status', 'iter', 'x', 'u', 'pi', 'lam', 'K', 'k')) -> bool:
    return all(np.array_equal(a[f], b[f]) for f in fields)


def interior(solver: AcadosOcpSolver, r: dict) -> bool:
    """finite, controls strictly inside their bounds, inequality multipliers positive"""
    ok = all(np.all(np.isfinite(r[f])) for f in ('x', 'u', 'pi', 'lam', 'K', 'k'))
    ok &= np.all(np.abs(r['u'].reshape(N, NU)[:, :2]) < 1.0)
    for stage in range(N):
        lam = solver.get(stage, 'lam')
        nbx = 4 if stage == 0 else 1
        lower, upper = lam[:lam.size//2], lam[lam.size//2:]
        ok &= np.all(lam >= 0.0)
        # bounds on u0, u1 and on the speed (stages 1..N-1), and the lower side of the obstacle row
        ok &= np.all(lower[:2] > 0.0) and np.all(upper[:2] > 0.0) and lower[2+nbx] > 0.0
        if stage > 0:
            ok &= lower[2] > 0.0 and upper[2] > 0.0
    return bool(ok)


def report(name: str, ok: bool) -> bool:
    print(f'{"ok  " if ok else "FAIL"} {name}')
    return ok


def build(name: str, timeout: float = 0.0, heuristic: str = 'ZERO') -> AcadosOcpSolver:
    """unicycle solver with the timeout options set at code generation"""
    ocp = setup('FILTERDDP')
    ocp.solver_options.timeout_max_time = timeout
    ocp.solver_options.timeout_heuristic = heuristic
    ocp.code_gen_options.code_export_directory = f'c_generated_code_test_filterddp_timeout_{name}'
    return AcadosOcpSolver(ocp, json_file=f'test_filterddp_timeout_{name}_ocp.json', verbose=False)


def warm_solve(solver: AcadosOcpSolver, x0: np.ndarray) -> int:
    """the policy of the last solve rolled out from x0, then the solve; -1 if there is no policy to take"""
    solver.constraints_set(0, 'lbx', x0)
    solver.constraints_set(0, 'ubx', x0)
    if solver.warm_start_from_policy(x0) != ACADOS_SUCCESS:
        return -1
    return solver.solve()


def main():
    ok = True

    # reference without timeout; the fastest of a few solves sets the time scale
    reference_solver = build('reference')
    times = []
    for _ in range(10):
        solve(reference_solver, X0)
        times.append(reference_solver.get_stats('time_tot'))
    reference = result(reference_solver)
    t_ref = min(times)
    print(f'reference: status {reference["status"]}, {reference["iter"]} iterations, {1e3*t_ref:.3f} ms')
    ok &= report('reference converges', reference['status'] == ACADOS_SUCCESS and reference['iter'] > 10)

    # a timeout that is never reached gives the same result as no timeout
    solver = build('unreached', 1e3, 'MAX_OVERALL')
    solve(solver, X0)
    ok &= report('unreached timeout identical to no timeout', identical(result(solver), reference))

    # a timeout part way, at 40 % of the fastest reference solve: the interior iterate of a solve capped at as
    # many iterations, and a warm-started solve from it
    timeout = 0.4*t_ref
    solver = build('partway', timeout, 'ZERO')
    solve(solver, X0)
    r = result(solver)
    print(f'timeout {1e3*timeout:.3f} ms: status {r["status"]}, {r["iter"]} iterations, '
          f'time_tot {1e3*solver.get_stats("time_tot"):.3f} ms')
    ok &= report('status ACADOS_TIMEOUT after fewer iterations, interior',
                 r['status'] == ACADOS_TIMEOUT and r['iter'] < reference['iter'] and interior(solver, r))
    ok &= report('time_tot reaches timeout_max_time (heuristic ZERO)', solver.get_stats('time_tot') >= timeout)
    if r['iter'] > 0:
        reference_solver.options_set('max_iter', r['iter'])
        solve(reference_solver, X0)
        ok &= report(f'iterate equals that of a solve capped at {r["iter"]} iterations',
                     identical(r, result(reference_solver), ('x', 'u')))
    status = warm_solve(solver, solver.get(1, 'x'))
    w = result(solver)
    print(f'warm start after the timeout: status {status}, {w["iter"]} iterations')
    ok &= report('warm-started solve after the timeout: iterations, interior',
                 status in (ACADOS_SUCCESS, ACADOS_TIMEOUT) and w['iter'] > 0 and interior(solver, w))

    # a timeout before the first forward pass returns the initial iterate, with the update rules of the backward
    # pass at it: the warm start shifts the same rules as after a solve capped at 0 iterations, whose extra
    # backward pass recomputes them at the returned iterate
    solver = build('immediate', 1e-9, 'LAST')
    solve(solver, X0)
    r = result(solver)
    ok &= report('timeout before the first iteration: status ACADOS_TIMEOUT, 0 iterations, interior',
                 r['status'] == ACADOS_TIMEOUT and r['iter'] == 0 and interior(solver, r))
    reference_solver.options_set('max_iter', 0)
    solve(reference_solver, X0)
    fields = ('x', 'u', 'pi', 'lam', 'K', 'k')
    ok &= report('initial iterate and rules equal those of a solve capped at 0 iterations',
                 identical(r, result(reference_solver), fields))
    x1 = reference_solver.get(1, 'x')
    warm_solve(solver, x1)
    warm_solve(reference_solver, x1)
    ok &= report('warm start after the timeout equals that after the capped solve',
                 identical(result(solver), result(reference_solver), fields) and interior(solver, result(solver)))
    reference_solver.options_set('max_iter', setup('FILTERDDP').solver_options.nlp_solver_max_iter)

    # the other heuristics stop a solve with a tight timeout, at an interior iterate
    for heuristic in ('MAX_CALL', 'LAST', 'AVERAGE'):
        solver = build(heuristic.lower(), 0.5*t_ref, heuristic)
        solve(solver, X0)
        r = result(solver)
        print(f'{heuristic:8s}: status {r["status"]}, {r["iter"]} iterations')
        ok &= report(f'{heuristic:8s}: timeout, interior',
                     r['status'] == ACADOS_TIMEOUT and r['iter'] < reference['iter'] and interior(solver, r))

    print('all checks passed' if ok else 'some checks FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

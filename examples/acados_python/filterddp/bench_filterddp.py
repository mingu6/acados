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
Benchmark FILTERDDP on the acrobot problems over all parameter sets of FilterDDP.jl, timed as in
the FilterDDP.jl experiments: one warm-up solve, then the wall time of a second solve from the same
initial guess. Writes "seed iterations status objective primal wall_ms wall_min_ms" per line.

    bench_filterddp.py <rk4|contact> <params_file> <results_file> [nrepeat]
"""

import sys
import time

import numpy as np
from acados_template import AcadosOcpSolver

from acrobot_filterddp import initialize, setup_contact, setup_rk4
from acrobot_model import load_parameter_sets


def main():
    problem, params_file, results_file = sys.argv[1], sys.argv[2], sys.argv[3]
    nrepeat = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    if problem == 'rk4':
        ocp = setup_rk4(None, 1e-7, 1000, 0)
        u_init = np.zeros(1)
    else:
        ocp = setup_contact(1e-7, 1000, 0)
        u_init = np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01])
    nu = ocp.model.u.rows()
    solver = AcadosOcpSolver(ocp, json_file=f'bench_{problem}_ocp.json', verbose=False)
    parameter_sets = load_parameter_sets(params_file)

    with open(results_file, 'w') as f:
        f.write('seed iterations status objective primal wall_ms wall_min_ms\n')
        for seed, parameters in enumerate(parameter_sets, start=1):
            initialize(solver, nu, parameters, u_init)
            solver.solve()
            walls = []
            for _ in range(nrepeat):
                initialize(solver, nu, parameters, u_init)
                t0 = time.perf_counter()
                status = solver.solve()
                walls.append(1e3*(time.perf_counter() - t0))
            iters = solver.get_stats('nlp_iter')
            cost = solver.get_cost()
            primal = solver.get_stats('statistics')[2, -1]
            f.write(f'{seed} {iters} {status} {cost:.12e} {primal:.12e} {walls[0]:.6f} {min(walls):.6f}\n')
            print(f'{problem} seed {seed:3d}: status {status} iters {iters:4d} obj {cost:.8e} wall {walls[0]:.2f} ms (min {min(walls):.2f})')


if __name__ == '__main__':
    main()

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
Solve the unconstrained acrobot RK4 problem with FILTERDDP a given number of times from the
same initial guess and print the statistics of each solve, to compare against FilterDDP.jl.
"""

import sys

import numpy as np
from acados_template import AcadosOcpSolver

from acrobot_filterddp import initialize, setup_rk4
from acrobot_model import DEFAULT_PARAMETERS, NX, load_parameter_sets


def main():
    params = load_parameter_sets(sys.argv[1])[0] if len(sys.argv) > 1 else DEFAULT_PARAMETERS
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_solves = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    ocp = setup_rk4(None, 1e-7, max_iter, 0)
    solver = AcadosOcpSolver(ocp, json_file='check_rk4_ocp.json', verbose=False)
    np.set_printoptions(precision=6, linewidth=200)
    for k in range(n_solves):
        initialize(solver, 1, params, np.zeros(1))
        status = solver.solve()
        stats = solver.get_stats('statistics')
        print(f'solve {k}: status {status} iters {solver.get_stats("nlp_iter")} cost {solver.get_cost():.10e}')
        print('  columns: iter du_inf pr_inf cs_inf objective mu reg alpha ls')
        print(stats.T[:4])
    print('x[N]:', solver.get(ocp.solver_options.N_horizon, 'x'))


if __name__ == '__main__':
    main()

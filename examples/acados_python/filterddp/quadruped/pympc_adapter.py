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
Drop-in replacements for the AcadosOcpSolver object that Quadruped-PyMPC's Acados_NMPC_Nominal keeps
in `self.acados_ocp_solver`, so that its unmodified compute_control drives a different solver.

  StockRecorder   wraps Quadruped-PyMPC's own solver unchanged and records every call.
  SolverAdapter   forwards to one of the solvers of quadruped_ocp.setup_ocp (FILTERDDP_GN or
                  SQP_GN): the 29 model parameters and the stage references that compute_control
                  sets are merged into the 83-entry parameter vector of quadruped_model; bounds, x0 and
                  getters pass through. The iteration cap is set at run time (it must not exceed the
                  cap the solver was built with, since FILTERDDP sizes its memory from it). Whatever
                  the solver returns is used, including at the cap; compute_control itself only falls
                  back to its nominal plan for status 1 (NaN) and 4 (QP / backward pass failure).

Both record, per solve: x0, parameters, references, status, iterations, solve time and the first
control (mujoco_closed_loop.py --log_episodes writes the calls to output/calls/).

Initialization is each solver's default: the previous solution, unshifted, as upstream (its
use_warm_start is off). FILTERDDP additionally re-initializes its slacks, bound duals and barrier
parameter from that iterate at every solve.
"""

import time

import numpy as np

from quadruped_model import N_HORIZON, NH, NP, NP_MODEL, NU, NX, P_YREF, friction_cone_bounds

N = N_HORIZON


class _Recorder:
    def __init__(self, record_instances):
        self.record_instances = record_instances
        self.p_model = np.zeros((N + 1, NP_MODEL))
        self.yref = np.zeros((N, NX + NU))
        self.yref_e = np.zeros(NX)
        self.x0 = np.zeros(NX)
        self.calls = []            # (status, iters, solve_ms) per solve
        self.instances = []        # per solve: dict of arrays, if record_instances

    def _capture(self, stage, field, value):
        value = np.asarray(value, dtype=float).flatten()
        if field == 'p':
            self.p_model[stage] = value[:NP_MODEL]
        elif field == 'yref':
            if stage < N:
                self.yref[stage] = value[:NX + NU]
            else:
                self.yref_e = value[:NX].copy()
        elif field == 'lbx' and stage == 0:
            self.x0 = value.copy()

    def parameters(self):
        P = np.zeros((N + 1, NP))
        P[:, :NP_MODEL] = self.p_model
        P[N, :NP_MODEL] = self.p_model[N - 1]
        P[:N, P_YREF] = self.yref
        P[N, NP_MODEL:NP_MODEL + NX] = self.yref_e
        return P

    def _record(self, status, iters, ms, u0):
        self.calls.append((status, iters, ms))
        if self.record_instances:
            self.instances.append(dict(x0=self.x0.copy(), P=self.parameters(), status=status, iters=iters,
                                       ms=ms, u0=np.asarray(u0).copy()))


class StockRecorder(_Recorder):
    """Quadruped-PyMPC's own solver, unchanged, with call recording."""

    def __init__(self, solver, record_instances=False):
        super().__init__(record_instances)
        self.solver = solver

    def set(self, stage, field, value):
        self._capture(stage, field, value)
        return self.solver.set(stage, field, value)

    def cost_set(self, stage, field, value, *args, **kwargs):
        self._capture(stage, field, value)
        return self.solver.cost_set(stage, field, value, *args, **kwargs)

    def solve(self):
        t0 = time.perf_counter()
        status = self.solver.solve()
        ms = 1e3*(time.perf_counter() - t0)
        self._record(status, int(self.solver.get_stats('nlp_iter')), ms, self.solver.get(0, 'u'))
        return status

    def __getattr__(self, name):
        return getattr(self.solver, name)


class SolverAdapter(_Recorder):
    """One of our solvers behind the interface compute_control uses."""

    def __init__(self, solver_name, solver, f_max, max_iter=None, record_instances=False):
        super().__init__(record_instances)
        self.name = solver_name
        self.solver = solver
        self.linear_ls = solver_name == 'SQP_GN'
        self.f_max = f_max
        if max_iter is not None:
            self.solver.options_set('max_iter', int(max_iter))
        self.reset()

    def reset(self):
        self.solver.reset()
        lh, uh = friction_cone_bounds(self.f_max)
        for k in range(N):
            self.solver.constraints_set(k, 'lh', lh)
            self.solver.constraints_set(k, 'uh', uh)
        for k in range(N + 1):
            self.solver.set(k, 'x', np.zeros(NX))
        for k in range(N):
            self.solver.set(k, 'u', np.zeros(NU))

    def set(self, stage, field, value):
        self._capture(stage, field, value)
        if field in ('p', 'yref'):
            return
        self.solver.set(stage, field, value)

    def cost_set(self, stage, field, value, *args, **kwargs):
        self._capture(stage, field, value)
        if field != 'yref':
            self.solver.cost_set(stage, field, value, *args, **kwargs)

    def constraints_set(self, stage, field, value, *args, **kwargs):
        self.solver.constraints_set(stage, field, value, *args, **kwargs)

    def options_set(self, field, value):
        if field == 'rti_phase':          # upstream RTI switch; our solvers are not RTI
            return
        self.solver.options_set(field, value)

    def solve(self):
        P = self.parameters()
        for k in range(N + 1):
            self.solver.set(k, 'p', P[k])
        if self.linear_ls:
            for k in range(N):
                self.solver.cost_set(k, 'yref', self.yref[k])
            self.solver.cost_set(N, 'yref', self.yref_e)
        t0 = time.perf_counter()
        status = self.solver.solve()
        ms = 1e3*(time.perf_counter() - t0)
        self._record(status, int(self.solver.get_stats('nlp_iter')), ms, self.solver.get(0, 'u'))
        return status

    def get(self, stage, field):
        return self.solver.get(stage, field)

    def get_stats(self, field):
        return self.solver.get_stats(field)

    def print_statistics(self):
        return self.solver.print_statistics()

    def get_from_qp_in(self, stage, field):
        return self.solver.get_from_qp_in(stage, field)


def summarize_calls(calls, budget_ms=10.0):
    if not calls:
        return {}
    a = np.array(calls, dtype=float)
    status, iters, ms = a[:, 0].astype(int), a[:, 1], a[:, 2]
    return dict(n_calls=len(a), status0=int(np.sum(status == 0)), status2=int(np.sum(status == 2)),
                status_other=int(np.sum((status != 0) & (status != 2))), iters_mean=float(iters.mean()),
                iters_max=int(iters.max()), ms_mean=float(ms.mean()), ms_p99=float(np.percentile(ms, 99)),
                ms_max=float(ms.max()), over_budget=int(np.sum(ms > budget_ms)))

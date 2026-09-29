"""CPC time-optimal waypoint flight as an acados FilterDDP OCP with a terminal progress cost.

The CPC NLP (Foehn et al., Science Robotics 2021; ``src/planner.py`` of
https://github.com/uzh-rpg/rpg_time_optimal, revision f5541b6) is posed per stage
``k = 0..N-1`` with

    x = [xq = (p, v, q, omega) (13), mu (NW), t_prev (1)]
    u = [T (4), y (13), lambda (NW), tau (NW), t (1)]
    f(x, u) = [y, mu - lambda, t]

``y`` is the next quadrotor state (lifted dynamics); ``y - RK4(xq, T, t/N) = 0`` are stage
equalities. The CPC constraints on ``x_{k+1}`` are rows of the stage-``k`` ``h`` written in
``y``, so indices match the CPC NLP. Deviations from CPC:

* ``t`` is a control carried in the state, with the equality ``t_k - t_prev = 0`` at
  ``k >= 1``; the stage cost ``t/N`` sums to ``t``.
* CPC's terminal constraint ``mu_N = 0`` is the terminal cost ``(w/2) ||mu_N||^2``
  (FilterDDP has no terminal constraints). The penalty is inexact: ``mu_N ~ 1e-5`` at
  ``w = 1e3``.
"""

from __future__ import annotations

from ctypes import byref, c_double, c_int
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from quad_model import NQ, NT, QuadParameters, Track, rk4_step

T_BOUNDS = (0.1, 150.0)         # planner.py total-time bounds
Z_BOUNDS = (0.5, 100.0)         # planner.py altitude bounds
PROGRESS_UPPER = 0.01           # planner.py complementarity relaxation


@dataclass
class WaypointOptions:
    nodes_per_gate: int = 30
    tolerance: float = 0.3       # waypoint tolerance [m]
    vel_guess: float = 3.0       # rpg example value
    endpoint_weight: float = 1e3  # w of the terminal cost (w/2) ||mu_N||^2
    tol: float = 1e-6
    max_iter: int = 3000
    print_level: int = 0
    mu_init: float = 1e-2        # fork default 1.0
    kappa: float = 1e-4          # filterddp_kappa_1 = kappa_2; fork default 1e-2


@dataclass
class Layout:
    nw: int

    @property
    def nx(self) -> int:
        return NQ + self.nw + 1

    @property
    def nu(self) -> int:
        return NT + NQ + 2*self.nw + 1

    # state
    @property
    def x_mu(self) -> slice:
        return slice(NQ, NQ + self.nw)

    @property
    def x_t(self) -> int:
        return NQ + self.nw

    # control
    @property
    def u_thrust(self) -> slice:
        return slice(0, NT)

    @property
    def u_next(self) -> slice:
        return slice(NT, NT + NQ)

    @property
    def u_lam(self) -> slice:
        return slice(NT + NQ, NT + NQ + self.nw)

    @property
    def u_tau(self) -> slice:
        return slice(NT + NQ + self.nw, NT + NQ + 2*self.nw)

    @property
    def u_t(self) -> int:
        return NT + NQ + 2*self.nw

    # h rows: [progress (NW), monotone (NW-1), omega (3), z (1), defect (13), t (1, k >= 1)]
    @property
    def h_progress(self) -> slice:
        return slice(0, self.nw)

    @property
    def h_monotone(self) -> slice:
        return slice(self.nw, 2*self.nw - 1)

    @property
    def h_omega(self) -> slice:
        return slice(2*self.nw - 1, 2*self.nw + 2)

    @property
    def h_z(self) -> int:
        return 2*self.nw + 2

    @property
    def nh0(self) -> int:
        return 2*self.nw + 3 + NQ

    @property
    def nh(self) -> int:
        return self.nh0 + 1


@dataclass
class WaypointProblem:
    quad: QuadParameters
    track: Track
    options: WaypointOptions
    layout: Layout
    wp: np.ndarray                   # (3, NW)
    x0: np.ndarray
    N: int
    thrust_max: np.ndarray           # (N,) ramped T_max(i)
    omega_max_xy: np.ndarray         # (N,) ramped omega_max_xy(i)
    ocp: AcadosOcp | None = None
    functions: dict = field(default_factory=dict)


def _interpolate(x1, y1, x2, y2, x):
    """``planner.py:Planner.interpolate``."""
    if abs(x2 - x1) < 1e-5:
        return 0
    return y1 + (y2 - y1)/(x2 - x1)*(x - x1)


def build_problem(quad: QuadParameters, track: Track, options: WaypointOptions | None = None) -> WaypointProblem:
    options = options or WaypointOptions()
    if track.ring or track.end_att is not None or track.end_vel is not None or track.end_omega is not None:
        raise NotImplementedError("ring and end attitude/velocity/omega constraints are not supported")
    missing = [n for n in ("init_pos", "init_att", "init_vel", "init_omega") if getattr(track, n) is None]
    if missing:
        raise ValueError(f"FilterDDP needs a fixed initial state; the track does not specify {missing}.")

    wp = track.waypoints()
    nw = wp.shape[1]
    p_init = np.array(track.init_pos, dtype=float)
    dist = np.cumsum(np.linalg.norm(np.diff(np.column_stack([p_init, wp]), axis=1), axis=0))
    N = options.nodes_per_gate*nw
    dpn = dist[-1]/N

    # planner.py ramps T_max and omega_max_xy over the first rampup_dist meters.
    thrust_max = np.full(N, quad.thrust_max)
    omega_max_xy = np.full(N, quad.omega_max_xy)
    if quad.rampup_dist > 0:
        for i in range(N):
            thrust_max[i] = max(min(_interpolate(0, quad.thrust_ramp_start, quad.rampup_dist,
                                                 quad.thrust_max, i*dpn), quad.thrust_max),
                                quad.thrust_ramp_start)
            omega_max_xy[i] = max(min(_interpolate(0, quad.omega_ramp_start, quad.rampup_dist,
                                                   quad.omega_max_xy, i*dpn), quad.omega_max_xy),
                                  quad.omega_ramp_start)

    t_guess = dist[-1]/options.vel_guess
    x0 = np.concatenate([p_init, np.array(track.init_vel, float), np.array(track.init_att, float),
                         np.array(track.init_omega, float), np.ones(nw), [t_guess]])
    problem = WaypointProblem(quad, track, options, Layout(nw), wp, x0, N, thrust_max, omega_max_xy)
    problem.ocp = make_ocp(problem)
    return problem


def symbolic_stage(problem: WaypointProblem):
    """Symbolic x, u, next state, RK4 prediction, h rows of stage 0 and stages >= 1, costs."""
    L, N = problem.layout, problem.N
    x = ca.SX.sym("x", L.nx)
    u = ca.SX.sym("u", L.nu)
    thrust, y, lam, tau, t = u[L.u_thrust], u[L.u_next], u[L.u_lam], u[L.u_tau], u[L.u_t]
    mu_next = x[L.x_mu] - lam
    x_next = ca.vertcat(y, mu_next, t)
    xq_rk4 = rk4_step(problem.quad, x[:NQ], thrust, t/N)

    progress = ca.vertcat(*[lam[j]*(ca.sumsqr(y[0:3] - problem.wp[:, j]) - tau[j]) for j in range(L.nw)])
    monotone = ca.vertcat(*[mu_next[j + 1] - mu_next[j] for j in range(L.nw - 1)]) if L.nw > 1 \
        else ca.SX(0, 1)
    h0 = ca.vertcat(progress, monotone, y[10:13], y[2], y - xq_rk4)
    h = ca.vertcat(h0, t - x[L.x_t])
    cost = t/N
    cost_e = 0.5*problem.options.endpoint_weight*ca.sumsqr(x[L.x_mu])
    return x, u, x_next, xq_rk4, h0, h, cost, cost_e


def stage_bounds(problem: WaypointProblem, k: int) -> tuple[np.ndarray, np.ndarray]:
    """lh, uh of stage k. Equality rows (defect, t) have lh == uh == 0."""
    L = problem.layout
    nh = L.nh0 if k == 0 else L.nh
    lh, uh = np.zeros(nh), np.zeros(nh)
    uh[L.h_progress] = PROGRESS_UPPER
    uh[L.h_monotone] = 1.0
    w = np.array([problem.omega_max_xy[k], problem.omega_max_xy[k], problem.quad.omega_max_z])
    lh[L.h_omega], uh[L.h_omega] = -w, w
    lh[L.h_z], uh[L.h_z] = Z_BOUNDS
    return lh, uh


def control_bounds(problem: WaypointProblem, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """idxbu, lbu, ubu of stage k: thrust, lambda, tau, t (y is unbounded)."""
    L, nw = problem.layout, problem.layout.nw
    idxbu = np.concatenate([np.arange(NT), np.arange(L.u_lam.start, L.u_t + 1)])
    lbu = np.concatenate([np.full(NT, problem.quad.thrust_min), np.zeros(2*nw), [T_BOUNDS[0]]])
    ubu = np.concatenate([np.full(NT, problem.thrust_max[k]), np.ones(nw),
                          np.full(nw, problem.options.tolerance**2), [T_BOUNDS[1]]])
    return idxbu, lbu, ubu


def make_ocp(problem: WaypointProblem) -> AcadosOcp:
    L, N, opts = problem.layout, problem.N, problem.options
    x, u, x_next, xq_rk4, h0, h, cost, cost_e = symbolic_stage(problem)
    problem.functions = {
        "rk4": ca.Function("rk4", [x, u], [xq_rk4]),
        "h0": ca.Function("h0", [x, u], [h0]),
        "h": ca.Function("h", [x, u], [h]),
        "cost_e": ca.Function("cost_e", [x], [cost_e]),
    }

    model = AcadosModel()
    model.name = f"waypoint_nw{L.nw}_n{N}"
    model.x, model.u = x, u
    model.disc_dyn_expr = x_next
    model.cost_expr_ext_cost_0 = cost
    model.cost_expr_ext_cost = cost
    model.cost_expr_ext_cost_e = cost_e
    model.con_h_expr_0 = h0
    model.con_h_expr = h

    ocp = AcadosOcp()
    ocp.model = model
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.constraints.x0 = problem.x0
    ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = control_bounds(problem, 1)
    ocp.constraints.lh_0, ocp.constraints.uh_0 = stage_bounds(problem, 0)
    ocp.constraints.lh, ocp.constraints.uh = stage_bounds(problem, 1)

    so = ocp.solver_options
    so.N_horizon = N
    so.tf = 1.0  # time enters through the control t; with cost_scaling = 1, tf has no effect
    so.integrator_type = "DISCRETE"
    so.nlp_solver_type = "FILTERDDP"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = N
    so.hessian_approx = "EXACT"
    so.regularize_method = "NO_REGULARIZE"
    so.globalization = "FIXED_STEP"
    so.nlp_solver_max_iter = opts.max_iter
    so.nlp_solver_tol_stat = opts.tol
    so.nlp_solver_tol_eq = opts.tol
    so.nlp_solver_tol_ineq = opts.tol
    so.nlp_solver_tol_comp = opts.tol
    so.print_level = opts.print_level
    so.cost_scaling = np.ones(N + 1)
    return ocp


def set_filterddp_option(solver: AcadosOcpSolver, name: str, value: float | int) -> None:
    """FilterDDP options are not in the Python template; set them through ctypes."""
    lib = solver._AcadosOcpSolver__acados_lib
    cval = c_int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
        else c_double(float(value))
    lib.ocp_nlp_solver_opts_set(solver.nlp_config, solver.nlp_opts, name.encode("utf-8"), byref(cval))


def create_solver(problem: WaypointProblem, output_dir: str | Path, verbose: bool = False) -> AcadosOcpSolver:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ocp = problem.ocp
    ocp.code_export_directory = str(output_dir / f"c_generated_code_{ocp.model.name}")
    # acados writes the json next to the generated code; a path is not accepted.
    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json", verbose=verbose)
    opts = problem.options
    set_filterddp_option(solver, "filterddp_mu_init", opts.mu_init)
    set_filterddp_option(solver, "filterddp_kappa_1", opts.kappa)
    set_filterddp_option(solver, "filterddp_kappa_2", opts.kappa)
    for k in range(problem.N):
        lh, uh = stage_bounds(problem, k)
        solver.constraints_set(k, "lh", lh)
        solver.constraints_set(k, "uh", uh)
        _, lbu, ubu = control_bounds(problem, k)
        solver.constraints_set(k, "lbu", lbu)
        solver.constraints_set(k, "ubu", ubu)
    return solver


def rollout(problem: WaypointProblem, us: np.ndarray) -> np.ndarray:
    """States of the lifted dynamics: x_{k+1} = [y_k, mu_k - lambda_k, t_k]."""
    L, N = problem.layout, problem.N
    xs = np.zeros((N + 1, L.nx))
    xs[0] = problem.x0
    for k in range(N):
        xs[k + 1] = np.concatenate([us[k, L.u_next], xs[k, L.x_mu] - us[k, L.u_lam], [us[k, L.u_t]]])
    return xs


def initialize_solver(problem: WaypointProblem, solver: AcadosOcpSolver, us: np.ndarray) -> None:
    xs = rollout(problem, us)
    for k in range(problem.N):
        solver.set(k, "x", xs[k])
        solver.set(k, "u", us[k])
    solver.set(problem.N, "x", xs[problem.N])


def get_solution(problem: WaypointProblem, solver: AcadosOcpSolver) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([solver.get(k, "x") for k in range(problem.N + 1)])
    us = np.array([solver.get(k, "u") for k in range(problem.N)])
    return xs, us


def evaluate(problem: WaypointProblem, xs: np.ndarray, us: np.ndarray) -> dict:
    """Residuals recomputed from (xs, us), independent of acados multipliers.

    Worst entries are ``(value, stage, index)``; ``mu_final`` is ``max |mu_N|``, the
    penalty residual of CPC's ``mu_N = 0``.
    """
    L, N = problem.layout, problem.N
    rk4, h0, h = problem.functions["rk4"], problem.functions["h0"], problem.functions["h"]
    worst = {"rk4_defect": (0.0, -1, -1), "equality": (0.0, -1, -1), "inequality": (0.0, -1, -1),
             "control_bound": (0.0, -1, -1)}

    def update(key, v, k, idx):
        j = int(np.argmax(v))
        if v[j] > worst[key][0]:
            worst[key] = (float(v[j]), k, int(idx[j]))

    for k in range(N):
        update("rk4_defect", np.abs(np.asarray(rk4(xs[k], us[k])).ravel() - xs[k + 1, :NQ]), k, np.arange(NQ))
        hk = np.asarray((h0 if k == 0 else h)(xs[k], us[k])).ravel()
        lh, uh = stage_bounds(problem, k)
        viol = np.maximum(lh - hk, hk - uh).clip(min=0.0)
        eq = lh == uh
        update("equality", viol*eq, k, np.arange(hk.size))
        update("inequality", viol*~eq, k, np.arange(hk.size))
        idxbu, lbu, ubu = control_bounds(problem, k)
        ub = us[k, idxbu]
        update("control_bound", np.maximum(lbu - ub, ub - ubu).clip(min=0.0), k, idxbu)
    t = us[:, L.u_t]
    closest = [float(np.min(np.linalg.norm(xs[1:, 0:3] - problem.wp[:, j][None, :], axis=1)))
               for j in range(L.nw)]
    return {"t": float(t[0]), "t_spread": float(np.max(t) - np.min(t)),
            "mu_final": float(np.max(np.abs(xs[-1, L.x_mu]))), "closest_approach": closest, **worst}

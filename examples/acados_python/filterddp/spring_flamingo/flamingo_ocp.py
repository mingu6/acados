"""Spring Flamingo OCP for acados FilterDDP (README.md).

A half-stride walking gait that minimizes a smooth mechanical cost of
transport, subject to variational contact-implicit mechanics,
maximum-dissipation friction, a speed-dependent swing clearance, and
mirror periodicity with a fixed start state.  The initial guess is a Bezier
stepping motion; no reference trajectory enters the objective or the guess.
"""

from __future__ import annotations

from ctypes import byref, c_double, c_int
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
from acados_template import ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver

from contact_implicit_casadi import ContactImplicitSymbolic, build_contact_implicit
from contact_implicit_numpy import ContactLayout, contact_quantities
from flamingo_gait import FlamingoGait
from flamingo_model import (
    FlamingoParameters, actuation_selector, build_model, contact_positions,
    mirror_configuration,
)

PITCH_LIMIT = 0.5 * np.pi
HIP_LIMIT = 1.25
KNEE_LIMIT = 0.5 * np.pi
ANKLE_LIMIT = 1.1
INTERIOR = 1e-3


@dataclass
class FlamingoOcpOptions:
    horizon: int = 35  # half stride of the 70-interval gait
    mu: float = 0.74
    cot_epsilon: float = 0.1  # smoothing of |tau v| [W]
    clearance_time: float = 0.02  # tau_c [s]
    clearance_smoothing: float = 0.01  # delta [m/s]
    torque_limit: float = 100.0
    torque_weight: float = 1e-4
    impulse_weight: float = 1e-3
    relaxation_quadratic_weight: float = 1e6
    relaxation_linear_weight: float = 2.0
    loaded_slip_weight: float = 1000.0
    dummy_weight: float = 1e-3
    pinned_minimum_gap: float = 2e-6  # strict interiority of the pinned periodic knots [m]
    bezier_apex: float = 0.04  # swing apex of the initial guess [m]
    tol_stat: float = 1e-4
    max_iter: int = 1500
    print_level: int = 1
    mu_init: float = 0.1  # filterddp_mu_init
    reg_1: float = 1e-2  # filterddp_reg_1
    name: str = "flamingo"

    def validate(self) -> None:
        positive = ("mu", "cot_epsilon", "clearance_time", "clearance_smoothing",
                    "torque_limit", "pinned_minimum_gap", "bezier_apex", "tol_stat")
        if self.horizon < 3 or any(not getattr(self, name) > 0.0 for name in positive):
            raise ValueError("flamingo OCP options are invalid")


@dataclass
class FlamingoProblem:
    options: FlamingoOcpOptions
    params: FlamingoParameters
    model: pin.Model
    data: pin.Data
    frames: list[int]
    symbolic: ContactImplicitSymbolic
    gait: FlamingoGait
    h: float
    step: float  # advance over the half stride [m]
    lower: np.ndarray
    upper: np.ndarray
    bounded: np.ndarray
    x0: np.ndarray
    stage_parameters: np.ndarray  # (N + 1, np)
    periodic_stages: tuple[int, int]
    ocp: AcadosOcp | None = None

    @property
    def layout(self) -> ContactLayout:
        return self.symbolic.layout


def control_bounds(layout: ContactLayout, torque_limit: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(layout.nu, -np.inf)
    upper = np.full(layout.nu, np.inf)
    lower[layout.torque] = -torque_limit
    upper[layout.torque] = torque_limit
    offset = layout.next_configuration.start
    limits = [None, None, PITCH_LIMIT, HIP_LIMIT, KNEE_LIMIT, ANKLE_LIMIT,
              HIP_LIMIT, KNEE_LIMIT, ANKLE_LIMIT]
    for index, limit in enumerate(limits):
        if limit is not None:
            lower[offset + index] = -limit
            upper[offset + index] = limit
    lower[layout.nonnegative_indices()] = 0.0
    return lower, upper


def running_cost(symbolic: ContactImplicitSymbolic, options: FlamingoOcpOptions,
                 params: FlamingoParameters, step: float) -> ca.SX:
    """Smooth cost of transport plus regularization and relaxation penalties."""
    layout, h, u = symbolic.layout, symbolic.h, symbolic.u
    weight = params.total_mass * params.gravity
    tau = u[layout.torque]
    power = tau * (ca.DM(actuation_selector()) @ symbolic.joint_velocity)
    eps = options.cot_epsilon
    cot = h / (weight * step) * ca.sum1(ca.sqrt(power * power + eps * eps) - eps)
    impulses = u[layout.impulse_offset:layout.normal_relaxation_offset]
    sigma_n = u[layout.normal_relaxation_offset:layout.friction_dual_offset]
    sigma_f = u[layout.friction_relaxation_offset:layout.dummy_offset]
    normal = ca.vertcat(*[u[layout.impulse(i)][1] for i in range(layout.nc)])
    slip = normal / (0.25 * weight * h) * symbolic.tangential_velocity
    zeta = u[layout.dummy]
    return (cot
            + options.torque_weight * h * ca.dot(tau, tau)
            + options.impulse_weight * ca.dot(impulses, impulses)
            + options.relaxation_quadratic_weight * (ca.dot(sigma_n, sigma_n) + ca.dot(sigma_f, sigma_f))
            + options.relaxation_linear_weight * (ca.sum1(sigma_n) + ca.sum1(sigma_f))
            + 0.5 * options.loaded_slip_weight * h * ca.dot(slip, slip)
            + options.dummy_weight * ca.dot(zeta, zeta))


def periodic_targets(model: pin.Model, data: pin.Data, frames: list[int], gait: FlamingoGait,
                     step: float, minimum_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Mirror images of the start configurations advanced by ``step``, lifted so
    every contact gap is at least ``minimum_gap``."""
    targets = []
    for source in (gait.q[0], gait.q[1]):
        target = mirror_configuration(source, step)
        gaps = contact_positions(model, data, target, frames)[:, 1]
        target[1] += max(0.0, minimum_gap - np.min(gaps))
        targets.append(target)
    return targets[0], targets[1]


def build_problem(gait: FlamingoGait, options: FlamingoOcpOptions | None = None,
                  params: FlamingoParameters | None = None) -> FlamingoProblem:
    options = options or FlamingoOcpOptions()
    options.validate()
    params = params or FlamingoParameters()
    model, frames = build_model(params)
    data = model.createData()
    h, N = gait.h, options.horizon
    step = gait.stride * N / gait.horizon
    symbolic = build_contact_implicit(model, frames, actuation_selector(), h, options.mu,
                                      options.clearance_time, options.clearance_smoothing)
    lower, upper = control_bounds(symbolic.layout, options.torque_limit)
    bounded = np.flatnonzero(np.isfinite(lower) | np.isfinite(upper))

    # x_0 = (q_0, q_1) from the gait; x_N = (q_{N-1}, q_N) = mirror(x_0) + step.
    x0 = np.concatenate([gait.q[0], gait.q[1]])
    target_prev, target_last = periodic_targets(model, data, frames, gait, step,
                                                options.pinned_minimum_gap)
    P = symbolic.parameters
    stage_parameters = np.zeros((N + 1, P.size))
    periodic_stages = (N - 2, N - 1)  # their controls q_{k+1} are q_{N-1}, q_N
    for stage, target in zip(periodic_stages, (target_prev, target_last)):
        stage_parameters[stage, P.periodicity_target] = target
        stage_parameters[stage, P.periodicity_flag] = 1.0

    problem = FlamingoProblem(
        options=options, params=params, model=model, data=data, frames=frames,
        symbolic=symbolic, gait=gait, h=h, step=step, lower=lower, upper=upper,
        bounded=bounded, x0=x0, stage_parameters=stage_parameters,
        periodic_stages=periodic_stages)
    problem.ocp = make_ocp(problem)
    return problem


def make_ocp(problem: FlamingoProblem) -> AcadosOcp:
    options, symbolic = problem.options, problem.symbolic
    N = options.horizon
    model = AcadosModel()
    model.name = options.name
    model.x = symbolic.x
    model.u = symbolic.u
    model.p = symbolic.p
    model.disc_dyn_expr = symbolic.disc_dyn_expr
    model.con_h_expr = symbolic.con_h_expr
    model.con_h_expr_0 = symbolic.con_h_expr
    model.cost_expr_ext_cost = running_cost(symbolic, options, problem.params, problem.step)
    model.cost_expr_ext_cost_e = ca.SX(0.0)  # x_N is fixed by the periodicity rows

    ocp = AcadosOcp()
    ocp.model = model
    ocp.parameter_values = np.zeros(symbolic.parameters.size)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.constraints.x0 = problem.x0
    ocp.constraints.idxbu = problem.bounded
    ocp.constraints.lbu = np.where(np.isfinite(problem.lower[problem.bounded]),
                                   problem.lower[problem.bounded], -ACADOS_INFTY)
    ocp.constraints.ubu = np.where(np.isfinite(problem.upper[problem.bounded]),
                                   problem.upper[problem.bounded], ACADOS_INFTY)
    n_eq, n_ineq = symbolic.h_eq.shape[0], symbolic.h_ineq.shape[0]
    lh = np.zeros(symbolic.n_h)
    uh = np.zeros(symbolic.n_h)
    uh[n_eq:n_eq + n_ineq] = ACADOS_INFTY  # inequalities h >= 0, all other rows h = 0
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_0 = lh
    ocp.constraints.uh_0 = uh

    so = ocp.solver_options
    so.N_horizon = N
    so.tf = N * problem.h
    so.integrator_type = "DISCRETE"
    so.nlp_solver_type = "FILTERDDP"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = N
    so.hessian_approx = "EXACT"
    so.regularize_method = "NO_REGULARIZE"
    so.globalization = "FIXED_STEP"
    so.nlp_solver_max_iter = options.max_iter
    so.nlp_solver_tol_stat = options.tol_stat
    so.nlp_solver_tol_eq = options.tol_stat
    so.nlp_solver_tol_ineq = options.tol_stat
    so.nlp_solver_tol_comp = options.tol_stat
    so.print_level = options.print_level
    so.cost_scaling = np.ones(N + 1)  # the cost already carries h
    return ocp


def set_filterddp_option(solver: AcadosOcpSolver, name: str, value: float | int) -> None:
    """FilterDDP options are not in the Python template; set them through ctypes."""
    lib = solver._AcadosOcpSolver__acados_lib
    cval = c_int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, bool) \
        else c_double(float(value))
    lib.ocp_nlp_solver_opts_set(solver.nlp_config, solver.nlp_opts,
                                name.encode("utf-8"), byref(cval))


def create_solver(problem: FlamingoProblem, output_dir: Path,
                  verbose: bool = False) -> AcadosOcpSolver:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ocp = problem.ocp
    ocp.code_export_directory = str(output_dir / "c_generated_code")
    # acados writes the json next to the generated code; a path is not accepted.
    solver = AcadosOcpSolver(ocp, json_file=f"{ocp.model.name}_ocp.json", verbose=verbose)
    set_filterddp_option(solver, "filterddp_mu_init", problem.options.mu_init)
    set_filterddp_option(solver, "filterddp_reg_1", problem.options.reg_1)
    for k in range(problem.options.horizon + 1):
        solver.set(k, "p", problem.stage_parameters[k])
    return solver


def _ankle_positions(problem: FlamingoProblem, q: np.ndarray) -> list[np.ndarray]:
    pin.framesForwardKinematics(problem.model, problem.data, q)
    return [problem.data.oMi[problem.model.getJointId(f"ankle{leg}")].translation[[0, 2]].copy()
            for leg in (1, 2)]


def leg_inverse_kinematics(hip: np.ndarray, ankle: np.ndarray, pitch: float,
                           foot_angle: float, knee_sign: float,
                           params: FlamingoParameters) -> np.ndarray:
    """Relative (hip, knee, ankle) angles placing the ankle at ``ankle`` with
    absolute foot angle ``foot_angle``. Link directions are ``(sin a, -cos a)``
    for absolute angle ``a``."""
    l1, l2 = params.l_thigh, params.l_calf
    delta = np.asarray(ankle) - np.asarray(hip)
    distance = np.linalg.norm(delta)
    if not abs(l1 - l2) < distance < l1 + l2:
        raise ValueError("Bezier initial guess leaves the leg workspace")
    cos_knee = (distance ** 2 - l1 ** 2 - l2 ** 2) / (2.0 * l1 * l2)
    knee = knee_sign * np.arccos(np.clip(cos_knee, -1.0, 1.0))
    direction = np.arctan2(delta[0], -delta[1])
    thigh = direction - np.arctan2(l2 * np.sin(knee), l1 + l2 * np.cos(knee))
    return np.array([thigh - pitch, knee, foot_angle - thigh - knee])


def initial_knots(problem: FlamingoProblem) -> np.ndarray:
    """(N + 1, nq) Bezier stepping guess from q_1 to the periodic end configuration.

    Hip position and pitch are linear in time. A foot whose end position
    differs from its start swings flat-footed during the middle 70% of the
    horizon: cubic smoothstep in x, quartic Bernstein bump of apex
    ``bezier_apex`` in z. Joint angles follow from planar 2-link inverse
    kinematics on the start pose's knee branch.
    """
    N, params, apex = problem.options.horizon, problem.params, problem.options.bezier_apex
    start = problem.gait.q[1]
    end = mirror_configuration(start, problem.step)
    ankles_start = _ankle_positions(problem, start)
    ankles_end = _ankle_positions(problem, end)
    knee_signs = [np.sign(start[4]) or 1.0, np.sign(start[7]) or 1.0]
    foot_angles = [start[2] + start[3:6].sum(), start[2] + start[6:9].sum()]
    moving = [np.linalg.norm(ankles_end[leg] - ankles_start[leg]) > 1e-2 for leg in (0, 1)]
    knots = np.zeros((N + 1, start.size))
    for j in range(N + 1):
        t = j / N
        q = (1.0 - t) * start + t * end
        for leg in (0, 1):
            ankle = ankles_start[leg].copy()
            if moving[leg]:
                phase = np.clip((t - 0.15) / 0.7, 0.0, 1.0)
                smooth = phase * phase * (3.0 - 2.0 * phase)  # cubic Bezier (0, 0, 1, 1)
                ankle = ankles_start[leg] + smooth * (ankles_end[leg] - ankles_start[leg])
                ankle[1] += 16.0 * apex * phase ** 2 * (1.0 - phase) ** 2  # quartic Bernstein bump
            offset = 3 + 3 * leg
            q[offset:offset + 3] = leg_inverse_kinematics(
                q[:2], ankle, q[2], foot_angles[leg], knee_signs[leg], params)
        knots[j] = q
    return knots


def initial_controls(problem: FlamingoProblem, interior: float = INTERIOR) -> np.ndarray:
    """(N, nu) initial controls strictly inside every bound.

    Next configurations follow ``initial_knots``; torques and tangential
    impulses are zero and normal impulses ``interior`` (Posa's zero initial
    forces, kept strictly positive). Relaxations equal the resulting products,
    friction duals follow the signed tangential speed, dummy controls are zero.
    """
    layout, options = problem.layout, problem.options
    knots = initial_knots(problem)
    controls = np.full((options.horizon, layout.nu), interior)
    controls[:, layout.torque] = 0.0
    controls[:, layout.dummy] = 0.0
    for k in range(options.horizon):
        u = controls[k]
        u[layout.next_configuration] = knots[k + 1]
        gaps, tangential = contact_quantities(problem.model, problem.data, problem.frames,
                                              problem.h, knots[k], knots[k + 1])
        for i in range(layout.nc):
            p_n = interior
            u[layout.impulse(i)] = [0.0, p_n]
            u[layout.normal_relaxation_offset + i] = max(interior, p_n * max(gaps[i], 0.0))
            lam = np.array([max(-tangential[i], 0.0), max(tangential[i], 0.0)]) + interior
            u[layout.friction_dual(i)] = lam
            u[layout.friction_relaxation(i)] = np.maximum(interior, lam * options.mu * p_n)
    if np.any(controls <= problem.lower) or np.any(controls >= problem.upper):
        stage, index = np.argwhere((controls <= problem.lower) | (controls >= problem.upper))[0]
        raise RuntimeError(f"initial guess is not strictly interior at stage {stage}, "
                           f"control {index}")
    return controls


def rollout_states(problem: FlamingoProblem, controls: np.ndarray) -> np.ndarray:
    layout = problem.layout
    states = np.zeros((controls.shape[0] + 1, layout.nx))
    states[0] = problem.x0
    for k, u in enumerate(controls):
        states[k + 1] = np.concatenate([states[k][layout.nq:], u[layout.next_configuration]])
    return states


def initialize_solver(problem: FlamingoProblem, solver: AcadosOcpSolver,
                      controls: np.ndarray) -> None:
    states = rollout_states(problem, controls)
    for k, u in enumerate(controls):
        solver.set(k, "x", states[k])
        solver.set(k, "u", u)
    solver.set(controls.shape[0], "x", states[-1])

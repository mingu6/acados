"""CasADi stage equations of the Spring Flamingo OCP (README.md).

Given a Pinocchio model with a Euclidean configuration (``nq == nv``), an
actuation selector ``S`` (nu_torque x nv), and planar contact frames, this
builds the symbolic state, control, and parameter vectors, the explicit shift
dynamics, and the path-constraint rows with ``pinocchio.casadi``.
``contact_implicit_numpy`` is the double-precision oracle for every row.

Row order (``con_h_expr``):

- equalities: mechanics (nq); ``p_n g - sigma_n`` (nc);
  ``v_t + lambda+ - lambda-`` (nc); ``lambda+- s+- - sigma_f+-`` (2 nc);
- inequalities ``>= 0``: gaps ``g`` (nc); cone margins ``s+-`` (2 nc);
  clearance ``g - tau_c (sqrt(v_t^2 + delta^2) - delta)`` (nc);
- periodicity equalities (nq): ``f (q_next - q_target) + (1 - f) zeta``, with
  the per-stage flag ``f`` and target in the parameter vector and ``zeta`` the
  trailing dummy controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from contact_implicit_numpy import ContactLayout


@dataclass(frozen=True)
class ParameterLayout:
    """Per-stage parameters: periodic target configuration and its flag."""

    nq: int

    @property
    def periodicity_target(self) -> slice:
        return slice(0, self.nq)

    @property
    def periodicity_flag(self) -> int:
        return self.nq

    @property
    def size(self) -> int:
        return self.nq + 1


@dataclass
class ContactImplicitSymbolic:
    layout: ContactLayout
    parameters: ParameterLayout
    h: float
    mu: float
    clearance_time: float
    clearance_smoothing: float
    x: ca.SX
    u: ca.SX
    p: ca.SX
    disc_dyn_expr: ca.SX
    h_eq: ca.SX
    h_ineq: ca.SX
    h_periodic: ca.SX
    gaps: ca.SX
    tangential_velocity: ca.SX
    joint_velocity: ca.SX  # (q_next - q) / h
    functions: dict[str, ca.Function] = field(default_factory=dict)

    @property
    def con_h_expr(self) -> ca.SX:
        return ca.vertcat(self.h_eq, self.h_ineq, self.h_periodic)

    @property
    def n_h(self) -> int:
        return self.con_h_expr.shape[0]


def build_contact_implicit(model: pin.Model, contact_frames: list[int],
                           selector: np.ndarray, h: float, mu: float,
                           clearance_time: float,
                           clearance_smoothing: float = 0.01) -> ContactImplicitSymbolic:
    if model.nq != model.nv:
        raise ValueError("contact-implicit builder requires a Euclidean configuration")
    for name, value in (("time step", h), ("friction coefficient", mu),
                        ("clearance time", clearance_time),
                        ("clearance smoothing", clearance_smoothing)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive")
    nq = model.nq
    nc = len(contact_frames)
    selector = np.asarray(selector, dtype=float)
    if selector.shape[1] != nq:
        raise ValueError("actuation selector has the wrong number of columns")
    layout = ContactLayout(nq=nq, nu_torque=selector.shape[0], nc=nc, n_dummy=nq)
    parameters = ParameterLayout(nq=nq)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # Contact positions and Jacobians as functions of a fresh symbol so they
    # can be applied to expressions (midpoint, next configuration).
    q_sym = ca.SX.sym("q_contact", nq)
    cpin.framesForwardKinematics(cmodel, cdata, q_sym)
    positions = ca.vertcat(*[cdata.oMf[frame].translation[[0, 2]]
                             for frame in contact_frames])  # (2 nc,) as (x, z) pairs
    position_function = ca.Function("contact_positions", [q_sym], [positions])
    jacobian_function = ca.Function("contact_jacobians", [q_sym],
                                    [ca.jacobian(positions, q_sym)])  # (2 nc, nq)

    x = ca.SX.sym("x", 2 * nq)
    u = ca.SX.sym("u", layout.nu)
    p = ca.SX.sym("p", parameters.size)
    q_prev = x[:nq]
    q = x[nq:]
    q_next = u[layout.next_configuration]
    tau = u[layout.torque]
    q_bar = 0.5 * (q + q_next)
    v = (q_next - q) / h
    a = (q_prev - 2.0 * q + q_next) / (h * h)

    # Midpoint inverse dynamics: h [RNEA(qbar, v, a) - S^T tau] - sum_i J_i(qbar)^T p_i.
    jacobian = jacobian_function(q_bar)
    contact_force = ca.SX.zeros(nq)
    for i in range(nc):
        contact_force += ca.transpose(jacobian[2 * i:2 * i + 2, :]) @ u[layout.impulse(i)]
    mechanics = h * (cpin.rnea(cmodel, cdata, q_bar, v, a) - ca.DM(selector.T) @ tau) \
        - contact_force

    current_positions = position_function(q)
    next_positions = position_function(q_next)
    gaps = ca.vertcat(*[next_positions[2 * i + 1] for i in range(nc)])
    tangential = ca.vertcat(*[(next_positions[2 * i] - current_positions[2 * i]) / h
                              for i in range(nc)])

    normal_complementarity, stationarity, friction_complementarity, cone = [], [], [], []
    for i in range(nc):
        p_t = u[layout.impulse(i)][0]
        p_n = u[layout.impulse(i)][1]
        lam = u[layout.friction_dual(i)]
        margin = ca.vertcat(mu * p_n - p_t, mu * p_n + p_t)
        normal_complementarity.append(p_n * gaps[i] - u[layout.normal_relaxation_offset + i])
        stationarity.append(tangential[i] + lam[0] - lam[1])
        friction_complementarity.append(lam * margin - u[layout.friction_relaxation(i)])
        cone.append(margin)

    delta = clearance_smoothing
    clearance = gaps - clearance_time * (ca.sqrt(tangential * tangential + delta * delta) - delta)
    h_eq = ca.vertcat(mechanics, *normal_complementarity, *stationarity,
                      *friction_complementarity)
    h_ineq = ca.vertcat(gaps, *cone, clearance)
    flag = p[parameters.periodicity_flag]
    h_periodic = (flag * (q_next - p[parameters.periodicity_target])
                  + (1.0 - flag) * u[layout.dummy])
    disc_dyn_expr = ca.vertcat(q, q_next)

    lam_eq = ca.SX.sym("lam_eq", h_eq.shape[0])
    functions = {
        "h_eq": ca.Function("h_eq", [x, u], [h_eq]),
        "h_ineq": ca.Function("h_ineq", [x, u], [h_ineq]),
        "h_periodic": ca.Function("h_periodic", [x, u, p], [h_periodic]),
        "jac_eq_u": ca.Function("jac_eq_u", [x, u], [ca.jacobian(h_eq, u)]),
        "jac_eq_x": ca.Function("jac_eq_x", [x, u], [ca.jacobian(h_eq, x)]),
        "jac_ineq_u": ca.Function("jac_ineq_u", [x, u], [ca.jacobian(h_ineq, u)]),
        "jac_ineq_x": ca.Function("jac_ineq_x", [x, u], [ca.jacobian(h_ineq, x)]),
        "jac_periodic_u": ca.Function("jac_periodic_u", [x, u, p],
                                      [ca.jacobian(h_periodic, u)]),
        "hess_eq": ca.Function("hess_eq", [x, u, lam_eq],
                               [ca.hessian(ca.dot(lam_eq, h_eq), ca.vertcat(u, x))[0]]),
        "contact_positions": position_function,
        "contact_jacobians": jacobian_function,
    }
    return ContactImplicitSymbolic(
        layout=layout, parameters=parameters, h=h, mu=mu, clearance_time=clearance_time,
        clearance_smoothing=clearance_smoothing, x=x, u=u, p=p, disc_dyn_expr=disc_dyn_expr,
        h_eq=h_eq, h_ineq=h_ineq, h_periodic=h_periodic, gaps=gaps,
        tangential_velocity=tangential, joint_velocity=v, functions=functions)

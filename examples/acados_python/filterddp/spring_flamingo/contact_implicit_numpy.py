"""Numpy reference for the Spring Flamingo stage equations.

Double-precision oracle for the CasADi rows in ``contact_implicit_casadi``;
all equations are those of ``README.md``.  Contact impulse
``p[i] = (p_t, p_n)`` acts at contact frame ``i`` through the planar position
Jacobian ``J_i = d(x, z)/dq`` (2 x nv).

``cimpc_mechanics_residual`` is the ContactImplicitMPC.jl integrator. It is
not part of the OCP; it exists only to prove that the Pinocchio model
reproduces the stored Julia gait (``tests/test_gait_reconciliation.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from flamingo_model import contact_jacobians, contact_positions, mass_matrix


@dataclass(frozen=True)
class ContactLayout:
    """Control layout: torques, next configuration, impulses, complementarity
    relaxations, friction duals, friction relaxations, periodicity dummies."""

    nq: int
    nu_torque: int
    nc: int
    n_dummy: int = 0

    @property
    def torque(self) -> slice:
        return slice(0, self.nu_torque)

    @property
    def next_configuration(self) -> slice:
        return slice(self.nu_torque, self.nu_torque + self.nq)

    @property
    def impulse_offset(self) -> int:
        return self.nu_torque + self.nq

    def impulse(self, contact: int) -> slice:
        start = self.impulse_offset + 2 * contact
        return slice(start, start + 2)

    @property
    def normal_relaxation_offset(self) -> int:
        return self.impulse_offset + 2 * self.nc

    @property
    def friction_dual_offset(self) -> int:
        return self.normal_relaxation_offset + self.nc

    def friction_dual(self, contact: int) -> slice:
        start = self.friction_dual_offset + 2 * contact
        return slice(start, start + 2)

    @property
    def friction_relaxation_offset(self) -> int:
        return self.friction_dual_offset + 2 * self.nc

    def friction_relaxation(self, contact: int) -> slice:
        start = self.friction_relaxation_offset + 2 * contact
        return slice(start, start + 2)

    @property
    def dummy_offset(self) -> int:
        return self.friction_relaxation_offset + 2 * self.nc

    @property
    def dummy(self) -> slice:
        return slice(self.dummy_offset, self.dummy_offset + self.n_dummy)

    @property
    def nu(self) -> int:
        return self.dummy_offset + self.n_dummy

    @property
    def nx(self) -> int:
        return 2 * self.nq

    @property
    def n_equality(self) -> int:
        return self.nq + 4 * self.nc

    def normal_impulse_indices(self) -> np.ndarray:
        return np.array([self.impulse_offset + 2 * i + 1 for i in range(self.nc)])

    def tangential_impulse_indices(self) -> np.ndarray:
        return np.array([self.impulse_offset + 2 * i for i in range(self.nc)])

    def nonnegative_indices(self) -> np.ndarray:
        """Controls with a zero lower bound: p_n, sigma_n, lambda+-, sigma_f+-."""
        return np.concatenate([
            self.normal_impulse_indices(),
            np.arange(self.normal_relaxation_offset, self.dummy_offset),
        ])


def mechanics_residual(model: pin.Model, data: pin.Data, frames: list[int],
                       selector: np.ndarray, h: float,
                       q_prev: np.ndarray, q: np.ndarray, q_next: np.ndarray,
                       tau: np.ndarray, impulses: np.ndarray) -> np.ndarray:
    """``h [RNEA(qbar, v, a) - S^T tau] - sum_i J_i(qbar)^T p_i``; impulses (nc, 2)."""
    q_prev, q, q_next = (np.asarray(v, float) for v in (q_prev, q, q_next))
    q_bar = 0.5 * (q + q_next)
    v = (q_next - q) / h
    a = (q_prev - 2.0 * q + q_next) / (h * h)
    contact_force = sum(jac.T @ np.asarray(impulses[i], float)
                        for i, jac in enumerate(contact_jacobians(model, data, q_bar, frames)))
    return h * (np.array(pin.rnea(model, data, q_bar, v, a)) - selector.T @ tau) - contact_force


def cimpc_mechanics_residual(model: pin.Model, data: pin.Data, frames: list[int],
                             selector: np.ndarray, h: float,
                             q_prev: np.ndarray, q: np.ndarray, q_next: np.ndarray,
                             tau: np.ndarray, impulses: np.ndarray) -> np.ndarray:
    """ContactImplicitMPC.jl ``model.jl:12-41``, negated to share the sign above:
    ``-[M(qbar_prev) v_prev - M(qbar) v - h/2 (nle_prev + nle) + S^T (h tau)
    + sum_i J_i(q_next)^T p_i]``."""
    q_prev, q, q_next = (np.asarray(v, float) for v in (q_prev, q, q_next))
    q_bar_prev, q_bar = 0.5 * (q_prev + q), 0.5 * (q + q_next)
    v_prev, v = (q - q_prev) / h, (q_next - q) / h
    contact_force = sum(jac.T @ np.asarray(impulses[i], float)
                        for i, jac in enumerate(contact_jacobians(model, data, q_next, frames)))
    julia = (mass_matrix(model, data, q_bar_prev) @ v_prev - mass_matrix(model, data, q_bar) @ v
             - 0.5 * h * (np.array(pin.nonLinearEffects(model, data, q_bar_prev, v_prev))
                          + np.array(pin.nonLinearEffects(model, data, q_bar, v)))
             + selector.T @ (h * tau) + contact_force)
    return -julia


def contact_quantities(model: pin.Model, data: pin.Data, frames: list[int],
                       h: float, q: np.ndarray, q_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaps ``g_i = z_i(q_next)`` and tangential speeds from position differences."""
    current = contact_positions(model, data, q, frames)
    following = contact_positions(model, data, q_next, frames)
    return following[:, 1], (following[:, 0] - current[:, 0]) / h


def clearance_requirement(tangential: np.ndarray, clearance_time: float,
                          smoothing: float) -> np.ndarray:
    """``tau_c (sqrt(v^2 + delta^2) - delta)``: zero at rest, ``~ tau_c |v|`` when moving."""
    return clearance_time * (np.sqrt(tangential ** 2 + smoothing ** 2) - smoothing)


def stage_constraints(model: pin.Model, data: pin.Data, frames: list[int],
                      selector: np.ndarray, layout: ContactLayout, h: float,
                      mu: float, clearance_time: float, clearance_smoothing: float,
                      x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equalities (nq + 4 nc) and inequalities (4 nc) in the documented order."""
    nq = layout.nq
    q_prev, q = x[:nq], x[nq:]
    q_next = u[layout.next_configuration]
    impulses = np.array([u[layout.impulse(i)] for i in range(layout.nc)])
    mechanics = mechanics_residual(model, data, frames, selector, h, q_prev, q, q_next,
                                   u[layout.torque], impulses)
    gaps, tangential = contact_quantities(model, data, frames, h, q, q_next)
    normal_complementarity = np.zeros(layout.nc)
    stationarity = np.zeros(layout.nc)
    friction_complementarity = np.zeros(2 * layout.nc)
    cone = np.zeros(2 * layout.nc)
    for i in range(layout.nc):
        p_t, p_n = impulses[i]
        lam = u[layout.friction_dual(i)]
        margin = np.array([mu * p_n - p_t, mu * p_n + p_t])
        normal_complementarity[i] = p_n * gaps[i] - u[layout.normal_relaxation_offset + i]
        stationarity[i] = tangential[i] + lam[0] - lam[1]
        friction_complementarity[2 * i:2 * i + 2] = lam * margin - u[layout.friction_relaxation(i)]
        cone[2 * i:2 * i + 2] = margin
    clearance = gaps - clearance_requirement(tangential, clearance_time, clearance_smoothing)
    equalities = np.concatenate([mechanics, normal_complementarity, stationarity,
                                 friction_complementarity])
    return equalities, np.concatenate([gaps, cone, clearance])

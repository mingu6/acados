"""Numpy transcription of ``ContactImplicitMPC.jl/src/dynamics/flamingo/model.jl``.

Absolute-angle coordinates, used only as a test oracle for the Pinocchio
model.  Function names and body/mode keywords mirror the Julia source so the
transcription can be checked line by line (``model.jl:62-253`` kinematics,
``257-324`` Lagrangian, ``353-427`` mass matrix, gap, actuation, contact
Jacobian).
"""

from __future__ import annotations

import numpy as np

from flamingo_model import FlamingoParameters

NQ = 9


def kinematics_1(p: FlamingoParameters, q, body="torso", mode="ee"):
    x, z = q[0], q[1]
    if body == "torso":
        r = p.l_torso if mode == "ee" else p.d_torso
        th = q[2]
        return np.array([x - r * np.sin(th), z + r * np.cos(th)])
    if body == "thigh_1":
        r = p.l_thigh if mode == "ee" else p.d_thigh
        th = q[3]
    elif body == "thigh_2":
        r = p.l_thigh if mode == "ee" else p.d_thigh
        th = q[5]
    else:
        raise ValueError(body)
    return np.array([x + r * np.sin(th), z - r * np.cos(th)])


def jacobian_1(p: FlamingoParameters, q, body="torso", mode="ee"):
    jac = np.zeros((2, NQ))
    jac[0, 0] = 1.0
    jac[1, 1] = 1.0
    if body == "torso":
        r = p.l_torso if mode == "ee" else p.d_torso
        th = q[2]
        jac[0, 2] = -r * np.cos(th)
        jac[1, 2] = -r * np.sin(th)
    elif body == "thigh_1":
        r = p.l_thigh if mode == "ee" else p.d_thigh
        th = q[3]
        jac[0, 3] = r * np.cos(th)
        jac[1, 3] = r * np.sin(th)
    elif body == "thigh_2":
        r = p.l_thigh if mode == "ee" else p.d_thigh
        th = q[5]
        jac[0, 5] = r * np.cos(th)
        jac[1, 5] = r * np.sin(th)
    else:
        raise ValueError(body)
    return jac


def kinematics_2(p: FlamingoParameters, q, body="calf_1", mode="ee"):
    if body == "calf_1":
        base = kinematics_1(p, q, "thigh_1", "ee")
        th = q[4]
    elif body == "calf_2":
        base = kinematics_1(p, q, "thigh_2", "ee")
        th = q[6]
    else:
        raise ValueError(body)
    r = p.l_calf if mode == "ee" else p.d_calf
    return base + np.array([r * np.sin(th), -r * np.cos(th)])


def jacobian_2(p: FlamingoParameters, q, body="calf_1", mode="ee"):
    if body == "calf_1":
        jac = jacobian_1(p, q, "thigh_1", "ee")
        th, col = q[4], 4
    elif body == "calf_2":
        jac = jacobian_1(p, q, "thigh_2", "ee")
        th, col = q[6], 6
    else:
        raise ValueError(body)
    r = p.l_calf if mode == "ee" else p.d_calf
    jac[0, col] += r * np.cos(th)
    jac[1, col] += r * np.sin(th)
    return jac


def _foot_radius(p: FlamingoParameters, mode: str) -> float:
    if mode == "toe":
        return p.l_foot
    if mode == "heel":
        return -p.d_foot
    if mode == "com":
        return 0.5 * (p.l_foot - p.d_foot)
    raise ValueError(mode)


def kinematics_3(p: FlamingoParameters, q, body="foot_1", mode="toe"):
    if body == "foot_1":
        base = kinematics_2(p, q, "calf_1", "ee")
        th = q[7]
    elif body == "foot_2":
        base = kinematics_2(p, q, "calf_2", "ee")
        th = q[8]
    else:
        raise ValueError(body)
    r = _foot_radius(p, mode)
    return base + np.array([r * np.sin(th), -r * np.cos(th)])


def jacobian_3(p: FlamingoParameters, q, body="foot_1", mode="toe"):
    if body == "foot_1":
        jac = jacobian_2(p, q, "calf_1", "ee")
        th, col = q[7], 7
    elif body == "foot_2":
        jac = jacobian_2(p, q, "calf_2", "ee")
        th, col = q[8], 8
    else:
        raise ValueError(body)
    r = _foot_radius(p, mode)
    jac[0, col] += r * np.cos(th)
    jac[1, col] += r * np.sin(th)
    return jac


_BODIES = (
    # (kinematics fn, jacobian fn, body key, mass attr, inertia attr, rotational index)
    (kinematics_1, jacobian_1, "torso", "m_torso", "J_torso", 2),
    (kinematics_1, jacobian_1, "thigh_1", "m_thigh", "J_thigh", 3),
    (kinematics_2, jacobian_2, "calf_1", "m_calf", "J_calf", 4),
    (kinematics_3, jacobian_3, "foot_1", "m_foot", "J_foot", 7),
    (kinematics_1, jacobian_1, "thigh_2", "m_thigh", "J_thigh", 5),
    (kinematics_2, jacobian_2, "calf_2", "m_calf", "J_calf", 6),
    (kinematics_3, jacobian_3, "foot_2", "m_foot", "J_foot", 8),
)


def lagrangian(p: FlamingoParameters, q, qd) -> float:
    L = 0.0
    for kin, jac, body, mass_name, inertia_name, index in _BODIES:
        mass = getattr(p, mass_name)
        pos = kin(p, q, body, "com")
        vel = jac(p, q, body, "com") @ qd
        L += 0.5 * mass * vel @ vel
        L += 0.5 * getattr(p, inertia_name) * qd[index] ** 2
        L -= mass * p.gravity * pos[1]
    return L


def M_func(p: FlamingoParameters, q) -> np.ndarray:
    M = np.diag([0.0, 0.0, p.J_torso, p.J_thigh, p.J_calf, p.J_thigh,
                 p.J_calf, p.J_foot, p.J_foot])
    for _, jac, body, mass_name, _, _ in _BODIES:
        J = jac(p, q, body, "com")
        M = M + getattr(p, mass_name) * J.T @ J
    return M


def center_of_mass(p: FlamingoParameters, q) -> np.ndarray:
    total = np.zeros(2)
    for kin, _, body, mass_name, _, _ in _BODIES:
        total += getattr(p, mass_name) * kin(p, q, body, "com")
    return total / p.total_mass


def kinematics(p: FlamingoParameters, q) -> np.ndarray:
    """Contact points ``[toe1; heel1; toe2; heel2]`` as ``(x, z)`` pairs."""
    return np.concatenate([
        kinematics_3(p, q, "foot_1", "toe"),
        kinematics_3(p, q, "foot_1", "heel"),
        kinematics_3(p, q, "foot_2", "toe"),
        kinematics_3(p, q, "foot_2", "heel"),
    ])


def phi_func(p: FlamingoParameters, q) -> np.ndarray:
    """Flat-ground gaps (``model.jl:391-401`` with ``surf == 0``)."""
    return kinematics(p, q)[1::2]


def J_func(p: FlamingoParameters, q) -> np.ndarray:
    return np.vstack([
        jacobian_3(p, q, "foot_1", "toe"),
        jacobian_3(p, q, "foot_1", "heel"),
        jacobian_3(p, q, "foot_2", "toe"),
        jacobian_3(p, q, "foot_2", "heel"),
    ])


def B_func() -> np.ndarray:
    """``model.jl:403-410``: rows are [hip1, knee1, hip2, knee2, ankle1, ankle2]."""
    return np.array([
        [0, 0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 1],
    ], dtype=float)

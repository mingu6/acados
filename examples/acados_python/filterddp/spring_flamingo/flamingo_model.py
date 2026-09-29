"""Planar Spring Flamingo model in Pinocchio.

Parameters and conventions follow ``ContactImplicitMPC.jl``'s ``flamingo``
model (``src/dynamics/flamingo/model.jl``), see
``README.md``.  The model is built programmatically; the
URDF shipped with the Julia package does not match these parameters.

Coordinates
-----------
Pinocchio relative coordinates, ``nq == nv == 9``::

    q = (x, z, pitch, hip1, knee1, ankle1, hip2, knee2, ankle2)

Every revolute axis is ``-y`` so that a positive angle rotates ``+x`` toward
``+z`` (the Julia convention).  The Julia model uses absolute link angles::

    q_abs = (x, z, th_torso, th_thigh1, th_calf1, th_thigh2, th_calf2,
             th_foot1, th_foot2)

and ``q = T q_abs + c`` with ``c`` carrying ``-pi/2`` on the ankles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

NQ = 9
NV = 9
NU_TORQUE = 6
NC = 4

JOINT_NAMES = (
    "root_x", "root_z", "pitch",
    "hip1", "knee1", "ankle1",
    "hip2", "knee2", "ankle2",
)
ACTUATED_JOINTS = ("hip1", "knee1", "ankle1", "hip2", "knee2", "ankle2")
CONTACT_NAMES = ("toe1", "heel1", "toe2", "heel2")

# Absolute (Julia) coordinate indices, 0-based.
ABS_X, ABS_Z, ABS_TORSO = 0, 1, 2
ABS_THIGH1, ABS_CALF1, ABS_THIGH2, ABS_CALF2 = 3, 4, 5, 6
ABS_FOOT1, ABS_FOOT2 = 7, 8

# Julia torque order [hip1, knee1, hip2, knee2, ankle1, ankle2] -> Pinocchio
# actuated order [hip1, knee1, ankle1, hip2, knee2, ankle2].
JULIA_TO_PINOCCHIO_TORQUE = np.array([0, 1, 4, 2, 3, 5])

# Mirror permutation swapping leg 1 and leg 2 (relative coordinates).
MIRROR_PERMUTATION = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5])


@dataclass(frozen=True)
class FlamingoParameters:
    """``model.jl:452-495``. Lengths in m, masses in kg, inertias in kg m^2."""

    gravity: float = 9.81
    m_torso: float = 12.0
    m_thigh: float = 0.4598
    m_calf: float = 0.306
    m_foot: float = 0.3466
    l_torso: float = 0.385  # visual only
    l_thigh: float = 0.42
    l_calf: float = 0.45
    l_foot: float = 0.1725  # ankle -> toe
    d_torso: float = 0.20  # hip -> torso COM, upward
    d_thigh: float = 0.21
    d_calf: float = 0.225
    d_foot: float = 0.0525  # ankle -> heel
    J_torso: float = 0.10
    J_thigh: float = 0.01256
    J_calf: float = 0.00952
    J_foot: float = 0.0015

    @property
    def foot_com(self) -> float:
        """Foot COM distance from the ankle toward the toe (``model.jl:187``)."""
        return 0.5 * (self.l_foot - self.d_foot)

    @property
    def total_mass(self) -> float:
        return self.m_torso + 2.0 * (self.m_thigh + self.m_calf + self.m_foot)

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"flamingo parameter {name} must be positive")


def _planar_inertia(mass: float, com: np.ndarray, pitch_inertia: float) -> pin.Inertia:
    """Only the ``yy`` entry enters planar dynamics; ``xx`` and ``zz`` are set
    equal to it so the inertia is a valid rigid-body inertia."""
    return pin.Inertia(mass, np.asarray(com, dtype=float),
                       pitch_inertia * np.eye(3))


def build_model(params: FlamingoParameters | None = None) -> tuple[pin.Model, list[int]]:
    """Return the Pinocchio model and the frame ids of ``CONTACT_NAMES``."""
    params = params or FlamingoParameters()
    params.validate()
    model = pin.Model()
    model.name = "spring_flamingo"
    model.gravity.linear = np.array([0.0, 0.0, -params.gravity])

    axis = np.array([0.0, -1.0, 0.0])

    def revolute() -> pin.JointModelRevoluteUnaligned:
        return pin.JointModelRevoluteUnaligned(axis)

    def translation(vector) -> pin.SE3:
        return pin.SE3(np.eye(3), np.asarray(vector, dtype=float))

    identity = pin.SE3.Identity()
    root_x = model.addJoint(0, pin.JointModelPX(), identity, "root_x")
    model.addJointFrame(root_x)
    root_z = model.addJoint(root_x, pin.JointModelPZ(), identity, "root_z")
    model.addJointFrame(root_z)
    pitch = model.addJoint(root_z, revolute(), identity, "pitch")
    pitch_frame = model.addJointFrame(pitch)
    model.appendBodyToJoint(
        pitch, _planar_inertia(params.m_torso, [0.0, 0.0, params.d_torso],
                               params.J_torso), identity)
    model.addBodyFrame("torso", pitch, identity, pitch_frame)

    contact_frames: list[int] = []
    for leg in (1, 2):
        hip = model.addJoint(pitch, revolute(), identity, f"hip{leg}")
        hip_frame = model.addJointFrame(hip)
        model.appendBodyToJoint(
            hip, _planar_inertia(params.m_thigh, [0.0, 0.0, -params.d_thigh],
                                 params.J_thigh), identity)
        model.addBodyFrame(f"thigh{leg}", hip, identity, hip_frame)

        knee = model.addJoint(hip, revolute(),
                              translation([0.0, 0.0, -params.l_thigh]),
                              f"knee{leg}")
        knee_frame = model.addJointFrame(knee)
        model.appendBodyToJoint(
            knee, _planar_inertia(params.m_calf, [0.0, 0.0, -params.d_calf],
                                  params.J_calf), identity)
        model.addBodyFrame(f"calf{leg}", knee, identity, knee_frame)

        ankle = model.addJoint(knee, revolute(),
                               translation([0.0, 0.0, -params.l_calf]),
                               f"ankle{leg}")
        ankle_frame = model.addJointFrame(ankle)
        model.appendBodyToJoint(
            ankle, _planar_inertia(params.m_foot, [params.foot_com, 0.0, 0.0],
                                   params.J_foot), identity)
        model.addBodyFrame(f"foot{leg}", ankle, identity, ankle_frame)

        toe = model.addFrame(pin.Frame(
            f"toe{leg}", ankle, ankle_frame,
            translation([params.l_foot, 0.0, 0.0]), pin.FrameType.OP_FRAME))
        heel = model.addFrame(pin.Frame(
            f"heel{leg}", ankle, ankle_frame,
            translation([-params.d_foot, 0.0, 0.0]), pin.FrameType.OP_FRAME))
        contact_frames.extend([toe, heel])

    if model.nq != NQ or model.nv != NV:
        raise RuntimeError("unexpected flamingo configuration dimension")
    if [model.names[j] for j in range(1, model.njoints)] != list(JOINT_NAMES):
        raise RuntimeError("unexpected flamingo joint order")
    return model, contact_frames


def actuation_selector() -> np.ndarray:
    """``S`` with shape (6, 9): generalized force ``S^T tau``."""
    selector = np.zeros((NU_TORQUE, NV))
    for row, name in enumerate(ACTUATED_JOINTS):
        selector[row, JOINT_NAMES.index(name)] = 1.0
    return selector


def absolute_to_relative_map() -> tuple[np.ndarray, np.ndarray]:
    """``q = T q_abs + c`` (``visuals.jl:165-177`` without visual offsets)."""
    T = np.zeros((NQ, NQ))
    T[0, ABS_X] = 1.0
    T[1, ABS_Z] = 1.0
    T[2, ABS_TORSO] = 1.0
    T[3, ABS_THIGH1], T[3, ABS_TORSO] = 1.0, -1.0
    T[4, ABS_CALF1], T[4, ABS_THIGH1] = 1.0, -1.0
    T[5, ABS_FOOT1], T[5, ABS_CALF1] = 1.0, -1.0
    T[6, ABS_THIGH2], T[6, ABS_TORSO] = 1.0, -1.0
    T[7, ABS_CALF2], T[7, ABS_THIGH2] = 1.0, -1.0
    T[8, ABS_FOOT2], T[8, ABS_CALF2] = 1.0, -1.0
    c = np.zeros(NQ)
    c[5] = -0.5 * np.pi
    c[8] = -0.5 * np.pi
    return T, c


def absolute_to_relative(q_abs: np.ndarray) -> np.ndarray:
    T, c = absolute_to_relative_map()
    return T @ np.asarray(q_abs, dtype=float) + c


def relative_to_absolute(q: np.ndarray) -> np.ndarray:
    T, c = absolute_to_relative_map()
    return np.linalg.solve(T, np.asarray(q, dtype=float) - c)


def absolute_velocity_to_relative(v_abs: np.ndarray) -> np.ndarray:
    T, _ = absolute_to_relative_map()
    return T @ np.asarray(v_abs, dtype=float)


def mirror_configuration(q: np.ndarray, stride: float = 0.0) -> np.ndarray:
    """Swap the legs and advance ``x`` by ``stride``."""
    mirrored = np.asarray(q, dtype=float)[MIRROR_PERMUTATION].copy()
    mirrored[0] += stride
    return mirrored


def julia_torque_to_pinocchio(u_julia: np.ndarray) -> np.ndarray:
    return np.asarray(u_julia, dtype=float)[JULIA_TO_PINOCCHIO_TORQUE]


def contact_positions(model: pin.Model, data: pin.Data, q: np.ndarray,
                      contact_frames: list[int]) -> np.ndarray:
    """(4, 2) array of contact ``(x, z)`` positions."""
    pin.framesForwardKinematics(model, data, q)
    return np.array([data.oMf[frame].translation[[0, 2]]
                     for frame in contact_frames])


def contact_jacobians(model: pin.Model, data: pin.Data, q: np.ndarray,
                      contact_frames: list[int]) -> np.ndarray:
    """(4, 2, 9) array of planar position Jacobians ``d(x, z)/dq``."""
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.array([
        pin.getFrameJacobian(model, data, frame,
                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[[0, 2], :]
        for frame in contact_frames])


def mass_matrix(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    """Symmetrized CRBA mass matrix."""
    upper = np.array(pin.crba(model, data, q))
    return np.triu(upper) + np.triu(upper, 1).T

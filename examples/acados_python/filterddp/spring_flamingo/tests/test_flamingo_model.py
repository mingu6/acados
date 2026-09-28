"""Pinocchio flamingo model versus the ContactImplicitMPC.jl oracle.

Every comparison is in the Julia absolute-angle coordinates mapped through
``absolute_to_relative``; the oracle is ``flamingo_julia_oracle`` (a
transcription of ``model.jl``).  Seeds are fixed and failing samples print
the offending configuration.
"""

import numpy as np
import pinocchio as pin
import pytest

import flamingo_julia_oracle as oracle
from flamingo_model import (
    CONTACT_NAMES, FlamingoParameters, JOINT_NAMES, absolute_to_relative,
    absolute_to_relative_map, absolute_velocity_to_relative, actuation_selector,
    build_model, contact_jacobians, contact_positions, julia_torque_to_pinocchio,
    mass_matrix, mirror_configuration, relative_to_absolute,
)

SEED = 0
SAMPLES = 64
VALUE_TOL = 1e-12
JACOBIAN_TOL = 1e-10
FD_TOL = 1e-6


@pytest.fixture(scope="module")
def setup():
    params = FlamingoParameters()
    model, frames = build_model(params)
    return params, model, model.createData(), frames


def random_absolute_configurations(rng, count):
    q = rng.uniform(-np.pi, np.pi, size=(count, 9))
    q[:, 0] = rng.uniform(-1.0, 1.0, size=count)
    q[:, 1] = rng.uniform(0.3, 1.2, size=count)
    return q


def worst(a, b):
    err = np.abs(a - b)
    index = np.unravel_index(np.argmax(err), err.shape)
    return err[index], index


def test_dimensions_and_order(setup):
    params, model, _, frames = setup
    assert model.nq == 9 and model.nv == 9
    assert [model.names[j] for j in range(1, model.njoints)] == list(JOINT_NAMES)
    assert [model.frames[f].name for f in frames] == list(CONTACT_NAMES)
    assert abs(params.total_mass - 14.2248) < 1e-12
    assert abs(params.foot_com - 0.06) < 1e-12


def test_contact_kinematics_parity(setup):
    params, model, data, frames = setup
    rng = np.random.default_rng(SEED)
    for q_abs in random_absolute_configurations(rng, SAMPLES):
        expected = oracle.kinematics(params, q_abs).reshape(4, 2)
        actual = contact_positions(model, data, absolute_to_relative(q_abs), frames)
        err, index = worst(actual, expected)
        assert err < VALUE_TOL, f"q_abs={q_abs.tolist()} worst={index} err={err}"


def test_center_of_mass_parity(setup):
    params, model, data, frames = setup
    rng = np.random.default_rng(SEED + 1)
    for q_abs in random_absolute_configurations(rng, SAMPLES):
        expected = oracle.center_of_mass(params, q_abs)
        com = pin.centerOfMass(model, data, absolute_to_relative(q_abs))
        err, _ = worst(com[[0, 2]], expected)
        assert err < VALUE_TOL, f"q_abs={q_abs.tolist()} err={err}"
        assert abs(com[1]) < VALUE_TOL


def test_contact_jacobian_parity(setup):
    params, model, data, frames = setup
    T, _ = absolute_to_relative_map()
    T_inv = np.linalg.inv(T)
    rng = np.random.default_rng(SEED + 2)
    step = 1e-6
    for q_abs in random_absolute_configurations(rng, SAMPLES):
        q = absolute_to_relative(q_abs)
        expected = (oracle.J_func(params, q_abs) @ T_inv).reshape(4, 2, 9)
        actual = contact_jacobians(model, data, q, frames)
        err, index = worst(actual, expected)
        assert err < JACOBIAN_TOL, f"q_abs={q_abs.tolist()} worst={index} err={err}"
        finite = np.zeros_like(actual)
        for column in range(9):
            delta = np.zeros(9)
            delta[column] = step
            finite[:, :, column] = (
                contact_positions(model, data, q + delta, frames)
                - contact_positions(model, data, q - delta, frames)) / (2.0 * step)
        err, index = worst(actual, finite)
        assert err < FD_TOL, f"finite difference q={q.tolist()} worst={index} err={err}"


def test_mass_matrix_parity(setup):
    params, model, data, _ = setup
    T, _ = absolute_to_relative_map()
    T_inv = np.linalg.inv(T)
    rng = np.random.default_rng(SEED + 3)
    for q_abs in random_absolute_configurations(rng, SAMPLES):
        expected = T_inv.T @ oracle.M_func(params, q_abs) @ T_inv
        actual = mass_matrix(model, data, absolute_to_relative(q_abs))
        err, index = worst(actual, expected)
        assert err < JACOBIAN_TOL, f"q_abs={q_abs.tolist()} worst={index} err={err}"
        assert np.max(np.abs(actual - actual.T)) < 1e-14
        assert np.min(np.linalg.eigvalsh(actual)) > 0.0
        assert abs(actual[0, 0] - params.total_mass) < 1e-12
        assert abs(actual[1, 1] - params.total_mass) < 1e-12


def test_energy_parity(setup):
    params, model, data, _ = setup
    rng = np.random.default_rng(SEED + 4)
    for q_abs in random_absolute_configurations(rng, SAMPLES):
        v_abs = rng.normal(size=9)
        q = absolute_to_relative(q_abs)
        v = absolute_velocity_to_relative(v_abs)
        kinetic = pin.computeKineticEnergy(model, data, q, v)
        potential = pin.computePotentialEnergy(model, data, q)
        expected = oracle.lagrangian(params, q_abs, v_abs)
        err = abs(kinetic - potential - expected)
        assert err < 1e-10, f"q_abs={q_abs.tolist()} v_abs={v_abs.tolist()} err={err}"


def test_generalized_force_map():
    """``T^-T B_abs^T u == S^T Pi u``: Julia actuation equals the joint selector."""
    T, _ = absolute_to_relative_map()
    S = actuation_selector()
    rng = np.random.default_rng(SEED + 5)
    for _ in range(SAMPLES):
        u = rng.normal(size=6)
        expected = np.linalg.solve(T.T, oracle.B_func().T @ u)
        actual = S.T @ julia_torque_to_pinocchio(u)
        assert np.max(np.abs(actual - expected)) < 1e-14


def test_coordinate_maps_roundtrip_and_mirror():
    rng = np.random.default_rng(SEED + 6)
    for _ in range(SAMPLES):
        q_abs = rng.normal(size=9)
        assert np.max(np.abs(relative_to_absolute(absolute_to_relative(q_abs)) - q_abs)) < 1e-14
        q = rng.normal(size=9)
        mirrored = mirror_configuration(q, stride=0.5)
        assert np.max(np.abs(mirror_configuration(mirrored, stride=-0.5) - q)) < 1e-14
        assert np.allclose(mirrored[3:6], q[6:9]) and np.allclose(mirrored[6:9], q[3:6])
        assert mirrored[0] == q[0] + 0.5


def test_sign_conventions(setup):
    params, model, data, frames = setup
    zero = np.zeros(9)
    zero[1] = 1.0
    # Foot perpendicular to the calf at zero ankle angle: toe at +x of the ankle.
    positions = contact_positions(model, data, zero, frames)
    ankle_z = 1.0 - params.l_thigh - params.l_calf
    assert np.allclose(positions[0], [params.l_foot, ankle_z])
    assert np.allclose(positions[1], [-params.d_foot, ankle_z])
    # Increasing pitch moves the torso COM toward -x (Julia: x - d sin(theta)).
    pitched = zero.copy()
    pitched[2] = 0.1
    assert pin.centerOfMass(model, data, pitched)[0] < pin.centerOfMass(model, data, zero)[0]
    # Increasing hip1 moves foot 1 toward +x (Julia: x + l sin(theta)).
    hipped = zero.copy()
    hipped[3] = 0.1
    assert contact_positions(model, data, hipped, frames)[0, 0] > positions[0, 0]
    # Julia foot angle pi/2 (horizontal) maps to zero ankle angle.
    q_abs = np.zeros(9)
    q_abs[7] = q_abs[8] = 0.5 * np.pi
    assert np.allclose(absolute_to_relative(q_abs), 0.0)

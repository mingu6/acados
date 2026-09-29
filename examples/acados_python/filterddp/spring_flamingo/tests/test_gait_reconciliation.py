"""Model parity: the stored ContactImplicitMPC.jl gait through the Pinocchio model.

The gait ``gait_forward_36_4.jld2`` supplies the OCP's start state and stride.
It was produced with the Julia integrator (``cimpc_mechanics_residual``,
contact Jacobian at ``q_next``, impulses unscaled) and must satisfy that
residual to solver tolerance through the Pinocchio model. The OCP's
RNEA-midpoint residual differs from it by an exact, verified decomposition.
"""

import numpy as np
import pytest

from contact_implicit_numpy import (
    cimpc_mechanics_residual, contact_quantities, mechanics_residual,
)
from flamingo_gait import load_gait
from flamingo_model import FlamingoParameters, actuation_selector, build_model, contact_jacobians

CIMPC_RESIDUAL_TOL = 1e-6
RNEA_RESIDUAL_BOUND = 0.1  # fraction of m g h; integrator difference, not a defect
# The gait was solved by the Julia interior-point method with r_tol = kappa_tol = 1e-8,
# so complementarity and cone quantities are satisfied only to that order.
FIXTURE_TOL = 1e-7


@pytest.fixture(scope="module")
def setup():
    try:
        gait = load_gait()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    params = FlamingoParameters()
    model, frames = build_model(params)
    return gait, params, model, model.createData(), frames


def gait_impulses(gait, k):
    return np.column_stack([gait.tangential_impulse[k], gait.gamma[k]])


def test_gait_metadata(setup):
    gait, *_ = setup
    assert gait.horizon == 70
    assert abs(gait.h - 0.015672816555634485) < 1e-15
    assert abs(gait.mu - 0.1) < 1e-15
    assert abs(gait.stride - 0.23172) < 1e-4
    # Two mirrored half steps: q[35 + k] equals the mirror of q[k] shifted by half a stride.
    from flamingo_model import mirror_configuration
    q = gait.q
    half = 35
    shift = q[half, 0] - q[0, 0]
    errors = [np.max(np.abs(q[half + k] - mirror_configuration(q[k], shift)))
              for k in range(0, half + 2)]
    assert max(errors) < 1e-6, f"half-stride mirror symmetry error {max(errors)}"


def test_cimpc_residual_is_satisfied(setup):
    gait, _, model, data, frames = setup
    selector = actuation_selector()
    q = gait.q
    worst = 0.0
    worst_interval = -1
    for k in range(gait.horizon):
        residual = cimpc_mechanics_residual(
            model, data, frames, selector, gait.h, q[k], q[k + 1], q[k + 2],
            gait.torque[k], gait_impulses(gait, k))
        magnitude = np.max(np.abs(residual))
        if magnitude > worst:
            worst, worst_interval = magnitude, k
    print(f"cimpc residual max {worst:.3e} at interval {worst_interval}")
    assert worst < CIMPC_RESIDUAL_TOL


def test_rnea_midpoint_residual_decomposition(setup):
    """The RNEA-midpoint residual differs from the CIMPC one by exactly

        [M(qbar_prev) - M(qbar)] v_prev
        + h/2 [nle(qbar, v) - nle(qbar_prev, v_prev)]
        - [J(qbar) - J(q_next)]^T p,

    which is verified term by term; the magnitude is reported relative to
    the static weight impulse ``m g h`` and bounded only loosely.
    """
    import pinocchio as pin
    from flamingo_model import mass_matrix
    gait, params, model, data, frames = setup
    selector = actuation_selector()
    q = gait.q
    h = gait.h
    weight_impulse = params.total_mass * params.gravity * h
    magnitudes, mass_terms, bias_terms, jacobian_terms = [], [], [], []
    for k in range(gait.horizon):
        impulses = gait_impulses(gait, k)
        rnea = mechanics_residual(model, data, frames, selector, h, q[k], q[k + 1],
                                  q[k + 2], gait.torque[k], impulses)
        cimpc = cimpc_mechanics_residual(model, data, frames, selector, h, q[k], q[k + 1],
                                         q[k + 2], gait.torque[k], impulses)
        q_bar_prev, q_bar = 0.5 * (q[k] + q[k + 1]), 0.5 * (q[k + 1] + q[k + 2])
        v_prev, v = (q[k + 1] - q[k]) / h, (q[k + 2] - q[k + 1]) / h
        mass_term = (mass_matrix(model, data, q_bar_prev) - mass_matrix(model, data, q_bar)) @ v_prev
        bias_term = 0.5 * h * (np.array(pin.nonLinearEffects(model, data, q_bar, v))
                               - np.array(pin.nonLinearEffects(model, data, q_bar_prev, v_prev)))
        jac_mid = contact_jacobians(model, data, q_bar, frames)
        jac_next = contact_jacobians(model, data, q[k + 2], frames)
        jacobian_term = -sum((jac_mid[i] - jac_next[i]).T @ impulses[i] for i in range(4))
        predicted = cimpc + mass_term + bias_term + jacobian_term
        assert np.max(np.abs(rnea - predicted)) < 1e-10, f"interval {k}"
        magnitudes.append(np.max(np.abs(rnea)))
        mass_terms.append(np.max(np.abs(mass_term)))
        bias_terms.append(np.max(np.abs(bias_term)))
        jacobian_terms.append(np.max(np.abs(jacobian_term)))
    worst = int(np.argmax(magnitudes))
    print(f"rnea midpoint residual max {magnitudes[worst]:.3e} at interval {worst} "
          f"({100 * magnitudes[worst] / weight_impulse:.1f}% of m g h), "
          f"median {np.median(magnitudes):.3e}; term maxima: mass {max(mass_terms):.3e}, "
          f"bias {max(bias_terms):.3e}, jacobian {max(jacobian_terms):.3e}")
    assert max(magnitudes) < RNEA_RESIDUAL_BOUND * weight_impulse


def test_gait_contact_complementarity(setup):
    gait, _, model, data, frames = setup
    q = gait.q
    min_gap = np.inf
    max_normal_product = 0.0
    min_cone = np.inf
    max_velocity_difference = 0.0
    for k in range(gait.horizon):
        gaps, tangential = contact_quantities(model, data, frames, gait.h, q[k + 1], q[k + 2])
        jac = contact_jacobians(model, data, q[k + 2], frames)
        julia_tangential = np.array([jac[i][0] @ (q[k + 2] - q[k + 1]) / gait.h
                                     for i in range(4)])
        max_velocity_difference = max(max_velocity_difference,
                                      np.max(np.abs(julia_tangential - tangential)))
        min_gap = min(min_gap, np.min(gaps))
        max_normal_product = max(max_normal_product, np.max(gait.gamma[k] * gaps))
        margin = gait.mu * gait.gamma[k] - (gait.b[k, 0::2] + gait.b[k, 1::2])
        min_cone = min(min_cone, np.min(margin))
    print(f"min gap {min_gap:.3e}, max gamma*gap {max_normal_product:.3e}, "
          f"min cone margin {min_cone:.3e}, "
          f"max |J dq/h - dp/h| {max_velocity_difference:.3e}")
    assert min_gap > -FIXTURE_TOL
    assert max_normal_product < 1e-6
    assert min_cone > -FIXTURE_TOL


def test_gait_is_physically_loaded(setup):
    gait, params, *_ = setup
    mean_normal_force = np.mean(np.sum(gait.gamma, axis=1)) / gait.h
    assert abs(mean_normal_force - params.total_mass * params.gravity) < 2.0

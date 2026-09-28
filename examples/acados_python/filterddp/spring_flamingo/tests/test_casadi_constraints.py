"""CasADi stage rows versus the numpy oracle, plus rank and curvature invariants.

Random points are drawn interior to all bounds with a fixed seed; the
configurations come from the stored gait so the mechanics are evaluated in
realistic postures.  Failures print the offending sample.
"""

import numpy as np
import pytest

import contact_implicit_numpy as oracle
from contact_implicit_casadi import build_contact_implicit
from flamingo_gait import load_gait
from flamingo_model import FlamingoParameters, actuation_selector, build_model

SEED = 1
SAMPLES = 32
VALUE_TOL = 1e-10
JACOBIAN_TOL = 1e-6
H = 0.015672816555634485
MU = 0.74
TAU_C = 0.02
DELTA = 0.01


@pytest.fixture(scope="module")
def setup():
    model, frames = build_model(FlamingoParameters())
    try:
        configurations = load_gait().q
    except FileNotFoundError:
        configurations = None
    symbolic = build_contact_implicit(model, frames, actuation_selector(), H, MU, TAU_C, DELTA)
    return model, model.createData(), frames, configurations, symbolic


def random_point(rng, layout, configurations):
    """Interior sample: gait-based configurations, random positive controls."""
    if configurations is not None:
        k = rng.integers(0, configurations.shape[0] - 2)
        q_prev, q = configurations[k], configurations[k + 1]
        q_next = configurations[k + 2] + 0.01 * rng.normal(size=layout.nq)
    else:
        q_prev = rng.normal(size=layout.nq)
        q = q_prev + 0.02 * rng.normal(size=layout.nq)
        q_next = q + 0.02 * rng.normal(size=layout.nq)
    u = rng.uniform(0.1, 1.0, size=layout.nu)
    u[layout.torque] = rng.normal(size=layout.nu_torque)
    u[layout.next_configuration] = q_next
    u[layout.dummy] = rng.normal(size=layout.n_dummy)
    for i in range(layout.nc):
        u[layout.impulse(i)] = [0.2 * rng.normal(), rng.uniform(0.1, 2.0)]
    return np.concatenate([q_prev, q]), u


def worst(a, b):
    err = np.abs(np.asarray(a) - np.asarray(b))
    index = np.unravel_index(np.argmax(err), err.shape)
    return err[index], index


def test_dimensions(setup):
    *_, symbolic = setup
    layout = symbolic.layout
    assert (layout.nx, layout.nu, layout.n_dummy) == (18, 52, 9)
    assert symbolic.h_eq.shape[0] == 25
    assert symbolic.h_ineq.shape[0] == 16
    assert symbolic.h_periodic.shape[0] == 9
    assert symbolic.n_h == 50
    assert symbolic.parameters.size == 10


def test_values_match_numpy_oracle(setup):
    model, data, frames, configurations, symbolic = setup
    layout = symbolic.layout
    rng = np.random.default_rng(SEED)
    for _ in range(SAMPLES):
        x, u = random_point(rng, layout, configurations)
        expected_eq, expected_ineq = oracle.stage_constraints(
            model, data, frames, actuation_selector(), layout, H, MU, TAU_C, DELTA, x, u)
        actual_eq = np.array(symbolic.functions["h_eq"](x, u)).ravel()
        actual_ineq = np.array(symbolic.functions["h_ineq"](x, u)).ravel()
        err, index = worst(actual_eq, expected_eq)
        assert err < VALUE_TOL, f"equality row {index} err={err} x={x.tolist()} u={u.tolist()}"
        err, index = worst(actual_ineq, expected_ineq)
        assert err < VALUE_TOL, f"inequality row {index} err={err}"


def test_clearance_rows_are_inactive_at_rest_and_scale_with_speed(setup):
    """Zero horizontal speed leaves g >= 0; speed v demands tau_c (sqrt(v^2+d^2)-d)."""
    _, _, _, configurations, symbolic = setup
    layout = symbolic.layout
    rng = np.random.default_rng(SEED + 6)
    x, u = random_point(rng, layout, configurations)
    q = x[layout.nq:]
    u[layout.next_configuration] = q  # nothing moves
    ineq = np.array(symbolic.functions["h_ineq"](x, u)).ravel()
    assert np.max(np.abs(ineq[12:16] - ineq[:4])) < 1e-14
    moved = q.copy()
    moved[0] += 1.0 * H  # translate the whole robot at 1 m/s
    u[layout.next_configuration] = moved
    ineq = np.array(symbolic.functions["h_ineq"](x, u)).ravel()
    expected = TAU_C * (np.sqrt(1.0 + DELTA ** 2) - DELTA)
    assert np.max(np.abs(ineq[:4] - ineq[12:16] - expected)) < 1e-10


def test_jacobians_match_finite_differences(setup):
    _, _, _, configurations, symbolic = setup
    layout = symbolic.layout
    rng = np.random.default_rng(SEED + 1)
    step = 1e-6

    def values(x, u):
        return np.concatenate([np.array(symbolic.functions["h_eq"](x, u)).ravel(),
                               np.array(symbolic.functions["h_ineq"](x, u)).ravel()])

    for _ in range(SAMPLES // 4):
        x, u = random_point(rng, layout, configurations)
        analytic_u = np.vstack([np.array(symbolic.functions["jac_eq_u"](x, u)),
                                np.array(symbolic.functions["jac_ineq_u"](x, u))])
        analytic_x = np.vstack([np.array(symbolic.functions["jac_eq_x"](x, u)),
                                np.array(symbolic.functions["jac_ineq_x"](x, u))])
        finite_u = np.column_stack([(values(x, u + step * e) - values(x, u - step * e)) / (2 * step)
                                    for e in np.eye(layout.nu)])
        finite_x = np.column_stack([(values(x + step * e, u) - values(x - step * e, u)) / (2 * step)
                                    for e in np.eye(layout.nx)])
        for analytic, finite, name in ((analytic_u, finite_u, "du"), (analytic_x, finite_x, "dx")):
            scale = 1.0 + np.max(np.abs(analytic))
            err, index = worst(analytic, finite)
            assert err / scale < JACOBIAN_TOL, f"{name} worst {index} err={err} scale={scale}"


def test_equality_control_jacobian_has_full_row_rank(setup):
    """acados FilterDDP's null-space elimination needs rank(dh_eq/du) = n_eq; it does not check."""
    _, _, _, configurations, symbolic = setup
    layout, P = symbolic.layout, symbolic.parameters
    rng = np.random.default_rng(SEED + 2)
    smallest = np.inf
    for flag in (0.0, 1.0):
        for _ in range(SAMPLES // 2):
            x, u = random_point(rng, layout, configurations)
            p = rng.normal(size=P.size)
            p[P.periodicity_flag] = flag
            jac = np.vstack([np.array(symbolic.functions["jac_eq_u"](x, u)),
                             np.array(symbolic.functions["jac_periodic_u"](x, u, p))])
            singular = np.linalg.svd(jac, compute_uv=False)
            smallest = min(smallest, singular[-1])
            assert jac.shape == (34, 52)
            assert np.linalg.matrix_rank(jac) == 34, f"flag={flag} u={u.tolist()}"
    print(f"smallest singular value of the equality control Jacobian: {smallest:.3e}")
    assert smallest > 1e-8


def test_periodicity_rows(setup):
    """Flag 0 pins the dummy controls, flag 1 pins q_next to the target."""
    _, _, _, configurations, symbolic = setup
    layout, P = symbolic.layout, symbolic.parameters
    rng = np.random.default_rng(SEED + 5)
    for flag in (0.0, 1.0):
        x, u = random_point(rng, layout, configurations)
        p = rng.normal(size=P.size)
        p[P.periodicity_flag] = flag
        rows = np.array(symbolic.functions["h_periodic"](x, u, p)).ravel()
        expected = (flag * (u[layout.next_configuration] - p[P.periodicity_target])
                    + (1.0 - flag) * u[layout.dummy])
        assert np.max(np.abs(rows - expected)) < 1e-12


def test_contracted_hessian_is_symmetric_and_finite(setup):
    _, _, _, configurations, symbolic = setup
    layout = symbolic.layout
    rng = np.random.default_rng(SEED + 3)
    for _ in range(SAMPLES // 4):
        x, u = random_point(rng, layout, configurations)
        hess = np.array(symbolic.functions["hess_eq"](x, u, rng.normal(size=layout.n_equality)))
        assert hess.shape == (layout.nu + layout.nx, layout.nu + layout.nx)
        assert np.all(np.isfinite(hess))
        assert np.max(np.abs(hess - hess.T)) < 1e-12
        zero = np.array(symbolic.functions["hess_eq"](x, u, np.zeros(layout.n_equality)))
        assert np.max(np.abs(zero)) == 0.0

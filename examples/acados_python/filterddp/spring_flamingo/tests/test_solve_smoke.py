"""OCP assembly, initial guess, and solve.

``test_solve_converges`` runs the documented OCP end to end (about 15 s
including code generation) and checks the reproducible invariants: status,
feasibility, periodicity, and the contact-mode structure. It does not pin
the cost of transport or iteration count, which move between nearby local
solutions (README.md, Results).
Requires ``ACADOS_SOURCE_DIR`` to point at the FilterDDP fork.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("acados_template")

from flamingo_gait import load_gait  # noqa: E402
from flamingo_model import contact_positions, mirror_configuration  # noqa: E402
from flamingo_ocp import (  # noqa: E402
    FlamingoOcpOptions, _ankle_positions, build_problem, initial_controls, initial_knots,
    leg_inverse_kinematics, rollout_states,
)

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "test"


def fork_available() -> bool:
    source = os.environ.get("ACADOS_SOURCE_DIR", "")
    return Path(source, "acados", "ocp_nlp", "ocp_nlp_filterddp.c").is_file()


@pytest.fixture(scope="module")
def problem():
    try:
        gait = load_gait()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return build_problem(gait, FlamingoOcpOptions(print_level=0))


def test_periodic_targets(problem):
    """Targets are the mirrored start knots advanced by the step, lifted to the
    pinned gap; the stored gait satisfies them up to that lift."""
    gait, P, options = problem.gait, problem.symbolic.parameters, problem.options
    assert problem.periodic_stages == (options.horizon - 2, options.horizon - 1)
    assert abs(problem.step - gait.stride / 2) < 1e-15
    for stage, source in zip(problem.periodic_stages, (gait.q[0], gait.q[1])):
        target = problem.stage_parameters[stage, P.periodicity_target]
        assert problem.stage_parameters[stage, P.periodicity_flag] == 1.0
        lift = target - mirror_configuration(source, problem.step)
        assert np.allclose(lift[[0, *range(2, 9)]], 0.0) and 0.0 <= lift[1] <= options.pinned_minimum_gap + 1e-8
        gaps = contact_positions(problem.model, problem.data, target, problem.frames)[:, 1]
        assert gaps.min() >= options.pinned_minimum_gap - 1e-15
        # The gait's own knot N + 1 (= q[36]) matches the target up to the lift.
        assert np.max(np.abs(target - gait.q[stage + 2])) < 1e-6 + options.pinned_minimum_gap
    others = [k for k in range(options.horizon + 1) if k not in problem.periodic_stages]
    assert np.all(problem.stage_parameters[others] == 0.0)


def test_bezier_initial_guess(problem):
    options = problem.options
    knots = initial_knots(problem)
    targets = problem.stage_parameters[list(problem.periodic_stages),
                                       problem.symbolic.parameters.periodicity_target]
    assert np.max(np.abs(knots[0] - problem.gait.q[1])) < 1e-12
    assert np.max(np.abs(knots[-1] - targets[1])) < 1e-5  # target carries the pinned lift
    heights = np.array([contact_positions(problem.model, problem.data, q, problem.frames)[:, 1]
                        for q in knots])
    assert heights.min() > -1e-9
    assert abs(heights[:, :2].min(axis=1).max() - options.bezier_apex) < 1e-3  # foot 1 swings
    assert heights[:, 2:].min(axis=1).max() < 1e-9  # foot 2 stays in stance
    assert np.max(np.abs(heights[:, 0] - heights[:, 1])) < 1e-9  # feet stay flat
    assert np.max(np.abs(heights[:, 2] - heights[:, 3])) < 1e-9
    q = knots[options.horizon // 3]
    ankle = _ankle_positions(problem, q)[0]
    angles = leg_inverse_kinematics(q[:2], ankle, q[2], q[2] + q[3:6].sum(), np.sign(q[4]),
                                    problem.params)
    assert np.max(np.abs(angles - q[3:6])) < 1e-10


def test_initial_controls(problem):
    layout = problem.layout
    controls = initial_controls(problem)
    assert np.all(controls > problem.lower) and np.all(controls < problem.upper)
    assert np.all(controls[:, layout.torque] == 0.0)
    assert np.all(controls[:, layout.tangential_impulse_indices()] == 0.0)
    assert np.allclose(controls[:, layout.normal_impulse_indices()], 1e-3)
    assert np.all(controls[:, layout.dummy] == 0.0)
    states = rollout_states(problem, controls)
    assert np.allclose(states[0], problem.x0)
    assert np.allclose(states[1:, :9], states[:-1, 9:])
    assert np.allclose(states[1:, 9:], controls[:, layout.next_configuration])


@pytest.mark.skipif(not fork_available(), reason="ACADOS_SOURCE_DIR is not the FilterDDP fork")
def test_solve_converges():
    from solve_flamingo import run
    try:
        load_gait()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    summary = run(FlamingoOcpOptions(print_level=0), OUTPUT)
    solution, modes = summary["solution"], summary["gait"]["modes"]
    assert summary["status"] == 0
    assert solution["max_equality_violation"] < 1e-6
    assert solution["min_inequality"] > -1e-8
    assert solution["periodicity_residual"] < 1e-12
    # tau_c = 0.02 s at the ~1 m/s peak swing speed asks for about 2 cm; nothing
    # caps the apex, so only a lower bound and a sanity ceiling are checked.
    assert 0.015 < summary["gait"]["clearance"][0]["apex"] < 0.10
    assert 0.03 < solution["cost_of_transport"] < 0.12
    # Contact-mode structure of a toe-push-off walking gait.
    assert modes["fractions"]["toe_only"] > 0.10
    assert 0.25 < modes["fractions"]["swing"] < 0.40
    assert modes["distance_to_posa"] < 0.3

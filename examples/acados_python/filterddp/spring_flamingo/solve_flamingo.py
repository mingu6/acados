"""Solve the Spring Flamingo OCP with acados FilterDDP and write diagnostics.

Usage (see README.md for the environment)::

    python solve_flamingo.py --output output/flamingo --repeat 4

Writes ``trajectory.csv`` (solution), ``initial_trajectory.csv`` (initial
guess), ``statistics.csv`` (per-iteration FilterDDP statistics), and
``summary.json``. Every check in the summary is recomputed in numpy from the
returned trajectory (``contact_implicit_numpy``), not taken from acados
residuals, because the FilterDDP fork's multiplier export is unreliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from ctypes import byref, c_double
from pathlib import Path

import numpy as np

from contact_implicit_numpy import clearance_requirement, contact_quantities, stage_constraints
from flamingo_gait import load_gait
from flamingo_model import CONTACT_NAMES, actuation_selector, contact_positions
from flamingo_ocp import (
    FlamingoOcpOptions, FlamingoProblem, build_problem, create_solver, initial_controls,
    initialize_solver, rollout_states,
)
from gait_metrics import gait_metrics

STAT_ROWS = ("iter", "du_inf", "pr_inf", "cs_inf", "objective", "mu", "reg", "alpha", "ls")


def read_filterddp_value(solver, field: str) -> float:
    value = c_double(0.0)
    solver._AcadosOcpSolver__acados_lib.ocp_nlp_get(
        solver.nlp_solver, field.encode("utf-8"), byref(value))
    return value.value


def evaluate_trajectory(problem: FlamingoProblem, xs: np.ndarray, us: np.ndarray) -> dict:
    """Constraint residuals, slip, and cost of transport in the documented equations."""
    layout, options, h = problem.layout, problem.options, problem.h
    selector = actuation_selector()
    weight = problem.params.total_mass * problem.params.gravity
    p_ref = 0.25 * weight * h
    max_equality, worst_equality = 0.0, (-1, -1)
    min_inequality, worst_inequality = np.inf, (-1, -1)
    max_loaded_slip, work = 0.0, 0.0
    for k in range(us.shape[0]):
        u = us[k]
        eq, ineq = stage_constraints(problem.model, problem.data, problem.frames, selector,
                                     layout, h, options.mu, options.clearance_time,
                                     options.clearance_smoothing, xs[k], u)
        row = int(np.argmax(np.abs(eq)))
        if abs(eq[row]) > max_equality:
            max_equality, worst_equality = abs(eq[row]), (k, row)
        row = int(np.argmin(ineq))
        if ineq[row] < min_inequality:
            min_inequality, worst_inequality = ineq[row], (k, row)
        q, q_next = xs[k][layout.nq:], u[layout.next_configuration]
        _, tangential = contact_quantities(problem.model, problem.data, problem.frames, h,
                                           q, q_next)
        normal = u[layout.normal_impulse_indices()]
        max_loaded_slip = max(max_loaded_slip, float(np.max(normal / p_ref * np.abs(tangential))))
        work += h * np.sum(np.abs(u[layout.torque] * (selector @ ((q_next - q) / h))))
    targets = problem.stage_parameters[list(problem.periodic_stages),
                                       problem.symbolic.parameters.periodicity_target]
    periodicity = max(np.max(np.abs(us[k][layout.next_configuration] - target))
                      for k, target in zip(problem.periodic_stages, targets))
    distance = xs[-1][layout.nq] - xs[0][layout.nq]
    torques = us[:, layout.torque]
    return {
        "max_equality_violation": max_equality,
        "worst_equality_stage_row": worst_equality,
        "min_inequality": min_inequality,
        "worst_inequality_stage_row": worst_inequality,
        "periodicity_residual": periodicity,
        "max_sigma_n": float(us[:, layout.normal_relaxation_offset:layout.friction_dual_offset].max()),
        "max_sigma_f": float(us[:, layout.friction_relaxation_offset:layout.dummy_offset].max()),
        "max_loaded_slip_speed": max_loaded_slip,
        "distance": distance,
        "cost_of_transport": work / (weight * distance),
        "torque_range": [float(torques.min()), float(torques.max())],
    }


def write_trajectory_csv(problem: FlamingoProblem, xs: np.ndarray, us: np.ndarray,
                         path: Path) -> None:
    layout = problem.layout
    header = ["time"] + [f"q{i}" for i in range(layout.nq)]
    header += [f"{c}_{axis}" for c in CONTACT_NAMES for axis in ("x", "z")]
    header += [f"tau{i}" for i in range(layout.nu_torque)]
    for c in CONTACT_NAMES:
        header += [f"{c}_pt", f"{c}_pn", f"{c}_gap", f"{c}_clearance_margin", f"{c}_sigma_n",
                   f"{c}_lambda_plus", f"{c}_lambda_minus", f"{c}_sigma_f_plus",
                   f"{c}_sigma_f_minus"]
    options = problem.options
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for k in range(xs.shape[0]):
            q = xs[k][layout.nq:]
            row = [k * problem.h, *q, *contact_positions(problem.model, problem.data, q,
                                                         problem.frames).ravel()]
            if k < us.shape[0]:
                u = us[k]
                gaps, tangential = contact_quantities(problem.model, problem.data,
                                                      problem.frames, problem.h, q,
                                                      u[layout.next_configuration])
                margin = gaps - clearance_requirement(tangential, options.clearance_time,
                                                      options.clearance_smoothing)
                row += list(u[layout.torque])
                for i in range(layout.nc):
                    row += [*u[layout.impulse(i)], gaps[i], margin[i],
                            u[layout.normal_relaxation_offset + i],
                            *u[layout.friction_dual(i)], *u[layout.friction_relaxation(i)]]
            else:
                row += [""] * (layout.nu_torque + 9 * layout.nc)
            writer.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v
                             for v in row])


def run(options: FlamingoOcpOptions, output_dir: Path, repeat: int = 1,
        gait_path: str | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = build_problem(load_gait(gait_path), options)
    solver = create_solver(problem, output_dir)
    controls = initial_controls(problem)
    initial_states = rollout_states(problem, controls)
    write_trajectory_csv(problem, initial_states, controls, output_dir / "initial_trajectory.csv")

    results = []
    for _ in range(max(1, repeat)):
        initialize_solver(problem, solver, controls)
        start = time.perf_counter()
        status = solver.solve()
        results.append((int(status), time.perf_counter() - start))
    stats = np.array(solver.get_stats("statistics"))
    xs = np.array([solver.get(k, "x") for k in range(options.horizon + 1)])
    us = np.array([solver.get(k, "u") for k in range(options.horizon)])
    write_trajectory_csv(problem, xs, us, output_dir / "trajectory.csv")
    with open(output_dir / "statistics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(STAT_ROWS[:stats.shape[0]])
        writer.writerows([[repr(float(v)) for v in column] for column in stats.T])

    times = [wall for _, wall in results]
    weight = problem.params.total_mass * problem.params.gravity
    summary = {
        "status": results[-1][0],
        "statuses": [s for s, _ in results],
        "iterations": int(solver.get_stats("nlp_iter")),
        "objective": float(solver.get_cost()),
        # Median over the timed repeats; the first solve is a warmup when repeat > 1.
        "median_solve_time_s": float(np.median(times[1:] if len(times) > 1 else times)),
        "solve_times_s": times,
        "final_barrier": read_filterddp_value(solver, "filterddp_mu"),
        "final_regularization": read_filterddp_value(solver, "filterddp_reg_last"),
        "options": {k: v for k, v in options.__dict__.items()},
        "horizon_s": options.horizon * problem.h,
        "step_m": problem.step,
        "initial_guess": evaluate_trajectory(problem, initial_states, controls),
        "solution": evaluate_trajectory(problem, xs, us),
        "gait": gait_metrics(output_dir / "trajectory.csv", weight),
    }
    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2,
                  default=lambda v: v.item() if hasattr(v, "item") else str(v))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="output/flamingo")
    parser.add_argument("--repeat", type=int, default=1,
                        help="solves from the same initial guess; the first is a warmup")
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--print-level", type=int, default=1)
    parser.add_argument("--gait", default=None, help="ContactImplicitMPC.jl gait .jld2")
    parser.add_argument("--pinned-gap", type=float, default=FlamingoOcpOptions.pinned_minimum_gap,
                        help="minimum contact gap of the pinned periodic knots [m]; "
                             "perturbing it samples the spread of local solutions")
    args = parser.parse_args()
    options = FlamingoOcpOptions(max_iter=args.max_iter, print_level=args.print_level,
                                 pinned_minimum_gap=args.pinned_gap)
    summary = run(options, Path(args.output), repeat=args.repeat, gait_path=args.gait)
    solution, modes = summary["solution"], summary["gait"]["modes"]
    print(f"status {summary['status']}, {summary['iterations']} iterations, "
          f"{summary['median_solve_time_s']:.2f} s, objective {summary['objective']:.5f}")
    print(f"cost of transport {solution['cost_of_transport']:.4f}, max equality violation "
          f"{solution['max_equality_violation']:.1e}, periodicity residual "
          f"{solution['periodicity_residual']:.1e}, swing apex "
          f"{100 * summary['gait']['clearance'][0]['apex']:.1f} cm")
    print(f"modes {modes['pattern']}, distance to Posa Fig. 6 {modes['distance_to_posa']:.2f}")
    return 0 if summary["status"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

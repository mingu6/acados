"""Solve CPC time-optimal waypoint flight with acados FilterDDP (terminal progress cost).

Usage (see README.md for the environment)::

    python solve_waypoint_flight.py --track three_gates --ipopt
    python solve_waypoint_flight.py --track lap --ipopt

FilterDDP starts from ``planner.py``'s own initial guess, the one IPOPT starts from.
``--ipopt`` also solves the unmodified CPC planner with IPOPT and evaluates the FilterDDP
solution in the CPC constraint function. Residuals in ``summary.json`` are recomputed from the
returned trajectory; the fork's multiplier export is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from cpc_reference import (cpc_guess_controls, cpc_violation, from_cpc_vector, make_planner, solve_cpc,
                           to_cpc_vector)
from quad_model import QuadParameters, Track, default_quad_path, resolve_track
from waypoint_ocp import WaypointOptions, build_problem, create_solver, evaluate, get_solution, initialize_solver

STAT_ROWS = ("iter", "du_inf", "pr_inf", "cs_inf", "objective", "mu", "reg", "alpha", "ls")


def parse_args(argv=None):
    d = WaypointOptions()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="three_gates", help="three_gates, lap, or a track.yaml path")
    p.add_argument("--quad", default=None, help="quad.yaml (default: quads/quad.yaml of $RPG_TIME_OPTIMAL_DIR)")
    p.add_argument("--nodes-per-gate", type=int, default=d.nodes_per_gate)
    p.add_argument("--endpoint-weight", type=float, default=d.endpoint_weight,
                   help="w of the terminal cost (w/2)||mu_N||^2")
    p.add_argument("--mu-init", type=float, default=d.mu_init, help="filterddp_mu_init (fork default 1.0)")
    p.add_argument("--kappa", type=float, default=d.kappa, help="filterddp_kappa_1 = kappa_2 (fork default 0.01)")
    p.add_argument("--tol", type=float, default=d.tol)
    p.add_argument("--max-iter", type=int, default=d.max_iter)
    p.add_argument("--print-level", type=int, default=0)
    p.add_argument("--ipopt", action="store_true", help="solve the CPC NLP with IPOPT as the reference")
    p.add_argument("--output", default=None, help="output directory (default output/<track>)")
    return p.parse_args(argv)


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_trajectory(path: Path, problem, xs: np.ndarray, us: np.ndarray) -> None:
    """Time, quadrotor state and rotor thrusts per node (thrusts of the last node are NaN)."""
    L, N = problem.layout, problem.N
    dt = us[0, L.u_t]/N
    header = ["t", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",
              "T0", "T1", "T2", "T3"]
    write_csv(path, header, ([f"{v:.12g}" for v in [k*dt, *xs[k, :13], *(us[k, L.u_thrust] if k < N else [np.nan]*4)]]
                             for k in range(N + 1)))


def main(argv=None) -> int:
    args = parse_args(argv)
    args.quad = args.quad or str(default_quad_path())
    track_path = resolve_track(args.track)
    out = Path(args.output or Path("output") / Path(track_path).stem)
    out.mkdir(parents=True, exist_ok=True)
    opts = WaypointOptions(nodes_per_gate=args.nodes_per_gate, endpoint_weight=args.endpoint_weight,
                           tol=args.tol, max_iter=args.max_iter, print_level=args.print_level,
                           mu_init=args.mu_init, kappa=args.kappa)
    problem = build_problem(QuadParameters.from_yaml(args.quad), Track.from_yaml(track_path), opts)
    L = problem.layout
    print(f"track {track_path.name}: NW={L.nw} N={problem.N} nx={L.nx} nu={L.nu}")
    summary: dict = {"track": str(track_path), "N": problem.N, "options": vars(opts)}

    ipopt_options = {"max_iter": 10000, "print_level": 0, "tol": 1e-9} if args.ipopt else None
    planner = make_planner(args.quad, track_path, opts.nodes_per_gate, opts.tolerance, opts.vel_guess,
                           ipopt_options)
    if args.ipopt:
        ref = solve_cpc(planner)
        print(f"IPOPT:     {ref['status']} iter {ref['iter']} t {ref['t']:.6f} wall {ref['wall_s']:.1f} s")
        summary["ipopt"] = {k: ref[k] for k in ("t", "iter", "status", "wall_s")}

    t0 = time.perf_counter()
    solver = create_solver(problem, out)
    summary["build_s"] = time.perf_counter() - t0
    initialize_solver(problem, solver, cpc_guess_controls(problem, planner))
    t0 = time.perf_counter()
    status = solver.solve()
    wall = time.perf_counter() - t0
    xs, us = get_solution(problem, solver)
    n_iter = int(solver.get_stats("nlp_iter"))
    ev = evaluate(problem, xs, us)
    summary.update({"status": int(status), "iterations": n_iter, "wall_s": wall, "solution": ev,
                    "cpc_violation": cpc_violation(planner, to_cpc_vector(problem, xs, us))})
    print(f"FilterDDP: status {status} iter {n_iter} t {ev['t']:.6f} wall {wall:.1f} s")
    print(f"  rk4 defect {ev['rk4_defect'][0]:.1e}, equality {ev['equality'][0]:.1e}, "
          f"inequality {ev['inequality'][0]:.1e}, bounds {ev['control_bound'][0]:.1e}, mu_N {ev['mu_final']:.1e}")
    if args.ipopt:
        print(f"  t - t_IPOPT = {ev['t'] - ref['t']:+.1e}")

    write_trajectory(out / "trajectory.csv", problem, xs, us)
    if args.ipopt:
        write_trajectory(out / "ipopt_trajectory.csv", problem, *from_cpc_vector(problem, ref["x"]))
    stats = solver.get_stats("statistics")
    write_csv(out / "statistics.csv", STAT_ROWS,
              ([f"{v:.12g}" for v in stats[:, i]] for i in range(min(stats.shape[1], n_iter + 1))))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return 0 if status == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

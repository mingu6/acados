"""The unmodified CPC planner of https://github.com/uzh-rpg/rpg_time_optimal as the IPOPT reference.

Its ``src`` directory (from the checkout named by ``RPG_TIME_OPTIMAL_DIR``) is imported read-only. Its modules import PyYAML, which the example
environment lacks, so a shim backed by ``quad_model.load_yaml`` is registered first; the
planner's parameter derivations (``Quad.load``), NLP and initial guess (``Planner.setup``)
run unchanged.

The CPC decision vector is
``[t, x_0 (13), mu_0 (NW), {u_i (4), x_{i+1} (13), lambda_i, tau_i, mu_{i+1}}_{i<N}]``.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import types
from pathlib import Path

import casadi as ca
import numpy as np

from quad_model import NQ, NT, load_yaml, rpg_dir


def _import_rpg():
    if "yaml" not in sys.modules:
        shim = types.ModuleType("yaml")
        shim.FullLoader = None
        shim.load = lambda stream, Loader=None: load_yaml(stream.name)
        sys.modules["yaml"] = shim
    src = str(rpg_dir() / "src")
    if src not in sys.path:
        sys.path.append(src)
    import integrator, planner, quad, track  # noqa: E401
    return quad, track, planner, integrator


def rpg_objects(quad_path: str | Path, track_path: str | Path):
    quad_mod, track_mod, _, _ = _import_rpg()
    with contextlib.redirect_stdout(io.StringIO()):
        return quad_mod.Quad(str(quad_path)), track_mod.Track(str(track_path))


def make_planner(quad_path, track_path, nodes_per_gate: int, tolerance: float, vel_guess: float,
                 ipopt_options: dict | None = None):
    _, _, planner_mod, integrator_mod = _import_rpg()
    quad, track = rpg_objects(quad_path, track_path)
    options = {"tolerance": tolerance, "nodes_per_gate": nodes_per_gate, "vel_guess": vel_guess}
    if ipopt_options is not None:
        options["solver_options"] = {"ipopt": ipopt_options, "print_time": False}
    with contextlib.redirect_stdout(io.StringIO()):
        planner = planner_mod.Planner(quad, track, integrator_mod.RungeKutta4, options)
        planner.setup()
    return planner


def solve_cpc(planner) -> dict:
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        x = planner.solve()
    wall = time.perf_counter() - t0
    stats = planner.solver.stats()
    return {"x": np.asarray(x), "t": float(x[0]), "wall_s": wall, "iter": int(stats.get("iter_count", -1)),
            "status": stats.get("return_status", "?"), "success": bool(stats.get("success", False))}


def to_cpc_vector(problem, xs: np.ndarray, us: np.ndarray) -> np.ndarray:
    L = problem.layout
    out = [np.array([us[0, L.u_t]]), xs[0, :NQ], xs[0, L.x_mu]]
    for i in range(problem.N):
        out += [us[i, L.u_thrust], xs[i + 1, :NQ], us[i, L.u_lam], us[i, L.u_tau], xs[i + 1, L.x_mu]]
    return np.concatenate(out)


def from_cpc_vector(problem, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FilterDDP (xs, us) of a CPC decision vector; ``t_prev = t`` after stage 0."""
    L, N, nw = problem.layout, problem.N, problem.layout.nw
    stride, start = NT + NQ + 3*nw, 1 + NQ + nw
    xs, us = np.zeros((N + 1, L.nx)), np.zeros((N, L.nu))
    xs[0, :NQ] = z[1:1 + NQ]
    xs[0, L.x_mu] = z[1 + NQ:start]
    xs[0, L.x_t] = problem.x0[L.x_t]
    xs[1:, L.x_t] = z[0]
    us[:, L.u_t] = z[0]
    for i in range(N):
        b = start + i*stride
        us[i, L.u_thrust] = z[b:b + NT]
        xs[i + 1, :NQ] = us[i, L.u_next] = z[b + NT:b + NT + NQ]
        us[i, L.u_lam] = z[b + NT + NQ:b + NT + NQ + nw]
        us[i, L.u_tau] = z[b + NT + NQ + nw:b + NT + NQ + 2*nw]
        xs[i + 1, L.x_mu] = z[b + NT + NQ + 2*nw:b + stride]
    return xs, us


def cpc_guess_controls(problem, planner) -> np.ndarray:
    """``planner.py``'s initial guess (what IPOPT starts from) as FilterDDP controls.

    FilterDDP rolls ``mu`` out from ``lambda``; CPC's own ``mu`` guess is not used.
    """
    return from_cpc_vector(problem, np.asarray(planner.xg).ravel())[1]


def cpc_violation(planner, z: np.ndarray) -> dict:
    """Max violation of the CPC constraints ``lb <= g(z) <= ub`` and the worst row."""
    g = ca.Function("g", [planner.x], [planner.g])
    gz = np.asarray(g(z)).ravel()
    lb, ub = np.asarray(planner.lb).ravel(), np.asarray(planner.ub).ravel()
    viol = np.maximum(lb - gz, gz - ub).clip(min=0.0)
    j = int(np.argmax(viol))
    return {"max": float(viol[j]), "row": j, "objective": float(z[0]), "n_rows": int(gz.size)}

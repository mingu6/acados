"""Quadrotor model and track data of the CPC time-optimal waypoint-flight problem.

Reference: P. Foehn, A. Romero, D. Scaramuzza, "Time-Optimal Planning for Quadrotor
Waypoint Flight", Science Robotics 6(56), 2021, doi:10.1126/scirobotics.abh1221, and its
example code https://github.com/uzh-rpg/rpg_time_optimal (revision f5541b6,
``src/quad.py``, ``src/integrator.py``, ``src/track.py``).

State ``[p (3), v (3), q (4, Hamilton w-x-y-z, body->world), omega (3)]`` and
control ``T (4)`` are single-rotor thrusts in N. The dynamics are
``quad.py:Quad.dynamics`` verbatim:

    p' = v
    v' = R(q) [0, 0, sum(T)/m] + [0, 0, -g] - cd v
    q' = 0.5 q (x) [0, omega]
    omega' = J^{-1} ([l(T0-T1-T2+T3), l(-T0-T1+T2+T3), ctau(T0-T1+T2-T3)] - omega x J omega)

and one RK4 step of length ``dt`` is ``integrator.py:RungeKutta4``.

PyYAML is not available in the example environment, so ``load_yaml`` parses
the flat subset of YAML these files use (scalars, inline lists that may span
lines, and one level of nested mappings).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

NQ = 13  # quadrotor state dimension
NT = 4   # rotor thrusts

RPG_URL = "https://github.com/uzh-rpg/rpg_time_optimal"
RPG_REVISION = "f5541b6d9d3dee563e01a64ec1994b8ed6c0076c"
HERE = Path(__file__).resolve().parent


def rpg_dir() -> Path:
    """rpg_time_optimal checkout (quad file and IPOPT reference).

    ``RPG_TIME_OPTIMAL_DIR`` if set, else ``data/rpg_time_optimal`` as cloned by reproduce.sh.
    """
    path = Path(os.environ.get("RPG_TIME_OPTIMAL_DIR", HERE / "data" / "rpg_time_optimal"))
    if not (path / "src" / "planner.py").exists():
        raise RuntimeError(f"no rpg_time_optimal checkout at {path}: run reproduce.sh, or set "
                           f"RPG_TIME_OPTIMAL_DIR to a clone of {RPG_URL} at {RPG_REVISION[:7]}")
    return path


def load_yaml(path: str | Path) -> dict:
    """Parse the YAML subset used by rpg_time_optimal quad and track files."""
    lines = []
    for raw in Path(path).read_text().splitlines():
        text = raw.split("#", 1)[0].rstrip()
        if not text.strip():
            continue
        # A value continues on the next line while its brackets are open.
        if lines and lines[-1][1].count("[") > lines[-1][1].count("]"):
            lines[-1] = (lines[-1][0], lines[-1][1] + " " + text.strip())
            continue
        indent = len(text) - len(text.lstrip())
        lines.append((indent, text.strip()))

    root: dict = {}
    stack = [(-1, root)]
    for indent, text in lines:
        key, _, value = text.partition(":")
        key, value = key.strip(), value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _yaml_scalar(value)
    return root


def _yaml_scalar(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


@dataclass
class QuadParameters:
    """Physical parameters with the derivations of ``quad.py:Quad.load``."""

    mass: float = 1.0
    arm_length: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    thrust_max: float = 5.0
    thrust_min: float = 0.0
    omega_max_xy: float = 3.0
    omega_max_z: float = 3.0
    torque_coeff: float = 0.5
    drag_coeff: float = 0.0
    rampup_dist: float = 0.0
    thrust_ramp_start: float = 5.0
    omega_ramp_start: float = 3.0
    gravity: float = 9.801  # quad.py uses 9.801 in the dynamics and 9.81 in TWR conversions

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuadParameters":
        d = load_yaml(path)
        p = cls()
        p.mass = float(d["mass"])
        p.arm_length = float(d["arm_length"])
        p.inertia = np.array(d["inertia"], dtype=float)
        if "TWR_max" in d:
            p.thrust_max = d["TWR_max"] * 9.81 * p.mass / 4
        else:
            p.thrust_max = float(d["thrust_max"])
        if "TWR_min" in d:
            p.thrust_min = d["TWR_min"] * 9.81 * p.mass / 4
        else:
            p.thrust_min = float(d["thrust_min"])
        p.omega_max_xy = float(d["omega_max_xy"])
        p.omega_max_z = float(d["omega_max_z"])
        p.torque_coeff = float(d["torque_coeff"])
        if "v_max" in d:
            a_max = 4 * p.thrust_max / p.mass
            p.drag_coeff = float(np.sqrt(a_max**2 - p.gravity**2) / d["v_max"])
        if "drag_coeff" in d:
            p.drag_coeff = float(d["drag_coeff"])
        if "rampup_dist" in d:
            p.rampup_dist = float(d["rampup_dist"])
            if "TWR_ramp_start" in d and "omega_ramp_start" in d:
                p.thrust_ramp_start = min(d["TWR_ramp_start"] * 9.81 * p.mass / 4, p.thrust_max)
                p.omega_ramp_start = min(d["omega_ramp_start"], p.omega_max_xy)
            # quad.py assigns a local ``rampup_dist = 0`` here, which leaves the ramp enabled
            # with the default start values; keep that behavior.
        return p

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity / 4


@dataclass
class Track:
    """Waypoint track with the fields of ``track.py:Track``."""

    gates: list = field(default_factory=list)
    init_pos: list | None = None
    init_att: list | None = None
    init_vel: list | None = None
    init_omega: list | None = None
    end_pos: list | None = None
    end_att: list | None = None
    end_vel: list | None = None
    end_omega: list | None = None
    ring: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Track":
        d = load_yaml(path)
        t = cls(gates=d.get("gates", []))
        initial = d.get("initial", d)
        t.init_pos = initial.get("position")
        t.init_att = initial.get("attitude")
        t.init_vel = initial.get("velocity")
        t.init_omega = initial.get("omega")
        end = d.get("end", {})
        t.end_pos = end.get("position")
        t.end_att = end.get("attitude")
        t.end_vel = end.get("velocity")
        t.end_omega = end.get("omega")
        t.ring = bool(d.get("ring", False))
        return t

    def waypoints(self) -> np.ndarray:
        """Waypoints as a (3, NW) array: gates, then the end position (``planner.py``)."""
        wp = [list(g) for g in self.gates]
        if self.end_pos is not None:
            wp.append(list(self.end_pos))
        return np.array(wp, dtype=float).T


def resolve_track(name: str) -> Path:
    """A track name from ``tracks/`` of this example (``three_gates``, ``lap``) or a path."""
    local = HERE / "tracks" / f"{name}.yaml"
    return local if local.exists() else Path(name)


def default_quad_path() -> Path:
    return rpg_dir() / "quads" / "quad.yaml"


def quat_mult(a, b):
    """Hamilton product a (x) b for w-x-y-z quaternions."""
    return ca.vertcat(
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0])


def rotate(q, v):
    """q (x) [0, v] (x) conj(q); equals R(q) v for unit q, as in ``quaternion.py:rotate_quat``."""
    qc = ca.vertcat(q[0], -q[1], -q[2], -q[3])
    r = quat_mult(quat_mult(q, ca.vertcat(0, v)), qc)
    return r[1:4]


def continuous_dynamics(params: QuadParameters, x, thrust):
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    J = ca.DM(params.inertia)
    J_inv = ca.DM(np.linalg.inv(params.inertia))
    l, ctau = params.arm_length, params.torque_coeff
    T = thrust
    torque = ca.vertcat(l*(T[0] - T[1] - T[2] + T[3]),
                        l*(-T[0] - T[1] + T[2] + T[3]),
                        ctau*(T[0] - T[1] + T[2] - T[3]))
    acc = rotate(q, ca.vertcat(0, 0, (T[0] + T[1] + T[2] + T[3])/params.mass)) \
        + ca.DM([0, 0, -params.gravity]) - v*params.drag_coeff
    return ca.vertcat(v, acc, 0.5*quat_mult(q, ca.vertcat(0, w)),
                      ca.mtimes(J_inv, torque - ca.cross(w, ca.mtimes(J, w))))


def rk4_step(params: QuadParameters, x, thrust, dt):
    k1 = continuous_dynamics(params, x, thrust)
    k2 = continuous_dynamics(params, x + dt/2*k1, thrust)
    k3 = continuous_dynamics(params, x + dt/2*k2, thrust)
    k4 = continuous_dynamics(params, x + dt*k3, thrust)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def rk4_function(params: QuadParameters) -> ca.Function:
    x = ca.SX.sym("x", NQ)
    T = ca.SX.sym("T", NT)
    dt = ca.SX.sym("dt")
    return ca.Function("rk4", [x, T, dt], [rk4_step(params, x, T, dt)], ["x", "u", "dt"], ["xn"])

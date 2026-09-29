#!/usr/bin/env python3
"""Posa-style figures and animation of the Spring Flamingo solution.

Follows the layout of Posa, Cantu, Tedrake (2013): Fig. 5(a) filmstrip of
stick figures with the fan-shaped torso, Fig. 5(b) centre-of-mass height
against normalized time, and Fig. 6 per-foot contact-mode sequence, with the
initial guess overlaid as "Initial Sequence".  The robot drawing follows
ContactImplicitMPC.jl's ``flamingo/visuals.jl``: thin black links, joint dots,
orange toe/heel contact spheres, and traced torso and toe paths.  Kinematics
come from the Pinocchio model.

The half-stride solution is unrolled to a full stride with its mirror
symmetry.  Modes are those of ``gait_metrics`` (stance, toe only, heel only,
swing); Posa's "plantarflexion" and "push off" are both toe-only phases.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.patches import Wedge

from flamingo_model import CONTACT_NAMES, FlamingoParameters, build_model
from gait_metrics import contact_modes

LINK_COLOR = "#111111"
FAN_EDGE = "#8a8a8a"
FAN_FILL = "#d9d9d9"
CONTACT_COLOR = "#ffa500"
TORSO_TRACE = "#e07b00"
TOE_TRACE = "#1f5fbf"
GROUND = "#000000"
OPT_COLOR = "#1f3fbf"
REF_COLOR = "#d62728"
FAN_HALF_ANGLE = np.deg2rad(38.0)
MODES = ("swing", "toe only\n(push-off)", "heel only\n(heel strike)", "stance")


class Trajectory:
    def __init__(self, times, q, normal):
        self.times = np.asarray(times)
        self.q = np.asarray(q)
        self.normal = np.asarray(normal)  # (T, 4) normal impulses, NaN on the last knot

    @classmethod
    def from_csv(cls, path: Path) -> "Trajectory":
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        times = [float(r["time"]) for r in rows]
        q = [[float(r[f"q{i}"]) for i in range(9)] for r in rows]
        normal = [[float(r[f"{c}_pn"]) if r[f"{c}_pn"] else np.nan for c in CONTACT_NAMES]
                  for r in rows]
        return cls(times, q, normal)

    @property
    def h(self) -> float:
        return float(self.times[1] - self.times[0])

    def unrolled_half_stride(self) -> "Trajectory":
        """Full stride from a mirror-periodic half stride: knot N + j is the
        mirror of knot j advanced by the half-stride step, with the feet's
        impulses swapped."""
        from flamingo_model import mirror_configuration
        N = self.q.shape[0] - 1
        step = self.q[N, 0] - self.q[0, 0]
        second = np.array([mirror_configuration(q, step) for q in self.q[1:]])
        normal_second = self.normal[:N][:, [2, 3, 0, 1]]
        q = np.vstack([self.q, second])
        normal = np.vstack([self.normal[:N], normal_second, np.full((1, 4), np.nan)])
        return Trajectory(self.h * np.arange(q.shape[0]), q, normal)


class Kinematics:
    def __init__(self):
        self.params = FlamingoParameters()
        self.model, self.frames = build_model(self.params)
        self.data = self.model.createData()
        self.joint = {name: self.model.getJointId(name) for name in
                      ("pitch", "knee1", "ankle1", "knee2", "ankle2")}

    def points(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        xz = lambda v: np.array([v[0], v[2]])  # noqa: E731
        hip = xz(self.data.oMi[self.joint["pitch"]].translation)
        up = xz(self.data.oMi[self.joint["pitch"]].rotation @ np.array([0.0, 0.0, 1.0]))
        legs = [(xz(self.data.oMi[self.joint[f"knee{leg}"]].translation),
                 xz(self.data.oMi[self.joint[f"ankle{leg}"]].translation)) for leg in (1, 2)]
        contacts = np.array([xz(self.data.oMf[f].translation) for f in self.frames])
        return hip, up, legs, contacts

    def com(self, q):
        c = pin.centerOfMass(self.model, self.data, q)
        return np.array([c[0], c[2]])


def draw_robot(ax, kin: Kinematics, q, alpha=1.0, fan=True, zorder=2):
    hip, up, legs, contacts = kin.points(q)
    artists = []
    if fan:
        center = np.degrees(np.arctan2(up[1], up[0]))
        wedge = Wedge(hip, kin.params.l_torso, center - np.degrees(FAN_HALF_ANGLE),
                      center + np.degrees(FAN_HALF_ANGLE), facecolor=FAN_FILL,
                      edgecolor=FAN_EDGE, linewidth=0.6, alpha=alpha, zorder=zorder)
        ax.add_patch(wedge)
        artists.append(wedge)
        # Posa draws the torso as a fan of spokes.
        for angle in np.linspace(-FAN_HALF_ANGLE, FAN_HALF_ANGLE, 9):
            c, s = np.cos(angle), np.sin(angle)
            direction = np.array([c * up[0] - s * up[1], s * up[0] + c * up[1]])
            tip = hip + kin.params.l_torso * direction
            artists += ax.plot([hip[0], tip[0]], [hip[1], tip[1]], color=FAN_EDGE,
                               linewidth=0.4, alpha=alpha, zorder=zorder)
    for knee, ankle in legs:
        artists += ax.plot([hip[0], knee[0], ankle[0]], [hip[1], knee[1], ankle[1]],
                           color=LINK_COLOR, linewidth=1.4, alpha=alpha, zorder=zorder + 1,
                           solid_capstyle="round")
        artists += ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]], "o", color=LINK_COLOR,
                           markersize=2.5, alpha=alpha, zorder=zorder + 1)
    for leg in (0, 1):
        toe, heel = contacts[2 * leg], contacts[2 * leg + 1]
        artists += ax.plot([heel[0], toe[0]], [heel[1], toe[1]], color=LINK_COLOR,
                           linewidth=1.4, alpha=alpha, zorder=zorder + 1)
    artists += ax.plot(contacts[:, 0], contacts[:, 1], "o", color=CONTACT_COLOR,
                       markersize=3.5, markeredgecolor="k", markeredgewidth=0.4,
                       alpha=alpha, zorder=zorder + 2)
    artists += ax.plot([hip[0]], [hip[1]], "o", color=LINK_COLOR, markersize=3,
                       alpha=alpha, zorder=zorder + 2)
    return artists


def filmstrip(ax, kin, traj, count, spacing):
    frames = np.linspace(0, traj.q.shape[0] - 1, count).round().astype(int)
    x_min, x_max = np.inf, -np.inf
    for index, frame in enumerate(frames):
        q = traj.q[frame].copy()
        q[0] += index * spacing  # spread the figures as in Posa Fig. 5(a)
        draw_robot(ax, kin, q)
        x_min, x_max = min(x_min, q[0]), max(x_max, q[0])
    ax.axhline(0.0, color=GROUND, linewidth=1.2)
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(-0.03, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def com_plot(ax, kin, trajectories):
    for label, traj, color, style in trajectories:
        com = np.array([kin.com(q) for q in traj.q])
        normalized = (traj.times - traj.times[0]) / (traj.times[-1] - traj.times[0])
        ax.plot(normalized, com[:, 1], style, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("CM Height [m]")
    ax.set_title("Spring Flamingo CM Height")
    ax.legend(loc="lower center", fontsize=8)
    ax.grid(alpha=0.3)


def mode_plot(axes, kin, trajectories):
    weight = kin.params.total_mass * kin.params.gravity
    for foot, ax in enumerate(axes):
        for label, traj, color, style in trajectories:
            modes = contact_modes(traj.normal[:-1], traj.h, weight)[:, foot]
            t = traj.times[:-1]
            ax.step(np.append(t, traj.times[-1]), np.append(modes, modes[-1]), style,
                    where="post", color=color, label=label, linewidth=2)
        ax.set_yticks(range(len(MODES)))
        ax.set_yticklabels(MODES, fontsize=7)
        ax.set_ylim(-0.4, len(MODES) - 0.6)
        ax.set_title(f"Foot {foot + 1} Mode", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=2, fontsize=8,
                   frameon=False)
    axes[-1].set_xlabel("Time [s]")


def animate(path, kin, traj, fps, trace=True):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = traj.q[:, 0]
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    ax.set_ylim(-0.03, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.axhline(0.0, color=GROUND, linewidth=2)
    points = [kin.points(q) for q in traj.q]
    torso_com = np.array([p[0] + kin.params.d_torso * p[1] for p in points])
    toes = np.array([p[3][[0, 2]] for p in points])
    title = ax.set_title("")
    dynamic = []

    def update(frame):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()
        if trace:
            # Above the opaque torso fan (zorder 2), below links and contacts.
            dynamic.extend(ax.plot(torso_com[:frame + 1, 0], torso_com[:frame + 1, 1],
                                   color=TORSO_TRACE, linewidth=1.2, zorder=2.5))
            for leg in (0, 1):
                dynamic.extend(ax.plot(toes[:frame + 1, leg, 0], toes[:frame + 1, leg, 1],
                                       color=TOE_TRACE, linewidth=1.0, zorder=2.5))
        dynamic.extend(draw_robot(ax, kin, traj.q[frame]))
        title.set_text(f"t = {traj.times[frame]:.3f} s")
        return dynamic + [title]

    movie = animation.FuncAnimation(fig, update, frames=traj.q.shape[0],
                                    interval=1000.0 / fps, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trajectory", type=Path, help="trajectory.csv from solve_flamingo.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="output prefix; writes <prefix>.png and <prefix>.gif")
    parser.add_argument("--initial", type=Path, default=None,
                        help="initial_trajectory.csv to overlay as 'Initial Sequence'")
    parser.add_argument("--label", default="Optimized Sequence")
    parser.add_argument("--title", default="Spring Flamingo, one stride")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.55,
                        help="extra horizontal spacing between filmstrip figures [m]")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    kin = Kinematics()
    traj = Trajectory.from_csv(args.trajectory).unrolled_half_stride()
    series = [(args.label, traj, OPT_COLOR, "-")]
    if args.initial is not None:
        initial = Trajectory.from_csv(args.initial).unrolled_half_stride()
        series.append(("Initial Sequence", initial, REF_COLOR, "--"))

    fig = plt.figure(figsize=(11, 10))
    grid = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.3, 1, 1], hspace=0.55)
    filmstrip(fig.add_subplot(grid[0]), kin, traj, args.frames, args.spacing)
    com_plot(fig.add_subplot(grid[1]), kin, series)
    mode_axes = [fig.add_subplot(grid[2])]
    mode_axes.append(fig.add_subplot(grid[3], sharex=mode_axes[0]))
    mode_plot(mode_axes, kin, series)
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_name(args.output.name + ".png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    if not args.no_gif:
        gif = args.output.with_name(args.output.name + ".gif")
        animate(gif, kin, traj, args.fps)
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()

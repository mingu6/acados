"""Plot FilterDDP (and, if present, IPOPT) trajectories written by solve_waypoint_flight.py.

One column per track: the top-down path coloured by speed, with the thrust direction every
``--arrow-nth`` nodes and the waypoints with their tolerance, as in rpg_time_optimal's
``plotPos(plot_axis='xyq')``; below it, the speed over time.

    python plot_trajectories.py three_gates lap --output output/trajectories.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from quad_model import Track, resolve_track  # noqa: E402


def load(path: Path) -> dict:
    d = np.genfromtxt(path, delimiter=",", names=True)
    q = np.stack([d["qw"], d["qx"], d["qy"], d["qz"]])
    body_z = np.stack([2*(q[1]*q[3] + q[0]*q[2]), 2*(q[2]*q[3] - q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)])
    v = np.stack([d["vx"], d["vy"], d["vz"]])
    return {"t": d["t"], "p": np.stack([d["px"], d["py"], d["pz"]]), "speed": np.linalg.norm(v, axis=0),
            "z": body_z}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracks", nargs="+", help="track names; data are read from output/<track>/")
    p.add_argument("--output", default="output/trajectories.png")
    p.add_argument("--arrow-nth", type=int, default=6)
    p.add_argument("--tolerance", type=float, default=0.3)
    args = p.parse_args()

    runs = {name: {"fddp": load(Path("output")/name/"trajectory.csv")} for name in args.tracks}
    for name in args.tracks:
        ipopt = Path("output")/name/"ipopt_trajectory.csv"
        if ipopt.exists():
            runs[name]["ipopt"] = load(ipopt)
    vmax = max(r["fddp"]["speed"].max() for r in runs.values())

    fig, axes = plt.subplots(2, len(args.tracks), figsize=(5*len(args.tracks), 7), layout="constrained",
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for col, name in enumerate(args.tracks):
        f, ax, axv = runs[name]["fddp"], axes[0, col], axes[1, col]
        wp = Track.from_yaml(resolve_track(name)).waypoints()
        pts = f["p"][:2].T.reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(0, vmax), linewidth=2, zorder=2)
        lc.set_array(0.5*(f["speed"][:-1] + f["speed"][1:]))
        ax.add_collection(lc)
        k = np.arange(0, f["t"].size, args.arrow_nth)
        ax.quiver(f["p"][0, k], f["p"][1, k], f["z"][0, k], f["z"][1, k], color="k", width=0.004,
                  scale=1.0, scale_units="xy", angles="xy", zorder=3)
        if "ipopt" in runs[name]:
            ax.plot(*runs[name]["ipopt"]["p"][:2], "--", color="tab:red", lw=0.8, label="IPOPT", zorder=4)
        labels: dict = {}  # waypoints sharing (x, y) get one label, e.g. "3,4"
        for j in range(wp.shape[1]):
            labels.setdefault(tuple(np.round(wp[:2, j], 3)), []).append(str(j))
            ax.add_patch(plt.Circle(wp[:2, j], args.tolerance, fill=False, color="tab:orange", lw=1.5))
        for xy, names in labels.items():
            ax.annotate(",".join(names), xy, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.plot(*f["p"][:2, 0], "ko", ms=4)
        ax.set_title(f"{name}: FilterDDP t = {f['t'][-1]:.4f} s")
        ax.set_xlabel("$p_x$ [m]")
        ax.set_ylabel("$p_y$ [m]")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.3)
        if "ipopt" in runs[name]:
            ax.legend(loc="best", fontsize=8)

        axv.plot(f["t"], f["speed"], label="FilterDDP")
        if "ipopt" in runs[name]:
            r = runs[name]["ipopt"]
            axv.plot(r["t"], r["speed"], "--", color="tab:red", lw=1, label=f"IPOPT t = {r['t'][-1]:.4f} s")
        axv.set_xlabel("t [s]")
        axv.set_ylabel("|v| [m/s]")
        axv.grid(True, alpha=0.3)
        axv.legend(loc="best", fontsize=8)
    fig.colorbar(lc, ax=axes, label="speed [m/s]", shrink=0.6)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Gait diagnostics: swing clearance and contact modes compared with Posa Fig. 6.

Contact mode per interval and foot, from which points carry more than
``LOAD_THRESHOLD`` of body weight: S stance (toe and heel), T toe only,
H heel only, W swing.  A mirror-periodic half stride is one foot's full cycle
when foot 1's half is followed by foot 2's half.

Posa et al. (2013) Fig. 6, read at full resolution (1.88 s cycle): right foot
S 0-0.56 s, T (push off) 0.56-0.81, W 0.81-1.48, T (plantarflexion)
1.48-1.53, S 1.53-1.88; left foot W 0-0.51, T 0.51-0.56, S 0.56-1.53, T
1.53-1.78, W 1.78-1.88.  Per foot: S 0.50, T 0.16, H 0, W 0.34; double
support 0.32.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CONTACTS = ("toe1", "heel1", "toe2", "heel2")
LETTERS = "WTHS"  # mode index 0 swing, 1 toe only, 2 heel only, 3 stance
LOAD_THRESHOLD = 0.01
MOVING_SPEED = 0.05  # m/s; a foot moving faster than this counts as swinging
POSA_FRACTIONS = np.array([0.34, 0.16, 0.00, 0.50])  # W, T, H, S
POSA_DOUBLE_SUPPORT = 0.32


def load_csv(path: Path) -> dict:
    rows = list(csv.DictReader(Path(path).open()))
    t = np.array([float(r["time"]) for r in rows])
    points = np.array([[[float(r[f"{c}_x"]), float(r[f"{c}_z"])] for c in CONTACTS]
                       for r in rows])
    normal = np.array([[float(r[f"{c}_pn"]) for c in CONTACTS] for r in rows[:-1]])
    return {"time": t, "h": t[1] - t[0], "points": points, "normal": normal}


def contact_modes(normal: np.ndarray, h: float, weight: float) -> np.ndarray:
    """(T, 2) mode index per interval and foot."""
    loaded = normal / h > LOAD_THRESHOLD * weight
    modes = np.zeros((normal.shape[0], 2), dtype=int)
    for foot in (0, 1):
        toe, heel = loaded[:, 2 * foot], loaded[:, 2 * foot + 1]
        modes[:, foot] = np.select([toe & heel, heel, toe], [3, 2, 1], default=0)
    return modes


def swing_clearance(points: np.ndarray, h: float) -> list[dict]:
    """Per foot: apex of the lower of toe and heel, and intervals moving below 1 cm."""
    result = []
    for foot in (0, 1):
        pts = points[:, 2 * foot:2 * foot + 2, :]
        height = pts[:, :, 1].min(axis=1)
        speed = np.abs(np.diff(pts[:, :, 0], axis=0)).max(axis=1) / h
        moving = speed > MOVING_SPEED
        low = np.minimum(height[:-1], height[1:]) < 0.01
        result.append({"apex": float(height.max()), "moving_below_1cm": int((moving & low).sum()),
                       "peak_speed": float(speed.max())})
    return result


def cyclic_runs(sequence: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encoding of a cyclic sequence, rotated to start at the longest stance."""
    runs: list[list[int]] = []
    for mode in sequence:
        if runs and runs[-1][0] == mode:
            runs[-1][1] += 1
        else:
            runs.append([int(mode), 1])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1] += runs.pop()[1]
    stance = [i for i, (mode, _) in enumerate(runs) if mode == 3]
    if stance:
        longest = max(stance, key=lambda i: runs[i][1])
        runs = runs[longest:] + runs[:longest]
    return [(mode, count) for mode, count in runs]


def mode_summary(modes: np.ndarray, half_stride: bool) -> dict:
    cycles = ([np.concatenate([modes[:, 0], modes[:, 1]]),
               np.concatenate([modes[:, 1], modes[:, 0]])] if half_stride
              else [modes[:, 0], modes[:, 1]])
    fractions = np.mean([[np.mean(c == m) for m in range(4)] for c in cycles], axis=0)
    double = float(np.mean((modes[:, 0] > 0) & (modes[:, 1] > 0)))
    runs = cyclic_runs(cycles[0])
    return {
        "fractions": {"stance": fractions[3], "toe_only": fractions[1],
                      "heel_only": fractions[2], "swing": fractions[0]},
        "double_support": double,
        "pattern": "-".join(f"{LETTERS[m]}{n}" for m, n in runs),
        "distance_to_posa": float(np.abs(fractions - POSA_FRACTIONS).sum()
                                  + abs(double - POSA_DOUBLE_SUPPORT)),
    }


def gait_metrics(path: Path, weight: float, half_stride: bool = True) -> dict:
    data = load_csv(path)
    modes = contact_modes(data["normal"], data["h"], weight)
    return {"clearance": swing_clearance(data["points"], data["h"]),
            "modes": mode_summary(modes, half_stride)}


if __name__ == "__main__":
    from flamingo_model import FlamingoParameters
    params = FlamingoParameters()
    for argument in sys.argv[1:]:
        m = gait_metrics(Path(argument), params.total_mass * params.gravity)
        f = m["modes"]["fractions"]
        print(f"{argument}: stance {f['stance']:.2f}, toe only {f['toe_only']:.2f}, "
              f"heel only {f['heel_only']:.2f}, swing {f['swing']:.2f}, "
              f"double support {m['modes']['double_support']:.2f}, "
              f"pattern {m['modes']['pattern']}, distance to Posa "
              f"{m['modes']['distance_to_posa']:.2f}, swing apex "
              f"{100 * m['clearance'][0]['apex']:.1f} cm")

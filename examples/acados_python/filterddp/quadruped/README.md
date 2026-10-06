# Quadruped NMPC in MuJoCo

The nominal centroidal NMPC of [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)
solved with FILTERDDP, in closed loop in MuJoCo. A Unitree Go2 trots forward at 0.5 m/s on flat
ground, optionally pushed by a random wrench on its base.

The OCP is Quadruped-PyMPC's own:

- a single rigid body with point feet;
- a LINEAR_LS tracking cost, ERK dynamics and a friction cone;
- 12 stages of 20 ms.

Only its solver options change. Quadruped-PyMPC runs one SQP iteration per MPC call. Here FILTERDDP
runs 5 iterations with the Gauss-Newton Hessian, warm started from the shifted policy of the previous
solve, rolled out in closed loop from the measured state (`AcadosOcpSolver.warm_start_from_policy`).
The MPC runs at 100 Hz and the physics at 500 Hz. Everything else,
including the whole-body layer that turns the plan into joint torques, is Quadruped-PyMPC's.

## Sources and revisions

| Source | Role | Revision |
| --- | --- | --- |
| [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC) | Controller, OCP, simulation loop | [`6dcbb35`](https://github.com/iit-DLSLab/Quadruped-PyMPC/tree/6dcbb35a3d20a29e9ab2fc70aa2f890e8f4d73a7) |
| [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped) | MuJoCo environment, Go2 model | [`f8c470d`](https://github.com/iit-DLSLab/gym-quadruped/tree/f8c470d0d685d815e59d385ebdc8fa79db8aa979) (1.1.6) |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Simulator | 3.14.0 |
| [Turrisi et al. 2024](https://arxiv.org/abs/2403.11383) | Push disturbances (`--wrench`) | IROS 2024 |
| [mingu6/acados, branch `filterddp`](https://github.com/mingu6/acados/tree/filterddp) | FILTERDDP NLP solver in acados | [`da3971695`](https://github.com/mingu6/acados/commit/da3971695) |

## Run

```bash
conda env create -f environment.yml && conda activate quadruped-filterddp
pip install --no-deps -e ../../../../interfaces/acados_template
git clone https://github.com/iit-DLSLab/Quadruped-PyMPC data/Quadruped-PyMPC
git -C data/Quadruped-PyMPC checkout 6dcbb35a3d20a29e9ab2fc70aa2f890e8f4d73a7
pip install --no-deps -e data/Quadruped-PyMPC
export ACADOS_SOURCE_DIR=$(cd ../../../.. && pwd) LD_LIBRARY_PATH=$(cd ../../../.. && pwd)/lib
python quadruped_nmpc.py --wrench 7.5
```

This acados tree must be built and installed into itself (`lib/`, `include/`). Quadruped-PyMPC is
installed without its dependencies: its nominal NMPC needs only the packages in `environment.yml`.
The first run generates and compiles the solver into `c_generated_code/`.

The run prints the outcome, the velocity tracking error, and the iterations and solve times of the
MPC calls. Options:

| Option | Effect |
| --- | --- |
| `--render` | Open MuJoCo's viewer with Quadruped-PyMPC's overlays: swing trajectories, footholds, contact forces (needs a display; on macOS run with `mjpython`) |
| `--wrench A`, `--seed` | Push the base every 2 s from t = 1 s, each force and torque component uniform in [-A, A] (N, N m) |
| `--solver sqp` | Quadruped-PyMPC's own solver, for comparison |
| `--max_iter k`, `--no_warm_start` | FILTERDDP iteration cap and warm start |
| `--seconds`, `--speed` | Episode length (default 10 s) and forward velocity command (default 0.5 m/s) |

In the viewer, drag to rotate the camera and scroll to zoom. To push the robot yourself, double-click
the body to select it and Ctrl + right-drag.

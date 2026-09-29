# Quadruped NMPC under random pushes

FILTERDDP and acados SQP as the solver of the nominal centroidal NMPC of
[Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC), in closed loop in MuJoCo: a Go2
trots forward at 0.5 m/s while a random wrench on its base is redrawn every 2 s, with each force and
torque component uniform in [-A, A], at A = 5 to 15 N / N m. This follows the disturbance test of
Turrisi et al., [IROS 2024](https://arxiv.org/abs/2403.11383). Only the solver is swapped; the rest of
the controller is Quadruped-PyMPC's.

## Setup

```bash
conda env create -f environment.yml && conda activate quadruped-filterddp
pip install --no-deps -e ../../../../interfaces/acados_template
git clone https://github.com/iit-DLSLab/Quadruped-PyMPC data/Quadruped-PyMPC
git -C data/Quadruped-PyMPC checkout 6dcbb35a3d20a29e9ab2fc70aa2f890e8f4d73a7
pip install --no-deps -e data/Quadruped-PyMPC
export ACADOS_SOURCE_DIR=$(cd ../../../.. && pwd) LD_LIBRARY_PATH=$(cd ../../../.. && pwd)/lib
```

This acados tree must be built and installed into itself (`lib/`, `include/`). Quadruped-PyMPC is
installed without its dependencies: its nominal NMPC needs only the packages in `environment.yml`, not
its bundled acados, jax or pinocchio. The first run downloads the Go2 model through gym-quadruped.

## Run

Build the OCP and solve one recorded MPC call (`example_calls.npz`) with FILTERDDP_GN and SQP_GN; no
simulator needed:

```bash
python solve_quadruped_ocp.py
```

The closed-loop experiment (all levels, 25 episodes each, about 3 hours on 8 cores), its summary, a
figure and example videos, all written to `output/`:

```bash
./reproduce.sh          # LEVELS="5 7.5 10 12.5 15" EPISODES=25 WORKERS=8 VIDEOS=1
```

Or step by step:

```bash
python mujoco_closed_loop.py --controllers STOCK FILTERDDP_GN:15 FILTERDDP_GN:15:vg+ws SQP_GN:3 \
    --episodes 25 --wrench 10 --tag closed_loop_go2_trot_w10
python summarize_closed_loop.py "output/closed_loop_go2_trot_w*.txt" --pool 10 12.5 15
python plot_results.py "output/closed_loop_go2_trot_w*.txt" --output output/closed_loop.png
MUJOCO_GL=egl python visualize_episode.py --controllers STOCK FILTERDDP_GN:15 SQP_GN:3 \
    --wrench 7.5 --episode 21 --seconds 7 --out output/videos/w7.5_ep21.mp4
```

Videos need an OpenGL backend for MuJoCo: `MUJOCO_GL=egl` without a display, the default (glfw)
with one.

Controllers (`--controllers`); every capped solve returns its last iterate:

| spec | solver |
|---|---|
| `STOCK` | Quadruped-PyMPC's own solver: SQP, one Gauss-Newton iteration per call |
| `SQP_GN:<k>` | the same settings, at most k iterations |
| `FILTERDDP_GN:<k>` | FILTERDDP with the dynamics Hessian dropped, at most k iterations |
| `FILTERDDP_GN:<k>:vg+ws` | with value-gradient stationarity (`vg`) and the shifted-policy warm start (`ws`) |

An episode lasts 10 s (MPC at 100 Hz, physics at 500 Hz). It fails on gym-quadruped's termination,
roll or pitch above 1 rad, or base height below 0.12 m. Episodes are paired across controllers
(same seeds, same pushes).

## Files

- `quadruped_model.py`: centroidal model, cost and friction cone, with Quadruped-PyMPC's equations.
- `quadruped_ocp.py`: the acados OCP for FILTERDDP_GN and SQP_GN, and the FILTERDDP options.
- `solve_quadruped_ocp.py`, `example_calls.npz`: one OCP build and solve.
- `pympc_adapter.py`: stands in for Quadruped-PyMPC's `AcadosOcpSolver` object.
- `closed_loop_core.py`: one episode of Quadruped-PyMPC's simulation loop, with pushes and the fall test.
- `mujoco_closed_loop.py`: episodes in parallel, one line per episode in `output/<tag>.txt`.
- `summarize_closed_loop.py`, `plot_results.py`, `visualize_episode.py`: tables, figure, videos.

## Sources

| Source | Role | Revision |
|---|---|---|
| [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC) | controller, NMPC model and settings, simulation loop | [`6dcbb35`](https://github.com/iit-DLSLab/Quadruped-PyMPC/tree/6dcbb35a3d20a29e9ab2fc70aa2f890e8f4d73a7) |
| [gym-quadruped](https://pypi.org/project/gym-quadruped/) | MuJoCo environment, Go2 model | 1.1.6 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | simulator | 3.14.0 |
| [mingu6/acados, branch `filterddp-nmpc`](https://github.com/mingu6/acados/tree/filterddp-nmpc) | FILTERDDP NLP solver in acados | [`ecdfef0`](https://github.com/mingu6/acados/commit/ecdfef033) |
| [Turrisi et al. 2024](https://arxiv.org/abs/2403.11383) | disturbance protocol | IROS 2024 |

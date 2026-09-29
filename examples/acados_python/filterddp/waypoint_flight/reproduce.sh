#!/usr/bin/env bash
# Reproduce the results and the figure of README.md: FilterDDP and the IPOPT reference on the
# three-gate track and the one-lap track, then output/trajectories.png.
set -euo pipefail
cd "$(dirname "$0")"
# ACADOS_SOURCE_DIR may point at an upstream acados, which has no FILTERDDP;
# always use this checkout unless FILTERDDP_ACADOS_DIR says otherwise.
export ACADOS_SOURCE_DIR="${FILTERDDP_ACADOS_DIR:-$(cd ../../../.. && pwd)}"
if [[ ! -f "$ACADOS_SOURCE_DIR/acados/ocp_nlp/ocp_nlp_filterddp.c" ]]; then
  echo "ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR is not the FilterDDP fork" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-python}

# The CPC planner (IPOPT reference, quadrotor, initial guess) at the pinned revision.
RPG_REVISION=f5541b6d9d3dee563e01a64ec1994b8ed6c0076c
if [[ -z "${RPG_TIME_OPTIMAL_DIR:-}" ]]; then
  export RPG_TIME_OPTIMAL_DIR=data/rpg_time_optimal
  if [[ ! -d "$RPG_TIME_OPTIMAL_DIR" ]]; then
    git clone --quiet https://github.com/uzh-rpg/rpg_time_optimal "$RPG_TIME_OPTIMAL_DIR"
    git -C "$RPG_TIME_OPTIMAL_DIR" checkout --quiet "$RPG_REVISION"
  fi
fi

for track in three_gates lap; do
  "$PYTHON" solve_waypoint_flight.py --track "$track" --ipopt
done
"$PYTHON" plot_trajectories.py three_gates lap --output output/trajectories.png

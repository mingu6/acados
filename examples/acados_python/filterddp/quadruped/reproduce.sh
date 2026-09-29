#!/usr/bin/env bash
# Closed-loop runs over the push levels, their summary and figure, and example videos.
#   LEVELS="5 7.5 10 12.5 15" EPISODES=25 WORKERS=8 VIDEOS=1 OUTPUT_DIR=output ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
# ACADOS_SOURCE_DIR may point at an upstream acados, which has no FILTERDDP;
# always use this checkout unless FILTERDDP_ACADOS_DIR says otherwise.
export ACADOS_SOURCE_DIR="${FILTERDDP_ACADOS_DIR:-$(cd ../../../.. && pwd)}"
if [[ ! -f "$ACADOS_SOURCE_DIR/include/acados/ocp_nlp/ocp_nlp_filterddp.h" ]]; then
  echo "ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR is not an installed FilterDDP fork" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-python}
"$PYTHON" -c "import quadruped_pympc, gym_quadruped" 2>/dev/null || {
  echo "Quadruped-PyMPC or gym-quadruped is not installed, see README.md" >&2; exit 1; }

LEVELS=${LEVELS:-"5 7.5 10 12.5 15"}
EPISODES=${EPISODES:-25}
WORKERS=${WORKERS:-8}
OUT=${OUTPUT_DIR:-output}
CONTROLLERS="STOCK FILTERDDP_GN:12 FILTERDDP_GN:15 FILTERDDP_GN:15:vg FILTERDDP_GN:20 FILTERDDP_GN:300
             FILTERDDP_GN:5:vg+ws FILTERDDP_GN:15:vg+ws SQP_GN:3 SQP_GN:300"

for A in $LEVELS; do
  "$PYTHON" mujoco_closed_loop.py --controllers $CONTROLLERS --episodes "$EPISODES" --wrench "$A" \
      --workers "$WORKERS" --tag "closed_loop_go2_trot_w$A" --output_dir "$OUT"
done
"$PYTHON" summarize_closed_loop.py "$OUT/closed_loop_go2_trot_w*.txt" --pool 10 12.5 15 --out "$OUT/closed_loop_summary.md"
"$PYTHON" plot_results.py "$OUT/closed_loop_go2_trot_w*.txt" --output "$OUT/closed_loop.png"

if [[ "${VIDEOS:-1}" == 1 ]]; then
  PANELS="STOCK FILTERDDP_GN:15 FILTERDDP_GN:15:vg+ws SQP_GN:3"
  for run in "5 0 10" "7.5 21 7" "10 0 10" "15 0 10"; do
    set -- $run
    "$PYTHON" visualize_episode.py --controllers $PANELS --wrench "$1" --episode "$2" --seconds "$3" \
        --out "$OUT/videos/w$1_ep$2.mp4"
  done
fi

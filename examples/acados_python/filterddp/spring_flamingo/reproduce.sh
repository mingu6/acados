#!/usr/bin/env bash
# Reproduce the Spring Flamingo results of README.md:
# the solution with timings (one warmup plus three timed solves), its
# Posa-style figure and animation, and the spread of local solutions under
# perturbations of the pinned-knot gap.
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

"$PYTHON" solve_flamingo.py --output output/flamingo --repeat 4 --print-level 0
"$PYTHON" visualize_flamingo.py output/flamingo/trajectory.csv \
  --initial output/flamingo/initial_trajectory.csv --output output/flamingo/flamingo

for gap in 1.5e-6 2.5e-6 3e-6 4e-6; do
  "$PYTHON" solve_flamingo.py --output "output/spread/gap$gap" --pinned-gap "$gap" \
    --print-level 0 | sed "s/^/gap $gap: /"
done

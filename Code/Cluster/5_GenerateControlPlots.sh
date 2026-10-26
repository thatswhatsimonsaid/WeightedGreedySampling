#!/bin/bash
# Regenerate the manuscript's figures from the control sweep, relative to iGS.
#
# Mirrors 5_GenerateManuscriptPlots.sh, but writes to results_grfp_controls/
# instead of Results/original_run/, and produces every figure twice: once on
# full-pool (the metric the manuscript's quoted numbers match) and once on
# held-out test error (the metric its Evaluation Metrics section names).
#
# BUDGET controls the AUC integration window; the trace plots always use the
# full curve, and label efficiency keeps its own 70%/80% gain milestones.

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

BUDGET="${BUDGET:-0.70}"
OUT="../results_grfp_controls/figures"
mkdir -p "$OUT"

for METRIC in full_pool test; do
    SRC="../results_grfp_controls/full_suite/${METRIC}"
    if [ ! -f "${SRC}/exp_a_runs.csv" ]; then
        echo "--- skipping ${METRIC}: run 3_ProcessControlResults.sh first ---"
        continue
    fi
    echo ""
    echo "--- Figures for ${METRIC} (AUC over first ${BUDGET} of budget) ---"
    "${PYTHON:-python}" -m utils.GRFPControls.ManuscriptFigures \
        --dirs "$SRC" --metrics "$METRIC" --out "$OUT" --budget "$BUDGET"
done

echo ""
echo "--- Summary tables (relative to iGS) ---"
"${PYTHON:-python}" -m utils.GRFPControls.RelativeToIGS --ref iGS
"${PYTHON:-python}" -m utils.GRFPControls.WinLossTie
echo ""
echo "Figures written to $(realpath "$OUT")"

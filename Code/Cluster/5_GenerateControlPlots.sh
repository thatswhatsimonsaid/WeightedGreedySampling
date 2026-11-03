#!/bin/bash
# Regenerate the manuscript's figures from the control sweep.
#
# Uses the repository's own plotting code rather than reimplementing it:
# GeneratePlots.generate_all_plots for the per-dataset trace plots,
# GenerateAUCTable for the relative-AUC heatmaps, and
# GenerateLabellingComplexity for the N_rel boxplot. The control sweep is first
# exported into the aggregated layout those expect, so the figures come out
# with the repository's colours, linestyles, legend and axis labels, one file
# per dataset.
#
# Everything is produced for both metrics: full-pool, which the manuscript's
# quoted numbers match, and held-out test error, which its Evaluation Metrics
# section names.

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

SUITE="../results_grfp_controls/full_suite"
OUT="../results_grfp_controls/manuscript_figures"

echo "--- Exporting to the aggregated layout and generating figures ---"
"${PYTHON:-python}" -m utils.GRFPControls.ExportToAggregated \
    --suite "$SUITE" --out "$OUT"

echo ""
echo "--- Summary tables (relative AUC) ---"
for METRIC in full_pool test; do
    SRC="${SUITE}/${METRIC}"
    [ -f "${SRC}/exp_a_runs.csv" ] || continue
    echo ""
    echo "=============================================================="
    echo " Summary tables -- ${METRIC}"
    echo "=============================================================="
    "${PYTHON:-python}" -m utils.GRFPControls.RelativeToIGS \
        --dirs "$SRC" --ref iGS --label "${METRIC} (relative to iGS)"
    "${PYTHON:-python}" -m utils.GRFPControls.RelativeToIGS \
        --dirs "$SRC" --ref "WiGS (MAB-UCB1, c=2.0)" \
        --label "${METRIC} (relative to the bandit)"
done

echo ""
echo "Per-dataset trace plots:"
find "$OUT" -name "*TracePlot.png" | sed 's|.*/||' | sort | head -8
echo "  ... $(find "$OUT" -name '*TracePlot.png' | wc -l) trace plots total"
echo "Figures under $(realpath "$OUT")"

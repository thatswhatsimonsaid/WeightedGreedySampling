#!/bin/bash
# Aggregate the control sweep and generate every figure, as a batch job.
#
# Reading 36,000 task pickles takes long enough that an interactive run dies
# with the SSH session, and login nodes kill long processes regardless. This
# does the aggregation and the plotting in one job.
#
# Submit from the Code/ directory:
#     sbatch Cluster/4_ProcessAndPlotControls.sbatch
#
# Follow it with:
#     tail -f Cluster/RunControls/ClusterMessages/out/processplot_*.out

#SBATCH --job-name=CTRL_process
#SBATCH --partition=cpu-g2
#SBATCH --output=Cluster/RunControls/ClusterMessages/out/processplot_%j.out
#SBATCH --error=Cluster/RunControls/ClusterMessages/error/processplot_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
case "$CODE_DIR" in
  */Cluster) CODE_DIR="${CODE_DIR%/Cluster}" ;;
esac
cd "$CODE_DIR" || exit 1

export PYTHONDONTWRITEBYTECODE=1

# Activate the project env if present, so the job does not depend on whatever
# the submitting shell happened to have loaded.
if [ -z "${PYTHON:-}" ] && [ -f "../.WiGS_Env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ../.WiGS_Env/bin/activate
fi
PY="${PYTHON:-python}"
echo "python: $("$PY" -c 'import sys; print(sys.executable)')"

echo ""
echo "=== 1/2  Aggregating 36,000 task pickles ==="
"$PY" -m utils.GRFPControls.AggregateControls \
    --in ../results_grfp_controls/cluster \
    --out ../results_grfp_controls/full_suite || exit 1

echo ""
echo "=== 2/2  Generating figures and summary tables ==="
BUDGET="${BUDGET:-0.70}"
OUT="../results_grfp_controls/figures"
mkdir -p "$OUT"

for METRIC in full_pool test; do
    SRC="../results_grfp_controls/full_suite/${METRIC}"
    [ -f "${SRC}/exp_a_runs.csv" ] || { echo "skipping ${METRIC}"; continue; }
    "$PY" -m utils.GRFPControls.ManuscriptFigures \
        --dirs "$SRC" --metrics "$METRIC" --out "$OUT" --budget "$BUDGET"
done

for METRIC in full_pool test; do
    SRC="../results_grfp_controls/full_suite/${METRIC}"
    [ -f "${SRC}/exp_a_runs.csv" ] || continue
    echo ""
    echo "=============================================================="
    echo " Summary tables -- ${METRIC}"
    echo "=============================================================="
    "$PY" -m utils.GRFPControls.RelativeToIGS \
        --dirs "$SRC" --ref iGS --label "${METRIC} (relative to iGS)"
    "$PY" -m utils.GRFPControls.RelativeToIGS \
        --dirs "$SRC" --ref "WiGS (MAB-UCB1, c=2.0)" \
        --label "${METRIC} (relative to the bandit)"
done

echo ""
echo "Done. Figures in $(realpath "$OUT")"
du -sh ../results_grfp_controls/full_suite/*/* 2>/dev/null

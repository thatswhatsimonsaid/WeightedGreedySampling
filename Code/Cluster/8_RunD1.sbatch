#!/bin/bash
# Diagnostic 1: rerun the September paper-protocol comparison (9 selectors,
# 14 datasets, 100 seeds) logging full-pool, full-pool excluding labelled
# points, and test RMSE from the same runs.
# 14 datasets x 9 selectors x 5 seed-chunks of 20 = 630 tasks (array indices
# must stay under the cluster's MaxArraySize, typically 1001).
#
# Submit from the Code/ directory:
#     sbatch Cluster/8_RunD1.sbatch
# Then summarise (checks bit-for-bit reproduction of the September curves first):
#     python -m utils.GRFPControls.AnalyzeD1
#
# The slowest task (dgp_three_regime, a CV-reading selector, 20 seeds) is
# ~70 min for 20 seeds on a laptop core; the limit leaves ~4x headroom.

#SBATCH --job-name=CTRL_D1
#SBATCH --partition=cpu-g2
#SBATCH --array=1-630
#SBATCH --output=Cluster/RunControls/ClusterMessages/out/d1_%A_%a.out
#SBATCH --error=Cluster/RunControls/ClusterMessages/error/d1_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4000MB
#SBATCH --cpus-per-task=1

CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
case "$CODE_DIR" in
  */Cluster) CODE_DIR="${CODE_DIR%/Cluster}" ;;
esac
cd "$CODE_DIR" || exit 1

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ -z "${PYTHON:-}" ] && [ -f "../.WiGS_Env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ../.WiGS_Env/bin/activate
fi

"${PYTHON:-python}" -m utils.GRFPControls.RunD1 --TaskID "$SLURM_ARRAY_TASK_ID"

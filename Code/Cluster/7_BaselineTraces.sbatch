#!/bin/bash
# RMSE and CC trace plots for the five non-WiGS baselines (Passive Learning,
# QBC, GSx, GSy, iGS), on full-pool and held-out test error, for every dataset
# in the control sweep. Reads the per-task pickles under
# results_grfp_controls/cluster/ (which only exist on the cluster); nothing is
# rerun. Output: results_grfp_controls/baseline_traces/{full_pool,test}/images/
#
# Submit from the Code/ directory:
#     sbatch Cluster/7_BaselineTraces.sbatch
# Follow it with:
#     tail -f Cluster/RunControls/ClusterMessages/out/baselines_*.out

#SBATCH --job-name=CTRL_baselines
#SBATCH --partition=cpu-g2
#SBATCH --output=Cluster/RunControls/ClusterMessages/out/baselines_%j.out
#SBATCH --error=Cluster/RunControls/ClusterMessages/error/baselines_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
case "$CODE_DIR" in
  */Cluster) CODE_DIR="${CODE_DIR%/Cluster}" ;;
esac
cd "$CODE_DIR" || exit 1

export PYTHONDONTWRITEBYTECODE=1

if [ -z "${PYTHON:-}" ] && [ -f "../.WiGS_Env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ../.WiGS_Env/bin/activate
fi

"${PYTHON:-python}" -m utils.GRFPControls.BaselineTraces \
    --in ../results_grfp_controls/cluster \
    --out ../results_grfp_controls/baseline_traces

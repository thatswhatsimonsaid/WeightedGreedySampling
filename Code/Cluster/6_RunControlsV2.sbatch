#!/bin/bash
# Step 5 of the bandit-scale diagnosis: rerun the random-arm control with the
# rescaled bandit (reward_scaling="std") and no CV fallback, under the September
# controls' protocol. 13 datasets x 2 selectors x 10 seed-chunks = 260 tasks.
#
# Submit from the Code/ directory:
#     sbatch Cluster/6_RunControlsV2.sbatch
# Then summarise (a few minutes; fine on a login node, or wrap in sbatch):
#     python -m utils.GRFPControls.AnalyzeControlsV2
#
# The slowest task (dgp_three_regime, 10 seeds) measured ~12 min locally.

#SBATCH --job-name=CTRL_v2
#SBATCH --partition=cpu-g2
#SBATCH --array=1-260
#SBATCH --output=Cluster/RunControls/ClusterMessages/out/v2_%A_%a.out
#SBATCH --error=Cluster/RunControls/ClusterMessages/error/v2_%A_%a.err
#SBATCH --time=02:00:00
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

"${PYTHON:-python}" -m utils.GRFPControls.RunControlsV2 --TaskID "$SLURM_ARRAY_TASK_ID"

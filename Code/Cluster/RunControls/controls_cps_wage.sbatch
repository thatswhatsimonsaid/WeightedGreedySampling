#!/bin/bash
#SBATCH --job-name=CTRL_cps_wage
#SBATCH --partition=cpu-g2
#SBATCH --array=1-180
#SBATCH --output=Cluster/RunControls/ClusterMessages/out/cps_wage_%A_%a.out
#SBATCH --error=Cluster/RunControls/ClusterMessages/error/cps_wage_%A_%a.err
#SBATCH --time=00:20:59
#SBATCH --mem-per-cpu=4000MB
#SBATCH --cpus-per-task=1

# Submit from the Code/ directory (or from Cluster/RunControls/); the script
# locates itself rather than carrying a hard-coded path.
CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
case "$CODE_DIR" in
  */Cluster/RunControls) CODE_DIR="${CODE_DIR%/Cluster/RunControls}" ;;
esac
cd "$CODE_DIR" || exit 1

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Override with e.g.  PYTHON=python3  or a module/conda path, no edit needed.
"${PYTHON:-python}" -m utils.GRFPControls.RunControlsCluster \
    --Data "cps_wage" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --Protocol paper \
    --Standardize 1

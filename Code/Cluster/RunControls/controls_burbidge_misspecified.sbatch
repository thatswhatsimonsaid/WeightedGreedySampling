#!/bin/bash
#SBATCH --job-name=CTRL_burbidge_misspecified
#SBATCH --partition=cpu-g2
#SBATCH --array=1-1800
#SBATCH --output=/gscratch/stf/troy/WeightedGreedySampling/Code/Cluster/RunControls/ClusterMessages/out/burbidge_misspecified_%A_%a.out
#SBATCH --error=/gscratch/stf/troy/WeightedGreedySampling/Code/Cluster/RunControls/ClusterMessages/error/burbidge_misspecified_%A_%a.err
#SBATCH --time=00:20:59
#SBATCH --mem-per-cpu=4000MB
#SBATCH --cpus-per-task=1

cd "/gscratch/stf/troy/WeightedGreedySampling/Code"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m utils.GRFPControls.RunControlsCluster \
    --Data "burbidge_misspecified" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --Protocol paper \
    --Standardize 1

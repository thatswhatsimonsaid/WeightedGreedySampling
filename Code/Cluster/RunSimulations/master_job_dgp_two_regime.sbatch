#!/bin/bash
#SBATCH --job-name=AL_dgp_two_regime
#SBATCH --partition=cpu-g2
#SBATCH --array=1-100
#SBATCH --output=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/dgp_two_regime_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/dgp_two_regime_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1

cd /mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code
export PYTHONDONTWRITEBYTECODE=1

python RunSimulation.py \
    --Data "dgp_two_regime" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --CandidateProportion 0.95

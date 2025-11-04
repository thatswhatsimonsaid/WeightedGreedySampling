#!/bin/bash
#SBATCH --job-name=AL_dgp_three_regime
#SBATCH --partition=cpu-g2
#SBATCH --array=1-50
#SBATCH --output=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/out/dgp_three_regime_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code/Cluster/RunSimulations/ClusterMessages/error/dgp_three_regime_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1

cd /mmfs1/gscratch/stf/trusso96/WeightedGreedySampling/Code

python RunSimulation.py \
    --Data "dgp_three_regime" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 50 \
    --CandidateProportion 0.8

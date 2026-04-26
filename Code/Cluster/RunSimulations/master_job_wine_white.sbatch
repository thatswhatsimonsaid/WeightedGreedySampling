#!/bin/bash
#SBATCH --job-name=AL_wine_white
#SBATCH --partition=cpu-g2
#SBATCH --array=1-100
#SBATCH --output=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code/Cluster/RunSimulations/ClusterMessages/out/wine_white_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code/Cluster/RunSimulations/ClusterMessages/error/wine_white_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --mem-per-cpu=3000MB
#SBATCH --cpus-per-task=1

cd /mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code
export PYTHONDONTWRITEBYTECODE=1

python RunSimulation.py \
    --Data "wine_white" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --CandidateProportion 0.99

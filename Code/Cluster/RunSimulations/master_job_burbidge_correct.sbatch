#!/bin/bash
#SBATCH --job-name=AL_burbidge_correct
#SBATCH --partition=cpu-g2
#SBATCH --array=1-100
#SBATCH --output=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code/Cluster/RunSimulations/ClusterMessages/out/burbidge_correct_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code/Cluster/RunSimulations/ClusterMessages/error/burbidge_correct_%A_%a.err
#SBATCH --time=09-23:00:00
#SBATCH --mem-per-cpu=30000MB
#SBATCH --cpus-per-task=1

cd /mmfs1/gscratch/stf/trusso96/WeightedGreedySampling_Test_Split/Code
export PYTHONDONTWRITEBYTECODE=1

python RunSimulation.py \
    --Data "burbidge_correct" \
    --TaskID "$SLURM_ARRAY_TASK_ID" \
    --NReplications 100 \
    --CandidateProportion 0.99

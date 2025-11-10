#!/bin/bash

# --- This script generates ONE average heatmap per selector ---
#SBATCH --job-name=Heatmap_Avg
#SBATCH --output=Logs/heatmap_avg_job_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --partition=stf

echo "======================================================"
echo "Starting Average Heatmap Generation"
echo "Job started at: $(date)"
echo "======================================================"

# --- 1. Define Configuration Variables ---
DGP_NAME="dgp_three_regime"
N_SEEDS=50 # Tell the script to use 50 seeds
OUTPUT_DIR="Results/visualizations"
PYTHON_SCRIPT="Code/utils/Auxiliary/PlotAverageWeightHeatmap.py"

# --- 2. Define Selector to Run ---
# Hard-code the list to ONLY run WiGS (SAC)
SELECTORS=("WiGS (SAC)")
echo "Targeting selector: ${SELECTORS[0]}"

# --- 3. Loop Through Selector and Run Script ---
for selector in "${SELECTORS[@]}"; do
    echo "------------------------------------------------------"
    echo "Processing AVG for Selector: ${selector}"

    python3 "$PYTHON_SCRIPT" \
        --dgp_name "${DGP_NAME}" \
        --selector "${selector}" \
        --n_seeds ${N_SEEDS} \
        --output_dir "${OUTPUT_DIR}"
    
    echo "--- Finished ${selector} ---"
done

echo "======================================================"
echo "All average heatmaps complete."
echo "======================================================"
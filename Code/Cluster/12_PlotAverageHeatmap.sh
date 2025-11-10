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
SELECTOR_LIST_PATH="Results/simulation_results/aggregated/${DGP_NAME}/selection_history"

# --- 2. Find all Selectors ---
echo "Finding selectors in: $SELECTOR_LIST_PATH"
SELECTORS=() 
for f in "${SELECTOR_LIST_PATH}"/*_SelectionHistory.csv; do
    filename=$(basename "$f")
    selector_name="${filename%_SelectionHistory.csv}"
    SELECTORS+=("${selector_name}")
done
echo "Found ${#SELECTORS[@]} selectors."

# --- 3. Loop Through Selectors and Run Script ---
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
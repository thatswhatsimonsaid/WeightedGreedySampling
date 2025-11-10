#!/bin/bash

# --- This script generates 50 individual heatmaps (0-49) ---
#SBATCH --job-name=Heatmap_Indiv
#SBATCH --output=Logs/heatmap_indiv_job_%j.out
#SBATCH --time=01:00:00      # 1 hour
#SBATCH --mem=8G
#SBATCH --partition=stf      # Use the stf partition

echo "======================================================"
echo "Starting Individual Heatmap Generation"
echo "Job started at: $(date)"
echo "======================================================"

# --- 1. Define Configuration Variables ---
DGP_NAME="dgp_three_regime"
OUTPUT_DIR="Results/visualizations"
PYTHON_SCRIPT="Code/utils/Auxiliary/PlotWeightHeatmap.py"
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

# --- 3. Loop Through Selectors AND Seeds 0-49 ---
for selector in "${SELECTORS[@]}"; do
    echo "------------------------------------------------------"
    echo "Processing Selector: ${selector}"
    
    for seed in {0..49}; do
        echo "--- Generating heatmap for Seed: ${seed} ---"
        python3 "$PYTHON_SCRIPT" \
            --dgp_name "${DGP_NAME}" \
            --selector "${selector}" \
            --seed ${seed} \
            --output_dir "${OUTPUT_DIR}"
    done
    
    echo "--- Finished ${selector} ---"
done

echo "======================================================"
echo "All individual heatmaps complete."
echo "======================================================"
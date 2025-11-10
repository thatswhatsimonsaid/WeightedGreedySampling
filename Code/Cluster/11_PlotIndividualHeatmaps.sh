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

# --- 2. Define Selector to Run ---
# Hard-code the list to ONLY run WiGS (SAC)
SELECTORS=("WiGS (SAC)")
echo "Targeting selector: ${SELECTORS[0]}"

# --- 3. Loop Through Selector AND Seeds 0-49 ---
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
            
        if [ $? -ne 0 ]; then
            echo "ERROR: Python script failed for ${selector}, Seed ${seed}"
        fi
    done
    
    echo "--- Finished ${selector} ---"
done

echo "======================================================"
echo "All individual heatmaps complete."
echo "======================================================"
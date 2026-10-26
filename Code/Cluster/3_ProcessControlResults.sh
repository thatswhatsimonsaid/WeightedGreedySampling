#!/bin/bash
# Merge the per-task control pickles into the analysis format.
#
# Mirrors 3_ProcessResults.sh. It cannot call AggregateResults.py: that reads
# Results/simulation_results/raw in the manuscript's single-metric layout,
# whereas the control tasks write one pickle per (dataset, seed, selector)
# carrying both metrics.
#
# Prints a per-dataset completeness table. Check it before trusting any figure:
# a partially returned job array otherwise produces plots with missing seeds.

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

echo "--- Aggregating control results ---"
"${PYTHON:-python}" -m utils.GRFPControls.AggregateControls \
    --in ../results_grfp_controls/cluster \
    --out ../results_grfp_controls/full_suite

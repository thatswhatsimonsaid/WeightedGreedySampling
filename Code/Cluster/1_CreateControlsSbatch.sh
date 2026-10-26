#!/bin/bash
# Generate the sbatch files for the GRFP control sweep.
#
# Mirrors 1_CreateSimulationSbatch.sh, but targets the control harness rather
# than RunSimulation.py: it adds the random-arm control to the manuscript's 17
# strategies, records both metrics, and writes to results_grfp_controls/ so
# nothing under Results/ is touched.
#
# Wall times come from timing a probe per dataset, so run this ON the cluster.

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
PROJECT_ROOT=$(realpath "$PWD/..")

mkdir -p Cluster/RunControls/ClusterMessages/{out,error}
mkdir -p "$PROJECT_ROOT/results_grfp_controls/cluster"

echo "--- Generating control-sweep jobs (probing each dataset for wall time) ---"
"${PYTHON:-python}" -m utils.GRFPControls.CreateControlsSbatch --all "$@"
echo "--- Job generation complete. ---"

#!/bin/bash
# Submit the control-sweep job arrays.
#
# Mirrors 2_RunAllSimulations.sh, with two differences the control sweep needs:
#
#   * a --time override, so the limit can be raised without editing 20 files;
#   * QOS-aware submission. SLURM counts array elements against
#     QOSMaxSubmitJobPerUser, so submitting everything at once can be rejected
#     partway through. Failed datasets are retried after a pause rather than
#     silently skipped.
#
# Usage:
#   ./2_RunAllControls.sh                      # all datasets, 4h limit
#   TIME_LIMIT=08:00:00 ./2_RunAllControls.sh  # different limit
#   ./2_RunAllControls.sh beer bodyfat         # only these datasets

set -uo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
RETRY_WAIT="${RETRY_WAIT:-300}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"

if [ "$#" -gt 0 ]; then
    PENDING=(); for d in "$@"; do PENDING+=("Cluster/RunControls/controls_${d}.sbatch"); done
else
    PENDING=(Cluster/RunControls/controls_*.sbatch)
fi

echo "Submitting ${#PENDING[@]} job arrays with --time=${TIME_LIMIT}"
for (( round=1; round<=MAX_ROUNDS; round++ )); do
    FAILED=()
    for f in "${PENDING[@]}"; do
        [ -f "$f" ] || { echo "  missing: $f"; continue; }
        if out=$(sbatch --time="$TIME_LIMIT" "$f" 2>&1); then
            echo "  ok   $(basename "$f"): ${out##* }"
        elif grep -qi "QOSMaxSubmitJobPerUser\|job submit limit" <<<"$out"; then
            FAILED+=("$f")
        else
            echo "  FAIL $(basename "$f"): $(tail -1 <<<"$out")"
        fi
    done
    [ "${#FAILED[@]}" -eq 0 ] && { echo "All submitted."; exit 0; }
    echo "  ${#FAILED[@]} hit the submit limit; waiting ${RETRY_WAIT}s for capacity (round ${round}/${MAX_ROUNDS})"
    PENDING=("${FAILED[@]}")
    sleep "$RETRY_WAIT"
done
echo "Still pending after ${MAX_ROUNDS} rounds: ${PENDING[*]}"
exit 1

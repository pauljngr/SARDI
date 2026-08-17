#!/usr/bin/env bash
# Reproduce all ten DLM cells of Table 1.
#
# ~6-8 GPU-hours total on a single B200.
#
# Usage:
#   bash scripts/reproduce_table1.sh              # all ten cells
#   bash scripts/reproduce_table1.sh 2wiki        # one dataset, all three methods
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-results}"
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    # Cheapest first, so setup problems surface in minutes.
    DATASETS=(2wiki hotpotqa cofca musique synthworlds)
fi

mkdir -p "$OUT"

for ds in "${DATASETS[@]}"; do
    for spec in "sardi 0.9" "sardi 0.95" "ret_static 0.9"; do
        read -r method tau <<< "$spec"
        echo
        echo "############################################################"
        echo "# $ds | $method | tau_c=$tau"
        echo "############################################################"
        python evaluate.py --dataset "$ds" --method "$method" --threshold "$tau" \
            --output_dir "$OUT"
    done
done

echo
echo "Done. Per-cell summaries:"
echo "  grep -h accuracy $OUT/*.metrics.json"

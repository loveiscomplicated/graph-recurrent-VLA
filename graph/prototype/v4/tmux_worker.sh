#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mode=${1:?state or both required}
case "$mode" in state|both) ;; *) exit 2 ;; esac
log="outputs/graph/v4/training/surface_${mode}_mps.log"
failed="outputs/graph/v4/tmux_mps_${mode}.failed"
trap 'task_status=$?; if [ "$task_status" -ne 0 ]; then printf "%s\n" "$task_status" > "$failed"; fi' EXIT
.venv/bin/python -u -m graph.prototype.v4.train \
  --model surface --history "$mode" --length 4 --seed 0 --epochs 30 \
  --device mps --mps-memory-fraction .4 \
  --resume "outputs/graph/v4/training/surface_${mode}_cpu_interrupted/best.pt" \
  --output "outputs/graph/v4/training/surface_${mode}" 2>&1 | tee "$log"
printf 'Completed MPS %s training\n' "$mode"
touch "outputs/graph/v4/tmux_mps_${mode}.done"

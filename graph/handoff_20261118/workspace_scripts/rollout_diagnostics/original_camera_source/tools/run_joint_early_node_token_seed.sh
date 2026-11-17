#!/usr/bin/env bash
# Jointly fine-tune early-stopped source models, then evaluate their paired best.
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/node_token_launch.sh"
read_node_token_seed "${1:-}"
shift
configure_node_token_launch /workspace/joint_node_token_early_stopping "$@"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
trap 'printf "Seed %s failed at line %s\n" "$seed" "$LINENO" > "$experiment_root/seed_${seed}_FAILED"' ERR
run_node_token_recipe "graph/configs/temporal/insertion/train_joint_early_node_token_mamba_${seed}.yaml" "seed_${seed}.log" stream
printf 'Seed %s joint training and evaluation completed.\n' "$seed" > "$experiment_root/seed_${seed}_COMPLETE"

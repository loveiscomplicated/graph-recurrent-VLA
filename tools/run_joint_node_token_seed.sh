#!/usr/bin/env bash
# Run one seed; launch seeds 0, 1 and 2 in separate tmux windows for parallelism.
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/node_token_launch.sh"
read_node_token_seed "${1:-}"
shift
configure_node_token_launch /workspace/joint_node_token_mamba "$@"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
trap 'printf "Seed %s failed at line %s\n" "$seed" "$LINENO" > "$experiment_root/seed_${seed}_FAILED"' ERR
run_node_token_recipe "graph/configs/temporal/insertion/train_joint_node_token_mamba_${seed}.yaml" "seed_${seed}.log" stream
printf 'Seed %s completed 20 joint epochs.\n' "$seed" > "$experiment_root/seed_${seed}_COMPLETE"

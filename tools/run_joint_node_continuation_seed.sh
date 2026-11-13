#!/usr/bin/env bash
# Continue one paired seed and evaluate its selected checkpoint after stopping.
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/node_token_launch.sh"
read_node_token_seed "${1:-}"
shift
configure_node_token_launch /workspace/joint_node_token_continuation "$@"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
trap 'printf "Seed %s failed at line %s\n" "$seed" "$LINENO" > "$experiment_root/seed_${seed}_FAILED"' ERR
run_node_token_recipe "graph/configs/temporal/insertion/continue_joint_node_token_mamba_${seed}.yaml" "seed_${seed}.log" stream
printf 'Seed %s continuation and evaluation completed.\n' "$seed" > "$experiment_root/seed_${seed}_COMPLETE"

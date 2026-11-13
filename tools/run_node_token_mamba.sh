#!/usr/bin/env bash
# Run three independent seeds and aggregate the same selected validation frames.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/node_token_launch.sh"
configure_node_token_launch /workspace/node_token_mamba "$@"
trap 'printf "Failed at line %s\n" "$LINENO" > "$experiment_root/FAILED"' ERR
for seed in 0 1 2; do
    run_node_token_recipe "graph/configs/temporal/insertion/train_node_token_mamba_${seed}.yaml" "seed_${seed}.log"
done
run_node_token_recipe graph/configs/temporal/insertion/aggregate_node_token_mamba.yaml aggregate.log
printf 'Three seeds and selected-frame aggregation completed.\n' > "$experiment_root/COMPLETE"

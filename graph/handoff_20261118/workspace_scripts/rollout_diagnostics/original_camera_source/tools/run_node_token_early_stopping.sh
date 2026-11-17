#!/usr/bin/env bash
# Train three independent seed runs concurrently, then compare completed runs.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/node_token_launch.sh"
configure_node_token_launch /workspace/node_token_early_stopping "$@"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_seed() {
    local seed="$1"
    local log="$log_root/seed_${seed}.log"
    if run_node_token_recipe "graph/configs/temporal/insertion/train_node_token_early_stopping_${seed}.yaml" "seed_${seed}.log"; then
        printf 'Seed %s completed.\n' "$seed" > "$experiment_root/seed_${seed}_COMPLETE"
    else
        printf 'Seed %s failed; see %s\n' "$seed" "$log" > "$experiment_root/seed_${seed}_FAILED"
        return 1
    fi
}

pids=()
for seed in 0 1 2; do
    run_seed "$seed" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    printf 'At least one seed failed; comparison was not run.\n' > "$experiment_root/FAILED"
    exit 1
fi
if ! run_node_token_recipe graph/configs/temporal/insertion/compare_node_token_early_stopping.yaml comparison.log; then
    printf 'Comparison failed; see %s/comparison.log.\n' "$log_root" > "$experiment_root/FAILED"
    exit 1
fi
printf 'Three seeds and matched comparison completed.\n' > "$experiment_root/COMPLETE"

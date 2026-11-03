#!/bin/bash
set -euo pipefail
TASK10I_OPT_ROOT=/workspace/task10i_robot_filter_optimization_20261103_v1
TASK10I_OPT_CODE="$TASK10I_OPT_ROOT/code_final_v4"
run_main() {
  local name="$1"
  shift
  printf '%s\n' "$name" > "$TASK10I_OPT_ROOT/current_stage.txt"
  bash "$TASK10I_OPT_CODE/remote.sh" --output "$TASK10I_OPT_ROOT/$name" "$@" > "$TASK10I_OPT_ROOT/$name.log" 2>&1
}
printf 'base_tests_v4\n' > "$TASK10I_OPT_ROOT/current_stage.txt"
TASK10I_FILTER_ENTRY=base_tests.py bash "$TASK10I_OPT_CODE/remote.sh" > "$TASK10I_OPT_ROOT/base_tests_v4.log" 2>&1
run_main regression_v4 --cases "$TASK10I_OPT_CODE/regression.json" --variants baseline strict --pixels --repeats 0
run_main extra_v4 --cases "$TASK10I_OPT_CODE/extra.json" --variants baseline strict --pixels --repeats 0
run_main full_masks_v4 --cases "$TASK10I_OPT_CODE/full_masks.json" --variants baseline strict --pixels --repeats 0
run_main final_t4 --cases "$TASK10I_OPT_CODE/timed_interleaved.json" --variants baseline strict --warmups 3 --repeats 15 --threads 4
run_main cpu_graph_t4 --cases "$TASK10I_OPT_CODE/cpu_threads.json" --variants baseline cpu strict --warmups 3 --repeats 15 --threads 4
run_main threads_t1 --cases "$TASK10I_OPT_CODE/cpu_threads.json" --variants baseline strict --warmups 3 --repeats 15 --threads 1
run_main threads_t2 --cases "$TASK10I_OPT_CODE/cpu_threads.json" --variants baseline strict --warmups 3 --repeats 15 --threads 2
printf 'filter_small_v4\n' > "$TASK10I_OPT_ROOT/current_stage.txt"
TASK10I_FILTER_ENTRY=filter_benchmark.py bash "$TASK10I_OPT_CODE/remote.sh" --output "$TASK10I_OPT_ROOT/filter_small_v4" --cases "$TASK10I_OPT_CODE/filter_small.json" --variants cpu baseline hulls strict --warmups 3 --repeats 15 --threads 4 > "$TASK10I_OPT_ROOT/filter_small_v4.log" 2>&1
printf 'filter_full_v4\n' > "$TASK10I_OPT_ROOT/current_stage.txt"
TASK10I_FILTER_ENTRY=filter_benchmark.py bash "$TASK10I_OPT_CODE/remote.sh" --output "$TASK10I_OPT_ROOT/filter_full_v4" --cases "$TASK10I_OPT_CODE/full_masks.json" --variants baseline hulls strict --warmups 3 --repeats 15 --threads 4 > "$TASK10I_OPT_ROOT/filter_full_v4.log" 2>&1
TRITON_CACHE_DIR="$TASK10I_OPT_ROOT/jit_cold_cache_v4" run_main jit_cold_v4 --cases "$TASK10I_OPT_CODE/smoke.json" --variants baseline strict --pixels --repeats 0
printf 'full_contracts_v4\n' > "$TASK10I_OPT_ROOT/current_stage.txt"
TASK10I_FILTER_ENTRY=full_contracts.py TASK10I_CONTRACT_OUTPUT="$TASK10I_OPT_ROOT/full_contracts_v4" bash "$TASK10I_OPT_CODE/remote.sh" > "$TASK10I_OPT_ROOT/full_contracts_v4.log" 2>&1
printf 'diagnostic_v4\n' > "$TASK10I_OPT_ROOT/current_stage.txt"
TASK10I_FILTER_ENTRY=diagnostic.py bash "$TASK10I_OPT_CODE/remote.sh" --output "$TASK10I_OPT_ROOT/diagnostic_v4" --variants baseline hulls strict > "$TASK10I_OPT_ROOT/diagnostic_v4.log" 2>&1
TASK10I_FILTER_ENTRY=summarize.py bash "$TASK10I_OPT_CODE/remote.sh" "$TASK10I_OPT_ROOT" > "$TASK10I_OPT_ROOT/summary.log" 2>&1
ps -eo pid,ppid,stat,comm > "$TASK10I_OPT_ROOT/processes_after.txt"
printf 'COMPLETE\n' > "$TASK10I_OPT_ROOT/current_stage.txt"

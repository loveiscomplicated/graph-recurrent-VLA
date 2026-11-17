#!/usr/bin/env bash
set -euo pipefail
source <(sed -n '/^cd /,/^export WANDB/p' /workspace/recurrent_final_gtfree/run.sh)
export PYTHONPATH=/workspace/recurrent_final_gtfree:$PYTHONPATH
root=/workspace/recurrent_final_gtfree/case_videos
for begin in 0 4 8 12; do
    pids=()
    for ((i=begin;i<begin+4 && i<13;i++)); do
        /venv/main/bin/python "$root/record_cases.py" "$i" > "$root/case_$i.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    test "$status" = 0
done
echo 'Completed all representative videos'

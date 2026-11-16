#!/usr/bin/env bash
set -eu
base=/workspace/spatiotemporal_corner_tracker
reference=/workspace/moving_box_gt20_training
previous=/workspace/gt_corner_input_ablation
cd /workspace/graph-recurrent-VLA
for attempt in {1..150}; do
    done_count=$(find "$base"/shard_*/ -maxdepth 1 -name summary.json | wc -l)
    if [ "$done_count" -eq 6 ]; then break; fi
    if grep -q Traceback "$base"/shard_*.log; then
        tail -n 10 "$base"/shard_*.log >&2
        exit 1
    fi
    sleep 5
done
test "$done_count" -eq 6
export WANDB_MODE=disabled
export PYTHONPATH=.:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
python=/venv/main/bin/python
"$python" "$base/train_tracker.py" --reference "$reference" --capture "$base" --test "$previous" --output "$base/train"
"$python" "$base/evaluate_tracker.py" --train "$base/train" --previous "$previous" --output "$base/eval" --gnn "$reference/train/best.pt" --reference "$reference/train/test_moving.npz"

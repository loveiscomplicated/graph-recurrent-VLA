#!/usr/bin/env bash
set -eu
base=/workspace/gt_corner_input_ablation
cd /workspace/graph-recurrent-VLA
for attempt in {1..90}; do
  complete=$(find "$base"/shard_*/ -maxdepth 1 -name summary.json | wc -l)
  if [ "$complete" -eq 4 ]; then break; fi
  if grep -q Traceback "$base"/shard_*.log; then
    tail -n 12 "$base"/shard_*.log >&2
    exit 1
  fi
  sleep 5
done
test "$complete" -eq 4
export PYTHONPATH=.:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
export WANDB_MODE=disabled
source=/workspace/moving_box_gt20_training
exec /venv/main/bin/python "$base/evaluate_ablation.py" --dataset "$source/data_3100132_3100176/dataset.npz" --checkpoint "$source/train/best.pt" --reference "$source/train/test_moving.npz" --matches "$base" --output "$base/full"

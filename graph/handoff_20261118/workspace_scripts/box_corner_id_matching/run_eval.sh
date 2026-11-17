#!/usr/bin/env bash
set -eu
base=/workspace/box_corner_id_matching
reference=/workspace/moving_box_gt20_training
previous=/workspace/spatiotemporal_corner_tracker_convergence
cd /workspace/graph-recurrent-VLA
for attempt in {1..120}; do
    finished=$(find "$base"/shard_*/ -maxdepth 1 -name summary.json | wc -l)
    if [ "$finished" -eq 6 ]; then break; fi
    if grep -q Traceback "$base"/shard_*.log; then tail -n 12 "$base"/shard_*.log >&2; exit 1; fi
    sleep 5
done
test "$finished" -eq 6
export WANDB_MODE=disabled
export PYTHONPATH=.:$previous:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
exec /venv/main/bin/python "$base/evaluate_ids.py" --capture "$base" --reference "$reference" --output "$base/eval"

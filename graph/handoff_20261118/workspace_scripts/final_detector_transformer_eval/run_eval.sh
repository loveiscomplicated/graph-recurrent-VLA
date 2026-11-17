#!/usr/bin/env bash
set -eu
base=/workspace/final_detector_transformer_eval
model=/workspace/spatiotemporal_corner_tracker_convergence
reference=/workspace/moving_box_gt20_training/data_3100132_3100176/dataset.npz
cd /workspace/graph-recurrent-VLA
for attempt in {1..100}; do
    done_count=$(find "$base"/shard_*/ -maxdepth 1 -name summary.json | wc -l)
    if [ "$done_count" -eq 4 ]; then break; fi
    if grep -q Traceback "$base"/shard_*.log; then
        tail -n 10 "$base"/shard_*.log >&2
        exit 1
    fi
    sleep 5
done
test "$done_count" -eq 4
export WANDB_MODE=disabled
export PYTHONPATH=.:$model:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
exec /venv/main/bin/python "$base/evaluate_final.py" --reference "$reference" --capture "$base" --old-test "$model/train/test_predictions.npz" --checkpoint "$model/train/spatial_temporal/best.pt" --output "$base/eval"

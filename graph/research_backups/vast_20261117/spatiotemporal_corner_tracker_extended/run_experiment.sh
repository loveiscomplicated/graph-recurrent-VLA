#!/usr/bin/env bash
set -eu
base=/workspace/spatiotemporal_corner_tracker_extended
reference=/workspace/moving_box_gt20_training
capture=/workspace/spatiotemporal_corner_tracker
previous=/workspace/gt_corner_input_ablation
cd /workspace/graph-recurrent-VLA
export WANDB_MODE=disabled
export PYTHONPATH=.:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
python=/venv/main/bin/python
"$python" "$base/train_tracker.py" --mode spatial_temporal --epochs 80 --reference "$reference" --capture "$capture" --test "$previous" --output "$base/train"
"$python" "$base/evaluate_tracker.py" --train "$base/train" --previous "$previous" --output "$base/eval" --gnn "$reference/train/best.pt" --reference "$reference/train/test_moving.npz"

#!/usr/bin/env bash
set -eu
base=/workspace/spatiotemporal_corner_tracker_convergence
source=/workspace/spatiotemporal_corner_tracker_long/train/spatial_temporal
reference=/workspace/moving_box_gt20_training
capture=/workspace/spatiotemporal_corner_tracker
previous=/workspace/gt_corner_input_ablation
cd /workspace/graph-recurrent-VLA
export WANDB_MODE=disabled
export PYTHONPATH=.:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
python=/venv/main/bin/python
"$python" "$base/train_tracker.py" --mode spatial_temporal --resume "$source/resume.pt" --epochs 1600 --patience 60 --min-delta-mm 0.10 --lr-floors 0.0001 0.00003 --reference "$reference" --capture "$capture" --test "$previous" --output "$base/train"
"$python" "$base/evaluate_tracker.py" --train "$base/train" --previous "$previous" --output "$base/eval" --gnn "$reference/train/best.pt" --reference "$reference/train/test_moving.npz"

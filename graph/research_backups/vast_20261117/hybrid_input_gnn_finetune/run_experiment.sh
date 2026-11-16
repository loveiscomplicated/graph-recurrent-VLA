#!/usr/bin/env bash
set -eu
base=/workspace/hybrid_input_gnn_finetune
prior=/workspace/moving_box_gt20_training
capture=/workspace/spatiotemporal_corner_tracker
tracker=/workspace/spatiotemporal_corner_tracker_convergence
hybrid=/workspace/spatiotemporal_rigid_box_eval/eval/predictions.npz
cd /workspace/graph-recurrent-VLA
export WANDB_MODE=disabled
export PYTHONPATH=.:$tracker:/workspace/spatiotemporal_rigid_box_eval:/workspace/rigid_box_object21_eval:/workspace/gt20_object21_retraining:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector
python=/venv/main/bin/python
"$python" "$base/cache_inputs.py" --reference "$prior" --capture "$capture" --test /workspace/gt_corner_input_ablation --checkpoint "$tracker/train/spatial_temporal/best.pt" --prior-hybrid "$hybrid" --output "$base/cache"
"$python" "$base/train_finetune.py" --cache "$base/cache" --initial "$prior/train/best.pt" --prior-hybrid "$hybrid" --output "$base/train"

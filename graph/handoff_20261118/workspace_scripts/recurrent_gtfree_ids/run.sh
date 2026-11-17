#!/usr/bin/env bash
set -euo pipefail
dir=/workspace/recurrent_gtfree_ids
source=/workspace/recurrent_full_insertion
prior=/workspace/rollout_diagnostics/source
front=/workspace/full_insertion_frontend/trained_until_plateau
python=/venv/main/bin/python
cd /workspace/graph-recurrent-VLA
export PYTHONPATH=.:recurrent/exp2:/workspace/recurrent_gtfree_ids:/workspace/gtfree_corner_ids:/workspace/box_corner_id_matching:/workspace/recurrent_full_insertion:/workspace/recurrent_new_frontend:/workspace/full_insertion_frontend:/workspace/closed_loop_hybrid_mamba:/workspace/closed_loop_moving_box_mamba:/workspace/moving_box_gt20_training:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector:/workspace/gt20_gnn_detected_tracking_test:/workspace/gt20_object21_retraining:/workspace/spatiotemporal_corner_tracker_convergence:/workspace/rigid_box_object21_eval:/workspace/spatiotemporal_rigid_box_eval:/workspace/hybrid_input_gnn_finetune
export WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$dir/eval"
pids=()
for a in 0 4 8 12 16 20 24 28; do
    b=$((a+4))
    out="$dir/eval/shard_${a}_${b}"
    "$python" "$dir/evaluate_gtfree_ids.py" --prior "$prior" --train "$source/train" \
        --tracker "$front/tracker/best.pt" --gnn "$front/gnn/best.pt" \
        --moving-source /workspace/closed_loop_moving_box_mamba --output "$out" \
        --start "$a" --stop "$b" > "$out.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
echo 'Completed GT-free ID evaluation'

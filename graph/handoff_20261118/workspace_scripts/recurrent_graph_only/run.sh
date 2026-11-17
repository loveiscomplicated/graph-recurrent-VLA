#!/usr/bin/env bash
set -euo pipefail
root=/workspace/recurrent_graph_only
mixed=/workspace/recurrent_full_insertion
prior=/workspace/rollout_diagnostics/source
front=/workspace/full_insertion_frontend/trained_until_plateau
python=/venv/main/bin/python
cd /workspace/graph-recurrent-VLA
export PYTHONPATH=.:recurrent/exp2:/workspace/recurrent_graph_only:/workspace/recurrent_full_insertion:/workspace/recurrent_new_frontend:/workspace/full_insertion_frontend:/workspace/closed_loop_hybrid_mamba:/workspace/closed_loop_moving_box_mamba:/workspace/moving_box_gt20_training:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector:/workspace/gt20_gnn_detected_tracking_test:/workspace/gt20_object21_retraining:/workspace/spatiotemporal_corner_tracker_convergence:/workspace/rigid_box_object21_eval:/workspace/spatiotemporal_rigid_box_eval:/workspace/hybrid_input_gnn_finetune
export WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$root/train" "$root/eval"
pids=()
for i in 0 1 2 3 4; do
    "$python" "$root/train_graph_only.py" --data "$mixed/demos" \
        --seed "$i" --prior "$prior/recurrent/exp2/checkpoints/runs_r6_mix_filter/mamba_step_s$i/last.pt" \
        --output "$root/train/mamba_s$i" > "$root/train/mamba_s$i.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
echo 'Completed five Mamba checkpoints'
pids=()
for a in 0 4 8 12 16 20 24 28; do
    b=$((a+4))
    "$python" "$mixed/evaluate_full_front_mamba.py" --prior "$prior" --train "$root/train" \
        --tracker "$front/tracker/best.pt" --gnn "$front/gnn/best.pt" \
        --moving-source /workspace/closed_loop_moving_box_mamba --output "$root/eval/shard_${a}_${b}" \
        --start "$a" --stop "$b" > "$root/eval/shard_${a}_${b}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
echo 'Completed Graph-only Mamba pipeline'

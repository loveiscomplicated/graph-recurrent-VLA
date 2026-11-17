#!/usr/bin/env bash
set -euo pipefail
root=/workspace/recurrent_final_gtfree
previous=/workspace/recurrent_full_insertion/train
prior=/workspace/rollout_diagnostics/source
front=/workspace/gtfree_frontend_retrain/finetuned
python=/venv/main/bin/python
cd /workspace/graph-recurrent-VLA
export PYTHONPATH=.:recurrent/exp2:/workspace/gtfree_frontend_retrain:/workspace/recurrent_gtfree_ids:/workspace/gtfree_corner_ids:/workspace/box_corner_id_matching:/workspace/recurrent_full_insertion:/workspace/recurrent_new_frontend:/workspace/full_insertion_frontend:/workspace/closed_loop_hybrid_mamba:/workspace/closed_loop_moving_box_mamba:/workspace/moving_box_gt20_training:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector:/workspace/gt20_gnn_detected_tracking_test:/workspace/gt20_object21_retraining:/workspace/spatiotemporal_corner_tracker_convergence:/workspace/rigid_box_object21_eval:/workspace/spatiotemporal_rigid_box_eval:/workspace/hybrid_input_gnn_finetune
export WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_group() {
    local status=0
    for pid in "$@"; do wait "$pid" || status=1; done
    return "$status"
}

evaluate() {
    local arm=$1 models=$2
    shift 2
    local out="$root/$arm"
    local pids=()
    mkdir -p "$out"
    for a in 0 4 8 12 16 20 24 28; do
        local b=$((a+4))
        local shard="$out/shard_${a}_${b}"
        if [[ -s "$shard/summary.json" ]]; then continue; fi
        if [[ -d "$shard" ]]; then mv "$shard" "$shard.incomplete.$(date +%s)"; fi
        "$python" "$root/evaluate_final.py" --prior "$prior" --train "$models" "$@" \
            --tracker "$front/tracker/best.pt" --gnn "$front/gnn/best.pt" \
            --moving-source /workspace/closed_loop_moving_box_mamba --output "$shard" \
            --start "$a" --stop "$b" > "$shard.log" 2>&1 &
        pids+=("$!")
    done
    wait_group "${pids[@]}"
    echo "Completed $arm closed-loop evaluation"
}

prepare() {
    if [[ -f "$root/demos.complete" ]]; then return; fi
    if [[ -d "$root/demos" ]]; then mv "$root/demos" "$root/demos.incomplete.$(date +%s)"; fi
    "$python" "$root/prepare_final.py" "$root/demos" > "$root/prepare.log" 2>&1
    touch "$root/demos.complete"
}

prepare &
prepare_pid=$!
evaluate baseline "$previous"
wait "$prepare_pid"
echo 'Completed Expert data alignment'

mkdir -p "$root/train"
pids=()
for i in 0 1 2 3 4; do
    out="$root/train/mamba_s$i"
    if [[ -s "$out/summary.json" ]]; then continue; fi
    if [[ -d "$out" ]]; then mv "$out" "$out.incomplete.$(date +%s)"; fi
    "$python" "$root/train_final.py" --data "$root/demos" --seed "$i" \
        --prior "$previous/mamba_s$i/last.pt" --output "$out" > "$out.log" 2>&1 &
    pids+=("$!")
done
wait_group "${pids[@]}"
echo 'Completed five Mamba fine-tunes'
evaluate finetuned "$root/train" --warm-start "$previous"

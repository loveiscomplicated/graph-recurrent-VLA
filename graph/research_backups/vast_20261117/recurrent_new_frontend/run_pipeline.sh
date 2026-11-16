#!/usr/bin/env bash
set -eu
base=/workspace/recurrent_new_frontend
prior=/workspace/rollout_diagnostics/source
tracker=/workspace/spatiotemporal_corner_tracker_convergence/train/spatial_temporal/best.pt
gnn=/workspace/hybrid_input_gnn_finetune/train/best.pt
cd /workspace/graph-recurrent-VLA
export WANDB_MODE=disabled
export PYTHONPATH=.:recurrent/exp2:/workspace/closed_loop_hybrid_mamba:/workspace/closed_loop_moving_box_mamba:/workspace/moving_box_gt20_training:/workspace/full_sim_gt_direct_gnn:/workspace/hole_front_object21_latest_detector:/workspace/gt20_gnn_detected_tracking_test:/workspace/gt20_object21_retraining:/workspace/spatiotemporal_corner_tracker_convergence:/workspace/rigid_box_object21_eval
python=/venv/main/bin/python
for attempt in {1..360}; do
    count=$(grep -l '^Completed full Expert collection' "$base"/shard_*.log | wc -l)
    if [ "$count" -eq 8 ]; then break; fi
    if grep -q Traceback "$base"/shard_*.log; then tail -n 15 "$base"/shard_*.log >&2; exit 1; fi
    sleep 5
done
test "$count" -eq 8
mkdir -p "$base/train" "$base/eval"
pids=()
for m in 0 1 2 3 4; do
    "$python" "$base/train_mamba.py" --seed "$m" --data "$base" --prior "$prior/recurrent/exp2/checkpoints/runs_r6_mix_filter/mamba_step_s$m/last.pt" --output "$base/train/mamba_s$m" >"$base/train/mamba_s$m.log" 2>&1 &
    pids+=("$!")
done
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
test "$code" -eq 0
pids=()
for a in 0 4 8 12 16 20 24 28; do
    b=$((a+4))
    shard="$base/eval/shard_${a}_${b}"
    "$python" "$base/evaluate_mamba.py" --prior "$prior" --train "$base/train" --tracker "$tracker" --gnn "$gnn" --moving-source /workspace/closed_loop_moving_box_mamba --output "$shard" --start "$a" --stop "$b" >"$shard.log" 2>&1 &
    pids+=("$!")
done
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
test "$code" -eq 0
echo 'Completed collection, Mamba training, and fixed/moving evaluation'

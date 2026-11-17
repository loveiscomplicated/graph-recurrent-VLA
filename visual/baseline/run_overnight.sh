#!/usr/bin/env bash
# Unattended queue: diagnostic evals of the 100k baselines (eval + training seeds,
# videos, failure stages), intermediate checkpoints, latency, and two variant
# trainings (ACT images-only, Diffusion with state noise) run in parallel on the GPU.
# Progress is pushed to ntfy (topic in outputs/visual_baseline/ntfy_topic.txt).
# Usage: bash visual/baseline/run_overnight.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2 WANDB_MODE=offline
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)

notify() {  # title, message
    curl -s -m 15 -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" > /dev/null || true
}

evaluate() {  # run, step, seeds file, output name, videos, notify(yes/no)
    local checkpoint=$base/$1/checkpoint_$2.pt output=$base/$1/$4
    if [ ! -f "$checkpoint" ]; then echo "missing $checkpoint"; notify "missing checkpoint" "$checkpoint"; return; fi
    [ -f "$output" ] && return
    echo "== $1 $2 -> $output ($(date -u))"
    local result
    result=$($python -m visual.baseline.evaluate --checkpoint "$checkpoint" --seeds "$3" --output "$output" \
        --workers 12 --videos "$5" 2>&1 | grep -v -i warn | tail -1)
    echo "$result"
    [ "$6" = yes ] && notify "$1 $2 ($4)" "$result"
}

train() {  # config name
    local log=$base/run_$1.log
    if $python main.py --config visual/configs/baseline_$1.yaml > "$log" 2>&1; then
        notify "training done: $1" "$(grep validation_loss "$log" | tail -1)"
    else
        notify "training FAILED: $1" "$(tail -2 "$log" | cut -c1-180)"
    fi
}

notify "overnight start" "evals of ACT/DP 100k + training act_nostate, diffusion_statenoise; pod stops by 16:00 UTC"
train act_nostate &
train diffusion_statenoise &

for run in act_main diffusion_main; do evaluate $run 0100000 $base/eval_seeds.json eval_0100000.json 10 yes; done
for run in act_main diffusion_main; do evaluate $run 0100000 $base/train_seeds.json train_0100000.json 0 yes; done
for step in 0075000 0050000 0025000; do
    for run in act_main diffusion_main; do evaluate $run $step $base/eval_seeds.json eval_$step.json 0 no; done
done
notify "phase 1 done" "$($python visual/baseline/summarize.py $base | cut -c1-900)"

wait  # variant trainings
for run in act_nostate diffusion_statenoise; do
    evaluate $run 0060000 $base/eval_seeds.json eval_0060000.json 10 yes
    evaluate $run 0060000 $base/train_seeds.json train_0060000.json 0 yes
done
for step in 0040000 0020000; do
    for run in act_nostate diffusion_statenoise; do evaluate $run $step $base/eval_seeds.json eval_$step.json 0 no; done
done
# Single-worker latency on 10 seeds with the GPU otherwise idle.
for run in act_main diffusion_main; do
    [ -f $base/$run/latency_0100000.json ] || $python -m visual.baseline.evaluate \
        --checkpoint $base/$run/checkpoint_0100000.pt --seeds $base/latency_seeds.json \
        --output $base/$run/latency_0100000.json --workers 1 2>&1 | grep -v -i warn | tail -1
done
echo "queue done $(date -u)"

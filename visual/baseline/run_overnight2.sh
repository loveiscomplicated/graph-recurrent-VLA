#!/usr/bin/env bash
# Unattended queue sized for a 24 GB GPU / 125 GB container:
# one training at a time, eval workers limited (ACT ~1.5 GB, Diffusion ~2.9 GB each).
# 1) evals of the 100k baselines while act_nostate trains, 2) act_nostate evals,
# 3) diffusion_statenoise training alone, then its evals, 4) latency runs.
# Progress is pushed to ntfy (topic in outputs/visual_baseline/ntfy_topic.txt).
# Usage: bash visual/baseline/run_overnight2.sh <pid of the running act_nostate training>
set -uo pipefail
cd "$(dirname "$0")/../.."
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2 WANDB_MODE=offline
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
act_training_pid=$1

notify() {  # title, message
    curl -s -m 15 -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" > /dev/null || true
}

evaluate() {  # run, step, seeds file, output name, videos, workers, notify(yes/no)
    local checkpoint=$base/$1/checkpoint_$2.pt output=$base/$1/$4
    if [ ! -f "$checkpoint" ]; then echo "missing $checkpoint"; notify "missing checkpoint" "$checkpoint"; return; fi
    [ -f "$output" ] && return
    echo "== $1 $2 -> $output ($(date -u))"
    local result
    result=$($python -m visual.baseline.evaluate --checkpoint "$checkpoint" --seeds "$3" --output "$output" \
        --workers "$6" --videos "$5" 2>&1 | grep -v -i warn | tail -1 | cut -c1-300)
    echo "$result"
    if [ ! -f "$output" ]; then notify "eval FAILED: $1 $2 ($4)" "$result"
    elif [ "$7" = yes ]; then notify "$1 $2 ($4)" "$result"; fi
}

notify "queue v2 start" "24GB GPU: DP evals + ACT nostate training now; DP statenoise training after. Pod stops by 16:00 UTC"

# 1) Baseline evals alongside the act_nostate training (~6 GB GPU).
evaluate diffusion_main 0100000 $base/eval_seeds.json eval_0100000.json 10 5 yes
evaluate diffusion_main 0100000 $base/train_seeds.json train_0100000.json 0 5 yes
for step in 0075000 0050000 0025000; do
    evaluate act_main $step $base/eval_seeds.json eval_$step.json 0 8 no
    evaluate diffusion_main $step $base/eval_seeds.json eval_$step.json 0 5 no
done
notify "phase 1 done" "$($python visual/baseline/summarize.py $base | cut -c1-900)"

# 2) act_nostate evals once its training ends.
while kill -0 "$act_training_pid" 2>/dev/null; do sleep 30; done
notify "training ended: act_nostate" "$(grep validation_loss $base/run_act_nostate.log | tail -1)"
evaluate act_nostate 0060000 $base/eval_seeds.json eval_0060000.json 10 8 yes
evaluate act_nostate 0060000 $base/train_seeds.json train_0060000.json 0 8 yes
for step in 0040000 0020000; do evaluate act_nostate $step $base/eval_seeds.json eval_$step.json 0 8 no; done

# 3) diffusion_statenoise training alone, then its evals.
rm -rf $base/diffusion_statenoise  # partial output of the earlier OOM-killed attempt
if $python main.py --config visual/configs/baseline_diffusion_statenoise.yaml > $base/run_diffusion_statenoise.log 2>&1; then
    notify "training done: diffusion_statenoise" "$(grep validation_loss $base/run_diffusion_statenoise.log | tail -1)"
else
    notify "training FAILED: diffusion_statenoise" "$(tail -2 $base/run_diffusion_statenoise.log | cut -c1-180)"
fi
evaluate diffusion_statenoise 0060000 $base/eval_seeds.json eval_0060000.json 10 6 yes
evaluate diffusion_statenoise 0060000 $base/train_seeds.json train_0060000.json 0 6 yes
for step in 0040000 0020000; do evaluate diffusion_statenoise $step $base/eval_seeds.json eval_$step.json 0 6 no; done

# 4) Single-worker latency on 10 seeds with the GPU otherwise idle.
for run in act_main diffusion_main; do evaluate $run 0100000 $base/latency_seeds.json latency_0100000.json 0 1 no; done
echo "queue done $(date -u)"

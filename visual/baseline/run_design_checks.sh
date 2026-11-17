#!/usr/bin/env bash
# Experiment-design checks for the low closed-loop success (24 GB GPU: one training
# at a time plus a limited number of eval workers):
#   H4 temporal ensembling lag   -> re-evaluate ACT 100k with chunk execution / per-step replanning
#   H1 overfitting               -> intermediate checkpoints (25k/50k/75k)
#   proprioception reliance      -> act_nostate (training already running)
#   H2 random shift hurts        -> act_noshift training
#   state reliance in Diffusion  -> diffusion_statenoise training (last; may hit the deadline)
# Progress is pushed to ntfy (topic in outputs/visual_baseline/ntfy_topic.txt).
# Usage: bash visual/baseline/run_design_checks.sh <pid of the running act_nostate training>
set -uo pipefail
cd "$(dirname "$0")/../.."
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2 WANDB_MODE=offline
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
act_nostate_pid=$1

notify() {  # title, message
    curl -s -m 15 -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" > /dev/null || true
}

evaluate() {  # run, step, seeds file, output name, videos, workers, notify(yes/no), [--policy-override ...]
    local checkpoint=$base/$1/checkpoint_$2.pt output=$base/$1/$4 seeds=$3 videos=$5 workers=$6 announce=$7
    shift 7
    if [ ! -f "$checkpoint" ]; then echo "missing $checkpoint"; notify "missing checkpoint" "$checkpoint"; return; fi
    [ -f "$output" ] && return
    echo "== $checkpoint -> $output $* ($(date -u))"
    local result
    result=$($python -m visual.baseline.evaluate --checkpoint "$checkpoint" --seeds "$seeds" --output "$output" \
        --workers "$workers" --videos "$videos" "$@" 2>&1 | grep -v -i warn | tail -1 | cut -c1-300)
    echo "$result"
    if [ ! -f "$output" ]; then notify "eval FAILED: $(basename $(dirname $output)) $4" "$result"
    elif [ "$announce" = yes ]; then notify "$(basename $(dirname $output)) $4" "$result"; fi
}

start_training() {  # config name; runs in background, notifies at the end
    (
        rm -rf "$base/$1"  # partial output of an earlier interrupted attempt, if any
        if $python main.py --config visual/configs/baseline_$1.yaml > $base/run_$1.log 2>&1; then
            notify "training done: $1" "$(grep validation_loss $base/run_$1.log | tail -1)"
        else
            notify "training FAILED: $1" "$(tail -2 $base/run_$1.log | cut -c1-180)"
        fi
    ) &
    training_pid=$!
}

notify "design checks start" "H4 ensemble, overfitting ckpts, act_nostate, act_noshift, diffusion_statenoise. Pod stops by 16:00 UTC"

# H4: inference-only variants of ACT 100k.
evaluate act_main 0100000 $base/eval_seeds.json eval_0100000_chunk10.json 5 8 yes \
    --policy-override temporal_ensemble_coeff=null --policy-override n_action_steps=10
evaluate act_main 0100000 $base/eval_seeds.json eval_0100000_replan1.json 0 8 yes \
    --policy-override temporal_ensemble_coeff=null --policy-override n_action_steps=1

# H1: intermediate checkpoints of both baselines.
for step in 0075000 0050000 0025000; do
    evaluate act_main $step $base/eval_seeds.json eval_$step.json 0 8 no
    evaluate diffusion_main $step $base/eval_seeds.json eval_$step.json 0 5 no
done
notify "baseline checks done" "$($python visual/baseline/summarize.py $base | cut -c1-900)"

# act_nostate evals, overlapped with the act_noshift training.
while kill -0 "$act_nostate_pid" 2>/dev/null; do sleep 30; done
notify "training ended: act_nostate" "$(grep validation_loss $base/run_act_nostate.log | tail -1)"
start_training act_noshift
evaluate act_nostate 0060000 $base/eval_seeds.json eval_0060000.json 10 8 yes
evaluate act_nostate 0060000 $base/train_seeds.json train_0060000.json 0 8 yes
for step in 0040000 0020000; do evaluate act_nostate $step $base/eval_seeds.json eval_$step.json 0 8 no; done
wait "$training_pid"

# act_noshift evals, overlapped with the diffusion_statenoise training.
start_training diffusion_statenoise
evaluate act_noshift 0060000 $base/eval_seeds.json eval_0060000.json 10 6 yes
evaluate act_noshift 0060000 $base/train_seeds.json train_0060000.json 0 6 yes
for step in 0040000 0020000; do evaluate act_noshift $step $base/eval_seeds.json eval_$step.json 0 6 no; done
wait "$training_pid"

evaluate diffusion_statenoise 0060000 $base/eval_seeds.json eval_0060000.json 10 5 yes
evaluate diffusion_statenoise 0060000 $base/train_seeds.json train_0060000.json 0 5 yes

# Single-worker latency on 10 seeds with the GPU otherwise idle.
for run in act_main diffusion_main; do evaluate $run 0100000 $base/latency_seeds.json latency_0100000.json 0 1 no; done
for step in 0040000 0020000; do evaluate diffusion_statenoise $step $base/eval_seeds.json eval_$step.json 0 5 no; done
echo "queue done $(date -u)"

#!/usr/bin/env bash
# Wait for a training process to exit, then evaluate checkpoints in priority order:
# final (100k) checkpoints first, intermediate ones after, then a single-worker
# latency run on 10 seeds with the GPU otherwise idle.
# Usage: bash visual/baseline/run_eval_queue.sh <training pid>
set -uo pipefail
cd "$(dirname "$0")/../.."
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json
base=outputs/visual_baseline

while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "training pid $1 exited at $(date -u)"

evaluate() {  # run, step, workers, seeds file, output name
    local checkpoint=$base/$1/checkpoint_$2.pt output=$base/$1/$5
    if [ ! -f "$checkpoint" ]; then echo "missing $checkpoint"; return; fi
    [ -f "$output" ] && return
    echo "== $1 $2 -> $output ($(date -u))"
    $python -m visual.baseline.evaluate --checkpoint "$checkpoint" --seeds "$4" \
        --output "$output" --workers "$3" 2>&1 | grep -v -i warn | tail -1
}

for run in act_main diffusion_main; do evaluate $run 0100000 16 $base/eval_seeds.json eval_0100000.json; done
for step in 0075000 0050000 0025000; do
    for run in act_main diffusion_main; do evaluate $run $step 16 $base/eval_seeds.json eval_$step.json; done
done
for run in act_main diffusion_main; do evaluate $run 0100000 1 $base/latency_seeds.json latency_0100000.json; done
echo "queue done $(date -u)"

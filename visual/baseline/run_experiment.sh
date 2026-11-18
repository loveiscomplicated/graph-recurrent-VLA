#!/usr/bin/env bash
# Train one baseline config, then evaluate every saved checkpoint closed-loop on
# new scenes, and the final checkpoint also on training scenes.
#
# Needs the seed files written by visual.baseline.eval_seeds (README step 3):
#   outputs/visual_baseline_shared/eval_seeds.json   new scenes
#   outputs/visual_baseline_shared/train_seeds.json  training scenes
#
# Usage: PYTHON=/path/to/venv/bin/python bash visual/baseline/run_experiment.sh visual/baseline/configs/act.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -ne 1 ]; then
    echo "usage: bash visual/baseline/run_experiment.sh <config.yaml>" >&2
    exit 1
fi
config=$1
python=${PYTHON:-python}
export WANDB_MODE=${WANDB_MODE:-offline}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$PWD/visual/vulkan/nvidia_icd_egl.json}

seeds=outputs/visual_baseline_shared
for name in eval_seeds train_seeds; do
    if [ ! -f "$seeds/$name.json" ]; then
        echo "missing $seeds/$name.json; create it with visual.baseline.eval_seeds first" >&2
        exit 1
    fi
done

output=$("$python" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$config")

"$python" main.py --config "$config"

final_checkpoint=""
for checkpoint in "$output"/checkpoint_*.pt; do
    step=$(basename "$checkpoint" .pt | cut -d_ -f2)
    "$python" -m visual.baseline.evaluate --checkpoint "$checkpoint" \
        --seeds "$seeds/eval_seeds.json" --output "$output/eval_$step.json" --videos 10
    final_checkpoint=$checkpoint
done

final_step=$(basename "$final_checkpoint" .pt | cut -d_ -f2)
"$python" -m visual.baseline.evaluate --checkpoint "$final_checkpoint" \
    --seeds "$seeds/train_seeds.json" --output "$output/train_$final_step.json"

"$python" visual/baseline/summarize.py "$(dirname "$output")"

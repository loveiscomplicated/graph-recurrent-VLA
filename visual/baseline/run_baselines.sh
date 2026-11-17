#!/usr/bin/env bash
# Train one baseline config, then evaluate every saved checkpoint closed-loop.
# Usage: bash visual/baseline/run_baselines.sh visual/configs/baseline_act.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."
config=$1
python=/workspace/envs/baseline/bin/python
export WANDB_MODE=offline VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json
output=$($python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$config")

$python main.py --config "$config"
for checkpoint in "$output"/checkpoint_*.pt; do
    step=$(basename "$checkpoint" .pt | cut -d_ -f2)
    $python -m visual.baseline.evaluate --checkpoint "$checkpoint" \
        --seeds outputs/visual_baseline/eval_seeds.json --output "$output/eval_$step.json"
done

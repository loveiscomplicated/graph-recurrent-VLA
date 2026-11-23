#!/usr/bin/env bash
# End-to-end smoke test of the Visual baselines on a Linux CUDA host:
# collect 10 expert episodes -> prepare -> pick 4 eval seeds -> train ACT and
# Diffusion for 200 steps -> evaluate each checkpoint closed-loop.
# It only checks that every stage runs; the success rates mean nothing.
#
# Usage: PYTHON=/path/to/venv/bin/python bash visual/baseline/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

python=${PYTHON:-python}
export WANDB_MODE=${WANDB_MODE:-offline}
# Keep a host-provided Vulkan ICD; otherwise use the repository's NVIDIA EGL ICD.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$PWD/visual/vulkan/nvidia_icd_egl.json}

collection=outputs/visible_peg_shared_camera_smoke
episodes=data/visual_baseline/episodes_smoke
results=outputs/visual_baseline_smoke

for path in "$collection" "$episodes" "$results"; do
    if [ -e "$path" ]; then
        echo "$path already exists; remove it to rerun the smoke test" >&2
        exit 1
    fi
done

echo "== 1/4 collect 10 expert episodes"
"$python" -m data_collection.collect_dataset \
    --config data_collection/configs/visible_peg_shared_camera_smoke.yaml

echo "== 2/4 prepare episodes"
"$python" -m visual.baseline.prepare_data \
    --archives "$collection" --output "$episodes" --expected-shards 2 --workers 2

echo "== 3/4 pick evaluation seeds"
mkdir -p "$results"
"$python" -m visual.baseline.eval_seeds \
    --output "$results/eval_seeds.json" --num-seeds 4 --scan 80 --workers 2

echo "== 4/4 train and evaluate"
for policy in act diffusion; do
    "$python" main.py --config "visual/baseline/configs/${policy}_smoke.yaml"
    "$python" -m visual.baseline.evaluate \
        --checkpoint "$results/$policy/checkpoint_0000200.pt" \
        --seeds "$results/eval_seeds.json" \
        --output "$results/$policy/eval_0000200.json" \
        --workers 2 --videos 1
done

echo "smoke test passed: $results"

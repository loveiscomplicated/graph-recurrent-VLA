#!/usr/bin/env bash
# Early signal: evaluate act_dart 20k while training continues (6 workers ≈ 9 GB GPU).
cd /workspace/graph-recurrent-VLA
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
until [ -f $base/act_dart/checkpoint_0020000.pt ]; do sleep 30; done
sleep 20  # let torch.save finish
result=$($python -m visual.baseline.evaluate --checkpoint $base/act_dart/checkpoint_0020000.pt \
    --seeds $base/eval_seeds.json --output $base/act_dart/eval_0020000.json --workers 6 2>&1 | grep -v -i warn | tail -1 | cut -c1-300)
echo "$result"
curl -s -m 15 -H "Title: act_dart 20k (early)" -d "$result" "https://ntfy.sh/$topic" > /dev/null

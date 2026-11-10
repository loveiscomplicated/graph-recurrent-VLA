#!/usr/bin/env bash
# Ad-hoc check (2026-11-10): does binarizing the gripper command fix the weak grasp?
# Runs after the queue's current eval, before act_nostate training ends (~08:35 UTC).
cd /workspace/graph-recurrent-VLA
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
while pgrep -f "visual.baseline.evaluate" > /dev/null; do sleep 15; done
run() {  # name, checkpoint, workers, extra args
    local out=$base/$1 result
    result=$($python -m visual.baseline.evaluate --checkpoint $2 --seeds $base/eval_seeds.json --output $out \
        --workers $3 "${@:4}" 2>&1 | grep -v -i warn | tail -1 | cut -c1-300)
    echo "$1: $result"
    curl -s -m 15 -H "Title: gripper check $1" -d "$result" "https://ntfy.sh/$topic" > /dev/null
}
ens="--policy-override temporal_ensemble_coeff=null --policy-override n_action_steps=10"
run act_main/eval_0100000_chunk10_stages.json $base/act_main/checkpoint_0100000.pt 8 $ens
run act_main/eval_0100000_chunk10_bingrip.json $base/act_main/checkpoint_0100000.pt 8 $ens --binary-gripper --videos 6
run diffusion_main/eval_0100000_bingrip.json $base/diffusion_main/checkpoint_0100000.pt 5 --binary-gripper
echo "done $(date -u)"

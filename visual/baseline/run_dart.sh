#!/usr/bin/env bash
# DART round (2026-11-10). Evidence: 85-90% of failures reach insertion depth but stay
# 2-4 cm laterally misaligned; the error is unrelated to hole position and in-hand offset
# is tiny, so the demos lack recovery behaviour. Collect burst-noise expert data with
# clean labels, then train ACT and Diffusion on original 900 + DART 600 episodes.
# Usage: bash visual/baseline/run_dart.sh <pid of the running act_nostate training>
set -uo pipefail
cd "$(dirname "$0")/../.."
python=/workspace/envs/baseline/bin/python
export VK_ICD_FILENAMES=$PWD/visual/vulkan/nvidia_icd_egl.json OMP_NUM_THREADS=2 WANDB_MODE=offline
base=outputs/visual_baseline
data=data/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
act_nostate_pid=$1

notify() { curl -s -m 15 -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" > /dev/null || true; }

evaluate() {  # run, step, seeds file, output name, videos, workers, notify(yes/no)
    local checkpoint=$base/$1/checkpoint_$2.pt output=$base/$1/$4
    if [ ! -f "$checkpoint" ]; then echo "missing $checkpoint"; notify "missing checkpoint" "$checkpoint"; return; fi
    [ -f "$output" ] && return
    echo "== $1 $2 -> $output ($(date -u))"
    local result
    result=$($python -m visual.baseline.evaluate --checkpoint "$checkpoint" --seeds "$3" --output "$output" \
        --workers "$6" --videos "$5" 2>&1 | grep -v -i warn | tail -1 | cut -c1-300)
    echo "$result"
    if [ ! -f "$output" ]; then notify "eval FAILED: $1 $4" "$result"
    elif [ "$7" = yes ]; then notify "$1 $4" "$result"; fi
}

train() {  # config name (foreground)
    if $python main.py --config visual/configs/baseline_$1.yaml > $base/run_$1.log 2>&1; then
        notify "training done: $1" "$(grep validation_loss $base/run_$1.log | tail -1)"
    else
        notify "training FAILED: $1" "$(tail -2 $base/run_$1.log | cut -c1-180)"
    fi
}

notify "DART round start" "lateral misalignment = 85-90% of failures -> collecting 600 burst-noise expert episodes, then ACT+DP on 900+600"

# 1) DART collection (8 workers share the GPU with the act_nostate training).
if [ ! -f $data/episodes_dart/index.json ]; then
    rm -rf $data/episodes_dart  # incomplete earlier attempt, if any
    $python -m visual.baseline.collect_dart --output $data/episodes_dart --episodes 600 --workers 8 \
        > $base/collect_dart.log 2>&1
    notify "DART collected" "$(grep -v -i warn $base/collect_dart.log | tail -1)"
fi

# 2) Mixed dataset: symlinks to both sets plus a merged index.
if [ ! -f $data/episodes_mix/index.json ]; then
    $python - <<'PY'
import json
from pathlib import Path
root = Path("data/visual_baseline")
mix = root / "episodes_mix"
mix.mkdir(exist_ok=True)
rows = []
for name in ("episodes", "episodes_dart"):
    index = json.loads((root / name / "index.json").read_text())
    for row in index["rows"]:
        link = mix / row["file"]
        if not link.exists():
            link.symlink_to((root / name / row["file"]).resolve())
        rows.append(row)
merged = dict(rgb_size=256, depth_size=128, episodes=len(rows), steps=sum(r["steps"] for r in rows), rows=rows)
(mix / "index.json").write_text(json.dumps(merged, indent=1))
print("mix episodes", len(rows))
PY
fi

# 3) act_nostate evals once its training ends (GPU otherwise free).
while kill -0 "$act_nostate_pid" 2>/dev/null; do sleep 30; done
evaluate act_nostate 0060000 $base/eval_seeds.json eval_0060000.json 6 8 yes

# 4) ACT on the mix, then its evals overlapped with the Diffusion training.
train act_dart
(train diffusion_dart) &
diffusion_pid=$!
evaluate act_dart 0060000 $base/eval_seeds.json eval_0060000.json 10 6 yes
evaluate act_dart 0060000 $base/train_seeds.json train_0060000.json 0 6 yes
evaluate act_dart 0040000 $base/eval_seeds.json eval_0040000.json 0 6 no
evaluate act_nostate 0060000 $base/train_seeds.json train_0060000.json 0 6 no
wait "$diffusion_pid"

# 5) Diffusion evals.
evaluate diffusion_dart 0060000 $base/eval_seeds.json eval_0060000.json 10 5 yes
evaluate diffusion_dart 0060000 $base/train_seeds.json train_0060000.json 0 5 yes
evaluate diffusion_dart 0040000 $base/eval_seeds.json eval_0040000.json 0 5 no
echo "queue done $(date -u)"

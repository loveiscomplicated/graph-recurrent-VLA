#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" || ! -f outputs/recurrent/env.sh ]]; then
  echo "Server setup is missing. Run: bash recurrent/setup_server.sh" >&2
  exit 1
fi
source outputs/recurrent/env.sh
export WANDB_MODE="${WANDB_MODE:-disabled}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-30}
if [[ ! "$EVAL_NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_NUM_ENVS must be a positive integer." >&2
  exit 2
fi

GPU_COUNT=$($PY -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
if (( GPU_COUNT < 2 )); then
  echo "Need at least two CUDA GPUs; Torch sees $GPU_COUNT." >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != all ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  if (( ${#GPU_IDS[@]} != GPU_COUNT )); then
    echo "CUDA_VISIBLE_DEVICES does not match Torch's visible GPU count." >&2
    exit 1
  fi
else
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do GPU_IDS+=("$i"); done
fi

mkdir -p outputs/recurrent/exp
RUN_ROOT=$(mktemp -d outputs/recurrent/exp/gpu_smoke_XXXXXX)
DEMO="$RUN_ROOT/smoke_demos.npz"
CONFIG="$RUN_ROOT/smoke.yaml"
INITIAL="$RUN_ROOT/initial.pt"
sed -e "s#^data: .*#data: $DEMO#" \
    -e "s#^out_dir: .*#out_dir: $RUN_ROOT#" \
    recurrent/configs/smoke.yaml > "$CONFIG"

echo "Smoke output: $RUN_ROOT"
CUDA_VISIBLE_DEVICES="" $PY -m recurrent.collect --num-demos 12 --workers 2 --out "$DEMO"
CUDA_VISIBLE_DEVICES="" $PY -m recurrent.initial_states \
  --episodes "$EVAL_NUM_ENVS" --start-seed 2000 --out "$INITIAL"

# Keep the 14-job training shard check, without the launcher's CPU evaluation.
pids=()
for ((i = 0; i < GPU_COUNT; i++)); do
  echo "Launching training shard $i/$GPU_COUNT on CUDA GPU ${GPU_IDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" $PY -m recurrent.run_experiments \
    --config "$CONFIG" --shard-count "$GPU_COUNT" --shard-index "$i" \
    --train-workers 1 --train-only --no-report > "$RUN_ROOT/shard-$i.log" 2>&1 &
  pids+=("$!")
done

status=0
for ((i = 0; i < GPU_COUNT; i++)); do
  if wait "${pids[$i]}"; then
    echo "Shard $i PASS"
  else
    echo "Shard $i FAIL: $RUN_ROOT/shard-$i.log" >&2
    tail -n 25 "$RUN_ROOT/shard-$i.log" >&2
    status=1
  fi
done
if (( status != 0 )); then
  exit 1
fi

shopt -s nullglob
train_logs=("$RUN_ROOT"/logs/train_*.log)
if (( ${#train_logs[@]} != 14 )); then
  echo "Incomplete smoke run: ${#train_logs[@]}/14 train logs" >&2
  exit 1
fi
for log in "${train_logs[@]}"; do
  if ! grep -q 'device cuda' "$log"; then
    echo "Training did not use CUDA: $log" >&2
    exit 1
  fi
done

CKPT="$RUN_ROOT/runs/gru_none_s0/best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing smoke checkpoint: $CKPT" >&2
  exit 1
fi

# Use the same policy and seeds on every GPU, with independent output writers.
mkdir -p "$RUN_ROOT/eval_gpu"
pids=()
for ((i = 0; i < GPU_COUNT; i++)); do
  echo "Launching GPU evaluation $i on CUDA GPU ${GPU_IDS[$i]} ($EVAL_NUM_ENVS envs)"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" $PY -m recurrent.evaluate \
    --ckpt "$CKPT" --sim-backend physx_cuda --num-envs "$EVAL_NUM_ENVS" \
    --initial-states "$INITIAL" --episodes "$EVAL_NUM_ENVS" --start-seed 2000 \
    --horizon 250 --conditions normal --levels none --name "gru_gpu_$i" \
    --out "$RUN_ROOT/eval_gpu/gpu-$i.jsonl" > "$RUN_ROOT/eval-gpu-$i.log" 2>&1 &
  pids+=("$!")
done

status=0
for ((i = 0; i < GPU_COUNT; i++)); do
  if wait "${pids[$i]}"; then
    echo "GPU evaluation $i PASS"
  else
    echo "GPU evaluation $i FAIL: $RUN_ROOT/eval-gpu-$i.log" >&2
    tail -n 25 "$RUN_ROOT/eval-gpu-$i.log" >&2
    status=1
  fi
done
if (( status != 0 )); then
  exit 1
fi

$PY - "$RUN_ROOT" "$EVAL_NUM_ENVS" "${GPU_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = int(sys.argv[2])
devices = sys.argv[3:]
expected = {("normal", "none", seed) for seed in range(2000, 2000 + episodes)}
evaluations = []
for index, device in enumerate(devices):
    path = root / "eval_gpu" / f"gpu-{index}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = {(r["condition"], r["eval_level"], r["episode_seed"]) for r in rows}
    if len(rows) != episodes or keys != expected:
        raise SystemExit(f"GPU {device}: incomplete or duplicate evaluation rows")
    identity = json.loads(path.with_suffix(".identity.json").read_text())
    tracking = json.loads(path.with_name(path.name + ".tracking.json").read_text())
    metadata = json.loads((Path(tracking["experiment_directory"]) / "metadata.json").read_text())
    environment = metadata["environment"]
    if (identity["backend"] != "physx_cuda"
            or environment["sim_backend"] != "physx_cuda"
            or environment["render_backend"] != "none"
            or not environment["device"].startswith("cuda")
            or environment["num_envs"] != episodes
            or metadata["status"] != "finished"):
        raise SystemExit(f"GPU {device}: evaluation did not finish with GPU physics")
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    batches = summary["performance"]["batches"]
    if len(batches) != 1 or batches[0]["seeds"] != list(range(2000, 2000 + episodes)):
        raise SystemExit(f"GPU {device}: expected one fresh batch of {episodes} episodes")
    evaluations.append({"gpu": device, "output": str(path), "performance": summary["performance"]})

result = {"training_jobs": 14, "gpu_count": len(devices), "envs_per_gpu": episodes,
          "evaluation_episodes": episodes * len(devices), "evaluations": evaluations}
(root / "smoke_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"Verified {len(devices)} GPU evaluations, {episodes * len(devices)} episodes")
PY

echo "Multi-GPU smoke PASS: 14 CUDA training jobs, $GPU_COUNT GPU batch evaluations"
echo "Results: $RUN_ROOT"

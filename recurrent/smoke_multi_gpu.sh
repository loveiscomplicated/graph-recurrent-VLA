#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" || ! -f outputs/recurrent/env.sh ]]; then
  echo "Server setup is missing. Run: bash recurrent/setup_server.sh" >&2
  exit 1
fi
source outputs/recurrent/env.sh

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
sed -e "s#^data: .*#data: $DEMO#" \
    -e "s#^out_dir: .*#out_dir: $RUN_ROOT#" \
    recurrent/configs/smoke.yaml > "$CONFIG"

echo "Smoke output: $RUN_ROOT"
$PY -m recurrent.collect --num-demos 12 --out "$DEMO"

pids=()
for ((i = 0; i < GPU_COUNT; i++)); do
  echo "Launching shard $i/$GPU_COUNT on CUDA GPU ${GPU_IDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" $PY -m recurrent.run_experiments \
    --config "$CONFIG" --shard-count "$GPU_COUNT" --shard-index "$i" \
    --train-workers 1 --eval-workers 1 > "$RUN_ROOT/shard-$i.log" 2>&1 &
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

train_logs=("$RUN_ROOT"/logs/train_*.log)
eval_summaries=("$RUN_ROOT"/eval/*.summary.json)
if (( ${#train_logs[@]} != 14 || ${#eval_summaries[@]} != 15 )); then
  echo "Incomplete smoke run: ${#train_logs[@]}/14 train logs, ${#eval_summaries[@]}/15 evaluation summaries" >&2
  exit 1
fi
for log in "${train_logs[@]}"; do
  if ! grep -q 'device cuda' "$log"; then
    echo "Training did not use CUDA: $log" >&2
    exit 1
  fi
done

$PY -m recurrent.report --config "$CONFIG"
echo "Multi-GPU smoke PASS: 14 CUDA training jobs, 15 evaluation summaries"
echo "Results: $RUN_ROOT"

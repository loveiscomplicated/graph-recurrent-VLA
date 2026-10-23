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

EVAL_WORKERS=${EVAL_WORKERS:-1}
if [[ ! "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_WORKERS must be a positive integer." >&2
  exit 2
fi

RUN_ROOT=outputs/recurrent/exp/main
CONFIG=recurrent/configs/experiment.yaml
mkdir -p "$RUN_ROOT"

# collect skips an existing dataset; all shards read the same file.
$PY -m recurrent.collect --num-demos 100
$PY -m recurrent.count_params
$PY -m recurrent.latency

pids=()
for ((i = 0; i < GPU_COUNT; i++)); do
  echo "Launching shard $i/$GPU_COUNT on CUDA GPU ${GPU_IDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" $PY -m recurrent.run_experiments \
    --config "$CONFIG" --shard-count "$GPU_COUNT" --shard-index "$i" \
    --train-workers 1 --eval-workers "$EVAL_WORKERS" > "$RUN_ROOT/shard-$i.log" 2>&1 &
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
  echo "Retry this script after checking failed job logs in $RUN_ROOT/logs/." >&2
  exit 1
fi

$PY - <<'PY'
import sys

from recurrent.common import load_config, resolve
from recurrent.run_experiments import eval_done, expected_rows, experiment_jobs

cfg = load_config("recurrent/configs/experiment.yaml")
exp = resolve(cfg["out_dir"])
jobs = experiment_jobs(cfg)
missing = []
for name, _, level, _ in jobs:
    run = exp / "runs" / name
    out = exp / "eval" / f"{name}.jsonl"
    if not (run / "last.pt").exists() or not (run / "history.json").exists():
        missing.append(f"train {name}")
    if not eval_done(out, expected_rows(cfg, level)):
        missing.append(f"eval {name}")
if cfg["eval"].get("with_expert", True):
    out = exp / "eval" / "expert.jsonl"
    if not eval_done(out, expected_rows(cfg, "none", expert=True)):
        missing.append("eval expert")
if len(jobs) != 84 or missing:
    print(f"Incomplete main run: {len(jobs)}/84 jobs; missing: {', '.join(missing[:20])}", file=sys.stderr)
    sys.exit(1)
print("All 84 training and evaluation jobs are complete; expert evaluation is complete.")
PY

$PY -m recurrent.report --config "$CONFIG"
echo "Main multi-GPU run PASS: $RUN_ROOT/report/report.md"

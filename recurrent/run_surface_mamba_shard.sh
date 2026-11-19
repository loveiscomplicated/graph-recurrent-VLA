#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 SHARD_INDEX SHARD_COUNT" >&2
  exit 2
fi

SHARD_INDEX="$1"
SHARD_COUNT="$2"
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "shard index/count must be nonnegative integers with count >= 1" >&2
  exit 2
fi
if [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
  echo "shard index must be smaller than shard count" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "missing .venv; run recurrent/setup_surface_mamba_server.sh first" >&2
  exit 1
fi

if [ -f outputs/recurrent/surface_env.sh ]; then
  # shellcheck disable=SC1091
  source outputs/recurrent/surface_env.sh
elif [ -f outputs/recurrent/env.sh ]; then
  # shellcheck disable=SC1091
  source outputs/recurrent/env.sh
fi

export PYTHONPATH="$REPOSITORY_ROOT/visual/third_party/tapnet:$REPOSITORY_ROOT/visual/third_party/co_tracker:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SHARD_NAME="$(printf 'shard_%03d' "$SHARD_INDEX")"
WORK_ROOT="/workspace/surface_mamba/$SHARD_NAME"
CONFIG_ROOT="$WORK_ROOT/configs"
LOG_ROOT="$WORK_ROOT/logs"
mkdir -p "$CONFIG_ROOT" "$LOG_ROOT"

STATUS_PATH="$WORK_ROOT/status.json"
write_status() {
  local state="$1"
  local stage="$2"
  "$PYTHON" - "$STATUS_PATH" "$state" "$stage" "$SHARD_INDEX" "$SHARD_COUNT" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

path, state, stage, index, count = sys.argv[1:]
Path(path).write_text(json.dumps({
    "state": state,
    "stage": stage,
    "shard": {"index": int(index), "count": int(count)},
    "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}

failed() {
  write_status "FAILED" "${CURRENT_STAGE:-startup}"
}
trap failed ERR

for required in \
  /workspace/surface_mamba/excluded_and_evaluation_seeds.json \
  /workspace/surface_mamba/checkpoints/surface_tracker_best.pt \
  /workspace/surface_mamba/checkpoints/surface_gnn_best.pt \
  visual/checkpoints/tapnextpp/tapnextpp_256_official.pt; do
  if [ ! -f "$required" ]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

"$PYTHON" - "$SHARD_INDEX" "$SHARD_COUNT" "$CONFIG_ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

index = int(sys.argv[1])
count = int(sys.argv[2])
output = Path(sys.argv[3])
root = Path("recurrent/configs/surface_mamba")
name = f"shard_{index:03d}"
work = f"/workspace/surface_mamba/{name}"

def load(name):
    return yaml.safe_load((root / name).read_text())

collect = load("collect_distributed.yaml")
collect["output"] = f"{work}/raw"
collect["shard"] = {"index": index, "count": count}

export = load("export_distributed.yaml")
export["raw_root"] = f"{work}/raw"
export["output"] = f"{work}/state36"

package = load("package_distributed.yaml")
package["raw_root"] = f"{work}/raw"
package["derived_root"] = f"{work}/state36"
package["output"] = f"{work}/package"
package["experiments_dir"] = f"{work}/experiments"

for filename, payload in (
    ("collect.yaml", collect),
    ("export.yaml", export),
    ("package.yaml", package),
):
    (output / filename).write_text(yaml.safe_dump(payload, sort_keys=False))
PY

CURRENT_STAGE="contracts"
write_status "RUNNING" "$CURRENT_STAGE"
"$PYTHON" -m unittest \
  tests.test_surface_mamba_pipeline \
  tests.test_surface_mamba_distributed -v \
  2>&1 | tee "$LOG_ROOT/contracts.log"

CURRENT_STAGE="collection"
write_status "RUNNING" "$CURRENT_STAGE"
"$PYTHON" main.py --config "$CONFIG_ROOT/collect.yaml" \
  2>&1 | tee "$LOG_ROOT/collection.log"

CURRENT_STAGE="state36_export"
write_status "RUNNING" "$CURRENT_STAGE"
"$PYTHON" main.py --config "$CONFIG_ROOT/export.yaml" \
  2>&1 | tee "$LOG_ROOT/state36_export.log"

CURRENT_STAGE="package_and_publish"
write_status "RUNNING" "$CURRENT_STAGE"
"$PYTHON" main.py --config "$CONFIG_ROOT/package.yaml" \
  2>&1 | tee "$LOG_ROOT/package_and_publish.log"

trap - ERR
write_status "COMPLETE" "published"
echo "Surface-Mamba $SHARD_NAME COMPLETE"

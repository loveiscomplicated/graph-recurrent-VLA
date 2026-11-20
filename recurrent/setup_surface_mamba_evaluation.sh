#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "FAIL missing Python environment: $PYTHON" >&2
  exit 1
fi

uv pip install --python "$PYTHON" \
  -r requirements.txt \
  -r requirements-tracking.txt \
  -r recurrent/requirements.txt \
  -r recurrent/requirements_surface_mamba.txt

TRACKER_CHECKPOINT="/workspace/surface_mamba/checkpoints/surface_tracker_best.pt"
GNN_CHECKPOINT="/workspace/surface_mamba/checkpoints/surface_gnn_best.pt"
EVALUATION_SEEDS="/workspace/surface_mamba/evaluation_seeds.json"
TAP_CHECKPOINT="$REPOSITORY_ROOT/visual/checkpoints/tapnextpp/tapnextpp_256_official.pt"

verify_sha256() {
  local path="$1"
  local expected="$2"
  if [ ! -f "$path" ]; then
    echo "FAIL missing required artifact: $path" >&2
    exit 1
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "FAIL artifact SHA-256 differs: $path" >&2
    exit 1
  fi
}

verify_sha256 \
  "$TRACKER_CHECKPOINT" \
  "f98c4ef1c77d25fa059e9f9cf048fb1f190bcf74302a0fe1367bfe1341236afc"
verify_sha256 \
  "$GNN_CHECKPOINT" \
  "0f4d9d6fd6ce615125698e243b259fad92557a3f5c5848aafe24a85277720241"
verify_sha256 \
  "$EVALUATION_SEEDS" \
  "7fe07e3645117a7cd897751fd4a4917f05bd648175a0db16219b6d95b7013765"

mkdir -p "$(dirname "$TAP_CHECKPOINT")"
if [ ! -f "$TAP_CHECKPOINT" ] || \
   [ "$(sha256sum "$TAP_CHECKPOINT" | awk '{print $1}')" != \
     "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da" ]; then
  curl --fail --location --retry 5 --retry-all-errors \
    https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
    -o "$TAP_CHECKPOINT.partial"
  verify_sha256 \
    "$TAP_CHECKPOINT.partial" \
    "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"
  mv "$TAP_CHECKPOINT.partial" "$TAP_CHECKPOINT"
fi
verify_sha256 \
  "$TAP_CHECKPOINT" \
  "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"

export PYTHONPATH="$REPOSITORY_ROOT/visual/third_party/tapnet:$REPOSITORY_ROOT/visual/third_party/co_tracker:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"

"$PYTHON" examples/check_gpu_observations.py
"$PYTHON" -m recurrent.check_surface_mamba_frontend

echo "surface closed-loop evaluation setup PASS"

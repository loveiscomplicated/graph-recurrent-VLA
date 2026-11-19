#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPOSITORY_ROOT"

bash recurrent/setup_server.sh

PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install -q -r recurrent/requirements_surface_mamba.txt
else
  uv pip install -q --python "$PYTHON" -r recurrent/requirements_surface_mamba.txt
fi

export PYTHONPATH="$REPOSITORY_ROOT/visual/third_party/tapnet:$REPOSITORY_ROOT/visual/third_party/co_tracker:${PYTHONPATH:-}"
"$PYTHON" - <<'PY'
import cv2
import einops
import torch
import torchvision
from tapnet.tapnextpp.votsp2026.model import TAPNextPP
from graph.surface_context_frontend import SurfaceContextFrontend

assert torch.cuda.is_available(), "Surface-Mamba collection requires CUDA"
print("surface imports PASS", torch.__version__, torchvision.__version__, cv2.__version__)
PY

SURFACE_ENV="$REPOSITORY_ROOT/outputs/recurrent/surface_env.sh"
GPU_ICD=""
for candidate in \
  /etc/vulkan/icd.d/nvidia_icd.json \
  /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [ -f "$candidate" ] && \
     VK_ICD_FILENAMES="$candidate" "$PYTHON" examples/check_gpu_observations.py; then
    GPU_ICD="$candidate"
    break
  fi
done
if [ -z "$GPU_ICD" ] && env -u VK_ICD_FILENAMES "$PYTHON" examples/check_gpu_observations.py; then
  GPU_ICD="default"
fi
if [ -z "$GPU_ICD" ]; then
  echo "FAIL native CUDA RGB-D verification did not pass" >&2
  exit 1
fi
if [ "$GPU_ICD" = "default" ]; then
  echo "unset VK_ICD_FILENAMES" > "$SURFACE_ENV"
else
  echo "export VK_ICD_FILENAMES=$GPU_ICD" > "$SURFACE_ENV"
fi

# shellcheck disable=SC1090
source "$SURFACE_ENV"
"$PYTHON" -m recurrent.check_surface_mamba_frontend

echo "surface setup PASS"

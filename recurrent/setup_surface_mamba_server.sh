#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv || pip install -q uv
    uv venv --python 3.11 .venv
  fi
fi

install_packages() {
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install -q "$@"
  else
    uv pip install -q --python "$PYTHON" "$@"
  fi
}

"$PYTHON" -m pip --version >/dev/null 2>&1 && \
  "$PYTHON" -m pip install -q --upgrade pip
install_packages -r requirements.txt
install_packages -r requirements-tracking.txt
install_packages -r recurrent/requirements.txt
install_packages -r recurrent/requirements_surface_mamba.txt

if ! "$PYTHON" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | awk '{print $3}' || true)
  TAG=$("$PYTHON" - "$DRIVER_CUDA" <<'PY'
import sys
version = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "0"
major, minor = (int(part) for part in (version.split(".") + ["0"])[:2])
for tag, required in (("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)),
                      ("cu124", (12, 4)), ("cu121", (12, 1)), ("cu118", (11, 8))):
    if (major, minor) >= required:
        print(tag)
        break
PY
)
  if [ -z "$TAG" ]; then
    echo "FAIL no compatible CUDA PyTorch index for driver CUDA $DRIVER_CUDA" >&2
    exit 1
  fi
  install_packages --force-reinstall torch torchvision \
    --index-url "https://download.pytorch.org/whl/$TAG"
fi

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
  apt-get update -q >/dev/null
  apt-get install -y -q \
    libglvnd0 \
    libvulkan1 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrandr2 \
    >/dev/null
fi

if [ ! -f /root/.secrets/hf.env ]; then
  echo "FAIL /root/.secrets/hf.env is required for shard publication" >&2
  exit 1
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

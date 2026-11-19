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

echo "surface setup PASS"

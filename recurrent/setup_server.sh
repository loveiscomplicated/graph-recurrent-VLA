#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/recurrent

VPY=.venv/bin/python
if [ ! -x "$VPY" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
  else
    command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv || pip install -q uv
    uv venv --python 3.11 .venv
  fi
fi

pipi() {
  if "$VPY" -m pip --version >/dev/null 2>&1; then
    "$VPY" -m pip install -q "$@"
  else
    uv pip install -q --python "$VPY" "$@"
  fi
}

"$VPY" -m pip --version >/dev/null 2>&1 && "$VPY" -m pip install -q --upgrade pip
pipi -r requirements.txt
[ -f requirements-tracking.txt ] && pipi -r requirements-tracking.txt
pipi -r recurrent/requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$VPY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | awk '{print $3}' || true)
    TAG=$("$VPY" - "$DRIVER_CUDA" <<'EOF'
import sys
v = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "0"
major, minor = (int(x) for x in (v.split(".") + ["0"])[:2])
for tag, need in (("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1)), ("cu118", (11, 8))):
    if (major, minor) >= need:
        print(tag)
        break
EOF
)
    if [ -n "$TAG" ]; then
      echo "torch cuda unavailable, driver cuda $DRIVER_CUDA, reinstall torch from $TAG"
      pipi --force-reinstall torch --index-url "https://download.pytorch.org/whl/$TAG"
    fi
  fi
fi

try_vk() {
  env ${1:+VK_ICD_FILENAMES="$1"} CUDA_VISIBLE_DEVICES="" "$VPY" -c "from recurrent.common import make_env; e = make_env(); e.reset(seed=0); e.close()" >/dev/null 2>&1
}

FOUND=""
for c in "${VK_ICD_FILENAMES:-}" /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/lvp_icd.json; do
  if [ -n "$c" ] && [ -f "$c" ] && try_vk "$c"; then FOUND="$c"; break; fi
done
if [ -z "$FOUND" ] && try_vk ""; then FOUND="default"; fi
if [ -z "$FOUND" ] && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
  apt-get update -q >/dev/null && apt-get install -y -q libvulkan1 mesa-vulkan-drivers >/dev/null
  for c in /usr/share/vulkan/icd.d/lvp_icd.json; do
    if [ -f "$c" ] && try_vk "$c"; then FOUND="$c"; break; fi
  done
fi
if [ -z "$FOUND" ]; then
  echo "FAIL simulator could not start; set VK_ICD_FILENAMES to a working Vulkan ICD"
  exit 1
fi
if [ "$FOUND" = default ]; then
  : > outputs/recurrent/env.sh
else
  echo "export VK_ICD_FILENAMES=$FOUND" > outputs/recurrent/env.sh
fi
echo "vulkan $FOUND"

"$VPY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
EOF
if [ -f /root/.secrets/wandb.env ] || [ -n "${WANDB_ENV_FILE:-}" ] || [ -n "${WANDB_API_KEY:-}" ]; then
  echo "wandb credentials found"
else
  echo "wandb credentials missing, runs are recorded offline under experiments/"
fi
. outputs/recurrent/env.sh
"$VPY" -m unittest discover -s recurrent/tests -t . 2>&1 | tail -3
echo "setup PASS"

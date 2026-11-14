#!/usr/bin/env bash
# Create an isolated project environment for the pinned SAM2 tracker.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

if [[ -e .venv ]]; then
  echo 'Refusing to replace the existing project .venv.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
command -v python3.11 >/dev/null || { echo 'Python 3.11 is required.' >&2; exit 1; }

ensure_source() {
  local path="$1" url="$2" revision="$3"
  if [[ ! -e "$path" ]]; then
    git clone "$url" "$path"
    git -C "$path" checkout --detach "$revision"
  fi
  if [[ ! -d "$path/.git" ]] || [[ "$(git -C "$path" rev-parse HEAD)" != "$revision" ]]; then
    echo "Existing source $path is missing or not at required revision $revision." >&2
    exit 1
  fi
  if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
    echo "Existing source $path contains local changes; inspect them before setup." >&2
    exit 1
  fi
}

mkdir -p visual/third_party
ensure_source visual/third_party/sam2 https://github.com/facebookresearch/sam2.git 2b90b9f5ceec907a1c18123530e92e794ad901a4
ensure_source visual/third_party/sam2_mlx https://github.com/avbiswas/sam2-mlx.git fbeafa7531be454b8f42d89c13d1e90024a63f6b
python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r visual/sam2_mlx/requirements_macos.txt

# Use relative source paths so the environment never references another checkout.
.venv/bin/python - <<'PY'
from pathlib import Path
import os
import site

root = Path.cwd()
site_packages = Path(site.getsitepackages()[0])
sources = [root / 'visual/third_party/sam2', root / 'visual/third_party/sam2_mlx/src']
(site_packages / 'roarm_sam2_sources.pth').write_text(
    ''.join(os.path.relpath(path, site_packages) + '\n' for path in sources),
    encoding='utf-8',
)
PY

echo 'Created .venv. See docs/sam2_mlx.md for checkpoint restoration and execution.'

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Install uv >=0.9 using https://docs.astral.sh/uv/getting-started/installation/ then rerun bash setup.sh.' >&2
    exit 1
fi
if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
    echo 'This environment supports Linux x86_64 with an NVIDIA GPU.' >&2
    exit 1
fi
export UV_CACHE_DIR="${VISION_CACHE:-$PWD/cache}/uv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_PYTHON_INSTALL_DIR="$UV_CACHE_DIR/python"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
# uv selects/downloads Python 3.12 without altering global Python packages.
if ! uv python find 3.12 >/dev/null 2>&1; then
    uv python install 3.12 || { echo 'Python 3.12 installation failed; check network access or provide Python 3.12 on PATH.' >&2; exit 1; }
fi
if [[ -f .venv/pyvenv.cfg ]] && grep -Eq 'include-system-site-packages = true' .venv/pyvenv.cfg; then
    echo 'Existing .venv shares system packages. Move it aside manually, then rerun setup.sh.' >&2
    exit 1
fi
if [[ -e .venv ]] && [[ ! -f .venv/pyvenv.cfg || ! -x .venv/bin/python ]]; then
    echo 'Existing .venv is incomplete. Move it aside manually, then rerun setup.sh.' >&2
    exit 1
fi
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo 'Existing .venv does not use Python 3.12. Move it aside manually, then rerun setup.sh.' >&2
    exit 1
fi
uv sync --locked || { echo 'Dependency synchronization failed. Check network/TLS access to PyPI, download.pytorch.org and GitHub; git is required for SAM2. No GPU runtime validation was performed.' >&2; exit 1; }
source ./env.sh
python scripts/check_install.py
printf '\nEnvironment ready. Run: source env.sh\nThen: python scripts/smoke_test.py --out results/step1\n'

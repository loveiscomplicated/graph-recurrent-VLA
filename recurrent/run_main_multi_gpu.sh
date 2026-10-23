#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" || ! -f outputs/recurrent/env.sh ]]; then
  echo "Server setup is missing. Run: bash recurrent/setup_server.sh" >&2
  exit 1
fi
source outputs/recurrent/env.sh
# The Python launcher owns flock and separate shard process groups, including
# termination/reaping and inherited writer locks after an abrupt launcher exit.
exec "$PY" -m recurrent.main_multi_gpu --config "${MAIN_CONFIG:-recurrent/configs/experiment.yaml}"

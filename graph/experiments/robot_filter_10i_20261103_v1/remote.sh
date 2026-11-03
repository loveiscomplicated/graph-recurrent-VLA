#!/bin/bash
set -euo pipefail
TASK10I_FILTER_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source
export TASK10I_REPO="$PWD"
export PYTHONPATH="$TASK10I_REPO:${PYTHONPATH:-}"
export TASK10I_ROOT=/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2
export TASK10I_LATENCY_CODE=/workspace/task10i_graph_latency_20261103_v2/code
export TASK10I_REFERENCE_CODE=/workspace/task10i_graph_latency_20261103_v2/reference_code
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python "$TASK10I_FILTER_CODE/${TASK10I_FILTER_ENTRY:-run.py}" "$@"

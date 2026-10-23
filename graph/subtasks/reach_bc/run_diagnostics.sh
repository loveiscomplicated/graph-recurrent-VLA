#!/bin/bash
# Foreground supervisor job: never automatically restart or overwrite results.
set -euo pipefail
cd /Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA
REACH_PY=/workspace/reach-bc-55512455/.venv/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode reference
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --repeat

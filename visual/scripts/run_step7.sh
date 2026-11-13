#!/usr/bin/env bash
# Step 7: tracker 교체 비교. venv 3개를 번갈아 쓰며 순차 실행 (GPU 타이밍 간섭 방지) → eval_step7.py 로 표/그림.
#   bash scripts/run_step7.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME=${VISION_CACHE:-$PWD/cache}/hf PIP_CACHE_DIR=${VISION_CACHE:-$PWD/cache}/pip TORCH_HOME=${VISION_CACHE:-$PWD/cache}/torch
mkdir -p logs/step7
run() {  # venv backend model mode [compile: default|on|off]
  local c=${5:-default} log=logs/step7/$2_$3_$4_${5:-default}.log
  echo "=== $2 $3 $4 compile=$c ==="
  ( source ${VISION_ENVS:-$HOME/envs}/$1/bin/activate && timeout 900 python scripts/track_compare.py --backend $2 --model $3 --mode $4 --compile $c ) \
    > $log 2>&1 || echo "FAIL ($?) → $log"
  grep -E "^(ep0|\[)" $log
}
run vision       sam2         small      joint
run vision       sam2         small      single
run vision       sam2         base_plus  joint
run samurai      samurai      small      single
run samurai      samurai      base_plus  single
run efficienttam efficienttam s          joint
run efficienttam efficienttam ti         joint
run efficienttam efficienttam s_512x512  joint
run efficienttam efficienttam ti_512x512 joint
run efficienttam efficienttam s          single
# 공정 비교: image encoder torch.compile 조건 맞추기 (EfficientTAM 기본 ON, SAM2 기본 OFF)
run efficienttam efficienttam s          joint off
run efficienttam efficienttam ti         joint off
run vision       sam2         small      joint on
[ -f scripts/eval_step7.py ] && ( source ${VISION_ENVS:-$HOME/envs}/vision/bin/activate && python scripts/eval_step7.py )

#!/usr/bin/env bash
# E0 수집 전체 (순차, ~25 분). source env.sh 후 실행.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v1
timeout 30m python scripts/collect_v1.py --split dev  --seeds $(seq 100 109) --out data/v1/dev > logs/v1/collect_dev.log 2>&1
timeout 60m python scripts/collect_v1.py --split eval --seeds $(seq 200 239) --out data/v1/eval > logs/v1/collect_eval.log 2>&1
timeout 30m python scripts/collect_v1.py --split dev  --seeds $(seq 100 109) --perturb box_move --out data/v1/dev_box_move > logs/v1/collect_dev_box_move.log 2>&1
timeout 60m python scripts/collect_v1.py --split eval --seeds $(seq 200 239) --perturb box_move --out data/v1/eval_box_move > logs/v1/collect_eval_box_move.log 2>&1
timeout 60m python scripts/collect_v1.py --split eval --seeds $(seq 200 239) --perturb lighting --out data/v1/eval_lighting > logs/v1/collect_eval_lighting.log 2>&1
echo COLLECT_DONE

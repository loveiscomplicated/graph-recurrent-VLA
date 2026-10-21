#!/usr/bin/env bash
# E2: 단독 GPU 타이밍 ablation. GPU 에 다른 compute 프로세스가 없을 때까지 기다렸다가 순차 실행.
cd "$(dirname "$0")/.."; source env.sh; mkdir -p logs/v1 results/v1/E2
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 120; done
echo "$(date +%T) GPU idle, start" >> logs/v1/e2_all.log
EPS="ep200 ep201 ep202 ep203 ep204 ep205 ep206 ep207 ep208 ep209"
R=results/v1/E1/runs
run() { v=$1; ref=$2; shift 2; timeout 120m python scripts/e2_stream.py --set eval --variant $v --ref $R/$ref/eval --eps ${EPS} "$@" > logs/v1/e2_$v.log 2>&1; echo "$(date +%T) done $v" >> logs/v1/e2_all.log; }
run stream sam2s_none
run stream+compile sam2s_c_none
run stream+compile+gpupc sam2s_c_none
run stream+compile+gpupc+gpupre sam2s_c_none
run stream+compile+gpupc+gpupre+prune sam2s_c_none
run stream+compile+gpupc+gpupre+prune+bf16w sam2s_c_none
run stream+compile+gpupc+gpupre+prune+img768 sam2s_c_none
run stream+compile+gpupc+gpupre+prune+img512 sam2s_c_none
run stream+compile+gpupc+gpupre+prune+cam384 sam2s_c_none
run stream+compile+gpupc+gpupre+prune+cam320 sam2s_c_none
echo E2_DONE >> logs/v1/e2_all.log

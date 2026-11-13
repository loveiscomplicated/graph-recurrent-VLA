#!/usr/bin/env bash
# E9: eval 40 ep export (v1 기본 tracker = SAM2.1-small compile + conf_e5), CPU 4 프로세스
cd "$(dirname "$0")/.."; source env.sh
RUN=results/v1/E1/runs/sam2s_c_conf_e5/eval
EPS=($(ls data/v1/eval | grep -o 'ep[0-9]*' | sort -u))
for i in 0 1 2 3; do
  sub=(); for j in "${!EPS[@]}"; do [ $((j % 4)) -eq $i ] && sub+=(${EPS[$j]}); done
  OMP_NUM_THREADS=4 timeout 300m python scripts/e9_export.py --run $RUN --set eval --eps ${sub[@]} > logs/v1/e9_export_$i.log 2>&1 &
done
wait; echo E9_DONE

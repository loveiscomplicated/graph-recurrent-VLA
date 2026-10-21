#!/usr/bin/env bash
# E5: 성공률 sweep (eval seed 200-239, expert 정책 계획을 state source 로)
set -u
cd "$(dirname "$0")/.."; source env.sh; mkdir -p logs/v1 results/v1/E5
r() { name=$1; shift; timeout 300m python scripts/e5_loop.py --name $name "$@" > logs/v1/e5_$name.log 2>&1; echo "$(date +%T) done $name $(grep SUMMARY logs/v1/e5_$name.log)"; }
r gt --cond gt &
r pos1 --cond noise --pos-mm 1 &
r pos2 --cond noise --pos-mm 2 &
r pos5 --cond noise --pos-mm 5 &
wait
r pos10 --cond noise --pos-mm 10 &
r rot1 --cond noise --rot-deg 1 &
r rot3 --cond noise --rot-deg 3 &
r rot5 --cond noise --rot-deg 5 &
wait
r pos0.5 --cond noise --pos-mm 0.5 &
r pos1_rot1 --cond noise --pos-mm 1 --rot-deg 1 &
r gt_boxmove --cond gt --box-move &
wait
echo E5_NOISE_DONE

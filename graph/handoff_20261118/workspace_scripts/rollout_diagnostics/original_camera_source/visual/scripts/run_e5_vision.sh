#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source env.sh
until grep -q E5_NOISE_DONE logs/v1/e5_all.log; do sleep 60; done
r() { name=$1; shift; timeout 600m python scripts/e5_loop.py --name $name "$@" > logs/v1/e5_$name.log 2>&1; echo "$(date +%T) done $name $(grep SUMMARY logs/v1/e5_$name.log)"; }
r vision_none --cond vision --vcfg none &
r vision_v1 --cond vision --vcfg v1 &
r vision_v1_boxmove --cond vision --vcfg v1 --box-move &
wait; echo E5_VISION_DONE

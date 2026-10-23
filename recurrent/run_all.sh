#!/bin/sh
set -e
cd "$(dirname "$0")/.."
[ -f outputs/recurrent/env.sh ] && . outputs/recurrent/env.sh
PY=.venv/bin/python
MODE=${1:-main}
case "$MODE" in
  smoke)
    $PY -m recurrent.collect --num-demos 12 --out outputs/recurrent/data/smoke_demos.npz
    CONFIG=smoke.yaml
    ;;
  official)
    $PY -m recurrent.official_demos --download
    CONFIG=official.yaml
    ;;
  main)
    $PY -m recurrent.collect --num-demos 100
    CONFIG=experiment.yaml
    ;;
  *)
    echo "usage: sh recurrent/run_all.sh [smoke|main|official]"
    exit 2
    ;;
esac
$PY -m recurrent.count_params
$PY -m recurrent.latency
$PY -m recurrent.run_experiments --config "$CONFIG"

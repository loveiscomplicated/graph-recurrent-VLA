#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }
EV="--no-move --cands mamba_step@step,mlp_step@step --train-seeds 0-4 --episodes 30 --env-offset 20 --workers 8"
K="--kin --kin-cfg /workspace/kin_best.json"
COMMON="--types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --splits train --data-dir /workspace/data_kin $K"

for c in heavy over long; do
  [ -e /workspace/exp2/runs_r6_mixf_$c ] && { echo "FAIL runs_r6_mixf_$c exists"; exit 1; }
done

t "train heavy (more weight on graph)"
python exp2_train.py run $COMMON --mix 0,0.5,1,1,1 --out /workspace/exp2/runs_r6_mixf_heavy
t "train over (alpha up to 1.5)"
python exp2_train.py run $COMMON --mix 0,0.25,0.5,0.75,1,1.25,1.5 --out /workspace/exp2/runs_r6_mixf_over
t "train long (6000 steps)"
python exp2_train.py run $COMMON --mix 0,0.25,0.5,0.75,1 --steps 6000 --out /workspace/exp2/runs_r6_mixf_long

for c in heavy over long; do
  O=/workspace/eval/r6_mixf_${c}_graph
  t "eval $c"
  mkdir -p $O && cp /workspace/eval/diag/env_seeds.json $O/
  python exp2_eval.py run $EV --runs /workspace/exp2/runs_r6_mixf_$c --out $O
done

t "table"
python tci.py /workspace/eval/r6_mix_filter_graph /workspace/eval/r6_mixf_heavy_graph /workspace/eval/r6_mixf_over_graph \
  /workspace/eval/r6_mixf_long_graph | tee /workspace/fix3_table.txt
t "DONE"

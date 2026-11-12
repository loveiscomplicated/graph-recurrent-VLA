#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }
EV="--no-move --cands mamba_step@step,mlp_step@step --train-seeds 0-4 --episodes 30 --env-offset 20 --workers 8"
MIX=0,0.25,0.5,0.75,1

t "wait fix"
while pgrep -f "fix.sh" > /dev/null; do sleep 30; done
cp exp2_train_mix.py exp2_train.py

for c in r6_mix_none r6_mix_filter; do
  [ -e /workspace/exp2/runs_$c ] && { echo "FAIL runs_$c exists"; exit 1; }
done

t "train mix"
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --mix $MIX --splits train \
  --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_mix_none
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --mix $MIX --splits train \
  --kin --kin-cfg /workspace/kin_best.json --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_mix_filter

ev() {
  mkdir -p $2 && cp /workspace/eval/diag/env_seeds.json $2/
  python exp2_eval.py run $EV $3 --runs $1 --out $2
}
t "eval mix_none graph"
ev /workspace/exp2/runs_r6_mix_none /workspace/eval/r6_mix_none_graph ""
t "eval mix_filter graph"
ev /workspace/exp2/runs_r6_mix_filter /workspace/eval/r6_mix_filter_graph ""
t "eval gt_none graph"
ev /workspace/exp2/runs_r6_gt_none /workspace/eval/r6_gt_none_graph ""

t "table"
python tci.py /workspace/eval/r6_mix_none_graph /workspace/eval/r6_mix_filter_graph /workspace/eval/r6_gt_none_graph \
  /workspace/eval/r6_gt_none_oracle /workspace/eval/r6_graph_filter_graph /workspace/eval/r6_graph_none_graph | tee /workspace/fix2_table.txt
t "DONE"

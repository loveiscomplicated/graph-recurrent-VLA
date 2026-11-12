#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }
EV="--no-move --cands mamba_step@step,mlp_step@step --train-seeds 0-4 --episodes 30 --env-offset 20 --workers 8"

t "wait gtnn"
while pgrep -f "gtnn.sh|eval/gtin_oracle|fac_nk_gt_oracle" > /dev/null; do sleep 30; done

for c in r6_gt_none r6_graph_filter r6_graph_none; do
  [ -e /workspace/exp2/runs_$c ] && { echo "FAIL runs_$c exists"; exit 1; }
done

t "train"
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --input gt --splits train \
  --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_gt_none
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --input transformer --splits train \
  --kin --kin-cfg /workspace/kin_best.json --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_graph_filter
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --input transformer --splits train \
  --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_graph_none

ev() {
  mkdir -p $2 && cp /workspace/eval/diag/env_seeds.json $2/
  python exp2_eval.py run $EV $3 --runs $1 --out $2
}
t "eval gt_none oracle"
ev /workspace/exp2/runs_r6_gt_none /workspace/eval/r6_gt_none_oracle --oracle
t "eval graph_filter graph"
ev /workspace/exp2/runs_r6_graph_filter /workspace/eval/r6_graph_filter_graph ""
t "eval graph_none graph"
ev /workspace/exp2/runs_r6_graph_none /workspace/eval/r6_graph_none_graph ""

t "table"
python tci.py /workspace/eval/fac_nk_gt_oracle /workspace/eval/r6_gt_none_oracle /workspace/eval/r6_graph_filter_graph \
  /workspace/eval/r6_graph_none_graph /workspace/eval/diag /workspace/eval/gtin_oracle | tee /workspace/fix_table.txt
t "DONE"

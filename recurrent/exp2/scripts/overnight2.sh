#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/exp2/runs_lock
E=/workspace/eval/lock_s2
T=mlp:step,gru:step,mamba:step
C=mlp_step@step,gru_step@step,mamba_step@step
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }

t "1 lock tune"
python lock_tune.py
cat /workspace/kin_lock.json; echo

t "2 train"
python exp2_train.py run --types $T --seeds 0-9 --jobs 10 --input transformer --splits train \
  --kin --kin-cfg /workspace/kin_lock.json --data-dir /workspace/data_kin --out $R

t "3 eval"
python exp2_eval.py run --no-move --cands $C --train-seeds 0-9 --episodes 30 --env-offset 20 --workers 8 --runs $R --out $E
python exp2_eval.py report --out $E
python fail_breakdown.py $E

cd /workspace
tar czf lock_results.tgz kin_lock.json eval/lock_s2/report.json overnight2.log
t "DONE"

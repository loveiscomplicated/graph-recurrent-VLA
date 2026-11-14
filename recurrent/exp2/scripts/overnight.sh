#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/exp2/runs_final
S1=/workspace/eval/final_s1
S2=/workspace/eval/final_s2
V=/workspace/videos_final
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }

t "1 hole correction"
python hole_corr.py
cat /workspace/kin_final.json; echo

t "2 train 8 types x 10 seeds"
python exp2_train.py run --types all --seeds 0-9 --jobs 10 --input transformer --splits train \
  --kin --kin-cfg /workspace/kin_final.json --data-dir /workspace/data_kin --out $R
ls $R | wc -l

t "3 stage1 eval 20 candidates"
python exp2_eval.py run --no-move --cands all --train-seeds 0-4 --episodes 20 --workers 8 --runs $R --out $S1
python exp2_eval.py report --out $S1

C=$(python pick_top.py $S1/report.json 3)
t "4 stage2 eval top3: $C"
python exp2_eval.py run --no-move --cands $C --train-seeds 0-9 --episodes 30 --env-offset 20 --workers 8 --runs $R --out $S2
python exp2_eval.py report --out $S2

B=$(python pick_top.py $S2/report.json 1)
t "5 videos best $B"
python exp2_eval.py video --no-move --cands expert,$B --train-seeds 0 --runs $R --out $V || echo "video failed"
python exp2_eval.py video --no-move --cands expert,$B --train-seeds 0 --env-seed 700058 --runs $R --out $V || echo "video failed"

t "6 pack"
cd /workspace
tar czf final_results.tgz kin_final.json eval/final_s1/report.json eval/final_s2/report.json videos_final overnight.log
t "DONE best $B"

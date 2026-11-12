#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/exp2/runs_nk_gt
O=/workspace/eval/fac_nk_gt_oracle
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }

if [ -e $R ] || [ -e $O ]; then echo "FAIL $R or $O exists"; exit 1; fi
t "train gt input, no filter"
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --input gt --splits train \
  --data-dir /workspace/data_kin --out $R
for m in mamba_step mlp_step; do for s in 0 1 2 3 4; do
  [ -f $R/${m}_s$s/last.pt ] || { echo "FAIL missing $R/${m}_s$s/last.pt"; exit 1; }
done; done

t "eval oracle"
mkdir -p $O && cp /workspace/eval/diag/env_seeds.json $O/
python exp2_eval.py run --no-move --oracle --cands mamba_step@step,mlp_step@step --train-seeds 0-4 --episodes 30 \
  --env-offset 20 --workers 8 --runs $R --out $O
python - <<'PY'
import json, glob
import numpy as np
from scipy import stats
rs = [json.loads(l) for f in glob.glob("/workspace/eval/fac_nk_gt_oracle/results_*.jsonl") for l in open(f)]
for c in sorted({r["cand"] for r in rs}):
    err = sum("error" in r for r in rs if r["cand"] == c)
    by = {}
    for r in rs:
        if r["cand"] == c and "none" in r:
            by.setdefault(r["train_seed"], []).append(r["none"]["success"])
    p = np.array([np.mean(v) for v in by.values()]) * 100
    h = stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p))
    eps = sum(len(v) for v in by.values())
    print(f"GT/none/GT {c} {p.mean():.1f}+-{h:.1f} seeds {len(p)} eps {eps} errors {err} {'OK' if eps == 150 and err == 0 else 'INCOMPLETE'}")
PY
t "DONE"

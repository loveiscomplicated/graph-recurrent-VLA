#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t() { echo "=== $(date '+%m-%d %H:%M') $*"; }
E=/workspace/exp2
NEW="--no-move --env-offset 50 --episodes 100 --workers 8"

for o in hold_single hold_ens; do
  [ -e /workspace/eval/$o ] && { echo "FAIL /workspace/eval/$o exists"; exit 1; }
done

python - <<'PY'
import json
E = "/workspace/exp2"
base = f"{E}/runs_r6_mix_filter"
var = [f"{E}/runs_r6_mix_filter", f"{E}/runs_r6_mixf_heavy", f"{E}/runs_r6_mixf_over", f"{E}/runs_r6_mixf_long"]
spec = {
    "ens_mamba5": [[base, "mamba_step", s] for s in range(5)],
    "ens_mlp5": [[base, "mlp_step", s] for s in range(5)],
    "ens_both10": [[base, m, s] for m in ("mamba_step", "mlp_step") for s in range(5)],
    "ens_mamba20": [[d, "mamba_step", s] for d in var for s in range(5)],
}
json.dump(spec, open("/workspace/ens.json", "w"), indent=1)
print("ens", {k: len(v) for k, v in spec.items()})
PY

t "single models on new scenes"
python exp2_eval.py run $NEW --cands mamba_step@step --train-seeds 0-4 --runs $E/runs_r6_mix_filter --out /workspace/eval/hold_single
t "ensembles on new scenes"
mkdir -p /workspace/eval/hold_ens && cp /workspace/eval/hold_single/env_seeds.json /workspace/eval/hold_ens/
python exp2_eval.py run $NEW --cands ens_mamba5@step,ens_mlp5@step,ens_both10@step,ens_mamba20@step --train-seeds 0 \
  --ens /workspace/ens.json --out /workspace/eval/hold_ens

t "table"
python - <<'PY' | tee /workspace/fix4_table.txt
import glob, json
import numpy as np
from scipy import stats
def rows(d):
    return [json.loads(l) for f in glob.glob(f"/workspace/eval/{d}/results_*.jsonl") for l in open(f)]
r = [x for x in rows("hold_single") if "none" in x]
by = {}
for x in r:
    by.setdefault(x["train_seed"], []).append(x["none"]["success"])
for s in sorted(by):
    print(f"single mamba seed {s} {100 * np.mean(by[s]):.0f}% n {len(by[s])}")
p = np.array([np.mean(v) for v in by.values()]) * 100
print(f"single mamba mean {p.mean():.1f}+-{stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p)):.1f} eps {len(r)}")
e = rows("hold_ens")
for c in sorted({x["cand"] for x in e}):
    v = [x["none"]["success"] for x in e if x["cand"] == c and "none" in x]
    k, n = sum(v), len(v)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95, "wilson")
    print(f"{c} {100 * k / n:.1f}% ({k}/{n}) wilson95 {100 * lo:.0f}-{100 * hi:.0f} errors {sum('error' in x for x in e if x['cand'] == c)}")
PY
t "DONE"

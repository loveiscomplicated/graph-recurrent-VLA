#!/bin/bash
set -eo pipefail
cd /workspace/graph-recurrent-VLA
O=/workspace/videos_ens
mkdir -p $O
S=$(python - <<'PY'
import json, glob
r = [json.loads(l) for f in glob.glob("/workspace/eval/final_ens/results_*.jsonl") for l in open(f)]
r = [x for x in r if x["cand"] == "ens_mamba5@step" and "none" in x]
ok = [x["env_seed"] for x in r if x["none"]["success"]][:3]
bad = [x["env_seed"] for x in r if not x["none"]["success"]][:2]
print(" ".join(map(str, ok + bad)))
PY
)
echo "env seeds $S"
for e in $S; do
  python exp2_eval.py video --no-move --cands expert,ens_mamba5@step --env-seed $e --ens /workspace/ens.json --out $O
done
ls -lh $O
cd /workspace && tar czf videos_ens.tgz videos_ens && ls -lh videos_ens.tgz

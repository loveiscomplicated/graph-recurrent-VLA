#!/usr/bin/env bash
cd /workspace/graph-recurrent-VLA
export OMP_NUM_THREADS=4
base=outputs/visual_baseline
topic=$(cat $base/ntfy_topic.txt)
/workspace/envs/baseline/bin/python -m visual.baseline.probe_lateral --output $base/probe_lateral_held.json \
    --variants both_256_x2,main_256 --held-only 2>&1 | grep -v -i warn
summary=$(/workspace/envs/baseline/bin/python -c "
import json; r=json.load(open('$base/probe_lateral_held.json'))['results']
print('; '.join(f\"{x['name']}: lateral median {x['lateral_norm_mm_median']}mm p90 {x['lateral_norm_mm_p90']}mm\" for x in r))" 2>&1)
curl -s -m 15 -H "Title: probe round 3 (held peg only)" -d "$summary" "https://ntfy.sh/$topic" > /dev/null
echo "$summary"

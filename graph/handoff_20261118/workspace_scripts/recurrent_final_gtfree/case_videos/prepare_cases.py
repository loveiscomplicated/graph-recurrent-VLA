"""Classify the final policy's failures and select videos by observed failure stage."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source_runs'
spec = importlib.util.spec_from_file_location('failure_classifier',
    ROOT.parent/'recurrent_full_insertion/failure_classification/classify.py')
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)

rows = [json.loads(line) for path in sorted((SOURCE/'finetuned').glob('shard_*/rollouts.jsonl'))
        for line in path.open()]
assert len(rows) == len({r['seed'] for r in rows}) == 32
result = {}
for condition in ('fixed','moving'):
    failures = []
    successes = []
    for row in rows:
        trial = row[condition]
        d = classifier.summarize(row['seed'], trial)
        trace = trial['trace']
        errors = np.array([f['raw_lateral_mm'] for f in trace])
        zero = np.array([f['observed'][0] == 0 for f in trace])
        edges = np.flatnonzero(np.diff(np.r_[False, zero, False]))
        d.update(success=trial['success'], lateral_error_p90_mm=float(np.percentile(errors,90)),
                 max_no_peg_frames=max((int(b-a) for a,b in zip(edges[::2],edges[1::2])), default=0),
                 lost_grasp_after_first=bool(trial['ever_grasped'] and any(not f['grasp'] for f in
                    trace[next(i for i,f in enumerate(trace) if f['grasp']):])))
        (successes if trial['success'] else failures).append(d)
    result[condition] = dict(total=32, success=len(successes), failed=len(failures),
                            failure_counts=dict(Counter(r['group'] for r in failures)),
                            failures=failures, successes=successes)
selection = [
    ('fixed','success',700882), ('moving','success',700878),
    ('fixed','before_entrance',700451), ('fixed','shallow',700674),
    ('fixed','partial',700732), ('fixed','deep_then_stalled',700866),
    ('moving','never_aligned_aperture',700807), ('moving','before_entrance',700729),
    ('moving','at_entrance',700549), ('moving','shallow',700674),
    ('moving','deep_then_stalled',700806), ('moving','deep_then_retreated',700882),
    ('moving','deep_then_retreated_grasp_loss',700831),
]
cases = []
for condition, group, seed in selection:
    trial = next(r for r in result[condition]['successes']+result[condition]['failures'] if r['seed']==seed)
    assert (group=='success') == trial['success']
    if group!='success':
        assert trial['group'] == group.replace('_grasp_loss','')
    cases.append(dict(condition=condition, group=group, seed=seed, stats=trial))
out = ROOT/'case_videos'
(out/'classification.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'selection.json').write_text(json.dumps(cases,indent=2)+'\n')
for condition, d in result.items():
    print(condition, d['success'], d['failure_counts'])
print('Selected',len(cases),'videos')

"""Hide future fitted Box translations to measure the causal gap fallback."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SCALE=1000


def main():
    rows=defaultdict(list)
    for line in (ROOT/'full/frames.jsonl').read_text().splitlines():
        row=json.loads(line)
        rows[row['seed']].append(row)
    result={}
    for horizon in (1,3,5):
        gap_errors=defaultdict(list)
        for episode in rows.values():
            velocity=np.zeros(3)
            last=np.zeros(3)
            previous=0
            for i,row in enumerate(episode):
                f=row['rigid_fit']['shared_hold']
                if f['support']<3:
                    continue
                measured=np.asarray(f['pose_m'])
                velocity=.35*velocity+.65*(measured-last)/max(1,i-previous)
                last,previous=measured,i
                if not row['moving'] or i+horizon>=len(episode):
                    continue
                future=episode[i+1:i+1+horizon]
                labels='moving_through_gap' if all(r['moving'] for r in future) else 'stops_during_gap'
                target=np.asarray(future[-1]['offset_m'])
                gap_errors[labels].append((np.linalg.norm(last-target)*SCALE,
                                           np.linalg.norm(last+velocity*horizon-target)*SCALE))
        result[horizon]={label:dict(n=len(values),hold_p90_mm=float(np.percentile(np.array(values)[:,0],90)),
                                    velocity_p90_mm=float(np.percentile(np.array(values)[:,1],90)))
                         for label,values in gap_errors.items()}
    (ROOT/'full/controlled_gaps.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

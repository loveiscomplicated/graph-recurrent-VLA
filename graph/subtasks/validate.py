"""Coverage, exact original alignment, deterministic segmentation and provenance."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import load_segments,load_audit,segment_arrays
from .segmentation import segment_episode,runs
from .specs import NAMES
from graph.prototype.common import write_json


def validate(index_path,output):
    data=load_segments(index_path);rows=[];total=covered=0;overlaps=[];deterministic=True;alignment_checks=0
    for ep in data['episodes']:
        audit=load_audit(ep['audit_file']);first,_,_=segment_episode(audit,data['thresholds']);second,_,_=segment_episode(audit,data['thresholds'])
        deterministic &= first==second
        for expected,stored in zip(first,ep['segments']):
            if any(expected[k]!=stored[k] for k in expected):raise ValueError('stored segmentation differs from recomputation')
        mask=np.zeros(ep['total_actions'],dtype=int)
        for s in ep['segments']:
            if s['status']=='detected':
                arrays=segment_arrays(data,s);alignment_checks+=1
                i,j=s['start_t'],s['end_t'];mask[i:j]+=1
                assert len(arrays['states'])==j-i+1 and len(arrays['actions'])==j-i
        if np.any(mask>1):overlaps.append(ep['episode_id'])
        total+=len(mask);covered+=int(np.count_nonzero(mask))
        rows.append(dict(episode_id=ep['episode_id'],split=ep['split'],total_actions=len(mask),covered_actions=int(np.count_nonzero(mask)),
            uncovered_runs=runs(mask==0),
            segments=ep['segments']))
    tables={}
    for split in ('all','train','validation','test'):
        eps=[e for e in data['episodes'] if split=='all' or e['split']==split];table=[]
        for n in NAMES:
            ss=[next(s for s in e['segments'] if s['subtask']==n) for e in eps];valid=[s for s in ss if s['status']=='detected'];length=[s['end_t']-s['start_t'] for s in valid]
            table.append(dict(subtask=n,episodes=len(eps),detected=len(valid),success_condition_episodes=sum(e['diagnostics'][n]['first_success'] is not None for e in eps),
                ordered_success=sum(s['success_reached'] for s in ss),start_valid=sum(s['start_valid'] for s in ss),
                mean_length=float(np.mean(length)) if length else None,
                length_quantiles=np.quantile(length,[0,.25,.5,.75,1]).tolist() if length else [],
                ambiguous=sum(s['status']=='unresolved' for s in ss),multiple_candidate_episodes=sum(s['multiple_candidates'] for s in ss),
                exits_after_success=sum(e['diagnostics'][n]['exits_after_success'] for e in eps),
                unresolved_reasons=dict(Counter(s['boundary_reason'] for s in ss if s['status']=='unresolved'))))
        tables[split]=table
    report=dict(episodes=len(rows),tables=tables,total_actions=total,covered_actions=covered,coverage=covered/total,
        overlaps=overlaps,alignment_checks=alignment_checks,deterministic=bool(deterministic),
        replay_valid=sum(e['replay']['replay_valid'] for e in data['episodes']),
        official_success=sum(e['replay']['official_success'] for e in data['episodes']),
        max_observation_error=max(e['replay']['max_observation_error'] for e in data['episodes']),
        rows=rows)
    output=Path(output);write_json(output/'validation.json',report)
    lines=['| Subtask | Detected | Condition satisfied | Mean actions | Unresolved | Multiple runs |', '|---|---:|---:|---:|---:|---:|']
    for r in tables['all']:lines.append(f"| {r['subtask']} | {r['detected']} | {r['success_condition_episodes']} | {r['mean_length']:.2f} | {r['ambiguous']} | {r['multiple_candidate_episodes']} |")
    (output/'coverage.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print({k:report[k] for k in ('episodes','total_actions','covered_actions','alignment_checks','deterministic','replay_valid','official_success','max_observation_error')})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',default='outputs/subtasks/segments/index.json');p.add_argument('--output',default='outputs/subtasks/reports')
    a=p.parse_args();validate(a.index,a.output)
if __name__=='__main__':main()

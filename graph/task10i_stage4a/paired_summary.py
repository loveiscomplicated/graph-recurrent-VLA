"""Explicit paired frame->episode->parent aggregation, including undefined support."""
from collections import defaultdict
from pathlib import Path
import argparse
import h5py
import numpy as np
from .run_pipeline import STAGE3, read, write
from .temporal_audit import phases_for


def collect(root, condition, entries):
    values=defaultdict(list);support=[]
    for p in sorted((root/'perception').glob('*/receipt.json')):
        eid=str(read(p)['episode_id']);e=entries[eid]
        with h5py.File(e['raw_path'],'r') as f:phases,segments=phases_for(e,f)
        def add(t,metric,value,stratum):
            for scope in ('all','t0' if t==0 else 'later','phase:'+phases[t],'segment:'+segments[t],'support:'+stratum):
                values[eid,e['parent_id'],metric,scope].append(value)
        for r in read(p.parent/'stage_rows.json'):
            if r['stage']!='sam2':continue
            m=r['metrics'];t=r['timestep'];prefix=f"mask/{r['camera']}/{r['object_id']}/"
            n=m['input_gt_support'] if m else None
            stratum='GT_UNSUPPORTED' if n is None else 'absent' if n==0 else 'tiny_1_7' if n<8 else 'visible_8_31' if n<32 else 'visible_32_127' if n<128 else 'visible_128_plus'
            support.append(dict(condition=condition,episode=eid,timestep=t,camera=r['camera'],object_id=r['object_id'],reference=r['reference_kind'],support=n))
            for k in ('precision','recall','iou'):add(t,prefix+k,m[k] if m else None,stratum)
            add(t,prefix+'visible_object_recall',float(m['true_positive']>0) if m and n>=8 else None,stratum)
            add(t,prefix+'false_presence',float(m['output_support']>0) if m and n==0 else None,stratum)
        for r in read(p.parent/'representative_points.json'):
            t=r['timestep'];oid=r['object_id'];prefix=f'graph/{oid}/'
            good=r['reference_kind']=='EXACT_GT';n=r.get('visible_reference_points') if good else None
            stratum='GT_UNSUPPORTED' if n is None else 'absent' if n==0 else 'tiny_1_7' if n<8 else 'visible_8_plus'
            visible=good and n is not None and n>=8
            add(t,prefix+'object_recall',float(r['correct']>0) if visible else None,stratum)
            add(t,prefix+'missing',float(r['points']==0) if visible else None,stratum)
            add(t,prefix+'false_presence',float(r['points']>0) if good and n==0 else None,stratum)
            for k in ('visible_coverage_4mm','projected_boundary_coverage_4mm'):
                add(t,prefix+k,r.get(k) if visible else None,stratum)
            for k in ('robot','wrong_object','background','correct'):
                add(t,prefix+k+'_fraction',r[k]/r['points'] if good and r['points'] else None,stratum)
            for k in ('surface_distance_mean_mm','surface_distance_p95_mm','normal_valid_fraction'):
                add(t,prefix+k,r.get(k) if good else None,stratum)
    episode=[]
    for (eid,parent,metric,scope),vs in values.items():
        finite=[v for v in vs if v is not None]
        episode.append(dict(condition=condition,episode=eid,parent_id=parent,metric=metric,scope=scope,
             eligible_frames=len(finite),total_frames=len(vs),value=float(np.mean(finite)) if finite else None))
    parents=defaultdict(list)
    for r in episode:parents[r['parent_id'],r['metric'],r['scope']].append(r)
    parent=[]
    for (pid,metric,scope),rr in parents.items():
        vs=[r['value'] for r in rr if r['value'] is not None]
        parent.append(dict(condition=condition,parent_id=pid,metric=metric,scope=scope,
             value=float(np.mean(vs)) if vs else None,episodes=len(rr),supported_episodes=len(vs),
             eligible_frames=sum(r['eligible_frames'] for r in rr),total_frames=sum(r['total_frames'] for r in rr)))
    return episode,parent,support


def run(output,baseline,candidate):
    output.mkdir(parents=True,exist_ok=False)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    be,bp,bs=collect(baseline,'baseline',entries);ce,cp,cs=collect(candidate,'candidate',entries)
    assert {(r['episode'],r['timestep'],r['camera'],r['object_id'],r['support'],r['reference']) for r in bs}=={(r['episode'],r['timestep'],r['camera'],r['object_id'],r['support'],r['reference']) for r in cs}
    b={(r['parent_id'],r['metric'],r['scope']):r for r in bp};c={(r['parent_id'],r['metric'],r['scope']):r for r in cp}
    assert b.keys()==c.keys()
    grouped=defaultdict(list);paired=[]
    for key,br in b.items():
        cr=c[key];assert br['total_frames']==cr['total_frames']
        row=dict(parent_id=key[0],metric=key[1],scope=key[2],baseline=br['value'],candidate=cr['value'],
             delta=cr['value']-br['value'] if cr['value'] is not None and br['value'] is not None else None,
             baseline_eligible_frames=br['eligible_frames'],candidate_eligible_frames=cr['eligible_frames'],total_frames=br['total_frames'])
        paired.append(row);grouped[key[1:]].append(row)
    macro=[]
    for (metric,scope),rr in sorted(grouped.items()):
        mean=lambda field:float(np.mean([r[field] for r in rr if r[field] is not None])) if any(r[field] is not None for r in rr) else None
        macro.append(dict(metric=metric,scope=scope,parents=len(rr),paired_supported_parents=sum(r['delta'] is not None for r in rr),
             baseline=mean('baseline'),candidate=mean('candidate'),delta=mean('delta'),
             baseline_eligible_frames=sum(r['baseline_eligible_frames'] for r in rr),candidate_eligible_frames=sum(r['candidate_eligible_frames'] for r in rr)))
    write(output/'episode_metrics.json',be+ce);write(output/'parent_metrics.json',bp+cp)
    write(output/'paired_parent_metrics.json',paired);write(output/'paired_macro.json',macro)
    write(output/'reference_support.json',dict(total_camera_object_frames=len(bs),exact=sum(r['reference']=='EXACT_GT' for r in bs),
         unsupported=sum(r['reference']!='EXACT_GT' for r in bs),same_denominator=True,unsupported_rows_retained=True))
    write(output/'summary.json',dict(status='PAIRED_PARENT_MACRO_FINISHED',baseline=str(baseline),candidate=str(candidate),
         aggregation='frame mean -> episode mean -> parent mean; R/D shared logical episode once',
         precision_undefined_when_no_prediction=True,scope_rows_are_separate_strata_not_additive=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();run(a.output,a.baseline,a.candidate)

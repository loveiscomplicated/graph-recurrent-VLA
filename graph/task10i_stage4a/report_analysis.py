"""Read-only matched-parent/frame-weighted reporting; never changes selection decisions."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .run_pipeline import read,write


def run(output,comparison,metrics_root):
    output.mkdir(parents=True,exist_ok=False)
    paired=read(comparison/'paired_parent_metrics.json');groups=defaultdict(list)
    for r in paired:groups[r['metric'],r['scope']].append(r)
    rng=np.random.default_rng(42);matched=[]
    for (metric,scope),rr in sorted(groups.items()):
        valid=[r for r in rr if r['delta'] is not None]
        values=np.array([r['delta'] for r in valid]);ci=np.percentile(rng.choice(values,(2000,len(values)),replace=True).mean(1),[2.5,97.5]).tolist() if len(values) else None
        matched.append(dict(metric=metric,scope=scope,total_parents=len(rr),paired_supported_parents=len(valid),
            baseline=float(np.mean([r['baseline'] for r in valid])) if valid else None,
            candidate=float(np.mean([r['candidate'] for r in valid])) if valid else None,
            delta=float(values.mean()) if len(values) else None,parent_bootstrap_95ci=ci,
            baseline_only_defined_parents=sum(r['baseline'] is not None and r['candidate'] is None for r in rr),
            candidate_only_defined_parents=sum(r['candidate'] is not None and r['baseline'] is None for r in rr)))
    weighted=defaultdict(list)
    for r in read(comparison/'episode_metrics.json'):weighted[r['condition'],r['metric'],r['scope']].append(r)
    frame=[]
    for (condition,metric,scope),rr in sorted(weighted.items()):
        denominator=sum(r['eligible_frames'] for r in rr if r['value'] is not None)
        numerator=sum(r['value']*r['eligible_frames'] for r in rr if r['value'] is not None)
        frame.append(dict(condition=condition,metric=metric,scope=scope,value=numerator/denominator if denominator else None,
            eligible_frames=denominator,total_frames=sum(r['total_frames'] for r in rr)))
    micro=defaultdict(lambda:dict(tp=0,fp=0,fn=0,robot=0,background=0,other=0,frames=0,unsupported=0))
    for condition,root in metrics_root.items():
        for p in (root/'perception').glob('*/stage_rows.json'):
            for r in read(p):
                if r['stage']!='sam2':continue
                x=micro[condition,r['camera'],r['object_id']];m=r['metrics']
                if m is None:x['unsupported']+=1;continue
                x['tp']+=m['true_positive'];x['fp']+=m['false_positive'];x['fn']+=m['false_negative'];x['robot']+=m['robot_pixels'];x['background']+=m['background_pixels'];x['other']+=m['wrong_object_pixels'];x['frames']+=1
    pooled=[]
    for (condition,camera,oid),x in sorted(micro.items()):
        pooled.append(dict(condition=condition,camera=camera,object_id=oid,**x,
            precision=x['tp']/(x['tp']+x['fp']) if x['tp']+x['fp'] else None,
            recall=x['tp']/(x['tp']+x['fn']) if x['tp']+x['fn'] else None,
            iou=x['tp']/(x['tp']+x['fp']+x['fn']) if x['tp']+x['fp']+x['fn'] else None))
    write(output/'matched_parent_macro.json',matched);write(output/'frame_weighted_metrics.json',frame);write(output/'pooled_pixel_counts.json',pooled)
    write(output/'scope.json',dict(status='REPORT_STATISTICS_FINISHED',all_parent_rows_retained=True,
        matched_means_use_identical_supported_parent_set=True,original_guard_deltas_not_changed=True,
        undefined_precision_has_explicit_counts=True,bootstrap_unit='parent',bootstrap_samples=2000,
        exploratory_development_CI_not_independent_evidence=True,metric_reference_support=read(comparison/'reference_support.json')))

"""Post-forward feature matching strata and source-clock-aware grasp labels.

Historical native grasp flags are after-action observations (frame i+1).
Final collection records intended expert phase, not the native grasp predicate;
that proxy is kept separate. Nothing here creates a practical trigger.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from .reference_pair_audit import curves
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


def phase_clock(trace,frames):
    phase=['NOT_RECORDED']*frames;grasp=[None]*frames
    if len(trace)==frames and all('expert_phase' in r for r in trace):
        phase=[r['expert_phase'] for r in trace]
    elif len(trace)==frames-1 and all('phase' in r for r in trace):
        phase=[r['phase'] for r in trace]+['TERMINAL_OBSERVATION']
        for i,r in enumerate(trace):
            if 'is_grasping' in r.get('physical',{}):grasp[i+1]=bool(r['physical']['is_grasping'])
    first=next((i for i,v in enumerate(grasp) if v is True),None)
    result=[]
    for t in range(frames):
        role=(('BEFORE_FIRST_OBSERVED_GRASP' if t<first else 'AFTER_FIRST_OBSERVED_GRASP') if first is not None else
            'NO_OBSERVED_GRASP' if any(v is not None for v in grasp) else 'GRASP_NOT_RECORDED')
        result.append(dict(phase=phase[t],native_grasp_at_frame=grasp[t],native_grasp_role=role,
            expert_after_close_phase=phase[t] in ('close','lift','align','insert'),
            expert_phase_is_not_a_native_grasp_flag=True))
    return result


def visible_bin(n):
    if n is None:return 'GT_UNSUPPORTED'
    if n==0:return 'ZERO'
    if n<8:return '1_7'
    if n<32:return '8_31'
    if n<128:return '32_127'
    if n<512:return '128_511'
    return '512_PLUS'


def run(root,crops,final,expected,references,candidate,evaluation_scope='final'):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    historical=read(crops/'dinov2_reference_provenance.json')
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    actual=read(final/'matching/dinov2_actual_positive_negative_pairs.json')
    final_entries={str(e['episode_id']):e for e in read(expected)['entries']};cache={};inputs=[]
    def labels(eid,is_final):
        key=eid,is_final
        if key in cache:return cache[key]
        e=final_entries[eid] if is_final else entries[eid]
        path=Path(e['raw_path']).parent/'trace.json'
        trace=read(path) if path.exists() else []
        clock=phase_clock(trace,e['actions']+1)
        if is_final:ref=references/'references'/eid
        else:
            cr=read(Path('/workspace/task10i_stage4a_completion_20261101_v1/run/corrected/perception')/eid/'receipt.json')
            ref=Path(cr['corrected_reference']).parent
        rr=read(ref/'receipt.json')
        assert rr['raw_sha256']==e['raw_sha256']
        assert sha(ref/'segmentation.npz')==rr['files']['segmentation.npz'] if 'files' in rr else sha(ref/'segmentation.npz')==rr['segmentation_sha256']
        with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        visible={(c,oid):[int(np.isin(seg[t,ci],rr['object_ids'][name]).sum()) if exact[t] else None for t in range(e['actions']+1)]
            for ci,c in enumerate(CAMERAS) for oid,name in enumerate(('peg','box'))}
        inputs.append(dict(episode=eid,trace=str(path) if path.exists() else None,trace_sha256=sha(path) if path.exists() else None,
            reference=str(ref/'segmentation.npz'),reference_sha256=sha(ref/'segmentation.npz')))
        cache[key]=(clock,visible);return cache[key]
    config=dict(source_sha256=sha(Path(__file__)),crop_pairs_sha256=sha(crops/'dinov2_positive_negative_pairs.csv'),
        metadata_sha256=sha(crops/'dinov2_reference_provenance.json'),actual_pairs_sha256=sha(final/'matching/dinov2_actual_positive_negative_pairs.json'),
        no_forwards=True,no_threshold_fitting=True,not_new_adoption_gates=True,
        evaluation_scope=evaluation_scope,
        native_grasp_clock='historical physical.is_grasping after action i labels observation i+1',
        native_grasp_contract='use native after-action flags where recorded; new Final expert_phase has no native grasp flag and remains a separate proxy')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_matching_phase_occlusion_grasp_audit',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        pairs=[]
        with (crops/'dinov2_positive_negative_pairs.csv').open() as stream:
            for p in csv.DictReader(stream):
                a,b=historical[int(p['a'])],historical[int(p['b'])]
                if a['timestep']>b['timestep']:a,b=b,a
                assert a['episode']==b['episode'] and a['parent_id']==b['parent_id']
                clock,visibility=labels(str(b['episode']),False);t=b['timestep']
                actor=a['actual_actor'];oid={'peg':0,'box':1}.get(actor)
                vis=visibility[b['camera'],oid] if oid is not None else [None]*len(clock)
                pairs.append(dict(parent_id=p['parent_id'],episode=b['episode'],mode=p['mode'],branch=p['branch'],
                    reference_timestep=a['timestep'],timestep=t,camera=b['camera'],cross_camera=p['cross_camera']=='True',
                    intended_positive=p['positive']=='True',similarity=float(p['similarity']),reference_actor=actor,
                    visibility_bin=visible_bin(vis[t]),after_prior_exact_zero_visibility=any(v==0 for v in vis[:t]),
                    grasp_pair_transition=clock[a['timestep']]['native_grasp_role']+' -> '+clock[t]['native_grasp_role'],**clock[t]))
        runtime=[]
        for p in actual:
            eid=p['episode'];clock,visibility=labels(eid,True);t=p['timestep'];v=visibility[p['camera'],p['object_id']]
            runtime.append(dict(**p,visibility_bin=visible_bin(v[t]),after_prior_exact_zero_visibility=any(n==0 for n in v[:t]),**clock[t]))
        csv_write(root/'historical_grasp_occlusion_matching_pairs.csv',pairs)
        csv_write(root/'historical_grasp_occlusion_matching_metrics.csv',curves(pairs,
            ['mode','branch','cross_camera','grasp_pair_transition','after_prior_exact_zero_visibility','visibility_bin']))
        write(root/'independent_actual_matching_phase_pairs.json',runtime)
        csv_write(root/'independent_actual_matching_phase_metrics.csv',curves([r for r in runtime if r.get('intended_positive') is not None],
            ['stratum','camera','object_id','phase','expert_after_close_phase','after_prior_exact_zero_visibility','visibility_bin']))
        counts=Counter((r['stratum'],r['phase'],r['kind']) for r in runtime)
        csv_write(root/'actual_matching_stratum_counts.csv',[dict(stratum=s,phase=p,kind=k,comparisons=n) for (s,p,k),n in sorted(counts.items())])
        write(root/'scope.json',dict(status='POST_FORWARD_MATCHING_PHASE_OCCLUSION_GRASP_AUDIT_COMPLETE',run_id=logger.run_id,**config,
            inputs=inputs,historical_pairs=len(pairs),actual_independent_pairs=len(runtime),
            native_grasp_labels_used_only_for_evaluation=True,correlated_pairs_not_independent_samples=True,
            actual_empty_strata_have_no_TPR_FPR_claim=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','crops','final','expected','references'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);p.add_argument('--evaluation-scope',default='final');run(**vars(p.parse_args()))

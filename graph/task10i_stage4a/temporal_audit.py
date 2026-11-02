"""Evaluation-only temporal definitions; no GT event drives inference."""
from collections import defaultdict
import h5py
import numpy as np
from .run_pipeline import read,write,STAGE3,CAMERAS
from .instrumentation import support

def runs(flags):
    out=[];start=None
    for t,value in enumerate(list(flags)+[False]):
        if value and start is None:start=t
        elif not value and start is not None:out.append([start,t]);start=None
    return out

def sequence_metrics(visible,correct,present):
    visible=np.asarray(visible);correct=np.asarray(correct);present=np.asarray(present)
    recovery=[]
    for a,b in runs(visible>=8):
        if a==0 or visible[a-1]!=0:continue
        hits=np.flatnonzero(correct[a:b]>0)
        recovery.append(dict(start=a,end=b,delay_frames=int(hits[0]) if len(hits) else None,recovered=bool(len(hits))))
    return dict(visible_frames=int((visible>=8).sum()),visible_missing_frames=int(((visible>=8)&(correct==0)).sum()),
        false_presence_absent_frames=int(((visible==0)&present).sum()),ghost_runs=runs((visible==0)&present),
        visible_failure_runs=runs((visible>=8)&(correct==0)),reappearances=recovery,
        temporal_fragment_count=sum(len(runs(correct[a:b]>0)) for a,b in runs(visible>=8)))

def action_phases(rows,actions):
    """Never align suffix-only metadata to a full raw episode by list position.

    Legacy Task10F corrective traces contain only the post-anchor suffix and
    no expert phase labels. Their raw HDF5 segment array is complete, while
    expert phases must stay explicitly unavailable for all action frames.
    A partial trace WITH phase labels needs a separately proven clock join.
    """
    unavailable='EXPERT_PHASE_UNAVAILABLE'
    if len(rows)!=actions:
        if any(r.get('phase') for r in rows):
            raise ValueError('partial phase metadata requires a proven action-clock join')
        return [unavailable]*actions
    return [r.get('phase') or unavailable for r in rows]


def phases_for(e,f):
    folder=__import__('pathlib').Path(e['raw_path']).parent
    if (folder/'trace.json').exists():
        data=read(folder/'trace.json');phase=action_phases(data,e['actions'])
    elif (folder/'expert_stages.json').exists():phase=action_phases(read(folder/'expert_stages.json'),e['actions'])
    else:phase=['EXPERT_PHASE_UNAVAILABLE']*e['actions']
    assert len(phase)==e['actions']
    segment=f['segment'][:] if 'segment' in f else np.zeros(e['actions'],np.uint8)
    assert len(segment)==e['actions']
    segment_names=['full_task' if e['branch']=='full_task' else {0:'corrective_prefix',1:'tcp_disturbance',2:'expert_recovery'}[int(x)] for x in segment]
    return phase+['terminal'],segment_names+['terminal']

def run(root,baseline,corrected):
    from pathlib import Path
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()};temporal=[];frame_rows=[];phase_rows=[]
    for receipt in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(receipt);eid=str(cr['episode_id']);e=entries[eid];rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
        with np.load(rrpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        with np.load(baseline/'perception'/eid/'masks.npz') as z:masks=np.unpackbits(z['mask_bits'],axis=1).reshape(e['actions']+1,2,2,128,128).astype(bool)
        with h5py.File(e['raw_path']) as f:phase,segments=phases_for(e,f)
        for ci,c in enumerate(CAMERAS):
            for oid in (0,1):
                visible=[];correct=[];present=[];xor=[];labels=[];previous=None
                for t in range(e['actions']+1):
                    m=masks[t,ci,oid];metrics=support(m,seg[t,ci],oid,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']]) if exact[t] else None
                    if metrics is None:raise AssertionError('temporal audit needs explicit support handling, not dropped unknown frames')
                    visible.append(metrics['input_gt_support']);correct.append(metrics['true_positive']);present.append(bool(m.any()))
                    flicker=float(np.logical_xor(m,previous).sum()/max(1,np.logical_or(m,previous).sum())) if previous is not None else None;previous=m
                    identity=max([(metrics['true_positive'],'correct'),(metrics['wrong_object_pixels'],'other_object'),(metrics['robot_pixels'],'robot'),(metrics['background_pixels'],'background')])[1] if m.any() else 'empty'
                    frame_rows.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,camera=c,object_id=oid,phase=phase[t],segment=segments[t],dominant_source=identity,
                        mask_xor_over_union=flicker,metrics=metrics,reference_kind='EXACT_GT'))
                    labels.append(identity)
                tm=sequence_metrics(visible,correct,present);temporal.append(dict(episode=eid,parent_id=e['parent_id'],camera=c,object_id=oid,**tm,
                    dominant_actor_transitions=sum(a!=b for a,b in zip(labels,labels[1:]) if a!='empty' and b!='empty'),internal_tracker_id_switch='separate from dominant actor changes; original IDs fixed and adapter checked them'))
        grouped=defaultdict(list)
        for r in read(receipt.parent/'stage_rows.json'):grouped[r['camera'],r['object_id'],r['stage'],phase[r['timestep']],segments[r['timestep']]].append(r)
        for key,rows in grouped.items():
            mm=[r['metrics'] for r in rows if r['metrics'] is not None]
            mean=lambda vs:float(np.mean(vs)) if len(vs) else None
            phase_rows.append(dict(episode=eid,parent_id=e['parent_id'],camera=key[0],object_id=key[1],stage=key[2],phase=key[3],segment=key[4],frames=len(rows),supported_frames=len(mm),
                pixel_precision=mean([m['precision'] for m in mm if m['precision'] is not None]),pixel_recall=mean([m['recall'] for m in mm if m['recall'] is not None]),
                visible_empty_rate=mean([float(m['output_support']==0) for m in mm if m['input_gt_support']>=8]),false_absent_presence=mean([float(m['output_support']>0) for m in mm if m['input_gt_support']==0])))
    write(root/'temporal_episode_metrics.json',temporal);write(root/'temporal_frame_metrics.json',frame_rows);write(root/'phase_episode_metrics.json',phase_rows)
    write(root/'temporal_scope.json',dict(status='TEMPORAL_AND_PHASE_AUDIT_FINISHED',visibility_minimum_pixels=8,
        reappearance='visible>=8 run immediately following true zero visible support; tiny-support transitions separate',
        flicker='raw mask XOR/union, not motion compensated; object motion can raise it',
        labels_used_only_for_evaluation=True,no_GT_triggered_student_reset=True,phase_unavailable_explicit=True))

"""Current-evidence handoff test on0250, with no GT-triggered intervention."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from .evaluation_metrics import pixel_metrics
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def run(root,baseline,partition,previous,references):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    eid='10i_stage3_10i_s1_0250_full_task';candidate='BC_WB_HANDOFF'
    entry=next(e for e in read(baseline/'expected_counts.json')['entries'] if str(e['episode_id'])==eid)
    config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        conditions=['B0','previous_full_BC','BC_WB_HANDOFF'],GT_practical_input=False,
        trigger='actual first original baseline found edge after an earlier experimental anchor; never t16 error time',
        handoff=dict(minimum_nonrobot_fraction=.9,old_support_overlap=.8,minimum_clean_area_ratio=1.,reid_similarity=.6,reid_margin=.05),
        enabled_slots=[['hand_camera',0],['base_camera',1]],Internal_and_Final_access=False)
    write(root/'protocol.json',config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_later_baseline_initialization_handoff',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        subprocess.run([sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts','-v'],check=True)
        command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.development_correction',str(root/'inference'),
            str(baseline),str(partition),'--engineering','--engineering-episode',eid,'--engineering-last',str(entry['actions']),
            '--conditions',candidate,'--geometry-cache']
        with (root/'inference.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        folder=root/'inference/perception'/eid;source=previous/'perception'/eid
        ir=read(folder/'receipt.json');assert all(sha(folder/p)==h for p,h in ir['files'].items())
        with np.load(folder/(candidate+'.npz')) as z:current=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
        with np.load(folder/'B0.npz') as z:b0=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
        with np.load(source/'BC_KEEP.npz') as z:old=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
        assert np.array_equal(old[:,0],b0[:,0]),'historical control has other Main interventions; cannot isolate handoff'
        assert np.array_equal(current[:,0,0],b0[:,0,0]) and np.array_equal(current[:,1,1],b0[:,1,1])
        events=read(folder/candidate/'baseline_handoff_events.json');accepted=[e for e in events if e['accepted']]
        # After inference, use original model records to verify the actual event
        # time. The practical decision never receives a known GT failure time.
        actual_new=[r['timestep'] for r in read(baseline/'perception'/eid/'vision_records.json')
                    if r['condition']=='B0' and r['camera']=='hand_camera' and 1 in r['new_object_ids']]
        assert actual_new==[15]
        resolved=bool(accepted and accepted[0]['timestep']==actual_new[0] and
                      np.array_equal(current[15:,1,0],b0[15:,1,0]))
        # Only now open GT labels for the saved outcomes.
        ref=references/'references'/eid;rr=read(ref/'receipt.json')
        assert all(sha(ref/p)==h for p,h in rr['files'].items())
        with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        metrics=[]
        for condition,values in [('B0',b0),('previous_full_BC',old),(candidate,current)]:
            for t in range(len(values)):
                if not exact[t]:continue
                metrics.append(dict(condition=condition,timestep=t,reference_kind='EXACT_GT',
                    **pixel_metrics(values[t,1,0],np.isin(seg[t,1],rr['object_ids']['peg']),
                        np.isin(seg[t,1],rr['robot_ids']),np.isin(seg[t,1],rr['object_ids']['box']))))
        csv_write(root/'handoff_mask_metrics.csv',metrics)
        write(root/'scope.json',dict(status='OBSERVED_BASELINE_HANDOFF_ENGINEERING_COMPLETE',run_id=logger.run_id,
            **config,frames=len(current),original_new_initialization_times=actual_new,handoff_events=events,
            causal_recovery_demonstrated=resolved,oracle_not_used_as_practical_trigger=True,
            full_Development_candidate_validation_required=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition','previous','references'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.baseline,a.partition,a.previous,a.references)
    except BaseException as exc:
        if a.root.exists():write(a.root/'failure.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise

"""Real-data development feasibility smoke; selection and validation stay closed."""
import argparse
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import STAGE3,CAMERAS,read,write,sha
from .reliable_reanchor import WristPegReanchor
from .instrumentation import support


def run(root,baseline,reference_root,refine=False):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    torch.set_num_threads(1);torch.manual_seed(42);root.mkdir(parents=True,exist_ok=False)
    runtime_root=Path('/workspace/task10g_f1_abc_20261031/full');runtime=read(runtime_root/'runtime_paths.json');old=Path('/workspace/task10f1_gt_pretrain_20261030')
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    choices=['10i_stage3_10i_s1_0362_full_task','10i_stage3_10i_s1_0666_full_task','2000003']
    dev=set(read(baseline/'protocol.json')['development']);assert all(entries[e]['parent_id'] in dev for e in choices)
    spec=dict(candidate='RGBD_MAIN_SUPPORTED_WRIST_PEG_REANCHOR_V2' if refine else 'RGBD_MAIN_SUPPORTED_WRIST_PEG_REANCHOR_V1',shared_depth_and_robot_refinement=refine,raw_sam2_valid_and_clean_fraction_guards_preserved=refine,camera='hand_camera',object_id=0,detector_prompt='red and white block',detector_threshold=.1,
        interval=5,topk=8,minimum_positive_depth_agreeing_main_witness_pixels=6,minimum_witness_box_and_mask_fraction=.6,
        depth_agreement_m=.004,wrist_gate='all original rules unchanged',proposal_large_area='permitted only through positive witness and unchanged gate validation',
        trigger='current baseline/correction Wrist gate failure at registered cadence',GT_trigger=False,student_GT_input=False,production_adoption=False,
        scope='development first20-frame causal prefixes, engineering feasibility only; not candidate selection or independent test')
    write(root/'candidate_spec.json',spec)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_RGBD_reanchor_development_smoke',seed=42,root=root/'experiments',job_type='evaluation',config=spec) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(__file__),candidate_source_sha256=sha(Path(__file__).with_name('reliable_reanchor.py'))))
        detector=GDino(device='cuda',model_id=runtime['dino']);tracker=SAM2Tracker(device='cuda',checkpoint=runtime['sam']);summaries=[]
        try:
            for eid in choices:
                e=entries[eid];geometry=CurrentGeometry(runtime_root/'inputs',read(old/'normalizer.json')['schema']);policy=WristPegReanchor(detector,tracker,geometry,refine=refine);policy.reset(eid)
                folder=root/eid;folder.mkdir();rows=[];costs=[];maskbits=[]
                write(root/'pipeline_state.json',dict(status='RUNNING',stage='DEVELOPMENT_FEASIBILITY_SMOKE',episode=eid,run_id=log.run_id,pid=os.getpid()))
                prior=baseline/'perception'/eid/'masks.npz'
                if not prior.exists():prior=old/'cache'/('ep'+eid)/'graphs.npz'
                rrpath=baseline/'references'/eid/'receipt.json'
                corrected=reference_root/'perception'/eid/'receipt.json'
                if corrected.exists():rrpath=Path(read(corrected)['corrected_reference'])
                if not rrpath.exists():rrpath=reference_root/'replay_reference'/eid/'receipt.json'
                rr=read(rrpath);assert rr['raw_sha256']==e['raw_sha256']
                with np.load(rrpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                try:
                    with h5py.File(e['raw_path']) as f,np.load(prior) as stored:
                        for t in range(min(20,e['actions']+1)):
                            cameras,data,surfaces=geometry.from_h5(f,t,eid);mm=np.unpackbits(stored['mask_bits'][t]).reshape(2,2,128,128).astype(bool);base=dict(zip(CAMERAS,mm))
                            tick=time.perf_counter();changed,event=policy.process(cameras,data,surfaces,base);torch.cuda.synchronize();costs.append((time.perf_counter()-tick)*1000)
                            graph,points,gate=geometry.build(cameras,data,surfaces,changed);maskbits.append(np.packbits(np.stack([changed[c] for c in CAMERAS]).ravel()))
                            if exact[t]:
                                for condition,masks in [('baseline',base),('candidate',changed)]:
                                    m=support(masks['hand_camera'][0],seg[t,1],0,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']])
                                    rows.append(dict(episode=eid,timestep=t,condition=condition,camera='hand_camera',object_id=0,metrics=m))
                            print('reanchor smoke',eid,t,'used',event['correction_used'],'attempted',event['attempted'],flush=True)
                    np.savez_compressed(folder/'candidate_masks.npz',mask_bits=np.asarray(maskbits),proposal_mask_bits=np.asarray(policy.proposal_masks,dtype=np.uint8).reshape(-1,2048),validated_proposal_mask_bits=np.asarray(policy.validated_proposal_masks,dtype=np.uint8).reshape(-1,2048),active_raw_mask_bits=np.asarray([x[1] for x in policy.active_raw_masks],dtype=np.uint8).reshape(-1,2048),active_raw_timestep=np.asarray([x[0] for x in policy.active_raw_masks],np.int64),correction_processing_ms=np.asarray(costs))
                    write(folder/'events.json',policy.events);write(folder/'metrics.json',rows)
                    summary=dict(episode=eid,frames=len(maskbits),reanchored=sum(bool(x.get('reanchored')) for x in policy.events),correction_used=sum(x['correction_used'] for x in policy.events),attempts=sum(x['attempted'] for x in policy.events),
                        baseline_correct=sum(r['metrics']['true_positive'] for r in rows if r['condition']=='baseline'),candidate_correct=sum(r['metrics']['true_positive'] for r in rows if r['condition']=='candidate'),
                        baseline_false_pixels=sum(r['metrics']['false_positive'] for r in rows if r['condition']=='baseline'),candidate_false_pixels=sum(r['metrics']['false_positive'] for r in rows if r['condition']=='candidate'),
                        correction_ms_mean=float(np.mean(costs)),correction_ms_p95=float(np.percentile(costs,95)),latency_scope='concurrent baseline experiment, correction-only development timing; not final deployment benchmark')
                    summaries.append(summary);write(root/'smoke_progress.json',summaries)
                finally:policy.close()
            write(root/'smoke_summary.json',dict(status='FEASIBILITY_SMOKE_FINISHED',rows=summaries,not_independent_validation=True,not_candidate_selection=True,student_GT_used=False));log.link_output(root/'smoke_summary.json',kind='metrics')
            write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='DEVELOPMENT_FEASIBILITY_SMOKE_FINISHED',completed_goal=False,run_id=log.run_id))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);p.add_argument('reference_root',type=Path);p.add_argument('--refine',action='store_true');a=p.parse_args();run(a.root,a.baseline,a.reference_root,a.refine)

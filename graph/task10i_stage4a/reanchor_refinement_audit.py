"""Post-decision raw SAM2 -> depth/robot -> shared-surface retention for V2.

Only saved masks and current sensor geometry are replayed; no model forwards,
GT-triggered decisions or inference/config changes occur.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import time
import h5py
import numpy as np
from .run_pipeline import CAMERAS,STAGE3,read,write
from .instrumentation import support


def run(output,final,audits):
    from utils.experiment_logger import ExperimentLogger
    from graph.persistent_geometry.contracts import CameraObservation
    from .reliable_reanchor import refine_shared_surface,raw_support_guard
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_raw_reanchor_refinement_retention',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(model_forwards=0,post_selection_diagnostic=True,no_candidate_retuning=True)) as log:
        write(output/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_FROZEN_EVALUATION',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        dev=Path(read(final/'handoff.json')['selected_development_root'])
        jobs=[('development',audits/'baseline',audits/'corrected',dev)]
        jobs += [(p,final/p/'baseline',final/p/'corrected',final/p/'candidate') for p in ('internal_test','validation')]
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        cfg=read(Path(read(final/'handoff.json')['final_source_root'])/'graph/plan_a_geometry/protocol.json')['gate']
        rows=[];checks=Counter()
        for partition,base,corrected,candidate in jobs:
            for p in sorted((candidate/'perception').glob('*/receipt.json')):
                cr=read(corrected/'perception'/p.parent.name/'receipt.json');eid=str(cr['episode_id']);e=entries[eid]
                rrpath=Path(cr['corrected_reference']);rr=read(rrpath);events=read(p.parent/'events.json')
                with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                with np.load(p.parent/'correction_masks.npz',allow_pickle=False) as z:
                    raw_proposals=z['raw_proposal_bits'].copy();refined_proposals=z['refined_proposal_bits'].copy()
                    active={int(t):bits.copy() for t,bits in zip(z['active_raw_timestep'],z['active_raw_bits'])}
                with h5py.File(e['raw_path'],'r') as f,np.load(base/'perception'/eid/'masks.npz',allow_pickle=False) as bm,np.load(p.parent/'masks.npz',allow_pickle=False) as cm:
                    for t,event in enumerate(events):
                        proposals=[r for r in event['proposals'] if 'proposal_mask_index' in r]
                        if t not in active and not proposals:continue
                        cameras=[CameraObservation(eid,t,c,float(f['timestamp'][t]),f[c+'/rgb'][t],f[c+'/depth_raw_mm'][t],f[c+'/intrinsic'][t],f[c+'/extrinsic_cv'][t],f[c+'/cam2world_gl'][t],f['tcp_pose'][t]) for c in CAMERAS]
                        base_masks=np.unpackbits(bm['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
                        with np.load(base/'traces'/eid/f't{t:04d}/support.npz',allow_pickle=False) as z:
                            def mask(key):
                                out=np.zeros((128,128),bool);out.ravel()[z[key]]=True;return out
                            # Native prepare.clean removes Robot/overlap, but
                            # retains invalid-depth mask pixels. Its dilation
                            # is not the depth-clamped stagewise support.
                            main=base_masks[0,0]&~base_masks[0,1]&~mask('base_camera__confirmed_robot')&~mask('base_camera__uncertain_robot')
                            robot=dict(confirmed_robot=mask('hand_camera__confirmed_robot'),uncertain=mask('hand_camera__uncertain_robot'))
                        clean={'base_camera':np.stack([main,np.zeros_like(main)])};wrist=cameras[1]
                        final_mask=np.unpackbits(cm['mask_bits'][t]).reshape(2,2,128,128)[1,0].astype(bool)
                        box=base_masks[1,1]
                        conditions=[('proposal',r['proposal_mask_index'],raw_proposals[r['proposal_mask_index']],r) for r in proposals]
                        if t in active:conditions.append(('active_propagation',-1,active[t],dict(raw_support_guard=event['raw_support_guard'],refinement=event['propagation_refinement'])))
                        for kind,index,bits,detail in conditions:
                            raw=np.unpackbits(bits).reshape(128,128).astype(bool)
                            refined,trace=refine_shared_surface(raw,cameras,clean,robot)
                            passed,guard=raw_support_guard(raw,wrist,box,robot,cfg)
                            assert trace==detail['refinement'] and guard==detail['raw_support_guard']
                            if kind=='proposal':np.testing.assert_array_equal(refined,np.unpackbits(refined_proposals[index]).reshape(128,128))
                            selected=bool(kind=='active_propagation' and event['correction_used'] and not event.get('reanchored',False))
                            if selected:np.testing.assert_array_equal(refined,final_mask)
                            checks['replayed_refinements']+=1;checks['selected_active_masks_verified']+=int(selected)
                            stages=dict(raw_sam2=raw,depth_valid=raw&np.isfinite(wrist.depth),
                                robot_removed=raw&np.isfinite(wrist.depth)&~robot['confirmed_robot']&~robot['uncertain'],shared_surface_refined=refined)
                            for stage,m in stages.items():
                                rows.append(dict(partition=partition,episode=eid,parent_id=e['parent_id'],timestep=t,kind=kind,proposal_index=index,
                                    selected_active=selected,proposal_accepted=detail.get('accepted'),raw_guard_passed=passed,stage=stage,
                                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                                    metrics=support(m,seg[t,1],0,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']]) if exact[t] else None))
        totals=defaultdict(Counter)
        for r in rows:
            if r['metrics'] is None:continue
            for scope in ('all_alternatives','selected_active_only' if r['selected_active'] else 'not_selected_active'):
                c=totals[r['partition'],r['kind'],scope,r['stage']];c['mask_instances']+=1
                c.update({k:r['metrics'][k] for k in ('input_gt_support','output_support','true_positive','false_positive','robot_pixels','wrong_object_pixels','background_pixels')})
        write(output/'refinement_rows.json',rows)
        write(output/'retention_counts.json',[dict(partition=k[0],kind=k[1],scope=k[2],stage=k[3],**v) for k,v in totals.items()])
        write(output/'scope.json',dict(status='RAW_REANCHOR_REFINEMENT_RETENTION_FINISHED',run_id=log.run_id,**checks,
            no_model_forwards=True,no_candidate_changes=True,GT_labels_only_after_replaying_observation_based_transform=True,
            proposal_counts_not_independent_frames=True,selected_active_mask_counts_separate_from_rejected_alternatives=True,
            unsupported_rows_retained=True))
        log.link_output(output/'scope.json',kind='metrics');log.link_output(output/'retention_counts.json',kind='metrics')
        write(output/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='RAW_REANCHOR_REFINEMENT_RETENTION_FINISHED',run_id=log.run_id))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('final',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.final,a.audits)

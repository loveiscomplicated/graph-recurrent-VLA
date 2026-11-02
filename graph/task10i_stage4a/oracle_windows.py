"""O0--O4 diagnosis on registered failure/positive-control causal prefixes."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(root,baseline,corrected,choices=None):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    from .reliable_reanchor import AnchorState
    from .detector_audit import bbox
    from .stagewise import trace_geometry
    from .reevaluate_references import updated_points
    from graph.persistent_geometry.metrics import sample_meshes
    torch.set_num_threads(1);torch.manual_seed(42);root.mkdir(parents=True,exist_ok=False)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    if choices is None:choices={eid:dict(episode=eid,last_timestep=19,reason=reason) for eid,reason in
        [('10i_stage3_10i_s1_0362_full_task','known wrong Wrist mask'),('10i_stage3_10i_s1_0666_full_task','known Wrist initial missing'),('2000003','positive control')]}
    else:choices={k:dict(v) for k,v in choices.items()}
    if len(choices)==3 and all(eid in choices for eid in ('10i_stage3_10i_s1_0362_full_task','10i_stage3_10i_s1_0666_full_task','2000003')):
        failure=read(corrected/'failure_case_inventory.json')['cases']
        for category in ('sam2','robot_uncertain_removed','gate_accepted'):
            candidates=[r for r in failure if r['first_zero_true_support_stage']==category and r['episode'] not in choices]
            if candidates:
                r=min(candidates,key=lambda r:(str(r['episode']),r['timestep']))
                choices[r['episode']]=dict(episode=r['episode'],last_timestep=r['timestep']+5,reason=category)
    for eid,w in choices.items():w['last_timestep']=min(w['last_timestep'],entries[eid]['actions'])
    write(root/'registered_windows.json',dict(windows=list(choices.values()),selection_before_oracle_forward=True,
         scope='six or fewer Development failure/positive-control full causal prefixes, not a population estimate'))
    rt=Path('/workspace/task10g_f1_abc_20261031/full');schema=read(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'))['schema']
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_O0_O4_causal_diagnostics',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(condition_type='ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE',production_adoption=False)) as log:
        tracker=SAM2Tracker(device='cuda',checkpoint=read(rt/'runtime_paths.json')['sam'])
        allrows=[];allpoints=[];summary=[]
        try:
            for eid,window in choices.items():
                e=entries[eid];rrpath=Path(read(corrected/'perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
                with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:segs=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                with np.load(baseline/'references'/eid/'geometry_reference.npz',allow_pickle=False) as z:poses=z['actor_pose'].copy()
                meshes=sample_meshes(baseline/'references'/eid/'meshes.npz',samples_per_object=10000)
                records=read(baseline/'perception'/eid/'vision_records.json');known={c:set() for c in CAMERAS};states={}
                geoms={condition:CurrentGeometry(rt/'inputs',schema) for condition in ('O1','O2','O3','O4')}
                original=geoms['O4'].system.prepare
                def bypass(*args):
                    out=original(*args);out['conditions']['C3']={c:m.copy() for c,m in out['conditions']['C2'].items()};out['gate']['diagnostic_bypass']=True;return out
                geoms['O4'].system.prepare=bypass
                try:
                    with h5py.File(e['raw_path'],'r') as f,np.load(baseline/'perception'/eid/'masks.npz',allow_pickle=False) as cached:
                        for t in range(window['last_timestep']+1):
                            assert exact[t]
                            cameras,data,surfaces=geoms['O1'].from_h5(f,t,eid)
                            truth={c:np.stack([np.isin(segs[t,ci],rr['object_ids'][name]) for name in ('peg','box')]) for ci,c in enumerate(CAMERAS)}
                            o1={c:np.zeros((2,128,128),bool) for c in CAMERAS}
                            for ci,c in enumerate(CAMERAS):
                                record=next(r for r in records if r['camera']==c and r['timestep']==t)
                                known[c].update(record['new_object_ids'])
                                for oid in (0,1):
                                    key=(c,oid)
                                    if key not in states and truth[c][oid].any():states[key]=AnchorState(tracker,f[c+'/rgb'][t],bbox(truth[c][oid]),t)
                                    elif key in states:states[key].append(f[c+'/rgb'][t],t)
                                    if key in states:o1[c][oid]=states[key].current()
                            conditions=dict(O1=o1,O2={c:np.stack([truth[c][oid] if oid+1 in known[c] else np.zeros((128,128),bool) for oid in (0,1)]) for c in CAMERAS},
                                O3=truth,O4=dict(zip(CAMERAS,np.unpackbits(cached['mask_bits'][t]).reshape(2,2,128,128).astype(bool))))
                            for condition,masks in conditions.items():
                                folder=root/'traces'/condition/eid/f't{t:04d}'
                                _,_,rows=trace_geometry(geoms[condition],cameras,data,surfaces,masks,dict(zip(CAMERAS,segs[t])),
                                    [rr['object_ids']['peg'],rr['object_ids']['box']],rr['robot_ids'],folder)
                                for r in rows:r.update(condition=condition,parent_id=e['parent_id'],condition_type='ORACLE_DIAGNOSTIC')
                                allrows.extend(rows)
                                with np.load(folder/'support.npz',allow_pickle=False) as arrays:
                                    pp=updated_points(arrays,read(folder/'trace.json'),dict(zip(CAMERAS,segs[t])),rr,cameras,meshes,poses,t,e)
                                for r in pp:r.update(condition=condition,condition_type='ORACLE_DIAGNOSTIC')
                                allpoints.extend(pp)
                    last=window['last_timestep'];allrows.extend(dict(r,condition='O0',condition_type='ORACLE_DIAGNOSTIC') for r in read(corrected/'perception'/eid/'stage_rows.json') if r['timestep']<=last)
                    allpoints.extend(dict(r,condition='O0',condition_type='ORACLE_DIAGNOSTIC') for r in read(corrected/'perception'/eid/'representative_points.json') if r['timestep']<=last)
                    summary.append(dict(**window,frames=last+1,parent_id=e['parent_id']))
                    write(root/'progress.json',summary);print('oracle',len(summary),'/',len(choices),eid,flush=True)
                finally:
                    for state in states.values():state.close()
            write(root/'stage_rows.json',allrows);write(root/'representative_points.json',allpoints)
            write(root/'oracle_summary.json',dict(status='ORACLE_DIAGNOSTICS_FINISHED',condition_type='SIM_ONLY_ORACLE',windows=summary,
                O1='GT box at first currently GT-visible frame per object/camera, then causal SAM2 propagation, no GT mask correction',
                O2='GT visible mask only after original detector initialized that object; current/past tracking_groups availability',
                O3='GT-visible masks unconditionally, unchanged downstream',O4='original masks with Wrist gate bypass only; downstream filters remain',
                practical_improvement_claim=False,not_full_dataset_estimate=True))
            log.link_output(root/'oracle_summary.json',kind='metrics')
        finally:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path)
    a=p.parse_args();run(a.root,a.baseline,a.corrected)

"""Stage4A task-owned audit coordinator. Never changes source observations."""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGE3=Path('/workspace/task10i_stage3_20261031_v5_finalization/run')
CAMERAS=('base_camera','hand_camera')

def logical_bytes(p):
    p=Path(p)
    if p.exists():return p.read_bytes()
    compressed=p.with_name(p.name+'.gz')
    if compressed.exists():
        with gzip.open(compressed,'rb') as f:return f.read()
    raise FileNotFoundError(p)

def read(p): return json.loads(logical_bytes(p))
def sha(p):
    h=hashlib.sha256()
    path=Path(p)
    if not path.exists() and path.with_name(path.name+'.gz').exists():
        stream=gzip.open(path.with_name(path.name+'.gz'),'rb')
    else:stream=path.open('rb')
    with stream as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rank(p): return hashlib.sha256(('task10i-stage4a-v1:'+p).encode()).hexdigest()

def audit(root):
    import h5py
    import numpy as np
    receipt=read(STAGE3/'stage4_raw_dataset_receipt.json')
    assert receipt['STAGE4_READY']
    manifest=STAGE3/'stage4_recache_input_manifest.json';contract=STAGE3/'stage4_source_and_schema_contract.json'
    assert sha(manifest)==receipt['recache_manifest_sha256'] and sha(contract)==receipt['schema_sha256']
    m=read(manifest);entries=list(m['unique_sources'].values())
    for k,v in m['datasets'].items(): assert sha(v['path'])==v['sha256']
    train=sorted({e['parent_id'] for e in entries if e['split']=='train'},key=rank)
    val=sorted({e['parent_id'] for e in entries if e['split']=='validation'})
    assert not set(train)&set(val)
    protocol=dict(version=1,development=train[:24],internal_test=train[24:48],train_reserve=train[48:],
        independent_validation=val,validation_pixels_opened=False,selection_before_validation=True,
        selection=dict(minimum_parent_macro_visible_graph_recall_gain=.02,maximum_pixel_precision_loss=.005,
            maximum_robot_contamination_increase=.002,maximum_wrong_object_contamination_increase=.002,
            maximum_false_object_presence_increase=.005,maximum_box_graph_recall_loss=.005,
            maximum_p95_latency_ratio=2.,decision='all safety constraints AND recall gain; otherwise NO_GO or TRADEOFF_UNRESOLVED'),
        gt_support_strata=[0,1,8,32,128],visibility_minimum_pixels=8,
        frame_scope='all T+1 observations; terminal frames have no action; development-selected temporal windows must retain full causal prefix',
        evaluation='paired same raw/GT; episode macro then parent macro; dedup logical source; phase evaluation only',
        forbidden_inputs=['GT','future','expert_phase','association_labels'],production_adoption=False,
        oracle_label='ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE',reference_acceptance='EXACT_GT only if both cameras full RGB/depth/calibration array-equal at that frame; otherwise GT_UNSUPPORTED',
        task10g_final_test_access=False,coverage_holdout60_access=False)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    rows=[]
    for e in entries:
        # No validation image reads before candidate freeze. Byte hashes/headers only.
        assert sha(e['raw_path'])==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
        with h5py.File(e['raw_path'],'r') as f:
            schema={};f.visititems(lambda k,v:schema.update({k:dict(shape=list(v.shape),dtype=str(v.dtype))}) if isinstance(v,h5py.Dataset) else None)
            segments=[k for k in schema if 'segment' in k.lower() and k!='segment']
            assert all(f[c+'/rgb'].shape==(e['actions']+1,128,128,3) for c in CAMERAS)
            row=dict(episode_id=e['episode_id'],parent_id=e['parent_id'],split=e['split'],source=e['source'],branch=e['branch'],
                actions=e['actions'],observations=e['actions']+1,actor_segmentation_fields=segments,schema=schema,
                raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256'])
        rows.append(row)
    source=Path(__file__).resolve().parents[2]
    files={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils','envs','recurrent','legacy_fps') for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.toml')}
    write(root/'source_freeze.json',dict(files=files,source_root=str(source),digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()))
    write(root/'source_data_audit.json',dict(status='PASS',receipt_sha256=sha(STAGE3/'stage4_raw_dataset_receipt.json'),
        manifest_sha256=sha(manifest),schema_sha256=sha(contract),unique_logical=len(rows),unique_observations=sum(r['observations'] for r in rows),
        unique_actions=sum(r['actions'] for r in rows),train_parents=len(train),validation_parents=len(val),
        segmentation_stored_episodes=sum(bool(r['actor_segmentation_fields']) for r in rows),rows=rows))
    print('SOURCE_AND_DATA_AUDIT PASS',len(rows),flush=True)

def reference_probe(root):
    import h5py
    import numpy as np
    import torch
    from mani_skill.utils.structs.pose import Pose
    from graph.geometry_reach10d.runtime import environment
    from graph.integration_baseline.direct_train import seed_all
    seed_all(42,1)
    entries=list(read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values())
    dev=set(read(root/'protocol.json')['development'])
    chosen=[]
    # Each source family and branch family represented without validation inspection.
    for kind in sorted({(e['source'],e['branch']=='full_task') for e in entries if e['parent_id'] in dev}):
        pool=sorted([e for e in entries if e['parent_id'] in dev and (e['source'],e['branch']=='full_task')==kind],key=lambda e:str(e['episode_id']))
        chosen.append(pool[0])
    env=environment(Path('/workspace/task10g_f1_abc_20261031/full'));rows=[]
    try:
        for e in chosen:
            seed=e.get('parent_seed')
            if seed is None: seed=read(Path(e['raw_path']).parent.parent/'parent_result.json')['seed']
            obs,_=env.reset(seed=int(seed));task=env.unwrapped
            initial_hole=task.box_hole_pose.raw_pose[0].cpu().numpy().copy()
            ids={int(k):v.name for k,v in task.segmentation_id_map.items()}
            mapping={name:[k for k,n in ids.items() if n.startswith(name)] for name in ('peg','box')}
            assert all(mapping.values()) and not set(mapping['peg'])&set(mapping['box'])
            with h5py.File(e['raw_path']) as f,np.load(e['gt_path'],allow_pickle=False) as z:
                # Frame restore is render-only. No physics/action or student computation.
                for t in sorted(set([0,e['actions']//3,2*e['actions']//3,e['actions']])):
                    robot=f['robot_state'][t]
                    task.agent.robot.set_qpos(torch.tensor(robot[:9])[None])
                    task.agent.robot.set_qvel(torch.tensor(robot[9:18])[None])
                    task.peg.set_pose(Pose.create(torch.tensor(z['raw_peg_pose'][t])[None]))
                    # hole = box * randomized fixed local hole offset; no GT student input.
                    if 'physics_box_pose' in z:
                        task.box.set_pose(Pose.create(torch.tensor(z['physics_box_pose'][t])[None]))
                    elif not np.array_equal(z['raw_hole_pose'][t],initial_hole):
                        task.box.set_pose(Pose.create(torch.tensor(z['raw_hole_pose'][t])[None])*task.box_hole_offsets.inv())
                    task.scene.update_render();obs=task.get_obs()
                    comparisons={};segs={}
                    for c in CAMERAS:
                        sd=obs['sensor_data'][c];sp=obs['sensor_param'][c]
                        vals=dict(rgb=sd['rgb'][0],depth_raw_mm=sd['depth'][0,...,0],intrinsic=sp['intrinsic_cv'][0],extrinsic_cv=sp['extrinsic_cv'][0],cam2world_gl=sp['cam2world_gl'][0])
                        checks={}
                        for k,v in vals.items():
                            a=v.detach().cpu().numpy();b=f[c+'/'+k][t]
                            checks[k]=dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
                        seg=task._sensors[c].get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'][0,...,0].cpu().numpy()
                        segs[c]=seg;comparisons[c]=checks
                    exact=all(x['exact'] for c in comparisons.values() for x in c.values())
                    folder=root/'reference_probe'/str(e['episode_id']);folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f't{t:04d}.npz',**{c+'_segmentation':s for c,s in segs.items()},**{c+'_render_rgb':obs['sensor_data'][c]['rgb'][0].cpu().numpy() for c in CAMERAS})
                    rows.append(dict(episode_id=e['episode_id'],parent_id=e['parent_id'],source=e['source'],branch=e['branch'],timestep=t,
                        reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',comparisons=comparisons,actor_ids=ids,object_ids=mapping,
                        visible_support={c:{n:int(np.isin(s,mapping[n]).sum()) for n in mapping} for c,s in segs.items()}))
                    write(root/'gt_reference_probe.json',dict(status='RUNNING',rows=rows))
                    print('reference',e['episode_id'],t,rows[-1]['reference_kind'],flush=True)
    finally:env.close()
    write(root/'gt_reference_probe.json',dict(status='PASS' if all(r['reference_kind']=='EXACT_GT' for r in rows) else 'REFERENCE_RECONSTRUCTION_REQUIRES_INVESTIGATION',rows=rows,physics_steps=0,student_GT_used=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from utils.experiment_logger import ExperimentLogger
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_perception_investigation',seed=42,root=root/'experiments',config=dict(stage3_root=str(STAGE3),production_changes=False),job_type='evaluation') as log:
            write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source=str(Path(__file__).resolve())))
            try:
                from .baseline import run as baseline
                from .replay_reference import run as replay
                from .diagnose_cached import run as diagnose
                for stage,fn in [('SOURCE_AND_DATA_AUDIT',audit),('BASELINE_EXACT_REPRODUCTION',baseline),('GT_REFERENCE_FEASIBILITY_PROBE',reference_probe),('GT_REFERENCE_VALIDATION',replay),('STAGEWISE_FAILURE_INSTRUMENTATION',diagnose)]:
                    write(root/'pipeline_state.json',dict(stage=stage,status='RUNNING',run_id=log.run_id,pid=os.getpid()))
                    fn(root)
                    artifact={audit:'source_data_audit.json',baseline:'baseline_forward_equivalence.json',reference_probe:'gt_reference_probe.json',replay:'replay_reference_probe.json',diagnose:'stagewise_development_summary.json'}[fn]
                    log.link_output(root/artifact,kind='metrics')
                write(root/'pipeline_state.json',dict(stage='FAILURE_CASE_COLLECTION_AND_ROOT_CAUSE_QUANTIFICATION',status='PENDING_IMPLEMENTATION',run_id=log.run_id,completed=False))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(stage=stage,status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':main()

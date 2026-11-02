"""Capture fresh benchmark RGB-D and synchronous GT after explicit source freeze.

No perception model runs here. Initial eligibility reuses unchanged Stage1 gates;
expert success, object-mask performance and later visibility never select parents.
This entrypoint is deliberately not launched by development experiments.
"""
import argparse
import fcntl
import importlib.metadata
import os
from pathlib import Path
import pickle
import time
import h5py
import numpy as np
import torch
from .gt_reference import simultaneous_sidecar,array
from .stress_capture import VisualOccluders,occlusion_active,active_cameras,state
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def capture_identity():
    source=Path(__file__).resolve().parents[2]
    files={str(p):sha(p) for folder in ('graph','visual/runtime','utils','recurrent','envs/maniskill')
        for p in (source/folder).rglob('*.py')}
    for p in [Path('/workspace/task10i_stage1_20261031/code/pipeline.py'),
              Path('/workspace/task10i_stage1_20261031/audit/geometry.py'),
              Path('/workspace/task10i_stage1_20261031/audit/protocol.json')]:files[str(p)]=sha(p)
    packages={name:importlib.metadata.version(name) for name in ('torch','numpy','mani_skill','sapien','h5py','scipy')}
    return dict(files=files,packages=packages,device='CPU PhysX / CUDA SAPIEN')


def freeze_barrier(directory,protocol,design,scope='final'):
    """Reject missing/changed freeze receipts before any registered seed reset."""
    frozen={name:read(directory/(name+'.json')) for name in
        ('candidate_source_frozen','candidate_config_frozen','adoption_criteria_frozen')}
    assert all(r['status']=='FROZEN' for r in frozen.values())
    config=frozen['candidate_config_frozen']
    if scope=='final':
        assert config['allow_independent_final_capture'] is True
        assert config['new_final_benchmark_protocol_sha256']==sha(protocol)
        assert config['final_collection_design_sha256']==sha(design)
    elif scope=='internal':
        assert config['allow_internal_evaluation'] is True
    else:raise ValueError('unknown independent evaluation scope')
    assert config['adoption_criteria_sha256']==sha(directory/'adoption_criteria_frozen.json')
    source=frozen['candidate_source_frozen']
    assert source['files'] and all(sha(p)==h for p,h in source['files'].items())
    return {name:sha(directory/(name+'.json')) for name in frozen}


def eligibility(env,entry,folder,s1):
    from geometry import evaluate
    d,snapshot,timing=s1.capture(env,entry['seed'])
    visible,evidence=evaluate(s1.camera(d,entry['parent_id']),d['peg_pose'],d['peg_half'],d['tcp_pose'])
    physical=s1.physical(env,d,snapshot)
    np.savez_compressed(folder/'initial.npz',**d)
    np.savez_compressed(folder/'visibility_evidence.npz',**evidence)
    with (folder/'snapshot.pkl').open('wb') as stream:pickle.dump(snapshot,stream,protocol=5)
    replay=dict(status='NOT_REQUIRED')
    if visible['requested_pass'] and physical['status']=='PHYSICAL_VALID':
        env.reset(seed=42424242)
        actual,other,_=s1.capture(env,entry['seed'])
        checks=s1.compare(d,actual)
        sim=s1.ahash(snapshot['sim_state'])==s1.ahash(other['sim_state'])
        controller=s1.ahash(snapshot['controller'])==s1.ahash(other['controller'])
        rv,_=evaluate(s1.camera(actual,entry['parent_id']),actual['peg_pose'],actual['peg_half'],actual['tcp_pose'])
        ok=all(v['exact'] for v in checks.values()) and sim and controller and rv['requested_pass']==visible['requested_pass']
        replay=dict(status='REPLAY_PASS' if ok else 'REPLAY_FAIL',checks=checks,
            sim_state_exact=sim,controller_exact=controller,visibility_exact=rv['requested_pass']==visible['requested_pass'])
    record=dict(**entry,visibility=visible,physical=physical,replay=replay,timing=timing,
        gate_pass=visible['requested_pass'] and physical['status']=='PHYSICAL_VALID' and replay['status']=='REPLAY_PASS',
        physical_state_sha256=s1.ahash({k:d[k] for k in
            ('peg_pose','peg_half','box_pose','hole_pose','hole_radius','qpos','qvel','tcp_pose')}),
        expert_success_selection=False,mask_performance_selection=False)
    write(folder/'eligibility.json',record)
    return record


def sequence(env,entry,folder,stress_stratum,horizon):
    from recurrent.expert import ScriptedExpert
    from graph.integration_baseline.collect_recurrent_expert import capture,append_h5
    from graph.integration_baseline.robot import robot_observation,state_schema
    from graph.task10i_stage4a.development_reference import mesh_reference
    observation=env.unwrapped.get_obs();task=env.unwrapped
    schema=state_schema(robot_observation(env)[0]);expert=ScriptedExpert(env)
    slab=VisualOccluders(env,stress_stratum) if stress_stratum else None
    refs=[];segs=[];trace=[];peg=[];box=[];hole=[];radii=[]
    folder.mkdir(parents=True,exist_ok=False)
    with h5py.File(folder/'rgbd.h5','w') as raw:
        raw.attrs['benchmark_only']=True;raw.attrs['GT_sidecar_input']=False
        raw.attrs['parent_id']=entry['parent_id'];raw.attrs['reset_seed']=entry['seed']
        for t in range(horizon+1):
            assert int(task.elapsed_steps.item())==t
            if slab:
                before=state(env);slab.place(observation,occlusion_active(t))
                observation=task.get_obs()
                for key,value in before.items():np.testing.assert_array_equal(value,state(env)[key])
            row,exact=capture(env,observation,schema);assert exact
            # RGB-D and proprioception only. Actor labels/poses remain separate.
            assert not any('segmentation' in name or 'peg_pose' in name or 'hole_pose' in name for name in row)
            for key,value in row.items():append_h5(raw,key,value)
            sidecar,receipt=simultaneous_sidecar(env,observation)
            assert receipt['reference_kind']=='EXACT_GT' and receipt['timestep']==t
            refs.append(receipt);segs.append(sidecar['actor_segmentation'])
            peg.append(array(task.peg.pose.raw_pose)[0]);box.append(array(task.box.pose.raw_pose)[0])
            hole.append(array(task.box_hole_pose.raw_pose)[0])
            radii.append(array(task.box_hole_radii)[0])
            phase=expert.phase
            item=dict(timestep=t,expert_phase=phase,stress_stratum=stress_stratum,
                artificial_occlusion=bool(slab and occlusion_active(t)),
                intended_cameras=list(active_cameras(stress_stratum)) if slab else [],
                termination_policy='continue fixed horizon, no expert-success selection')
            limits=array(task.agent.robot.get_qlimits())[0]
            qpos=array(task.agent.robot.get_qpos())[0]
            item['finite_physical_state']=all(np.isfinite(v).all() for v in state(env).values())
            item['joint_limits_valid']=bool(((qpos>=limits[:,0]-1e-5)&(qpos<=limits[:,1]+1e-5)).all())
            contacts=[]
            for contact in task.scene.px.get_contacts():
                contacts.append(dict(names=[body.entity.name for body in contact.bodies],
                    min_separation_m=min((float(p.separation) for p in contact.points),default=0.)))
            item['current_contacts']=contacts
            item['contact_scope']='current boundary contacts; not exhaustive physics-substep safety certification'
            if t<horizon:
                action=np.asarray(expert.act(),np.float32)
                assert action.shape==(7,) and np.isfinite(action).all()
                append_h5(raw,'actions',action)
                observation,reward,terminated,truncated,info=env.step(torch.tensor(action)[None])
                item.update(success=bool(task.evaluate()['success'].item()),
                    terminated=bool(terminated.item()),truncated=bool(truncated.item()))
                assert not bool(truncated.item()),'fixed horizon exceeds installed time limit'
            trace.append(item)
        raw.flush()
    np.savez_compressed(folder/'gt_geometry.npz',raw_peg_pose=np.asarray(peg),
        physics_box_pose=np.asarray(box),raw_hole_pose=np.asarray(hole),raw_hole_radius=np.asarray(radii))
    mesh_reference(task,folder,dict(gt_path=str(folder/'gt_geometry.npz')))
    np.savez_compressed(folder/'segmentation.npz',actor_segmentation=np.asarray(segs),exact_frame=np.ones(horizon+1,bool))
    write(folder/'trace.json',trace)
    rr=dict(episode_id=entry['parent_id'],parent_id=entry['parent_id'],source='new_independent_benchmark',
        branch='stress_'+stress_stratum if slab else 'normal_full_task',seed=entry['seed'],
        raw_path=str(folder/'rgbd.h5'),raw_sha256=sha(folder/'rgbd.h5'),
        gt_path=str(folder/'gt_geometry.npz'),gt_sha256=sha(folder/'gt_geometry.npz'),
        actions=horizon,frames=horizon+1,exact_frames=horizon+1,total_frames=horizon+1,
        object_ids=refs[0]['object_ids'],robot_ids=refs[0]['robot_ids'],actor_ids=refs[0]['actor_ids'],
        rows=refs,GT_forward_input=False,synchronous_capture=True,
        stratum='stress' if slab else 'normal',stress_stratum=stress_stratum,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    with h5py.File(folder/'rgbd.h5') as raw:
        np.testing.assert_array_equal(raw['timestep'][:],np.arange(horizon+1))
        assert len(raw['actions'])==horizon
        assert all(len(raw[c+'/rgb'])==len(raw[c+'/depth_raw_mm'])==horizon+1 for c in CAMERAS)
    with np.load(folder/'segmentation.npz') as z:
        assert z['actor_segmentation'].shape==(horizon+1,2,128,128) and z['exact_frame'].all()
    write(folder/'receipt.json',rr)
    return rr


def run(root,protocol,design,freeze):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage3.worker import initial_api
    from graph.geometry_reach10d.runtime import environment
    barriers=freeze_barrier(freeze,protocol,design)
    plan=read(design);registration=read(protocol)
    assert plan['status']=='FROZEN' and plan['stress_engineering_pass']
    assert plan['normal_parents']>=32 and plan['stress_parents']>=12
    assert plan['horizon']==120 and plan['occlusion_steps']==list(range(8,20))
    assert sha(plan['stress_engineering_scope'])==plan['stress_engineering_scope_sha256']
    assert read(plan['stress_engineering_scope'])['status']=='STRESS_CAPTURE_ENGINEERING_COMPLETE'
    # Conservative RGB-D+sidecar+metadata budget; no deletion or paid allocation.
    import shutil
    required=int(plan['storage_reserve_gib']*2**30)
    if shutil.disk_usage(root.parent).free<required:raise OSError('insufficient reserved benchmark storage')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(freezes=barriers,protocol_sha256=sha(protocol),design_sha256=sha(design),
            source_identity=capture_identity())
        if (root/'source_config_frozen.json').exists():assert read(root/'source_config_frozen.json')==identity
        else:write(root/'source_config_frozen.json',identity)
        torch.set_num_threads(1);torch.manual_seed(42)
        s1=initial_api();accepted=set();entries=[];attempts=[]
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_independent_benchmark_capture',
                seed=42,root=root/'experiments',job_type='evaluation',config=identity) as logger:
            for stratum,key,count in [('normal','normal_conditional_random',plan['normal_parents']),
                    ('stress','occlusion_reappearance_negative_stress',plan['stress_parents'])]:
                completed=0
                for entry in registration['queues'][key]:
                    if completed>=count:break
                    folder=root/'attempts'/entry['parent_id'];folder.mkdir(parents=True,exist_ok=True)
                    record_path=folder/'record.json'
                    if record_path.exists():
                        record=read(record_path)
                        assert all(sha(folder/p)==h for p,h in record['files'].items())
                        attempts.append(record)
                        if record['eligible']:
                            assert record['physical_state_sha256'] not in accepted
                            accepted.add(record['physical_state_sha256']);completed+=1
                            receipt=read(folder/'sequence/receipt.json')
                            assert all(sha(folder/'sequence'/p)==h for p,h in receipt['files'].items())
                            entries.append(receipt)
                        continue
                    # Incomplete episode cannot resume from unrecorded hidden physics.
                    if (folder/'eligibility.json').exists() or (folder/'sequence').exists():
                        folder.rename(folder.with_name(folder.name+'.partial_'+str(time.time_ns())))
                        folder.mkdir()
                    write(root/'pipeline_state.json',dict(status='CAPTURING',stratum=stratum,
                        parent_id=entry['parent_id'],accepted=completed,target=count,pid=os.getpid()))
                    env=environment(Path('/workspace/task10g_f1_abc_20261031/full'))
                    try:
                        record=eligibility(env,entry,folder,s1)
                        record['eligible']=record['gate_pass'] and record['physical_state_sha256'] not in accepted
                        if record['eligible']:
                            stress=('main','wrist','both')[completed%3] if stratum=='stress' else None
                            receipt=sequence(env,entry,folder/'sequence',stress,plan['horizon'])
                            accepted.add(record['physical_state_sha256']);completed+=1;entries.append(receipt)
                        record['files']={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
                        write(record_path,record);attempts.append(record)
                        write(root/'progress.json',dict(attempted=len(attempts),accepted=len(entries),
                            entries=[{k:v for k,v in e.items() if k not in ('rows','files')} for e in entries]))
                    finally:env.close()
                assert completed==count,('registered queue exhausted',stratum,completed,count)
            assert len(entries)==plan['normal_parents']+plan['stress_parents']
            write(root/'independent_benchmark_manifest.json',dict(status='SYNCHRONOUS_BENCHMARK_CAPTURE_COMPLETE',
                run_id=logger.run_id,source_config=identity,attempts=attempts,
                entries=[{k:v for k,v in e.items() if k not in ('rows','files')} for e in entries],
                normal_parents=plan['normal_parents'],stress_parents=plan['stress_parents'],
                expert_success_filter=False,mask_success_filter=False,student_training_data=False))
            write(root/'pipeline_state.json',dict(status='SYNCHRONOUS_BENCHMARK_CAPTURE_COMPLETE',
                parents=len(entries),frames=sum(e['frames'] for e in entries),whole_goal_complete=False))
            logger.link_output(root/'independent_benchmark_manifest.json',kind='metrics')


def engineering(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage3.worker import initial_api
    from graph.geometry_reach10d.runtime import environment
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    config=dict(engineering_only=True,seed=30080,horizon=25,
        final_seeds_opened=False,final_protocol_read=False,source_sha256=sha(Path(__file__)))
    write(root/'source_identity.json',capture_identity())
    config['source_identity_sha256']=sha(root/'source_identity.json')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_benchmark_collector_engineering',
            seed=42,root=root/'experiments',job_type='evaluation',config=config) as logger:
        s1=initial_api();rows=[]
        for stratum in (None,'main','wrist','both'):
            name=stratum or 'normal';folder=root/name;folder.mkdir()
            env=environment(Path('/workspace/task10g_f1_abc_20261031/full'))
            try:
                entry=dict(parent_id='engineering30080_'+name,seed=30080,attempt=0)
                gate=eligibility(env,entry,folder,s1)
                # Exercise serialization even if this historical seed does not
                # pass Stage1. Such an engineering sequence is never a Final parent.
                env.reset(seed=entry['seed'])
                receipt=sequence(env,entry,folder/'sequence',stratum,25)
                assert receipt['frames']==receipt['exact_frames']==26
                rows.append(dict(stratum=name,initial_gate_pass=gate['gate_pass'],
                    receipt=str(folder/'sequence/receipt.json'),sha256=sha(folder/'sequence/receipt.json')))
            finally:env.close()
        write(root/'scope.json',dict(status='BENCHMARK_COLLECTOR_ENGINEERING_COMPLETE',
            run_id=logger.run_id,frames=104,rows=rows,final_seeds_opened=False,
            exact_synchronous_GT=True,student_training_data=False,whole_goal_complete=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    for name in ('protocol','design','freeze'):p.add_argument(name,type=Path,nargs='?')
    p.add_argument('--engineering',action='store_true')
    p.add_argument('--wait-stress-engineering',type=Path)
    a=p.parse_args()
    try:
        if a.wait_stress_engineering:
            assert a.engineering,'automatic wait is engineering-only'
            deadline=time.monotonic()+24*3600
            while True:
                scope=a.wait_stress_engineering/'scope.json'
                if scope.exists():
                    assert read(scope)['status']=='STRESS_CAPTURE_ENGINEERING_COMPLETE';break
                assert not (a.wait_stress_engineering.parent/'failure.json').exists(),'stress engineering failed'
                live=False
                for path in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        command=path.read_bytes()
                        if b'graph.task10i_stage4a2.stress_capture' in command and str(a.wait_stress_engineering).encode() in command:live=True
                    except OSError:pass
                if not live:raise RuntimeError('live stress engineering dependency missing')
                write(a.root.parent/'wait_state.json',dict(status='WAITING_FOR_LIVE_STRESS_ENGINEERING',pid=os.getpid()))
                if time.monotonic()>deadline:raise TimeoutError('benchmark engineering dependency deadline')
                time.sleep(10)
        if a.engineering:engineering(a.root)
        else:
            assert all((a.protocol,a.design,a.freeze)),'Final requires protocol, design and freeze receipts'
            run(a.root,a.protocol,a.design,a.freeze)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,
            message=str(exc),partial_preserved=True,whole_goal_complete=False))
        raise

"""Opt-in canonical HDF5 collection, causal graphs, frozen direct-data joins.

The Task06-B recorder, original expert, Vision algorithms and model inputs are
unchanged. Each finalized episode/receipt and final manifest is immutable.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from graph.integration_baseline.direct_runtime import (
    atomic, append, read, sha, finalize, check_space, stop_gate, now,
)
from graph.integration_baseline.collect_recurrent_expert import (
    append_h5, capture, reset_for_recording, validate_episode, array, digest,
)
from graph.integration_baseline.data import AlignedFrame
from graph.integration_baseline.io import read_packet, save_packet
from graph.integration_baseline.robot import robot_observation, state_schema, geometry_in_ee, RobotNormalizer
from graph.vision_input import validate_packet, RAW_EDE_POLICY


def load_episode(path):
    with h5py.File(path, 'r') as f:
        keys = []
        f.visititems(lambda k, v: keys.append(k) if isinstance(v, h5py.Dataset) else None)
        return {k: f[k][()] for k in keys}


def export_npz(path, output):
    """Explicit compatibility export; not a second canonical dataset."""
    from graph.integration_baseline.io import save_npz
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    tmp = output.with_name(output.stem+'.partial.npz')
    save_npz(tmp, **load_episode(path)); finalize(tmp, output)


def receipts(run):
    return [read(p) for p in sorted((Path(run)/'collection/receipts').glob('*.json'))]


def validate_receipt(run, e, schema):
    path = Path(run)/e['path']
    if sha(path) != e['sha256']: raise ValueError('episode checksum mismatch')
    a = load_episode(path)
    validate_episode(a, schema, e['seed'])
    if e['actions'] != len(a['actions']) or e['success'] != bool(a['success'][-1]):
        raise ValueError('episode receipt/array mismatch')
    if not np.isfinite(a['actions']).all(): raise ValueError('nonfinite expert action')
    if len(e['phases']) != e['actions']: raise ValueError('diagnostic phase length')
    return a


def collect(run, config, events):
    from envs.maniskill import make_peg_insertion_env, get_gt_state
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.expert import ScriptedExpert
    from recurrent.common import to_action_tensor, versions
    run = Path(run); c = config['collection']
    assert c['noise'] == 0 and c['hole_shift'] is None
    assert c['horizon'] == 250 and c['control_mode'] == 'pd_ee_delta_pose'
    if (run/'dataset_manifest.json').exists():
        m = read(run/'dataset_manifest.json')
        for e in m['episodes']: validate_receipt(run, e, read(run/'robot_state_schema.json'))
        return m
    events.emit('collection_start', target=c['target_successes'], maximum_attempts=c['max_attempts'])
    env = make_peg_insertion_env(obs_mode='rgb+depth', control_mode=c['control_mode'], device='cpu')
    env = TimeLimitWrapper(env.unwrapped, c['horizon'])
    try:
        schema = state_schema(robot_observation(env)[0])
        previous = read('experiments/9fec4f02fc9d4172a48e97d5eff90abd/robot_state_schema.json')
        if schema != previous or schema['dimension'] != 76: raise ValueError('Robot76 schema changed')
        atomic(run/'robot_state_schema.json', schema, immutable=True)
        task = env.unwrapped
        conditions = dict(versions=versions(), backend=str(task.backend), robot=task.agent.uid,
                          control_freq=int(task.control_freq), physics_freq=int(task.sim_freq),
                          horizon=env._max_episode_steps,
                          controller_config={k: asdict(v.config) for k,v in task.agent.controller.controllers.items()})
        if (conditions['control_freq'], conditions['physics_freq']) != (20,100):
            raise ValueError('unexpected simulator frequencies')
        atomic(run/'executed_conditions.json', conditions, immutable=True)
        atomic(run/'expert_action_schema.json', dict(id='pd_ee_delta_pose_regression7', dimension=7,
            dtype='float32', target='native expert = step input = actual controller input',
            clipping=False, components=[3,3,1]), immutable=True)
        source_sha = sha('recurrent/expert.py')
        completed = receipts(run)
        for e in completed: validate_receipt(run,e,schema)
        # Intent without receipt is never silently recollected or counted complete.
        intents = {read(p)['seed'] for p in (run/'collection/intents').glob('*.json')}
        done = {e['seed'] for e in completed}
        for seed in sorted(intents-done):
            partial = run/'collection/partial'/f'ep{seed}.h5.partial'
            final = run/'collection/episodes'/f'ep{seed}.h5'
            if final.exists():
                raise ValueError('final episode without receipt; explicit integrity recovery required')
            interrupted = run/'collection/interrupted'/f'ep{seed}.json'
            atomic(interrupted, dict(seed=seed, episode_id=seed, status='INTERRUPTED',
                   path=str(partial.relative_to(run)), training_eligible=False), immutable=True)
        successes = [e for e in completed if e['success']]
        for index in range(c['max_attempts']):
            if len(successes) >= c['target_successes']: break
            seed = c['start_seed']+index
            if seed in intents: continue
            stop_gate(run); check_space(run,config)
            intent = dict(dataset_id=run.name, episode_id=seed, seed=seed,
                          expert_sha256=source_sha, config_sha256=sha(run/'config.yaml'))
            atomic(run/'collection/intents'/f'ep{seed}.json', intent, immutable=True)
            tick=perf_counter(); phases=[]
            partial = run/'collection/partial'/f'ep{seed}.h5.partial'
            final = run/'collection/episodes'/f'ep{seed}.h5'
            partial.parent.mkdir(parents=True,exist_ok=True); final.parent.mkdir(parents=True,exist_ok=True)
            with h5py.File(partial,'x') as f:
                f.attrs['completion']='PARTIAL'; f.attrs['dataset_id']=run.name
                f.attrs['expert_source_sha256']=source_sha
                obs,_,audit = reset_for_recording(env,seed,schema)
                expert=ScriptedExpert(env)
                gt=get_gt_state(env)
                # Reset identity only: GT never enters stored robot/graph inputs.
                group=digest(np.concatenate([array(getattr(gt,k)).ravel() for k in
                    ('qpos','peg_pose','box_pose','hole_pose','peg_half_sizes')]))
                f.attrs['reset_group']=group
                def record(values):
                    for k,v in values.items(): append_h5(f,k,v)
                def observation():
                    row,exact=capture(env,obs,schema)
                    if not exact: raise ValueError('camera observation mismatch')
                    row.update(episode_id=np.int64(seed),reset_seed=np.int64(seed)); record(row)
                observation()
                for t in range(c['horizon']):
                    phase=expert.phase; action=expert.act()
                    if action.shape!=(7,) or action.dtype!=np.float32 or not np.isfinite(action).all():
                        raise ValueError('expert native Action7 contract')
                    tensor=to_action_tensor(action.copy(),env); audit.clear()
                    obs,_,terminated,truncated,info=env.step(tensor)
                    success=bool(info['success'].item())
                    np.testing.assert_array_equal(action,audit['controller_input'])
                    record(dict(actions=action, env_step_input=array(tensor)[0],
                        success=np.bool_(success),terminated=np.bool_(terminated.item()),
                        truncated=np.bool_(truncated.item()),**audit))
                    phases.append(dict(timestep=t,before=phase,after=expert.phase))
                    observation(); f.flush()
                    if success or terminated.item() or truncated.item(): break
            a=load_episode(partial)
            validation=validate_episode(a,schema,seed)
            # Read the canonical file again, not an NPZ copy.
            b=load_episode(partial)
            if any(a[k].tobytes()!=b[k].tobytes() for k in a): raise ValueError('HDF5 readback mismatch')
            with h5py.File(partial,'r+') as f:
                f.attrs['completion']='COMPLETE'; f.flush()
            finalize(partial,final)
            ep=dict(**intent,reset_group=group,path=str(final.relative_to(run)),sha256=sha(final),
                    success=bool(a['success'][-1]),actions=len(a['actions']),observations=len(a['robot_state']),
                    terminated=bool(a['terminated'][-1]),truncated=bool(a['truncated'][-1]),
                    validation=validation,phases=phases,elapsed_seconds=perf_counter()-tick,
                    bytes=final.stat().st_size,first_success_step=len(a['actions']) if a['success'][-1] else None)
            atomic(run/'collection/receipts'/f'ep{seed}.json',ep,immutable=True)
            append(run/'collection/audit.jsonl',dict(seed=seed,success=ep['success'],actions=ep['actions']))
            completed.append(ep)
            if ep['success']: successes.append(ep)
            atomic(run/'collection/progress.json',dict(successes=len(successes),completed=len(completed),last_seed=seed))
            if len(completed)==c['gate_episodes']:
                for entry in completed: validate_receipt(run,entry,schema)
                atomic(run/'collection/readback_gate.json',dict(status='PASS',episodes=len(completed)),immutable=True)
                events.emit('collection_readback_gate',episodes=len(completed),status='PASS')
            if len(completed)%10==0:
                events.emit('collection_progress',successes=len(successes),attempts=len(completed),seed=seed)
        if len(successes)!=c['target_successes']:
            raise RuntimeError('target successful episodes not reached; no split reduction permitted')
        failed=[e for e in completed if not e['success']]
        interrupted=list((run/'collection/interrupted').glob('*.json'))
        manifest=dict(dataset_id=run.name,source_kind='direct_scripted_expert_hdf5_v2',
            expert_sha256=source_sha,config_sha256=sha(run/'config.yaml'),robot_schema_sha256=sha(run/'robot_state_schema.json'),
            episodes=successes,failed_episodes=failed,interrupted_attempts=[read(p) for p in interrupted],
            successes=len(successes),completed_attempts=len(completed),attempts=len(completed)+len(interrupted),
            success_rate_completed=len(successes)/len(completed),
            success_rate_all_attempts=len(successes)/(len(completed)+len(interrupted)),
            selection='success only for training; failed and interrupted attempts retained separately',
            status='COMPLETE',created=now())
        atomic(run/'dataset_manifest.json',manifest,immutable=True)
        events.emit('collection_complete',successes=len(successes),attempts=manifest['attempts'],
                    success_rate=manifest['success_rate_all_attempts'])
        return manifest
    finally: env.close()


def freeze_split(run,config):
    run=Path(run); manifest=read(run/'dataset_manifest.json')
    episodes=manifest['episodes']; sizes=config['splits']
    if sum(sizes.values())!=len(episodes): raise ValueError('no automatic split resizing')
    ids=[e['episode_id'] for e in episodes]
    groups=[e['reset_group'] for e in episodes]
    if len(set(ids))!=len(ids) or len(set(groups))!=len(groups):
        raise ValueError('duplicate episode or reset group; split requires explicit resolution')
    order=np.random.default_rng(config['seed']).permutation(len(ids))
    result=dict(dataset_id=run.name,dataset_sha256=sha(run/'dataset_manifest.json'),seed=config['seed'],splits={},reset_groups={})
    start=0
    for name,n in sizes.items():
        selected=order[start:start+n];start+=n
        result['splits'][name]=[ids[i] for i in selected]
        result['reset_groups'][name]=[groups[i] for i in selected]
    atomic(run/'split_manifest.json',result,immutable=True)
    return result


def process_vision(run,config,events,*,frame_limit=None,episode_limit=None):
    """Existing causal streaming algorithm; interrupted episode output is preserved."""
    from graph.integration_baseline.vision import causal_masks,packet_from_frame
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from visual.runtime.rgbd import TrackedRGBDFrame
    run=Path(run); manifest=read(run/'dataset_manifest.json'); camera=config['vision']['camera']
    summary_path=run/'vision_manifest.json'
    if summary_path.exists() and frame_limit is None and episode_limit is None: return read(summary_path)
    detector=GDino(device=config['vision_device']);tracker=SAM2Tracker(device=config['vision_device'])
    results=[]
    for ep in manifest['episodes'][:episode_limit]:
        stop_gate(run); check_space(run,config)
        eid=ep['episode_id']; receipt=run/'vision/receipts'/f'ep{eid}.json'
        if receipt.exists():
            entry=read(receipt)
            for row in entry['packets']:
                if sha(run/row['path'])!=row['sha256']: raise ValueError('vision packet corruption')
            if sha(run/entry['mask_path'])!=entry['mask_sha256']: raise ValueError('mask corruption')
            results.append(entry);continue
        a=validate_receipt(run,ep,read(run/'robot_state_schema.json'))
        root=run/'vision'/f'ep{eid}'
        # Never replace partial graphs; a fresh attempt gets a distinct directory.
        from uuid import uuid4
        folder=root/uuid4().hex;folder.mkdir(parents=True)
        rows=[]; rgb=a[camera+'/rgb'][:frame_limit];tick=perf_counter()
        def process(t,masks,timing):
            stop_gate(run);check_space(run,config)
            frame=TrackedRGBDFrame(t,camera,rgb[t],a[camera+'/depth'][t],a[camera+'/intrinsic'][t],
                a[camera+'/cam2world'][t],a['tcp_pose'][t],{0:masks[0],1:masks[1]},t,float(a['timestamp'][t]))
            packet,cost=packet_from_frame(frame,run.name+f':ep{eid}',keep_ratio=.5)
            packet['observation_provenance']=dict(dataset_id=run.name,episode_id=eid,timestep=t,
                episode_sha256=ep['sha256'],rgb_sha256=digest(rgb[t]),
                mask_source='prefix_causal_rgb_gdino_forward_sam2')
            validate_packet(packet)
            path=folder/f'f{t:04d}.npz';tmp=folder/f'f{t:04d}.partial.npz'
            save_packet(tmp,packet);finalize(tmp,path)
            restored=read_packet(path);validate_packet(restored)
            counts=[int((packet['object_id']==i).sum()) for i in (0,1)]
            overlap=masks[0]&masks[1];depth_valid=np.isfinite(frame.depth)&(frame.depth>0)
            missing=[]
            for oid in (0,1):
                mask=masks[oid]; support=mask&~overlap&depth_valid
                reason=('present' if counts[oid] else 'not_observed' if not mask.any() else
                        'overlap_removed' if not (mask&~overlap).any() else
                        'depth_invalid' if not support.any() else 'empty_after_existing_selection')
                missing.append(dict(reason=reason,mask_pixels=int(mask.sum()),overlap_pixels=int((mask&overlap).sum()),
                    invalid_depth_pixels=int((mask&~depth_valid).sum()),observed_support=int(support.sum())))
            rows.append(dict(timestep=t,path=str(path.relative_to(run)),sha256=sha(path),
                object_counts=counts,point_count=len(packet['points']),missing=missing,
                timings_ms=dict(**cost,mask_ms=timing['elapsed_ms']),has_action_target=t<ep['actions']))
            if t%50==0: events.emit('vision_progress',episode=eid,frame=t,total=len(rgb),objects=counts)
        masks,provenance=causal_masks(rgb,detector,tracker,process)
        mask_path=folder/'masks.npz';tmp=folder/'masks.partial.npz'
        np.savez_compressed(tmp,masks=masks,timestep=np.arange(len(rgb)),episode_id=np.int64(eid))
        with np.load(tmp,allow_pickle=False) as f: np.testing.assert_array_equal(f['masks'],masks)
        finalize(tmp,mask_path)
        entry=dict(episode_id=eid,episode_sha256=ep['sha256'],packets=rows,mask_provenance=provenance,
                   mask_path=str(mask_path.relative_to(run)),mask_sha256=sha(mask_path),elapsed_seconds=perf_counter()-tick)
        if len(rows)!=len(rgb): raise ValueError('vision processing did not produce all observations')
        # A limited fixture must never advertise full-episode coverage.
        if len(rgb)!=ep['observations']: entry['scope']='PARTIAL_SMOKE_ONLY'
        else: atomic(receipt,entry,immutable=True)
        results.append(entry)
        events.emit('vision_episode_complete',episode=eid,frames=len(rows),elapsed_seconds=entry['elapsed_seconds'])
    summary=dict(dataset_id=run.name,dataset_sha256=sha(run/'dataset_manifest.json'),
        camera=camera,policy=dict(keep_ratio=.5,raw_ede=True,variable_N=True,point_budget=None),episodes=results)
    if frame_limit is None and episode_limit is None:
        atomic(summary_path,summary,immutable=True)
    else: atomic(run/'vision_smoke_manifest.json',summary,immutable=True)
    return summary


class DirectDataset:
    """Small LRU of robot arrays; RGB-D is only read during integrity validation."""
    def __init__(self,run,*,smoke=False):
        self.run=Path(run);self.manifest=read(self.run/'dataset_manifest.json')
        self.schema=read(self.run/'robot_state_schema.json')
        split=read(self.run/'split_manifest.json'); self.splits=split['splits']
        self.graph_manifest=read(self.run/('vision_smoke_manifest.json' if smoke else 'vision_manifest.json'))
        if (split['dataset_id']!=self.run.name or self.manifest['dataset_id']!=self.run.name or
            split['dataset_sha256']!=sha(self.run/'dataset_manifest.json') or
            self.graph_manifest['dataset_sha256']!=split['dataset_sha256'] or
            self.graph_manifest['dataset_id']!=self.run.name): raise ValueError('dataset identity/hash mismatch')
        if self.manifest.get('config_sha256',sha(self.run/'config.yaml'))!=sha(self.run/'config.yaml'):
            raise ValueError('frozen dataset config changed')
        sets=[set(v) for v in self.splits.values()]; groups=[set(v) for v in split['reset_groups'].values()]
        if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]) or any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):
            raise ValueError('episode/reset group leakage')
        self.episodes={e['episode_id']:e for e in self.manifest['episodes']}
        if set.union(*sets)!=set(self.episodes) or len(self.episodes)!=sum(map(len,sets)):
            raise ValueError('split coverage mismatch')
        expected_groups={e['episode_id']:e['reset_group'] for e in self.manifest['episodes']}
        for name,ids in self.splits.items():
            if split['reset_groups'][name]!=[expected_groups[e] for e in ids]:raise ValueError('reset group mismatch')
        if self.manifest['robot_schema_sha256']!=sha(self.run/'robot_state_schema.json'):raise ValueError('schema checksum mismatch')
        self.packet_rows={};self.cache=OrderedDict()
        for e in self.graph_manifest['episodes']:
            if e['episode_sha256']!=self.episodes[e['episode_id']]['sha256']:raise ValueError('vision/raw source mismatch')
            p=e['mask_provenance']
            if p['future_observations_used'] or p['gt_segmentation_used'] or not p['sam2_horizon'].startswith('num_frames=t+1;'):
                raise ValueError('causal mask provenance contract')
            if sha(self.run/e['mask_path'])!=e['mask_sha256']:raise ValueError('mask file checksum mismatch')
            for row in e['packets']:
                key=(e['episode_id'],row['timestep'])
                if key in self.packet_rows:raise ValueError('duplicate packet identity')
                self.packet_rows[key]=row

    def raw(self,eid):
        if eid not in self.cache:
            e=self.episodes[eid];path=self.run/e['path']
            if sha(path)!=e['sha256']:raise ValueError('episode corrupted')
            with h5py.File(path,'r') as f:
                if f.attrs['completion']!='COMPLETE' or f.attrs['dataset_id']!=self.run.name:raise ValueError('partial/foreign episode')
                keys=['actions','robot_state','robot_world_geometry','tcp_pose','timestamp','terminated','truncated']
                self.cache[eid]={k:f[k][()] for k in keys}
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(eid);return self.cache[eid]

    def frame(self,eid,t):
        a=self.raw(eid);row=self.packet_rows[(eid,t)]
        if not 0<=t<len(a['actions']) or not row['has_action_target']:raise ValueError('unlabelled final observation')
        path=self.run/row['path']
        if sha(path)!=row['sha256']:raise ValueError('packet checksum mismatch')
        p=read_packet(path);validate_packet(p);pr=p['observation_provenance']
        if (p['episode']!=self.run.name+f':ep{eid}' or p['source_frame']!=t or p['frame_idx']!=t or
            p['timestamp']!=a['timestamp'][t] or pr['dataset_id']!=self.run.name or pr['episode_id']!=eid or
            pr['timestep']!=t or pr['episode_sha256']!=self.episodes[eid]['sha256'] or p['edge_policy']!=RAW_EDE_POLICY):
            raise ValueError('graph/robot/action join mismatch')
        np.testing.assert_array_equal(p['observation_ee_pose'],a['tcp_pose'][t].astype(np.float64))
        return AlignedFrame(eid,t,p,a['robot_state'][t].copy(),geometry_in_ee(a['robot_world_geometry'][t],a['tcp_pose'][t]),
                            a['actions'][t].copy(),bool(a['terminated'][t]),bool(a['truncated'][t]))

    def sequence(self,eid,start,length):
        seq=[self.frame(eid,t) for t in range(start,start+length)]
        if any(r.terminated or r.truncated for r in seq[:-1]):raise ValueError('physical episode boundary crossing')
        return seq

    def fit_normalizer(self):
        ids=self.splits['train']
        return RobotNormalizer.fit(np.concatenate([self.raw(e)['robot_state'][:-1] for e in ids]),ids,self.splits,self.schema)

    def validate(self,*,partial=False):
        coverage=[]
        for eid,e in self.episodes.items():
            a=validate_receipt(self.run,e,self.schema)
            present=[];phases={}
            timesteps=sorted(t for i,t in self.packet_rows if i==eid) if partial else range(e['observations'])
            if partial and list(timesteps)!=list(range(len(timesteps))):raise ValueError('smoke observations must be a contiguous prefix')
            for t in timesteps:
                row=self.packet_rows[(eid,t)];p=read_packet(self.run/row['path'])
                if sha(self.run/row['path'])!=row['sha256']:raise ValueError('packet corruption')
                validate_packet(p)
                pr=p['observation_provenance'];camera=self.graph_manifest['camera']
                if pr['rgb_sha256']!=digest(a[camera+'/rgb'][t]):raise ValueError('RGB identity mismatch')
                np.testing.assert_array_equal(p['camera_to_world'],a[camera+'/cam2world'][t].astype(np.float64))
                np.testing.assert_array_equal(p['camera_intrinsic'],a[camera+'/intrinsic'][t].astype(np.float64))
                np.testing.assert_array_equal(p['image_shape'],a[camera+'/rgb'][t].shape[:2])
                np.testing.assert_array_equal(p['observation_ee_pose'],a['tcp_pose'][t].astype(np.float64))
                if t<e['actions']:
                    frame=self.frame(eid,t)
                    np.testing.assert_array_equal(frame.action,a['actions'][t])
                counts=row['object_counts'];present.append([int(v>0) for v in counts])
                phase=e['phases'][min(t,e['actions']-1)]['before']
                phase='grasp' if phase in ('reach','descend','close','lift') else 'alignment' if phase=='align' else 'insertion'
                phases.setdefault(phase,[]).append(present[-1])
            pp=np.asarray(present)
            coverage.append(dict(episode_id=eid,observations=len(pp),presence_counts=pp.sum(0).tolist(),
                box_never_observed=bool(not pp[:,1].any()),presence_recoveries=((pp[1:]>pp[:-1])&(pp[1:]>0)).sum(0).tolist(),
                phase_counts={k:dict(frames=len(v),present=np.asarray(v).sum(0).tolist()) for k,v in phases.items()},
                point_counts=[self.packet_rows[(eid,t)]['object_counts'] for t in timesteps]))
        report=dict(status='PASS',episodes=len(self.episodes),actions=sum(e['actions'] for e in self.episodes.values()),
            observations=sum(e['observations'] for e in self.episodes.values()),alignment='PASS',
            splits=self.splits,gt_student_inputs=False,coverage=coverage,
            box_never_observed_fraction=sum(e['box_never_observed'] for e in coverage)/len(coverage))
        report['scope']='PARTIAL_REAL_SMOKE_PREFIX' if partial else 'FULL_DATASET'
        report['validated_graph_observations']=sum(e['observations'] for e in coverage)
        atomic(self.run/('smoke_dataset_validation.json' if partial else 'dataset_validation.json'),report,immutable=True);return report

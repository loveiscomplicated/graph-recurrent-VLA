"""Index-only segments with exact source arrays and prefix restoration references."""
import json
from pathlib import Path
import h5py
import numpy as np
from graph.prototype.common import write_json
from graph.prototype.data import state_at, sha256

STATE_KEYS=('ee_pose','peg_pose','hole_pose','box_pose','peg_half_sizes','hole_radius','gripper_qpos','qpos','qvel','success','grasp','contact_valid','finger_forces','peg_box_force')


def load_audit(path):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    n=len(data['actions'])
    if any(len(data[k])!=n+1 for k in STATE_KEYS):
        raise ValueError('audit needs T+1 states and T actions')
    data['record']=json.loads(str(data['replay_record']))
    return data


def state_row(data,t):
    if not 0<=t<=len(data['actions']): raise IndexError(t)
    return dict(**{k:data[k][t] for k in STATE_KEYS},reach_local=data['reach_local'],
        grasp_local=data['grasp_local'],initial_peg_pose=data['peg_pose'][0])


def save_segments(path, payload): write_json(path,payload)


def load_segments(path,verify_sources=True):
    data=json.loads(Path(path).read_text())
    if verify_sources:
        validate_splits(data)
        for key in ('source','initial_state_source'):
            if sha256(data[key])!=data[key+'_sha256']: raise ValueError(f'{key} changed')
        for ep in data['episodes']:
            if sha256(ep['audit_file'])!=ep['audit_sha256']: raise ValueError('audit changed')
    return data


def validate_splits(data):
    """Episode IDs and identical seed/reset groups cannot cross splits."""
    episode_ids=[e['episode_id'] for e in data['episodes']]
    split_ids=[eid for ids in data['splits'].values() for eid in ids]
    if len(episode_ids)!=len(set(episode_ids)) or len(split_ids)!=len(set(split_ids)) or set(episode_ids)!=set(split_ids):
        raise ValueError('duplicate/missing episode in splits')
    reset_splits={}
    for ep in data['episodes']:
        split=next(k for k,v in data['splits'].items() if ep['episode_id'] in v)
        if ep['split']!=split:
            raise ValueError('episode split label differs')
        for segment in ep['segments']:
            ref=segment['initial_state_ref']
            kwargs=dict(ref['reset_kwargs'])
            kwargs['seed']=ref['episode_seed']
            kwargs['options']=kwargs.get('options') or {}
            reset_key=json.dumps(kwargs,sort_keys=True)
            if reset_key in reset_splits and reset_splits[reset_key]!=split:
                raise ValueError('same reset seed/config leaks across splits')
            reset_splits[reset_key]=split
            if segment['source_episode_id']!=ep['episode_id'] or segment['split']!=split:
                raise ValueError('segment episode/split differs')


def segment_arrays(dataset,segment):
    if segment['status']!='detected': raise ValueError('unresolved segment has no usable slice')
    i,j=segment['start_t'],segment['end_t'];eid=segment['source_episode_id']
    episode=next(e for e in dataset['episodes'] if e['episode_id']==eid)
    audit=load_audit(episode['audit_file'])
    if not 0<=i<j<=len(audit['actions']): raise ValueError('invalid segment boundary')
    with h5py.File(dataset['source']) as h5, h5py.File(dataset['initial_state_source']) as orig:
        group=h5[f'traj_{eid}']
        actions=np.asarray(group['actions'][i:j])
        # Legacy conversion has a corrupt s0; native s0 is separately verified.
        # New recordings have matching native and converted initial states.
        states=[state_at(orig[f"traj_{segment['initial_state_ref']['original_episode_id']}/env_states"],0)
            if t==0 else state_at(group['env_states'],t) for t in range(i,j+1)]
        obs=np.asarray(group['obs'][i:j+1])
    if not np.array_equal(actions,audit['actions'][i:j]): raise ValueError('source actions differ')
    geometry={k:audit[k][i:j+1] for k in STATE_KEYS}
    assert len(states)==len(actions)+1==len(obs)
    # Compare the actual simulator state at every source index to replay GT.
    for row, saved in enumerate(states):
        np.testing.assert_allclose(saved['actors']['peg'][:7], geometry['peg_pose'][row], atol=2e-6, rtol=0)
        np.testing.assert_allclose(saved['actors']['box_with_hole'][:7], geometry['box_pose'][row], atol=2e-6, rtol=0)
        np.testing.assert_allclose(saved['articulations']['panda_wristcam'][13:22], geometry['qpos'][row], atol=2e-6, rtol=0)
    return dict(states=states,actions=actions,observations=obs,geometry=geometry)

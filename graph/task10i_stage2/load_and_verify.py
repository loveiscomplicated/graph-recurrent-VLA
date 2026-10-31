"""Read authoritative manifests/GT, preserving every input byte."""
from collections import Counter
import hashlib
from pathlib import Path

import h5py
import numpy as np

from graph.initial_pool10i.mirror.verify import read, sha, inventory, compare, consumer, write
from .core import descriptor, canonical_hash, freeze

POOL=Path('/workspace/task10i_stage1_20261031/verified_pool_v2')
HANDOFF=Path('/workspace/task10i_stage1_5_mirror_7934732624c148ad86a8f4d518bf7c18/handoff')
AUDIT=Path('/workspace/task10i_stage1_20261031/audit')
F1=Path('/workspace/task10f1_gt_pretrain_20261030')
G=Path('/workspace/task10g_f1_abc_20261031/full')


def verify_inputs(out):
    receipt=read(HANDOFF/'stage2_input_receipt.json')
    assert receipt['STAGE2_READY'] and receipt['instance_id']==57987772
    assert receipt['verified_remote_root']==str(POOL)
    idx=read(HANDOFF/'artifact_index.json')
    for rel,info in idx['files'].items():
        p=HANDOFF/rel; assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],rel
    expected=read(HANDOFF/'source_inventory.json')['files']
    actual=inventory(POOL);audit=compare(expected,actual)
    assert audit['counts']==dict(MATCHED=8514,MISSING=0,MISMATCHED=0,EXTRA=0,UNREADABLE=0)
    assert sha(POOL/'candidate_pool_manifest.json')==receipt['candidate_pool_manifest_sha256']
    proof=consumer(POOL,receipt['canonical_source_root'])
    assert proof['manifest']==read(HANDOFF/'remote_manifest_verification.json')
    assert proof['candidates']==read(HANDOFF/'remote_candidate_readback.json')
    assert proof['rgbd']==read(HANDOFF/'remote_raw_rgbd_readback.json')
    freeze(out/'stage2_input_receipt_verified.json',dict(status='PASS',receipt=receipt,
        receipt_sha256=sha(HANDOFF/'stage2_input_receipt.json'),counts=audit['counts'],
        candidate_readback=300, raw_rgbd_readback=300,handoff_index_sha256=sha(HANDOFF/'artifact_index.json')))
    freeze(out/'input_pool_inventory.json',actual)
    freeze(out/'input_candidate_readback.json',proof)
    return receipt


def load_candidates(out):
    entries=read(POOL/'eligible_candidate_pool.json'); rows=[];states={}
    for e in sorted(entries,key=lambda e:e['candidate_id']):
        cid=e['candidate_id']
        with np.load(POOL/e['initial_npz'],allow_pickle=False) as z:
            state={k:z[k] for k in z.files}
        d=descriptor(state); old=e['descriptor']
        for new,prior in [('peg_xyz','peg_xyz'),('g1','peg_minus_tcp_world'),('g3p','peg_minus_hole_world'),
                          ('g4','peg_half'),('g5','image_centroid_uv'),('boundary_margin_px','boundary_margin_px')]:
            v=np.array(old[prior]);v=v[:2]*2 if new=='g4' else v/128 if new=='g5' else v
            np.testing.assert_allclose(d[new],v,rtol=0,atol=1e-7)
        rows.append(dict(id=cid,seed=e['reset_seed'],role='UNASSIGNED',kind='new',descriptor=d,
            source=dict(pool=str(POOL),export=e,initial_sha256=sha(POOL/e['initial_npz']),
                snapshot_sha256=sha(POOL/e['snapshot']),visibility_sha256=sha(POOL/e['gt_visibility_evidence'])),
            expert_status='NOT_TESTED'))
        states[cid]=state
    assert len(rows)==300 and len({r['seed'] for r in rows})==300
    freeze(out/'candidate_descriptor_readback.json',dict(status='PASS',count=len(rows),
        ids=[r['id'] for r in rows],descriptor_contract='recomputed from actual NPZ, original stage1 cross-check'))
    return rows,states


def load_existing(out):
    inp=AUDIT/'task10g_initial_visibility_inputs_v1'; proof=read(inp/'extraction_verification.json')
    for name,expected in proof['files'].items(): assert sha(inp/name)==expected,name
    episodes=read(inp/'episodes.json'); verdicts=read(AUDIT/'results_v1/episodes.json')
    eligibility=read(AUDIT/'results_v1/eligibility_ids.json')
    assert len(episodes)==len(verdicts)==320
    manifest=read(F1/'final_dataset_manifest.json'); cfg=read(G/'config.json')
    assert sha(F1/'final_dataset_manifest.json')==cfg['dataset_manifest_sha256']==proof['source_manifest_sha256']
    assert sha(G/'dataset_identity.json')==proof['task10g_dataset_identity_sha256']
    assert sha(G/'host_data/metadata.json')==proof['host_metadata_sha256']
    specs={e['episode_id']:e for e in manifest['episode_specs']}
    release={e['episode_id']:e for e in manifest['episodes']}
    meta=read(G/'host_data/metadata.json')['episodes']
    robot_schema=read(F1/'normalizer.json')['schema']
    assert robot_schema['dimension']==76 and robot_schema['pose_order']=='xyz,qw,qx,qy,qz'
    robot_fields={r['name']:slice(r['offset'],r['offset']+r['dimension']) for r in robot_schema['fields']}
    assert {str(e['episode_id']) for e in episodes}==set(meta)
    with np.load(inp/'first_observations.npz',allow_pickle=False) as z: saved={k:z[k] for k in z.files}
    parents={};inventory_rows=[];states={};source_files={}
    for i,(e,v) in enumerate(zip(episodes,verdicts)):
        eid=e['episode_id'];seed=e['parent_seed'];spec=specs[eid];rel=release[eid]
        assert v['episode_id']==eid and v['index']==e['index']==i
        assert spec['seed']==seed==meta[str(eid)]['parent']
        assert spec['split']==e['split']==meta[str(eid)]['split']
        passed=eid in eligibility[e['split']]['primary_geometry_depth_pass']
        assert v['requested_pass']==passed
        raw=Path(e['raw_path']);gt=Path(e['gt_path'])
        assert sha(raw)==e['raw_sha256']==rel['raw_episode_sha256']
        assert sha(gt)==e['gt_sha256']==rel['gt_sidecar_sha256']
        source_files[str(raw)]=dict(bytes=raw.stat().st_size,sha256=e['raw_sha256'])
        source_files[str(gt)]=dict(bytes=gt.stat().st_size,sha256=e['gt_sha256'])
        with h5py.File(raw,'r') as f,np.load(gt,allow_pickle=False) as g:
            assert int(f['timestep'][0])==int(g['timestep'][0])==0
            assert float(f['timestamp'][0])==float(g['timestamp'][0])==0
            assert len(f['actions'])==e['actions']==meta[str(eid)]['actions']
            s={k:g['raw_'+k][0].copy() for k in ['tcp_pose','peg_pose','peg_half','hole_pose','hole_radius']}
            for k in ['rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl']:
                s[k]=f['base_camera/'+k][0];np.testing.assert_array_equal(s[k],saved[k][i])
            for k in ['tcp_pose','peg_pose','peg_half']:
                np.testing.assert_array_equal(s[k],saved[k][i])
            np.testing.assert_array_equal(s['tcp_pose'],f['tcp_pose'][0])
            robot=f['robot_state'][0]
            s['qpos']=robot[robot_fields['qpos']];s['qvel']=robot[robot_fields['qvel']]
            np.testing.assert_array_equal(robot[robot_fields['tcp_pose']],s['tcp_pose'])
            s['gripper']=g['raw_gripper'][0]
            np.testing.assert_array_equal(robot[robot_fields['gripper_position']],s['gripper'])
            for k,a in s.items():assert np.isfinite(a).all(),(eid,k)
        h=hashlib.sha256()
        for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','peg_pose','peg_half','tcp_pose'):
            h.update(k.encode()+s[k].tobytes())
        assert h.hexdigest()==e['first_observation_sha256']
        pid='existing_'+str(seed)
        if pid in parents:
            assert parents[pid]['split']==e['split'] and parents[pid]['passed']==passed
            for k in s: np.testing.assert_array_equal(s[k],states[pid][k],err_msg=f'parent copied prefix {pid} {k}')
        else:
            parents[pid]=dict(split=e['split'],passed=passed,episodes=[],seed=seed,representative=eid,verdict=v)
            states[pid]=s
        parents[pid]['episodes'].append(e)
        inventory_rows.append(dict(**e,parent_id=pid,visibility_pass=passed,collection_spec=spec,
                                   disposition='PRESERVE' if passed else 'REMOVE_IN_STAGE3_PLAN_ONLY'))
        if (i+1)%40==0: print('existing source/GT readback',i+1,flush=True)
    rows=[]
    for pid,p in sorted(parents.items()):
        if not p['passed']:continue
        d=descriptor(states[pid]);v=p['verdict']
        np.testing.assert_allclose(d['corners_uv'],v['corners_uv'],atol=1e-6,rtol=0)
        rows.append(dict(id=pid,seed=p['seed'],kind='existing',role='existing_'+p['split'],
            descriptor=d,episodes=p['episodes'],source=dict(original_episode_id=p['representative'],
                first_observations=str(inp/'first_observations.npz'),
                visibility_evidence=str(AUDIT/'results_v1/pixel_evidence.npz'),
                approved_eligibility_sha256=sha(AUDIT/'results_v1/eligibility_ids.json'),
                visibility_verdict=p['verdict']),expert_status='HISTORICAL_RECORDED_NOT_REEXECUTED'))
    assert Counter(r['role'] for r in rows)==dict(existing_train=33,existing_validation=7)
    assert Counter(e['split'] for e in inventory_rows if e['visibility_pass'])==dict(train=82,validation=23)
    candseeds={e['reset_seed'] for e in read(POOL/'eligible_candidate_pool.json')}
    assert not candseeds & {e['seed'] for e in manifest['episode_specs']}
    freeze(out/'existing_valid_parent_inventory.json',dict(status='PASS',parents=rows,
        counts=dict(train=33,validation=7),logical_episodes=dict(train=82,validation=23),
        original_inventory_counts=dict(train_parents=96,validation_parents=16,train_episodes=272,validation_episodes=48)))
    freeze(out/'existing_episode_preservation_manifest.json',dict(episodes=inventory_rows,
        counts=dict(Counter(e['disposition'] for e in inventory_rows)),source_manifest_sha256=sha(F1/'final_dataset_manifest.json'),
        source_manifest=str(F1/'final_dataset_manifest.json'), grouping='one parent per reset seed; all copied-prefix branches same split',
        final_test_used=False))
    freeze(out/'existing_source_file_inventory.json',source_files)
    return rows,states


def preserve(out):
    assert inventory(POOL)==read(out/'input_pool_inventory.json')
    for name,info in read(out/'existing_source_file_inventory.json').items():
        assert Path(name).stat().st_size==info['bytes'] and sha(name)==info['sha256'],name
    freeze(out/'source_preservation_audit.json',dict(status='PASS',pool_files=8514,
        existing_files=len(read(out/'existing_source_file_inventory.json')),hash_before_after_identical=True))

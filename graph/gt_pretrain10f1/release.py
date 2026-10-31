"""Normalize a verified Task10F export and freeze actual data/budget before fitting.

Runtime mismatches are reported and do not release training. Existing source,
plans, raw data, cache and initial configuration files are never overwritten.
"""
import argparse
from pathlib import Path
from collections import Counter

import h5py
import numpy as np

from graph.plan_a_bc.common import read,write,sha,npz,now
from recurrent.features import build_features
from .contract import measured_robot,object_target
from .ready_contract import validate,safe_file


def runtime_mismatches(expected,actual):
    """Compare executed semantics, retaining distinct collector provenance.

    Native rebuilding is an explicit conversion, not a waiver for a failed
    cross-runtime test. Historical camera backend annotations are not live
    backend evidence; the simulator fields pin the actual backend separately.
    """
    failures=[]
    required=('tensor_runtime','robot_schema','geometry','geometry_protocol_sha256','source_sha256','gt38_encoder_sha256','quaternion_order','simulator','camera_contract','vision_weight_sha256')
    if any(k not in expected for k in required):return ['expected_runtime_contract_schema']
    for k in ('tensor_runtime','robot_schema','geometry','geometry_protocol_sha256','source_sha256','gt38_encoder_sha256','quaternion_order'):
        if actual.get(k)!=expected[k]:failures.append(k)
    if any(actual.get('simulator',{}).get(k)!=v for k,v in expected['simulator'].items()):failures.append('simulator')
    camera_keys=('roles','installed','camera_shapes','same_physics_timestep','depth_unit','coordinate_convention','schema')
    if any(actual.get('camera_contract',{}).get(k)!=expected['camera_contract'].get(k) for k in camera_keys):failures.append('camera_contract')
    def weights(value):
        return dict(dino_files={k:v for k,v in value.get('dino_files',{}).items() if k!='download_provenance.json'},sam_sha256=value.get('sam_sha256'))
    if weights(actual.get('vision_weight_sha256',{}))!=weights(expected['vision_weight_sha256']):failures.append('vision_weight_sha256')
    collector=actual.get('collector_execution')
    if collector and any(collector['versions'][k]!=expected['tensor_runtime'][k] for k in ('python','torch','numpy')):
        proof=actual.get('native_runtime_acceptance',{})
        if (proof.get('status')!='PASS' or proof.get('mode')!='ALL_GRAPHS_REBUILT_IN_F1_NATIVE_RUNTIME'
            or proof.get('old576_graphs_included') is not False):failures.append('collector_runtime_requires_complete_native_conversion')
    return failures


def expected_runtime(root):
    env=read(root/'environment_worker0.json');versions=env['versions']
    source=Path(__file__).resolve().parents[2]
    reach=Path(read(root/'run_identity.json')['reach'])
    vision=read(reach/'optimization_equivalence.json')['model_identity']
    vision=dict(dino_files={k:v for k,v in vision['dino_files'].items() if k!='download_provenance.json'},sam_sha256=vision['sam_sha256'])
    deps=['graph/plan_a_geometry/core.py','graph/plan_a_geometry/protocol.json','graph/plan_a_bc/geometry.py',
        'graph/persistent_geometry/graph.py','graph/persistent_geometry/self_filter.py',
        'graph/integration_baseline/online_vision_adapter.py','visual/runtime/tracking.py','visual/runtime/grounding.py','recurrent/features.py']
    return dict(simulator=dict(mani_skill=versions['mani_skill'],sapien=versions['sapien'],gymnasium=versions['gymnasium'],
            sim_backend='physx_cpu',render_backend='sapien_cuda',control_mode='pd_ee_delta_pose',control_freq=20,physics_freq=100),
        tensor_runtime={k:versions[k] for k in ('python','torch','numpy')},
        camera_contract=read(root/'inputs/sensor_contract.json'),robot_schema=read(root/'normalizer.json')['schema'],
        geometry=read(root/'config.json')['geometry'],geometry_protocol_sha256=sha(source/'graph/plan_a_geometry/protocol.json'),
        vision_weight_sha256=vision,source_sha256={k:sha(source/k) for k in deps},
        gt38_encoder_sha256=sha(source/'recurrent/features.py'),quaternion_order='wxyz')


def run(root,imported,marker_sha):
    if len(marker_sha)!=64:raise ValueError('trusted source marker SHA is required')
    if (root/'training').exists() or (root/'final_dataset_manifest.json').exists():
        raise FileExistsError('existing release/training must be preserved; inspect before reusing')
    expected=expected_runtime(root);validation=validate(imported,expected['gt38_encoder_sha256'])
    if validation['marker_sha256']!=marker_sha:raise ValueError('source vs transferred marker identity')
    actual=validation['runtime_contract'];mismatches=runtime_mismatches(expected,actual)
    if 'native_runtime_acceptance' in actual:
        proof=actual['native_runtime_acceptance']
        if proof.get('target_environment_sha256')!=sha(root/'environment_worker0.json'):
            mismatches.append('native_target_environment_identity')
    check=dict(created_at=now(),status='PASS' if not mismatches else 'WAITING_RUNTIME_COMPATIBILITY',
        expected=expected,actual=actual,mismatched_contract_fields=mismatches,source_marker_sha256=marker_sha,
        no_silent_runtime_mixing=True)
    checkpath=root/'release_checks'/f'{marker_sha}.json'
    if checkpath.exists():
        prior=read(checkpath)
        assert prior['source_marker_sha256']==marker_sha and prior['mismatched_contract_fields']==mismatches
    else:write(checkpath,check,immutable=True)
    if mismatches:
        print('WAITING_RUNTIME_COMPATIBILITY:',','.join(mismatches),'No training released.',flush=True)
        return check
    if not (root/'graph_cache_manifest.json').exists():
        print('WAITING_FULL_TASK_CACHE: preserve ongoing data job; no duplicate launch.',flush=True)
        return dict(status='WAITING_FULL_TASK_CACHE')
    marker=read(imported/'gnn_dataset_ready.json');dataset=read(safe_file(imported,marker['dataset_manifest']))
    old=read(root/'dataset_manifest.json');graphmanifest=read(root/'graph_cache_manifest.json')
    specs=list(old['episodes']);episodes=list(graphmanifest['episodes']);schema=expected['robot_schema']
    split=read(root/'split_manifest.json');forbidden=set(split['forbidden'])|set(split['final'])
    parents={p:k for k in ('train','validation','final') for p in split[k]};counts=Counter();derived=imported/'derived_labels'
    derived.mkdir(exist_ok=True);records=[]
    for j,ep in enumerate(dataset['episodes']):
        parent=int(ep['parent_id']);group=ep['split']
        if parent in forbidden or parent in parents and parents[parent]!=group:
            raise ValueError('Task10F parent overlaps held-out/other split')
        parents[parent]=group;eid=3000000+j
        rawpath=safe_file(imported,ep['raw_path']);gtpath=safe_file(imported,ep['gt_path'])
        graphpath=safe_file(imported,ep['graph_path']);maskpath=safe_file(imported,ep['gnn_mask_path'])
        with h5py.File(rawpath) as f,np.load(gtpath,allow_pickle=False) as gt,np.load(maskpath,allow_pickle=False) as masks:
            robot=measured_robot(f['robot_state'][:],schema)
            np.testing.assert_allclose(robot['tcp_pose'],f['tcp_pose'][:],atol=1e-7,rtol=0)
            np.testing.assert_allclose(robot['left_tip'],f['robot_world_geometry'][:,1,:],atol=1e-7,rtol=0)
            np.testing.assert_allclose(robot['right_tip'],f['robot_world_geometry'][:,2,:],atol=1e-7,rtol=0)
            raw={**robot,**{k:gt['raw_'+k].copy() for k in ('peg_pose','hole_pose','peg_half','hole_radius')}}
            n=int(ep['actions']);segments=f['segment'][:]
            distribution=np.array([{0:'nominal_reach',1:'disturbed_robot',2:'expert_recovery'}[int(s)] for s in segments]+['terminal'])
            arrays={**{'raw_'+k:v for k,v in raw.items()},'target18':object_target(raw),
                'gt38':build_features(raw,np.ones(n+1,'f'),np.zeros(n+1,'f')),
                'timestep':gt['timestep'],'timestamp':gt['timestamp'],'distribution':distribution}
            eligible=masks['gnn_supervision_mask'];counts.update(distribution[:n][eligible].tolist())
        derivedpath=derived/f'ep{eid}_gt.npz'
        if derivedpath.exists():
            with np.load(derivedpath,allow_pickle=False) as oldlabels:
                for k,v in arrays.items():np.testing.assert_array_equal(oldlabels[k],v)
        else:npz(derivedpath,arrays)
        specs.append(dict(episode_id=eid,seed=parent,parent_id=parent,split=group,paired=False,
            branch_id=ep['branch_id'],source='Task10F_corrective',distribution='per_timestep_sidecar',
            gnn_mask_path=str(maskpath),source_episode_id=ep['episode_id']))
        receipt=read(safe_file(imported,ep['graph_receipt_path']))
        episodes.append(dict(receipt,episode_id=eid,seed=parent,split=group,path=str(graphpath),graphs=n,actions=n,
            gt_path=str(derivedpath),gt_sidecar_sha256=sha(derivedpath),parent_seed=parent,
            source_episode_id=ep['episode_id'],branch_id=ep['branch_id'],source_raw_path=str(rawpath)))
        records.append(dict(episode_id=eid,source_episode_id=ep['episode_id'],parent=parent,split=group,
            raw_sha256=sha(rawpath),source_gt_sha256=sha(gtpath),derived_gt_sha256=sha(derivedpath),
            graphs_sha256=sha(graphpath),gnn_mask_sha256=sha(maskpath)))
    final=dict(episodes=episodes,paired_episodes=graphmanifest['paired_episodes'],episode_specs=specs,
        contains_verified_Task10F=True,source_marker_sha256=marker_sha,Task10F_validation=validation,
        source_before_GNN=True,corrective_distribution_counts=dict(counts),normalized_source_provenance=records,
        runtime_compatibility_sha256=sha(checkpath),created_at=now())
    config=read(root/'pretrain_config.json')
    config.update(final_freeze_at=now(),final_dataset_sources=['validated Full-task F1','sealed approved Task10F'],
        corrective_eligible_frame_counts=dict(counts),budget_decision='retain fixed12k updates after actual support audit; no result-based tuning',
        frame_sampling='uniform eligible frames; source common prefixes retained, same-parent correlation disclosed',
        parent_split='single split per parent across original and P/N branches',source_marker_sha256=marker_sha)
    write(root/'final_dataset_manifest.json',final,immutable=True);write(root/'final_pretrain_config.json',config,immutable=True)
    release=dict(status='READY_FOR_GNN_PRETRAINING',Task10F_GT_validation='PASS',transfer_readback='PASS',
        runtime_camera_geometry_compatibility='PASS',source_marker_sha256=marker_sha,
        dataset_manifest_sha256=sha(root/'final_dataset_manifest.json'),pretrain_config_sha256=sha(root/'final_pretrain_config.json'),
        final_frozen_before_training=True,at=now())
    write(root/'data_ready_release.json',release,immutable=True)
    print('FINAL DATA AND BUDGET FROZEN. GNN training is now permitted.',flush=True)
    return release


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--imported',type=Path,required=True);p.add_argument('--marker-sha',required=True)
    a=p.parse_args();run(a.root,a.imported,a.marker_sha)

"""Audit every original episode and freeze BC/evaluation conditions."""
import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np

from graph.integration_baseline.direct_dataset import validate_receipt
from graph.integration_baseline.bc08_codec import ActionCodec, digest
from graph.integration_baseline.robot import RobotNormalizer
from graph.integration_baseline.direct_train import normalizer_state
from .common import read, write, sha, protocol, tree_identity, now
from .geometry import RobotFK


def prepare(root, dataset, task08_plan):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    manifest = read(dataset/'dataset_manifest.json'); split = read(dataset/'split_manifest.json')
    schema = read(dataset/'robot_state_schema.json')
    if manifest['status']!='COMPLETE' or len(manifest['episodes'])!=120:
        raise ValueError('complete successful original120 required')
    if split['dataset_sha256']!=sha(dataset/'dataset_manifest.json'):
        raise ValueError('split/dataset identity mismatch')
    if schema['dimension']!=76: raise ValueError('Robot76 required')
    fk = RobotFK(root/'inputs',schema); rows=[]; train_actions={};robot=[];fitted=[]
    eps = {e['episode_id']:e for e in manifest['episodes']}
    if len(eps)!=120 or len(set(e['reset_group'] for e in eps.values()))!=120:
        raise ValueError('duplicate episode/reset group')
    original = read(task08_plan)
    if original['splits'] != split['splits']: raise ValueError('Task08 split changed')
    source = Path(__file__).resolve().parents[2]
    # Original final geometry source plus new opt-in code, independent of dirty workspace.
    sources=tree_identity(source); source_sha=digest(sources)
    for eid,ep in eps.items():
        a = validate_receipt(dataset,ep,schema)
        if not ep['success'] or not a['success'][-1] or a['success'][:-1].any():
            raise ValueError('non-success or post-success tail')
        error=0.
        for t in range(ep['observations']):
            error=max(error,fk.verify_robot(a['robot_state'][t],a['robot_world_geometry'][t],a['tcp_pose'][t]))
        membership=[k for k,v in split['splits'].items() if eid in v]
        if len(membership)!=1: raise ValueError('episode split cardinality')
        rows.append(dict(episode_id=eid,seed=ep['seed'],split=membership[0],actions=ep['actions'],
            observations=ep['observations'],path=ep['path'],sha256=sha(dataset/ep['path']),
            reset_group=ep['reset_group'],RGBD_cameras=['base_camera','hand_camera'],
            robot_fk_max_abs=error,observation_t_to_action_t='PASS',receipt='PASS'))
        if eid in split['splits']['train']:
            train_actions[eid]=a['actions']; robot.append(a['robot_state'][:-1]);fitted.append(eid)
    cfg=protocol(task08_plan,manifest,split,source_sha)
    norm=RobotNormalizer.fit(np.concatenate(robot),fitted,split['splits'],schema)
    ns=normalizer_state(norm)
    # Preserve the exact original Train96 Robot76 statistics (including reduction
    # order); freshly recomputed means can differ by a float32 rounding unit.
    import torch
    original_norm=torch.load(root/'inputs/task08_initial.pt',map_location='cpu',weights_only=False)['normalizer']
    if original_norm['schema']!=schema or original_norm['fitted_episode_ids']!=sorted(fitted):
        raise ValueError('original Train96 normalizer identity')
    torch.testing.assert_close(original_norm['mean'],ns['mean'],atol=2e-7,rtol=0)
    torch.testing.assert_close(original_norm['std'],ns['std'],atol=2e-7,rtol=0)
    ns=original_norm
    codec=ActionCodec.fit('B',train_actions,split['splits']['train'])
    # Compare statistics to the actual Task08 Full B preserved execution plan.
    old=original['codecs']['full']['B']
    for k in ['mean6','std6','fitted_episode_ids','count']:
        if codec.state[k]!=old[k]: raise ValueError('Full B action normalization differs: '+k)
    write(root/'normalizer.json',dict(schema=schema,mean=ns['mean'].tolist(),std=ns['std'].tolist(),
        fitted_episode_ids=ns['fitted_episode_ids']),immutable=True)
    write(root/'action_codec.json',codec.state,immutable=True)
    write(root/'source_manifest.json',dict(files=sources,sha256=source_sha),immutable=True)
    write(root/'config.json',cfg,immutable=True)
    write(root/'dataset_audit.json',dict(status='PASS',directory=str(dataset),
        dataset_sha256=sha(dataset/'dataset_manifest.json'),split_sha256=sha(dataset/'split_manifest.json'),
        source_plan_sha256=sha(task08_plan),episodes=rows,total_actions=sum(e['actions'] for e in rows),
        train_actions=sum(e['actions'] for e in rows if e['split']=='train'),
        validation_actions=sum(e['actions'] for e in rows if e['split']=='validation'),
        terminal_observations_unlabelled=True,action_codec_matches_actual_Task08_Full_B=True),immutable=True)
    write(root/'seeds_manifest.json',dict(frozen_at=now(),before_any_policy_output=True,
        train=[30003],development=cfg['evaluation']['dev_seeds'],smoke=[44001],
        test=[dict(episode_id=eid,seed=eps[eid]['seed'],reset_group=eps[eid]['reset_group']) for eid in split['splits']['test']],
        excluded_09d_final=[73141,73142],excluded_previous_final=[62091,62092],
        reset_identity_audit='NOT_TESTED until simulator preflight',selection=cfg['training']['validation_best']),immutable=True)
    print(dict(status='PASS',episodes=120,train_actions=sum(map(len,train_actions.values())),
        source=source_sha,max_fk_error=max(e['robot_fk_max_abs'] for e in rows)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--task08-plan',type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.dataset,a.task08_plan)


if __name__=='__main__':main()

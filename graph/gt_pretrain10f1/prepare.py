"""Source/contract audit and prospective data/optimizer freeze, before fitting."""
import argparse
import importlib.util
import inspect
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from graph.plan_a_bc.common import read, write, sha, digest, now, tree_identity
from utils.experiment_logger import ExperimentLogger
from .contract import feature_contract, OBJECT_FIELDS

PARENTS = ['3788798a719f4ec4a66c224ca42f5d63', '60ba447bf1b541e793a3d54ce68aa719',
           '771c27f66f6d4f27850a41a87478dc16']


def run(root, baseline, reach, official):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'run_identity.json').exists():
        raise FileExistsError('preparation already frozen')
    with ExperimentLogger(team='graph', model='Task10F1_GT_supervised_GNN', seed=42,
            config=dict(task='10-F1', parent_run_ids=PARENTS, no_policy=True),
            root=root/'experiments', job_type='prepare') as log:
        write(root/'run_identity.json', dict(run_id=log.run_id, parent_run_ids=PARENTS,
            created_at=now(), root=str(root), baseline=str(baseline), reach=str(reach), official=str(official)), immutable=True)
        for name in ('normalizer.json', 'renderer_gpu_gate.json', 'runtime_paths.json'):
            shutil.copy2(reach/name, root/name)
        shutil.copytree(reach/'inputs', root/'inputs')
        layout = read(official/'gt38_input_contract.json') if (official/'gt38_input_contract.json').exists() else None
        original = official/'inputs/original/recurrent'
        official_encoder = original/'features.py'
        if not official_encoder.exists():
            original = official/'source/recurrent'; official_encoder = original/'features.py'
        local_source = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('task10f1_official_encoder', official_encoder)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        from recurrent.features import FEATURE_LAYOUT
        assert module.FEATURE_LAYOUT == FEATURE_LAYOUT
        assert sha(official_encoder) == sha(local_source/'recurrent/features.py')
        checkpoints = []
        for seed in (0, 1, 2):
            path = official/f'inputs/training/outputs/recurrent/exp/main/runs/gru_none_s{seed}/best.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['obs_dim'] == 38
            stats = {k: np.asarray(v).tolist() for k, v in ck['stats'].items()}
            write(root/f'gt_gru_normalizer_seed{seed}.json', stats, immutable=True)
            checkpoints.append(dict(seed=seed, path=str(path), sha256=sha(path), obs_dim=38,
                stats=stats, model_config=ck.get('cfg'), state_dict_not_reused=True))
        import mani_skill.envs.tasks.tabletop.peg_insertion_side as task_module
        task_cls = task_module.PegInsertionSideEnv
        task_source = inspect.getsource(task_cls)
        (root/'installed_task_source.txt').write_text(task_source)
        source_identity = dict(created_at=now(), parent_run_ids=PARENTS,
            source_tree=tree_identity(local_source), original_encoder=dict(path=str(official_encoder), sha256=sha(official_encoder)),
            official_source={str(p):sha(p) for p in original.glob('*.py')},
            checkpoint_sources=checkpoints, installed_task_path=inspect.getfile(task_cls),
            installed_task_sha256=sha(inspect.getfile(task_cls)),
            source_backup_disclaimer='Task10E original source is archived backup; original dirty training execution equality unverified')
        write(root/'source_identity.json', source_identity, immutable=True)
        write(root/'gt38_feature_contract.json', feature_contract(), immutable=True)
        write(root/'target_observability_audit.json', dict(
            status='CONDITIONAL', definition_verified=True, inference_provenance_verified=True,
            measured_dimensions=list(range(15)), independent_object_targets=list(OBJECT_FIELDS),
            derived_dimensions=[22,23,24], clean_dimensions=[36,37],
            random_geometry='peg_half_sizes=[length,radius,radius]; hole_radius=radius+clearance; randomized during scene load',
            fixed_geometry_prior_used=False, no_runtime_gt_dimensions=True,
            hole_pose='box.pose * box_hole_offsets; randomized local offset; box centroid is NOT hole center',
            peg_rotation='square cross section and no RGB in Graph21: local x roll and end polarity can be ambiguous',
            orientation_loss='sign-invariant raw quaternion rotation loss; report unquotiented SO3 errors, not symmetry-resolved claims',
            unsupported='unique arbitrary-pose orientation under geometric symmetry, fully hidden object/rim/extent',
            policy_A_readiness='BLOCKED pending empirical errors and explicit handling of ambiguous/missing fields; no GT/default fill',
            permitted_scope='supervised estimates on task distribution; no claim of complete universal observability'), immutable=True)
        oldcfg = read(baseline/'config.json')
        train = list(oldcfg['splits']['train']); val = list(oldcfg['splits']['validation'])
        assert (len(train), len(val)) == (96, 12)
        final = list(range(91000, 91012))
        forbidden = set(oldcfg['splits']['test']) | set(range(61000, 61006)) | {73141,73142,62091,62092}
        assert not (set(train+val) & set(final) or set(final) & forbidden or set(train+val) & forbidden)
        paired_parents = dict(train=sorted(train)[::8], validation=sorted(val)[:4], final=final)
        pairs = dict(train=dict(anchors=[0,20], offsets_mm=[[7.5,0,0],[-7.5,0,0],[0,7.5,0],[0,-7.5,0]]),
            validation=dict(anchors=[0,20,80], offsets_mm=[[3.6,4.8,0],[-3.6,-4.8,0],[3.6,-4.8,0],[-3.6,4.8,0]]),
            final=dict(anchors=[0,20,80], offsets_mm=[[v if a==0 else 0,v if a==1 else 0,0] for a in (0,1) for v in (-10,-5,5,10)]))
        episodes = []
        for split, seeds in [('train',train),('validation',val),('final',final)]:
            for i, seed in enumerate(seeds):
                episodes.append(dict(episode_id=2000000+len(episodes), seed=int(seed), parent_id=int(seed), split=split,
                    paired=seed in paired_parents[split], paired_protocol=pairs[split], horizon=250))
        write(root/'split_manifest.json', dict(created_at=now(), train=train, validation=val, final=final,
            forbidden=sorted(forbidden), parent_disjoint=True, model_selection='validation only; final accessed after selection freeze'), immutable=True)
        write(root/'dataset_manifest.json', dict(version='10F1_full_task_v1', created_at=now(), episodes=episodes,
            scale_frozen_before_training=True, collection='unchanged ScriptedExpert full task, 250 actions or success/terminal',
            success_filter=False, record='same pre-action t: RGBD/Robot76/GT sidecar; retain T+1 unlabeled terminal obs',
            runtime='Task10D CPU PhysX + verified GPU renderer; no mixed historical inputs',
            collection_workers=4, graph_workers=4), immutable=True)
        write(root/'paired_intervention_manifest.json', dict(created_at=now(), parents=paired_parents, protocol=pairs,
            objects=['peg','box'], actual_render=True, robot_camera_prefix_exact=True,
            physics_steps_between_pair=0, physical_validation='original SAT collision_audit; invalid pairs retained and excluded from loss',
            pair_loss_fraction='16 pairs/update plus64 uniformly sampled absolute frames',
            prior10B_interventions_used=False), immutable=True)
        cfg = dict(oldcfg)
        cfg['splits']={k:[e['episode_id'] for e in episodes if e['split']==k] for k in ('train','validation','final')}
        cfg['splits']['test']=[]
        write(root/'config.json', cfg, immutable=True)
        write(root/'dataset_audit.json', dict(dataset_sha256=sha(root/'dataset_manifest.json'),
            split_sha256=sha(root/'split_manifest.json'), episodes=[]), immutable=True)
        pretrain = dict(created_at=now(), seed=42, architecture='E0 exact GraphEncoder L2/d64; Node21/Edge4; 5x64 slot order TCP/Left/Right/Peg/Box',
            head='MLP 398->256->128->18; readout320 + presence2 + train-standardized measured Robot76',
            independent_targets=list(OBJECT_FIELDS), no_GRU=True, no_action_head=True,
            optimizer='AdamW', lr=.0003, weight_decay=.00001, betas=[.9,.999], eps=1e-8,
            scheduler='cosine per optimizer update to zero', max_updates=12000, batch_absolute=64, batch_pairs=16,
            gradient_clip=1., validation_interval=250, early_stopping=None,
            selection='first minimum episode-macro absolute objective on validation; final inaccessible until freeze',
            position_scale_m=.1, size_scale_m=.02, rotation_loss='mean(1-abs(dot(unit_q,unit_gt_q))^2)',
            absolute_loss='MSE(pos6/.1) + mean quaternion losses peg/hole + MSE(shape4/.02)',
            delta_loss='MSE((predicted_delta_pos6 - GT_delta_pos6)/.01)', delta_weight=1.,
            normalization='positions/shapes train-only mean; fixed physical scales; quat unit wxyz; Robot76 train-only mean/std floor.001',
            baseline='Robot-only 76->256->128->18, same12k optimizer budget, same loss/split; raw Geometry30 ridge alpha1 fixed; train mean',
            primary_seed_count=1, no_architecture_search=True, bc_weights_loaded=False,
            target_scale_distinct_from_gt_gru_normalizer=True)
        write(root/'pretrain_config.json', pretrain, immutable=True)
        # Bounded, schema-level audits: do not pretend historical source has GT labels.
        audits=[]
        for label, folder in [('PlanA_Train96',baseline/'dataset'),('Task10D_Reach',reach/'raw')]:
            files=sorted(folder.rglob('*.h5')); examples=[]
            for p in files[:2]:
                with h5py.File(p) as f:
                    keys=[]; f.visititems(lambda n,o: keys.append(n) if isinstance(o,h5py.Dataset) else None)
                examples.append(dict(path=str(p), datasets=keys))
            audits.append(dict(source=label, hdf5_count=len(files), inspected=examples,
                reusable_for_full_task_gt_supervision=False,
                reason='no per-action object GT fields in HDF5; Task10D also reach-only'))
        write(root/'existing_data_audit.json', dict(audits=audits,
            task09D='Train8/Validation4 pilot labels documented; small and different historical runtime, not merged',
            task10B='evaluation interventions excluded from all fitting',
            recurrent='GT38 checkpoint and encoder reused as contract only; historical RGBD-aligned Graph absent',
            decision='fresh synchronized120 full-task episodes'), immutable=True)
        for path in root.glob('*.json'): log.link_output(path,kind='metrics')
        log.record(parent_run_ids=PARENTS)
        print('PREPARED', log.run_id, len(episodes), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reach',type=Path,required=True)
    p.add_argument('--official',type=Path,required=True);a=p.parse_args();run(a.root,a.baseline,a.reach,a.official)

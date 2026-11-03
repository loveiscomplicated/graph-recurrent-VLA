"""Freeze identities, reconstruct host tensors, cache frozen sensor estimates."""
import argparse
import copy
import os
import shutil
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read, write, sha, npz, now, tensor_identity
from graph.graph_recurrent10g.data import prepare as prepare_g, Dataset
from graph.gt_pretrain10f1.model import restore_checkpoint
from graph.gt_pretrain10f1.contract import feature_contract
from graph.integration_baseline.direct_train import seed_all
from .model import Policy, fit_normalizer

F1_SHA='781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6'
MANIFEST_SHA='f1b61e7635be3a9b1533d5158e8be8df869edf67dddee926335c68486d5cf6b1'
OLD=Path('/workspace/task10g_f1_abc_20261031/full')


def initialize(root):
    if (root/'config.json').exists():
        return read(root/'config.json')
    old=read(OLD/'config.json'); cfg=copy.deepcopy(old)
    cfg.update(task='10-I',root=str(root),reference_root=str(OLD),miniature=False,
               miniature_root=None,reuse_verified_data=None,no_callback=True,no_publish=True)
    cfg['evaluation']['final_seeds']=list(range(118200,118212))
    cfg['evaluation']['smoke_seed']=old['evaluation']['dev_seeds'][0]
    cfg['policy']=dict(input_dimension=38,projection=128,gru=144,action=7,
        frozen=['F1 E0 GNN','parameter-free Readout','F1 state head'],fresh_control_seed=42,
        object_GT_input=False,Readout_bypass=False)
    cfg['runtime']=dict(instance=58752018,rollout_workers=1,training_workers=1,
        allocator='expandable_segments:True',simulation='physx_cpu',renderer='sapien_cuda',
        cuda='vision/policy',tmux='task10i_predstate38_bc',automatic_retry=False,
        persistence='tmux survives SSH disconnect, not instance stop/delete/recycle')
    cfg['tolerances']=dict(state_atol=2e-6,state_rtol=1e-6,action_atol=2e-6,action_rtol=1e-6,
        quaternion_atol=2e-5,oracle_assembly_atol=2e-7,resume='exact tensors/loss',
        positive_shape_min=0.,positive_shape_max_m=1.)
    # All decisions precede any new policy outputs.
    write(root/'config.json',cfg,immutable=True)
    write(root/'train_config.json',cfg,immutable=True)
    selection=read(OLD/'model_selection_frozen.json')['conditions']
    comparisons={k:dict(condition=c,path=str(OLD/selection[c]['path']),sha256=selection[c]['sha256'])
                 for k,c in [('R','A'),('J','B')]}
    assert comparisons['J']['sha256']=='8ab82f390dccf054b0bd6539dbd55f7a60af2ae7934f53234702ab66c1369ea3'
    assert sha(cfg['f1_checkpoint'])==F1_SHA
    assert sha(Path(cfg['f1_root'])/'final_dataset_manifest.json')==MANIFEST_SHA
    for row in comparisons.values(): assert sha(row['path'])==row['sha256']
    write(root/'comparison_policy_manifest.json',comparisons,immutable=True)
    split=read(Path(cfg['f1_root'])/'final_dataset_manifest.json')
    seen={s['seed'] for s in split['episode_specs']}
    prior=set(range(98200,98212))|set(range(108200,108212))|set(range(91000,91012))
    final=cfg['evaluation']['final_seeds']
    assert not set(final)&(seen|prior|set(cfg['evaluation']['dev_seeds']))
    write(root/'final_seed_manifest.json',dict(created_at=now(),parents=final,
        conditions=['E','S0','S1','S2','R0','R1','R2','J0','J1','J2'],episodes=120,
        horizon_including_prefix=250,representative_parent=final[0],
        excluded_previous_final=sorted(prior),selection_locked_before_final=True,
        handoff='Task10H original lift->align and align->insert; no reinterpretation as contact-near'),immutable=True)
    write(root/'state38_input_contract.json',feature_contract(),immutable=True)
    with ExperimentLogger(team='graph',model='Task10I_predicted_State38_protocol',seed=42,
             config=cfg,root=root/'experiments',job_type='prepare') as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,created_at=now(),instance=os.getenv('CONTAINER_ID')))
        log.record(f1_checkpoint_sha256=F1_SHA,dataset_sha256=MANIFEST_SHA)
        log.link_output(root/'train_config.json')
    return cfg


def source_audit(root):
    cfg=read(root/'config.json');old=Path(cfg['reference_root'])
    protected={str(p):sha(p) for p in old.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    # Entire original G source, F1 source, H runner/analysis + checkpoints/contracts.
    for base in [old.parent/'source',Path(cfg['f1_root'])/'source']:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
    for base in [Path(cfg['f1_root']),Path('/workspace/task10h_phase_handoff_20261031_v5')]:
        for p in base.iterdir():
            if p.is_file() and p.suffix in ('.py','.pt','.json','.md'):protected[str(p)]=sha(p)
    f1,ck=restore_checkpoint(cfg['f1_checkpoint'],'cuda')
    layers=[list(m.weight.shape) for m in f1.state_head.net if isinstance(m,torch.nn.Linear)]
    assert layers==[[256,398],[128,256],[18,128]],layers
    write(root/'source_checkpoint_audit.json',dict(status='PASS',protected=protected,
        f1_sha256=sha(cfg['f1_checkpoint']),state_head_weight_shapes=layers,
        readout='5x64 parameter-free structured readout',robot_schema=ck['robot_schema'],
        train_budget=cfg['training'],packages={p:importlib.metadata.version(p) for p in
        ['torch','torchvision','mani_skill','sapien','gymnasium','numpy']},
        torch_cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        free_vram=torch.cuda.mem_get_info()[0],disk_free=shutil.disk_usage(root).free,
        source_runtime_snapshot=str(root/'source'),originals_used_readonly=True))
    del f1;torch.cuda.empty_cache()


def prepare_data(root):
    cfg=read(root/'config.json')
    if not (root/'dataset_identity.json').exists():
        prepare_g(root)
    for name in ['sampling_schedule.json','sampling_manifest.json','action_codec.json','normalizer.json']:
        assert sha(root/name)==sha(OLD/name),('G exact reuse',name)
    write(root/'data_reconstruction_verification.json',dict(status='PASS',
        prior_mmap_missing=True,graph_recache=False,exact_schedule_sha256=sha(root/'sampling_schedule.json'),
        codec_sha256=sha(root/'action_codec.json'),normalizer_sha256=sha(root/'normalizer.json')))


def cache(root):
    cfg=read(root/'config.json');ds=Dataset(root)
    model,ck=restore_checkpoint(cfg['f1_checkpoint'],'cuda')
    before=tensor_identity(model.state_dict());states={};records=[]
    for eid,e in ds.episodes.items():
        ix=e['indices'];parts=[]
        for start in range(0,len(ix),32):
            g,r,_=ds.select(ix[start:start+32],'cuda')
            # Official F1 extraction. GT arrays do not enter this call.
            parts.append(model.extract(g,r,ds.schema)['state38'])
        state=np.concatenate(parts).astype('f');assert state.shape==(e['actions'],38)
        assert np.isfinite(state).all();states[eid]=state
        path=root/'features'/e['split']/(eid+'.npz')
        arrays=dict(state38=state,action7=ds.actions[ix],supervision_mask=ds.mask[ix],
            timestep=np.arange(len(ix),dtype=np.int64),episode=np.full(len(ix),int(eid),np.int64),
            parent=np.full(len(ix),e['parent'],np.int64),branch=np.array(e['branch']),
            boundary=np.arange(len(ix))==0)
        npz(path,arrays)
        source=read(root/'dataset_identity.json')['sources'][list(ds.episodes).index(eid)]
        records.append(dict(episode=eid,parent=e['parent'],branch=e['branch'],split=e['split'],
            frames=len(ix),path=str(path.relative_to(root)),sha256=sha(path),
            graph_cache_sha256=source['graph_sha256'],raw_source_sha256=source['sha256']))
        if len(records)%25==0: print('FEATURE_CACHE',len(records),flush=True)
    assert before==tensor_identity(model.state_dict())
    train=np.concatenate([states[eid] for eid,e in ds.episodes.items() if e['split']=='train'])
    stats=fit_normalizer(train)
    write(root/'state38_normalizer.json',stats,immutable=True)
    source={str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*.py') if '__pycache__' not in p.parts}
    write(root/'state38_feature_cache_manifest.json',dict(status='PASS',episodes=records,
        F1_checkpoint_sha256=sha(cfg['f1_checkpoint']),frozen_weight_hash=before,
        robot_schema=ds.schema,input_feature_contract=read(root/'state38_input_contract.json'),
        normalizer_sha256=sha(root/'state38_normalizer.json'),normalization_fit='train predicted frames only',
        source_hashes=source,runtime=cfg['runtime'],GT_policy_input=False),immutable=True)
    write(root/'training_protocol_frozen.json',dict(config_sha256=sha(root/'config.json'),
        sampling_sha256=sha(root/'sampling_schedule.json'),normalizer_sha256=sha(root/'state38_normalizer.json'),
        codec_sha256=sha(root/'action_codec.json'),feature_cache_sha256=sha(root/'state38_feature_cache_manifest.json'),
        checkpoint_steps=cfg['training']['checkpoint_steps'],development_seeds=cfg['evaluation']['dev_seeds'],
        selection_rule=cfg['evaluation']['selection'],final_manifest_sha256=sha(root/'final_seed_manifest.json')),immutable=True)


def verify_cache(root):
    manifest=read(root/'state38_feature_cache_manifest.json')
    runtime=read(root/'state38_feature_cache_runtime_identity.json')
    assert runtime['feature_manifest_sha256']==sha(root/'state38_feature_cache_manifest.json')
    assert runtime['source_audit_sha256']==sha(root/'source_checkpoint_audit.json')
    for package,version in runtime['packages'].items():assert importlib.metadata.version(package)==version
    assert runtime['torch_cuda']==torch.version.cuda
    assert sha(read(root/'config.json')['f1_checkpoint'])==manifest['F1_checkpoint_sha256']
    assert sha(root/'state38_normalizer.json')==manifest['normalizer_sha256']
    for row in manifest['episodes']:
        assert sha(root/row['path'])==row['sha256']
        with np.load(root/row['path'],allow_pickle=False) as z:
            assert z['state38'].shape==(row['frames'],38) and z['state38'].dtype==np.float32
            assert np.isfinite(z['state38']).all()
            np.testing.assert_array_equal(z['timestep'],np.arange(row['frames']))
    write(root/'train_feature_cache_verification.json',dict(status='PASS',episodes=len(manifest['episodes'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['initialize','audit','data','cache','verify'],required=True)
    a=p.parse_args();torch.set_num_threads(1);seed_all(42,1)
    {'initialize':initialize,'audit':source_audit,'data':prepare_data,'cache':cache,'verify':verify_cache}[a.stage](a.root)

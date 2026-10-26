"""Freeze Task08 inputs/selection before inspecting a trained policy output."""
import argparse
import copy
from pathlib import Path

import h5py
import numpy as np
import torch

from graph.integration_baseline.bc08_codec import ActionCodec, digest
from graph.integration_baseline.bc08_train import VERSION, tensor_hash, normalizer_hash
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.direct_dataset import DirectDataset, validate_receipt
from graph.integration_baseline.direct_train import seed_all, normalizer_state
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.rollout_checkpoint import load_bundle
from utils.experiment_logger import ExperimentLogger


def select_overfit(dataset):
    """First five sorted train IDs with distinct reset group and complete phases.

    Missing coverage/invalid input is an error, not a policy-performance filter.
    """
    selected=[];seen=set()
    for eid in sorted(dataset.splits['train']):
        ep=dataset.episodes[eid]
        if ep['reset_group'] in seen:continue
        stages={r['before'] for r in ep['phases']}
        if not {'reach','close','lift','align','insert'} <= stages:
            raise ValueError(f'first deterministic candidate {eid} lacks complete stages: {stages}')
        if not ep['success'] or ep['observations'] != ep['actions']+1 or ep['first_success_step'] != ep['actions']:
            raise ValueError('candidate is not complete through first success')
        selected.append(eid);seen.add(ep['reset_group'])
        if len(selected)==5:break
    if len(selected)!=5:raise ValueError('five distinct complete train episodes required')
    return selected


def prepare(dataset_directory, checkpoint_manifest, output_root, *, device='cuda'):
    dataset_directory=Path(dataset_directory).resolve();output_root=Path(output_root).resolve()
    ds=DirectDataset(dataset_directory)
    if {k:len(v) for k,v in ds.splits.items()} != dict(train=96,validation=12,test=12) or len(ds.graph_manifest['episodes'])!=120:
        raise ValueError('full existing Train96/Validation12/Test12 and 120 causal graph episodes required')
    source_model,norm,provenance=load_bundle(checkpoint_manifest,device='cpu',check_runtime=False)
    if provenance['manifest']['source_run_id'] not in ('2be5e62357a44b14a5651a1a346edc7a','002abd72baf84c4da185fd7134c3146f'):
        raise ValueError('unexpected reference run')
    if provenance['manifest']['files']['checkpoint']['sha256']!='3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776':
        raise ValueError('reference best checkpoint mismatch')
    if ds.schema != norm.schema or norm.fitted_episode_ids != sorted(ds.splits['train']):
        raise ValueError('reuse original Train96 Robot76 normalizer exactly')
    for root,name,key in [(ds.run,'dataset_manifest.json','dataset_sha256'),(ds.run,'split_manifest.json','split_sha256'),
                          (ds.output,'vision_manifest.json','vision_sha256')]:
        if sha(root/name)!=provenance['identity'][key]:raise ValueError('reference/dataset identity mismatch: '+key)
    config=copy.deepcopy(read('graph/integration_baseline/direct_config.json'))
    config.update(task='Task08',device=device,vision_device=device,threads=1,auto_callback_required=False,auto_repair_attempts=0)
    config['training']['patience']=None  # explicitly disabled; always same 50 epochs
    config['training']['loss']='method-dependent Task08 codec'
    if config['model']!=provenance['identity']['model_schema'] or config['graph']!=provenance['identity']['graph_config']:
        raise ValueError('fixed model/graph architecture mismatch')
    ids=select_overfit(ds)
    with ExperimentLogger(team='integration',model='task08_comparison',seed=42,
        config=config,root=output_root,job_type='comparison') as logger:
        run=logger.directory
        input_hashes={};episode_hashes={};selected=[]
        for root,names in [(ds.run,['config.yaml','dataset_manifest.json','split_manifest.json','robot_state_schema.json','expert_action_schema.json']),
                           (ds.output,['vision_manifest.json'])]:
            for name in names:input_hashes[str(root/name)]=sha(root/name)
        input_hashes[str(Path(checkpoint_manifest).resolve())]=sha(checkpoint_manifest)
        for item in provenance['manifest']['files'].values():
            input_hashes[str((Path(checkpoint_manifest).resolve().parent/item['path']).resolve())]=item['sha256']
        protection=Path(__file__).resolve().parents[2]/'protected_before.json'
        if protection.exists():input_hashes[str(protection)]=sha(protection)
        action_arrays={}
        # No writes into dataset: validate through existing read-only accessors.
        for eid,ep in ds.episodes.items():
            a=validate_receipt(ds.run,ep,ds.schema)
            if np.flatnonzero(a['success']).tolist()!=[ep['actions']-1]:
                raise ValueError('episode must end at its first success')
            if not np.array_equal(a['actions'],a['controller_input']) or not np.array_equal(a['actions'],a['env_step_input']):
                raise ValueError('native expert/controller target mismatch')
            rows=[ds.packet_rows[(eid,t)] for t in range(ep['observations'])]
            for t in range(ep['actions']):ds.frame(eid,t)
            if sha(ds.output/rows[-1]['path'])!=rows[-1]['sha256']:
                raise ValueError('terminal packet SHA mismatch')
            entry=dict(episode_id=eid,seed=ep['seed'],reset_group=ep['reset_group'],actions=ep['actions'],
                observations=ep['observations'],raw_sha256=ep['sha256'],
                packet_sequence_sha256=digest([r['sha256'] for r in rows]))
            episode_hashes[str(eid)]=digest(entry)
            input_hashes[str(ds.run/ep['path'])]=ep['sha256']
            if eid in ids:selected.append(entry)
            if eid in ds.splits['train']:action_arrays[eid]=a['actions']
        seed_all(42,1);model=IntegrationPolicy(ds.schema,**config['model'])
        initial=run/'initial.pt';torch.save(dict(model=model.state_dict(),normalizer=normalizer_state(norm)),initial)
        input_hashes[str(initial.resolve())]=sha(initial)
        code={str(p.resolve()):sha(p) for folder in ['graph','recurrent','envs','visual/runtime','utils','tools']
              for p in sorted(Path(folder).rglob('*.py'))}
        # Freeze new development/final seeds and runtime reset groups BEFORE training.
        # Reset preflight is its own process (bc08_eval --freeze); no model outputs.
        plan=dict(version=VERSION,config=config,experiments_root=str(output_root),
            dataset_directory=str(dataset_directory),reference_checkpoint_manifest=str(Path(checkpoint_manifest).resolve()),
            reference_provenance=provenance,overfit_ids=ids,splits=ds.splits,
            overfit_selection_rule='ascending frozen train episode ID, distinct reset group; first five; complete phases required, no policy outputs',
            overfit_selection=sorted(selected,key=lambda e:e['episode_id']),episode_input_hashes=episode_hashes,
            episode_lengths={str(e):v['actions'] for e,v in ds.episodes.items()},
            initial_path=str(initial.resolve()),initial_weight_identity=tensor_hash(model.state_dict()),
            robot_normalizer_identity=normalizer_hash(normalizer_state(norm)),
            input_hashes=input_hashes,source_hashes=code,overfit_updates=2000,
            protection_manifest=str(protection) if protection.exists() else None,
            offline_fit_thresholds=dict(native_action7_mse=.001,native_motion6_mse=.0001),
            offline_fit_additional='>=95% native Action7 error reduction and gripper accuracy >=.99; operational fit criterion, not manipulation success',
            minimum_free_bytes=8*1024**3,probe_barrier=False,
            development_seeds=list(range(42000,42010)),final_seeds=list(range(43000,43040)),
            diagnostic_seeds=[41005],old_comparison_seeds=list(range(41000,41007)),
            selection='Full best by validation native_action7_mse only; all ABC evaluated; no development/test tuning',
            codecs={phase:{m:ActionCodec.fit(m,{e:action_arrays[e] for e in subset},ds.splits['train']).state
                for m in ('A','B','C')} for phase,subset in [('overfit',ids),('full',ds.splits['train'])]})
        candidates=[]
        for start in range(0,max(e['actions'] for e in ds.episodes.values()),32):
            available=[e for e in ds.splits['train'] if ds.episodes[e]['actions']>=start+32]
            ranked=sorted(available,key=lambda e:(-sum(ds.packet_rows[(e,t)]['point_count'] for t in range(start,start+32)),e))
            if len(ranked)>=2:
                pair=ranked[:2];weight=sum(ds.packet_rows[(e,t)]['point_count'] for e in pair for t in range(start,start+32))
                candidates.append((weight,pair,start))
        weight,pair,start=max(candidates,key=lambda x:x[0])
        plan['probe_batch']=dict(episode_ids=pair,start=start,point_count=weight,
            rule='largest observed point-count B2xT32 among existing train packets; discarded optimization probes only')
        atomic(run/'plan.json',plan,immutable=True)
        atomic(run/'overfit_selection.json',dict(rule=plan['overfit_selection_rule'],episodes=plan['overfit_selection']),immutable=True)
        atomic(run/'status.json',dict(status='PREPARED',stage='reset_identity_freeze'))
        from graph.integration_baseline.bc08_report import snapshot
        snapshot(run,plan)
        logger.link_output(run/'plan.json',kind='metrics')
        print('PLAN='+str(run/'plan.json'),flush=True)
        return run/'plan.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--checkpoint-manifest',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    prepare(a.dataset,a.checkpoint_manifest,a.output_root,device=a.device)


if __name__=='__main__':main()

"""Freeze collection candidates before any A/B weights or policy outcomes."""
import argparse
from pathlib import Path
import numpy as np
from graph.plan_a_bc.common import read,write,sha,digest,now
from graph.geometry_reach10d.analysis_pilot import run as pilot_analysis


def prepare(root):
    if (root/'collection_protocol.json').exists():return
    assert read(root/'optimization_equivalence.json')['status']=='PASS'
    pilot_analysis(root)
    assert read(root/'pilot_data_verification.json')['status']=='PASS'
    baseline=Path(read(root/'run_identity.json')['baseline']);old=read(baseline/'config.json')
    seeds=sorted(old['splits']['train']);parents=seeds[::4]
    assert len(parents)==24
    offsets=[[0,0,0],[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]]
    pilot={ (r['spec']['seed'],r['spec']['variant']):r['spec'] for r in read(root/'capture_manifest.json')['episodes'] }
    entries=[]
    for j,seed in enumerate(seeds):
        spec=pilot.get((seed,'P0'),dict(episode_id=1100000+j*10,seed=seed,variant='P0',
            offset_mm=offsets[0],split='train',parent_id=seed))
        entries.append(dict(spec=spec,effective_split='train',data_kind='original'))
    for j,seed in enumerate(parents):
        for k in range(1,5):
            spec=pilot.get((seed,'P'+str(k)),dict(episode_id=1200000+j*10+k,seed=seed,variant='P'+str(k),
                offset_mm=offsets[k],split='train',parent_id=seed))
            entries.append(dict(spec=spec,effective_split='train',data_kind='perturbed'))
    for j,seed in enumerate(range(60000,60004)):
        for k,offset in enumerate([[0,0,0],[3,4,0]]):
            spec=dict(episode_id=1300000+j*10+k,seed=seed,variant=['nominal','diagonal'][k],
                offset_mm=offset,split='validation',parent_id=seed)
            entries.append(dict(spec=spec,effective_split='validation',data_kind='development'))
    final=[dict(seed=seed,offset_mm=offset,variant=variant,group=variant)
        for seed in range(61000,61006)
        for variant,offset in [('nominal',[0,0,0]),('seen_type',[5,0,0]),('unseen_diagonal',[3,4,0])]]
    for specs in [entries,final]:
        assert not (set(e.get('spec',e)['seed'] for e in specs)&set(old['splits']['test']))
    protocol=dict(version='10D_v1',created_at=now(),comparison_uuid=read(root/'run_identity.json')['run_id'],
        simulator='same CPU PhysX/controller/camera contract as Plan A; current-runtime recollection for ALL originals',
        renderer_selection=read(root/'renderer_gpu_gate.json'),
        train_original=96,perturbed_parents=24,perturbed_per_parent=4,perturbed_candidates=96,
        pilot_in_training='reuse accepted raw episodes and completed pilot graph tensors after source/identity audit',
        reach_phases=['reach','descend'],stop='after open descend action transitions unchanged ScriptedExpert to close; never record close',
        threshold=dict(approach_position_m=.01,approach_orientation_rad=.05,
            final_position_m=.004,final_orientation_rad=.03,max_peg_motion_m=.002,opening_min_m=.065),
        source_expert_sha256=sha(Path(__file__).resolve().parents[2]/'recurrent/expert.py'),
        training_updates=11487,seed=42,mixture='one original + one perturbed per episode pair, about 50% frames; exact matching of A replacement episode lengths',
        unmatched_length_rule='all captures retained; perturbed lengths absent from original96 cannot enter paired sampler; report exclusion before training',
        optimizer='existing AdamW and original Full96 scheduler progression by reference update boundaries',
        sequence_length=32,batch_size=2,hidden_carry=True,TBPTT=True,
        normalizers='unchanged Train96 Robot76 and Method B motion6',
        offline_validation_steps=list(range(1000,11001,1000))+[11487],
        live_development_steps=[3000,6000,9000,11487],
        live_development=[dict(seed=60000,offset_mm=[0,0,0],variant='nominal'),dict(seed=60001,offset_mm=[3,4,0],variant='unseen_diagonal')],
        checkpoint_selection='first minimum validation Method B loss; final on selected checkpoint only',
        final_cases=final,final_intervention_seeds=[61004,61005],development_intervention_seeds=[60002],
        intervention_offsets_mm=[[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]],
        evaluation_horizon=130,primary='initial Peg offset Reach; not dynamic recovery, Grasp or Insertion',
        no_policy_outputs_used=True)
    write(root/'parent_seed_manifest.json',dict(method='sorted original Train96[::4], before policy results',parents=parents),immutable=True)
    write(root/'peg_offset_manifest.json',dict(train=offsets[1:],development=[[3,4,0]],final=final),immutable=True)
    write(root/'collection_candidates.json',entries,immutable=True)
    write(root/'collection_protocol.json',protocol,immutable=True)
    cfg=read(root/'config.json');cfg['training']['updates']=11487
    cfg['splits']={k:[e['spec']['episode_id'] for e in entries if e['effective_split']==k] for k in ['train','validation']}
    cfg['splits']['test']=[];cfg['episode_lengths']={}
    write(root/'config.json',cfg)
    write(root/'dataset_audit.json',dict(dataset_sha256=sha(root/'collection_candidates.json'),
        split_sha256=digest([(e['spec']['episode_id'],e['effective_split']) for e in entries]),
        identity_semantics='candidate generation protocol; raw episode SHA stored in each graph receipt',episodes=[]),immutable=True)
    print('COLLECTION FROZEN',len(entries),'including192 train +8 development candidates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root)

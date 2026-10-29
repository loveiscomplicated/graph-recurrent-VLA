"""Freeze paired episode-contiguous exposures after engineering/data gates."""
import argparse
import collections
from pathlib import Path
import numpy as np
from graph.plan_a_bc.common import read,write,sha,npz,append_csv,digest,now
from graph.plan_a_bc.train import chunks


def run(root):
    if (root/'dataset_ab_manifest.json').exists():return read(root/'dataset_ab_manifest.json')
    assert read(root/'optimization_equivalence.json')['status']=='PASS'
    assert read(root/'seed_novelty_audit.json')['status']=='PASS'
    assert read(root/'analysis_protocol.json')['before_AB_models']
    inherited=read(root/'inherited_source_identity.json')
    source_root=Path(__file__).resolve().parents[2]
    mismatches=[name for name,value in inherited.items() if not (source_root/name).is_file() or sha(source_root/name)!=value]
    assert not mismatches,('inherited source changed',mismatches)
    inherited_gate=dict(status='PASS',files=len(inherited),mismatches=[])
    if (root/'inherited_source_final_gate.json').exists():
        assert read(root/'inherited_source_final_gate.json')==inherited_gate
    else:write(root/'inherited_source_final_gate.json',inherited_gate,immutable=True)
    m=read(root/'graph_cache_manifest.json');assert m['complete_train_validation']
    from graph.geometry_reach10d.audit_graph_signal import run as graph_signal
    graph_signal(root)
    ep={e['episode_id']:e for e in m['episodes']}
    original=[e for e in m['episodes'] if e['data_kind']=='original']
    perturbed=[e for e in m['episodes'] if e['data_kind']=='perturbed']
    validation=[e for e in m['episodes'] if e['split']=='validation']
    assert len(original)==96 and len(validation)==8,'baseline/validation missing; do not silently shrink'
    by_length=collections.defaultdict(list)
    for e in original:by_length[e['actions']].append(e['episode_id'])
    original_by_seed={e['seed']:e['episode_id'] for e in original}
    eligible=[e for e in perturbed if e['actions'] in by_length]
    excluded=[e for e in perturbed if e['actions'] not in by_length]
    assert eligible,'no perturbed episode can match exposure budget'
    cfg=read(root/'config.json');cfg['splits']['train']=[e['episode_id'] for e in original+eligible]
    cfg['splits']['validation']=[e['episode_id'] for e in validation]
    cfg['episode_lengths']={str(e['episode_id']):e['actions'] for e in m['episodes']}
    write(root/'config.json',cfg)
    rng=np.random.default_rng(42);plan=[];cycle=0
    while len(plan)<11487:
        cycle+=1
        orig=rng.permutation([e['episode_id'] for e in original]).tolist()
        pert=rng.permutation([e['episode_id'] for e in eligible]).tolist()
        for i in range(max(len(orig),len(pert))):
            o,p=orig[i%len(orig)],pert[i%len(pert)]
            parent_original=original_by_seed[ep[p]['seed']]
            replacement=parent_original if ep[parent_original]['actions']==ep[p]['actions'] else int(rng.choice(by_length[ep[p]['actions']]))
            for start in range(0,max(ep[o]['actions'],ep[p]['actions']),32):
                aa=[e for e in [o,replacement] if start<ep[e]['actions']]
                bb=[e for e in [o,p] if start<ep[e]['actions']]
                na=sum(min(32,ep[e]['actions']-start) for e in aa)
                nb=sum(min(32,ep[e]['actions']-start) for e in bb)
                assert na==nb and [ep[e]['actions'] for e in aa]==[ep[e]['actions'] for e in bb]
                plan.append(dict(A=aa,B=bb,start=start,cycle=cycle,frames=na))
                if len(plan)==11487:break
            if len(plan)==11487:break
    freq={c:{e:np.zeros(ep[e]['actions'],np.int64) for e in ep} for c in ['A','B']}
    for row in plan:
        for c in freq:
            for e in row[c]:freq[c][e][row['start']:row['start']+32]+=1
    arrays={c:np.array([r[c]+[-1]*(2-len(r[c])) for r in plan],np.int64) for c in ['A','B']}
    arrays.update(start=np.array([r['start'] for r in plan]),cycle=np.array([r['cycle'] for r in plan]),
        frames=np.array([r['frames'] for r in plan]))
    npz(root/'paired_training_schedule.npz',arrays)
    exposures={}
    for c in freq:
        exposures[c]={}
        for e,counts in freq[c].items():
            exposures[c][str(e)]=dict(frames=int(counts.sum()),unique_frames=int((counts>0).sum()),
                min_frequency=int(counts.min()),max_frequency=int(counts.max()),parent_seed=ep[e]['seed'],
                data_kind=ep[e]['data_kind'])
            for t,count in enumerate(counts):append_csv(root/'sampling_frequency.csv',
                dict(condition=c,episode_id=e,parent_seed=ep[e]['seed'],timestep=t,count=int(count),data_kind=ep[e]['data_kind']))
    baseline=read(Path(read(root/'run_identity.json')['baseline'])/'config.json')
    boundaries=[];steps=0;epoch=1
    while steps<11487:
        steps+=len(chunks(baseline,epoch));boundaries.append(steps);epoch+=1
    result=dict(created_at=now(),frozen_before_AB_model_initialization=True,
        config_sha256=sha(root/'config.json'),cache_manifest_sha256=sha(root/'cache_manifest.json'),
        collection_protocol_sha256=sha(root/'collection_protocol.json'),
        sampler_sha256=sha(root/'paired_training_schedule.npz'),normalizer_sha256=sha(root/'normalizer.json'),
        seed_novelty_audit_sha256=sha(root/'seed_novelty_audit.json'),analysis_protocol_sha256=sha(root/'analysis_protocol.json'),
        codec_sha256=sha(root/'action_codec.json'),updates=11487,equal_timesteps_each_update=True,
        total_timesteps={c:sum(x['frames'] for x in exposures[c].values()) for c in exposures},
        exposures=exposures,eligible_perturbed=len(eligible),captured_perturbed=len(perturbed),
        excluded_unmatched_lengths=excluded,reference_scheduler_boundaries=boundaries,
        dataset_cycles=cycle,unique_original_episodes=96,validation_episodes=8,
        parent_replacement_rule='same parent original whenever length matches; otherwise original of identical length; no truncation',
        parent_reweighting_caveat='Length-matched replacements can have another parent; all parent/frame exposure weights are explicit. Geometry inclusion is not isolated from this remaining reweighting.',
        exact_mixture_B_perturbed_frame_fraction=sum(v['frames'] for v in exposures['B'].values() if v['data_kind']=='perturbed')/sum(arrays['frames']),
        source_hashes={str(p.relative_to(Path(__file__).resolve().parents[2])):sha(p)
            for p in Path(__file__).resolve().parent.glob('*.py')})
    assert result['total_timesteps']['A']==result['total_timesteps']['B']
    write(root/'training_config.json',dict(model=cfg['model'],training=cfg['training'],
        schedule='original Full96 scheduler at identical reference update boundaries',
        offline_validation_steps=read(root/'collection_protocol.json')['offline_validation_steps'],
        live_development_steps=read(root/'collection_protocol.json')['live_development_steps']),immutable=True)
    write(root/'dataset_ab_manifest.json',result,immutable=True)
    print('A/B FROZEN',result['total_timesteps'],'perturbed',len(eligible),'excluded',len(excluded),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

"""Diagnostic moments only. Never creates or changes an ActionCodec."""
import csv
from pathlib import Path
import h5py
import numpy as np
from graph.plan_a_bc.common import read,write


def run(root,captures,freeze):
    by_id={};groups={k:[] for k in ['current_original_reach','perturbed_reach','development_reach']}
    kinds={e['spec']['episode_id']:e['data_kind'] for e in read(root/'collection_candidates.json')}
    for r in captures:
        if r['status']!='ACCEPTED':continue
        with h5py.File(r['raw_path']) as f:action=f['actions'][:].astype(float)
        eid=r['spec']['episode_id'];by_id[eid]=action
        group={'original':'current_original_reach','perturbed':'perturbed_reach','development':'development_reach'}[kinds[eid]]
        groups[group].append(action)
    stats=read(root/'action_codec.json');mu=np.array(stats['mean6']);sd=np.array(stats['std6'])
    result={};table=[]
    def summarize(name,a):
        z=(a[:,:6]-mu)/sd
        data=dict(count=len(a),native_mean=a.mean(0).tolist(),native_std=a.std(0).tolist(),
            motion6_z_mean=z.mean(0).tolist(),motion6_z_std=z.std(0).tolist(),
            motion6_z_rms=np.sqrt((z*z).mean(0)).tolist())
        result[name]=data
        for j in range(7):table.append(dict(group=name,channel=j,count=len(a),native_mean=data['native_mean'][j],
            native_std=data['native_std'][j],frozen_Train96_mean=mu[j] if j<6 else None,
            frozen_Train96_std=sd[j] if j<6 else None,z_mean=data['motion6_z_mean'][j] if j<6 else None,
            z_std=data['motion6_z_std'][j] if j<6 else None))
    for name,arrays in groups.items():
        if arrays:summarize(name,np.concatenate(arrays))
    if freeze:
        with np.load(root/'paired_training_schedule.npz') as schedule:
            for condition in ['A','B']:
                exposed=[]
                for ids,start in zip(schedule[condition],schedule['start']):
                    for eid in ids:
                        if eid>=0:exposed.append(by_id[int(eid)][int(start):int(start)+32])
                summarize(condition+'_frozen_sampler_exposure',np.concatenate(exposed))
    write(root/'action_distribution.json',dict(groups=result,frozen_reference_codec=stats,
        normalization_refitted=False,used_for_training_inputs=False,
        interpretation='Reach-only and extra-position action shifts relative to the unchanged full-Train96 statistics are diagnostics, not new normalizers'))
    with (root/'action_distribution.csv').open('w',newline='') as f:
        if table:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    return result

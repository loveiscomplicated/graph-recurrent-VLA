"""Apply already selected GNNs/probes to newly collected evaluation episodes.

This runner has no optimizer, fit or model-selection call. Target scalers are
loaded from each development checkpoint; probe scalers from its saved decoder.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch

from graph.integration_baseline.io import save_npz,write_json,sha256
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.geometry09b_probe import csv_write,absolute
from utils.experiment_logger import ExperimentLogger
from .train import Dataset,parameter_hash,permutation_audit
from .encoder import GeometryRegressor,geometry_summary
from .probes import read_features,probe_features,decode_saved,metric_rows
from .audit import protection_snapshot,verify_protection
from .collect import inventory_sources


def final_gate(training,probes,geometry):
    freeze=json.loads((probes/'frozen_selection.json').read_text())
    if sha256(training/'model_manifest.json')!=freeze['model_manifest_sha256']:
        raise ValueError('model selection changed after freeze')
    if sha256(probes/'probe_manifest.json')!=freeze['probe_manifest_sha256']:
        raise ValueError('probe selection changed after freeze')
    for path,digest in {**freeze['checkpoints'],**freeze['decoders']}.items():
        if sha256(path)!=digest:raise ValueError('frozen weights changed')
    manifest=json.loads((geometry/'geometry_manifest.json').read_text())
    source=Path(manifest['config']['capture'])
    captured=json.loads((source/'capture_manifest.json').read_text())
    capture_metadata=json.loads((source/'metadata.json').read_text())
    if datetime.fromisoformat(capture_metadata['started_at'])<=datetime.fromisoformat(freeze['frozen_at']):
        raise ValueError('independent capture preceded final selection freeze')
    if set(e['seed'] for e in captured['episodes'])!=set(freeze['independent_evaluation_seeds']):
        raise ValueError('final reserved seed identity')
    if any(f['split']!='evaluation' for f in manifest['frames']):
        raise ValueError('final runner accepts independent evaluation only')
    development=json.loads((training/'model_manifest.json').read_text())
    devgeom=Path(development['config']['geometry'])
    devcapture=Path(json.loads((devgeom/'geometry_manifest.json').read_text())['config']['capture'])
    old=json.loads((devcapture/'capture_manifest.json').read_text())
    oldgroups={e['actual_reset_group'] for e in old['episodes']}
    newgroups={e['actual_reset_group'] for e in captured['episodes']}
    if oldgroups & newgroups:raise ValueError('development/final actual reset group overlap')
    return manifest,dict(status='PASS',capture_started_at=capture_metadata['started_at'],
        selection_frozen_at=freeze['frozen_at'],development_groups=sorted(oldgroups),
        evaluation_groups=sorted(newgroups),weight_identity_exact=True,fit_or_selection_executed=False)


def extract(folder,data,checkpoint):
    model=GeometryRegressor(checkpoint['encoder']).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    identity=parameter_hash(model)
    if identity!=checkpoint['selected_parameter_hash']:raise ValueError('selected GNN identity')
    write_json(folder/'permutation_audit.json',permutation_audit(model.encoder,data.graphs[0]))
    outputs={k:[] for k in ('readout','object_slots','raw_geometry30','prediction36')}
    with torch.inference_mode():
        for begin in range(0,len(data.graphs),16):
            graph=pack_graphs(data.graphs[begin:begin+16]).to('cuda')
            z,_=model.encoder(graph)
            outputs['readout'].append(z.cpu().numpy())
            outputs['object_slots'].append(z[:,192:320].cpu().numpy())
            outputs['raw_geometry30'].append(geometry_summary(graph).cpu().numpy())
            outputs['prediction36'].append(model.decoder(z).cpu().numpy()*data.std+data.mean)
    if parameter_hash(model)!=identity:raise ValueError('frozen encoder changed during extraction')
    arrays={k:np.concatenate(v) for k,v in outputs.items()}
    arrays.update(target6=data.targets[:,:6],target36=data.targets,target_valid=data.valid,robot76=data.robot,
        seed=np.array([f['seed'] for f in data.frames]),timestep=np.array([f['timestep'] for f in data.frames]),
        split=np.array([f['split'] for f in data.frames]),train_fit_mask=data.train,
        condition=np.array([f.get('condition','baseline') for f in data.frames]),
        physical_valid=np.array([f.get('physical_valid',True) for f in data.frames],bool),
        validation_selection_mask=data.validation)
    if arrays['train_fit_mask'].any() or arrays['validation_selection_mask'].any():
        raise ValueError('final split contains fitting rows')
    save_npz(folder/'frozen_features.npz',**arrays)
    write_json(folder/'sample_manifest.json',data.identities)
    return arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training','probes','geometry','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for parent in (args.training,args.probes,args.geometry):
        if json.loads((parent/'metadata.json').read_text())['status']!='finished':raise ValueError('unfinished source')
    manifest,gate=final_gate(args.training,args.probes,args.geometry)
    models=json.loads((args.training/'model_manifest.json').read_text())['models']
    probes=json.loads((args.probes/'probe_manifest.json').read_text())['probes']
    protected=protection_snapshot(p for parent in (args.training,args.probes,args.geometry)
        for p in parent.rglob('*') if p.is_file())
    config=dict(task='09-C',stage='frozen_independent_evaluation',training=str(args.training),
        probes=str(args.probes),geometry=str(args.geometry),no_fit=True,no_selection=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_frozen_final',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_FINAL_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'independent_final_gate.json',gate)
        write_json(root/'protection_before.json',protected)
        feature_paths={};regressor=[]
        for entry in models:
            checkpoint=torch.load(args.training/entry['directory']/'selected.pt',map_location='cpu',weights_only=False)
            folder=root/entry['directory'];folder.mkdir(parents=True,exist_ok=False)
            data=Dataset(args.geometry,entry['variant'],manifest['frames'],
                target_statistics=(checkpoint['target_mean'],checkpoint['target_std']))
            arrays=extract(folder,data,checkpoint)
            feature_paths[str((args.training/entry['directory']).resolve())]=folder/'frozen_features.npz'
            for oid in (0,1):
                valid=arrays['physical_valid'] & (arrays['condition']!='baseline_repeat')
                regressor.append(dict(variant=entry['variant'],encoder=entry['encoder'],object_id=oid,
                    **absolute(arrays['target6'][valid,3*oid:3*oid+3],arrays['prediction36'][valid,3*oid:3*oid+3])))
            print('frozen final '+entry['variant']+'/'+entry['encoder'],flush=True)
        absolute_rows=[];episodes=[];pairs=[]
        for probe in probes:
            data=read_features(feature_paths[str(Path(probe['source_model_directory']).resolve())])
            oid=probe['object_id'];path=args.probes/probe['directory']/'decoder.npz'
            if probe['probe']=='Constant':
                with np.load(path,allow_pickle=False) as a:pred=np.broadcast_to(a['constant'],(len(data['seed']),3)).copy()
            elif probe['probe']=='RawCentroidDirect':pred=data['raw_geometry30'][:,oid*15:oid*15+3].astype(float)
            else:pred=decode_saved(path,probe_features(data,probe['probe'],oid).astype(float))
            target=data['target6'][:,oid*3:oid*3+3]
            identity={k:probe[k] for k in ('model','variant','encoder','probe','decoder','object_id')}
            aa,ee,pp=metric_rows(data,target,pred,identity)
            absolute_rows.extend(aa);episodes.extend(ee);pairs.extend(pp)
            folder=root/probe['directory'];folder.mkdir(parents=True,exist_ok=False)
            save_npz(folder/'predictions.npz',prediction=pred,target=target,seed=data['seed'],
                timestep=data['timestep'],split=data['split'],condition=data['condition'])
        csv_write(root/'geometry_regressor_metrics.csv',regressor)
        csv_write(root/'absolute_metrics.csv',absolute_rows);csv_write(root/'per_episode_metrics.csv',episodes)
        csv_write(root/'paired_delta_metrics.csv',pairs)
        audit=verify_protection(protected);write_json(root/'protection_audit.json',audit)
        if audit['status']!='PASS':raise ValueError('frozen final changed source artifact')
        for name in ('independent_final_gate.json','protection_audit.json','absolute_metrics.csv',
                     'per_episode_metrics.csv','paired_delta_metrics.csv','geometry_regressor_metrics.csv'):
            logger.link_output(root/name,kind='metrics')


if __name__=='__main__':main()

"""Exact logical graph work from stored node/edge counts and recorded batch RNG."""
import argparse,json
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());geometry=Path(state['completed']['development_geometry']);training=Path(state['completed']['training']);gm=json.loads((geometry/'geometry_manifest.json').read_text());models=json.loads((training/'model_manifest.json').read_text());rows=[];cache={}
    for entry in models['models']:
        v,e=entry['variant'],entry['encoder'];folder=training/entry['directory'];identity=json.loads((folder/'model_identity.json').read_text());config=identity['config']
        with np.load(folder/'frozen_features.npz',allow_pickle=False) as f:
            fit=np.flatnonzero(f['train_fit_mask']);validation=np.flatnonzero(f['validation_selection_mask'])
        if v not in cache:
            counts=[]
            for frame in gm['frames']:
                with np.load(geometry/frame['folder']/(v+'_graph.npz'),allow_pickle=False) as g:counts.append([len(g['node_input']),sum(g['node_type']>=3),g['edge_index'].shape[1]])
            cache[v]=np.array(counts,dtype=np.int64)
        n=cache[v];rng=np.random.default_rng(config['seed']);total=np.zeros(3,np.int64)
        for step in range(config['updates']):
            indices=rng.choice(fit,config['batch'],replace=len(fit)<config['batch']);total+=n[indices].sum(0)
        curve=json.loads((folder/'training_curve.json').read_text());assert curve[-1]['step']==2000 and config['updates']==2000
        rows.append(dict(variant=v,encoder=e,parameter_count=entry['parameter_count'],encoder_parameter_count=identity['encoder_parameter_count'],
            optimizer_updates=config['updates'],batch=config['batch'],training_graph_forwards=config['updates']*config['batch'],
            train_total_nodes=int(total[0]),train_total_object_points=int(total[1]),train_total_edges_packed=int(total[2]),
            train_message_edge_applications=int(total[2]*2) if e=='E0' else 0,
            mean_input_nodes=float(n[:,0].mean()),mean_input_object_points=float(n[:,1].mean()),mean_input_edges=float(n[:,2].mean()),
            validation_passes=len(curve),validation_nodes=int(n[validation,0].sum()*len(curve)),
            training_seconds=identity['training_seconds'],cuda_peak_allocated_bytes=identity['cuda_peak_allocated_bytes'],
            process_peak_rss_bytes=identity['process_peak_rss_bytes']))
    write_json(a.output/'compute_counts.json',dict(models=rows,method='stored graph dimensions + exact train_one default_rng(seed).choice replay; logical forward work, not a hardware profiler'))
    csv_write(a.output/'compute_counts.csv',rows)
if __name__=='__main__':main()

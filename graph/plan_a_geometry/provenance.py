"""Unique supporting-source export; never changes frozen geometry or graph rows."""
import argparse,json
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import save_npz,write_json
from graph.integration_baseline.geometry09b_probe import csv_write

def unique_sources(source,object_id):
    # Each node's first source is its actual measured representative. Auxiliary
    # support footprints can overlap after SF/SOR fusion near the3mm threshold.
    # Reserve every primary first; assign remaining support once by stable ID.
    entries={};primary={};node_primary={}
    for j,node in enumerate(source['node_row']):
        node=int(node);identity=tuple(str(source[k][j]) for k in ('episode','timestep','camera','pixel'))
        key=(int(object_id[node]),identity)
        entries.setdefault(key,[]).append((node,j))
        if node not in node_primary:
            node_primary[node]=key
            if key in primary and primary[key]!=node:raise ValueError('duplicate primary physical source across geometry nodes')
            primary[key]=node
    keep=[]
    for key,options in entries.items():
        owner=primary.get(key,min((node for node,j in options),key=lambda n:node_primary[n]))
        keep.append(next(j for node,j in options if node==owner))
    keep=sorted(keep)
    result={k:v[keep] for k,v in source.items() if k!='structural'}
    result['structural']=source['structural'].copy()
    return result,len(source['node_row'])-len(keep)

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());rows=[];count=0
    for stage in ('development_geometry','final_geometry'):
        root=Path(state['completed'][stage]);manifest=json.loads((root/'geometry_manifest.json').read_text())
        for f in manifest['frames']:
            folder=root/f['folder']
            with np.load(f['gt_path'],allow_pickle=False) as z:gt={k:z[k] for k in z.files}
            for path in list(folder.glob('C*_native.npz'))+list(folder.glob('*_points.npz')):
                with np.load(path,allow_pickle=False) as z:point={k:z[k] for k in z.files}
                with np.load(path.with_name(path.stem+'_sources.npz'),allow_pickle=False) as z:source={k:z[k] for k in z.files}
                unique,repeated=unique_sources(source,point['object_id']);out=a.output/stage/f['folder']/path.with_name(path.stem+'_canonical_sources.npz').name;out.parent.mkdir(parents=True,exist_ok=True);save_npz(out,**unique)
                for oid in (0,1):
                    selected=np.array([point['object_id'][i]==oid for i in unique['node_row']],bool)
                    labels=[int(gt[str(c)+'/segmentation'].ravel()[int(p)])==int(gt['actor_ids'][oid]) for c,p in zip(unique['camera'][selected],unique['pixel'][selected])]
                    rows.append(dict(stage=stage,seed=f['seed'],split=f['split'],timestep=f['timestep'],condition=f['condition'],point_set=path.stem,object_id=oid,
                        nodes=int((point['object_id']==oid).sum()),unique_source_observations=len(labels),unique_support_object_purity=float(np.mean(labels)) if labels else None,
                        repeated_support_entries_in_point_set=repeated))
                count+=repeated
    csv_write(a.output/'unique_source_purity.csv',rows)
    write_json(a.output/'canonicalization_audit.json',dict(status='PASS',repeated_support_entries_removed=count,duplicate_primary_source_conflicts=0,point_geometry_changed=False,graph_tensors_changed=False,model_weights_changed=False,
        reason='SF/SOR supporting-source lists can repeat or share an auxiliary pixel near the3mm merge threshold. Reserve each real representative first and assign remaining physical support once by stable primary source ID. Raw traces, points, graphs and model inputs remain unchanged'))
if __name__=='__main__':main()

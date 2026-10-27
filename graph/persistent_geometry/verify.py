"""Independent readback and joins for finalized Task09-C diagnostic runs."""
import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from graph.integration_baseline.io import write_json,sha256
from utils.experiment_logger import ExperimentLogger
from .audit import protection_snapshot,verify_protection
from .collect import inventory_sources


def graph_contract(a,path):
    nodes=a['node_input'];edges=a['edge_index'];types=a['node_type']
    if nodes.ndim!=2 or nodes.shape[1] not in (14,21):raise ValueError('graph feature dimension '+str(path))
    if not np.isfinite(nodes).all() or not np.isfinite(a['edge_attr']).all():raise ValueError('nonfinite graph '+str(path))
    if len(types)!=len(nodes) or not np.array_equal(types[:3],[0,1,2]):raise ValueError('robot/object type ordering')
    if edges.shape!=(2,len(a['edge_type'])) or len(a['edge_attr'])!=edges.shape[1]:raise ValueError('edge tensor join')
    if edges.size and (edges.min()<0 or edges.max()>=len(nodes)):raise ValueError('edge endpoint outside graph')


def source_snapshot(root,*,required=True):
    path=root/'source_hashes.json'
    if not path.exists():return dict(status='NOT_AVAILABLE')
    mapping=json.loads(path.read_text());missing=[];changed=[]
    for name,digest in mapping.items():
        saved=root/'source_snapshot'/name
        if not saved.exists():missing.append(name)
        elif sha256(saved)!=digest:changed.append(name)
    if changed:raise ValueError('source snapshot readback '+str(root))
    if missing:
        if required:raise ValueError('missing required source snapshot '+str(root))
        return dict(status='NOT_AVAILABLE',missing_files=len(missing),
                    note='legacy pilot; no claim of an execution-time source snapshot')
    return dict(status='PASS',files=len(mapping))


def relocate(path,path_map=None):
    path=Path(path)
    for old,new in sorted((path_map or {}).items(),key=lambda item:len(item[0]),reverse=True):
        try:relative=path.relative_to(old)
        except ValueError:continue
        return Path(new)/relative
    return path


def verify_run(root,*,deep=True,path_map=None,check_joins=True):
    metadata=json.loads((root/'metadata.json').read_text())
    if metadata['status']!='finished':raise ValueError('readback requires finalized run '+str(root))
    if metadata['run_id']!=root.name:raise ValueError('UUID/directory identity')
    counts=dict(json=0,npz=0,csv=0,graph=0,png=0,mp4=0,checkpoint=0)
    for path in sorted(root.rglob('*')):
        if not path.is_file() or 'source_snapshot' in path.parts:continue
        if path.suffix=='.json':json.loads(path.read_text());counts['json']+=1
        elif path.suffix=='.csv':
            with path.open(newline='') as f:
                reader=csv.DictReader(f)
                for row in reader:
                    if None in row:raise ValueError('CSV field-count mismatch '+str(path))
            counts['csv']+=1
        elif path.suffix=='.npz' and deep:
            with np.load(path,allow_pickle=False) as a:
                for key in a.files:a[key]
                if path.name.endswith('_graph.npz') or path.name=='graph.npz':graph_contract(a,path);counts['graph']+=1
                if path.name=='evaluation_targets.npz':
                    if a['target36'].shape!=(36,) or a['target_valid'].shape!=(36,) or a['robot76'].shape!=(76,):raise ValueError('target tensor shape')
                    if not np.isfinite(a['target36']).all():raise ValueError('nonfinite targets')
                if path.name=='frozen_features.npz':
                    for key in ('readout','object_slots','raw_geometry30','prediction36','target36','robot76'):
                        if not np.isfinite(a[key]).all() or len(a[key])!=len(a['seed']):raise ValueError('model feature finite/row join')
                    if np.any(a['train_fit_mask']&a['validation_selection_mask']):raise ValueError('fitting split overlap')
                    sample=json.loads((path.parent/'sample_manifest.json').read_text())
                    if len(sample)!=len(a['seed']):raise ValueError('sample-manifest row count')
                    for i,entry in enumerate(sample):
                        if entry['seed']!=int(a['seed'][i]) or entry['timestep']!=int(a['timestep'][i]) or entry['split']!=str(a['split'][i]) or entry['condition']!=str(a['condition'][i]):raise ValueError('model/map episode-time-condition join')
            counts['npz']+=1
        elif path.suffix=='.pt' and deep:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint.get('schema')!='task09c_geometry_training_v1':raise ValueError('unexpected new checkpoint schema')
            if any(not torch.isfinite(v).all() for v in checkpoint['state_dict'].values()):raise ValueError('nonfinite checkpoint')
            if checkpoint['actor_position_dimensions']!=6 or checkpoint['surface_summary_dimensions']!=30 or not checkpoint['no_policy_or_gru']:raise ValueError('diagnostic checkpoint scope')
            counts['checkpoint']+=1
        elif path.suffix=='.png':
            with Image.open(path) as image:image.verify()
            counts['png']+=1
        elif path.suffix=='.mp4':
            reader=imageio.get_reader(path);n=reader.count_frames()
            first=reader.get_data(0);last=reader.get_data(n-1);reader.close()
            if n<2 or first.shape!=last.shape:raise ValueError('temporal MP4 decode')
            counts['mp4']+=1
    if check_joins and (root/'geometry_manifest.json').exists():
        manifest=json.loads((root/'geometry_manifest.json').read_text())
        unique=set()
        for frame in manifest['frames']:
            identity=(frame['seed'],frame['timestep'],frame.get('condition','baseline'))
            if identity in unique:raise ValueError('duplicate physical frame condition')
            unique.add(identity)
            folder=relocate(root/frame['folder'],path_map)
            with np.load(folder/'evaluation_targets.npz',allow_pickle=False) as a:
                if int(a['timestep'])!=frame['timestep']:raise ValueError('map/target timestep mismatch')
            for name in ('G0','G1','G2','G3','G4','O1','O2','O3'):
                with np.load(folder/(name+'_map.npz'),allow_pickle=False) as a:
                    if np.any(a['first_seen_timestep']>frame['timestep']) or np.any(a['last_seen_timestep']>frame['timestep']):raise ValueError('map future lifecycle')
                    tracker=folder/(name+'_tracker.json')
                    if tracker.exists():
                        state=json.loads(tracker.read_text())
                        if state['timestep']!=frame['timestep'] or state.get('geometry_variant',name)!=name:raise ValueError('map/tracker variant/time identity')
                        for oid in (0,1):
                            selected=a['object_id']==oid;matrix=np.array(state['objects'][str(oid)]['world_from_object'])
                            world=a['object_local_xyz'][selected]@matrix[:3,:3].T+matrix[:3,3]
                            np.testing.assert_allclose(world,a['world_xyz'][selected],atol=1e-10,rtol=0)
                provenance=folder/(name+'_point_provenance.json')
                if provenance.exists():
                    for node,sources in json.loads(provenance.read_text()).items():
                        if any(int(source[1])>frame['timestep'] for source in sources):raise ValueError('branch contains future provenance')
                        for episode,t,camera,pixel in sources:
                            if not camera or int(pixel)<0:raise ValueError('invalid camera/pixel provenance')
    for name in ('protection_audit.json','frozen_gnn_protection_audit.json','historical_frozen_identity.json'):
        path=root/name
        if path.exists() and json.loads(path.read_text())['status']!='PASS':raise ValueError('recorded protection failure')
    return dict(run_id=root.name,status='PASS',counts=counts,
        join_scope='final-schema frame/model joins' if check_joins else 'legacy file-format readback only',
        source_snapshot=source_snapshot(root,required=check_joins))


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(value for name,value in attrs if name in ('href','src') and value)


def verify_links(path):
    if path.suffix=='.html':
        parser=Links();parser.feed(path.read_text());links=parser.links
    else:links=re.findall(r'\]\(([^)]+)\)',path.read_text())
    checked=[]
    for link in links:
        target=urlsplit(link.strip('<>'))
        if target.scheme or not target.path:continue
        actual=path.parent/unquote(target.path)
        if not actual.exists():raise ValueError('broken relative artifact link '+str(actual))
        checked.append(link)
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protected=protection_snapshot(p for root in args.runs for p in root.rglob('*') if p.is_file())
    with ExperimentLogger(team='integration',config=dict(task='09-C',stage='independent_readback',runs=list(map(str,args.runs))),
                          seed=42,model='task09c_artifact_readback',job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_READBACK_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'protection_before.json',protected)
        results=[]
        for source in args.runs:
            results.append(verify_run(source));print('READBACK_PASS '+str(source),flush=True)
        after=verify_protection(protected)
        if after['status']!='PASS':raise ValueError('readback modified inputs')
        write_json(root/'verification.json',dict(status='PASS',runs=results,protection=after))
        logger.link_output(root/'verification.json',kind='metrics')


if __name__=='__main__':main()

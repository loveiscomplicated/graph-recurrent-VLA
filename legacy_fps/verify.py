"""Strict artifact, tensor/provenance and original-workspace protection audit."""
import argparse,csv,hashlib,json,os,subprocess,time
from html.parser import HTMLParser
from pathlib import Path
import cv2
import numpy as np
from graph.integration_baseline.io import sha256,write_json
from graph.persistent_geometry.evaluate import load_cameras
from .benchmark import BASE

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value and not value.startswith(('http:','https:','data:','#')):self.links.append(value.split('#')[0])

def protection(root):
    before=json.loads((BASE/'protection_before.json').read_text());unchanged=0;changed=[];missing=[]
    for name,identity in before['stable'].items():
        p=Path(name)
        if not p.exists():missing.append(name)
        elif p.stat().st_size==identity['size'] and sha256(p)==identity['sha256']:unchanged+=1
        else:changed.append(name)
    processes=[]
    for old in before['processes']:
        p=Path('/proc')/str(old['pid'])
        try:
            start=(p/'stat').read_text().split(') ')[1].split()[19]
            same=start==old['starttime'];state=(p/'stat').read_text().split(') ')[1].split()[0]
            processes.append(dict(pid=old['pid'],same_process=same,state=state,naturally_completed=False))
        except OSError:processes.append(dict(pid=old['pid'],same_process=False,naturally_completed='not inferred; inspect parent pipeline audit'))
    result=dict(status='PASS' if not changed and not missing else 'FAIL',stable_files_unchanged=unchanged,changed=changed,missing=missing,
        process_observations=processes,active_outputs_policy='expected to grow naturally; inventoried pre-state, not immutable',
        extension_actions='only own task directory written; no signal/control action sent to any existing process',post_at=time.time())
    write_json(root/'protection_audit.json',result)
    return result

def pinned_source_readback(root):
    original=json.loads((root/'dependency_source_hashes.json').read_text());checked=0;changed=[]
    for relative,digest in original.items():
        file=BASE/'source'/relative
        if not file.exists() or sha256(file)!=digest:changed.append(relative)
        checked+=1
    result=dict(status='FAIL' if changed else 'PASS',checked_files=checked,changed=changed,
        scope='independent frozen dependency copy used by this extension, distinct from concurrently changing original source')
    if changed:raise ValueError('extension frozen dependency snapshot changed')
    return result

def check(root):
    counts=dict(json=0,csv=0,csv_rows=0,npz=0,png=0,mp4=0,mp4_decoded_frames=0,html=0,links=0,tensor_graphs=0,source_rows=0,provenance_nodes=0)
    digests={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in ('verification.json','artifact_hashes.json'):continue
        suffix=p.suffix.lower()
        if suffix=='.json' or p.name=='config.yaml':
            json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x)));counts['json']+=1
            if p.name.endswith('_provenance.json'):
                t=int(p.parent.name[1:])
                for row in json.loads(p.read_text()):
                    if any(int(s[1])>t for s in row['source_observation_ids']):raise ValueError('future point source')
                    counts['provenance_nodes']+=1
        elif suffix=='.csv':
            with p.open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames
                if not fields:raise ValueError('empty CSV header')
                for row in reader:
                    if None in row:raise ValueError('CSV field mismatch')
                    counts['csv_rows']+=1
            counts['csv']+=1
        elif suffix=='.npz':
            with np.load(p,allow_pickle=False) as a:
                arrays={k:a[k] for k in a.files}
                for value in arrays.values():value.tobytes()
            counts['npz']+=1
            if 'node_input' in arrays:
                n=len(arrays['node_input']);edge=arrays['edge_index']
                if edge.shape[0]!=2 or (edge<0).any() or (edge>=n).any():raise ValueError('invalid graph endpoints')
                if len(arrays['edge_attr'])!=edge.shape[1] or len(arrays['edge_type'])!=edge.shape[1]:raise ValueError('edge row join')
                if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('nonfinite graph')
                if p.name=='L0_graph.npz' and edge.shape[1]:raise ValueError('L0 must be no-edge')
                counts['tensor_graphs']+=1
            if 'source_pixel_id' in arrays:
                source=arrays['source_pixel_id'];counts['source_rows']+=len(source)
                if len(np.unique(source))!=len(source):raise ValueError('duplicated source pixel')
                if not set(source).issubset(set(arrays['sor_source'])):raise ValueError('source selection join')
                stem=p.parent.name;seed=p.parent.parent.name;capture=Path(json.loads((root/'geometry_manifest.json').read_text())['config']['capture'])
                if 'conditions' in p.parts:
                    condition=p.parts[p.parts.index('conditions')+1]
                    capture=root/'branch_inputs'/condition
                sourcepath=capture/'collection'/seed/(stem+'_input.npz')
                if sourcepath.exists():
                    cameras,_=load_cameras(sourcepath)
                    camera_name=p.name[len('fps_'):].split('_object')[0].replace('_unfiltered','')
                    cam=next(c for c in cameras if c.camera==camera_name)
                    v,u=np.divmod(source,cam.depth.shape[1]);z=cam.depth.astype(np.float32)[v,u];K=cam.intrinsic
                    xyz=np.c_[(u+.5-K[0,2])*z/K[0,0],(v+.5-K[1,2])*z/K[1,1],z].astype(np.float32)
                    np.testing.assert_array_equal(xyz,arrays['point_xyz_camera'])
                    world=(xyz@cam.world_from_camera[:3,:3].T+cam.world_from_camera[:3,3]).astype(np.float32)
                    np.testing.assert_array_equal(world,arrays['point_xyz_world'])
        elif suffix=='.png':
            im=cv2.imread(str(p))
            if im is None:raise ValueError('PNG decode failed')
            counts['png']+=1
        elif suffix=='.mp4':
            cap=cv2.VideoCapture(str(p));n=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                n+=1
            cap.release()
            if n==0:raise ValueError('empty MP4')
            counts['mp4']+=1;counts['mp4_decoded_frames']+=n
        elif suffix=='.html':
            parser=Links();parser.feed(p.read_text())
            for link in parser.links:
                if not (p.parent/link).exists():raise ValueError('broken HTML relative link: '+link)
            counts['links']+=len(parser.links);counts['html']+=1
        digests[str(p.relative_to(root))]=sha256(p)
    write_json(root/'artifact_hashes.json',digests)
    return counts

def semantic_readback(root):
    from .adapter import frame_points
    from graph.persistent_geometry.fusion import filtered_masks
    from visual.runtime.vision_packet import pose_matrix
    manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture'])
    rows=0;features=0;historical=0;same_point_edges=0
    for frame in manifest['frames']:
        folder=root/frame['folder'];src=capture/'collection'/f'seed{frame["seed"]}'
        cameras,data=load_cameras(src/f'f{frame["timestep"]:04d}_input.npz')
        main=json.loads((src/'sensor_contract.json').read_text())['roles']['main'];camera=next(c for c in cameras if c.camera==main)
        with np.load(src/f'f{frame["timestep"]:04d}_masks.npz',allow_pickle=False) as a:masks=a[main]
        with np.load(folder/'robot_filter.npz',allow_pickle=False) as a:filter_output={k.split('/',1)[1]:a[k] for k in a.files if k.startswith(main+'/')}
        clean,_=filtered_masks(masks,filter_output)
        for name,mask in (('L0',masks),('L1',clean)):
            obs,_,_=frame_points(camera,mask,f5=True)
            with np.load(folder/(name+'_map.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(a['world_xyz'],np.concatenate([o.xyz for o in obs]))
                np.testing.assert_array_equal(a['last_f5'],np.concatenate([o.f5 for o in obs]))
                np.testing.assert_array_equal(a['last_feature_valid'],np.concatenate([o.feature_valid for o in obs]))
                features+=len(a['world_xyz'])
        with np.load(folder/'L0_graph.npz') as base:
            for name in ('L0_surface','L0_rawede'):
                path=folder/(name+'_graph.npz')
                if not path.exists():continue
                with np.load(path) as other:
                    for key in ('node_input','node_type'):np.testing.assert_array_equal(base[key],other[key])
                same_point_edges+=1
        for name in ('L2','L4','G2','G4'):
            graph=folder/(name+'_graph.npz');meta=folder/(name+'_graph.json');mapfile=folder/(name+'_map.npz')
            if not graph.exists() or not meta.exists():continue
            node_ids=json.loads(meta.read_text()).get('node_ids')
            if node_ids is None:continue
            with np.load(mapfile) as a:lookup={str(s):bool(h) for s,h in zip(a['persistent_node_id'],a['estimated_only'])}
            is_old=np.array([lookup[s] for s in node_ids],bool)
            with np.load(graph) as g:
                node=g['node_input'][3:]
                if (node[is_old][:,[6,7,9,10,12,13]]!=0).any():raise ValueError('historical jump/variation reused as current feature')
            historical+=int(is_old.sum())
        rows+=1
    return dict(status='PASS',frames=rows,FPS_source_feature_rows=features,historical_feature_rows=historical,same_point_edge_controls=same_point_edges)

def metric_readback(root):
    from scipy.spatial import cKDTree
    from graph.persistent_geometry.metrics import sample_meshes
    from graph.persistent_geometry.contracts import transform
    from visual.runtime.vision_packet import pose_matrix
    manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture'])
    index={(f['seed'],f['timestep']):root/f['folder'] for f in manifest['frames']};cache={};meshes={};gt_cache={};checked=0
    for row in csv.DictReader((root/'geometry_metrics.csv').open()):
        seed,t=int(row['seed']),int(row['timestep']);name=row['variant'];mode=row['budget_mode'];oid=int(row['object_id'])
        folder=index[(seed,t)]
        if seed not in meshes:meshes[seed]=sample_meshes(capture/'collection'/f'seed{seed}'/'evaluation_meshes.npz')
        if (seed,t) not in gt_cache:
            with np.load(capture/'collection'/f'seed{seed}'/f'f{t:04d}_gt.npz') as a:gt_cache[(seed,t)]=a['actor_pose']
        key=(seed,t,name,mode)
        if key not in cache:
            with np.load(folder/(name+('_matched_map.npz' if mode=='matched_native_minN_posthoc' else '_map.npz'))) as a:arr={k:a[k] for k in a.files}
            if mode=='cap128_per_object':
                meta=json.loads((folder/(name+'_graph.json')).read_text())
                if name=='G0':selection=np.array([i-3 for i in meta['selected_original_rows'] if i>=3],int)
                else:
                    lookup={s:i for i,s in enumerate(arr['persistent_node_id'])};selection=np.array([lookup[s] for s in meta['node_ids']],int)
                arr={k:v[selection] for k,v in arr.items()}
            cache[key]=arr
        arr=cache[key];selected=(arr['object_id']==oid)&(arr['status']!=3);world=arr['world_xyz'][selected]
        if len(world)!=int(row['count']):raise ValueError('geometry CSV point count mismatch')
        local=transform(world,np.linalg.inv(pose_matrix(gt_cache[(seed,t)][oid])))
        distance=meshes[seed][oid]['tree'].query(local)[0]*1000
        if len(distance):
            for k,actual in (('surface_distance_mean_mm',distance.mean()),('surface_distance_p95_mm',np.percentile(distance,95)),('surface_distance_median_mm',np.median(distance))):
                if not np.isclose(float(row[k]),actual,atol=1e-9,rtol=1e-10):raise ValueError('geometry CSV numeric mismatch: '+k)
        elif row['surface_distance_mean_mm']!='':raise ValueError('invented empty geometry distance')
        checked+=1
    return dict(status='PASS',geometry_rows_recomputed=checked,reference='actual stored map/selected rows and actual GT-sidecar collision union samples')

def learning_readback(root):
    models=json.loads((root/'model_manifest.json').read_text())['models'];features={};checkpoint_hashes={};identities=None
    for model in models:
        folder=root/model['directory'];meta=json.loads((folder/'model_identity.json').read_text())
        checkpoint_hashes[(model['variant'],model['encoder'])]=meta['selected_parameter_hash']
        if sha256(folder/'selected.pt')!=model['checkpoint_sha256'] or sha256(folder/'frozen_features.npz')!=model['features_sha256']:raise ValueError('frozen GNN input/checkpoint hash mismatch')
        with np.load(folder/'frozen_features.npz') as a:
            current={k:a[k] for k in ('target36','target_valid','seed','timestep','split','condition','train_fit_mask','validation_selection_mask')}
            if identities is None:identities=current
            else:
                for k,value in current.items():np.testing.assert_array_equal(value,identities[k])
            if model['encoder']=='E1' and model['variant'] in ('L0','L0_surface','L0_rawede'):features[model['variant']]=a['readout']
        audit=json.loads((folder/'permutation_audit.json').read_text())
        if audit['status']!='PASS' or audit['device']!='cuda':raise ValueError('actual CUDA permutation audit missing')
    if len(features)==3:
        for name in ('L0_surface','L0_rawede'):
            np.testing.assert_array_equal(features['L0'],features[name])
            if checkpoint_hashes[('L0','E1')]!=checkpoint_hashes[(name,'E1')]:raise ValueError('E1 changed with edge-only control')
    train=set(identities['seed'][identities['train_fit_mask']]);val=set(identities['seed'][identities['validation_selection_mask']])
    if train&val:raise ValueError('GNN/probe split overlap')
    return dict(status='PASS',same_targets_splits_budget_rows_models=len(models),CUDA_permutation_models=len(models),
        same_point_E1_edge_controls_exact=True,train_seeds=sorted(int(x) for x in train),validation_seeds=sorted(int(x) for x in val))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--additional',type=Path,nargs='*',default=[]);args=parser.parse_args()
    protected=protection(args.run);pinned=pinned_source_readback(args.run);result={}
    semantic=semantic_readback(args.run);recomputed=metric_readback(args.run);learning=[]
    for root in [args.run,*args.additional]:
        result[str(root)]=check(root)
        if (root/'model_manifest.json').exists():learning.append(learning_readback(root))
    references=[]
    for root in args.additional:
        path=root/'reference_source_identity.json'
        if path.exists():
            identities=json.loads(path.read_text())
            for filename,digest in identities.items():
                if sha256(filename)!=digest:raise ValueError('completed reference artifact changed')
            references.append(dict(run=str(root),stable_reference_files=len(identities),status='PASS'))
    write_json(args.run/'verification.json',dict(status='PASS' if protected['status']=='PASS' else 'PARTIAL',
        artifact_readback_status='PASS',checked_at=time.time(),readback=result,protection=protected,pinned_dependency_source=pinned,
        semantic_tensor_identity=semantic,completed_reference_source_identity=references,
        geometry_numeric_readback=recomputed,learning_tensor_identity=learning,
        no_simulator_execution_claim=True,devices='CPU saved observations; CUDA diagnostic only if learning metadata finished',
        visual_inspection='contact sheet reviewed by agent separately; see report'))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

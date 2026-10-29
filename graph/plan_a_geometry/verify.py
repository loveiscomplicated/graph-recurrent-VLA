"""Readback and source/graph/weight/split protection checks, no model fitting."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json,sha256
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix


def verify_geometry(root):
    manifest=json.loads((root/'geometry_manifest.json').read_text());checks=[];framekeys=set();source_cache={}
    for frame in manifest['frames']:
        identity=(frame['seed'],frame['timestep'],frame['condition'])
        if identity in framekeys:raise ValueError('duplicate geometry frame')
        framekeys.add(identity);folder=root/frame['folder']
        cameras,data=load_cameras(frame['input_path']);by={c.camera:c for c in cameras};worlds={c.camera:c.world_points().reshape(-1,3) for c in cameras}
        if sha256(frame['input_path'])!=frame['input_sha256']:raise ValueError('source changed')
        if sha256(frame['masks_path'])!=frame['masks_sha256']:raise ValueError('mask changed')
        if sha256(frame['gt_path'])!=frame['gt_sha256']:raise ValueError('GT changed')
        for path in list(folder.glob('*_points.npz'))+list(folder.glob('C*_native.npz')):
            with np.load(path,allow_pickle=False) as a:points={k:a[k].copy() for k in a.files}
            with np.load(path.with_name(path.stem+'_sources.npz'),allow_pickle=False) as a:src={k:a[k].copy() for k in a.files}
            for i in range(len(points['world_xyz'])):
                j=np.flatnonzero(src['node_row']==i)[0];camera=by[str(src['camera'][j])];pixel=int(src['pixel'][j])
                if str(src['episode'][j])!=camera.episode or int(src['timestep'][j])!=camera.timestep:raise ValueError('source identity mismatch')
                xyz=worlds[camera.camera][pixel]
                np.testing.assert_allclose(points['world_xyz'][i],xyz,atol=3e-7,rtol=1e-6)
                if points['object_id'][i]==1 and path.name.startswith(('C2','C3')):
                    if np.any(src['camera'][src['node_row']==i]!='base_camera'):raise ValueError('Wrist Box leakage')
            if points['estimated_only'].any() or not points['currently_observed'].all():raise ValueError('noncurrent geometry')
        def load(p):
            with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
        sf=load(folder/'C3_SF_native.npz');sfs=load(folder/'C3_SF_native_sources.npz')
        hp=load(folder/'C3_H_native.npz');hs=load(folder/'C3_H_native_sources.npz')
        cap=load(folder/'C3_H_cap128_points.npz');cs=load(folder/'C3_H_cap128_points_sources.npz')
        for oid in (0,1):
            structural=set(sf['persistent_node_id'][(sf['object_id']==oid)&sfs['structural']])
            hybrid=set(hp['persistent_node_id'][hp['object_id']==oid])
            if len(structural)<=1024:assert structural<=hybrid
            hstruct=set(hp['persistent_node_id'][(hp['object_id']==oid)&hs['structural']])
            capids=set(cap['persistent_node_id'][cap['object_id']==oid])
            if len(hstruct)<=128:assert hstruct<=capids
            else:assert cs['structural'][cap['object_id']==oid].all()
        sor={}
        for name in by:
            for oid in (0,1):sor[name,oid]=set(load(folder/f'legacy_{name}_object{oid}_stages.npz')['sor_source'])
        for i,oid in enumerate(hp['object_id']):
            if hs['structural'][i]:continue
            j=np.flatnonzero(hs['node_row']==i)[0]
            assert int(hs['pixel'][j]) in sor[str(hs['camera'][j]),int(oid)]
        for path in folder.glob('*_graph.npz'):
            with np.load(path,allow_pickle=False) as a:g={k:a[k].copy() for k in a.files}
            n=len(g['node_input']);e=g['edge_index'];meta=json.loads(path.with_suffix('.json').read_text())
            assert g['node_input'].shape==(n,21) and not g['node_input'][:,14:].any()
            assert e.shape[0]==2 and g['edge_attr'].shape==(e.shape[1],4) and g['edge_type'].shape==(e.shape[1],)
            assert (e>=0).all() and (e<n).all()
            delta=g['node_input'][e[1],:3].astype(float)-g['node_input'][e[0],:3].astype(float)
            distance=np.linalg.norm(delta,axis=1)
            direction=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>0)
            np.testing.assert_allclose(g['edge_attr'][:,0],distance,atol=1e-7)
            np.testing.assert_allclose(g['edge_attr'][:,1:],direction,atol=1e-6)
            variant=path.name.split('_matched')[0].split('_graph')[0];mode='matched_N' if '_matched_N_' in path.name else 'cap128'
            p=folder/(variant+'_'+mode+'_points.npz')
            with np.load(p,allow_pickle=False) as a:points={k:a[k] for k in a.files}
            lookup={s:i for i,s in enumerate(points['persistent_node_id'])};indices=[lookup[s] for s in meta['node_ids']]
            expected=transform(points['world_xyz'][indices],np.linalg.inv(pose_matrix(data['tcp_pose']))).astype('float32')
            np.testing.assert_allclose(g['node_input'][3:,:3],expected,atol=1e-7)
            for oid in (0,1):assert sum(g['node_type']==oid+3)<=128
            checks.append(dict(seed=frame['seed'],timestep=frame['timestep'],condition=frame['condition'],variant=variant,budget=mode,nodes=n,edges=e.shape[1]))
    return dict(status='PASS',frames=len(manifest['frames']),graphs=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=json.loads(a.state.read_text());runs={k:Path(v) for k,v in state['completed'].items() if k!='runtime'};result={}
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('development_geometry','final_geometry'):result[name]=verify_geometry(runs[name])
    groups={}
    for name in ('development_geometry','final_geometry'):
        m=json.loads((runs[name]/'geometry_manifest.json').read_text())
        for ep in m['episodes']:
            if ep['actual_reset_group'] in groups and groups[ep['actual_reset_group']]!=ep['split']:raise ValueError('reset split leakage')
            groups[ep['actual_reset_group']]=ep['split']
    count=dict(json=0,csv=0,npz=0)
    for root in runs.values():
        for path in root.rglob('*'):
            if not path.is_file():continue
            if path.suffix=='.json':json.loads(path.read_text());count['json']+=1
            elif path.suffix=='.csv':
                with path.open(newline='') as f:list(csv.DictReader(f))
                count['csv']+=1
            elif path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as arrays:
                    for k in arrays.files:np.asarray(arrays[k]).tobytes()
                count['npz']+=1
    before=json.loads((Path('/workspace/task09d_plan_a/state.json')).read_text());preflight=Path(before['preflight'])
    protected=json.loads((preflight/'protection_before.json').read_text());changed=[];missing=[]
    for filename,entry in protected.items():
        path=Path(filename)
        if not path.is_file():missing.append(filename)
        elif path.stat().st_size!=entry['size'] or sha256(path)!=entry['sha256']:changed.append(filename)
    result['protection']=dict(status='PASS' if not changed and not missing else 'FAIL',checked_files=len(protected),changed=changed,missing=missing)
    result.update(readback=count,split_groups=len(groups),status='PASS' if not changed and not missing else 'FAIL')
    write_json(a.output/'verification.json',result)
    assert result['status']=='PASS'
if __name__=='__main__':main()

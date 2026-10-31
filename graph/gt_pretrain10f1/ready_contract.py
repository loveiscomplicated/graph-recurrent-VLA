"""Sealed producer contract. Standalone source can run read-only on Task10-F.

No generated marker is trusted without actual file/GT/graph/approval readback.
This module uses no simulator environment and cannot control any process.
"""
import hashlib
import json
from pathlib import Path
import time


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def safe_file(root,relative):
    root=Path(root).resolve();p=Path(relative)
    if p.is_absolute() or not p.parts or any(s in ('..','') for s in p.parts):
        raise ValueError('unsafe producer relative path')
    result=root/p
    for parent in (result,*result.parents):
        if parent==root:break
        if parent.is_symlink():raise ValueError('symlink in sealed export')
    if not result.resolve().is_relative_to(root) or not result.is_file():
        raise ValueError('missing/nonfile sealed export')
    return result


def read_json(path):return json.loads(Path(path).read_text())


def probe(root):
    """Cheap normal-progress polling; no HDF5/model/CUDA/CLI invocation."""
    root=Path(root);marker=root/'gnn_dataset_ready.json'
    if not marker.exists():
        return dict(status='WAITING_DATA',reason='sealed gnn_dataset_ready.json absent',marker_exists=False)
    value=read_json(marker)
    return dict(status=value.get('status','NOT_READY'),marker_exists=True,marker_sha256=sha_file(marker),
        dataset_id=value.get('dataset_id'),source_instance=value.get('source_instance'))


def validate(root,expected_encoder_sha=None,*,marker_path=None):
    """All-file hash/readback and actual semantic gates; returns JSON-only data."""
    import numpy as np
    import h5py
    from recurrent.features import build_features
    root=Path(root).resolve()
    marker=root/'gnn_dataset_ready.json' if marker_path is None else Path(marker_path).resolve()
    if not marker.is_relative_to(root):raise ValueError('candidate marker outside producer root')
    m=read_json(marker)
    required=dict(schema_version=1,status='DATA_READY',task='10-F',source_instance=57640710,
        collection_complete=True,graph_cache_readback='PASS',physical_validity='PASS',gt38_alignment='PASS',
        parent_branch_provenance='PASS',files_sealed=True)
    for k,v in required.items():
        if m.get(k)!=v:raise ValueError('producer readiness gate '+k)
    if not isinstance(m.get('dataset_id'),str) or not m['dataset_id']:raise ValueError('dataset identity required')
    files={};before={}
    for row in m.get('files',[]):
        path=safe_file(root,row['path'])
        if row['path'] in files:raise ValueError('duplicate sealed file')
        stat=path.stat();before[row['path']]=(stat.st_size,stat.st_mtime_ns)
        if stat.st_size!=row['bytes'] or len(row['sha256'])!=64 or sha_file(path)!=row['sha256']:
            raise ValueError('sealed SHA/size mismatch '+row['path'])
        files[row['path']]=row
    if not files:raise ValueError('empty sealed file list')
    def declared(relative):
        if relative not in files:raise ValueError('referenced file not sealed: '+relative)
        return safe_file(root,relative)
    mp=declared(m['dataset_manifest']);rp=declared(m['runtime_contract'])
    if sha_file(mp)!=m['dataset_manifest_sha256'] or sha_file(rp)!=m['runtime_contract_sha256']:
        raise ValueError('manifest/runtime identity mismatch')
    runtime=read_json(rp);dataset=read_json(mp)
    for key in ('simulator','tensor_runtime','camera_contract','robot_schema','geometry','geometry_protocol_sha256',
                'vision_weight_sha256','source_sha256','gt38_encoder_sha256','quaternion_order'):
        if key not in runtime:raise ValueError('missing runtime provenance '+key)
    if runtime['quaternion_order']!='wxyz':raise ValueError('quaternion order')
    if expected_encoder_sha and runtime['gt38_encoder_sha256']!=expected_encoder_sha:
        raise ValueError('official GT38 encoder SHA mismatch')
    import inspect
    if sha_file(inspect.getfile(build_features))!=runtime['gt38_encoder_sha256']:
        raise ValueError('validator encoder differs from producer contract')
    schema=runtime['robot_schema']
    if schema['dimension']!=76:raise ValueError('Robot76 schema dimension')
    fields={s['name']:slice(s['offset'],s['offset']+s['dimension']) for s in schema['fields']}
    parents={};episode_ids=set();rows=[];alignment_max=0.;fit_frames=0;masked_action_fit=0
    for ep in dataset.get('episodes',[]):
        for key in ('episode_id','parent_id','branch_id','split','distribution','actions','raw_path','gt_path',
                    'graph_path','graph_receipt_path','physical_audit_path','gnn_mask_path'):
            if key not in ep:raise ValueError('episode field missing '+key)
        if ep['episode_id'] in episode_ids:raise ValueError('duplicate episode identity')
        episode_ids.add(ep['episode_id'])
        parent=ep['parent_id'];split=ep['split']
        if split not in ('train','validation'):raise ValueError('Task10F fitting export split must be train/validation')
        if parent in parents and parents[parent]!=split:raise ValueError('parent/branch split leakage')
        parents[parent]=split
        for key in ('raw_path','gt_path','graph_path','graph_receipt_path','physical_audit_path','gnn_mask_path'):declared(ep[key])
        physical=read_json(declared(ep['physical_audit_path']))
        if physical.get('approved') is not True or physical.get('physical_safety') is not True:
            raise ValueError('branch lacks explicit physical approval')
        if physical.get('legacy_failed_pilot',False):raise ValueError('failed legacy pilot is not approved data')
        physical_checks=physical.get('checks')
        if (not isinstance(physical_checks,list) or len(physical_checks)!=int(ep['actions'])
            or physical.get('initial',{}).get('valid') is not True
            or not all(c.get('valid') is True for c in physical_checks)):
            raise ValueError('missing or failed per-step physical approval evidence')
        receipt=read_json(declared(ep['graph_receipt_path']))
        if receipt.get('readback')!='PASS' or receipt.get('sha256')!=files[ep['graph_path']]['sha256']:
            raise ValueError('graph receipt readback/SHA')
        if receipt.get('raw_episode_sha256')!=files[ep['raw_path']]['sha256']:
            raise ValueError('graph/raw observation identity mismatch')
        if receipt['identity']['graph_config']!=runtime['geometry']:
            raise ValueError('graph producer geometry differs')
        if receipt['identity']['frozen_geometry_protocol_sha256']!=runtime['geometry_protocol_sha256']:
            raise ValueError('graph protocol hash differs')
        native=runtime.get('native_runtime_acceptance')
        if native:
            acceptance=read_json(declared(native['path']))
            if (acceptance.get('status')!='PASS' or acceptance.get('mode')!='ALL_GRAPHS_REBUILT_IN_F1_NATIVE_RUNTIME'
                or acceptance.get('old576_graphs_included') is not False or receipt.get('graph_generation_instance')!=57987772
                or len(receipt.get('graph_runtime_sha256',''))!=64
                or receipt['graph_runtime_sha256']!=acceptance.get('graph_runtime_sha256')):
                raise ValueError('unified native graph runtime proof missing')
            if sha_file(declared(native['path']))!=native['sha256']:raise ValueError('native runtime acceptance SHA mismatch')
            if sha_file(declared(acceptance['graph_runtime_path']))!=acceptance['graph_runtime_sha256']:
                raise ValueError('native graph runtime identity mismatch')
        with h5py.File(declared(ep['raw_path']),'r') as f,np.load(declared(ep['gt_path']),allow_pickle=False) as gt, \
                np.load(declared(ep['graph_path']),allow_pickle=False) as graph,np.load(declared(ep['gnn_mask_path']),allow_pickle=False) as mask:
            n=int(ep['actions'])
            sampling_id=int(ep.get('graph_sampling_episode_id',ep['episode_id']))
            if int(graph['episode_id'])!=sampling_id:raise ValueError('graph sampling episode identity mismatch')
            if int(receipt.get('graph_sampling_episode_id',receipt.get('episode_id',sampling_id)))!=sampling_id:
                raise ValueError('graph receipt sampling identity mismatch')
            if n<1 or len(f['actions'])!=n or len(f['robot_state'])!=n+1:raise ValueError('raw T/T+1 alignment')
            if receipt.get('graphs')!=n or receipt.get('actions')!=n:raise ValueError('graph receipt frame count mismatch')
            for camera in ('base_camera','hand_camera'):
                for key in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'):
                    if len(f[camera+'/'+key])!=n+1:raise ValueError('camera observation alignment')
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1))
            np.testing.assert_array_equal(f['timestamp'][:],np.arange(n+1)/20)
            np.testing.assert_array_equal(gt['timestep'],f['timestep'][:])
            np.testing.assert_array_equal(gt['timestamp'],f['timestamp'][:])
            np.testing.assert_array_equal(graph['timestep'],np.arange(n))
            np.testing.assert_array_equal(graph['timestamp'],f['timestamp'][:n])
            np.testing.assert_array_equal(graph['robot76'],f['robot_state'][:n])
            selected=np.asarray(mask['gnn_supervision_mask']);valid=np.asarray(mask['physical_valid'])
            if selected.dtype!=bool or valid.dtype!=bool or selected.shape!=(n,) or valid.shape!=(n,):
                raise ValueError('explicit pre-action boolean GNN/physical mask required [T]')
            if (selected&~valid).any() or not selected.any():raise ValueError('invalid GNN supervision approval')
            raw={}
            for name,size in [('peg_pose',7),('hole_pose',7),('peg_half',3),('hole_radius',1)]:
                arr=gt['raw_'+name]
                if arr.shape!=(n+1,size) or not np.isfinite(arr).all():raise ValueError('missing/nonfinite GT '+name)
                raw[name]=arr
            for name in ('peg_pose','hole_pose'):
                np.testing.assert_allclose(np.linalg.norm(raw[name][:,3:],axis=-1),1,atol=2e-5,rtol=0)
            if (raw['peg_half']<=0).any() or (raw['hole_radius']<=0).any():raise ValueError('nonphysical shape labels')
            raw['tcp_pose']=f['tcp_pose'][:]
            np.testing.assert_array_equal(raw['tcp_pose'],f['robot_state'][:,fields['tcp_pose']])
            raw['left_tip']=f['robot_world_geometry'][:,1,:];raw['right_tip']=f['robot_world_geometry'][:,2,:]
            raw['gripper']=f['robot_state'][:,fields['gripper_position']]
            original=build_features(raw,np.ones(n+1,np.float32),np.zeros(n+1,np.float32))
            if 'gt38' in gt.files:
                error=float(np.max(abs(gt['gt38']-original)));alignment_max=max(alignment_max,error)
                np.testing.assert_allclose(gt['gt38'],original,atol=1e-7,rtol=0)
            if len(graph['node_ptr'])!=n+1 or len(graph['edge_ptr'])!=n+1:raise ValueError('graph action alignment')
            nn=len(graph['node_type']);ne=len(graph['edge_type'])
            if (graph['node_ptr'][0]!=0 or graph['node_ptr'][-1]!=nn or (np.diff(graph['node_ptr'])<3).any()
                or graph['edge_ptr'][0]!=0 or graph['edge_ptr'][-1]!=ne or (np.diff(graph['edge_ptr'])<0).any()
                or graph['edge_index'].shape!=(2,ne) or graph['edge_attr'].shape!=(ne,4)
                or graph['presence2'].shape!=(n,2)):
                raise ValueError('invalid graph pointers/types/presence shape')
            if graph['node_input'].shape[1]!=21 or graph['edge_attr'].shape[1]!=4:raise ValueError('Node21/Edge4')
            if (graph['node_input'][:,14:]!=0).any():raise ValueError('temporal extras must be zero')
            if not np.isfinite(graph['node_input']).all() or not np.isfinite(graph['edge_attr']).all():raise ValueError('nonfinite graph')
            for t in range(n):
                lo,hi=graph['node_ptr'][t:t+2];a,b=graph['edge_ptr'][t:t+2];typ=graph['node_type'][lo:hi]
                np.testing.assert_array_equal(typ[:3],np.arange(3))
                if not np.isin(typ[3:],[3,4]).all():raise ValueError('invalid object node type')
                np.testing.assert_array_equal(graph['presence2'][t],[(typ==3).any(),(typ==4).any()])
                if any((typ==kind).sum()>128 for kind in (3,4)):raise ValueError('object cap')
                edge=graph['edge_index'][:,a:b]
                if edge.size and (edge.min()<0 or edge.max()>=hi-lo):raise ValueError('bad graph-local edge')
            # Every stored Box surface node must originate only from Main.
            point_type=graph['node_type'][graph['point_graph_node']]
            source_type=point_type[graph['source_point']]
            if (graph['source_camera'][source_type==4]!=0).any():raise ValueError('Wrist Box source forbidden')
            off=np.asarray(f['supervision_mask'][:],bool) if 'supervision_mask' in f else np.ones(n,bool)
            masked_action_fit+=int((selected&~off).sum());fit_frames+=int(selected.sum())
            rows.append(dict(episode_id=ep['episode_id'],parent_id=parent,branch_id=ep['branch_id'],split=split,
                distribution=ep['distribution'],frames=n,fit_frames=int(selected.sum()),gt38_readback='PASS'))
    if not rows or fit_frames<1:raise ValueError('empty approved dataset')
    # A sealed marker plus two complete reads detects changing/truncated writes.
    for relative,row in files.items():
        path=safe_file(root,relative);stat=path.stat()
        if before[relative]!=(stat.st_size,stat.st_mtime_ns) or sha_file(path)!=row['sha256']:
            raise ValueError('sealed file changed during validation')
    return dict(status='PASS',dataset_id=m['dataset_id'],marker_sha256=sha_file(marker),
        dataset_manifest_sha256=m['dataset_manifest_sha256'],runtime_contract=runtime,
        files_verified=len(files),bytes_verified=sum(r['bytes'] for r in files.values()),
        episodes=rows,parent_count=len(parents),gnn_fit_frames=fit_frames,
        action_mask_off_eligible_frames=masked_action_fit,gt38_readback_max_abs=alignment_max,
        original_source_modified=False,all_files_readback='PASS',validated_at_epoch=time.time())

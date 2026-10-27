"""Independent evaluation-only edge, aperture, timing and provenance audit.

GT mesh/derived aperture coordinates are used solely as output metrics here.
They never enter graph sampling, registration, node features or GNN training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from graph.integration_baseline.io import write_json,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .contracts import transform
from .evaluate import load_cameras,VARIANTS
from .metrics import sample_meshes,visible_reference
from .graph import free_space_violation
from .audit import protection_snapshot,verify_protection
from .collect import inventory_sources


def aperture_reference(mesh,gt):
    """Square aperture rims, verified installed PegInsertionSide collision boxes.

    Full rim and actually observed rim support have different denominators.
    The GT-derived hole frame is evaluation-only, never an input or sampler.
    """
    hole=pose_matrix(gt['hole_pose']);actor=pose_matrix(gt['actor_pose'][1])
    points=transform(transform(mesh[1]['points'],actor),np.linalg.inv(hole))
    radius=float(np.asarray(gt['hole_radius']).reshape(-1)[0])
    lo,hi=points[:,0].min(),points[:,0].max()
    u=np.linspace(-radius,radius,64)
    rim=np.concatenate([np.c_[np.full(len(u),x),y,z] for x in (lo,hi)
        for y,z in ((u,np.full(len(u),-radius)),(u,np.full(len(u),radius)),
                    (np.full(len(u),-radius),u),(np.full(len(u),radius),u))])
    return hole,radius,lo,hi,transform(rim,hole)


def graph_audit(arrays,cameras,gt,meshes,aperture,visible):
    positions=transform(arrays['node_input'][:,:3],pose_matrix(cameras[0].tcp_pose))
    edges=arrays['edge_index'];types=arrays['node_type']
    same=(arrays['edge_type']==0)&(edges[0]<edges[1])
    edges=edges[:,same]
    rows=[]
    hole,radius,xlo,xhi,rim=aperture
    for oid in (0,1):
        selected=np.flatnonzero(types==oid+3);world=positions[selected]
        chosen=edges[:,types[edges[0]]==oid+3]
        reverse={int(k):i for i,k in enumerate(selected)}
        pairs=np.array([[reverse[int(a)],reverse[int(b)]] for a,b in chosen.T],int).reshape(-1,2)
        adjacency=coo_matrix((np.ones(len(pairs)*2),
            (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(world),len(world)))
        components=connected_components(adjacency,directed=False,return_labels=False) if len(world) else 0
        delta=positions[chosen[1]]-positions[chosen[0]]
        length=np.linalg.norm(delta,axis=1)
        samples=positions[chosen[0]][:,None]+np.linspace(.1,.9,9)[None,:,None]*delta[:,None]
        flat=samples.reshape(-1,3)
        local=transform(flat,np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
        distance=meshes[oid]['tree'].query(local)[0].reshape(-1,9)
        free=free_space_violation(flat,cameras).reshape(-1,9)
        aperture_crossing=np.zeros(len(chosen.T),bool)
        if oid==1 and len(flat):
            h=transform(flat,np.linalg.inv(hole)).reshape(-1,9,3)
            aperture_crossing=((abs(h[:,:,1])<radius-.001)&(abs(h[:,:,2])<radius-.001)&
                (h[:,:,0]>xlo-.001)&(h[:,:,0]<xhi+.001)).any(1)
        row=dict(object_id=oid,nodes=len(world),undirected_surface_edges=len(chosen.T),
            connected_components=int(components),
            mesh_unsupported_edges_4mm=int((distance>.004).any(1).sum()),
            camera_free_space_crossing_edges=int(free.any(1).sum()),
            aperture_crossing_edges=int(aperture_crossing.sum()),
            edge_length_mean_mm=float(length.mean()*1000) if len(length) else None,
            edge_length_p95_mm=float(np.percentile(length,95)*1000) if len(length) else None)
        for axis,label in enumerate('xyz'):
            row['edge_abs_direction_'+label]=float((abs(delta[:,axis])/np.maximum(length,1e-12)).mean()) if len(length) else None
        if oid==1:
            observed=cKDTree(visible[1]).query(rim)[0]<.004 if len(visible[1]) else np.zeros(len(rim),bool)
            coverage=cKDTree(world).query(rim)[0]<.004 if len(world) else np.zeros(len(rim),bool)
            row.update(aperture_rim_reference=len(rim),aperture_rim_observed_reference=int(observed.sum()),
                aperture_full_rim_coverage_4mm=float(coverage.mean()),
                aperture_observed_rim_coverage_4mm=float(coverage[observed].mean()) if observed.any() else None,
                aperture_metric_scope='evaluation-only square collision aperture; full versus current-both-visible support')
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=5)
    args=parser.parse_args()
    from .handoff import maybe_delegate
    if maybe_delegate('geometry_audit',args):return
    manifest=json.loads((args.geometry/'geometry_manifest.json').read_text())
    capture=Path(manifest['config']['capture'])
    collected=json.loads((capture/'capture_manifest.json').read_text())
    episodes={e['seed']:e for e in collected['episodes']}
    if json.loads((args.geometry/'metadata.json').read_text())['status']!='finished':raise ValueError('unfinished geometry')
    protected=protection_snapshot(p for p in args.geometry.rglob('*') if p.is_file())
    config=dict(task='09-C',stage='independent_geometry_audit',geometry=str(args.geometry),
        interval=args.interval,GT_scope='evaluation-only',metrics_do_not_modify_inputs=True)
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_geometry_audit',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_GEOMETRY_AUDIT_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'protection_before.json',protected)
        metrics=[];causal=[];meshes={};last_seen={};all_ids={};retained=[]
        for frame in manifest['frames']:
            seed,t=frame['seed'],frame['timestep'];episode=episodes[seed]
            source=capture/'collection'/f'seed{seed}'
            folder=args.geometry/frame['folder']
            condition=frame.get('condition','baseline')
            if condition=='baseline':
                input_path=source/f'f{t:04d}_input.npz';gt_path=source/f'f{t:04d}_gt.npz'
            else:
                branch=next(b for b in episode['branches'] if b['timestep']==t and b['condition']==condition)
                input_path=source/branch['source']/'input.npz';gt_path=source/branch['source']/'gt.npz'
            if seed not in meshes:meshes[seed]=sample_meshes(source/'evaluation_meshes.npz')
            # Causality and identity are audited every baseline frame, whereas
            # expensive edge/mesh metrics use a declared fixed time interval.
            for name in VARIANTS:
                path=folder/(name+'_source_observations.npz')
                if path.exists():
                    with np.load(path,allow_pickle=False) as a:
                        if np.any(a['source_timestep']>t):raise ValueError('future source found')
                        causal.append(dict(seed=seed,timestep=t,variant=name,observations=len(a['source_timestep']),future=0))
                if condition=='baseline' and name in ('G2','G4','O1','O2','O3'):
                    with np.load(folder/(name+'_map.npz'),allow_pickle=False) as a:
                        active=a['status']!=3;ids=set(a['persistent_node_id'][active].tolist())
                        if any(int(v)>t for v in a['last_seen_timestep']):raise ValueError('map future observation')
                        key=(seed,name);old=last_seen.get(key,set())
                        retained.append(dict(seed=seed,timestep=t,variant=name,active_nodes=len(ids),
                            retained_from_previous=len(ids&old),previous_active=len(old),
                            currently_observed=int((a['currently_observed']&active).sum()),
                            estimated_only=int((a['estimated_only']&active).sum())))
                        if t==0:
                            for previous_seed,previous_ids in all_ids.get(name,{}).items():
                                if ids&previous_ids:raise ValueError('persistent identity crosses reset')
                        last_seen[key]=ids;all_ids.setdefault(name,{}).setdefault(seed,set()).update(ids)
            if t%args.interval and condition=='baseline':continue
            cameras,_=load_cameras(input_path)
            with np.load(gt_path,allow_pickle=False) as a:gt={k:a[k] for k in a.files}
            aperture=aperture_reference(meshes[seed],gt);visible=visible_reference(cameras,gt)
            for name in VARIANTS:
                for budget,suffix in (('cap128_per_object','_graph.npz'),('matched_predicted_N','_matched_N_graph.npz')):
                    path=folder/(name+suffix)
                    if not path.exists():continue
                    with np.load(path,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
                    if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('nonfinite graph tensor')
                    metrics.extend(dict(seed=seed,timestep=t,condition=condition,split=frame['split'],
                        variant=name,budget=budget,graph_sha256=sha256(path),**r)
                        for r in graph_audit(arrays,cameras,gt,meshes[seed],aperture,visible))
            if t%25==0:print(f'geometry audit seed={seed} t={t} {condition}',flush=True)
        csv_write(root/'graph_boundary_metrics.csv',metrics);csv_write(root/'temporal_retention.csv',retained)
        write_json(root/'causality_identity_audit.json',dict(status='PASS',source_events=causal,
            reset_identity_disjoint=True,future_sources=0,graph_metrics_interval=args.interval))
        result=verify_protection(protected);write_json(root/'protection_audit.json',result)
        if result['status']!='PASS':raise ValueError('audit changed geometry')
        for name in ('graph_boundary_metrics.csv','temporal_retention.csv','causality_identity_audit.json','protection_audit.json'):
            logger.link_output(root/name,kind='metrics')


if __name__=='__main__':main()

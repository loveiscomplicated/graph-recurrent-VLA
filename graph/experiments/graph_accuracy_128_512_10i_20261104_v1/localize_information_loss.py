"""Fine-grained frozen-input loss ledger and point-budget controls.

Observers preserve the original graph. GT defines evaluation regions only.
Variable-N/256/matched-N controls are point sets, not production Graph variants.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha, sensor_frame, world_points
from .evaluate import actor_labels
from .loss_metrics import covered, transition, high_resolution_gain, retention
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_geometry.core import budget
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.persistent_geometry.tensor_graph import numpy_graph, FIELDS
from graph.persistent_geometry.tensor_records import numpy_records, _identity_ranks
from graph.persistent_geometry.geometry_audit import aperture_reference
from graph.persistent_geometry.metrics import sample_meshes
from graph.task10i_stage4a2.dinov2_matching_audit import csv_write
from visual.runtime.tensor_packet import numpy_packet
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger

ACTUAL = ('raw', 'clean', 'gate', 'shape_first', 'cross_view', 'dedup', 'fps128')
BRANCH = ('PRIORITY_C_PLUS_S', 'PRIORITY_R_PLUS_S', 'PRIORITY_P_PLUS_S')
CONTROLS = ('FPS256', 'NATIVE_VARIABLE_N', 'MATCHED_N', 'ORACLE_CORRECT_POOL_FPS128')


def from_records(rows, tcp):
    return dict(xyz=world_points(rows, tcp),
        oid=np.asarray([r['object_id'] for r in rows], int),
        cam=np.asarray([CAMERAS.index(r['source_observation_ids'][0][2]) for r in rows], int),
        pix=np.asarray([r['source_observation_ids'][0][3] for r in rows], int))


def from_sources(world, sources):
    xyz, oids, cams, pixels = [], [], [], []
    for (ci, oid), source in sorted(sources.items()):
        source = np.asarray(source, np.int64)
        xyz.append(world[ci].reshape(-1, 3)[source])
        oids.extend([oid]*len(source)); cams.extend([ci]*len(source)); pixels.extend(source)
    return dict(xyz=np.concatenate(xyz).reshape(-1,3), oid=np.asarray(oids,int),
                cam=np.asarray(cams,int), pix=np.asarray(pixels,int))


def source_keys(array):
    return set(zip(array['oid'].tolist(), array['cam'].tolist(), array['pix'].tolist()))


def masks_to_sources(host, masks):
    return {(ci,oid):np.flatnonzero(masks[ci,oid]&np.isfinite(c.depth)&~(masks[ci,0]&masks[ci,1]))
            for ci,c in enumerate(host) for oid in (0,1) if oid==0 or ci==0}


def reference(capture, index, parent, t, mesh):
    cameras, _ = sensor_frame(capture,index,parent,t,1024,parent)
    scene=index[parent,t,1024,CAMERAS[0]]['scene']
    state=read(capture/'GT'/scene/'state.json')
    segs=[]; ids=[]; worlds=[c.world_points() for c in cameras]
    for c in CAMERAS:
        e=index[parent,t,1024,c]
        with np.load(capture/e['gt_file']) as z:segs.append(z['actor_segmentation'].copy())
        ids.append(e['object_ids'])
    result={}
    for oid,name in enumerate(('peg','box')):
        xyz=[];cam=[];pix=[];boundary=[]
        for ci,c in enumerate(cameras):
            if oid==1 and ci==1:continue
            target=np.isin(segs[ci],ids[ci][name])&np.isfinite(c.depth)
            source=np.flatnonzero(target);edge=target&~binary_erosion(target)
            xyz.append(worlds[ci].reshape(-1,3)[source]);cam.extend([ci]*len(source));pix.extend(source)
            boundary.extend(edge.ravel()[source])
        xyz=np.concatenate(xyz).reshape(-1,3)
        regions=dict(all=np.ones(len(xyz),bool),boundary=np.asarray(boundary,bool),
            main=np.asarray(cam)==0,wrist=np.asarray(cam)==1)
        if oid==0:
            local=transform(xyz,np.linalg.inv(pose_matrix(state['peg_pose'])))
            length=float(state['peg_half'][0])
            regions.update(end_minus=local[:,0]<=-.8*length,end_plus=local[:,0]>=.8*length,
                           middle=abs(local[:,0])<.8*length)
        result[oid]=dict(xyz=xyz,cam=np.asarray(cam,int),pix=np.asarray(pix,int),regions=regions)
    _,_,_,_,rim=aperture_reference(mesh,dict(hole_pose=state['hole_pose'],
        actor_pose=[state['peg_pose'],state['box_pose']],hole_radius=state['hole_radius']))
    approach_rim=rim[:len(rim)//2]
    support=cKDTree(result[1]['xyz']).query(approach_rim)[0]<.004 if len(result[1]['xyz']) else np.zeros(len(approach_rim),bool)
    result[1]['observed_approach_rim']=approach_rim[support]
    return result


class Observer:
    def __init__(self, geo):self.geo=geo;self.data={};self.camera=None
    def __enter__(self):
        import visual.runtime.cuda_sampling as sampling
        import graph.persistent_geometry.tensor_records as records_module
        self.stack=ExitStack()
        build=self.geo._build_prepared; represent=self.geo.system.represent
        packet=self.geo.system.tensor_packet_builder;shape=sampling.shape_first_cuda;dedup=records_module.deduplicate_tensor
        def observe_build(cs,ds,ss,ms,clean_masks=None,**kw):
            self.data['clean']=clean_masks
            return build(cs,ds,ss,ms,clean_masks,**kw)
        def observe_packet(c,*a,**kw):
            self.camera=c.camera
            return packet(c,*a,**kw)
        def observe_shape(info):
            result=shape(info);self.data.setdefault('shape',{})[self.camera]=(info,result)
            return result
        def observe_dedup(records,*a,**kw):
            self.data['before_dedup']=records
            result=dedup(records,*a,**kw);self.data['pool']=result[0]
            return result
        def observe_represent(cs,prepared,*a,**kw):
            result=represent(cs,prepared,*a,**kw)
            self.data.update(gate_masks=prepared['conditions']['C3'],packets=result[3])
            return result
        for obj,name,fn in [(self.geo,'_build_prepared',observe_build),
                (self.geo.system,'represent',observe_represent),(self.geo.system,'tensor_packet_builder',observe_packet),
                (sampling,'shape_first_cuda',observe_shape),(records_module,'deduplicate_tensor',observe_dedup)]:
            self.stack.enter_context(patch.object(obj,name,fn))
        return self
    def __exit__(self,*args):return self.stack.__exit__(*args)


def packing_audit(graph, records):
    order=torch.argsort(_identity_ranks(records),stable=True)
    order=order[torch.argsort(records.object_id[order],stable=True)]
    valid=records.feature_valid[order]
    expected=torch.where(valid[:,[0,0,0,1,2]],records.f5[order],0).to(torch.float32)
    assert torch.equal(graph.node_input[3:,:3],records.xyz[order].to(torch.float32))
    assert torch.equal(graph.node_input[3:,3:8],expected)
    assert torch.equal(graph.node_input[3:,11:14],valid.to(torch.float32))
    assert torch.equal(graph.node_type[3:],records.object_id[order]+3)
    assert len(graph.node_type)==len(records)+3
    return dict(nodes=len(records),xyz_exact=True,f5_valid_values_exact=True,validity_exact=True,
                invalid_feature_values=int((~valid).sum()),discarded_nodes=0)


def connectivity(graph, oid):
    types=graph.node_type.numpy();edges=graph.edge_index.numpy();et=graph.edge_type.numpy()
    nodes=np.flatnonzero(types==oid+3);reverse={int(v):i for i,v in enumerate(nodes)}
    e=edges[:,(et==0)&(types[edges[0]]==oid+3)]
    a=np.asarray([reverse[int(v)] for v in e[0]],int);b=np.asarray([reverse[int(v)] for v in e[1]],int)
    matrix=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(nodes),len(nodes)))
    degree=np.bincount(a,minlength=len(nodes))
    return dict(intra_edges_directed=len(a),isolated_intra_nodes=int((degree==0).sum()),
        components=int(connected_components(matrix,directed=False,return_labels=False)) if len(nodes) else 0)


def score(array, oid, segs, ids, ref):
    take=array['oid']==oid;xyz=array['xyz'][take]
    labels=actor_labels(segs,ids,oid,array['cam'][take],array['pix'][take]);good=labels=='correct'
    correct_coverage=covered(xyz,good,ref['xyz'])
    all_coverage=covered(xyz,np.ones(len(xyz),bool),ref['xyz'])
    row=dict(points=len(xyz),correct_points=int(good.sum()),fp_points=int((~good).sum()),
        main_points=int((array['cam'][take]==0).sum()),wrist_points=int((array['cam'][take]==1).sum()),
        precision=float(good.mean()) if len(xyz) else None,
        all_node_coverage=float(all_coverage.mean()) if len(all_coverage) else None)
    for name,region in ref['regions'].items():
        row['ref_'+name]=int(region.sum());row['coverage_'+name]=retention(correct_coverage,region)
    if oid==1:
        rim=ref['observed_approach_rim'];row['ref_observed_rim']=len(rim)
        row['coverage_observed_rim']=float(covered(xyz,good,rim).mean()) if len(rim) else None
    return row,correct_coverage


def aggregate(rows, keys, metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,group in sorted(groups.items(),key=lambda v:str(v[0])):
        row=dict(zip(keys,key));row['rows']=len(group)
        for metric in metrics:
            values=[r[metric] for r in group if r.get(metric) is not None]
            row[metric+'_mean']=float(np.mean(values)) if values else None
            row[metric+'_defined_rows']=len(values)
        result.append(row)
    return result


def main(capture,inference,runtime,output):
    frozen=read(inference/'inference_complete.json')
    assert frozen['status']=='COMPLETE' and frozen['graph_frames']==168
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    index=entries(capture);schema=read(runtime/'normalizer.json')['schema']
    config=dict(parents=PARENTS,timesteps=list(range(21)),sizes=SIZES,
        frozen_inference_receipt_sha256=sha(inference/'inference_complete.json'),model_forwards=0,
        code_sha256=sha(Path(__file__)),loss_metrics_sha256=sha(Path(__file__).with_name('loss_metrics.py')),
        reference='common native1024 current visible GT; Peg both cameras, Box Main; radius strict4mm',
        branch_controls='C/R/P + identical actual Surface sources; GT-free, not a new sampling policy',
        budget_controls='same pool FPS256, native variable N, paired per-object matched N; point sets only',
        oracle='GT-correct source filtering before FPS128, diagnostic only',
        end_regions='observed outer10% at each local X end; not end-face center pose accuracy')
    rows=[];transitions=[];gains=[];proof=[];merges=[];feature_rows=[];pair_counts=[]
    with ExperimentLogger(team='integration',model='task10i_512_information_loss_localization',seed=42,
            job_type='evaluate',root=output/'experiments',config=config) as log:
        for parent in PARENTS:
            geometries={s:CurrentGeometry(runtime/'inputs',schema) for s in SIZES}
            mesh_path=capture/'GT'/index[parent,0,1024,CAMERAS[0]]['scene']/'meshes.npz'
            mesh=sample_meshes(mesh_path,samples_per_object=10000)
            try:
                for t in range(21):
                    refs=reference(capture,index,parent,t,mesh);pair={}
                    for size,geo in geometries.items():
                        identity=f'{parent}:native{size}'
                        host,data=sensor_frame(capture,index,parent,t,size,identity)
                        worlds=[c.world_points() for c in host]
                        cameras=[tensor_camera(c) for c in host];surfaces=geo.fk.surfaces(data['robot76'])
                        data.update(episode=np.array(identity),timestep=np.int64(t),timestamp=np.float64(t/20),
                            qpos=data['robot76'][:9],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
                        with np.load(inference/parent/str(size)/f't{t:04d}.npz') as z:expected={k:z[k].copy() for k in z.files}
                        raw={c:torch.as_tensor(expected['raw_masks'][ci],device='cuda') for ci,c in enumerate(CAMERAS)}
                        with Observer(geo) as observer:graph,final,_=geo.build(cameras,data,surfaces,lambda:raw)
                        audit=packing_audit(graph,final);saved=numpy_graph(graph)
                        assert all(np.array_equal(getattr(saved,k).numpy(),expected[k]) for k in FIELDS)
                        cap=observer.data;pool_rows=numpy_records(cap['pool']);final_rows=numpy_records(final)
                        array={}
                        for stage,mm in [('raw',expected['raw_masks']),
                                ('clean',np.stack([cap['clean'][c].cpu().numpy() for c in CAMERAS])),
                                ('gate',np.stack([cap['gate_masks'][c].cpu().numpy() for c in CAMERAS]))]:
                            array[stage]=from_sources(worlds,masks_to_sources(host,mm))
                        branches={stage:{} for stage in BRANCH};packet_parts=[]
                        for ci,c in enumerate(host):
                            p=numpy_packet(cap['packets'][c.camera]);part=dict(xyz=transform(p['points'],pose_matrix(data['tcp_pose'])),
                                oid=p['object_id'],cam=np.full(len(p['object_id']),ci),pix=p['provenance']['source_index'])
                            packet_parts.append(part)
                            info,result=cap['shape'][c.camera]
                            for oid in (0,1):
                                if oid==1 and ci==1:continue
                                prep=result['prepared'][oid];sel=result['selected'][oid]
                                s=sel['S'].cpu().numpy()
                                cands=np.union1d(prep['C']['pca'].cpu().numpy(),prep['C']['sil'].cpu().numpy())
                                reps=np.union1d(prep['R']['pca'].cpu().numpy(),prep['R']['sil'].cpu().numpy())
                                priority=sel['P'].cpu().numpy()
                                for name,source in zip(BRANCH,(cands,reps,priority)):
                                    branches[name][ci,oid]=np.union1d(source,s)
                                final_sources=p['provenance']['source_index'][p['object_id']==oid]
                                assert np.array_equal(np.union1d(priority,s),np.sort(final_sources))
                                take=info['projected']['object_id']==oid
                                flags=np.column_stack([info['metadata'][name][take].cpu().numpy()
                                    for name in ('normal_valid','normal_variation_valid','depth_pair_valid')])
                                feature_rows.append(dict(parent=parent,timestep=t,size=size,object_id=oid,camera=c.camera,
                                    stage='dense_F5',points=len(flags),normal_valid=int(flags[:,0].sum()),
                                    variation_valid=int(flags[:,1].sum()),depth_pair_valid=int(flags[:,2].sum())))
                        array['shape_first']={k:np.concatenate([v[k] for v in packet_parts]) for k in ('xyz','oid','cam','pix')}
                        array['cross_view']=from_records(numpy_records(cap['before_dedup']),data['tcp_pose'])
                        array['dedup']=from_records(pool_rows,data['tcp_pose']);array['fps128']=from_records(final_rows,data['tcp_pose'])
                        for name,source in branches.items():array[name]=from_sources(worlds,source)
                        cpu128=from_records(budget(pool_rows,128),data['tcp_pose'])
                        assert source_keys(cpu128)==source_keys(array['fps128'])
                        array['FPS256']=from_records(budget(pool_rows,256),data['tcp_pose'])
                        array['NATIVE_VARIABLE_N']=array['dedup']
                        segs=[];ids=[]
                        for camera in CAMERAS:
                            e=index[parent,t,size,camera]
                            with np.load(capture/e['gt_file']) as z:segs.append(z['actor_segmentation'].copy())
                            ids.append(e['object_ids'])
                        correct_pool=[]
                        for r in pool_rows:
                            oid=r['object_id'];e,ts,c,p=r['source_observation_ids'][0];ci=CAMERAS.index(c)
                            labels=actor_labels(segs,ids,oid,[ci],[p])
                            if labels[0]=='correct':correct_pool.append(r)
                            all_labels=actor_labels(segs,ids,oid,
                                [CAMERAS.index(v[2]) for v in r['source_observation_ids']],
                                [v[3] for v in r['source_observation_ids']])
                            merges.append(dict(parent=parent,timestep=t,size=size,object_id=oid,
                                source_count=len(all_labels),correct_sources=int((all_labels=='correct').sum()),
                                representative_correct=bool(labels[0]=='correct'),
                                wrong_representative_with_correct_source=bool(labels[0]!='correct' and (all_labels=='correct').any())))
                        array['ORACLE_CORRECT_POOL_FPS128']=from_records(budget(correct_pool,128),data['tcp_pose'])
                        for stage,table in [('cross_view',cap['before_dedup']),('dedup',cap['pool']),('fps128',final)]:
                            flags=table.feature_valid.cpu().numpy();oids=table.object_id.cpu().numpy();cams=table.camera_id.cpu().numpy()
                            for oid in (0,1):
                                for ci,camera in enumerate(table.camera_names):
                                    take=(oids==oid)&(cams==ci);f=flags[take]
                                    feature_rows.append(dict(parent=parent,timestep=t,size=size,object_id=oid,camera=camera,
                                        stage=stage,points=len(f),normal_valid=int(f[:,0].sum()),
                                        variation_valid=int(f[:,1].sum()),depth_pair_valid=int(f[:,2].sum())))
                        proof.append(dict(parent=parent,timestep=t,size=size,five_tensors_exact=True,
                            cpu_fps128_same_sources=True,**audit))
                        pair[size]=dict(arrays=array,pool_rows=pool_rows,tcp=data['tcp_pose'],segs=segs,ids=ids,
                            topology={oid:connectivity(saved,oid) for oid in (0,1)})
                    matched={oid:min(sum(r['object_id']==oid for r in pair[size]['pool_rows']) for size in SIZES) for oid in (0,1)}
                    case_arrays={}
                    for size in SIZES:
                        p=pair[size];p['arrays']['MATCHED_N']=from_records(budget(p['pool_rows'],matched),p['tcp'])
                        for oid in (0,1):pair_counts.append(dict(parent=parent,timestep=t,size=size,object_id=oid,matched_budget=matched[oid]))
                        p['coverage']={}
                        for oid in (0,1):
                            p['coverage'][oid]={}
                            for stage,a in p['arrays'].items():
                                metric,cov=score(a,oid,p['segs'],p['ids'],refs[oid])
                                p['coverage'][oid][stage]=cov
                                rows.append(dict(parent=parent,timestep=t,size=size,object_id=oid,stage=stage,
                                    **metric,**(p['topology'][oid] if stage=='fps128' else {})))
                                if t in (0,10,20):case_arrays[f'coverage_s{size}_o{oid}_{stage}']=np.packbits(cov)
                            for before,after in zip(ACTUAL,ACTUAL[1:]):
                                for region,mask in refs[oid]['regions'].items():
                                    transitions.append(dict(parent=parent,timestep=t,size=size,object_id=oid,
                                        before_stage=before,after_stage=after,region=region,
                                        **transition(p['coverage'][oid][before],p['coverage'][oid][after],mask)))
                    for oid in (0,1):
                        gain=high_resolution_gain(pair[128]['coverage'][oid]['raw'],pair[512]['coverage'][oid]['raw'])
                        for size in SIZES:
                            for stage,cov in pair[size]['coverage'][oid].items():
                                for region,mask in refs[oid]['regions'].items():
                                    selected=gain&mask
                                    gains.append(dict(parent=parent,timestep=t,size=size,object_id=oid,stage=stage,region=region,
                                        gain_reference_samples=int(selected.sum()),retained_samples=int((selected&cov).sum()),
                                        retained_fraction=retention(cov,selected)))
                        if t in (0,10,20):
                            ref=refs[oid];case_arrays.update({f'ref_xyz_o{oid}':ref['xyz'],f'ref_cam_o{oid}':ref['cam'],
                                f'ref_pix_o{oid}':ref['pix'],f'gain_o{oid}':np.packbits(gain)})
                            for name,mask in ref['regions'].items():case_arrays[f'region_o{oid}_{name}']=np.packbits(mask)
                    if case_arrays:
                        folder=output/'reference_ledgers';folder.mkdir(exist_ok=True)
                        np.savez_compressed(folder/f'{parent}_t{t:04d}.npz',**case_arrays)
                    if t%10==0:print('LOCALIZED',parent,t,flush=True)
                write(output/'progress.json',dict(completed_parent=parent,graph_frames=len(proof)))
            finally:
                for geo in geometries.values():geo.close()
        for name,data in [('stage_metrics',rows),('transitions',transitions),('resolution_gain_retention',gains),
                          ('dedup_groups',merges),('feature_validity',feature_rows),('matched_budgets',pair_counts)]:
            csv_write(output/(name+'.csv'),data)
        write(output/'replay_and_encoding.json',proof)
        summary=aggregate(rows,('size','object_id','stage'),
            ('points','precision','all_node_coverage','coverage_all','coverage_boundary','coverage_main','coverage_wrist',
             'coverage_end_minus','coverage_end_plus','coverage_middle','coverage_observed_rim'))
        transition_summary=aggregate(transitions,('size','object_id','before_stage','after_stage','region'),
            ('reference_samples','lost','recovered','loss_fraction','net_loss_fraction'))
        gain_summary=aggregate(gains,('size','object_id','stage','region'),('gain_reference_samples','retained_fraction'))
        write(output/'transition_summary.json',transition_summary);write(output/'gain_summary.json',gain_summary)
        write(output/'summary.json',dict(status='COMPLETE',run_id=log.run_id,config=config,stages=summary,
            graph_frames=len(proof),all_five_tensors_exact=True,all_valid_features_encoded_exact=True,
            point_set_controls_are_not_production_graphs=True))
        log.log(dict(graph_frames=len(proof),model_forwards=0))
        for name in ('summary.json','transition_summary.json','gain_summary.json','replay_and_encoding.json'):
            log.link_output(output/name)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('capture','inference','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.capture,a.inference,a.runtime,a.output)

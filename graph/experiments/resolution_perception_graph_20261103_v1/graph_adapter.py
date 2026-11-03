"""Sensor-only adapter and observational hooks around the unchanged PlanA builder."""
from common import *
from contextlib import contextmanager
from functools import wraps,cached_property
import time
import numpy as np
import torch
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.map import current_records
from visual.runtime.vision_packet import pose_matrix
from estimator import estimate_graph


class Spans:
    """Exclusive wall spans; diagnostic runs only, never official total latency."""
    def __init__(self):
        self.stack=[];self.rows=[];self.original=[]

    @contextmanager
    def span(self,name):
        item=[name,time.perf_counter(),0.0];self.stack.append(item)
        try:yield
        finally:
            elapsed=(time.perf_counter()-item[1])*1000;self.stack.pop()
            if self.stack:self.stack[-1][2]+=elapsed
            self.rows.append(dict(stage=name,inclusive_ms=elapsed,exclusive_ms=max(0.,elapsed-item[2])))

    def hook(self,module,name,label,after=None):
        original=getattr(module,name);self.original.append((module,name,original))
        @wraps(original)
        def call(*a,**kw):
            with self.span(label):result=original(*a,**kw)
            if after is not None:after(result,a,kw)
            return result
        setattr(module,name,call)

    def close(self):
        for module,name,fn in reversed(self.original):setattr(module,name,fn)


def geometry():
    return CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs',
                           read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema'])


class QualityCamera(CameraObservation):
    """Quality-only memoization of pure, read-only sensor conversions; never timed."""
    @cached_property
    def depth(self):
        value=super().depth;value.setflags(write=False);return value

    @cached_property
    def world_from_camera(self):
        value=super().world_from_camera;value.setflags(write=False);return value


def load_inputs(scene,size,geo,*,cache_sensors=False):
    folder=ROOT/'sensors'/scene/str(size)
    io_start=time.perf_counter()
    with np.load(folder/'robot.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    spec=next(s for s in read(ROOT/'protocol.json')['scenes'] if s['id']==scene)
    io_ms=(time.perf_counter()-io_start)*1000;cameras=[];join_ms=0.
    for c in CAMERAS:
        io_start=time.perf_counter()
        with np.load(folder/(c+'.npz'),allow_pickle=False) as z:
            values={k:z[k].copy() for k in ('rgb','depth_raw_mm','intrinsic_cv','extrinsic_cv','cam2world_gl')}
        io_ms+=(time.perf_counter()-io_start)*1000;join_start=time.perf_counter()
        cls=QualityCamera if cache_sensors else CameraObservation
        cameras.append(cls(scene,int(spec['timestep']),c,float(spec['timestep']/20),
            values['rgb'],values['depth_raw_mm'],values['intrinsic_cv'],values['extrinsic_cv'],
            values['cam2world_gl'],data['tcp_pose'].copy()).validate())
        join_ms+=(time.perf_counter()-join_start)*1000
    start=time.perf_counter()
    geo.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
    surfaces=geo.fk.surfaces(data['robot76'])
    fk_ms=(time.perf_counter()-start)*1000
    data.update(episode=np.array(scene),timestep=np.int64(spec['timestep']),timestamp=np.float64(spec['timestep']/20),
                qpos=data['robot76'][:9],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    return cameras,data,surfaces,fk_ms,dict(input_join_ms=join_ms,file_read_decode_ms=io_ms)


def records_arrays(records):
    return dict(xyz=np.asarray([r['world_xyz'] for r in records],np.float32).reshape(-1,3),
                object_id=np.asarray([r['object_id'] for r in records],np.int8),
                camera=np.asarray([CAMERAS.index(r['source_observation_ids'][0][2]) for r in records],np.int8),
                pixel=np.asarray([r['source_observation_ids'][0][3] for r in records],np.int32),
                merged_sources=np.asarray([len(r['source_observation_ids']) for r in records],np.int32))


def dense_arrays(cameras,masks):
    arrays={k:[] for k in ('xyz','object_id','camera','pixel','merged_sources')}
    for ci,c in enumerate(cameras):
        world=c.world_points().reshape(-1,3)
        for oid in (0,1):
            pix=np.flatnonzero(masks[c.camera][oid]&np.isfinite(c.depth))
            for k,v in dict(xyz=world[pix].astype(np.float32),object_id=np.full(len(pix),oid,np.int8),
                            camera=np.full(len(pix),ci,np.int8),pixel=pix.astype(np.int32),merged_sources=np.ones(len(pix),np.int32)).items():arrays[k].append(v)
    return {k:np.concatenate(v) for k,v in arrays.items()}


def build(geo,inputs,masks,trace=False):
    cameras,data,surfaces,fk_ms,diagnostics=inputs
    spans=Spans();stages={};cap={};phase={'value':'prepare'}
    if trace:
        import graph.plan_a_geometry.core as core
        import graph.persistent_geometry.graph as graphmod
        import graph.integration_baseline.vision as vision
        import visual.runtime.sampling as sampling
        import visual.runtime.vision_packet as packetmod
        stages['raw']=dense_arrays(cameras,masks)
        def prepared(out,a,k):
            stages['filtered']=dense_arrays(cameras,out['conditions']['C3'])
        def packet(out,a,k):
            cap.setdefault('shape',[]).extend(current_records(core.observations_from_packet(out[0],a[0]),fuse_views=False))
        def consistent(out,a,k):
            if phase['value']=='represent':cap.setdefault('cross',[]).extend(current_records(out[0],fuse_views=False))
        def represent_after(out,a,k):
            stages['pre_budget']=records_arrays(out[0]['SF']);phase['value']='final'
        original_represent=geo.system.represent
        def represent(*a,**kw):
            phase['value']='represent'
            with spans.span('represent_total'):out=original_represent(*a,**kw)
            represent_after(out,a,kw);return out
        geo.system.represent=represent
        spans.hook(core.PlanA,'prepare','prepare_total',prepared)
        for name,label in [('classify_robot','robot_filter'),('filtered_masks','mask_overlap_filter'),('wrist_gate','wrist_gate'),
                           ('deduplicate','dedup_3mm'),('budget','fps_budget')]:spans.hook(core,name,label)
        # CurrentGeometry imports budget separately from core.
        import graph.plan_a_bc.geometry as adaptermod
        spans.hook(adaptermod,'budget','final_fps_128')
        spans.hook(core,'make_packet','camera_packet_total',packet)
        spans.hook(core,'consistent_views','cross_view',consistent)
        spans.hook(core,'observations_from_packet','packet_to_observations')
        spans.hook(core,'current_records','current_records')
        spans.hook(vision,'extract_sampling_evidence','evidence_total')
        for name,label in [('build_geometry','backprojection'),('local_plane_optimized','pca5_normals'),
                           ('cropped_features','geometric_features'),('pair_evidence','pair_evidence')]:spans.hook(sampling,name,label)
        spans.hook(vision,'extract_candidates','structural_candidates')
        for name,label in [('prepare_object','shape_prepare'),('select_object','shape_select'),('make_packet','selection_pack')]:spans.hook(vision.shape_first,name,label)
        spans.hook(vision,'complete_vision_packet','feature_ee_pack')
        spans.hook(vision,'observed_components','observed_components')
        spans.hook(vision,'build_raw_ede','packet_raw_ede')
        spans.hook(core,'build_graph','final_graph_total')
        for name,label in [('surface_edges','final_surface_edges'),('nearest_robot_pairs','robot_object_edges'),
                           ('global_peg_box_pairs','peg_box_edges'),('edge_attributes','edge_attributes')]:spans.hook(graphmod,name,label)
    start=time.perf_counter()
    try:
        if trace:
            with spans.span('graph_build'):g,records,gate=geo.build(cameras,data,surfaces,masks)
        else:g,records,gate=geo.build(cameras,data,surfaces,masks)
        total_ms=(time.perf_counter()-start)*1000
    finally:
        if trace:geo.system.represent=original_represent
        spans.close()
    if trace:
        stages['shape_first']=records_arrays(cap.get('shape',[]))
        stages['cross_view']=records_arrays(cap.get('cross',[]))
        stages['final']=records_arrays(records)
    arrays={k:getattr(g,k).numpy().copy() for k in FIELDS}
    hashes={k:array_sha(v) for k,v in arrays.items()}
    probe_start=time.perf_counter();est=estimate_graph(g,data['tcp_pose']);probe_ms=(time.perf_counter()-probe_start)*1000
    return dict(graph=arrays,hashes=hashes,estimate=est,stages=stages,gate=gate,
                graph_ms=total_ms,fk_ms=fk_ms,input_join_ms=diagnostics['input_join_ms'],file_read_decode_ms=diagnostics['file_read_decode_ms'],probe_ms=probe_ms,spans=spans.rows,
                records_order='original budget records; Graph tensor node ordering is independently stable-ID sorted',
                sources=[r['source_observation_ids'] for r in records],records=records_arrays(records))


def save_result(folder,result):
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'graph.npz',**result['graph'],**result['records'])
    if result['stages']:
        np.savez_compressed(folder/'stages.npz',**{stage+'__'+k:v for stage,a in result['stages'].items() for k,v in a.items()})
    write(folder/'result.json',{k:v for k,v in result.items() if k not in ('graph','stages','records')})
    write(folder/'receipt.json',dict(status='COMPLETE',files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

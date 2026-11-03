"""Task-local exact-output Graph optimization; existing modules remain unchanged."""
from time import perf_counter
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from copy import deepcopy
from gpu_graph import OptimizedGraph


@triton.jit
def _greedy_labels(xyz, oid, result, ambiguous, n:tl.constexpr, distance:tl.constexpr, B:tl.constexpr):
    i=tl.arange(0,B)
    active=i<n
    x=tl.load(xyz+3*i,active,0);y=tl.load(xyz+3*i+1,active,0);z=tl.load(xyz+3*i+2,active,0)
    objects=tl.load(oid+i,active,-1)
    labels=tl.full((B,),-1,tl.int32)
    near=tl.full((),0,tl.int32)
    first=tl.min(tl.where(active,i,2147483647),0)
    while first < n:
        qx=tl.load(xyz+3*first);qy=tl.load(xyz+3*first+1);qz=tl.load(xyz+3*first+2)
        obj=tl.load(oid+first)
        dx=x-qx;dy=y-qy;dz=z-qz
        d=libdevice.sqrt(dx*dx+dy*dy+dz*dz)
        pending=active & (labels<0) & (objects==obj)
        near=near | tl.max((pending & (tl.abs(d-distance)<1e-14)).to(tl.int32),0)
        hit=pending & (d<distance)
        labels=tl.where(hit,first,labels)
        first=tl.min(tl.where(active & (labels<0),i,2147483647),0)
    tl.store(result+i,labels,active)
    tl.store(ambiguous,near)


def gpu_deduplicate(records, distance=.003, *, fallback):
    import graph.plan_a_geometry.core as core
    if not records or len(records)>8192:
        return fallback(records,distance)
    ordered=sorted(records,key=lambda x:(x['object_id'],not x.get('structural',False),core.key(x)))
    xyz=np.ascontiguousarray([r['world_xyz'] for r in ordered],dtype=np.float64)
    # A boundary-distance CPU fallback preserves NumPy's strict comparison.
    ids=np.ascontiguousarray([r['object_id'] for r in ordered],dtype=np.int32)
    dev_xyz=torch.as_tensor(xyz,device='cuda');dev_ids=torch.as_tensor(ids,device='cuda')
    result=torch.empty(len(ordered)+1,dtype=torch.int32,device='cuda')
    _greedy_labels[(1,)](dev_xyz,dev_ids,result,result[len(ordered):],len(ordered),distance,B=triton.next_power_of_2(len(ordered)),num_warps=8,enable_fp_fusion=False)
    values=result.cpu().numpy();labels=values[:-1]
    # Verify all decisions close to the strict boundary with the reference routine.
    # Float64 CUDA sqrt and NumPy norm can differ in the last bit.
    if values[-1]:return fallback(records,distance)
    out=[];lookup={}
    for i,(r,representative) in enumerate(zip(ordered,labels)):
        if representative==i:
            item=deepcopy(r);item.setdefault('structural',False);lookup[i]=item;out.append(item)
        else:
            item=lookup[int(representative)]
            item['source_observation_ids'].extend(r['source_observation_ids'])
            item['structural']=item['structural'] or r.get('structural',False)
    duplicates=len(records)-len(out)
    return out,dict(input_points=len(records),unique_points=len(out),duplicates=duplicates,
        duplicate_fraction=duplicates/len(records),representative='stable real measurement, no XYZ/F5 averaging')


def tree_deduplicate(records, distance=.003, *, fallback):
    from scipy.spatial import cKDTree
    import graph.plan_a_geometry.core as core
    if not records:return fallback(records,distance)
    ordered=sorted(records,key=lambda x:(x['object_id'],not x.get('structural',False),core.key(x)))
    xyz=np.ascontiguousarray([r['world_xyz'] for r in ordered],dtype=np.float64)
    if not np.isfinite(xyz).all():return fallback(records,distance)
    tree=cKDTree(xyz);radius=distance+1e-14
    if tree.count_neighbors(tree,radius)>2_000_000:return fallback(records,distance)
    pairs=tree.query_pairs(radius,output_type='ndarray')
    neighbors=[[] for _ in ordered]
    for a,b in pairs:
        if ordered[a]['object_id']==ordered[b]['object_id']:neighbors[b].append(a)
    out=[];lookup={};duplicates=0
    for i,r in enumerate(ordered):
        hits=[a for a in neighbors[i] if a in lookup and np.linalg.norm(xyz[a]-xyz[i])<distance]
        if hits:
            representative=lookup[min(hits)]
            representative['source_observation_ids'].extend(r['source_observation_ids'])
            representative['structural']=representative['structural'] or r.get('structural',False)
            duplicates+=1
        else:
            item=deepcopy(r);item.setdefault('structural',False);lookup[i]=item;out.append(item)
    return out,dict(input_points=len(records),unique_points=len(out),duplicates=duplicates,
        duplicate_fraction=duplicates/len(records),representative='stable real measurement, no XYZ/F5 averaging')


def make_packet_without_unused_edges(camera, masks):
    """Keep all consumed features/validation; omit the unused Raw EDE branch.

    PlanA's observations_from_packet consumes points, feature fields and source
    provenance only; it constructs the final surface edges independently. The
    feature packet's existing 'not_implemented' edge status passes the same
    general receiver/schema and sensor-alignment validations.
    """
    from graph.integration_baseline import vision
    from visual.runtime.rgbd import TrackedRGBDFrame
    masks=np.asarray(masks,bool)
    frame=TrackedRGBDFrame(camera.timestep,camera.camera,camera.rgb,
        camera.depth.astype(np.float32),camera.intrinsic,camera.world_from_camera,
        camera.tcp_pose,{oid:masks[oid] for oid in (0,1)},camera.timestep,camera.timestamp)
    start=perf_counter();evidence=vision.extract_sampling_evidence(frame,estimator='pca5');after_evidence=perf_counter()
    candidates=vision.extract_candidates(evidence,vision.SamplingConfig(),method='S3')
    prepared={};selected={};cases={};geometry=evidence.frame.geometry
    for oid in (0,1):
        obj=candidates[oid];orientation=evidence.pairs[oid].candidates(20)['orientation']
        pca=obj.flat_index[orientation.ravel()[obj.flat_index]]
        prepared[oid]=vision.shape_first.prepare_object(obj.flat_index,obj.points,pca,frame.masks[oid],
            geometry.valid_depth,geometry.overlap_mask,frame.intrinsic)
        selected[oid]=vision.shape_first.select_object(prepared[oid],.5,0,vision.shape_first.SURFACE[oid])
        cases[oid]=dict(saved_flat=obj.flat_index,saved_features=obj.metadata)
    selection,_=vision.shape_first.make_packet(frame.source_frame,prepared,selected,cases)
    selection.update(episode=camera.episode,camera=frame.camera);after_selection=perf_counter()
    packet=vision.complete_vision_packet(selection,evidence,frame);after_feature=perf_counter()
    vision.receive_vision_packet(packet)
    return packet,dict(pca5_evidence_ms=(after_evidence-start)*1000,
        shape_first_ms=(after_selection-after_evidence)*1000,
        feature_ee_ms=(after_feature-after_selection)*1000,component_raw_ede_ms=0.,
        total_offline_graph_ms=(perf_counter()-start)*1000,
        raw_ede_omitted='not consumed by PlanA observations_from_packet')


class FastGraph(OptimizedGraph):
    def __init__(self, variant='fast_v1'):
        super().__init__('cuda_fused_v5')
        self.variant=variant

    def __enter__(self):
        super().__enter__()
        import graph.plan_a_geometry.core as core
        self.patch(core,'make_packet',make_packet_without_unused_edges)
        if self.variant in ('fast_v3_dedup','fast_v4_combined'):
            original=core.deduplicate
            self.patch(core,'deduplicate',lambda records,distance=.003:gpu_deduplicate(records,distance,fallback=original))
        if self.variant in ('fast_v5_tree','fast_v6_tree_exact'):
            original=core.deduplicate
            self.patch(core,'deduplicate',lambda records,distance=.003:tree_deduplicate(records,distance,fallback=original))
        if self.variant in ('fast_v2_pca','fast_v4_combined','fast_v5_tree'):
            import visual.runtime.sampling as sampling
            from visual.runtime.test.pca_triton import fused_normals
            context=self
            def normals_gpu(geometry,object_id,parameters,*,solver='analytic',profile=None):
                if solver!='analytic':raise ValueError('only existing analytic PCA5 path supported')
                key=(id(geometry),'pca_xyz')
                if key not in context.cache:
                    context.cache[key]=(geometry,torch.as_tensor(np.ascontiguousarray(geometry.xyz),device='cuda'))
                xyz=context.cache[key][1]
                support=torch.as_tensor(np.ascontiguousarray(geometry.observed_mask(object_id)),device='cuda')
                normals,valid=fused_normals(xyz,support,parameters)
                return normals.cpu().numpy(),valid.cpu().numpy()
            self.patch(sampling,'local_plane_optimized',normals_gpu)
        return self

"""Frozen-mask stage attribution; no detector, SAM2, policy or source changes.

GT source pruning below is an explicitly privileged posthoc control, never a
production filter. All stage observers must reproduce the saved five tensors.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha, sensor_frame, world_points
from .evaluate import actor_labels
from graph.task10i_stage4a2.evaluation_metrics import correct_source_coverage
from graph.task10i_stage4a2.paired_evaluation import reference_points
from graph.task10i_stage4a2.dinov2_matching_audit import csv_write
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.persistent_geometry.tensor_graph import numpy_graph, FIELDS
from graph.persistent_geometry.tensor_records import numpy_records, budget_tensor, _take_records
from visual.runtime.tensor_packet import numpy_packet
from utils.experiment_logger import ExperimentLogger


def record_arrays(records, tcp):
    rr = numpy_records(records)
    return dict(xyz=world_points(rr, tcp),
        oid=np.asarray([r['object_id'] for r in rr], int),
        camera=np.asarray([CAMERAS.index(r['source_observation_ids'][0][2]) for r in rr], int),
        pixel=np.asarray([r['source_observation_ids'][0][3] for r in rr], int))


def point_metrics(a, oid, segs, ids, reference):
    take = a['oid'] == oid
    labels = actor_labels(segs, ids, oid, a['camera'][take], a['pixel'][take])
    correct = labels == 'correct'
    return dict(points=int(take.sum()), correct=int(correct.sum()),
        precision=float(correct.mean()) if len(correct) else None,
        coverage=correct_source_coverage(a['xyz'][take], correct, reference),
        background=int((labels == 'background').sum()), robot=int((labels == 'robot').sum()),
        wrong_object=int((labels == 'wrong_object').sum()))


def main(capture, inference, runtime, output):
    if read(inference/'inference_complete.json')['status'] != 'COMPLETE':
        raise ValueError('frozen inference required')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    index = entries(capture)
    schema = read(runtime/'normalizer.json')['schema']
    import visual.runtime.cuda_sampling as sampling
    original_shape = sampling.shape_first_cuda
    observed = {}
    active = [None]
    def shape_observer(info):
        result = original_shape(info)
        observed[active[0]] = (info, result)
        return result
    sampling.shape_first_cuda = shape_observer
    stages, details, controls, equivalence = [], [], [], []
    config = dict(inference=str(inference), capture=str(capture), parents=PARENTS, timesteps=list(range(21)),
        sizes=SIZES, frozen_inference_sha256=sha(inference/'inference_complete.json'),
        diagnostic_source_sha256=sha(Path(__file__)), model_forwards=0,
        privileged_control='ORACLE_SOURCE_FILTER: GT-correct deduplicated pool -> existing FPS128; evaluation only')
    try:
        with ExperimentLogger(team='integration', model='task10i_frozen_graph_sampling_diagnosis', seed=42,
                job_type='evaluate', root=output/'experiments', config=config) as log:
            for parent in PARENTS:
                geometries = {size: CurrentGeometry(runtime/'inputs', schema) for size in SIZES}
                try:
                    for t in range(21):
                        # Predictions are already frozen; this process only diagnoses them.
                        ref_cameras, _ = sensor_frame(capture, index, parent, t, 1024, parent)
                        ref_segs, ref_ids = [], []
                        for camera in CAMERAS:
                            e = index[parent, t, 1024, camera]
                            with np.load(capture/e['gt_file']) as z:
                                ref_segs.append(z['actor_segmentation'].copy())
                            ref_ids.append(e['object_ids'])
                        rr = dict(object_ids={k: ref_ids[0][k] for k in ('peg', 'box')})
                        references = [reference_points(ref_cameras, ref_segs, rr, oid)[0] for oid in (0,1)]
                        for size, geo in geometries.items():
                            identity = f'{parent}:native{size}'
                            host, data = sensor_frame(capture, index, parent, t, size, identity)
                            cameras = [tensor_camera(c) for c in host]
                            surfaces = geo.fk.surfaces(data['robot76'])
                            data.update(episode=np.array(identity), timestep=np.int64(t), timestamp=np.float64(t/20),
                                qpos=data['robot76'][:9], robot_link_world=np.stack([s.world_from_link for s in surfaces]))
                            with np.load(inference/parent/str(size)/f't{t:04d}.npz') as z:
                                expected = {k: z[k].copy() for k in z.files}
                            raw = {c: torch.as_tensor(expected['raw_masks'][ci], device='cuda') for ci,c in enumerate(CAMERAS)}
                            cap = {}
                            build = geo._build_prepared
                            represent = geo.system.represent
                            packet_builder = geo.system.tensor_packet_builder
                            def prepare_observer(cs, ds, ss, ms, clean_masks=None, **kw):
                                cap['clean'] = clean_masks
                                return build(cs, ds, ss, ms, clean_masks, **kw)
                            def packet_observer(c, *a, **kw):
                                active[0] = c.camera
                                return packet_builder(c, *a, **kw)
                            def represent_observer(cs, prepared, *a, **kw):
                                result = represent(cs, prepared, *a, **kw)
                                cap.update(c3=prepared['conditions']['C3'], pool=result[0]['SF'], packets=result[3])
                                return result
                            geo._build_prepared = prepare_observer
                            geo.system.represent = represent_observer
                            geo.system.tensor_packet_builder = packet_observer
                            try:
                                graph, records, _ = geo.build(cameras, data, surfaces, lambda: raw)
                            finally:
                                geo._build_prepared = build
                                geo.system.represent = represent
                                geo.system.tensor_packet_builder = packet_builder
                            saved = numpy_graph(graph)
                            equal = all(np.array_equal(getattr(saved,k).numpy(), expected[k]) for k in FIELDS)
                            if not equal:
                                raise ValueError(('observer changed saved graph',parent,t,size))
                            equivalence.append(dict(parent=parent,timestep=t,size=size,five_tensors_equal=True))
                            masks = {'raw': expected['raw_masks'],
                                     'clean': np.stack([v.cpu().numpy() for v in (cap['clean'][c] for c in CAMERAS)]),
                                     'c3': np.stack([v.cpu().numpy() for v in (cap['c3'][c] for c in CAMERAS)])}
                            segs, ids = [], []
                            for camera in CAMERAS:
                                e = index[parent,t,size,camera]
                                with np.load(capture/e['gt_file']) as z:
                                    segs.append(z['actor_segmentation'].copy())
                                ids.append(e['object_ids'])
                            pool = record_arrays(cap['pool'], data['tcp_pose'])
                            final = record_arrays(records, data['tcp_pose'])
                            packets = {c: numpy_packet(cap['packets'][c]) for c in CAMERAS}
                            meta = dict(parent=parent, timestep=t, size=size)
                            for oid,name in enumerate(('peg','box')):
                                ref = references[oid]
                                for stage, array in [('dedup_pool',pool),('final',final)]:
                                    stages.append(dict(meta,object_id=oid,stage=stage,**point_metrics(array,oid,segs,ids,ref)))
                                for stage, mm in masks.items():
                                    points, labels = [], []
                                    for ci,c in enumerate(host):
                                        if oid==1 and ci==1:
                                            continue
                                        mask=mm[ci,oid]&np.isfinite(c.depth)&~(mm[ci,0]&mm[ci,1])
                                        pix=np.flatnonzero(mask);points.append(c.world_points().reshape(-1,3)[pix])
                                        labels.extend(actor_labels(segs,ids,oid,np.full(len(pix),ci),pix))
                                    xyz=np.concatenate(points).reshape(-1,3);labels=np.asarray(labels);good=labels=='correct'
                                    stages.append(dict(meta,object_id=oid,stage=stage,points=len(xyz),correct=int(good.sum()),
                                        precision=float(good.mean()) if len(good) else None,
                                        coverage=correct_source_coverage(xyz,good,ref),
                                        background=int((labels=='background').sum()),robot=int((labels=='robot').sum()),
                                        wrong_object=int((labels=='wrong_object').sum())))
                                selected_xyz=[];selected_labels=[]
                                for ci,c in enumerate(host):
                                    p=packets[c.camera];take=p['object_id']==oid;pix=p['provenance']['source_index'][take]
                                    labels=actor_labels(segs,ids,oid,np.full(len(pix),ci),pix)
                                    xyz=c.world_points().reshape(-1,3)[pix]
                                    selected_xyz.append(xyz);selected_labels.extend(labels)
                                    info,result=observed[c.camera]
                                    observed_n=int((info['projected']['object_id']==oid).sum())
                                    s=result['selected'][oid]
                                    details.append(dict(meta,object_id=oid,camera=c.camera,
                                        input_points=observed_n,pca_candidates=len(result['prepared'][oid]['C']['pca']),
                                        silhouette_candidates=len(result['prepared'][oid]['C']['sil']),
                                        voxel_pca=len(result['prepared'][oid]['R']['pca']),
                                        voxel_silhouette=len(result['prepared'][oid]['R']['sil']),
                                        priority_points=len(s['P']),surface_points=len(s['S']),selected_points=len(pix),
                                        selected_fp=int((labels!='correct').sum()),
                                        priority_fp=int(((labels!='correct')&~p['provenance']['surface_selected'][take]).sum()),
                                        voxel_scale_m=float(sampling.nominal_scale_cuda(
                                            info['projected']['points'][info['projected']['object_id']==oid],
                                            info['intrinsic'])) if observed_n else None))
                                xyz=np.concatenate(selected_xyz).reshape(-1,3);good=np.asarray(selected_labels)=='correct'
                                stages.append(dict(meta,object_id=oid,stage='sf_packet',points=len(xyz),correct=int(good.sum()),
                                    precision=float(good.mean()) if len(good) else None,coverage=correct_source_coverage(xyz,good,ref)))
                                take=final['oid']==oid;pix=final['pixel'][take];cis=final['camera'][take];xyz=final['xyz'][take]
                                labels=actor_labels(segs,ids,oid,cis,pix);fp=labels!='correct'
                                distances=cKDTree(ref).query(xyz[fp])[0] if len(ref) and fp.any() else np.empty(0)
                                controls.append(dict(meta,object_id=oid,control='FP_DISTANCE_ONLY',
                                    fp_count=int(fp.sum()),fp_distance_mm=(distances*1000).tolist(),
                                    far_fp_20mm=int((distances>.02).sum()),far_fp_100mm=int((distances>.1).sum())))
                            # Explicit privileged diagnostic: same pre-cap pool,
                            # prune only GT-wrong representatives, then existing FPS128.
                            correct=np.zeros(len(pool['oid']),bool)
                            for oid in (0,1):
                                take=pool['oid']==oid
                                correct[take]=actor_labels(segs,ids,oid,pool['camera'][take],pool['pixel'][take])=='correct'
                            oracle = budget_tensor(_take_records(cap['pool'],torch.as_tensor(np.flatnonzero(correct),device='cuda')),128)
                            oracle_arrays=record_arrays(oracle,data['tcp_pose'])
                            for oid in (0,1):
                                stages.append(dict(meta,object_id=oid,stage='ORACLE_SOURCE_FILTER_FPS128',
                                    **point_metrics(oracle_arrays,oid,segs,ids,references[oid])))
                            if t%10==0:print('DIAGNOSED',parent,t,size,flush=True)
                finally:
                    for geo in geometries.values():geo.close()
            csv_write(output/'stage_metrics.csv',stages);csv_write(output/'sampling_details.csv',details)
            write(output/'fp_distances.json',controls);write(output/'replay_equivalence.json',equivalence)
            summary=[]
            for size in SIZES:
                for oid in (0,1):
                    for stage in ('raw','clean','c3','sf_packet','dedup_pool','final','ORACLE_SOURCE_FILTER_FPS128'):
                        a=[r for r in stages if r['size']==size and r['object_id']==oid and r['stage']==stage]
                        total=sum(r['points'] for r in a);correct=sum(r['correct'] for r in a)
                        summary.append(dict(size=size,object_id=oid,stage=stage,frames=len(a),points=total,correct=correct,
                            pooled_precision=correct/total if total else None,
                            mean_coverage=float(np.mean([r['coverage'] for r in a if r['coverage'] is not None]))))
            write(output/'summary.json',dict(status='COMPLETE',run_id=log.run_id,stages=summary,
                replayed_graph_frames=len(equivalence),all_saved_five_tensors_equal=True,model_forwards=0,config=config))
            log.log(dict(replayed_graph_frames=len(equivalence),model_forwards=0));log.link_output(output/'summary.json')
    finally:
        sampling.shape_first_cuda=original_shape


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('capture','inference','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.capture,a.inference,a.runtime,a.output)

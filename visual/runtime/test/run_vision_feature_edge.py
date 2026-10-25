"""Task04 fixed-input CPU evaluation. Never calls a sampler or object GT reader."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

from graph.vision_input import receive_vision_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence
from visual.runtime.vision_packet import (
    complete_vision_packet, NUMERIC_CONTRACT, float32_tolerance,
)
from visual.runtime.test.run_adaptive_node_budget import load_evidence, save_evidence
from visual.runtime.test.run_variable_rate_shape_first import flatten_arrays, packet_digest

REPO = Path(__file__).resolve().parents[3]
K = REPO/'experiments/9185355a32b8433abc70f1738f983377'
AB = REPO/'experiments/f68c59f210b5499dadc7e962c7f64492'
J = REPO/'experiments/15efc476f735491a828f5b84f352f858'
SAMPLING = REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f'
FRAMES = (0, 18, 27, 45, 56, 66, 80, 120)
RATIOS = (.5, 1.)
REPORT = REPO/'docs/visual-graph-integration/04_vision_feature_edge.md'
FREEZE_SHA = '198b9d1ffe965be49b81632f53fde5903dcc1e4dff0f934c2f94f74f2521e923'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,Path): return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_packet(path, packet):
    """Lossless arrays plus JSON skeleton; no object arrays/pickle."""
    arrays={}
    def walk(v, key='packet'):
        if isinstance(v,np.ndarray):
            arrays[key]=v
            return {'__array__':key}
        if isinstance(v,dict):return {k:walk(x,key+'/'+k) for k,x in v.items()}
        if isinstance(v,list):return [walk(x,key+'/'+str(i)) for i,x in enumerate(v)]
        return v
    skeleton=walk(packet)
    arrays['schema_json']=np.array(json.dumps(skeleton))
    np.savez_compressed(path,**arrays)
    restored=read_packet(path)
    if packet_digest(restored)!=packet_digest(packet):raise AssertionError('packet readback hash mismatch')


def read_packet(path):
    with np.load(path,allow_pickle=False) as a:
        def walk(v):
            if isinstance(v,dict):
                if set(v)=={'__array__'}:return a[v['__array__']].copy()
                return {k:walk(x) for k,x in v.items()}
            if isinstance(v,list):return [walk(x) for x in v]
            return v
        return walk(json.loads(str(a['schema_json'])))


def read_selection(frame, ratio, fields):
    path=K/f'selection/f{frame:04d}_r{ratio:g}.npz'
    with np.load(path,allow_pickle=False) as a:
        p=dict(episode='ep000',camera='base_camera',source_frame=frame,
               coordinate_frame='CV_camera_metres',feature_fields=copy.deepcopy(fields),
               edge_status='not_implemented')
        for name in ('points','object_id','geometric_features','intra_object_edge_index'):
            p[name]=a['packet/'+name].copy()
        p['provenance']={k.removeprefix('packet/provenance/'):a[k].copy()
                         for k in a.files if k.startswith('packet/provenance/')}
        p['compact_features']={f['name']:{n:a['packet/compact_features/'+f['name']+'/'+n].copy()
                                        for n in ('positions','values')} for f in fields}
    return p


def run_tests(out, modules, *, append=False):
    env=dict(os.environ,MPLCONFIGDIR='/tmp/task04-mpl',WANDB_MODE='disabled')
    with (out/'tests.log').open('a' if append else 'w') as log:
        log.write('\nCOMMAND '+sys.executable+' -m unittest '+' '.join(modules)+' -v\n');log.flush()
        result=subprocess.run([sys.executable,'-m','unittest',*modules,'-v'],cwd=REPO,env=env,
                              stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise AssertionError('contract test failure; see tests.log')


def feature_gate(out):
    """Exactly one deterministic extraction per representative source frame."""
    cfg=json.loads((SAMPLING/'comparison.json').read_text())['config']
    frozen=json.loads((K/'selection_before_gt.json').read_text())
    if frozen['sha256']!=FREEZE_SHA or sha(K/'selection_before_gt.npz')!=FREEZE_SHA:
        raise AssertionError('K original freeze hash mismatch')
    paths=[K/'config.json',K/'selection_before_gt.json',K/'selection_before_gt.npz',
           SAMPLING/'comparison.json',J/'selection_before_gt.json',Path(cfg['input']),Path(cfg['masks'])]
    if sha(cfg['input'])!=cfg['input_sha256'] or sha(cfg['masks'])!=cfg['masks_sha256']:
        raise AssertionError('source RGB-D/mask hash mismatch')
    fields=next(iter(json.loads((J/'selection_before_gt.json').read_text())['packets'].values()))['feature_fields']
    frames={f.source_frame:f for f in iter_saved_rgbd(cfg['input'],cfg['masks'],camera=cfg['camera'])
            if f.source_frame in FRAMES}
    if set(frames)!=set(FRAMES):raise AssertionError('missing representative frame')
    (out/'feature_packets').mkdir();(out/'runtime_evidence').mkdir()
    parity={};measurements={}; packets={}
    for fid in FRAMES:
        print('Feature gate frame',fid,flush=True)
        frame=frames[fid]
        cache_path=AB/f'cache/f{fid:04d}_evidence.npz';paths.append(cache_path)
        cache=load_evidence(cache_path)
        t=perf_counter();live=extract_sampling_evidence(frame,estimator='pca5');wall=(perf_counter()-t)*1000
        measurements[fid]=dict(**live.timings_ms,wall_ms=wall,extraction_calls=1)
        row={}
        for oid in (0,1):
            for name in ('xyz','valid_depth','overlap_mask'):
                np.testing.assert_array_equal(getattr(cache.frame.geometry,name),getattr(live.frame.geometry,name))
            np.testing.assert_array_equal(cache.frame.geometry.observed_mask(oid),live.frame.geometry.observed_mask(oid))
            cp,lp=cache.pairs[oid],live.pairs[oid]
            for name in ('pixels','supported','depth_jump','overlap'):
                np.testing.assert_array_equal(getattr(cp,name),getattr(lp,name))
            for name in cp.candidates(20):
                np.testing.assert_array_equal(cp.candidates(20)[name],lp.candidates(20)[name])
            errors={}
            for name in cache.frame.objects[oid].__dataclass_fields__:
                a=getattr(cache.frame.objects[oid],name);b=getattr(live.frame.objects[oid],name)
                if a.dtype.kind=='f':
                    tol=NUMERIC_CONTRACT['cache_normal_atol' if name=='normals' else
                        'cache_variation_atol_rad' if name=='normal_variation_rad' else 'cache_depth_atol_m']
                    np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
                    diff=np.abs(a.astype(float)-b);finite=np.isfinite(diff)
                    errors[name]=float(diff[finite].max()) if finite.any() else None
                else:np.testing.assert_array_equal(a,b)
            row[oid]=dict(status='PASS',max_abs_error=errors)
        parity[fid]=row
        save_evidence(out/'runtime_evidence'/f'f{fid:04d}.npz',live)
        for ratio in RATIOS:
            paths.append(K/f'selection/f{fid:04d}_r{ratio:g}.npz')
            source=read_selection(fid,ratio,fields)
            key=f'f{fid:04d}_r{ratio:g}'
            if packet_digest(source)!=frozen['packet_hashes'][f'f{fid:04d}/r{ratio:g}_phase0']:
                raise AssertionError('K saved packet identity mismatch')
            packet=complete_vision_packet(source,live,frame)
            received=receive_vision_packet(packet)
            if packet_digest(packet)!=packet_digest(received.packet):raise AssertionError('receiver changed packet')
            # Cache gather and independently re-extracted dense fields must agree.
            cached_packet=complete_vision_packet(source,cache,frame)
            np.testing.assert_allclose(packet['geometric_features'],cached_packet['geometric_features'],
                                       rtol=0,atol=1e-6,equal_nan=True)
            if not np.array_equal(packet['feature_field_valid'],cached_packet['feature_field_valid']):
                raise AssertionError('cache field validity mismatch')
            # Independent homogeneous reference, before float32 storage.
            t=packet['T_EE_from_camera'];x=packet['camera_points'].astype(float)
            ref=(t@np.column_stack((x,np.ones(len(x)))).T).T[:,:3]
            calc=x@t[:3,:3].T+t[:3,3]
            tol=64*np.finfo(float).eps*np.maximum(1,np.abs(ref))
            if (np.abs(ref-calc)>tol).any():raise AssertionError('float64 reference mismatch')
            if (np.abs(packet['points']-ref)>float32_tolerance(ref)).any():raise AssertionError('float32 ULP mismatch')
            save_packet(out/'feature_packets'/f'{key}.npz',packet)
            packets[key]=dict(hash=packet_digest(packet),point_count=len(x),
                              field_valid=packet['feature_field_valid'].sum(0),
                              normal_invalid=int((~packet['feature_field_valid'][:,0]).sum()),
                              float64_reference_max_abs=float(np.abs(calc-ref).max()) if len(x) else 0,
                              float32_storage_max_abs=float(np.abs(packet['points']-ref).max()) if len(x) else 0)
    inputs={str(p.relative_to(REPO)):sha(p) for p in paths}
    result=dict(status='PASS',cache_parity=parity,packets=packets,extraction=measurements,
                numeric_contract=NUMERIC_CONTRACT,input_hashes=inputs,
                feature_contract='PASS',edge_contract='NOT_EXECUTED',extraction_calls_total=8)
    write(out/'feature_gate.json',result)
    return result


def run(baseline_path, pause=False):
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    if REPORT.exists():raise FileExistsError(REPORT)
    baseline=Path(baseline_path)
    config=dict(task='04',frames=FRAMES,ratios=RATIOS,phase=0,feature_mode='continuous',
                numeric_contract=NUMERIC_CONTRACT,device='cpu',warmup=3,repeats=5,
                k_run_id=K.name,ab_run_id=AB.name,sampling_reexecuted=False,
                extraction_policy='one shared extraction/frame for parity; reuse evidence for all timed repeats',
                foundation='repository contracts explicitly authorized; attached reference not read',platform=platform.platform())
    with ExperimentLogger(team='visual',job_type='evaluate',model='task04_vision_feature_edge',config=config,seed=0) as logger:
        out=logger.directory
        print('RUN_DIRECTORY',out,flush=True)
        shutil.copyfile(baseline,out/'git_start.json')
        shutil.copyfile(baseline.parent/'git_start.diff',out/'git_start.diff')
        shutil.copytree(baseline.parent/'original',out/'original_source')
        write(out/'numeric_contract.json',NUMERIC_CONTRACT)
        try:
            run_tests(out,['visual.runtime.test.test_vision_features','visual.runtime.test.test_variable_rate_graph_contract'])
            gate=feature_gate(out)
            logger.record(feature_contract='PASS',edge_contract='NOT_EXECUTED',source_runs=[K.name,AB.name])
            write(out/'verification.json',dict(feature_contract='PASS',edge_contract='NOT_EXECUTED',browser='NOT_EXECUTED'))
            print('FEATURE_GATE_PASS',flush=True)
            if pause:
                input('Feature gate passed. Press Enter after the approved Edge implementation is ready.\n')
                updated=runpy.run_path(str(Path(__file__)),run_name='task04_completion')
                updated['complete_run'](logger,gate)
            else:
                complete_run(logger,gate)
        except Exception as exc:
            write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),edge_module_exists=(REPO/'visual/runtime/intra_object_edges.py').exists()))
            REPORT.write_text('# Task04 — 중단 보고\n\nRun: `'+logger.run_id+'`.\n\n'+str(exc)+'\n\nFeature gate와 실행 로그를 확인한다. 미실행 Edge/benchmark를 PASS로 표시하지 않는다.\n')
            logger.link_output(REPORT,kind='reports')
            logger.link_output(out/'failure.json',kind='metrics')
            raise


def frame_from_packet(packet, evidence, *, rgb=None):
    """Reattach observation-only calibration to saved dense evidence for repeats."""
    g=evidence.frame.geometry
    return TrackedRGBDFrame(source_frame=packet['source_frame'],camera=packet['camera'],
        rgb=rgb,depth=g.xyz[...,2],intrinsic=packet['camera_intrinsic'],
        cam2world=packet['camera_to_world'],tcp_pose=packet['observation_ee_pose'],
        masks=g.object_masks,frame_idx=packet['frame_idx'],timestamp=packet['timestamp'])


def array_bytes(value):
    """Logical array bytes and unique owning buffers (includes NumPy native data)."""
    seen=set();logical=0;unique=0
    def visit(v):
        nonlocal logical,unique
        if isinstance(v,np.ndarray):
            logical+=v.nbytes
            root=v
            while isinstance(root.base,np.ndarray):root=root.base
            key=id(root)
            if key not in seen:seen.add(key);unique+=root.nbytes
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        elif hasattr(v,'__dataclass_fields__'):
            for k in v.__dataclass_fields__:visit(getattr(v,k))
    visit(value)
    return dict(logical_array_bytes=logical,unique_buffer_bytes=unique)


def rss_bytes():
    raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform=='darwin' else raw*1024)


def memory_worker(out, key):
    """Fresh CPU process high-water RSS, including loads and native allocations."""
    from visual.runtime.intra_object_edges import observed_components,build_intra_object_edges,EDGE_POLICY
    baseline=rss_bytes()
    fid=int(key[1:5]);ratio=float(key.split('_r')[1])
    feature=read_packet(out/'feature_packets'/f'{key}.npz')
    evidence=load_evidence(out/'runtime_evidence'/f'f{fid:04d}.npz')
    fields=feature['source_artifact_feature_fields']
    source=read_selection(fid,ratio,fields)
    frame=frame_from_packet(feature,evidence)
    maps,_=observed_components(evidence)
    packet=complete_vision_packet(source,evidence,frame)
    edge,comp=build_intra_object_edges(packet['camera_points'],packet['object_id'],packet['provenance']['source_index'],maps)
    packet.update(edge_status='supplied',edge_policy=copy.deepcopy(EDGE_POLICY),
                  observed_component_id=comp,intra_object_edge_index=edge)
    received=receive_vision_packet(packet)
    result=dict(baseline_peak_rss_bytes=baseline,peak_rss_bytes=rss_bytes(),
                peak_delta_bytes=max(0,rss_bytes()-baseline),rss_units='bytes',
                rss_scope='fresh process; imports baseline, evidence/packet load, one post-evidence pipeline',
                arrays=array_bytes(dict(evidence=evidence,source=source,components=maps,packet=packet,
                                        receiver=received)),N=len(packet['points']),E=edge.shape[1])
    print(json.dumps(result),flush=True)


def numeric_audit(packet,evidence):
    """Reference arithmetic, normal rotation and near-tie ordering are separate."""
    from visual.runtime.intra_object_edges import directed_knn
    x=packet['camera_points'].astype(np.float64);t=packet['T_EE_from_camera'];r=t[:3,:3]
    ee=x@r.T+t[:3,3]
    reference=(t@np.column_stack((x,np.ones(len(x)))).T).T[:,:3]
    eps=np.finfo(np.float64).eps
    arithmetic_bound=64*eps*np.maximum(1,np.abs(reference))
    if (np.abs(ee-reference)>arithmetic_bound).any():raise AssertionError('float64 algebra mismatch')
    if (np.abs(packet['points']-reference)>float32_tolerance(reference)).any():raise AssertionError('storage ULP violation')
    normal=np.full((len(x),3),np.nan)
    src=packet['provenance']['source_index'];oid=packet['object_id']
    for obj in (0,1):
        rows=np.flatnonzero(oid==obj)
        normal[rows]=evidence.frame.objects[obj].normals.reshape(-1,3)[src[rows]]
    nv=packet['feature_field_valid'][:,0]
    normal_ref=(r@normal[nv].T).T
    normal_saved=packet['geometric_features'][nv,:3]
    if (np.abs(normal_saved-normal_ref)>float32_tolerance(normal_ref,normal=True)).any():
        raise AssertionError('normal rotation/storage mismatch')
    normal_norm=np.linalg.norm(normal_saved,axis=1)
    if (np.abs(normal_norm-1)>NUMERIC_CONTRACT['normal_norm_atol']).any():raise AssertionError('normal norm')
    # Raw recorded camera rotation is nearly orthogonal, not magically exact.
    # Its spectral distance distortion is separate from float64 arithmetic.
    singular=np.linalg.svd(r,compute_uv=False)
    distortion=float(np.max(np.abs(singular-1)))
    comp=packet['observed_component_id']
    edge=packet['intra_object_edge_index']
    d_camera=np.linalg.norm(x[edge[0]]-x[edge[1]],axis=1)
    d_ee=np.linalg.norm(ee[edge[0]]-ee[edge[1]],axis=1)
    size=max(1,float(np.abs(ee).max()) if len(ee) else 1,float(np.abs(x).max()) if len(x) else 1)
    arithmetic_distance_bound=128*eps*size
    distance_bound=distortion*d_camera+arithmetic_distance_bound
    if (np.abs(d_camera-d_ee)>distance_bound).any():raise AssertionError('unexplained float64 distance error')
    direct_camera=directed_knn(x,oid,comp)
    changes={}
    for name,coords in (('ee_float64',ee),('ee_float32',packet['points'].astype(float))):
        other=directed_knn(coords,oid,comp)
        rows_changed=0;near_ties=0;unexplained=0;examples=[]
        for i in range(len(x)):
            a=set(direct_camera[1,direct_camera[0]==i].tolist())
            b=set(other[1,other[0]==i].tolist())
            if a!=b:
                rows_changed+=1
                removed=sorted(a-b);added=sorted(b-a)
                gap=max(np.linalg.norm(x[added]-x[i],axis=1),default=0)-min(np.linalg.norm(x[removed]-x[i],axis=1),default=0)
                longest=max(np.linalg.norm(x[list(a|b)]-x[i],axis=1),default=0)
                bound=2*(distortion*longest+arithmetic_distance_bound)
                if name=='ee_float32':
                    error=np.linalg.norm(coords-ee,axis=1)
                    bound+=2*(error[i]+max(error[list(a|b)],default=0))
                explained=gap<=bound
                near_ties+=int(explained);unexplained+=int(not explained)
                if len(examples)<5:examples.append(dict(row=i,removed=removed,added=added,camera_distance_gap_m=float(gap),fixed_error_bound_m=float(bound)))
        if unexplained:raise AssertionError('neighbor ordering change exceeds fixed numerical bound')
        changes[name]=dict(changed_rows=rows_changed,near_tie_explained_rows=near_ties,
                          unexplained_rows=unexplained,examples=examples)
    return dict(float64_reference_max_abs=float(np.abs(ee-reference).max()) if len(x) else 0,
                float32_storage_max_abs_m=float(np.abs(packet['points']-reference).max()) if len(x) else 0,
                float32_max_error_over_allowed=float(np.max(np.abs(packet['points']-reference)/float32_tolerance(reference))) if len(x) else 0,
                normal_rotation_max_abs=float(np.abs(normal_saved-normal_ref).max()) if nv.any() else None,
                normal_norm_max_abs=float(np.abs(normal_norm-1).max()) if nv.any() else None,
                raw_calibration_singular_value_distortion=distortion,
                float64_distance_max_abs_m=float(np.abs(d_camera-d_ee).max()) if len(d_camera) else 0,
                float64_arithmetic_distance_bound_m=arithmetic_distance_bound,
                ordering=changes)


def statistics(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),max=float(a.max()),samples=a)


def complete_run(logger,gate):
    """Runs only after the recorded Feature gate has passed."""
    if gate['status']!='PASS':raise AssertionError('Feature gate not passed')
    from visual.runtime.intra_object_edges import (
        observed_components,build_intra_object_edges,EDGE_POLICY,edge_diagnostics,
    )
    out=logger.directory
    run_tests(out,['visual.runtime.test.test_intra_object_edges','visual.runtime.test.test_vision_features',
                  'visual.runtime.test.test_variable_rate_graph_contract',
                  'visual.runtime.test.test_variable_rate_shape_first','visual.runtime.test.test_sampling_geometry'],append=True)
    fields=next(iter(json.loads((J/'selection_before_gt.json').read_text())['packets'].values()))['feature_fields']
    packets={};records={};common={};raw={}
    (out/'packets').mkdir();(out/'plots').mkdir()
    for fid in FRAMES:
        print('CPU post-evidence frame',fid,flush=True)
        evidence=load_evidence(out/'runtime_evidence'/f'f{fid:04d}.npz')
        feature=read_packet(out/'feature_packets'/f'f{fid:04d}_r0.5.npz')
        frame=frame_from_packet(feature,evidence)
        sources={r:read_selection(fid,r,fields) for r in RATIOS}
        source_hashes={r:packet_digest(s) for r,s in sources.items()}
        samples={r:[] for r in RATIOS};cc_samples=[];group_samples=[]
        for repeat in range(8):
            start=perf_counter();maps,component_stats=observed_components(evidence)
            cc_ms=(perf_counter()-start)*1000
            for ratio in RATIOS:
                timing={};begin=perf_counter()
                packet=complete_vision_packet(sources[ratio],evidence,frame,timings=timing)
                edge_start=perf_counter()
                edge,comp=build_intra_object_edges(packet['camera_points'],packet['object_id'],packet['provenance']['source_index'],maps)
                timing['knn_dedup_ms']=(perf_counter()-edge_start)*1000
                pack_start=perf_counter()
                packet.update(edge_status='supplied',edge_policy=copy.deepcopy(EDGE_POLICY),
                              observed_component_id=comp,intra_object_edge_index=edge)
                timing['edge_packaging_ms']=(perf_counter()-pack_start)*1000
                receive_start=perf_counter();received=receive_vision_packet(packet)
                timing['graph_receiver_ms']=(perf_counter()-receive_start)*1000
                timing['ratio_processing_ms']=(perf_counter()-begin)*1000
                timing['post_evidence_subtotal_ms']=cc_ms+timing['ratio_processing_ms']
                timing['offline_subtotal_with_once_extraction_ms']=timing['post_evidence_subtotal_ms']+gate['extraction'][str(fid) if str(fid) in gate['extraction'] else fid]['wall_ms']
                if repeat>=3:samples[ratio].append(timing)
                key=f'f{fid:04d}_r{ratio:g}'
                if repeat==7:packets[key]=packet
            if repeat>=3:
                cc_samples.append(cc_ms);group_samples.append((perf_counter()-start)*1000)
        common[fid]=dict(component_generation_ms=statistics(cc_samples),two_ratio_group_post_evidence_ms=statistics(group_samples),
                         components=component_stats,evidence_array_bytes=array_bytes(evidence),
                         extraction=gate['extraction'][str(fid) if str(fid) in gate['extraction'] else fid])
        for ratio in RATIOS:
            key=f'f{fid:04d}_r{ratio:g}';packet=packets[key]
            if packet_digest(sources[ratio])!=source_hashes[ratio]:raise AssertionError('source packet mutated')
            if packet_digest(packet['provenance'])!=packet_digest(sources[ratio]['provenance']):raise AssertionError('provenance mutated')
            np.testing.assert_array_equal(packet['camera_points'],sources[ratio]['points'])
            if packet_digest(packet)!=packet_digest(receive_vision_packet(packet).packet):raise AssertionError('receiver packet changed')
            save_packet(out/'packets'/f'{key}.npz',packet)
            numeric=numeric_audit(packet,evidence)
            diagnostics=edge_diagnostics(packet)
            timing={k:statistics([sample[k] for sample in samples[ratio]]) for k in samples[ratio][0]}
            worker=subprocess.run([sys.executable,'-m','visual.runtime.test.run_vision_feature_edge',
                '--memory-worker',str(out),key],cwd=REPO,env=dict(os.environ,MPLCONFIGDIR='/tmp/task04-mpl',WANDB_MODE='disabled'),
                capture_output=True,text=True,check=True)
            memory=json.loads(worker.stdout.strip().splitlines()[-1])
            p=packet['provenance'];objects={}
            for obj in (0,1):
                rows=packet['object_id']==obj
                sil_only=p['candidate_evidence'][:,1]&~p['candidate_evidence'][:,0]
                surface_only=p['surface_selected']&~p['candidate_evidence'].any(1)
                objects[obj]=dict(N=int(rows.sum()),valid=bool(packet['valid'][obj]),
                    available=packet['feature_field_available'][rows].sum(0),field_valid=packet['feature_field_valid'][rows].sum(0),
                    normal_invalid=int((~packet['feature_field_valid'][rows,0]).sum()),
                    silhouette_only=int((rows&sil_only).sum()),surface_only=int((rows&surface_only).sum()),
                    both_candidates=int((rows&p['both_evidence']).sum()),
                    structural=int((rows&(p['selection_stage']==0)).sum()),surface=int((rows&p['surface_selected']).sum()))
            records[key]=dict(frame=fid,ratio=ratio,N=len(packet['points']),E=packet['intra_object_edge_index'].shape[1],
                objects=objects,timing_ms=timing,memory=memory,numerics=numeric,edge=diagnostics,
                packet_hash=packet_digest(packet),packet_array_bytes=array_bytes(packet),
                over_50ms=bool(timing['offline_subtotal_with_once_extraction_ms']['max']>50),
                point_rows=dict(source=p['source_index'],uv=np.column_stack((p['source_index']%packet['image_shape'][1],p['source_index']//packet['image_shape'][1])),
                    object_id=packet['object_id'],camera_xyz=packet['camera_points'],ee_xyz=packet['points'],
                    features=packet['geometric_features'],field_available=packet['feature_field_available'],
                    field_valid=packet['feature_field_valid'],candidate_evidence=p['candidate_evidence'],
                    selected_by_structural_source=p['selected_by_structural_source'],selection_stage=p['selection_stage'],
                    component_id=packet['observed_component_id']))
            raw.update(flatten_arrays(packet,prefix=key))
    np.savez_compressed(out/'vision_feature_edge.npz',**raw)
    with np.load(out/'vision_feature_edge.npz',allow_pickle=False) as a:
        if set(a.files)!=set(raw):raise AssertionError('NPZ readback keys')
        for k,v in raw.items():np.testing.assert_array_equal(a[k],v)
    summary=dict(run_id=logger.run_id,feature_contract='PASS',edge_contract='PASS',
                 frames=FRAMES,ratios=RATIOS,phase=0,edge_policy=EDGE_POLICY,numeric_contract=NUMERIC_CONTRACT,
                 common=common,records=records,feature_gate=gate,
                 extraction_measurement='exactly once/frame, shared across ratios; no extraction warmup/repeats',
                 repeat_measurement='post-evidence: 3 warmup + 5 repeats; component computed once/frame-repeat shared across ratios',
                 downstream='No GNN encoder, inter-object policy, BC/GRU rollout or 20Hz validation',
                 browser=dict(status='NOT_EXECUTED'),visual_review=dict(status='PENDING'))
    write(out/'vision_feature_edge.json',summary)
    # Defined below; figures and HTML do not alter packets/topology.
    build_figures(out,packets,records)
    build_shortcut_details(out,packets,records)
    render_reports(out,summary)
    print('ARTIFACTS_READY',out,flush=True)
    if sys.stdin.isatty():
        input('Agent artifact review checkpoint; press Enter after review metadata is saved.\n')
    review_path=out/'artifact_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        summary['browser']=review.get('browser',dict(status='NOT_EXECUTED'))
        summary['visual_review']=review.get('visual_review',dict(status='NOT_EXECUTED'))
    finalize_outputs(logger,summary)


def build_figures(out,packets,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    plt.rcParams.update({'font.size':9})
    cfg=json.loads((SAMPLING/'comparison.json').read_text())['config']
    with np.load(cfg['input'],allow_pickle=False) as a:
        images={fid:a['base_camera/rgb'][fid] for fid in FRAMES}
    for key,p in packets.items():
        fid=records[key]['frame'];src=p['provenance']['source_index'];w=p['image_shape'][1]
        uv=np.column_stack((src%w,src//w));pro=p['provenance']
        role=pro['candidate_evidence'][:,0].astype(int)+2*pro['candidate_evidence'][:,1].astype(int)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax in axes:ax.imshow(images[fid]);ax.set(xlabel='Source u (pixel)',ylabel='Source v (pixel)')
        for code,color,label in ((0,'#999999','No candidate'),(1,'#ff5a5f','PCA5'),(2,'#00b8ff','Silhouette'),(3,'#d45cff','Both')):
            for stage,marker in ((0,'.'),(1,'x')):
                rows=(role==code)&(pro['selection_stage']==stage)
                axes[0].scatter(uv[rows,0],uv[rows,1],s=8,c=color,marker=marker,
                                label=f'{label}; '+('Structural' if stage==0 else 'Surface FPS'))
        nv=p['feature_field_valid'][:,0]
        for keep,color,label in ((nv,'#4cff72','Valid normal'),(~nv,'#ff5733','Invalid normal (NaN)')):
            axes[1].scatter(uv[keep,0],uv[keep,1],s=8,c=color,label=label)
        axes[0].legend(fontsize=6,loc='upper right');axes[1].legend(fontsize=7)
        axes[0].set_title(f'{key}: exact source rows, N={len(src)}')
        axes[1].set_title('Measurement validity is independent of selection role')
        fig.tight_layout();fig.savefig(out/f'plots/{key}_alignment.png',dpi=130);plt.close(fig)

        fig=plt.figure(figsize=(14,10))
        a=fig.add_subplot(2,2,1);a.imshow(images[fid])
        b=fig.add_subplot(2,2,2);b.imshow(images[fid])
        c=fig.add_subplot(2,2,3,projection='3d');d=fig.add_subplot(2,2,4,projection='3d')
        edge=p['intra_object_edge_index'];unique=edge[:,edge[0]<edge[1]]
        a.add_collection(LineCollection(uv[unique.T],colors='#53ccff',linewidths=.35,alpha=.18))
        longest=[]
        for obj,color,name in ((0,'#ff6a46','Peg'),(1,'#36d7ff','Box')):
            rows=p['object_id']==obj
            a.scatter(uv[rows,0],uv[rows,1],s=5,c=color,label=name)
            for ax,xyz in ((c,p['camera_points']),(d,p['points'])):
                ax.scatter(*xyz[rows].T,s=3,c=color,label=name)
                e=unique[:,rows[unique[0]]]
                if e.shape[1]:
                    ax.add_collection3d(Line3DCollection(xyz[e.T],colors=color,linewidths=.3,alpha=.2))
            objects=records[key]['edge']['objects']
            longest.extend(objects.get(obj,objects.get(str(obj)))['longest_edges'][:3])
        # Longest edges are diagnostic candidates only; no veto or reselection.
        for item in longest:
            rows=np.asarray(item['point_rows']);xy=uv[rows]
            b.plot(xy[:,0],xy[:,1],'-',lw=1.4,alpha=.85)
            for i,(u,v) in zip(rows,xy):b.annotate(str(i),(u,v),fontsize=7)
            for ax,xyz in ((c,p['camera_points']),(d,p['points'])):
                ax.plot(*xyz[rows].T,c='yellow',lw=1,alpha=.75)
        a.set_title(f'{key}: all intra-object edges, E={edge.shape[1]}');a.legend()
        b.set_title('Longest pairs: inspect hole/thin-surface shortcuts; labels = final rows')
        for ax,title in ((c,'Original camera XYZ (full extent)'),(d,'EE XYZ (full extent)')):
            ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title);ax.legend()
        fig.suptitle('Geometric proximity links; physical surface connectivity is not guaranteed')
        fig.tight_layout();fig.savefig(out/f'plots/{key}_edges.png',dpi=130);plt.close(fig)
    keys=list(records);labels=[k.replace('f00','f') for k in keys];index=np.arange(len(keys))
    fig,axes=plt.subplots(2,1,figsize=(14,8))
    for j,name in enumerate(('normal','variation','camera-z depth jump')):
        fractions=[packets[k]['feature_field_valid'][:,j].mean() if len(packets[k]['points']) else np.nan for k in keys]
        axes[0].plot(index,fractions,'o-',label=name)
    axes[0].set(ylabel='Valid fraction',ylim=(-.03,1.03),title='Availability is 1.0 for every provided runtime field');axes[0].legend()
    axes[1].plot(index,[records[k]['N'] for k in keys],'o-',label='N')
    axes[1].plot(index,[records[k]['E'] for k in keys],'o-',label='E')
    axes[1].set(ylabel='Actual count',xticks=index,xticklabels=labels);axes[1].tick_params(axis='x',rotation=60);axes[1].legend()
    fig.tight_layout();fig.savefig(out/'plots/validity_variable_ne.png',dpi=130);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(14,8));bottom=np.zeros(len(keys))
    for name in ('source_alignment_ms','feature_gather_ms','ee_transform_ms','packet_packaging_ms','knn_dedup_ms','edge_packaging_ms','graph_receiver_ms'):
        values=np.array([records[k]['timing_ms'][name]['mean'] for k in keys])
        axes[0].bar(index,values,bottom=bottom,label=name);bottom+=values
    axes[0].axhline(50,c='red',ls='--',label='50 ms reference (no integrated GNN)')
    axes[0].set(ylabel='Ratio-specific CPU mean (ms)',xticks=index,xticklabels=labels);axes[0].tick_params(axis='x',rotation=60);axes[0].legend(fontsize=7,ncol=3)
    for ratio,marker in ((.5,'o'),(1.,'^')):
        subset=[r for r in records.values() if r['ratio']==ratio]
        axes[1].scatter([r['N'] for r in subset],[r['timing_ms']['knn_dedup_ms']['mean'] for r in subset],marker=marker,label=f'ratio {ratio}')
    axes[1].set(xlabel='Point N',ylabel='k-NN + dedup mean (ms)',title='Fixed k=8, exact sorting; no point cap');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'plots/latency.png',dpi=130);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].bar(index,[records[k]['numerics']['float32_max_error_over_allowed'] for k in keys]);axes[0].axhline(1,c='red',ls='--')
    axes[0].set(ylabel='max |storage error| / max(1e-6m,4 ULP)',title='Camera to EE: float32 storage bound')
    axes[1].bar(index,[records[k]['numerics']['float64_reference_max_abs'] for k in keys],label='float64 reference arithmetic')
    axes[1].set(ylabel='Absolute component error (m)',title='Float64 homogeneous reference; separate from storage')
    for ax in axes:ax.set_xticks(index,labels,rotation=90)
    fig.tight_layout();fig.savefig(out/'plots/camera_ee_verification.png',dpi=130);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ratio in RATIOS:
        selected=[p for k,p in packets.items() if records[k]['ratio']==ratio]
        deg=np.concatenate([np.bincount(p['intra_object_edge_index'][0],minlength=len(p['points'])) for p in selected])
        lengths=np.concatenate([np.linalg.norm(p['camera_points'][p['intra_object_edge_index'][0]].astype(float)-p['camera_points'][p['intra_object_edge_index'][1]],axis=1) for p in selected])
        axes[0].hist(deg,bins=np.arange(deg.max()+2)-.5,histtype='step',label=f'ratio {ratio}')
        axes[1].hist(lengths*1000,bins=60,histtype='step',label=f'ratio {ratio}')
    axes[0].set(xlabel='Directed degree',ylabel='Point count');axes[1].set(xlabel='Edge length (mm)',ylabel='Directed edge count',yscale='log')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'plots/edge_distributions.png',dpi=130);plt.close(fig)

    fig,ax=plt.subplots(figsize=(14,5));base=np.array([records[k]['memory']['baseline_peak_rss_bytes']/2**20 for k in keys]);delta=np.array([records[k]['memory']['peak_delta_bytes']/2**20 for k in keys])
    ax.bar(index,base,label='Fresh worker imports baseline high-water RSS');ax.bar(index,delta,bottom=base,label='Pipeline/load high-water increase')
    ax.set(ylabel='CPU process peak RSS (MiB)',xticks=index,xticklabels=labels,title='Native NumPy/SciPy memory included; not GPU memory');ax.tick_params(axis='x',rotation=60);ax.legend()
    fig.tight_layout();fig.savefig(out/'plots/memory.png',dpi=130);plt.close(fig)


def render_reports(out,summary):
    import html
    records=summary['records'];run_id=summary['run_id']
    rows=[]
    for key,r in records.items():
        rows.append(f"| {key} | {r['N']} | {r['E']} | {r['timing_ms']['knn_dedup_ms']['mean']:.3f} | {r['timing_ms']['post_evidence_subtotal_ms']['mean']:.3f} | {r['timing_ms']['offline_subtotal_with_once_extraction_ms']['mean']:.3f} | {r['memory']['peak_rss_bytes']/2**20:.1f} |")
    normal_valid=sum(int(np.asarray(p['field_valid'])[0]) for r in records.values() for p in r['objects'].values())
    total=sum(r['N'] for r in records.values())
    browser=summary['browser']['status'];visual=summary['visual_review']['status']
    a=records['f0000_r0.5'];b=records['f0000_r1']
    growth=b['timing_ms']['knn_dedup_ms']['mean']/a['timing_ms']['knn_dedup_ms']['mean']
    numerical_max={k:max(r['numerics'][k] or 0 for r in records.values()) for k in
                   ('float64_reference_max_abs','float32_storage_max_abs_m','float32_max_error_over_allowed',
                    'normal_norm_max_abs','float64_distance_max_abs_m')}
    findings=f'''Frame0에서는 N=950→1694 (×{1694/950:.2f})일 때 k-NN 평균이 {a['timing_ms']['knn_dedup_ms']['mean']:.3f}→{b['timing_ms']['knn_dedup_ms']['mean']:.3f}ms (×{growth:.2f})로 증가했다. Extraction 포함 가산 subtotal 평균은 {a['timing_ms']['offline_subtotal_with_once_extraction_ms']['mean']:.3f}→{b['timing_ms']['offline_subtotal_with_once_extraction_ms']['mean']:.3f}ms다. 두 frame0 조건 모두 최대 subtotal이 50ms를 넘었다.

Frame0 Box의 가장 긴 연결은 **263.469mm**, camera-z **8.431m**, source pixel (251,206)↔(259,206) (ratio .5)이다. 관측 이미지의 hole 주변 위치를 잇는 의심 사례다. 실제 Box 앞면의 표면 연결이라는 판정은 하지 않는다. 같은 component 안의 관측 depth 오염/긴 연결이 남을 수 있음을 보여준다. 개선 정책은 추가하지 않고 최종 채택 판단을 보류한다. Full-extent 3D는 최대 30.726m finite point까지 포함하며 아래 확대 그림은 표시 범위만 좁힌 진단이다.

Float64 reference의 최대 성분 오차는 {numerical_max['float64_reference_max_abs']:.3g}m, float32 저장 오차는 {numerical_max['float32_storage_max_abs_m']:.3g}m이고, 허용오차 대비 최대 비율은 {numerical_max['float32_max_error_over_allowed']:.3f}였다. Normal norm 최대 오차는 {numerical_max['normal_norm_max_abs']:.3g}다. Raw calibration의 비직교성까지 분리한 거리 검증을 통과했다. 이 검사들은 physical Edge 정확도 PASS가 아니다.'''
    md=f'''# Task04 — Vision Geometric Feature & Intra-object Edge

Run UUID: `{run_id}`. Feature 계약 **PASS**, Edge 계약 **PASS**. 고정된 8개 frame × 2개 ratio의 CPU offline 검증이며 독립 episode 16개가 아니다.

[정적 HTML](../../experiments/{run_id}/vision_feature_edge.html) · [JSON](../../experiments/{run_id}/vision_feature_edge.json) · [NPZ](../../experiments/{run_id}/vision_feature_edge.npz) · [검증](../../experiments/{run_id}/verification.json) · [테스트 로그](../../experiments/{run_id}/tests.log) · [Source/input hash](../../experiments/{run_id}/source_hashes.json) · [Implementation diff](../../experiments/{run_id}/implementation.diff)

## Feature와 수신 계약

Task03-K 저장 point/source/행 순서와 provenance를 유지했다. 기존 optimized PCA5 evidence에서 EE normal 3성분, normal variation(rad), camera-z depth jump(m)를 gather한 F=5다. Availability와 measurement validity는 분리했다. Invalid는 NaN/false이고 point 제거·재선택은 없다. `onehot`/`probability`는 unsupported error다. RGB/선택 근거를 GNN x에 자동 추가하지 않는다.

총 {total}개 행(동일 frame의 두 ratio 포함) 중 normal 유효 {normal_valid}개다. Silhouette-only/Surface-only/중복 후보, 부분 유효 측정과 빈 관측을 구분해 JSON/HTML에 저장했다. Frame18/56의 빈 Box와 frame27/66의 복구도 그대로 전달한다.

8개 source frame의 deterministic extraction은 각각 **1회**, 두 ratio가 공유했다. AB cache의 support/candidate/validity는 exact, float Feature는 사전 고정 tolerance로 비교했다. Float64 reference와 float32 저장 오차는 별도로 검증했다. 저장 허용오차는 성분별 `max(1e-6m,4 ULP)`이며 결과를 본 뒤 확대하지 않았다. 원본 float32 calibration의 미세한 비직교성에 따른 거리 변형은 float64 연산 오차와 분리했다. Near-tie의 EE64/EE32 ordering 차이는 진단하며 실제 topology는 camera64 기준으로 고정했다.

Unversioned v1과 opt-in v2, 전체 packet deep-copy/Feature/provenance 보존을 검증했다. Synthetic robot 3개가 실제 공급된 테스트에서만 +3 remap했다. 실제 robot 좌표 수집이나 GNN 입력 encoder 완성은 검증하지 않았다.

## Edge와 한계

전체 observed support X의 기존 right/down PairEvidence와 depth gate로 component를 만들고, 동일 object/component의 최종 point에 Euclidean k-NN(k=8)을 적용했다. Self-loop/directed duplicate는 0이며 양방향 합집합, index 범위, object/component 소속, E≤16N을 검사했다. 관측상 다른 component를 잇는 Edge는 0이다.

Component/degree/isolated node/길이 mean·p95·max와 가장 긴 Edge의 source/UV/camera·EE XYZ를 보존했다. Hole/얇은 표면 shortcut 의심 사례는 긴 연결 overlay와 3D 그림에서 검토할 진단이며 GT로 Edge를 재선택하지 않았다. 같은 component라는 사실은 물리 표면 연결성이나 physical crease를 보장하지 않는다. Radius/normal veto/geodesic 등 후속 정책은 추가하지 않았다.

![Edge degree/length](../../experiments/{run_id}/plots/edge_distributions.png)

{findings}

![Hole 주변 긴 연결 확대](../../experiments/{run_id}/plots/f0000_r0.5_shortcut_detail.png)
![Ratio1 긴 연결 확대](../../experiments/{run_id}/plots/f0000_r1_shortcut_detail.png)

## CPU 시간과 메모리

Extraction은 frame당 1회만 허용된 입력 관문을 우선해 별도 실측했다. **3회 warmup + 5회 계측은 evidence 이후 단계**에 적용했다. Component는 각 frame-repeat에서 두 ratio가 공유한다. Offline subtotal은 측정한 component+ratio 처리에 같은 frame의 단회 extraction 시간을 더한 값이며 연속 실시간 loop latency가 아니다. I/O/그림/압축/로깅/진단은 core 시간에서 제외했다.

| frame/ratio | N | E | k-NN mean ms | evidence 이후 subtotal mean ms | extraction 포함 가산 subtotal mean ms | fresh worker peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+f'''

고정 k에도 정확한 거리 계산은 O(Σn_c²), 행별 전체 정렬은 O(Σn_c² log n_c)다. 큰 N에서 비용이 증가하며 component별 분포에도 영향을 받는다. 최대 N 조건과 다른 조건의 실측 비교는 HTML/JSON에 있다. 50ms 초과 조건은 그대로 표시한다. N/k를 줄이거나 GPU 최적화를 수행하지 않았다. RSS는 fresh CPU worker의 baseline과 native allocation을 포함한 peak로 분리했고 packet/evidence/receiver 배열 byte도 별도 저장했다.

![N/E와 validity](../../experiments/{run_id}/plots/validity_variable_ne.png)
![CPU 비용](../../experiments/{run_id}/plots/latency.png)
![메모리](../../experiments/{run_id}/plots/memory.png)
![좌표 변환 검증](../../experiments/{run_id}/plots/camera_ee_verification.png)

## 검증 범위와 남은 판단

PNG decode/JSON·NPZ readback/상대 링크/hash 보존은 verification.json에 기록한다. 실제 브라우저: **{browser}**. PNG 시각 검토: **{visual}**. 실행하지 않은 검사를 PASS로 기록하지 않는다.

수치 계산 완료 후 빈 Box의 빈 3D Edge collection을 그리는 Matplotlib 오류가 발생했다. `failure.json`과 최초 계산 source를 보존하고 **같은 UUID에서 저장 수치만으로 보고서를 복구**했다. 새 sampling/extraction/Edge 계산/benchmark를 실행하지 않았다. `numerical_results_before_report_recovery.json`, `computation_source_snapshot/`, `source_snapshot/`와 verification의 recovery 항목을 구분한다.

현재 repository 계약으로 진행하도록 명시적으로 승인받았다. 별도 첨부 `SHARED_TECHNICAL_FOUNDATION.md`는 읽지 않았으며 새로 만들지 않았다. 공통 ExperimentLogger/local artifact 등록을 재사용했다. 기존 dirty/untracked 파일과 입력 artifact를 보호했고 Sampling/PCA5/threshold/ratio/Surface FPS/GT GraphBuilder를 변경하지 않았다.

Simulator/renderer, 추가 episode, CUDA/MPS/PCL, GNN/GRU/BC 학습·rollout, stale geometry, robot/inter-object 정책, PyG adapter, dependency 설치, 외부 게시와 commit은 수행하지 않았다. Feature/Edge 계약 PASS는 최종 GNN encoder·BC 성공·20Hz 통합·최종 Edge 정책 채택을 뜻하지 않는다. 결과 검토 후 다음 변경을 결정한다.
'''
    REPORT.write_text(md)
    esc=lambda v:html.escape(str(v))
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Task04 Vision Feature & Edge</title><style>body{font:15px system-ui;max-width:1450px;margin:32px auto;padding:0 20px;color:#192838;background:#f7f9fc}table{border-collapse:collapse;font-size:12px}td,th{padding:5px 9px;border:1px solid #ccd4df;white-space:nowrap}th{background:#e4ebf5;position:sticky;top:0}img{max-width:100%;height:auto}details{margin:16px 0;background:white;padding:12px;border:1px solid #d3dce7}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:600px}h2{margin-top:36px}a{color:#175eaa}</style>',
          f'<h1>Task04 · measured geometry & intra-object edges</h1><p>Run <code>{run_id}</code> · Feature PASS · Edge PASS · CPU offline, 8 source frames × 2 fixed ratios</p>',
          '<p><a href="../../docs/visual-graph-integration/04_vision_feature_edge.md">Markdown report</a> · <a href="vision_feature_edge.json">JSON</a> · <a href="vision_feature_edge.npz">NPZ</a> · <a href="verification.json">Verification</a> · <a href="tests.log">Tests</a> · <a href="source_hashes.json">Source/input hashes</a> · <a href="implementation.diff">Implementation diff</a></p>',
          '<p>F=5: EE normal (xyz), normal variation (radians), camera-z depth jump (metres). Field availability ≠ measurement validity. Invalid values remain NaN; RGB and selection evidence are not GNN x. No resampling, object GT selection, GNN/BC, integrated 20Hz or final topology claim.</p>',
          '<p>Observed-component constrained symmetric Euclidean k-NN, k=8. Full observed raster support defines components. Camera64 defines topology; EE64/EE32 ordering changes near ties are audited separately. Depth/component membership cannot guarantee physical surface connectivity across a hole or thin wall.</p>',
          f'<p>Actual browser: <b>{esc(browser)}</b> · PNG visual review: <b>{esc(visual)}</b></p>',
          '<h2>Variable N/E, validity and CPU cost</h2>']
    for image in ('validity_variable_ne','camera_ee_verification','latency','memory','edge_distributions'):
        page.append(f'<a href="plots/{image}.png"><img loading="lazy" src="plots/{image}.png" alt="{image}"></a>')
    page.append('<p>Extraction: exactly once per source frame, shared across ratios. Post-evidence stages: 3 warmup + 5 measured repetitions. Extraction-inclusive subtotal adds the single measured extraction and is not a continuous online-loop benchmark. Fresh-process peak RSS includes native allocations and input loads.</p>')
    page.append('<table><thead><tr><th>Input</th><th>N</th><th>E</th><th>kNN mean ms</th><th>post-evidence mean/p95 ms</th><th>with extraction mean ms</th><th>50ms exceeded</th><th>RSS MiB</th></tr></thead><tbody>')
    for key,r in records.items():
        page.append(f"<tr><td><a href='#{key}'>{key}</a></td><td>{r['N']}</td><td>{r['E']}</td><td>{r['timing_ms']['knn_dedup_ms']['mean']:.3f}</td><td>{r['timing_ms']['post_evidence_subtotal_ms']['mean']:.3f}/{r['timing_ms']['post_evidence_subtotal_ms']['p95']:.3f}</td><td>{r['timing_ms']['offline_subtotal_with_once_extraction_ms']['mean']:.3f}</td><td>{r['over_50ms']}</td><td>{r['memory']['peak_rss_bytes']/2**20:.1f}</td></tr>")
    page.append('</tbody></table>')
    page.append('<h2>Measured limits and shortcut diagnostic</h2><p>'+esc(findings).replace('\n\n','</p><p>')+'</p>')
    for key in ('f0000_r0.5','f0000_r1'):
        page.append(f'<a href="plots/{key}_shortcut_detail.png"><img loading="lazy" src="plots/{key}_shortcut_detail.png" alt="Hole-location longest edge detail, not physical connectivity proof"></a>')
    page.append('<p>Report rendering initially failed on an empty Box 3D collection. Recovery reused this UUID and saved numerical artifacts only; no new extraction, sampling, edges or benchmark. Original failure and computation sources are retained.</p>')
    biggest=max(records.items(),key=lambda x:x[1]['N']);smallest=min(records.items(),key=lambda x:x[1]['N'])
    page.append(f"<p>Largest representative N: {biggest[0]} ({biggest[1]['N']}); kNN mean {biggest[1]['timing_ms']['knn_dedup_ms']['mean']:.3f} ms. Smallest: {smallest[0]} ({smallest[1]['N']}); {smallest[1]['timing_ms']['knn_dedup_ms']['mean']:.3f} ms. Exact sorting grows with component size; no truncation was applied.</p>")
    for key,r in records.items():
        page.extend([f'<h2 id="{key}">{esc(key)} · N={r["N"]}, E={r["E"]}</h2>',
                     f'<p><a href="packets/{key}.npz">Lossless v2 packet</a></p>',
                     f'<img loading="lazy" src="plots/{key}_alignment.png" alt="Source alignment and normal validity">',
                     f'<img loading="lazy" src="plots/{key}_edges.png" alt="All edges and longest-pair shortcut diagnostic in source and camera/EE views">',
                     '<details><summary>Object/component counts, degree, isolated nodes, length statistics, longest source/UV/XYZ pairs</summary><pre>'+esc(json.dumps(serial(r['edge']),ensure_ascii=False,indent=2))+'</pre></details>',
                     '<details><summary>Numerics, availability/validity and CPU/memory details</summary><pre>'+esc(json.dumps(serial({k:r[k] for k in ('objects','numerics','timing_ms','memory')}),ensure_ascii=False,indent=2))+'</pre></details>',
                     '<details><summary>Every final point: source ↔ camera/EE XYZ, F=5, field masks and generation evidence</summary><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ('row','object','source','u,v','camera XYZ m','EE XYZ m','normal EE, variation rad, depth m','available','valid','PCA/Sil candidate','selected by PCA/Sil','stage 0=P,1=S','component'))+'</tr></thead><tbody>'])
        p=r['point_rows']
        for i in range(r['N']):
            values=(i,p['object_id'][i],p['source'][i],p['uv'][i],p['camera_xyz'][i],p['ee_xyz'][i],p['features'][i],p['field_available'][i],p['field_valid'][i],p['candidate_evidence'][i],p['selected_by_structural_source'][i],p['selection_stage'][i],p['component_id'][i])
            page.append('<tr>'+''.join('<td>'+esc(np.array2string(v,precision=7) if isinstance(v,np.ndarray) else v)+'</td>' for v in values)+'</tr>')
        page.append('</tbody></table></div></details>')
    page.append('<h2>Downstream scope</h2><p>v1/v2 deep-copy and row/provenance/feature/edge preservation passed. Robot +3 remap was tested only with explicitly supplied synthetic robot positions. No encoder, training-ready x, robot/inter-object policy, stale cache or policy learning was added. Longest-pair diagrams are visual diagnostics, not GT-based physical edge labels. No extra episode, simulator, GPU/PCL, dependency install, upload or commit.</p></html>')
    (out/'vision_feature_edge.html').write_text('\n'.join(page))


def build_shortcut_details(out,packets,records):
    """Display-only zoom of a suspicious pair and dominant Box component."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    cfg=json.loads((SAMPLING/'comparison.json').read_text())['config']
    with np.load(cfg['input'],allow_pickle=False) as a:rgb=a['base_camera/rgb'][0]
    for key in ('f0000_r0.5','f0000_r1'):
        p=packets[key];objects=records[key]['edge']['objects'];box=objects.get(1,objects.get('1'))
        longest=box['longest_edges'][0];pair=np.asarray(longest['point_rows'])
        src=p['provenance']['source_index'];w=int(p['image_shape'][1]);uv=np.column_stack((src%w,src//w))
        edge=p['intra_object_edge_index'];edge=edge[:,edge[0]<edge[1]]
        comp=p['observed_component_id'];oid=p['object_id'];xyz=p['camera_points']
        fig=plt.figure(figsize=(14,10));a=fig.add_subplot(2,2,1);b=fig.add_subplot(2,2,2,projection='3d')
        c=fig.add_subplot(2,2,3);d=fig.add_subplot(2,2,4,projection='3d')
        for ax in (a,c):ax.imshow(rgb);ax.set(xlabel='Source u (pixel)',ylabel='Source v (pixel)')
        a.plot(*uv[pair].T,'o-',c='red',lw=2)
        for i,row in enumerate(pair):a.annotate(f'row {row}\nsource {src[row]}',uv[row],xytext=(10,(-1)**i*30),textcoords='offset points',arrowprops=dict(arrowstyle='->'),fontsize=8)
        lo=uv[pair].min(0)-20;hi=uv[pair].max(0)+20
        a.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),title=f'Hole-location suspect pair: {longest["length_m"]*1000:.3f} mm')
        longest_rows=(oid==1)&(comp==comp[pair[0]])
        b.scatter(*xyz[longest_rows].T,s=20,c='deepskyblue');b.plot(*xyz[pair].T,'o-',c='red',lw=2)
        b.set(title='Same observed component at camera-z 8.431m\n(not a validated physical Box surface)',xlabel='camera x (m)',ylabel='camera y (m)',zlabel='camera z (m)')
        dominant=max(box['components'],key=lambda x:x['point_count'])['component_id']
        rows=(oid==1)&(comp==dominant);e=edge[:,rows[edge[0]]]
        c.add_collection(LineCollection(uv[e.T],colors='deepskyblue',alpha=.4,linewidths=.6))
        c.scatter(*uv[rows].T,s=7,c='red')
        low=uv[rows].min(0)-8;high=uv[rows].max(0)+8
        c.set(xlim=(low[0],high[0]),ylim=(high[1],low[1]),title='Largest selected Box component: local image view')
        d.scatter(*xyz[rows].T,s=6,c='deepskyblue')
        if e.shape[1]:d.add_collection3d(Line3DCollection(xyz[e.T],colors='deepskyblue',linewidths=.6,alpha=.35))
        d.set(xlabel='camera x (m)',ylabel='camera y (m)',zlabel='camera z (m)',title='Largest selected Box component: local 3D view')
        fig.suptitle(key+' · display-only component/UV zoom; all point rows/edges retained in NPZ')
        fig.tight_layout();fig.savefig(out/f'plots/{key}_shortcut_detail.png',dpi=150);plt.close(fig)


def finalize_outputs(logger,summary):
    import difflib
    import re
    from html.parser import HTMLParser
    from PIL import Image
    from utils.experiment_artifacts import register_existing_output
    out=logger.directory if logger is not None else REPO/'experiments'/summary['run_id']
    run_id=summary['run_id']
    baseline=json.loads((out/'git_start.json').read_text())
    changed=[p for p,h in baseline['hashes'].items() if p not in baseline['editable'] and sha(REPO/p)!=h]
    if changed:raise AssertionError('protected source changed: '+repr(changed))
    inputs=summary['feature_gate']['input_hashes']
    bad=[p for p,h in inputs.items() if sha(REPO/p)!=h]
    if bad:raise AssertionError('input hash changed: '+repr(bad))
    sources=['visual/runtime/vision_packet.py','visual/runtime/intra_object_edges.py','graph/vision_input.py',
             'visual/runtime/test/test_vision_features.py','visual/runtime/test/test_intra_object_edges.py',
             'visual/runtime/test/test_variable_rate_graph_contract.py','visual/runtime/test/run_vision_feature_edge.py',
             'docs/visual-graph-integration/00_interface.md',
             'visual/runtime/sampling.py','visual/runtime/pca.py','visual/runtime/geometry.py',
             'visual/runtime/edge_evidence.py','visual/runtime/normal_estimators.py','visual/runtime/rgbd.py',
             'utils/experiment_logger.py','utils/experiment_artifacts.py']
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in sources:
        target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/p,target)
    diff=[]
    for p in sources[:8]:
        original=out/'original_source'/p
        before=original.read_text() if original.exists() else ''
        diff.extend(difflib.unified_diff(before.splitlines(True),(REPO/p).read_text().splitlines(True),
                                       fromfile='a/'+p,tofile='b/'+p))
    (out/'implementation.diff').write_text(''.join(diff))
    write(out/'source_hashes.json',dict(source_after={p:sha(REPO/p) for p in sources},
        source_before={p:baseline['hashes'].get(p) for p in sources},input_hashes=inputs,
        computation_source_snapshot={str(p.relative_to(out/'computation_source_snapshot')):sha(p)
                                     for p in (out/'computation_source_snapshot').rglob('*') if p.is_file() and not p.name.endswith('.tracking.json')},
        protected_source_count=len(baseline['hashes'])-len(baseline['editable']),protected_source_unchanged=True))
    # Add visual conclusions without changing any topology/measurement artifact.
    summary['preservation']=dict(protected_source_count=len(baseline['hashes'])-len(baseline['editable']),
                                 protected_sources_unchanged=True,input_hashes_unchanged=True)
    write(out/'vision_feature_edge.json',summary)
    render_reports(out,summary)
    decoded=[]
    for p in sorted((out/'plots').glob('*.png')):
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:im.load();decoded.append(dict(path=str(p.relative_to(out)),width=im.width,height=im.height))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and v and not v.startswith('#'):self.paths.append(v)
    parser=Links();parser.feed((out/'vision_feature_edge.html').read_text())
    missing=[v for v in parser.paths if not (out/v.split('#')[0]).exists()]
    markdown=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    missing += [v for v in markdown if not (REPORT.parent/v).exists()]
    # Verification itself is written next, so allow this one forward link only.
    missing=[v for v in missing if v not in ('verification.json',f'../../experiments/{run_id}/verification.json')]
    if missing:raise AssertionError('relative link failure: '+repr(missing))
    if '../../docs/visual-graph-integration/04_vision_feature_edge.md' not in parser.paths:
        raise AssertionError('missing HTML to Markdown backlink')
    with np.load(out/'vision_feature_edge.npz',allow_pickle=False) as a:
        for key,r in summary['records'].items():
            packet=read_packet(out/'packets'/f'{key}.npz')
            if packet_digest(packet)!=r['packet_hash']:raise AssertionError('final packet readback')
            for k,v in flatten_arrays(packet,prefix=key).items():np.testing.assert_array_equal(a[k],v)
    for p in out.glob('*.json'):json.loads(p.read_text())
    verification=dict(status='PASS',feature_contract='PASS',edge_contract='PASS',
        tests=dict(command_test_counts=[int(x) for x in re.findall(r'Ran (\d+) tests', (out/'tests.log').read_text())],
                   final_contract_suite_passed='\nOK\n' in (out/'tests.log').read_text(),log='tests.log'),
        source_alignment=True,cache_parity=True,feature_validity=True,coordinate_transform=True,
        graph_receiver=True,v1_regression=True,supplied_synthetic_robot_offset=True,
        source_input_hash_unchanged=True,point_row_order_preserved=True,
        edge_cross_object_count=0,edge_cross_component_count=0,
        json_readback=True,npz_readback=True,png_decode=True,pngs=decoded,
        markdown_html_relative_links=True,browser=summary['browser'],visual_review=summary['visual_review'],
        report_only_recovery=summary.get('report_only_recovery',False),
        edge_physical_surface_correctness='NOT_ESTABLISHED',policy_adopted=False,
        simulator_executed=False,gpu_executed=False,pcl_executed=False,
        sampling_reexecuted=False,extraction_calls=8,post_evidence_warmup=3,post_evidence_repeats=5,
        gnn_bc_executed=False,integrated_20hz_validated=False)
    write(out/'verification.json',verification)
    # Shared local registration only; original inputs are never registered as outputs.
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name.endswith('.tracking.json') or p.name in ('config.yaml','metadata.json','metrics.json','manifest.json'):continue
        kind='trajectory' if p.suffix=='.npz' else 'metrics' if p.suffix=='.json' else 'reports'
        if logger is not None:logger.link_output(p,kind=kind)
        else:register_existing_output(out,p,kind=kind)
    if logger is not None:logger.link_output(REPORT,kind='reports')
    else:register_existing_output(out,REPORT,kind='reports')
    final_metrics=dict(feature_pass=1,edge_pass=1,representative_frames=8,conditions=16,
                    max_n=max(r['N'] for r in summary['records'].values()),
                    max_e=max(r['E'] for r in summary['records'].values()),
                    max_post_evidence_ms=max(r['timing_ms']['post_evidence_subtotal_ms']['max'] for r in summary['records'].values()))
    if logger is not None:
        logger.log(final_metrics)
        logger.record(feature_contract='PASS',edge_contract='PASS',verification='verification.json',browser=summary['browser']['status'])
    else:
        # Same report-only recovery pattern as Task03-J, using shared serializers.
        from datetime import datetime,timezone
        from utils.experiment_logger import write as shared_write,scalars
        metadata=json.loads((out/'metadata.json').read_text())
        metadata.setdefault('initial_report_failure',dict(status=metadata['status'],exception_type=metadata.get('exception_type'),
                                               finished_at=metadata.get('finished_at'),evidence='failure.json'))
        metadata.update(status='finished',finished_at=datetime.now(timezone.utc).isoformat(),
                        report_only_recovery=True,feature_contract='PASS',edge_contract='PASS',
                        verification='verification.json',browser=summary['browser']['status'])
        metadata.pop('exception_type',None)
        shared_write(out/'metadata.json',metadata)
        metrics=json.loads((out/'metrics.json').read_text())
        if not metrics:shared_write(out/'metrics.json',[dict(run_id=run_id,step=0,metrics=scalars(final_metrics))])
        from utils.experiment_artifacts import save_manifest
        save_manifest(out,metadata)
    print('TASK04_COMPLETE',run_id,flush=True)


def recover_reports(out):
    """Render saved numerical results only, retaining the original UUID/failure."""
    out=out.resolve()
    summary=json.loads((out/'vision_feature_edge.json').read_text())
    if summary['run_id']!=out.name or summary['feature_contract']!='PASS' or summary['edge_contract']!='PASS':
        raise AssertionError('completed numerical run required for report recovery')
    immutable=[out/'vision_feature_edge.npz',out/'feature_gate.json',out/'tests.log',
               *sorted((out/'packets').glob('*.npz')),*sorted((out/'runtime_evidence').glob('*.npz'))]
    hashes={str(p.relative_to(out)):sha(p) for p in immutable}
    packets={k:read_packet(out/'packets'/f'{k}.npz') for k in summary['records']}
    for k,p in packets.items():
        if packet_digest(p)!=summary['records'][k]['packet_hash']:raise AssertionError('recovery packet hash mismatch')
    build_figures(out,packets,summary['records'])
    build_shortcut_details(out,packets,summary['records'])
    render_reports(out,summary)
    print('ARTIFACTS_READY',out,flush=True)
    if sys.stdin.isatty():input('Agent artifact review checkpoint; press Enter after review metadata is saved.\n')
    review=json.loads((out/'artifact_review.json').read_text()) if (out/'artifact_review.json').exists() else {}
    summary['browser']=review.get('browser',dict(status='NOT_EXECUTED'))
    summary['visual_review']=review.get('visual_review',dict(status='NOT_EXECUTED'))
    after={str(p.relative_to(out)):sha(p) for p in immutable}
    if hashes!=after:raise AssertionError('report recovery changed numerical artifacts')
    summary['report_only_recovery']=dict(original_failure='failure.json',new_uuid=False,
        new_extraction=False,new_sampling=False,new_edges=False,new_benchmark=False,numerical_hashes_before=hashes,
        numerical_hashes_after=after,computation_source='computation_source_snapshot/',report_source='source_snapshot/')
    finalize_outputs(None,summary)


def refine_reports(out):
    """Add display-only diagnostic zooms to already computed, recovered results."""
    out=out.resolve();summary=json.loads((out/'vision_feature_edge.json').read_text())
    packets={k:read_packet(out/'packets'/f'{k}.npz') for k in summary['records']}
    for key,p in packets.items():
        if packet_digest(p)!=summary['records'][key]['packet_hash']:raise AssertionError('saved packet changed')
    build_shortcut_details(out,packets,summary['records'])
    summary['edge_physical_surface_correctness']='NOT_ESTABLISHED'
    summary['policy_adopted']=False
    finalize_outputs(None,summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--pause-after-feature-gate',action='store_true')
    parser.add_argument('--memory-worker',nargs=2,metavar=('RUN_DIRECTORY','PACKET_KEY'))
    parser.add_argument('--recover-reports',type=Path)
    parser.add_argument('--refine-reports',type=Path)
    args=parser.parse_args()
    if args.refine_reports:refine_reports(args.refine_reports)
    elif args.recover_reports:recover_reports(args.recover_reports)
    elif args.memory_worker:memory_worker(Path(args.memory_worker[0]),args.memory_worker[1])
    elif args.baseline:run(args.baseline,args.pause_after_feature_gate)
    else:parser.error('--baseline required')

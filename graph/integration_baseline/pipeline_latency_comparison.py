"""Matched RGB-to-output latency of Task03 S3 and the actual Raw EDE pipeline.

Only this experiment opts into CUDA sampling. Production builders and defaults
are reused. Exclusive host partitions reconcile to each measured frame total;
CUDA event details are nested diagnostics and never added to host partitions.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import h5py
import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline import vision
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.robot import RobotNormalizer, geometry_in_ee
from graph.integration_baseline.streaming_vision_adapter import build_stream_tracker, StreamVisionAdapter
from graph.integration_baseline.streaming_vision_benchmark import fingerprint, statistics
from visual.runtime.grounding import GDino
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence, sample_points
from visual.runtime.cuda_sampling import CudaSampling
from visual.runtime.test import run_variable_rate_shape_first as shape

MODES = ('visual_s3_cpu', 'visual_s3_cuda', 'raw_ede_cpu', 'raw_ede_cuda')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Events:
    """CUDA event intervals on the real forward pass, without per-layer sync."""
    def __init__(self, model):
        self.rows = []; self.handles = []; self.active = {}
        modules = {'gnn_and_readout': model.graph_encoder,
                   'recurrent_encoder': model.policy.encoder,
                   'gru': model.policy.rnn, 'action_head': model.policy.head}
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self.start(name)))
            self.handles.append(module.register_forward_hook(self.end(name)))

    def start(self, name):
        def begin(*_):
            event = torch.cuda.Event(enable_timing=True); event.record(); self.active[name] = event
        return begin

    def end(self, name):
        def finish(*_):
            event = torch.cuda.Event(enable_timing=True); event.record()
            self.rows.append((name, self.active.pop(name), event))
        return finish

    def collect(self):
        out = {name: a.elapsed_time(b) for name, a, b in self.rows}
        self.rows.clear(); return out

    def close(self):
        for handle in self.handles: handle.remove()


class ModuleMeter:
    def __init__(self):
        self.values = defaultdict(float)

    def call(self, name, fn, *args, **kwargs):
        tick = perf_counter(); value = fn(*args, **kwargs)
        self.values[name] += (perf_counter()-tick)*1000
        return value

    def wrapped(self, name, fn):
        def run(*args, **kwargs): return self.call(name, fn, *args, **kwargs)
        return run


def downstream(mode, frame, episode, arrays, t, model, normalizer, hidden, gpu, event_detail=False):
    """Invoke the existing packet_from_frame and model.step, without forks."""
    meter = ModuleMeter(); detail = {}; events = Events(model) if event_detail and mode.startswith('raw') else None
    is_gpu = mode.endswith('cuda')
    evidence_fn = gpu.evidence if is_gpu else extract_sampling_evidence
    evidence = None
    def extraction(f, *args, **kwargs):
        nonlocal evidence
        # Existing caller fixes estimator='pca5'; GPU API has that fixed too.
        evidence = meter.call('pca5_evidence', evidence_fn, f)
        detail['evidence'] = evidence.timings_ms
        return evidence
    tick = perf_counter()
    if mode.startswith('visual'):
        extraction(frame)
        result = meter.call('s3_sampling', sample_points, evidence, method='S3',
                            fps_fn=gpu.fps_indices if is_gpu else None)
        detail['sampling'] = result.timings_ms
        output = result; counts = dict(points=len(result.points), edges=0)
    else:
        with ExitStack() as stack:
            stack.enter_context(patch.object(vision, 'extract_sampling_evidence', extraction))
            for attr, name in [('extract_candidates','candidates'), ('complete_vision_packet','f5_ee_packet'),
                               ('observed_components','observed_components'), ('build_raw_ede','raw_ede'),
                               ('receive_vision_packet','producer_receiver_validation')]:
                original = getattr(vision, attr)
                if attr == 'build_raw_ede':
                    def edges(*args, _original=original, **kwargs):
                        result = _original(*args, **kwargs)
                        detail['raw_ede'] = result[2]['timing_ms']
                        return result
                    original = edges
                elif attr == 'complete_vision_packet':
                    def features(*args, _original=original, **kwargs):
                        detail['f5_ee'] = {}
                        return _original(*args, **kwargs, timings=detail['f5_ee'])
                    original = features
                stack.enter_context(patch.object(vision, attr, meter.wrapped(name, original)))
            for attr, name in [('prepare_object','contour_scale_voxel'), ('select_object','shape_surface_selection'),
                               ('make_packet','camera_selection_packet')]:
                original = getattr(shape, attr)
                if attr == 'select_object' and is_gpu:
                    def selection(*args, _original=original, **kwargs):
                        return _original(*args, **kwargs, fps_fn=gpu.fps_indices)
                    original = selection
                stack.enter_context(patch.object(shape, attr, meter.wrapped(name, original)))
            packet, detail['packet'] = vision.packet_from_frame(frame, episode)
        robot_positions = meter.call('robot_geometry_ee', geometry_in_ee,
                                     arrays['robot_world_geometry'][t], arrays['tcp_pose'][t])
        graph = meter.call('graph_construction', build_graph, packet, robot_positions)
        detail['graph'] = graph.timings_ms
        packed_cpu = meter.call('graph_packing', pack_graphs, [graph], [(0,0)])
        begin = perf_counter()
        packed = packed_cpu.to('cuda')
        robot = torch.from_numpy(arrays['robot_state'][t].copy())[None].cuda()
        torch.cuda.synchronize(); meter.values['input_transfer'] = (perf_counter()-begin)*1000
        begin = perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            action, hidden = model.step(packed, robot, normalizer, hidden)
        torch.cuda.synchronize(); meter.values['gnn_gru_action'] = (perf_counter()-begin)*1000
        begin = perf_counter(); action_cpu = action.cpu().numpy().copy(); torch.cuda.synchronize()
        meter.values['action_return'] = (perf_counter()-begin)*1000
        if events:
            detail['policy_cuda_event_ms'] = events.collect(); events.close()
        output = (packet, action_cpu)
        counts = dict(points=len(packet['points']), edges=int(graph.edge_index.shape[1]),
                      intra_edges=int(packet['intra_object_edge_index'].shape[1]))
    total = (perf_counter()-tick)*1000
    meter.values['downstream_orchestration'] = total-sum(meter.values.values())
    return output, hidden, dict(meter.values), detail, counts


def load_inputs(root):
    source = root/'d1fd6173cb3648bfa4c03507f84b7e71'
    manifest=json.loads((source/'dataset_manifest.json').read_text())
    schema=json.loads((source/'robot_state_schema.json').read_text())
    split=json.loads((source/'split_manifest.json').read_text())
    entries={e['episode_id']: e for e in manifest['episodes']}
    arrays={}; proof={}
    for eid in (30002,30047):
        path=source/entries[eid]['path']
        digest=sha(path)
        if digest != entries[eid]['sha256']: raise ValueError('input episode hash mismatch')
        with h5py.File(path) as f:
            a={}
            def visit(k,v):
                if isinstance(v,h5py.Dataset): a[k]=v[()]
            f.visititems(visit)
        arrays[eid]=a;proof[str(eid)]=dict(sha256=digest,frames=len(a['base_camera/rgb']))
    with np.load(root/'train_robot.npz',allow_pickle=False) as z:
        ids=z['episode_ids'].tolist()
        if sorted(ids)!=sorted(split['splits']['train']): raise ValueError('normalizer train split mismatch')
        normalizer=RobotNormalizer.fit(z['robot'],ids,split['splits'],schema)
    proof['train_robot_sha256']=sha(root/'train_robot.npz')
    return arrays,schema,normalizer,proof


def make_frame(a,t,masks):
    return TrackedRGBDFrame(t,'base_camera',a['base_camera/rgb'][t],a['base_camera/depth'][t],
        a['base_camera/intrinsic'][t],a['base_camera/cam2world'][t],a['tcp_pose'][t],
        {0:masks[0],1:masks[1]},t,float(a['timestamp'][t]))


def summarize(rows):
    result={}
    for mode in MODES:
        result[mode]={}
        for group in ('all','tracking','dino','first'):
            selected=[r for r in rows if r['mode']==mode and (group=='all' or r['group']==group)]
            if not selected: continue
            fields=sorted({k for r in selected for k in r['modules_ms']})
            means={k:float(np.mean([r['modules_ms'].get(k,0) for r in selected])) for k in fields}
            result[mode][group]=dict(total_ms=statistics([r['total_ms'] for r in selected]),
                modules_ms={k:statistics([r['modules_ms'].get(k,0) for r in selected]) for k in fields},
                module_mean_sum_ms=sum(means.values()), points=statistics([r['counts']['points'] for r in selected]),
                edges=statistics([r['counts']['edges'] for r in selected]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--dino',required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--limit',type=int)
    p.add_argument('--no-compile',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(False)
    data,schema,normalizer,proof=load_inputs(args.inputs)
    if args.limit: data={eid:{k:v[:args.limit] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(a['base_camera/rgb']) else v for k,v in a.items()} for eid,a in data.items()}
    setup=perf_counter();detector=GDino(device='cuda',model_id=args.dino)
    tracker=build_stream_tracker(compile_image_encoder=not args.no_compile)
    adapter=StreamVisionAdapter(detector,tracker)
    model=IntegrationPolicy(schema,d=64,L=2,gru_hidden=144,gru_layers=1,enc_hidden=128).cuda().eval()
    model_id=fingerprint(model.state_dict()); gpu=CudaSampling()
    config=dict(instance_id='57987772',repeats=args.repeats,limit=args.limit,modes=MODES,inputs=proof,
        compile=not args.no_compile,tracking='same Visual BF16 compiled GPU streaming adapter for all modes',
        precision='FP32 policy; CPU FP64 evidence or optional CUDA FP32 evidence/FPS',
        policy='seed42 weights; identical architecture/weights for both Raw EDE modes',policy_weights_sha256=model_id,
        endpoints={'visual':'CPU SampledPoints, PCA5+S3 256 budget', 'raw_ede':'CPU Action7, unchanged packet_from_frame and model.step'},
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        setup_ms=(perf_counter()-setup)*1000,optimizer_updates=0,smoke=args.smoke,
        dino_model_sha256=sha(Path(args.dino)/'model.safetensors'),sam2_sha256=sha(vision.DEFAULT_CHECKPOINT))
    with ExperimentLogger(team='integration',model='full_pipeline_latency_comparison',config=config,seed=42,job_type='latency',root=args.output_root) as log:
        root=log.directory.resolve();print('RUN_DIR='+str(root),flush=True)
        def save(name,value):
            path=root/name;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');log.link_output(path,kind='metrics')
        source_root=Path(__file__).resolve().parents[2]
        source_files=['graph/integration_baseline/pipeline_latency_comparison.py','graph/integration_baseline/vision.py',
            'graph/integration_baseline/graph.py','graph/integration_baseline/model.py','graph/integration_baseline/robot.py',
            'graph/integration_baseline/streaming_vision_adapter.py','visual/scripts/stream_v1.py',
            'visual/runtime/sampling.py','visual/runtime/pca.py','visual/runtime/cuda_sampling.py',
            'visual/runtime/cuda_sampling.cu','visual/runtime/cuda_driver.py','visual/runtime/vision_packet.py',
            'visual/runtime/intra_object_edges.py','visual/runtime/extended_delaunay_edges.py',
            'visual/runtime/test/raw_ede_comparison_adapter.py','visual/runtime/test/run_variable_rate_shape_first.py']
        hashes={}
        for rel in source_files:
            source=source_root/rel;target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes());hashes[rel]=sha(source)
        save('source_hashes.json',hashes)
        masks_ref={}; outputs_ref={}; rows=[]; checks=[];detail_rows=[]
        def episode(mode,eid,a,repeat,phase):
            adapter.reset(f'{phase}:{eid}');hidden=None;mask_hashes=[]
            original_push=adapter.initializer.push; original_dino=adapter.initializer.detector.detector
            init_ms=0.; dino_ms=0.;dino_calls=0
            def dino(*v,**kw):
                nonlocal dino_ms,dino_calls
                start=perf_counter();out=original_dino(*v,**kw)
                dino_ms+=(perf_counter()-start)*1000;dino_calls+=1
                return out
            def push(image):
                nonlocal init_ms
                start=perf_counter();out=original_push(image);init_ms+=(perf_counter()-start)*1000;return out
            adapter.initializer.push=push;adapter.initializer.detector.detector=dino
            try:
                for t in range(len(a['base_camera/rgb'])):
                    init_ms=0.;dino_ms=0.;dino_calls=0
                    torch.cuda.synchronize();start=perf_counter()
                    masks,record=adapter.process(a['base_camera/rgb'][t],f'{phase}:{eid}',t)
                    torch.cuda.synchronize();mask_ms=(perf_counter()-start)*1000
                    frame=make_frame(a,t,masks)
                    output,hidden,modules,detail,counts=downstream(mode,frame,f'ep{eid}',a,t,model,normalizer,hidden,gpu,phase=='detail')
                    torch.cuda.synchronize();total=(perf_counter()-start)*1000
                    modules.update(dino=dino_ms,prefix_initialization=init_ms-dino_ms,
                                   sam2_preprocess_tracking_mask_return=mask_ms-init_ms)
                    modules['frame_orchestration']=total-sum(modules.values())
                    if any(v < -1e-6 for v in modules.values()): raise AssertionError('overlapping host partitions')
                    row=dict(mode=mode,episode=eid,timestep=t,repeat=repeat,phase=phase,
                        group='first' if t==0 else ('dino' if dino_calls else 'tracking'),
                        total_ms=total,modules_ms=modules,detail=detail,counts=counts,dino_calls=dino_calls)
                    if phase=='timed': rows.append(row)
                    if phase=='detail': detail_rows.append(row)
                    if phase=='timed' and repeat==0:
                        key=(eid,t);digest=fingerprint(masks);mask_hashes.append(digest)
                        if key in masks_ref:
                            checks.append(dict(mode=mode,episode=eid,timestep=t,mask_equal=digest==masks_ref[key]))
                        else:masks_ref[key]=digest
                        if mode.startswith('visual'):
                            identity=output.flat_index; objects=output.object_id
                            action=None
                        else:
                            identity=output[0]['provenance']['source_index'];objects=output[0]['object_id'];action=output[1]
                        family='visual' if mode.startswith('visual') else 'raw'
                        out_key=(family,eid,t)
                        if mode.endswith('cpu'):
                            outputs_ref[out_key]=(identity.copy(),objects.copy(),None if action is None else action.copy())
                        elif out_key in outputs_ref:
                            old,old_objects,old_action=outputs_ref[out_key]
                            aa=set(zip(old_objects.tolist(),old.tolist()));bb=set(zip(objects.tolist(),identity.tolist()))
                            checks.append(dict(mode=mode,episode=eid,timestep=t,
                                selected_source_jaccard=len(aa&bb)/max(1,len(aa|bb)) if aa or bb else 1.,
                                reference_points=len(old),points=len(identity),
                                action_max_abs_difference=None if action is None else float(np.max(np.abs(action-old_action)))))
                        # Exact CPU reference for this genuine production invocation.
                        if mode=='raw_ede_cpu' and t in (0,10,len(a['base_camera/rgb'])-1):
                            packet,_=vision.packet_from_frame(frame,f'ep{eid}')
                            checks.append(dict(mode=mode,episode=eid,timestep=t,packet_original_equal=fingerprint(packet)==fingerprint(output[0])))
                    if phase=='timed' and t%40==0: print(f'PROGRESS={mode}:{repeat}:{eid}:{t}',flush=True)
            finally:
                adapter.initializer.push=original_push;adapter.initializer.detector.detector=original_dino
                adapter.close()
        try:
            warm=perf_counter()
            for eid,a in data.items():episode(MODES[-1],eid,a,0,'warmup')
            # Warm every downstream configuration using actual warm masks through
            # a complete pass, preserving full-episode state/reset semantics.
            for mode in MODES[:-1]:
                for eid,a in data.items():episode(mode,eid,a,0,'warmup')
            save('warmup.json',dict(elapsed_ms=(perf_counter()-warm)*1000))
            print('WARMUP_COMPLETE',flush=True)
            for repeat in range(args.repeats):
                order=MODES[repeat%len(MODES):]+MODES[:repeat%len(MODES)]
                for mode in order:
                    for eid,a in data.items():episode(mode,eid,a,repeat,'timed')
                    print(f'COMPLETE={mode}:{repeat}',flush=True)
                save('frames_partial.json',rows)
            for mode in ('raw_ede_cpu','raw_ede_cuda'):
                for eid,a in data.items():episode(mode,eid,a,0,'detail')
            save('frames.json',rows);save('policy_detail_frames.json',detail_rows);save('checks.json',checks)
            summary=summarize(rows)
            masks_equal=all(x.get('mask_equal',True) for x in checks)
            original_equal=all(x.get('packet_original_equal',True) for x in checks)
            save('summary.json',dict(status='COMPLETE' if masks_equal and original_equal else 'COMPARISON_FAILED',
                run_id=log.run_id,config=config,results=summary,mask_equality=masks_equal,
                original_cpu_packet_equality=original_equal))
            if not masks_equal or not original_equal:
                raise AssertionError('comparison requires identical masks and unchanged CPU packet outputs')
            lines=['# Same-host full pipeline comparison','',f'Run `{log.run_id}`','',
                   '| Mode | Group | N | Mean ms | p50 ms | p95 ms | <=50ms |',
                   '|---|---|---:|---:|---:|---:|---:|']
            for mode,groups in summary.items():
                for group,value in groups.items():
                    s=value['total_ms'];lines.append(f'|{mode}|{group}|{s["n"]}|{s["mean"]:.3f}|{s["median"]:.3f}|{s["p95"]:.3f}|{100*s["deadline_50ms_fraction"]:.1f}%|')
            lines+=['','Exclusive module mean wall times (tracking frames):','', '| Module | '+' | '.join(MODES)+' |', '|---|'+'---:|'*len(MODES)]
            fields=sorted({k for mode in MODES for k in summary[mode]['tracking']['modules_ms']})
            for field in fields:
                lines.append('|'+field+'|'+'|'.join(f'{summary[m]["tracking"]["modules_ms"][field]["mean"]:.3f}' if field in summary[m]['tracking']['modules_ms'] else '—' for m in MODES)+'|')
            path=root/'report.md';path.write_text('\n'.join(lines)+'\n');log.link_output(path,kind='reports')
            log.record(validation_status='COMPLETE',optimizer_updates=0,published=False)
            print('\n'.join(lines),flush=True)
        finally:adapter.close();gpu.close()


if __name__=='__main__':main()

"""One frozen condition, live RGB-D to current Graph, with no GT file reads.

Run conditions in separate Python processes. Timing retains real detector,
feature, tracker, correction and Graph work; no cross-condition cache credit.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from graph.task10i_stage4a.run_pipeline import read, write, sha, CAMERAS


def array_digest(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(repr((value.shape,value.dtype.str)).encode());h.update(value.tobytes())
    return h.hexdigest()


class TimedDetector:
    def __init__(self,detector):self.detector=detector;self.calls=[];self.context={}
    def __call__(self,rgb,prompt):
        torch.cuda.synchronize();start=time.perf_counter()
        result=self.detector(rgb,prompt)
        torch.cuda.synchronize()
        elapsed=1000*(time.perf_counter()-start)
        self.calls.append(dict(**self.context,prompt=prompt,total_ms=elapsed,
            input_rgb_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            postprocessed_boxes=result[0],postprocessed_scores=result[1],raw_token_logits_saved=False,
            **{k+'_ms':v for k,v in self.detector.last_ms.items()}))
        return result


def run(root,expected,condition,geometry_cache=False,lazy_matching=False,engineering=False):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.plan_a_bc.geometry import CurrentGeometry,validate_graph
    from graph.task10i_stage4a.development_perception import Memo
    from graph.task10i_stage4a.perframe_cache import install
    from .preprocessing_audit import LosslessAdapter
    from .tri_model_integration import CausalCorrection
    from .dinov2_runtime import CurrentFeatures
    if condition not in ('B0','P','C_KEEP','BC_KEEP','BC_MAIN_BOX','BC_WB_HANDOFF'):raise ValueError('unregistered condition')
    if lazy_matching and not condition.startswith('BC'):raise ValueError('lazy matching requires BC condition')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        paths=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
        identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        assert sha(paths['sam'])==identity['sam_sha256']
        assert str(torch.__version__)==identity['torch_version']
        assert all(sha(Path(paths['dino'])/p)==h for p,h in identity['dino_files'].items())
        import sam2
        installed_sam=Path(sam2.__file__).parent
        runtime_inputs=Path('/workspace/task10g_f1_abc_20261031/full/inputs')
        specification=dict(condition=condition,expected_sha256=sha(expected),engineering=engineering,
            geometry_cache=geometry_cache,lazy_matching=lazy_matching,GT_forward_input=False,
            source={str(p.relative_to(source)):sha(p) for directory in ('graph','visual/runtime','utils')
                    for p in (source/directory).rglob('*.py')},model_identity=identity,
            installed_sam_files={str(p):sha(p) for p in installed_sam.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')},
            packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','tokenizers','numpy','scipy')},
            schema_sha256=sha('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'),
            robot_assets={str(p):sha(p) for p in runtime_inputs.rglob('*') if p.is_file()},
            runtime='same fixed CPU PhysX/CUDA-render source data; model CUDA on RTX3090',
            timing='live separate-condition process; no raw query/logit serialization; CUDA synchronized; input HDF5 read, current geometry, Graph included; model loading excluded')
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==specification,'changed source/config needs a new output root'
        else:write(frozen,specification)
        torch.set_num_threads(1);torch.manual_seed(42)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        manifest=read(expected)
        assert manifest['episodes']==len(manifest['entries'])
        assert manifest['frames']==sum(e['actions']+1 for e in manifest['entries'])
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_frozen_live_'+condition,
                seed=42,root=root/'experiments',job_type='evaluation',config=specification) as logger:
            detector=TimedDetector(GDino('cuda',model_id=paths['dino']));memo=Memo(detector)
            tracker=SAM2Tracker('cuda',checkpoint=paths['sam'])
            features=CurrentFeatures('/workspace/task10i_stage4a2_models/dinov2-small') if condition.startswith('BC') else None
            if features is not None:write(root/'dinov2_model_identity.json',features.identity)
            feature_events=[]
            if features is not None:
                native_embed=features.embed
                def timed_embed(camera,t,mask,*args,**kwargs):
                    torch.cuda.synchronize();tick=time.perf_counter();before=len(features.calls)
                    result=native_embed(camera,t,mask,*args,**kwargs)
                    torch.cuda.synchronize()
                    feature_events.append(dict(camera=camera,timestep=t,total_ms=1000*(time.perf_counter()-tick),
                        mask_pixels=int(np.asarray(mask).sum()),valid_embedding=result is not None,
                        actual_forwards=sum(r['model_forward'] for r in features.calls[before:]),
                        empty_or_patch_support_rejection=result is None))
                    return result
                features.embed=timed_embed
            schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
            actual_inference=tracker.predictor._run_single_frame_inference
            sam_times=[]
            sam_stages=[];native_start=tracker.start;native_track=tracker.track;native_reset=tracker.reset
            def timed_start(*args,**kwargs):
                torch.cuda.synchronize();tick=time.perf_counter()
                result=native_start(*args,**kwargs);torch.cuda.synchronize()
                sam_stages.append(dict(kind='start',total_ms=1000*(time.perf_counter()-tick)))
                return result
            def timed_track(*args,**kwargs):
                iterator=native_track(*args,**kwargs)
                try:
                    while True:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        try:value=next(iterator)
                        except StopIteration:break
                        torch.cuda.synchronize();sam_stages.append(dict(kind='track',total_ms=1000*(time.perf_counter()-tick)))
                        yield value
                finally:iterator.close()
            def timed_reset(*args,**kwargs):
                torch.cuda.synchronize();tick=time.perf_counter()
                result=native_reset(*args,**kwargs);torch.cuda.synchronize()
                sam_stages.append(dict(kind='reset',total_ms=1000*(time.perf_counter()-tick)))
                return result
            tracker.start=timed_start;tracker.track=timed_track;tracker.reset=timed_reset
            def measured(*args,**kwargs):
                torch.cuda.synchronize();tick=time.perf_counter()
                result=actual_inference(*args,**kwargs)
                torch.cuda.synchronize();sam_times.append(1000*(time.perf_counter()-tick))
                return result
            tracker.predictor._run_single_frame_inference=measured
            progress=[]
            try:
                for entry in manifest['entries']:
                    episode=str(entry['episode_id']);folder=root/'perception'/episode
                    if (folder/'receipt.json').exists():
                        receipt=read(folder/'receipt.json')
                        assert receipt['source_config_sha256']==sha(frozen) and receipt['raw_sha256']==entry['raw_sha256']
                        assert all(sha(folder/p)==h for p,h in receipt['files'].items())
                        progress.append(receipt);continue
                    if folder.exists():folder.rename(folder.with_name(episode+'.partial_'+str(time.time_ns())))
                    folder.mkdir(parents=True)
                    assert sha(entry['raw_path'])==entry['raw_sha256']
                    geometry=CurrentGeometry(Path('/workspace/task10g_f1_abc_20261031/full/inputs'),schema)
                    robot_memo=install(geometry) if geometry_cache else None
                    adapters={c:(LosslessAdapter if condition=='P' else OnlineVisionAdapter)(memo,tracker) for c in CAMERAS}
                    for adapter in adapters.values():adapter.reset(episode)
                    if features is not None:features.calls.clear();features.reset(episode)
                    policy=CausalCorrection(memo,tracker,geometry,features=features,uncertain_output_policy='preserve_current',
                        lazy_matching=lazy_matching,correction_slots=[('base_camera',1)] if condition=='BC_MAIN_BOX' else
                            [('hand_camera',0),('base_camera',1)] if condition=='BC_WB_HANDOFF' else None,
                        late_baseline_handoff=condition=='BC_WB_HANDOFF'
                        ) if condition in ('C_KEEP','BC_KEEP','BC_MAIN_BOX','BC_WB_HANDOFF') else None
                    if policy is not None:policy.reset(episode)
                    masks=[];internal_baseline=[];timings=[];records=[];graphs=[];initializations=[]
                    detector.calls.clear();sam_times.clear();sam_stages.clear();feature_events.clear();memo.cache.clear();memo.calls.clear()
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    write(root/'pipeline_state.json',dict(status='RUNNING',episode=episode,condition=condition,
                        completed_episodes=len(progress),pid=os.getpid()))
                    try:
                        with h5py.File(entry['raw_path']) as raw:
                            for t in range(entry['actions']+1):
                                torch.cuda.synchronize();start=time.perf_counter()
                                di=len(detector.calls);si=len(sam_times);fi=len(features.calls) if features is not None else 0
                                ss=len(sam_stages);ff=len(feature_events)
                                cameras,data,surfaces=geometry.from_h5(raw,t,episode)
                                values={};found={}
                                for camera in cameras:
                                    detector.context=dict(episode_id=episode,camera=camera.camera,decision_timestep=t,condition=condition)
                                    mask,record=adapters[camera.camera].process(camera.rgb,episode,t)
                                    values[camera.camera]=mask;records.append(dict(condition=condition,camera=camera.camera,**record))
                                    for model_id in record['new_object_ids']:
                                        name=('peg','box')[model_id-1]
                                        initializations.append(dict(camera=camera.camera,object_id=model_id-1,timestep=t,
                                            **adapters[camera.camera].found[name]))
                                    for oid,name in enumerate(('peg','box')):found[camera.camera,oid]=name in adapters[camera.camera].found
                                baseline_end=time.perf_counter()
                                baseline_snapshot=values
                                if policy is not None:values=policy.process(cameras,data,surfaces,values,found)
                                correction_end=time.perf_counter()
                                graph,provenance,gate=geometry.build(cameras,data,surfaces,values)
                                validate_graph(graph);torch.cuda.synchronize();end=time.perf_counter()
                                for r in provenance:
                                    assert len(r['source_observation_ids'])>0
                                    assert all(str(e)==episode and step==t and not(r['object_id']==1 and c!='base_camera')
                                        for e,step,c,p in r['source_observation_ids'])
                                graphs.append(dict(timestep=t,tensors={k:array_digest(getattr(graph,k).numpy()) for k in
                                    ('node_input','node_type','edge_index','edge_attr','edge_type')}))
                                masks.append(np.packbits(np.stack([values[c] for c in CAMERAS]).ravel()))
                                if policy is not None:internal_baseline.append(np.packbits(np.stack([baseline_snapshot[c] for c in CAMERAS]).ravel()))
                                feature_calls=features.calls[fi:] if features is not None else []
                                timings.append(dict(timestep=t,end_to_end_ms=1000*(end-start),
                                    baseline_and_input_ms=1000*(baseline_end-start),
                                    correction_ms=1000*(correction_end-baseline_end),graph_ms=1000*(end-correction_end),
                                    dino_ms=sum(v['total_ms'] for v in detector.calls[di:]),dino_calls=len(detector.calls)-di,
                                    sam_ms=sum(sam_times[si:]),sam_calls=len(sam_times)-si,
                                    sam_start_track_reset_ms=sum(v['total_ms'] for v in sam_stages[ss:]),
                                    dinov2_forward_ms=sum(v.get('forward_ms',0) for v in feature_calls),
                                    dinov2_feature_total_ms=sum(v['total_ms'] for v in feature_events[ff:]),
                                    dinov2_calls=sum(v['model_forward'] for v in feature_calls),
                                    peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                                    peak_GPU_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
                                if t%20==0:
                                    tmp=folder/'mask_prefix.npz.tmp'
                                    with tmp.open('wb') as stream:np.savez_compressed(stream,mask_bits=np.stack(masks))
                                    tmp.replace(folder/'mask_prefix.npz')
                                    write(folder/'partial_state.json',dict(frames=len(masks),restart_from_t0=True,complete=False))
                        assert len(masks)==entry['actions']+1
                        np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(masks))
                        if policy is not None:np.savez_compressed(folder/'internal_B0.npz',mask_bits=np.stack(internal_baseline))
                        for filename,value in [('vision_records.json',records),('timing.json',timings),
                                ('graph_fingerprints.json',graphs),('detector_timing.json',detector.calls),
                                ('detector_memo_calls.json',memo.calls),('sam_timing.json',sam_times),
                                ('sam_stage_timing.json',sam_stages),('feature_total_timing.json',feature_events),
                                ('original_initialization_prompts.json',initializations)]:write(folder/filename,value)
                        if policy is not None:
                            for filename,value in [('events.json',policy.events),('candidate_masks.json',policy.candidate_masks),
                                    ('reference_provenance.json',policy.reference_provenance),
                                    ('baseline_handoff_events.json',policy.handoffs),
                                    ('state_machine.json',{str(k):v.events for k,v in policy.slots.items()})]:write(folder/filename,value)
                        if features is not None:write(folder/'feature_calls.json',features.calls)
                        receipt=dict(episode=episode,parent_id=entry['parent_id'],frames=len(masks),condition=condition,
                            source_config_sha256=sha(frozen),raw_sha256=entry['raw_sha256'],GT_forward_input=False,
                            geometry_cache=geometry_cache,lazy_matching=lazy_matching,actual_dino_calls=len(detector.calls),
                            actual_sam_calls=len(sam_times),actual_dinov2_calls=sum(v['model_forward'] for v in features.calls) if features else 0,
                            robot_memo_hits=robot_memo.hits if robot_memo else 0,
                            robot_memo_misses=robot_memo.misses if robot_memo else 0,
                            files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
                        write(folder/'receipt.json',receipt);progress.append(receipt);write(root/'progress.json',progress)
                        print('frozen live',condition,len(progress),'/',manifest['episodes'],episode,len(masks),flush=True)
                    finally:
                        if policy is not None:policy.close()
                        for adapter in adapters.values():adapter.close()
                        memo.cache.clear();torch.cuda.empty_cache()
                assert len(progress)==manifest['episodes'] and sum(r['frames'] for r in progress)==manifest['frames']
                write(root/'pipeline_state.json',dict(status='FROZEN_LIVE_CONDITION_COMPLETE',condition=condition,
                    episodes=len(progress),frames=manifest['frames'],run_id=logger.run_id,GT_evaluation_pending=True,
                    model_loading_excluded_from_timing=True,whole_goal_complete=False))
            finally:tracker.predictor._run_single_frame_inference=actual_inference


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('expected',type=Path)
    p.add_argument('--condition',required=True);p.add_argument('--geometry-cache',action='store_true')
    p.add_argument('--lazy-matching',action='store_true');p.add_argument('--engineering',action='store_true');a=p.parse_args()
    try:run(a.root,a.expected,a.condition,a.geometry_cache,a.lazy_matching,a.engineering)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,restart_from_t0=True))
        raise

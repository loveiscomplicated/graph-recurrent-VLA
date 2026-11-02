"""Observer-free online common prefix with separately timed baseline/candidate branches."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(output,mode):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.plan_a_bc.geometry import CurrentGeometry
    from .reliable_reanchor import WristPegReanchor
    from .depth_component import MainPegDepthComponent
    import graph.plan_a_geometry.core as core
    torch.set_num_threads(1);torch.manual_seed(42);output.mkdir(parents=True,exist_ok=False)
    rt=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(rt/'runtime_paths.json')
    schema=read(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'))['schema']
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    choices=['10i_stage3_10i_s1_0362_full_task','10i_stage3_10i_s1_0666_full_task','2000003']
    class TimedDino(GDino):
        total_ms=0.
        def __call__(self,*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter();v=super().__call__(*args,**kwargs);torch.cuda.synchronize()
            self.total_ms+=(time.perf_counter()-start)*1000;return v
    timers=dict(robot=0.,depth_mask=0.,gate=0.,sam2=0.)
    class TimedTracker(SAM2Tracker):
        def start(self,*args,**kwargs):
            torch.cuda.synchronize();s=time.perf_counter();v=super().start(*args,**kwargs);torch.cuda.synchronize()
            timers['sam2']+=(time.perf_counter()-s)*1000;return v
        def track(self,*args,**kwargs):
            iterator=super().track(*args,**kwargs)
            try:
                while True:
                    torch.cuda.synchronize();s=time.perf_counter()
                    try:v=next(iterator)
                    except StopIteration:return
                    torch.cuda.synchronize();timers['sam2']+=(time.perf_counter()-s)*1000
                    yield v
            finally:iterator.close()
    originals={}
    for name,key in [('classify_robot','robot'),('filtered_masks','depth_mask'),('wrist_gate','gate')]:
        originals[name]=getattr(core,name)
        def wrapper(*args,_fn=originals[name],_key=key,**kwargs):
            s=time.perf_counter();v=_fn(*args,**kwargs);timers[_key]+=(time.perf_counter()-s)*1000;return v
        setattr(core,name,wrapper)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_observer_free_latency',seed=42,
         root=output/'experiments',job_type='evaluation',config=dict(mode=mode,choices=choices,frames_per_prefix=20,GT_input=False)) as log:
        detector=TimedDino(device='cuda',model_id=paths['dino']);tracker=TimedTracker(device='cuda',checkpoint=paths['sam']);rows=[]
        try:
            for eid in choices:
                e=entries[eid];bgeo=CurrentGeometry(rt/'inputs',schema);cgeo=CurrentGeometry(rt/'inputs',schema)
                from .perframe_cache import install
                install(bgeo);install(cgeo)
                if mode=='reanchor':policy=WristPegReanchor(detector,tracker,cgeo,refine=True)
                elif mode=='depth_component':policy=MainPegDepthComponent(cgeo)
                else:
                    from .depth_box_component import MainObjectDepthComponents
                    policy=MainObjectDepthComponents(cgeo)
                policy.reset(eid);adapters={c:OnlineVisionAdapter(detector,tracker) for c in CAMERAS}
                for a in adapters.values():a.reset(eid)
                try:
                    with h5py.File(e['raw_path'],'r') as f:
                        for t in range(min(20,e['actions']+1)):
                            s=time.perf_counter();cameras,data,surfaces=bgeo.from_h5(f,t,eid);common_ms=(time.perf_counter()-s)*1000
                            before=dict(timers);dino_before=detector.total_ms;s=time.perf_counter()
                            masks={c.camera:adapters[c.camera].process(c.rgb,eid,t)[0] for c in cameras}
                            torch.cuda.synchronize();vision_ms=(time.perf_counter()-s)*1000
                            base_sam=timers['sam2']-before['sam2'];base_dino=detector.total_ms-dino_before
                            s=time.perf_counter();bgeo.build(cameras,data,surfaces,masks);base_geom=(time.perf_counter()-s)*1000
                            baseline_timers={k:timers[k]-before[k] for k in ('robot','depth_mask','gate')}
                            before=dict(timers);dino_before=detector.total_ms;s=time.perf_counter()
                            changed,event=policy.process(cameras,data,surfaces,masks)
                            torch.cuda.synchronize();correction_ms=(time.perf_counter()-s)*1000
                            s=time.perf_counter();cgeo.build(cameras,data,surfaces,changed);candidate_geom=(time.perf_counter()-s)*1000
                            candidate_timers={k:timers[k]-before[k] for k in ('robot','depth_mask','gate')}
                            rows.append(dict(episode=eid,timestep=t,common_rgbd_fk_ms=common_ms,vision_ms=vision_ms,
                                baseline_end_to_end_ms=common_ms+vision_ms+base_geom,
                                candidate_end_to_end_ms=common_ms+vision_ms+correction_ms+candidate_geom,
                                baseline_dino_ms=base_dino,baseline_sam2_ms=base_sam,
                                extra_dino_ms=detector.total_ms-dino_before,extra_sam2_ms=timers['sam2']-before['sam2'],
                                baseline_geometry_graph_ms=base_geom,candidate_geometry_graph_ms=candidate_geom,
                                correction_ms=correction_ms,baseline_components_ms=baseline_timers,
                                candidate_components_ms=candidate_timers,correction_used=event['correction_used']))
                finally:
                    policy.close()
                    for a in adapters.values():a.close()
            summaries={}
            for scope,rr in [('all_60_frames',rows),('after_frame4',[r for r in rows if r['timestep']>=5])]:
                summaries[scope]={k:dict(mean=float(np.mean([r[k] for r in rr])),p95=float(np.percentile([r[k] for r in rr],95))) for k in rows[0] if k.endswith('_ms') and not isinstance(rows[0][k],dict)}
                summaries[scope]['candidate_to_baseline_p95_ratio']=summaries[scope]['candidate_end_to_end_ms']['p95']/summaries[scope]['baseline_end_to_end_ms']['p95']
            write(output/'latency.json',dict(status='OBSERVER_FREE_LATENCY_FINISHED',mode=mode,frames=len(rows),summaries=summaries,rows=rows,
                cpu_thread_environment={k:__import__('os').environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                raw_query_observer=False,GT_metrics_inside_timed_path=False,graph_artifact_IO_inside_timed_path=False,exact_current_frame_memo_both_branches=True,
                jpeg_adapter_IO_included=True,common_vision_forward_measured_once=True,
                scope='fixed three Development causal20-frame prefixes, initial and later separately; not all-validation latency',
                runtime_20hz_not_assumed=True))
            log.link_output(output/'latency.json',kind='metrics')
        finally:
            for name,fn in originals.items():setattr(core,name,fn)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--mode',choices=('reanchor','depth_component','depth_component_both'),required=True)
    a=p.parse_args();run(a.output,a.mode)

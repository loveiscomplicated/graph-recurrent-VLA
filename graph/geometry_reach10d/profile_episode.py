"""Bounded real-offset per-stage CUDA vision/CPU geometry profiling."""
import argparse
from pathlib import Path
import time
import h5py
import torch
from graph.plan_a_bc.common import read,write,append_csv
from graph.plan_a_bc.cache import make_vision
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter,ObservedFrames
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from graph.geometry_reach10d.profile import Resources


def run(root):
    torch.set_num_threads(1);torch.manual_seed(42)
    paths=read(root/'runtime_paths.json');dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    rows=[];context={'t':0}
    def measured(stage,fn):
        torch.cuda.synchronize();begin=time.perf_counter();out=fn();torch.cuda.synchronize()
        row=dict(stage=stage,episode_id=1000001,t=context['t'],seconds=time.perf_counter()-begin)
        rows.append(row);append_csv(root/'pipeline_profile.csv',row);return out
    class TimedDetector:
        def __call__(self,*args,**kwargs):return measured('DINO_detection_CUDA',lambda:dino(*args,**kwargs))
    detector=DetectionMemo(TimedDetector(),'stage_profile',read(root/'optimization_equivalence.json')['model_identity'])
    old_track=tracker.track
    def track(*args,**kwargs):
        iterator=old_track(*args,**kwargs)
        try:
            while True:
                try:out=measured('SAM2_tracking_CUDA',lambda:next(iterator))
                except StopIteration:return
                yield out
        finally:iterator.close()
    tracker.track=track
    old_append=ObservedFrames.append_jpeg
    def append(self,*args,**kwargs):return measured('image_tensor_processing_transfer_CUDA',lambda:old_append(self,*args,**kwargs))
    ObservedFrames.append_jpeg=append
    import graph.plan_a_geometry.core as core
    core.components=components
    geometry=CurrentGeometry(root/'inputs',read(root/'normalizer.json')['schema'])
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ['base_camera','hand_camera']}
    for v in adapters.values():v.reset('1000001')
    begin=time.perf_counter()
    try:
        with Resources(root,'bounded_stage_profile'),h5py.File(root/'raw/1000001/episode.h5') as f:
            for t in range(8):
                context['t']=t
                cameras,data,surfaces=measured('camera_FK_source_CPU',lambda:geometry.from_h5(f,t,'1000001'))
                masks={c.camera:adapters[c.camera].process(c.rgb,'1000001',t)[0] for c in cameras}
                measured('full_geometry_graph_CPU',lambda:geometry.build(cameras,data,surfaces,masks))
        write(root/'vision_stage_profile.json',dict(rows=rows,frames=8,wall_seconds=time.perf_counter()-begin,
            model_loaded_once=True,DINO_memo_hits=sum(r['hit'] for r in detector.trace),
            DINO_memo_calls=len(detector.trace),precision='same FP32, no AMP',
            note='synchronization used for attribution; bounded profile ran concurrently with production cache'),immutable=True)
    finally:
        for v in adapters.values():v.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

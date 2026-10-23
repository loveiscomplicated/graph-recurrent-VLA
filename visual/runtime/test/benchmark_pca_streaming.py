"""Causal SAM2 -> PCA5 timing, excluding cached initial grounding and startup.

Only currently implemented geometry is produced; final Graph sampling/edges
are deliberately not fabricated. Reports queue growth at 20Hz without drops.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_logger import ExperimentLogger
from utils.experiment_artifacts import digests
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.tracking import SAM2Tracker
from visual.scripts.stream_v1 import CausalFrames, StreamTracker
from visual.runtime.test.pca_optimized import process_optimized
from visual.runtime.test.pca_torch import TorchPCA5
from visual.runtime.test.normal_diagnostics import stats
from visual.runtime.test.run_normal_diagnostics import save_json


class DeviceStreamTracker(StreamTracker):
    """Reuse causal tracking operations; supply the predictor's actual device.

    The legacy StreamTracker initialization hardcodes CausalFrames' CUDA
    default. This harness only adapts its frame-buffer construction for CPU.
    """
    def _init_state(self, rgb0):
        mod = importlib.import_module(type(self.p).__module__)
        h, w = rgb0.shape[:2]
        frames = CausalFrames(self.p.image_size, device=self.p.device, keep=self.keep_frames, preprocess=self.pre)
        frames.push(rgb0)
        original = mod.load_video_frames
        mod.load_video_frames = lambda **kw: (frames, h, w)
        try: state = self.p.init_state(video_path='<stream>')
        finally: mod.load_video_frames = original
        self.frames = frames
        return state


def benchmark(input_path, masks_path, tracking_path, checkpoint, *, device, count, output_root, backend='eigh', compile_image_encoder=False, cpu_parallel=False, encoder_warmup=0, tracker_warmup_steps=0):
    if count < 2: raise ValueError('count must include startup and at least one subsequent frame')
    input_path, masks_path, tracking_path, checkpoint = map(Path, (input_path,masks_path,tracking_path,checkpoint))
    frames = list(iter_saved_rgbd(input_path,masks_path))[:count]
    tracking = json.loads(tracking_path.read_text())
    boxes = {i+1:tracking['initial_boxes'][name]['box'] for i,name in enumerate(('peg','box'))}
    cfg = dict(task='pca5_streaming_budget',device=device,backend=backend,compile_image_encoder=compile_image_encoder,cpu_parallel=cpu_parallel,encoder_warmup=encoder_warmup,tracker_warmup_steps=tracker_warmup_steps,warmup_uses_only_first_rgb=True,frames=len(frames),target_period_ms=50,
               initial_grounding='cached initial boxes; grounding not run or timed',startup_excluded=True,
               checkpoint_sha256=digests(checkpoint)[1], input_sha256=digests(input_path)[1],
               tracker_config=tracking['config'],preprocess='gpu',initial_boxes=boxes,
               final_graph_handoff_implemented=False,frame_drops=0,geometry='optimized_analytic' if device=='cpu' else 'torch_float64_pca5',
               clock='scheduled 50ms arrivals; all frames processed causally; queue delay included in arrival-to-geometry-ready',
               source_sha256={str(p.relative_to(REPO)):digests(p)[1] for p in [Path(__file__),REPO/'visual/scripts/stream_v1.py',REPO/'visual/runtime/test/pca_torch.py',REPO/'visual/runtime/test/pca_optimized.py']})
    with ExperimentLogger(team='visual',config=cfg,seed=0,model='pca5_causal_stream',job_type='evaluate') as logger:
        out=Path(output_root)/logger.run_id;out.mkdir(parents=True,exist_ok=False);print('Output:',out,flush=True)
        if device=='cuda' and not torch.cuda.is_available():
            save_json(out/'streaming.json',dict(run_id=logger.run_id,config=cfg,status='not_executed',reason='CUDA unavailable'))
            logger.link_output(out/'streaming.json',kind='metrics');logger.record(gpu_executed=False);return out
        torch.set_num_threads(4)
        start=time.perf_counter()
        overrides=['++model.compile_image_encoder=True'] if compile_image_encoder else None
        predictor=SAM2Tracker(device=device,checkpoint=checkpoint,config=tracking['config'],hydra_overrides=overrides).predictor
        tracker=DeviceStreamTracker(predictor,preprocess='gpu',prune=True,keep_mem=16,keep_frames=2)
        processor=TorchPCA5(frames[0].intrinsic,frames[0].depth.shape,device=device,backend=backend) if device=='cuda' else None
        autocast = torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device=='cuda')
        rows=[];saved_masks=[]
        def sync():
            if device=='cuda':torch.cuda.synchronize()
        with torch.inference_mode(),autocast:
            masks=tracker.start(frames[0].rgb,boxes)
            # Warm only the already available first RGB, without advancing or
            # resetting tracker state or accessing a future observation.
            for _ in range(encoder_warmup): predictor.image_encoder(tracker.frames[0][None])
            for _ in range(tracker_warmup_steps): tracker.step(frames[0].rgb)
            if tracker_warmup_steps:
                tracker.reset()
                masks=tracker.start(frames[0].rgb,boxes)
                assert tracker.frames.latest==0  # fresh actual observation state
            if device=='cpu':process_optimized(replace(frames[0],masks={i-1:m.cpu().numpy() for i,m in masks.items()}),solver='analytic',parallel=cpu_parallel)
            else:processor(torch.as_tensor(frames[0].depth,device='cuda'),{i-1:m for i,m in masks.items()})
            sync();startup=(time.perf_counter()-start)*1000
            epoch=time.perf_counter()
            for idx,frame in enumerate(frames[1:]):
                arrival=epoch+idx*.05
                delay=arrival-time.perf_counter()
                if delay>0:time.sleep(delay)
                sync();begin=time.perf_counter()
                fi,masks=tracker.step(frame.rgb);sync();after_tracking=time.perf_counter()
                if device=='cpu':
                    masks={i-1:m.detach().cpu().numpy() for i,m in masks.items()};after_transfer=time.perf_counter()
                    geometry=process_optimized(replace(frame,masks=masks),solver='analytic',parallel=cpu_parallel)
                    points=sum(int(f.valid_points.sum()) for f in geometry.objects.values())
                else:
                    depth=torch.as_tensor(frame.depth,device='cuda'); masks={i-1:m for i,m in masks.items()};sync();after_transfer=time.perf_counter()
                    geometry=processor(depth,masks)
                    points=None  # avoid a timed host reduction/synchronization for reporting
                sync();end=time.perf_counter()
                saved_masks.append(masks)
                rows.append(dict(source_frame=frame.source_frame,tracker_frame=fi,queue_wait_ms=(begin-arrival)*1000,
                                 tracking_ms=(after_tracking-begin)*1000,transfer_ms=(after_transfer-after_tracking)*1000,
                                 geometry_ms=(end-after_transfer)*1000,service_ms=(end-begin)*1000,
                                 arrival_to_geometry_ready_ms=(end-arrival)*1000,observed_points=points))
                print('Stream frame',frame.source_frame,'service_ms',round(rows[-1]['service_ms'],2),flush=True)
            elapsed=time.perf_counter()-epoch
        tracker.reset()
        # Keep references during timing; transfer/serialize the masks afterward.
        arrays={name:np.stack([(m[i].detach().cpu().numpy() if device=='cuda' else m[i]) for m in saved_masks]) for name,i in [('peg',0),('box',1)]}
        np.savez_compressed(out/'tracked_masks.npz',source_frame=np.array([r['source_frame'] for r in rows]),**arrays)
        logger.link_output(out/'tracked_masks.npz',kind='trajectory')
        keys=['tracking_ms','transfer_ms','geometry_ms','service_ms','queue_wait_ms','arrival_to_geometry_ready_ms']
        summary={k:stats([r[k] for r in rows]) for k in keys}
        data=dict(run_id=logger.run_id,config=cfg,status='executed',startup_excluded_ms=startup,summary=summary,frames=rows,
                  achieved_hz=len(rows)/elapsed,service_over_50ms=sum(r['service_ms']>50 for r in rows),
                  arrival_deadline_misses=sum(r['arrival_to_geometry_ready_ms']>50 for r in rows),
                  final_graph_handoff_measured=False,scope='RGB available to implemented geometry ready; sampling/edges/Graph packaging absent')
        save_json(out/'streaming.json',data);logger.link_output(out/'streaming.json',kind='metrics')
        logger.log(summary,prefix='eval/streaming');logger.record(gpu_executed=device=='cuda',initial_grounding_excluded=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=REPO/'visual/data/step2/ep000.npz')
    p.add_argument('--masks',type=Path,default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    p.add_argument('--tracking',type=Path,default=REPO/'visual/results/runtime_tracking/ep000/tracking.json')
    p.add_argument('--checkpoint',type=Path,default=REPO/'visual/checkpoints/sam2.1_hiera_small.pt')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--frames',type=int,default=12)
    p.add_argument('--backend',choices=['eigh','triton','triton_fused'],default='eigh')
    p.add_argument('--compile-image-encoder',action='store_true')
    p.add_argument('--cpu-parallel',action='store_true')
    p.add_argument('--encoder-warmup',type=int,default=0)
    p.add_argument('--tracker-warmup-steps',type=int,default=0)
    p.add_argument('--out',type=Path,default=REPO/'visual/results/pca_optimization')
    a=p.parse_args();print(benchmark(a.input,a.masks,a.tracking,a.checkpoint,device=a.device,count=a.frames,output_root=a.out,backend=a.backend,compile_image_encoder=a.compile_image_encoder,cpu_parallel=a.cpu_parallel,encoder_warmup=a.encoder_warmup,tracker_warmup_steps=a.tracker_warmup_steps))


if __name__=='__main__':main()

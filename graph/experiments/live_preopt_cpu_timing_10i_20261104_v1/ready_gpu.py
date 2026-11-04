"""Matched mask-ready endpoint, excluding SAM2 and including RF preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.experiments.live_preopt_cpu_timing_10i_20261104_v1.legacy import load,read,write,stats,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema','live'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert os.environ.get('CONTAINER_ID')=='58752018'
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.persistent_geometry.tensor_camera import tensor_camera
    from graph.persistent_geometry.tensor_graph import FIELDS
    manifest=read(a.inputs/'manifest.json');reference=read(a.live/'natural/measurement.json')['rows']
    staged=[]
    for row in manifest['frames']:
        file=a.inputs/row['file'];assert sha(file)==row['sha256']
        cameras,data,surfaces,masks=load(file)
        staged.append(([tensor_camera(c) for c in cameras],data,surfaces,
                       {k:torch.as_tensor(v,device='cuda') for k,v in masks.items()}))
    torch.cuda.synchronize()
    geometry=CurrentGeometry(a.assets,read(a.schema)['schema'],coordinate_frame='ee')
    config=dict(task='10i_matched_mask_ready_GPU_20261104',input_manifest_sha256=sha(a.inputs/'manifest.json'),
        boundary='resident current RGB/depth/raw masks plus CPU robot/calibration -> actual default CurrentGeometry.build -> resident Graph/records',
        preparation_included=True,SAM2_excluded=True,input_upload_outside_timer=True,
        note='unhidden robot preparation/join; one full trace warmup, then same39 inputs, t9..38 summary')
    with ExperimentLogger(team='graph',model='mask_ready_GPU_comparison',job_type='diagnostic',seed=42,config=config,root=a.output/'experiments') as log:
        try:
            for cameras,data,surfaces,masks in staged:
                geometry.build(cameras,data,surfaces,lambda:masks);torch.cuda.synchronize()
            rows=[]
            for t,(cameras,data,surfaces,masks) in enumerate(staged):
                torch.cuda.synchronize();start=time.perf_counter_ns()
                graph,records,gate=geometry.build(cameras,data,surfaces,lambda:masks)
                torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
                assert records.xyz.is_cuda and all(getattr(graph,k).is_cuda for k in FIELDS)
                hashes={k:hashlib.sha256(getattr(graph,k).cpu().numpy().tobytes()).hexdigest() for k in FIELDS}
                assert hashes==reference[t]['graph_sha256'],('GPU ready/live mismatch',t)
                rows.append(dict(timestep=t,ms=elapsed,Graph_matches_live=True))
            result=dict(status='PASS',run_id=log.run_id,config=config,rows=rows,summary=stats([r['ms'] for r in rows[9:]]))
            write(a.output/'summary.json',result);log.link_output(a.output/'summary.json')
            print('GPU_READY_COMPLETE',log.run_id,json.dumps(result['summary']),flush=True)
        finally:geometry.close()


if __name__=='__main__':main()

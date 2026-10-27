"""Measured CPU stage costs and CUDA diagnostic inference, with resource gates."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json
from graph.integration_baseline.graph import pack_graphs
from graph.persistent_geometry.train import load_graph,parameter_hash
from .readout_adapter import CanonicalRegressor as GeometryRegressor
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.self_filter import LinkSurface,classify_robot
from graph.persistent_geometry.fusion import filtered_masks
from graph.persistent_geometry.map import PersistentMap
from graph.persistent_geometry.graph import build_graph
from .adapter import frame_points
from .learning import gpu_gate,initialize_gpu_identity
from .benchmark import BASE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',type=Path,required=True);parser.add_argument('--learning',type=Path);args=parser.parse_args()
    m=json.loads((args.geometry/'geometry_manifest.json').read_text());capture=Path(m['config']['capture']);src=capture/'collection'/'seed30002'
    cameras,data=load_cameras(src/'f0000_input.npz');main_name=json.loads((src/'sensor_contract.json').read_text())['roles']['main']
    main=next(c for c in cameras if c.camera==main_name)
    with np.load(src/'f0000_masks.npz') as a:masks=a[main.camera]
    with np.load(src/'robot_meshes.npz') as a:links=[LinkSurface(name,a[f'vertices_{i}'],pose) for i,(name,pose) in enumerate(zip(a['links'].tolist(),data['robot_link_world']))]
    torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=dict(stage='latency',scope='seed30002t0 fixed shape; fresh map initialization, not steady-state tracking',geometry=str(args.geometry),learning=str(args.learning)),seed=42,model='legacy_fps_stage_costs',job_type='evaluate',root=BASE/'latency_runs') as logger:
        root=logger.directory;(BASE/'latest_latency.txt').write_text(str(root)+'\n');rows=[]
        for i in range(6):
            tick=time.perf_counter();filtered=classify_robot(main,links);filter_ms=(time.perf_counter()-tick)*1000
            clean,_=filtered_masks(masks,filtered)
            tick=time.perf_counter();obs,pts,_=frame_points(main,clean,f5=False);extract_ms=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();obs,_,_=frame_points(main,clean,f5=True);extract_f5_ms=(time.perf_counter()-tick)*1000
            memory=PersistentMap();tick=time.perf_counter();memory.update(main.episode,0,obs,cameras=[main]);map_ms=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();g=build_graph(memory.records(),data['robot_world_geometry'],data['tcp_pose'],0,cameras=[main],point_cap=128);graph_ms=(time.perf_counter()-tick)*1000
            rows.append(dict(repetition=i,warmup=i==0,filter_ms=filter_ms,legacy_extract_ms=extract_ms,legacy_extract_plus_F5_ms=extract_f5_ms,map_initialize_ms=map_ms,graph_ms=graph_ms,native_input_count=sum(len(p.source_pixel) for p in pts),graph_nodes=len(g.node_input)))
        write_json(root/'cpu_stage_costs.json',rows)
        inference=[]
        if args.learning:
            initialize_gpu_identity()
            mm=json.loads((args.learning/'model_manifest.json').read_text());frame=m['frames'][0];folder=args.geometry/frame['folder']
            for info in mm['models']:
                gpu_gate();checkpoint=torch.load(args.learning/info['directory']/'selected.pt',map_location='cpu',weights_only=False)
                model=GeometryRegressor(info['encoder']).cuda().eval().requires_grad_(False);model.load_state_dict(checkpoint['state_dict'])
                graph=pack_graphs([load_graph(folder/(info['variant']+'_graph.npz'))]).to('cuda')
                times=[]
                with torch.inference_mode():
                    for i in range(25):
                        torch.cuda.synchronize();start=time.perf_counter();model.encoder(graph);torch.cuda.synchronize()
                        if i>=5:times.append((time.perf_counter()-start)*1000)
                inference.append(dict(variant=info['variant'],encoder=info['encoder'],mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),repetitions=len(times),nodes=len(graph.node_input),CUDA_peak_bytes=torch.cuda.max_memory_allocated()))
                del model,graph;torch.cuda.empty_cache()
        write_json(root/'gnn_inference_costs.json',inference)
        logger.link_output(root/'cpu_stage_costs.json',kind='metrics');logger.link_output(root/'gnn_inference_costs.json',kind='metrics')
if __name__=='__main__':main()

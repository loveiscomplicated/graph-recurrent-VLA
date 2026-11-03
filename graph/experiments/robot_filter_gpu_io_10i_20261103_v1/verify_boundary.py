"""Correctness-only integration check. No latency/model/rollout execution."""
from pathlib import Path
import os,sys,json,unittest
HERE=Path(__file__).resolve().parent;OVERLAY=HERE.parents[2]
OLD=Path('/workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4')
sys.path.insert(0,str(OLD))
from run import C,FROZEN,geometry,load_inputs,masks_for,Candidate,ExperimentLogger,np,torch,exact,graph_compare
sys.path.insert(0,str(OVERLAY))
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.graph_boundary import GpuMaskBoundary


def build(geo,inputs,masks,gpu):
    with Candidate(9):
        if gpu:
            with GpuMaskBoundary() as boundary:
                graph,records,gate=geo.build(*inputs[:3],masks)
                assert all(m.is_cuda for m in boundary.clean_cuda.values())
        else:graph,records,gate=geo.build(*inputs[:3],masks)
    geo.reset()
    return dict(graph={k:getattr(graph,k).numpy().copy() for k in C.FIELDS},records=records,
        sources=[r['source_observation_ids'] for r in records],gate=gate,packet=graph.packet)


def main():
    output=Path(os.environ['TASK10I_GPU_IO_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);scene='10i_s1_0924_t080'
    spec=next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id']==scene)
    rows=[]
    with ExperimentLogger(team='integration',model='task10i_gpu_mask_graph_boundary',job_type='evaluate',seed=42,root=output/'experiments',config=dict(scene=scene,sizes=[128,512,768,1024],scope='correctness only; Graph implementation and rules unchanged')) as log:
        suite=unittest.defaultTestLoader.loadTestsFromName('graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_graph_boundary')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise AssertionError('boundary test failed')
        for size in (128,512,768,1024):
            geo=geometry();inputs=load_inputs(scene,size,geo);masks=masks_for(spec,size,0,'actual')
            reference=build(geo,inputs,masks,False)
            actual=build(geo,inputs,{k:torch.as_tensor(v,device='cuda') for k,v in masks.items()},True)
            comparison=graph_compare(actual,reference)
            exact(actual['records'],reference['records']);exact(actual['packet'],reference['packet'],'Graph.packet')
            rows.append(dict(scene=scene,size=size,comparison=comparison,all_records_exact=True,graph_packet_exact=True))
            print(json.dumps(rows[-1]),flush=True)
        path=output/'results.json';C.write(path,dict(status='PASS',rows=rows));log.link_output(path)


if __name__=='__main__':main()

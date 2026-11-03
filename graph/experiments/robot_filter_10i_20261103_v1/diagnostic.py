"""Diagnostic-only synchronized spans and operator/transfer counts, never latency."""
from run import *
from filter_benchmark import pair
from functools import wraps


def main_diagnostic(a):
    torch.set_num_threads(4);output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    geo=geometry();scene=a.scene;size=a.size
    spec=next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id']==scene)
    inputs=load_inputs(scene,size,geo);masks=masks_for(spec,size,0,'actual')
    rows=[]
    config=dict(identity=identity(),scene=scene,size=size,scope='instrumented Graph-only diagnostic; synchronization perturbs timing')
    C.write(output/'config.json',config)
    with ExperimentLogger(team='integration',model='task10i_robot_filter_diagnostic',job_type='evaluate',seed=42,root=output/'experiments',config=config) as log:
        for variant in a.variants:
            rays=RayCache();execute(geo,inputs,masks,variant,rays)
            import graph.plan_a_geometry.core as core
            spans=[]
            with context(variant,rays) as ctx:
                for name in ('classify_robot','filtered_masks','wrist_gate','make_packet','deduplicate','budget','build_graph'):
                    original=getattr(core,name)
                    def hook(*args,_original=original,_name=name,**kw):
                        torch.cuda.synchronize();start=time.perf_counter()
                        result=_original(*args,**kw)
                        torch.cuda.synchronize();spans.append(dict(name=_name,ms=(time.perf_counter()-start)*1000))
                        return result
                    ctx.patch(core,name,hook)
                torch.cuda.synchronize();start=time.perf_counter()
                graph,records,gate=geo.build(*inputs[:3],masks);torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000;geo.reset()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=True) as prof:
                measured=pair(inputs,masks,variant,rays)
            ops=[]
            for event in prof.key_averages():
                ops.append(dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,
                    self_device_us=getattr(event,'self_device_time_total',0),cpu_memory=event.cpu_memory_usage,device_memory=getattr(event,'device_memory_usage',0)))
            prof.export_chrome_trace(str(output/f'{variant}_filter_trace.json'))
            rows.append(dict(variant=variant,graph_synchronized_instrumented_ms=elapsed,spans=spans,
                filter_profiled_ms=measured['ms'],metrics=measured['metrics'],operators=ops))
        C.write(output/'diagnostic.json',dict(status='PASS',rows=rows));log.link_output(output/'diagnostic.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',default='10i_s1_0044_t080');p.add_argument('--size',type=int,default=1024);p.add_argument('--variants',nargs='+',default=['baseline','hulls','strict'])
    main_diagnostic(p.parse_args())

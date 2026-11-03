"""Bounded paired validation/latency driver; only saved actual masks are loaded."""
import argparse
from pathlib import Path
import os,sys,time,json,hashlib,platform,resource,signal,shutil
HERE=Path(__file__).resolve().parent
BASE=Path(os.environ['TASK10I_LATENCY_CODE'])
REF=Path(os.environ['TASK10I_REFERENCE_CODE'])
FROZEN=Path(os.environ['TASK10I_ROOT'])
sys.path[:0]=[str(FROZEN/'code'),str(BASE),str(REF)]
import common as C
import numpy as np
import torch
from graph_adapter import geometry,load_inputs
from graph_worker import masks_for
from candidates import Candidate
from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.self_filter import classify_robot as cpu_filter
sys.path.insert(0,str(HERE))
from adapter import SparseGraph
from roi_filter import RobotFilter,RayCache,filtered_masks
from benchmark import compare as graph_compare


def exact(a,b,path='records'):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    elif isinstance(a,float) and np.isnan(a):assert np.isnan(b),path
    else:assert a==b,(path,a,b)


def context(variant,rays):
    return Candidate(9) if variant in ('baseline','cpu') else SparseGraph(variant,rays)


def execute(geo,inputs,masks,variant,rays,capture=False):
    import graph.plan_a_geometry.core as core
    captured={}
    with context(variant,rays) as ctx:
        if variant=='cpu':ctx.patch(core,'classify_robot',cpu_filter)
        if capture:
            prepare=core.PlanA.prepare
            def observed(system,*a,**kw):
                out=prepare(system,*a,**kw)
                captured.update(clean=out['clean'],conditions=out['conditions'],gate=out['gate'])
                return out
            ctx.patch(core.PlanA,'prepare',observed)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if Path('/proc/self/clear_refs').exists():Path('/proc/self/clear_refs').write_text('5\n')
        signal.alarm(180)
        try:
            start=time.perf_counter();graph,records,gate=geo.build(*inputs[:3],masks)
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
        finally:signal.alarm(0);geo.reset()
        metrics=getattr(getattr(ctx,'robot_filter',None),'last_metrics',None)
    arrays={k:getattr(graph,k).numpy().copy() for k in C.FIELDS}
    return dict(ms=ms,graph=arrays,records=records,sources=[r['source_observation_ids'] for r in records],gate=gate,
        captured=captured,metrics=metrics,peak_rss_bytes=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def compare(a,b):
    result=graph_compare(a,b)
    captured=bool(a['captured'] and b['captured'])
    if captured:
        exact(a['records'],b['records'])
        exact(a['captured'],b['captured'],'prepared')
    return dict(**result,all_record_fields_compared=captured)


def pixel_check(inputs,masks,variants):
    checks=[]
    with Candidate(9) as ctx:
        # No CurrentGeometry.build here: explicitly restrict build-local caches.
        for camera in inputs[0]:
            expected=__import__('graph.plan_a_geometry.core',fromlist=['classify_robot']).classify_robot(camera,inputs[2])
            for variant in variants:
                filt=RobotFilter(variant);out=filt.classify(camera,inputs[2],masks[camera.camera],diagnostics=True)
                p=out.pixels
                for key,value in (('confirmed_robot',out.confirmed),('uncertain',out.uncertain)):
                    np.testing.assert_array_equal(value,expected[key].ravel()[p],err_msg=key)
                for key in ('candidate','link_owner','inside_known_link_volume'):
                    np.testing.assert_array_equal(out.diagnostics[key],expected[key].ravel()[p],err_msg=key)
                a,b=out.diagnostics['predicted_robot_depth'],expected['predicted_robot_depth'].ravel()[p]
                np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
                clean,_=filtered_masks(masks[camera.camera],out)
                from graph.persistent_geometry.fusion import filtered_masks as full
                np.testing.assert_array_equal(clean,full(masks[camera.camera],expected)[0])
                finite=np.isfinite(a)&np.isfinite(b)
                checks.append(dict(camera=camera.camera,variant=variant,pixels=len(p),flags_owner_interior_exact=True,
                    depth_max_abs=float(np.max(np.abs(a[finite]-b[finite]),initial=0)),metrics=filt.last_metrics))
    return checks


def identity():
    import scipy,triton,subprocess
    roots=[HERE,BASE,REF,FROZEN/'code']
    repo=Path(os.environ['TASK10I_REPO']).resolve()
    loaded={str(p.relative_to(repo)):C.sha(p) for module in list(sys.modules.values()) if (filename:=getattr(module,'__file__',None)) and (p:=Path(filename)).suffix=='.py' and p.is_relative_to(repo) and p.is_file()}
    return dict(cgroup_runtime={p:Path('/sys/fs/cgroup',p).read_text().strip() for p in ('cpu.stat','cpu.pressure','memory.current','memory.events') if Path('/sys/fs/cgroup',p).exists()},loaded_repo_sources=loaded,host=platform.platform(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),scipy=scipy.__version__,numpy=np.__version__,triton=triton.__version__,
        gpu_driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        affinity=sorted(os.sched_getaffinity(0)),quota={p:Path('/sys/fs/cgroup',p).read_text().strip() for p in ('cpu.max','memory.max') if Path('/sys/fs/cgroup',p).exists()},
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        sources={str(p):C.sha(p) for root in roots for p in root.iterdir() if p.suffix in ('.py','.cpp')},
        frozen_protocol_sha256=C.sha(FROZEN/'protocol.json'),gc_enabled=__import__('gc').isenabled())


def main(a):
    setup_start=time.perf_counter()
    C.assert_source();torch.set_num_threads(a.threads)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    def deadline(*_):raise TimeoutError('180s per build')
    signal.signal(signal.SIGALRM,deadline)
    import native
    native.library()
    cuda_start=time.perf_counter();torch.empty(0,device='cuda');torch.cuda.synchronize()
    cuda_setup_ms=(time.perf_counter()-cuda_start)*1000
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    cases=C.read(a.cases)['cases'];specs={s['id']:s for s in C.read(FROZEN/'protocol.json')['scenes']}
    config=dict(cuda_setup_ms=cuda_setup_ms,args=vars(a),identity=identity(),cases_sha256=C.sha(a.cases),
        boundary='CPU sensor arrays + raw actual masks + precomputed current robot surfaces -> CPU final Graph/records/gate; internal transfers included; excludes I/O,decode,join,FK,models,action,GT',
        cache='Only rays variant retains content-keyed O(H+W) calibration vectors, max 8; all world geometry/depth/current extrinsics renewed per build')
    config['setup_ms']=(time.perf_counter()-setup_start)*1000
    C.write(output/'config.json',config);shutil.copytree(HERE,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    rows=[];rays={v:RayCache() for v in a.variants}
    with ExperimentLogger(team='integration',model='task10i_robot_filter_optimization',job_type='evaluate',seed=42,root=output/'experiments',config=config) as log:
        for index,case in enumerate(cases):
            scene,size=case['scene'],case['size'];geo=geometry();inputs=load_inputs(scene,size,geo)
            masks=masks_for(specs[scene],size,0,'actual')
            if case.get('mask_mode')=='full':masks={k:np.stack((np.ones_like(m[0]),np.zeros_like(m[1]))) for k,m in masks.items()}
            tag=f'{scene}_{size}_{case.get("mask_mode","actual")}'
            inp=dict(case=case,masks={k:C.array_sha(v) for k,v in masks.items()},
                cameras={c.camera:{k:C.array_sha(getattr(c,k)) for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','tcp_pose')} for c in inputs[0]},
                surfaces=[dict(link=s.link,local=C.array_sha(s.local_vertices),pose=C.array_sha(s.world_from_link),world=C.array_sha(s.world_vertices)) for s in inputs[2]])
            C.write(output/'inputs'/f'{tag}.json',inp)
            reference=execute(geo,inputs,masks,'baseline',RayCache(),capture=True)
            validation={}
            for variant in a.variants:
                got=execute(geo,inputs,masks,variant,rays[variant],capture=True)
                validation[variant]=dict(comparison=compare(got,reference),metrics=got['metrics'],first_validation_ms=got['ms'],reference_first_ms=reference['ms'])
                folder=output/'graphs'/tag/variant;folder.mkdir(parents=True)
                np.savez_compressed(folder/'graph.npz',**got['graph'])
                C.write(folder/'sources_gate.json',dict(sources=got['sources'],gate=got['gate']))
            if a.pixels:validation['pixels']=pixel_check(inputs,masks,[v for v in a.variants if v not in ('baseline','cpu')])
            C.write(output/'validation'/f'{tag}.json',validation)
            if a.repeats:
                cold=[]
                for variant in a.variants:
                    rays[variant]=RayCache() # calibration-cold, CUDA/JIT may be warm
                    first=execute(geo,inputs,masks,variant,rays[variant]);compare(first,reference)
                    cold.append(dict(variant=variant,ms=first['ms'],ray_misses=rays[variant].misses))
                C.write(output/'cold'/f'{tag}.json',cold)
                for _ in range(a.warmups):
                    for variant in a.variants:compare(execute(geo,inputs,masks,variant,rays[variant]),reference)
                for repeat in range(a.repeats):
                    order=a.variants if (repeat+index)%2==0 else a.variants[::-1]
                    for variant in order:
                        result=execute(geo,inputs,masks,variant,rays[variant]);comparison=compare(result,reference)
                        row=dict(case=case,variant=variant,repeat=repeat,comparison=comparison,
                            **{k:result[k] for k in ('ms','peak_rss_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')})
                        rows.append(row)
            C.write(output/'results.json',dict(status='PARTIAL',completed=index+1,total=len(cases),rows=rows))
            print(json.dumps(dict(completed=index+1,total=len(cases),case=case,
                median_ms={v:float(np.median([r['ms'] for r in rows if r['case']==case and r['variant']==v])) for v in a.variants} if a.repeats else {},accuracy='PASS')),flush=True)
            log.log(dict(cases_complete=index+1),step=index+1)
        C.write(output/'runtime_after.json',identity())
        path=output/'results.json';C.write(path,dict(status='PASS',completed=len(cases),rows=rows))
        log.link_output(path);log.link_output(output/'config.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--variants',nargs='+',default=['baseline','strict']);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=15);p.add_argument('--pixels',action='store_true')
    main(p.parse_args())

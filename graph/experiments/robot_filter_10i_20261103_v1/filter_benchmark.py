"""Supplementary two-camera filter->CPU clean masks, separate from Graph timing."""
from run import *
from graph.persistent_geometry.fusion import filtered_masks as full_masks


def pair(inputs,masks,variant,rays):
    import graph.plan_a_geometry.core as core
    with Candidate(9) as ctx:
        ctx.cache={}
        filt=RobotFilter(variant,rays) if variant not in ('baseline','cpu') else None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();result={}
        for camera in inputs[0]:
            if variant=='cpu':out=cpu_filter(camera,inputs[2])
            elif variant=='baseline' or (variant=='strict' and camera.depth_raw_mm.size<=128*128):out=core.classify_robot(camera,inputs[2])
            else:out=filt.classify(camera,inputs[2],masks[camera.camera])
            fn=filtered_masks if filt else full_masks
            result[camera.camera]=fn(masks[camera.camera],out)[0]
        torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
        return dict(ms=ms,clean=result,metrics=filt.last_metrics if filt else None,
            allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())


def main_filter(a):
    C.assert_source();torch.set_num_threads(a.threads)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    config=dict(args=vars(a),identity=identity(),boundary='two camera CPU depth/calibration/surfaces/raw masks -> CPU clean masks; baseline/CPU computes full diagnostics, ROI private computes only mask union; all transfers included; not Graph latency')
    C.write(output/'config.json',config)
    rows=[];specs={s['id']:s for s in C.read(FROZEN/'protocol.json')['scenes']}
    with ExperimentLogger(team='integration',model='task10i_robot_filter_only',job_type='evaluate',seed=42,root=output/'experiments',config=config) as log:
        for case in C.read(a.cases)['cases']:
            geo=geometry();inputs=load_inputs(case['scene'],case['size'],geo)
            masks=masks_for(specs[case['scene']],case['size'],0,'actual')
            if case.get('mask_mode')=='full':masks={k:np.stack((np.ones_like(m[0]),np.zeros_like(m[1]))) for k,m in masks.items()}
            expected=pair(inputs,masks,'baseline',RayCache())['clean'];rays={v:RayCache() for v in a.variants}
            for _ in range(a.warmups):
                for v in a.variants:exact(pair(inputs,masks,v,rays[v])['clean'],expected)
            for repeat in range(a.repeats):
                for v in a.variants if repeat%2==0 else a.variants[::-1]:
                    out=pair(inputs,masks,v,rays[v]);exact(out['clean'],expected)
                    rows.append(dict(case=case,variant=v,repeat=repeat,**{k:out[k] for k in ('ms','allocated','reserved','metrics')}))
            C.write(output/'results.json',dict(status='PARTIAL',rows=rows))
            print(json.dumps(dict(case=case,median_ms={v:float(np.median([r['ms'] for r in rows if r['case']==case and r['variant']==v])) for v in a.variants})),flush=True)
        C.write(output/'results.json',dict(status='PASS',rows=rows));log.link_output(output/'results.json');log.link_output(output/'config.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--variants',nargs='+',default=['cpu','baseline','hulls','strict']);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=15)
    main_filter(p.parse_args())

"""Bounded pre-adoption checks on frozen actual inputs, including one old OOM."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path('/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2')
OUT=ROOT.parent/'graph_optimization_v1'
sys.path.insert(0,str(ROOT/'code'))
import common as C
from graph_adapter import geometry,load_inputs,build
from graph_worker import masks_for
from gpu_graph import OptimizedGraph
from utils.experiment_logger import ExperimentLogger
import torch


def timeout(*_):raise TimeoutError('validation case exceeded 180 seconds')


def main():
    C.assert_source();torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3));signal.signal(signal.SIGALRM,timeout)
    specs={x['id']:x for x in C.read(ROOT/'protocol.json')['scenes']}
    completed=[]
    for path in sorted((ROOT/'graphs/actual').glob('*/*/*/receipt.json')):
        r=C.read(path).get('outcome',{});scene=path.parent.parent.parent.name;size=int(path.parent.parent.name)
        if r.get('status')=='COMPLETE' and r.get('mask_hash'):
            completed.append(dict(scene=scene,size=size,repeat=int(path.parent.name[1:]),cohort=specs[scene]['cohort']))
    selected=[]
    for size in [128,512,768,1024]:
        for cohort in ['static','tracking','seed98202']:
            matches=[x for x in completed if x['size']==size and x['cohort']==cohort]
            if matches:selected.append(matches[-1])
    assert len(selected)>=8
    rows=[];config=dict(stage='pre_adoption_validation',cases=selected,old_oom=['10i_s1_0044_t080',512,0],
        GPU_backend='cuda_fused_v5',GT_used=False,new_API_requests=0)
    with ExperimentLogger(team='integration',model='task10i_graph_optimization',job_type='evaluate',seed=42,
                          root=OUT/'experiments',config=config) as log:
        context=OptimizedGraph('cuda_fused_v5');context.load_reference(ROOT)
        with context:
            geo=geometry()
            for case in selected:
                s=specs[case['scene']];size=case['size'];repeat=case['repeat']
                inputs=load_inputs(s['id'],size,geo);masks=masks_for(s,size,repeat,'actual')
                signal.alarm(180);start=time.perf_counter();before=context.reference_checks
                try:r=build(geo,inputs,masks)
                finally:signal.alarm(0);geo.reset()
                assert context.reference_checks==before+1,case
                rows.append(dict(case,status='MATCH',graph_ms=r['graph_ms'],hashes=r['hashes'],wall_ms=(time.perf_counter()-start)*1000))
                print('MATCH',case,round(r['graph_ms'],1),flush=True)
        scene='10i_s1_0044_t080';size=512;s=specs[scene]
        old=C.read(ROOT/f'graphs/actual/{scene}/{size}/r0/receipt.json')['outcome']
        assert old['status']=='OOM'
        inputs=load_inputs(scene,size,geometry());masks=masks_for(s,size,0,'actual')
        results={}
        for backend in ['cpu_cells','cuda_fused_v5']:
            geo=geometry()
            with OptimizedGraph(backend):
                signal.alarm(180);start=time.perf_counter()
                try:r=build(geo,inputs,masks)
                finally:signal.alarm(0);geo.reset()
                results[backend]=dict(hashes=r['hashes'],graph_ms=r['graph_ms'],wall_ms=(time.perf_counter()-start)*1000)
            print('OLD_OOM',backend,round(r['graph_ms'],1),flush=True)
        assert results['cpu_cells']['hashes']==results['cuda_fused_v5']['hashes'],'old OOM parity failure'
        path=OUT/'pre_adoption_validation.json'
        C.write(path,dict(status='PASS',cases=rows,old_OOM_case=dict(scene=scene,size=size,old_status='OOM',
            parity='all five graph tensor hashes match CPU bounded-memory implementation',results=results),
            GPU_source_sha256=C.sha(Path(__file__).with_name('gpu_graph.py')),new_API_requests=0,GT_used=False))
        log.log(dict(reference_cases=len(rows),old_oom_cases=1));log.link_output(path)
        print('PRE_ADOPTION_PASS',len(rows),flush=True)


if __name__=='__main__':main()

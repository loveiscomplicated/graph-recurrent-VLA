"""Compare fast prototype against the previously validated GPU Graph on frozen masks."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2')
OUT=ROOT.parent/'graph_optimization_v1'
sys.path.insert(0,str(ROOT/'code'))
import common as C
from graph_adapter import geometry,load_inputs,build
from graph_worker import masks_for
from gpu_graph import OptimizedGraph
from fast_graph import FastGraph
from utils.experiment_logger import ExperimentLogger

def main():
    C.assert_source();torch.set_num_threads(4)
    cases=C.read(OUT/'pre_adoption_validation.json')['cases']
    specs={x['id']:x for x in C.read(ROOT/'protocol.json')['scenes']}
    rows=[]
    with ExperimentLogger(team='integration',model='task10i_graph_optimization',job_type='evaluate',seed=42,
                          root=OUT/'experiments',config=dict(stage='fast_v5_validation',cases=cases,GT_used=False)) as log:
        for case in cases:
            scene=case['scene'];size=case['size'];repeat=case['repeat']
            geo=geometry();inputs=load_inputs(scene,size,geo);masks=masks_for(specs[scene],size,repeat,'actual')
            with OptimizedGraph('cuda_fused_v5'):
                reference=build(geo,inputs,masks)
            geo.reset()
            with FastGraph('fast_v5_tree'):
                actual=build(geo,inputs,masks)
            geo.reset()
            for k in ('node_type','edge_index','edge_type','edge_attr'):
                np.testing.assert_array_equal(actual['graph'][k],reference['graph'][k],err_msg=k)
            a=actual['graph']['node_input'];b=reference['graph']['node_input']
            np.testing.assert_array_equal(a[:,:3],b[:,:3]);np.testing.assert_array_equal(a[:,8:],b[:,8:])
            np.testing.assert_allclose(a[:,3:8],b[:,3:8],atol=2e-5,rtol=0)
            assert actual['sources']==reference['sources'],'selected source points differ'
            rows.append(dict(scene=scene,size=size,repeat=repeat,status='PASS',
                feature_max_abs=float(np.max(np.abs(a[:,3:8]-b[:,3:8]),initial=0))))
            C.write(OUT/'fast_v5_diverse_validation.json',dict(status='PARTIAL',rows=rows))
            print('PASS',scene,size,flush=True)
        path=OUT/'fast_v5_diverse_validation.json'
        C.write(path,dict(status='PASS',rows=rows,reference='cuda_fused_v5',feature_atol=2e-5,GT_used=False))
        log.log(dict(cases=len(rows)));log.link_output(path)

if __name__=='__main__':main()

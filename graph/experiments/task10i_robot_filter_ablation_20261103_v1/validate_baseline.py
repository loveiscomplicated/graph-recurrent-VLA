"""Compare each frozen full-filter output with prior fast_v5_tree, without GT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(os.environ['TASK10I_ROOT']);REF=Path(os.environ['TASK10I_REFERENCE_CODE'])
sys.path.insert(0,str(REF));sys.path.insert(0,str(ROOT/'code'))
import common as C
from graph_adapter import geometry,load_inputs
from graph_worker import masks_for
from fast_graph import FastGraph
from utils.experiment_logger import ExperimentLogger


def validate_masks(folder):
    with np.load(folder/'raw_masks.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    masks={}
    for mode in ('full','confirmed_only','off'):
        with np.load(folder/mode/'masks.npz',allow_pickle=False) as z:masks[mode]={k:z[k].copy() for k in z.files}
    for cam in raw:
        a,b,c=(masks[m]['clean__'+cam] for m in ('full','confirmed_only','off'))
        assert np.all(~a|b) and np.all(~b|c),'unexpected clean-mask relationship'
        expected=raw[cam].copy();expected[:,raw[cam][0]&raw[cam][1]]=False
        np.testing.assert_array_equal(c,expected,err_msg='off must retain overlap exclusion')


def main(args):
    C.assert_source();torch.set_num_threads(4)
    run=Path(args.run);freeze=C.read(run/'prediction_freeze.json');cases=C.read(args.cases)
    specs={x['id']:x for x in C.read(ROOT/'protocol.json')['scenes']}
    out=run/'baseline_validation';out.mkdir(exist_ok=False);rows=[]
    with ExperimentLogger(team='integration',model='task10i_filter_ablation_baseline_check',job_type='evaluate',seed=42,
        root=out/'experiments',config=dict(prediction_freeze_sha256=C.sha(run/'prediction_freeze.json'),GT_used=False)) as log:
        for scenario in cases['scenarios']:
            for size in cases['sizes']:
                scene=scenario['scene'];folder=run/'cases'/scene/str(size);row=dict(scene=scene,size=size,stratum=scenario['stratum'])
                try:
                    receipt=folder/'receipt.json'
                    if not receipt.exists():raise FileNotFoundError('missing frozen case receipt')
                    for path,h in C.read(receipt)['files'].items():assert C.sha(folder/path)==h
                    validate_masks(folder)
                    with np.load(folder/'full'/'graph.npz',allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
                    pred=C.read(folder/'full'/'prediction.json')
                    geo=geometry();inputs=load_inputs(scene,size,geo);masks=masks_for(specs[scene],size,0,'actual')
                    pinned=C.read(folder/'inputs.json')
                    assert {k:C.array_sha(v) for k,v in masks.items()}==pinned['masks'],'mask changed since freeze'
                    for name,digest in pinned['sensor_files'].items():
                        assert C.sha(ROOT/'sensors'/scene/str(size)/name)==digest,'sensor changed since freeze'
                    with FastGraph('fast_v5_tree'):
                        graph,records,gate=geo.build(*inputs[:3],masks)
                    for name in ('node_type','edge_index','edge_type','edge_attr'):
                        np.testing.assert_array_equal(actual[name],getattr(graph,name).numpy(),err_msg=name)
                    a,b=actual['node_input'],graph.node_input.numpy()
                    np.testing.assert_array_equal(a[:,:3],b[:,:3]);np.testing.assert_array_equal(a[:,8:],b[:,8:])
                    np.testing.assert_allclose(a[:,3:8],b[:,3:8],atol=2e-5,rtol=0)
                    sources=json.loads(json.dumps([r['source_observation_ids'] for r in records],default=C.serial))
                    assert [r['sources'] for r in pred['records']]==sources,'sources differ'
                    assert pred['gate']==gate,'gate differs'
                    row.update(status='PASS',feature_max_abs=float(np.max(np.abs(a[:,3:8]-b[:,3:8]),initial=0)))
                except Exception as exc:row.update(status='FAILED',error_type=type(exc).__name__,message=str(exc))
                rows.append(row);C.write(out/'results.json',dict(status='PARTIAL',rows=rows))
                log.log(dict(attempted=len(rows),failed=sum(x['status']!='PASS' for x in rows)),step=len(rows))
                print('validate',scene,size,row['status'],flush=True)
        path=out/'results.json';C.write(path,dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'PARTIAL',rows=rows,GT_used=False))
        log.link_output(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cases',required=True);main(p.parse_args())

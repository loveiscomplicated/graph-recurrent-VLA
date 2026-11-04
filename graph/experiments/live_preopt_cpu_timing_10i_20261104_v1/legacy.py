"""Replay the frozen pre-CUDA CurrentGeometry; instrument without shortcuts."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.experiments.live_stage_timing_10i_20261104_v1.profile import Profiler


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(values):
    return dict(n=len(values),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),min_ms=min(values),max_ms=max(values))


def target(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


def timed_block(function,span,stage,label,begin,end):
    module=ast.parse(inspect.getsource(function));definition,=module.body
    starts=[i for i,node in enumerate(definition.body) if begin(node)]
    ends=[i for i,node in enumerate(definition.body) if end(node)]
    assert len(starts)==len(ends)==1 and starts[0]<=ends[0],function.__name__
    a,b=starts[0],ends[0]
    context=ast.Call(func=ast.Name(id='_legacy_span',ctx=ast.Load()),args=[ast.Constant(stage),ast.Constant(label)],keywords=[ast.keyword(arg='cpu',value=ast.Constant(True))])
    definition.body=definition.body[:a]+[ast.With(items=[ast.withitem(context_expr=context)],body=definition.body[a:b+1])]+definition.body[b+1:]
    ast.fix_missing_locations(module)
    namespace=dict(function.__globals__);namespace['_legacy_span']=span
    exec(compile(module,inspect.getsourcefile(function),'exec'),namespace)
    return namespace[definition.name]


def load(path):
    from graph.persistent_geometry.contracts import CameraObservation
    from graph.persistent_geometry.self_filter import LinkSurface
    with np.load(path,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
    data={k[6:]:v for k,v in p.items() if k.startswith('data__')}
    cameras=[CameraObservation(str(data['episode']),int(data['timestep']),name,float(data['timestamp']),
        *[p[name+'__'+f] for f in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl','tcp_pose')]) for name in ('base_camera','hand_camera')]
    surfaces=[LinkSurface(str(p[f'surface{i}__link']),p[f'surface{i}__vertices'],p[f'surface{i}__world']) for i in range(int(p['surface_count']))]
    return cameras,data,surfaces,{name:p[name+'__mask'] for name in ('base_camera','hand_camera')}


def fingerprint(result):
    graph,records,gate=result
    return dict(graph={k:hashlib.sha256(getattr(graph,k).numpy().tobytes()).hexdigest() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')},
                ids=[r['persistent_node_id'] for r in records],gate=gate)


def install(stack,prof):
    from graph.plan_a_bc import geometry
    from graph.plan_a_geometry import core
    from graph.persistent_geometry import graph,fusion
    from graph.integration_baseline import vision
    from visual.runtime import sampling
    replace=lambda owner,name,value:stack.enter_context(patch.object(owner,name,value))
    # All final edges, including attributes constructed in the return call.
    function=timed_block(graph.build_graph,prof.span,'17','final_edges',
        lambda n:isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='surface_edges',
        lambda n:target(n,'types'))
    function.__globals__['edge_attributes']=prof.wrap(graph.edge_attributes,'17','edge_attributes',cpu=True)
    replace(core,'build_graph',prof.wrap(function,'18','CPU_nodes_graph_return',cpu=True))
    pairs=[(geometry.CurrentGeometry,'build','18'),(core.PlanA,'prepare','8'),
        (core,'classify_robot','6'),(core,'filtered_masks','6'),(core,'wrist_gate','7'),
        (core.PlanA,'represent','15'),(core,'make_packet','13'),
        (vision,'extract_sampling_evidence','10'),(sampling,'build_geometry','9'),
        (sampling,'local_plane_optimized','10'),(sampling,'cropped_features','10'),
        (sampling,'pair_evidence','11'),(vision,'extract_candidates','11'),
        (vision.shape_first,'prepare_object','12'),(vision.shape_first,'select_object','12'),
        (vision.shape_first,'make_packet','13'),(vision,'complete_vision_packet','13'),
        (vision,'receive_vision_packet','13'),(core,'observations_from_packet','14'),
        (core,'consistent_views','14'),(geometry,'budget','16'),(graph,'edge_attributes','17'),
        (core.PlanA,'graph','18'),(geometry,'validate_graph','18')]
    for owner,name,stage in pairs:replace(owner,name,prof.wrap(getattr(owner,name),stage,name,cpu=True))
    # Resolve the actual imported call site and capture the already instrumented
    # globals. No duplicate implementation of packet/edge logic is introduced.
    function=timed_block(vision.packet_from_frame,prof.span,'extra_RawEDE','component_and_RawEDE',
        lambda n:isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='observed_components',
        lambda n:isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='update' and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='packet')
    replace(fusion,'packet_from_frame',function)


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,default=39);p.add_argument('--warmup',type=int,default=8)
    a=p.parse_args();assert os.environ.get('CONTAINER_ID')=='58752018'
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[3]
    for file,digest in read(source/'source_manifest.json').items():assert sha(source/file)==digest,file
    from graph.plan_a_bc.geometry import CurrentGeometry
    schema=read(a.schema)['schema'];inputs=read(a.inputs/'manifest.json')
    prof=Profiler(lambda:None,detailed=True);geometry=CurrentGeometry(a.assets,schema)
    rows=[];checks=[]
    config=dict(task='10i_original_CPU_numpy_stage_timing_20261104',legacy_commit=inputs['legacy_commit'],
        input_manifest_sha256=sha(a.inputs/'manifest.json'),source_manifest_sha256=sha(source/'source_manifest.json'),
        frames=a.frames,warmup=a.warmup,threads=1,shared_vision=True,
        boundary='original CurrentGeometry.build with ready host RGB/depth/calibration/raw SAM2 masks; includes original CPU filter/gate/full XYZ/PCA/ShapeFirst/RawEDE/fusion/dedup/FPS/graph',
        controlled='512 native images and identical current Terra/native SAM2 raw masks, original CPU sensors; original128 renderer/DINO/JPEG adapter not benchmarked')
    with ExperimentLogger(team='graph',model='preopt_CPU_stages',job_type='diagnostic',seed=42,config=config,root=a.output/'experiments') as log:
        for item in inputs['frames'][:a.frames]:
            t=item['timestep'];file=a.inputs/item['file'];assert sha(file)==item['sha256']
            args=load(file)
            # Input file loading and validation are outside each CPU timer.
            prof.frame=t;start_index=len(prof.events)
            with ExitStack() as stack:
                install(stack,prof)
                start=time.perf_counter_ns();result=geometry.build(*args);end=time.perf_counter_ns()
            graph,records,gate=result
            assert all(getattr(graph,k).device.type=='cpu' for k in ('node_input','node_type','edge_index','edge_attr','edge_type'))
            stages={}
            for event in prof.events[start_index:]:stages[event['stage']]=stages.get(event['stage'],0.)+event['exclusive_ms']
            row=dict(timestep=t,build_ms=(end-start)/1e6,stage_ms=stages,legacy_sensors_ms=item['legacy_sensors_ms'],
                mask_readback_ms=item['mask_readback_ms'],nodes=len(graph.node_type),edges=graph.edge_index.shape[1],
                points=[sum(r['object_id']==i for r in records) for i in (0,1)],gate=gate,
                surface_audit=graph.packet['surface_edge_audit'],fingerprint=fingerprint(result))
            assert abs(sum(stages.values())-row['build_ms'])<1.,(t,sum(stages.values()),row['build_ms'])
            if t in (0,9,a.frames-1):
                expected=geometry.build(*args)
                assert fingerprint(expected)==row['fingerprint'],('instrumentation changed CPU output',t)
                checks.append(dict(timestep=t,exact=True))
            rows.append(row);write(a.output/'partial.json',dict(rows=rows,checks=checks))
            print('CPU_FRAME',t,round(row['build_ms'],3),row['nodes'],row['edges'],flush=True)
        steady=rows[a.warmup+1:]
        stages=sorted({k for row in steady for k in row['stage_ms']})
        summary=dict(status='PASS',run_id=log.run_id,config=config,rows=rows,checks=checks,
            stage_summary={k:stats([row['stage_ms'].get(k,0.) for row in steady]) for k in stages},
            build_summary=stats([r['build_ms'] for r in steady]),
            legacy_sensors_summary=stats([r['legacy_sensors_ms'] for r in steady]),
            mask_readback_summary=stats([r['mask_readback_ms'] for r in steady]))
        write(a.output/'summary.json',summary);write(a.output/'spans.json',prof.events)
        log.record(status='PASS',original_CPU_outputs_exact=checks)
        log.link_output(a.output/'summary.json');log.link_output(a.output/'spans.json')
        print('CPU_COMPLETE',log.run_id,json.dumps(summary['build_summary']),flush=True)


if __name__=='__main__':main()

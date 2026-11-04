"""Audit current live call wiring. Robot filtering and wrist gate are fixtures.

Runs actual512 CUDA cameras, installed SAM2, projection/PCA/SF/records/FPS/Graph,
and the real rollout loop with a zero-command policy stub. Grounding uses fixed
boxes: no paid API call or semantic/accuracy claim. Production code is unedited.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


class Trace:
    def __init__(self):self.events=[];self.scope=[];self.graphs=[];self.packets=[]
    def event(self,name,**fields):self.events.append(dict(event=name,scope=list(self.scope),**fields))
    def function(self,module,name,label=None,after=None):
        original=getattr(module,name);label=label or name;trace=self
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            trace.event(label);trace.scope.append(label)
            try:
                result=original(*args,**kwargs)
                if after:after(result,args,kwargs)
                return result
            finally:trace.scope.pop()
        return patch.object(module,name,wrapped)
    def kernel(self,module,name):
        original=getattr(module,name);trace=self
        class Launch:
            def __getitem__(self,grid):
                actual=original[grid]
                def execute(*args,**kwargs):trace.event('kernel.'+name);return actual(*args,**kwargs)
                return execute
        return patch.object(module,name,Launch())


class FixedBoxes:
    """Current-frame bbox fixture; no network/GT or learned detection."""
    def __init__(self):self.calls=0
    def initialize(self,frames,prompts=None):
        self.calls+=1;rgb=frames[len(frames)-1];h,w=rgb.shape[:2]
        boxes={'peg':[.12*w,.18*h,.43*w,.57*h],'box':[.55*w,.2*h,.85*w,.65*h]}
        return {name:dict(box=box,score=None,frame=len(frames)-1,prompt='audit fixed box',scans=1) for name,box in boxes.items()}
    def status(self):return dict(backend='audit_fixed_boxes',calls=self.calls,network_calls=0)


class ExcludedPreparation:
    """Ready raw masks stand in for RF output; no RF code is run."""
    def submit(self,cameras,surfaces):
        from graph.persistent_geometry.tensor_fusion import depth_from_mm
        self.depths={c.camera:depth_from_mm(c.depth_raw_mm) for c in cameras}
        return SimpleNamespace(cameras=cameras,surfaces=surfaces)
    def filter(self,job,masks):return masks
    def discard(self,job):pass
    def close(self):pass


def gate_fixture(cameras,data,masks,clean,protocol,*,coordinate_frame,depths):
    from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
    selected={k:v.clone() for k,v in clean.items()};selected['hand_camera'][1]=False
    views=prepare_tensor_views(cameras,selected,coordinate_frame=coordinate_frame,depths=depths)
    return dict(conditions={'C3':selected},gate=dict(accepted=True,reasons=['audit fixture; gate excluded']),
        wrist_gate_backend='audit_excluded',wrist_gate_coordinate_frame=coordinate_frame),views


class ZeroPolicy(torch.nn.Module):
    def __init__(self,codec,trace):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros((),device='cuda'));self.codec=codec;self.trace=trace;self.variant='AUDIT'
    def step(self,packed,robot,norm,hidden):
        self.trace.event('model.step',node_device=str(packed.node_input.device),edge_device=str(packed.edge_index.device))
        assert packed.node_input.is_cuda and packed.edge_index.is_cuda and packed.edge_attr.is_cuda
        values=torch.zeros((1,7),device='cuda')
        if self.codec.method!='A':values[:,:6]=-torch.tensor(self.codec.mean,device='cuda')/torch.tensor(self.codec.std,device='cuda')
        return values,torch.zeros((1,1),device='cuda')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('another CUDA process active')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[3];manifest=read(source/'source_manifest.json')
    for f,h in manifest.items():assert hashlib.sha256((source/f).read_bytes()).hexdigest()==h,f
    from graph.plan_a_bc import rollout
    from graph.persistent_geometry import tensor_fusion,tensor_records,tensor_graph,tensor_knn,tensor_gate
    from graph.plan_a_geometry import core
    from visual.runtime import graph_perception,torch_projection,cuda_sampling,tensor_packet,tracking
    from visual.runtime.test import pca_triton
    from graph.integration_baseline import tensor_online_vision
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter as LegacyAdapter
    from graph.integration_baseline.bc08_codec import ActionCodec
    fields=tensor_graph.FIELDS
    sam=read(a.runtime/'runtime_paths.json')['sam'];tracker=tracking.SAM2Tracker(device='cuda',checkpoint=sam)
    base=read(a.runtime/'config.json');schema=read(a.runtime/'normalizer.json')['schema'];codec=ActionCodec(read(a.runtime/'action_codec.json'))
    config=dict(task='10i_live_nonrobot_wiring_audit_20261104',instance='58752018',frames=['ee','world'],steps_each=2,
        real='native512 renderer, installed SAM2, original rollout.run_episode and all nonrobot Graph stages',
        fixtures='fixed grounding boxes, passthrough ready-clean masks, gate output, zero-command policy',
        robot_filter_and_gate_excluded=True,API_calls=0,
        sam_sha256=hashlib.sha256(Path(sam).read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='diagnostic',seed=42,root=a.output/'experiments',config=config)
    results={}
    try:
        for frame in ('ee','world'):
            trace=Trace();root=a.output/frame;root.mkdir();(root/'inputs').symlink_to(a.runtime/'inputs',target_is_directory=True)
            cfg=deepcopy(base);cfg['evaluation'].update(horizon=2,smoke_horizon=2);cfg.setdefault('geometry',{})['coordinate_frame']=frame;write(root/'config.json',cfg)
            original_geometry=rollout.CurrentGeometry
            def geometry(*args,**kwargs):
                g=original_geometry(*args,**kwargs);assert g.perception_backend=='triton_fused' and g.tensor_geometry is None
                g.robot_preparation=ExcludedPreparation();return g
            def graph_after(result,args,kwargs):
                g,records,_=result
                assert all(getattr(g,k).is_cuda for k in fields) and records.xyz.is_cuda
                assert g.packet['coordinate_frame']==('EE_metres' if frame=='ee' else 'World_metres')
                assert g.packet['graph_backend']==g.packet['records_dedup_backend']==g.packet['final_fps_backend']=='torch_cuda'
                summary=dict(nodes=len(g.node_type),edges=g.edge_index.shape[1],points=len(records),device=str(records.device),
                    frame=g.packet['coordinate_frame'],source_device=str(records.source_pixels.device))
                trace.graphs.append(summary);trace.last_graph=g
            def packet_after(packet,args,kwargs):
                assert packet['points'].is_cuda and packet['geometric_features'].is_cuda
                trace.packets.append(dict(frame=packet['coordinate_frame'],object_slices=packet['object_slices'],points=len(packet['points'])))
            original_cpu=torch.Tensor.cpu
            def cpu(value,*args,**kwargs):
                if value.is_cuda:trace.event('readback.cpu',shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
                return original_cpu(value,*args,**kwargs)
            def forbid(*args,**kwargs):raise AssertionError('legacy CPU point path reached')
            with ExitStack() as scope:
                scope.enter_context(patch.object(rollout,'CurrentGeometry',side_effect=geometry))
                scope.enter_context(patch.object(tensor_gate,'prepare_tensor_gate',side_effect=gate_fixture))
                scope.enter_context(patch.object(torch.Tensor,'cpu',cpu))
                functions=[(rollout,'sensors'),(original_geometry,'build','geometry.build',graph_after),
                    (tensor_online_vision.TensorOnlineVisionAdapter,'process','vision.process'),(tracking.SAM2Tracker,'start_frames','SAM2.start_frames'),
                    (tracking.SAM2Tracker,'track','SAM2.track'),(graph_perception,'make_packet_cuda'),(torch_projection,'project'),
                    (graph_perception,'features_cuda'),(pca_triton,'fused_normals_compact'),(pca_triton,'fused_features_compact'),(cuda_sampling,'build_sampling_info_cuda'),(cuda_sampling,'shape_first_cuda'),
                    (cuda_sampling,'surface_fps'),(cuda_sampling._FPSPlan,'run','FPSPlan.run'),
                    (tensor_packet,'selected_packet',None,packet_after),(tensor_fusion,'observations_from_tensor_packet'),
                    (tensor_fusion,'consistent_views_tensor'),(tensor_records,'current_records_tensor'),(tensor_records,'deduplicate_tensor'),
                    (tensor_records,'budget_tensor'),(tensor_graph,'build_graph_tensor'),(tensor_graph,'neighbor_membership'),
                    (tensor_graph,'free_space_tensor'),(rollout,'pack_graphs'),(tensor_packet,'numpy_packet'),(tensor_graph,'numpy_graph')]
                for item in functions:scope.enter_context(trace.function(*item))
                for name in ('mask_bounds','voxel_choices','compress','canonical','seed_tile_bounds','fps_seed_bounded','mark_membership','gather_membership'):
                    scope.enter_context(trace.kernel(cuda_sampling.kernels,name))
                for module,names in [(core,['current_records','deduplicate','budget','build_graph','frame_points']),
                    (graph_perception,['to_cpu_evidence']),(tensor_fusion,['numpy_observations']),
                    (tensor_records,['numpy_records']),(cuda_sampling,['numpy_selection'])]:
                    for name in names:scope.enter_context(patch.object(module,name,side_effect=forbid))
                row=rollout.run_episode(root,root/'episode',ZeroPolicy(codec,trace),SimpleNamespace(schema=schema),codec,
                    FixedBoxes(),tracker,dict(seed=42,group='audit'),500,smoke=True)
            assert len(trace.graphs)==2 and all(g['points']>0 for g in trace.graphs)
            counts=Counter(e['event'] for e in trace.events)
            expected=('vision.process','SAM2.start_frames','SAM2.track','project','features_cuda','build_sampling_info_cuda','shape_first_cuda',
                'selected_packet','observations_from_tensor_packet','consistent_views_tensor','current_records_tensor','deduplicate_tensor',
                'budget_tensor','build_graph_tensor','neighbor_membership','free_space_tensor','pack_graphs','model.step')
            write(root/'events.json',trace.events)
            assert all(counts[name]>0 for name in expected),counts
            probes={}
            legacy=LegacyAdapter(FixedBoxes(),tracker);legacy.reset('legacy-probe')
            try:
                try:legacy.process(torch.zeros((512,512,3),device='cuda',dtype=torch.uint8),'legacy-probe',0)
                except TypeError as error:probes['legacy_vision_cuda_input']=dict(type=type(error).__name__,message=str(error))
                else:raise AssertionError('legacy CUDA image conversion unexpectedly succeeded')
            finally:legacy.close()
            try:trace.last_graph.node_input.numpy()
            except TypeError as error:probes['direct_graph_numpy']=dict(type=type(error).__name__,message=str(error))
            else:raise AssertionError('direct CUDA Graph.numpy unexpectedly succeeded')
            write(root/'events.json',trace.events)
            results[frame]=dict(status='PASS',counts=dict(counts),graphs=trace.graphs,packets=trace.packets,
                secondary_contract_probes=probes,real_episode_steps=row['episode_length'],physics_calls=row['physics_calls'])
        # Native fixed boxes establish live wiring; also exercise nonempty
        # surface/free-space work against the saved pre-port graph reference.
        from graph.experiments.live_cuda_ingress_58752018_20261103_v1.baseline_single import load
        from graph.persistent_geometry.tensor_camera import tensor_camera
        saved=Path('/workspace/task10i_graph_depth_lookup_20261103_v1')
        cameras,data,masks=load(saved/'inputs');cameras=[tensor_camera(c) for c in cameras]
        raw={k:torch.as_tensor(v,device='cuda') for k,v in masks.items()}
        reference=np.load(saved/'measured/graph_reference.npz',allow_pickle=False)
        saved_results={}
        for frame in ('ee','world'):
            g=rollout.CurrentGeometry(a.runtime/'inputs',schema,coordinate_frame=frame)
            g.robot_preparation=ExcludedPreparation();surfaces=g.fk.surfaces(data['robot76'])
            original_free=tensor_graph.free_space_tensor;sizes=[]
            def free(samples,*args,**kwargs):sizes.append(samples.numel()//3);return original_free(samples,*args,**kwargs)
            try:
                with patch.object(tensor_gate,'prepare_tensor_gate',side_effect=gate_fixture),patch.object(tensor_graph,'free_space_tensor',side_effect=free):
                    graph,records,_=g.build(cameras,data,surfaces,lambda:raw)
                assert max(sizes)>0
                for field in fields:np.testing.assert_array_equal(getattr(graph,field).cpu().numpy(),reference[frame+'__'+field])
                saved_results[frame]=dict(nodes=len(graph.node_type),edges=graph.edge_index.shape[1],
                    nonempty_free_space_samples=sizes,all_five_tensors_bit_exact_vs_saved_CPU=True)
            finally:g.close()
        config['saved_inputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (saved/'inputs').glob('*.npz')}
        summary=dict(run_id=logger.run_id,config=config,results=results,saved_reference=saved_results,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        write(a.output/'summary.json',summary);logger.record(summary=summary)
        for path in list(a.output.rglob('*.json')):
            if 'experiments' not in path.parts:logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()

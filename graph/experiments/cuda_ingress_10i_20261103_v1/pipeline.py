"""CPU observations -> one upload -> resident gate/SF -> small exact graph.

The single explicit legacy boundary exports only selected, view-checked rows,
never full masks, dense XYZ or per-pixel features. Small CPU cKDTree operations
retain their original exact tie behavior; network graph arrays return on CUDA.
"""
from dataclasses import fields
from functools import cached_property
import numpy as np
import torch
from graph.persistent_geometry.contracts import CameraObservation,SurfaceObservation
from graph.persistent_geometry.map import current_records
from graph.plan_a_geometry import core
from visual.runtime.graph_perception import make_packet_cuda
from visual.runtime.tensor_packet import numpy_packet
from visual.runtime.object_groups import object_rows
from graph.persistent_geometry.coordinates import camera_origin
from graph.persistent_geometry.tensor_fusion import prepare_tensor_views, consistency_keep
from .ingress import TensorIngress
from .gate import wrist_gate
from .consumers import SmallGraphConsumers
from .selected_records import records_after_budget


class _FinalCamera(CameraObservation):
    # Final edge depth checks may ask repeatedly; one frame-local conversion.
    @cached_property
    def depth(self):return super().depth
    @cached_property
    def world_from_camera(self):return super().world_from_camera


class TensorGeometry:
    def __init__(self,system,robot_filter=None):
        if robot_filter is None:
            from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
            robot_filter=CudaRobotFilter()
        self.system=system;self.ingress=TensorIngress(robot_filter)
        self.last=None

    def stage(self,cameras,data,surfaces):return self.ingress.stage(cameras,data,surfaces)

    @torch.inference_mode()
    def build(self,frame,masks):
        if set(masks)!=set(frame.tensors):raise ValueError('camera mask identity')
        clean=self.ingress.filter(frame,masks)
        accepted,gate=wrist_gate(frame,masks,clean,self.system.protocol['gate'])
        cm={k:v.clone() for k,v in clean.items()}
        cm['hand_camera'][1]=False;cm['hand_camera'][0]=accepted
        packets={};exports={}
        coordinate_frame=self.system.coordinate_frame
        tensor_views=prepare_tensor_views(frame.cameras,cm,coordinate_frame=coordinate_frame or 'world',
            device=next(iter(cm.values())).device,depths={name:t.depth64 for name,t in frame.tensors.items()})
        for name in ('base_camera','hand_camera'):
            t=frame.tensors[name];c=t.camera
            target=coordinate_frame or 'ee'
            p=make_packet_cuda(c,cm[name],depth_cuda=t.depth32,intrinsic_cuda=t.intrinsic,
                transform_cuda=t.world if target=='world' else t.ee,coordinate_frame=target)
            packets[name]=p;n=len(p['points'])
            world=p['points'].to(torch.float64)
            if coordinate_frame is None:world=world@t.world_ee[:3,:3].T+t.world_ee[:3,3]
            valid=p['feature_field_valid'];f5=p['geometric_features'].clone()
            normals=f5[:,:3]@t.world_ee[:3,:3].T if coordinate_frame is None else f5[:,:3]
            normals=torch.where(valid[:,:1],normals,0.)
            f5[:,:3]=torch.where(valid[:,:1],normals,torch.nan)
            keep=torch.ones(n,device=world.device,dtype=torch.bool)
            for oid,rows in enumerate(object_rows(p)):
                keep[rows]=consistency_keep(world[rows],oid,name,tensor_views)
            exports[name]=dict(xyz=world[keep],normals=normals[keep],f5=f5[keep],valid=valid[keep],
                oid=p['object_id'][keep],pixels=p['provenance']['source_index'][keep],
                structural=~p['provenance']['surface_selected'][keep])
        # One packed readback, after all image-sized stages. Its size scales
        # with selected points, not H*W, and is recorded for measurement.
        byte_count=sum(a.numel()*a.element_size() for e in exports.values() for a in e.values())
        host=numpy_packet(exports)
        records=records_after_budget(host,frame,distance=self.system.protocol['merge_distance_m'],coordinate_frame=coordinate_frame)
        if records is None:
            # Exact pre-existing fallback for unusual native-domain/ambiguity
            # overflow; never change the policy to force the fast path.
            records=[]
            for name in ('base_camera','hand_camera'):
                e=host[name];c=frame.tensors[name].camera;observations=[]
                for oid in (0,1):
                    rows=e['oid']==oid
                    observations.append(SurfaceObservation(c.episode,c.timestep,name,oid,e['xyz'][rows],e['normals'][rows],
                        e['valid'][rows,0],e['f5'][rows],e['valid'][rows],e['pixels'][rows],
                        camera_origin(c,coordinate_frame or 'world'),coordinate_frame or 'world',
                        tuple(c.tcp_pose) if coordinate_frame=='ee' else None).validate())
                rr=current_records(observations,fuse_views=False,coordinate_frame=coordinate_frame)
                structural=set(e['pixels'][e['structural']].tolist())
                for r in rr:r['structural']=r['source_observation_ids'][0][3] in structural
                records.extend(rr)
            with SmallGraphConsumers():
                from graph.experiments.graph_latency_10i_20261103_v2 import native
                records,_=native.deduplicate(records,self.system.protocol['merge_distance_m'],fallback=core.deduplicate,owned=True)
                records=core.budget(records,128)
        cameras=[_FinalCamera(**{f.name:getattr(c,f.name) for f in fields(CameraObservation)}) for c in frame.cameras]
        with SmallGraphConsumers():
            graph=self.system.graph(records,frame.data,cameras)
        for name in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            setattr(graph,name,getattr(graph,name).to('cuda'))
        graph.packet['perception_backend']='cuda_resident'
        self.last=dict(frame=frame,clean=clean,conditions=cm,packets=packets,
            selected_readback_bytes=byte_count,gate=gate)
        return graph,records,gate

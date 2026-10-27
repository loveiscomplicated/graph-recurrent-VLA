"""Practical sensor-only G0--G4 path, shared by base and branch diagnostics."""
import numpy as np

from graph.integration_baseline.graph import build_graph as original_graph
from graph.integration_baseline.robot import geometry_in_ee
from visual.runtime.vision_packet import pose_matrix
from .contracts import synchronized,transform
from .self_filter import classify_robot
from .fusion import make_packet,filtered_masks,observations_from_packet,consistent_views
from .map import PersistentMap,current_records


def original_g0(camera,data,masks):
    packet,_=make_packet(camera,masks)
    robot_ee=geometry_in_ee(data['robot_world_geometry'],data['tcp_pose'])
    return packet,original_graph(packet,robot_ee)


class GeometrySystem:
    """Current observations, persistent maps and graph inputs stay separate.

    This class cannot accept an evaluation sidecar, actor poses or actor meshes.
    A diagnostic history fork may supply two already causal map states.
    """
    def __init__(self,main,wrist,main_map=None,fused_map=None):
        if any(memory is not None and type(memory) is not PersistentMap for memory in (main_map,fused_map)):
            raise ValueError('practical system only accepts sensor-derived PersistentMap states')
        self.main,self.wrist=main,wrist
        self.main_map=main_map if main_map is not None else PersistentMap()
        self.fused_map=fused_map if fused_map is not None else PersistentMap()

    def process(self,cameras,data,masks,robot_link_surfaces):
        synchronized(cameras)
        same={c.camera:c for c in cameras}
        if set(same)!={self.main,self.wrist} or set(masks)!=set(same):
            raise ValueError('configured camera identity mismatch')
        allowed={'robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world','timestep','timestamp','episode'}
        allowed.update(name+'/'+field for name in same for field in
                       ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
        if set(data)-allowed:
            raise ValueError('non-sensor fields are forbidden in practical input')
        episode,timestep=cameras[0].episode,cameras[0].timestep
        if str(data['episode'])!=episode or int(data['timestep'])!=timestep:
            raise ValueError('robot/sensor identity join')
        baseline_packet,baseline_graph=original_g0(same[self.main],data,masks[self.main])
        filters={};packets={};clean={};observed={}
        for camera in cameras:
            filters[camera.camera]=classify_robot(camera,robot_link_surfaces)
            clean[camera.camera],_=filtered_masks(masks[camera.camera],filters[camera.camera])
            packets[camera.camera],_=make_packet(camera,clean[camera.camera])
            observed[camera.camera]=observations_from_packet(packets[camera.camera],camera)
        dual,audit=consistent_views(observed[self.main]+observed[self.wrist],cameras,clean)
        records=dict(G0=current_records(observations_from_packet(baseline_packet,same[self.main]),fuse_views=False),
                     G1=current_records(observed[self.main],fuse_views=False),G3=current_records(dual))
        self.main_map.update(episode,timestep,observed[self.main],cameras=[same[self.main]])
        self.fused_map.update(episode,timestep,dual,cameras=cameras)
        records.update(G2=self.main_map.records(),G4=self.fused_map.records())
        return dict(records=records,baseline_packet=baseline_packet,baseline_graph=baseline_graph,
                    filter_outputs=filters,filtered_packets=packets,main_observations=observed[self.main],
                    dual_observations=dual,fusion_audit=audit)

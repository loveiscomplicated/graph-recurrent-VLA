"""Opt-in Legacy adapter using unchanged filter, map, registration and fusion."""
import numpy as np
from graph.persistent_geometry.contracts import synchronized
from graph.persistent_geometry.self_filter import classify_robot
from graph.persistent_geometry.fusion import filtered_masks,consistent_views
from graph.persistent_geometry.map import PersistentMap,current_records
from .adapter import frame_points

class LegacySystem:
    def __init__(self,main,wrist,*,seed=42,f5=True):
        self.main,self.wrist,self.seed,self.f5=main,wrist,seed,f5
        self.main_map=PersistentMap();self.fused_map=PersistentMap()
    def process(self,cameras,data,masks,robot_link_surfaces):
        synchronized(cameras)
        allowed={'robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world','timestep','timestamp','episode'}
        allowed.update(c.camera+'/'+k for c in cameras for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
        if set(data)-allowed:raise ValueError('GT/non-sensor fields forbidden')
        if (str(data['episode']),int(data['timestep']))!=(cameras[0].episode,cameras[0].timestep):raise ValueError('sensor/robot join')
        by={c.camera:c for c in cameras}
        if self.main not in by:raise ValueError('BLOCKED_SENSOR_INPUT: main camera missing')
        if self.wrist not in by:raise ValueError('BLOCKED_SENSOR_INPUT: wrist camera missing')
        base,base_points,base_components=frame_points(by[self.main],masks[self.main],seed=self.seed,f5=self.f5)
        observed={};filters={};clean={};all_points={};components={}
        for c in cameras:
            filters[c.camera]=classify_robot(c,robot_link_surfaces)
            clean[c.camera],_=filtered_masks(masks[c.camera],filters[c.camera])
            observed[c.camera],all_points[c.camera],components[c.camera]=frame_points(c,clean[c.camera],seed=self.seed,f5=self.f5)
        dual,audit=consistent_views(observed[self.main]+observed[self.wrist],cameras,clean)
        e,t=cameras[0].episode,cameras[0].timestep
        self.main_map.update(e,t,observed[self.main],cameras=[by[self.main]])
        self.fused_map.update(e,t,dual,cameras=cameras)
        records=dict(L0=current_records(base,fuse_views=False),L1=current_records(observed[self.main],fuse_views=False),
            L2=self.main_map.records(),L3=current_records(dual),L4=self.fused_map.records())
        return dict(records=records,base_observations=base,base_points=base_points,base_components=base_components,
            main_observations=observed[self.main],dual_observations=dual,filter_outputs=filters,
            filtered_masks=clean,filtered_points=all_points,fusion_audit=audit)

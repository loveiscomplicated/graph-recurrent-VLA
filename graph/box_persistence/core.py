"""Sensor-only Box memory. Reuses09-C fusion, lifecycle and registration.

B1/B2 execute identical code except application of the estimated pose update.
No evaluator, object mesh, actor pose or task goal is accepted by this module.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree
from graph.plan_a_geometry.core import components
from graph.persistent_geometry.contracts import SurfaceObservation
from graph.persistent_geometry.map import PersistentMap, ObjectMap, INVALID_REMOVED


def observations(records, episode, timestep, main, origin):
    if any(r['object_id'] != 1 for r in records):
        raise ValueError('Box only; Peg memory forbidden')
    for r in records:
        if any(s[0] != episode or s[1] != timestep or s[2] != main for s in r['source_observation_ids']):
            raise ValueError('foreign/future/Wrist source')
    n=len(records)
    return SurfaceObservation(episode,timestep,main,1,
        np.array([r['world_xyz'] for r in records]).reshape(n,3),
        np.array([r['world_normal'] for r in records]).reshape(n,3),
        np.array([r['normal_valid'] for r in records],bool),
        np.array([r['last_f5'] for r in records]).reshape(n,5),
        np.array([r['last_feature_valid'] for r in records],bool).reshape(n,3),
        np.array([r['source_observation_ids'][0][3] for r in records],int),origin).validate()


class BoxMemory(PersistentMap):
    def __init__(self, mode, cfg, main='base_camera'):
        if mode not in ('B1','B2'):raise ValueError('estimated B1/B2 only')
        super().__init__(merge_distance=cfg['merge_distance_m'])
        self.mode,self.cfg,self.main=mode,cfg,main
        self.trace=[]

    def reset(self, episode):
        super().reset(episode)
        self.objects={1:ObjectMap(1)}
        self.trace=[]

    def update_records(self, episode, timestep, records, camera):
        if camera.camera != self.main or camera.episode!=episode or camera.timestep!=timestep:
            raise ValueError('main same-time camera required')
        obs=observations(records,episode,timestep,self.main,camera.world_from_camera[:3,3])
        if episode != self.episode:
            if timestep != 0:raise ValueError('episode starts at zero')
            self.reset(episode)
        self.clock.advance(episode,timestep,[obs])
        obj=self.objects[1]
        for node in obj.active():node.currently_observed=False;node.estimated_only=True
        trace=dict(episode=episode,timestep=timestep,mode=self.mode,input_count=len(records),
            accepted_count=0,rejected_count=len(records),state='UNOBSERVABLE_UNKNOWN',
            static_violation=False,GT_used=False,hidden_motion_observable=False)
        labels=components(obs.xyz,self.cfg['component_radius_m'])
        if len(obs.xyz):
            counts=np.bincount(labels)
            local=cKDTree(obs.xyz).query_ball_point(obs.xyz,self.cfg['local_radius_m'],return_length=True)
            supported=(counts[labels]>=self.cfg['minimum_component_points'])&(local>=self.cfg['minimum_local_neighbors'])
            # Initialization has no map identity evidence: largest supported component only.
            if not obj.active():supported &= labels==counts.argmax()
            obs=obs.subset(supported)
        old_pose=obj.world_from_object.copy()
        registration=None
        if not len(obs.xyz):
            obj.tracking_state='UNOBSERVABLE_UNKNOWN'
            obj.pose_uncertainty+=.002;obj.pose_confidence*=.8
        elif not obj.active():
            # Literal world frame, never actor GT or observed centroid as actor origin.
            obj.world_from_object=np.eye(4)
            obj.pose_confidence=.8;obj.pose_uncertainty=.003
            obj.tracking_state='INITIALIZED_FROM_OBSERVATION'
        else:
            registration=self._register(obj,obs.xyz)
            obj.registration={k:v for k,v in vars(registration).items() if k!='pose'}
            trace['registration']=obj.registration
            # Displacement is measured at the map centroid, not world-origin translation.
            center=obj.world_points().mean(0)
            from graph.persistent_geometry.contracts import transform
            relative=registration.pose@np.linalg.inv(old_pose)
            movement=float(np.linalg.norm(transform(center,relative)-center))
            violation=registration.accepted and (movement>self.cfg['static_motion_threshold_m'] or registration.rotation_rad>self.cfg['static_rotation_threshold_rad'])
            trace.update(static_violation=bool(violation),estimated_surface_motion_m=movement)
            if registration.accepted:
                if self.mode=='B2':obj.world_from_object=registration.pose
                obj.pose_confidence=registration.confidence
                obj.pose_uncertainty=max(.001,registration.residual_m)
                if registration.rank<6:obj.pose_uncertainty=max(.004,obj.pose_uncertainty)
                obj.tracking_state='POSSIBLE_OBJECT_MOTION' if violation else registration.state
            else:
                obj.pose_confidence*=.5;obj.pose_uncertainty+=.005
                obj.tracking_state='TRACKING_FAILURE_OR_MOTION_UNKNOWN'
                obs=obs.subset(np.zeros(len(obs.xyz),bool))
        if len(obs.xyz):
            if obj.active():obs=obs.subset(self._connected_support(obs.xyz,obj.world_points()))
            added,updated=self._integrate(obj,obs,timestep)
            # Preserve complete PlanA same-time source support in addition to primary pixels.
            support={tuple(r['source_observation_ids'][0]):r['source_observation_ids'] for r in records}
            for node in obj.active():
                if node.last_seen_timestep==timestep:
                    extra=[s for src in list(node.source_observation_ids) if src[1]==timestep for s in support.get(tuple(src),[])]
                    node.source_observation_ids=list(dict.fromkeys(map(tuple,node.source_observation_ids+extra)))
            trace.update(accepted_count=len(obs.xyz),rejected_count=len(records)-len(obs.xyz),added=added,updated=updated)
        self._contradictions(obj,[camera],timestep)
        self._age(obj,timestep)
        if len(obj.active())>self.cfg['maximum_native_nodes']:
            for node in sorted(obj.active(),key=lambda n:(n.confidence,n.last_seen_timestep,n.persistent_node_id))[:len(obj.active())-self.cfg['maximum_native_nodes']]:
                node.removed=True
                self._event(timestep,1,'CAPACITY_REMOVED',node_id=node.persistent_node_id)
        trace.update(state=obj.tracking_state,pose=obj.world_from_object.copy(),
            pose_update_applied=self.mode=='B2' and registration is not None and registration.accepted,
            active_nodes=len(obj.active()),uncertainty_m=obj.pose_uncertainty)
        self.trace.append(trace)
        return self

    def records(self):
        rows=super().records()
        for r in rows:
            r.update(age=self.clock.timestep-r['last_seen_timestep'],source_camera=r['last_observed_camera'],
                source_timestep=r['last_seen_timestep'],source_pixel_id=r['source_observation_ids'][-1][3],
                structural=False)
        return rows

    def active_records(self):
        return [r for r in self.records() if r['status']!=INVALID_REMOVED]


def paired_records(current, memory):
    """Peg records are copied byte-for-byte; only Box records are replaced."""
    return deepcopy([r for r in current if r['object_id']==0])+deepcopy(memory)

"""Causal object-local surfels, explicit pose uncertainty and lifecycle events."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .contracts import CausalClock, transform
from .registration import register

CURRENTLY_OBSERVED = 0
PERSISTENT_ESTIMATED = 1
LOW_CONFIDENCE = 2
INVALID_REMOVED = 3


@dataclass
class Surfel:
    persistent_node_id: str
    object_id: int
    object_local_xyz: np.ndarray
    object_local_normal: np.ndarray
    normal_valid: bool
    confidence: float
    first_seen_timestep: int
    last_seen_timestep: int
    last_observed_camera: str
    observation_count: int
    currently_observed: bool
    estimated_only: bool
    position_uncertainty: float
    source_observation_ids: list
    last_f5: np.ndarray
    last_feature_valid: np.ndarray
    contradiction_count: int = 0
    removed: bool = False


@dataclass
class ObjectMap:
    object_id: int
    world_from_object: np.ndarray = field(default_factory=lambda: np.eye(4))
    nodes: list = field(default_factory=list)
    pose_uncertainty: float = .002
    pose_confidence: float = 0.
    tracking_state: str = 'UNINITIALIZED'
    last_tracking_timestep: int = -1
    registration: dict = field(default_factory=dict)

    def active(self):
        return [n for n in self.nodes if not n.removed]

    def world_points(self):
        return transform(np.asarray([n.object_local_xyz for n in self.active()]).reshape(-1, 3), self.world_from_object)


class PersistentMap:
    def __init__(self, *, merge_distance=.003, normal_cosine=.7):
        self.merge_distance = merge_distance
        self.normal_cosine = normal_cosine
        self.clock = CausalClock()
        self.episode = None
        self.objects = {}
        self.next_id = 0
        self.events = []

    def reset(self, episode):
        self.episode = episode
        self.objects = {oid: ObjectMap(oid) for oid in (0, 1)}
        self.next_id = 0
        self.events = []
        self.clock = CausalClock()

    @staticmethod
    def _register(obj, current):
        nodes=obj.active()
        return register(np.array([n.object_local_xyz for n in nodes]),current,obj.world_from_object,
                        local_normals=np.array([n.object_local_normal for n in nodes]),
                        normal_valid=np.array([n.normal_valid for n in nodes]))

    def _event(self, timestep, oid, kind, **detail):
        self.events.append(dict(episode=self.episode, timestep=timestep, object_id=oid,
                                event=kind, **detail))

    def update(self, episode, timestep, observations, *, cameras=()):
        if episode != self.episode:
            if timestep != 0:
                raise ValueError('explicit episode boundary at timestep zero required')
            self.reset(episode)
        self.clock.advance(episode, timestep, observations)
        for camera in cameras:
            if camera.episode != episode or camera.timestep != timestep:
                raise ValueError('future/foreign camera evidence')
        for oid, obj in self.objects.items():
            for node in obj.active():
                node.currently_observed = False
                node.estimated_only = True
            views = [o for o in observations if o.object_id == oid and len(o.xyz)]
            current = np.concatenate([o.xyz for o in views]) if views else np.empty((0, 3))
            if not len(current):
                if obj.nodes:
                    obj.tracking_state = 'FULL_OCCLUSION'
                    obj.pose_uncertainty += .002
                    obj.pose_confidence *= .8
                self._event(timestep, oid, obj.tracking_state)
                self._age(obj, timestep)
                continue
            own = obj.world_points()
            other = self.objects[1-oid].world_points()
            if len(own) and len(other):
                d_own = cKDTree(own).query(current)[0]
                d_other = cKDTree(other).query(current)[0]
                swap_fraction = float(((d_other < .004) & (d_own > .02)).mean())
                if swap_fraction > .6:
                    obj.tracking_state = 'IDENTITY_SWITCH_REJECTED'
                    obj.pose_uncertainty += .005
                    obj.pose_confidence = 0.
                    self._event(timestep, oid, obj.tracking_state, fraction=swap_fraction)
                    self._age(obj, timestep)
                    continue
            if not len(own):
                # An observation-defined frame. Never initialize from actor GT.
                obj.world_from_object = np.eye(4)
                obj.world_from_object[:3, 3] = np.median(current, axis=0)
                obj.tracking_state = 'INITIALIZED_FROM_OBSERVATION'
                obj.pose_confidence = .5
                obj.pose_uncertainty = .003
            else:
                registration = self._register(obj, current)
                obj.registration = {k: v for k, v in vars(registration).items() if k != 'pose'}
                obj.tracking_state = registration.state
                if not registration.accepted:
                    obj.pose_uncertainty += .005
                    obj.pose_confidence *= .5
                    self._event(timestep, oid, registration.state, **obj.registration)
                    self._age(obj, timestep)
                    continue
                obj.world_from_object = registration.pose
                obj.pose_confidence = registration.confidence
                if registration.state in ('PARTIALLY_OBSERVABLE','TRACKED_PLANAR_UNCERTAINTY'):
                    obj.pose_uncertainty=max(obj.pose_uncertainty+.003,registration.residual_m)
                else:
                    obj.pose_uncertainty=max(.001,registration.residual_m)
            obj.last_tracking_timestep = timestep
            # Novel surfaces must connect through current observed geometry to
            # an accepted registration component. Isolated mask islands cannot
            # permanently enlarge a previously established object map.
            supported = self._connected_support(current, obj.world_points()) if len(own) else np.ones(len(current), bool)
            offset = 0
            added, updated = 0, 0
            for view in views:
                selected = supported[offset:offset+len(view.xyz)]
                offset += len(view.xyz)
                counts = self._integrate(obj, view.subset(selected), timestep)
                added += counts[0]
                updated += counts[1]
            self._contradictions(obj, cameras, timestep)
            self._age(obj, timestep)
            self._event(timestep, oid, obj.tracking_state, added=added, updated=updated,
                        rejected_disconnected=int((~supported).sum()),
                        confidence=obj.pose_confidence, uncertainty_m=obj.pose_uncertainty)
        return self

    @staticmethod
    def _connected_support(current, reference):
        if not len(current):
            return np.zeros(0, bool)
        pairs = np.asarray(list(cKDTree(current).query_pairs(.018)), dtype=int).reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs)*2),
                            (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                           shape=(len(current), len(current)))
        _, label = connected_components(graph, directed=False)
        near = cKDTree(reference).query(current)[0] < .012
        supported_components = np.unique(label[near])
        return np.isin(label, supported_components)

    def _integrate(self, obj, observation, timestep):
        local = transform(observation.xyz, np.linalg.inv(obj.world_from_object))
        normals = observation.normals @ obj.world_from_object[:3, :3]
        # Spatial hash searches neighbouring voxels; boundaries cannot create
        # duplicate identities merely because a measurement crossed a voxel.
        active = obj.active()
        grid = {}
        for node in active:
            key = tuple(np.floor(node.object_local_xyz/self.merge_distance).astype(int))
            grid.setdefault(key, []).append(node)
        added = updated = 0
        sources = observation.sources
        for i, point in enumerate(local):
            voxel = np.floor(point/self.merge_distance).astype(int)
            neighbours = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbours.extend(grid.get(tuple(voxel+[dx, dy, dz]), ()))
            candidates = []
            for node in neighbours:
                distance = np.linalg.norm(node.object_local_xyz-point)
                normal_ok = (not node.normal_valid or not observation.normal_valid[i]
                             or node.object_local_normal @ normals[i] >= self.normal_cosine)
                if distance <= self.merge_distance and normal_ok:
                    candidates.append((distance, node.persistent_node_id, node))
            source = sources[i]
            if candidates:
                node = min(candidates, key=lambda x: x[:2])[2]
                old_key = tuple(np.floor(node.object_local_xyz/self.merge_distance).astype(int))
                weight = min(node.observation_count, 20)
                node.object_local_xyz = (node.object_local_xyz*weight+point)/(weight+1)
                new_key = tuple(np.floor(node.object_local_xyz/self.merge_distance).astype(int))
                if new_key != old_key:
                    grid[old_key] = [n for n in grid[old_key] if n is not node]
                    grid.setdefault(new_key, []).append(node)
                if observation.normal_valid[i]:
                    normal = normals[i]
                    if node.normal_valid and node.object_local_normal @ normal < 0:
                        normal = -normal
                    normal = normal + (node.object_local_normal*weight if node.normal_valid else 0)
                    node.object_local_normal = normal/np.linalg.norm(normal)
                    node.normal_valid = True
                # Two cameras at one timestep are corroboration, not two
                # independent temporal observations.
                if node.last_seen_timestep != timestep:
                    node.observation_count += 1
                node.confidence = min(.95, node.confidence+.08)
                node.last_seen_timestep = timestep
                node.last_observed_camera = observation.camera
                if source not in node.source_observation_ids:
                    node.source_observation_ids.append(source)
                node.last_f5 = observation.f5[i].copy()
                node.last_feature_valid = observation.feature_valid[i].copy()
                node.currently_observed, node.estimated_only = True, False
                node.position_uncertainty = obj.pose_uncertainty
                node.contradiction_count = max(0, node.contradiction_count-1)
                updated += 1
            else:
                identity = f'{self.episode}:{self.next_id:08d}'
                self.next_id += 1
                node = Surfel(identity, obj.object_id, point.copy(), normals[i].copy(),
                              bool(observation.normal_valid[i]), .45, timestep, timestep,
                              observation.camera, 1, True, False, obj.pose_uncertainty,
                              [source], observation.f5[i].copy(), observation.feature_valid[i].copy())
                obj.nodes.append(node)
                grid.setdefault(tuple(voxel), []).append(node)
                added += 1
        return added, updated

    def _contradictions(self, obj, cameras, timestep):
        nodes = obj.active()
        if not nodes:
            return
        world = obj.world_points()
        contradicted = np.zeros(len(nodes), bool)
        for camera in cameras:
            xyz = transform(world, np.linalg.inv(camera.world_from_camera))
            positive = xyz[:, 2] > 0
            k = camera.intrinsic
            u = np.floor(xyz[:, 0]/np.maximum(xyz[:, 2], 1e-12)*k[0, 0]+k[0, 2]).astype(int)
            v = np.floor(xyz[:, 1]/np.maximum(xyz[:, 2], 1e-12)*k[1, 1]+k[1, 2]).astype(int)
            h, w = camera.depth.shape
            inside = positive & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            rows = np.flatnonzero(inside)
            z = camera.depth[v[rows], u[rows]]
            contradicted[rows] |= np.isfinite(z) & (xyz[rows, 2] < z-.008-obj.pose_uncertainty)
        for i, node in enumerate(nodes):
            if contradicted[i] and not node.currently_observed:
                node.contradiction_count += 1
                node.confidence *= .65
                if node.contradiction_count >= 4:
                    node.removed = True
                    self._event(timestep, obj.object_id, 'REMOVED_FREE_SPACE_CONTRADICTION',
                                node_id=node.persistent_node_id)

    @staticmethod
    def _age(obj, timestep):
        for node in obj.active():
            age = timestep-node.last_seen_timestep
            # Current world measurements do not inherit an unobservable object
            # frame's full pose uncertainty: inverse/forward transforms cancel
            # at this timestep. Historical surfaces still depend on that pose.
            node.position_uncertainty = .003 if node.currently_observed else obj.pose_uncertainty + age*.0005
            if not node.currently_observed:
                node.confidence *= .985

    def records(self):
        rows = []
        for obj in self.objects.values():
            for node in obj.nodes:
                state = (INVALID_REMOVED if node.removed else LOW_CONFIDENCE
                         if node.confidence*obj.pose_confidence < .15 or node.position_uncertainty > .02
                         else CURRENTLY_OBSERVED if node.currently_observed else PERSISTENT_ESTIMATED)
                row = vars(node).copy()
                row.update(status=state, world_xyz=transform(node.object_local_xyz, obj.world_from_object),
                           world_normal=node.object_local_normal @ obj.world_from_object[:3, :3].T,
                           pose_confidence=obj.pose_confidence)
                rows.append(row)
        return rows


def current_records(observations, *, fuse_views=True):
    """One-frame fusion using the same surfel duplicate rule, without memory."""
    if not observations:
        return []
    if not fuse_views:
        rows = []
        for observation in observations:
            for i, source in enumerate(observation.sources):
                rows.append(dict(persistent_node_id=f'{source[0]}:current{source[1]}:{source[2]}:{source[3]}',
                    object_id=observation.object_id, object_local_xyz=observation.xyz[i].copy(),
                    object_local_normal=observation.normals[i].copy(), normal_valid=bool(observation.normal_valid[i]),
                    confidence=1., first_seen_timestep=observation.timestep, last_seen_timestep=observation.timestep,
                    last_observed_camera=observation.camera, observation_count=1, currently_observed=True,
                    estimated_only=False, position_uncertainty=.001, source_observation_ids=[source],
                    last_f5=observation.f5[i].copy(), last_feature_valid=observation.feature_valid[i].copy(),
                    removed=False, status=CURRENTLY_OBSERVED, world_xyz=observation.xyz[i].copy(),
                    world_normal=observation.normals[i].copy(), pose_confidence=1.))
        return rows
    first = observations[0]
    memory = PersistentMap()
    memory.reset(first.episode + f':current{first.timestep}')
    for oid, obj in memory.objects.items():
        views = [o for o in observations if o.object_id == oid and len(o.xyz)]
        if not views:
            continue
        obj.world_from_object[:3, 3] = np.median(np.concatenate([o.xyz for o in views]), axis=0)
        obj.pose_confidence = 1.
        for view in views:
            memory._integrate(obj, view, first.timestep)
    return memory.records()

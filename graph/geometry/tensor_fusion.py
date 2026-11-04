"""Selected-observation consistency on Torch tensors, with explicit readback.

Only small identity/calibration metadata is validated on the host. Selected
points, depth, masks, keep decisions and returned observations stay on device.
The NumPy fusion module remains the independent legacy/reference path.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
import torch

from graph.geometry.contracts import SurfaceObservation, synchronized
from graph.perception.coordinate_frames import check_frame, packet_frame, world_from_frame
from graph.perception.object_groups import object_rows


@lru_cache(maxsize=8)
def _depth_table(device):
    # Correctly rounded Python/IEEE division matches NumPy signed-mm depth,
    # including exact strict-4mm decisions. Cache constants, never sensor data.
    values = [v/1000. if v > 0 else float('nan') for v in range(-32768, 32768)]
    return torch.tensor(values, dtype=torch.float64, device=torch.device(device))


def depth_from_mm(raw):
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.int16 or raw.ndim != 2:
        raise ValueError('signed int16 [H,W] depth tensor required')
    return _depth_table(str(raw.device))[raw.to(torch.int64)+32768]


@dataclass(frozen=True)
class TensorCamera:
    depth: torch.Tensor
    mask: torch.Tensor
    intrinsic: torch.Tensor
    camera_from_frame: torch.Tensor
    frame_from_camera: torch.Tensor
    camera_from_world: torch.Tensor | None = None
    world_from_camera: torch.Tensor | None = None


@dataclass(frozen=True)
class TensorViews:
    identity: tuple
    coordinate_frame: str
    by_name: dict
    device: torch.device

    def world_views(self):
        """Same snapshot/masks in World, sharing depth and calibration tensors."""
        if self.coordinate_frame == 'world':
            return self
        if any(c.camera_from_world is None or c.world_from_camera is None for c in self.by_name.values()):
            raise ValueError('prepared World calibration required')
        return replace(self, coordinate_frame='world', by_name={name: replace(c,
            camera_from_frame=c.camera_from_world, frame_from_camera=c.world_from_camera)
            for name, c in self.by_name.items()})

    def check(self, observation):
        if (observation.episode, observation.timestep) != self.identity[:2] or observation.camera not in self.by_name:
            raise ValueError('foreign tensor observation identity')
        if observation.coordinate_frame != self.coordinate_frame:
            raise ValueError('tensor observation/view coordinate frame mismatch')
        if self.coordinate_frame == 'ee' and (observation.reference_ee_pose is None or tuple(observation.reference_ee_pose) != self.identity[3]):
            raise ValueError('stale tensor EE reference pose')
        observation.validate_layout(self.device)


@torch.inference_mode()
def prepare_tensor_views(cameras, masks, *, coordinate_frame='ee', device='cuda', depths=None):
    """Fresh frame snapshot; optional depths reuse already resident float64 data.

    Sensor identity/calibration checks use original small host metadata once.
    Image depth conversion and all observation-dependent math use Torch.
    CPU Torch is an explicit test/device choice; there is no CUDA fallback.
    """
    frame = check_frame(coordinate_frame)
    synchronized(cameras)
    device = torch.empty(0, device=device).device
    if set(masks) != {c.camera for c in cameras}:
        raise ValueError('tensor view mask/camera identity')
    if depths is not None and set(depths) != set(masks):
        raise ValueError('tensor depth/camera identity')
    result = {}
    from graph.perception.vision_packet import validate_transform
    world_from_graph = world_from_frame(frame, cameras[0].tcp_pose)
    graph_from_world = np.linalg.inv(world_from_graph) if frame == 'ee' else None
    for c in cameras:
        h, w = c.depth_raw_mm.shape
        mask = torch.as_tensor(masks[c.camera], device=device)
        if mask.dtype != torch.bool or tuple(mask.shape) != (2, h, w):
            raise ValueError('bool [2,H,W] view masks required')
        if depths is None:
            raw = torch.as_tensor(c.depth_raw_mm, device=device)
            depth = depth_from_mm(raw)
        else:
            depth = depths[c.camera]
            if not isinstance(depth, torch.Tensor) or depth.device != device or depth.dtype != torch.float64 or tuple(depth.shape) != (h, w):
                raise ValueError('resident float64 depth alignment')
        # Frame-to-camera matrix composes the same validated calibration as
        # the reference; no observation points cross to CPU for this work.
        world = c.world_from_camera
        # Invert the original calibration dtype before promotion, exactly as
        # camera_from_frame does. Share this snapshot with the World gate.
        camera_from_world = np.linalg.inv(world)
        world = validate_transform(world)
        inverse_world = torch.as_tensor(camera_from_world @ np.eye(4), device=device)
        forward_world = torch.as_tensor(world, device=device)
        inverse = inverse_world if frame == 'world' else torch.as_tensor(camera_from_world @ world_from_graph, device=device)
        forward = forward_world if frame == 'world' else torch.as_tensor(validate_transform(graph_from_world @ world), device=device)
        result[c.camera] = TensorCamera(depth, mask,
            torch.as_tensor(np.asarray(c.intrinsic, np.float64), device=device),
            inverse, forward, inverse_world, forward_world)
    first = cameras[0]
    return TensorViews((first.episode, first.timestep, first.timestamp, tuple(first.tcp_pose)), frame, result, device)


@dataclass(frozen=True)
class TensorSurfaceObservation(SurfaceObservation):
    xyz: torch.Tensor
    normals: torch.Tensor
    normal_valid: torch.Tensor
    f5: torch.Tensor
    feature_valid: torch.Tensor
    pixels: torch.Tensor
    camera_origin: torch.Tensor
    structural: torch.Tensor | None = None

    def validate_layout(self, device):
        n = len(self.xyz)
        for name, dtype, shape in (
            ('xyz', torch.float64, (n, 3)), ('normals', torch.float64, (n, 3)),
            ('normal_valid', torch.bool, (n,)), ('f5', torch.float64, (n, 5)),
            ('feature_valid', torch.bool, (n, 3)), ('pixels', torch.int64, (n,)),
            ('camera_origin', torch.float64, (3,))):
            a = getattr(self, name)
            if not isinstance(a, torch.Tensor) or a.device != device or a.dtype != dtype or tuple(a.shape) != shape:
                raise ValueError('tensor observation shape/dtype/device: '+name)
        if self.object_id not in (0, 1):
            raise ValueError('tensor observation object identity')
        if self.structural is not None and (not isinstance(self.structural, torch.Tensor) or self.structural.device != device or self.structural.dtype != torch.bool or tuple(self.structural.shape) != (n,)):
            raise ValueError('tensor structural row alignment')
        return self

    def validate(self):
        from graph.perception.vision_packet import pose_matrix
        check_frame(self.coordinate_frame)
        if not self.episode or not self.camera or self.timestep < 0:
            raise ValueError('tensor observation identity')
        if self.coordinate_frame == 'ee':
            pose_matrix(self.reference_ee_pose)
        elif self.reference_ee_pose is not None:
            raise ValueError('World observation must not declare an EE reference')
        self.validate_layout(self.xyz.device)
        usable = self.feature_valid[:, [0, 0, 0, 1, 2]]
        norm = torch.linalg.vector_norm(self.normals, dim=1)
        good = torch.isfinite(self.xyz).all() & torch.isfinite(self.normals).all() & torch.isfinite(self.camera_origin).all()
        good &= (self.normal_valid == self.feature_valid[:, 0]).all()
        good &= (~self.normal_valid | ((norm-1).abs() <= 1.1e-4)).all()
        good &= (~usable | torch.isfinite(self.f5)).all() & (self.pixels >= 0).all()
        if not bool(good) or torch.unique(self.pixels).numel() != len(self.pixels):
            raise ValueError('tensor observation contract')
        return self

    def subset(self, selection):
        if selection.dtype != torch.bool or tuple(selection.shape) != (len(self.xyz),) or selection.device != self.xyz.device:
            raise ValueError('tensor subset mask alignment')
        fields = vars(self).copy()
        # Compute the dynamic compact row list once, sharing it across fields.
        rows = torch.nonzero(selection).flatten()
        for key in ('xyz', 'normals', 'normal_valid', 'f5', 'feature_valid', 'pixels', 'structural'):
            if fields[key] is not None:
                fields[key] = fields[key].index_select(0, rows)
        return TensorSurfaceObservation(**fields)

    @property
    def sources(self):
        raise TypeError('use numpy_observations explicitly before Python source records')


def tensor_observations(observations, *, device='cuda'):
    """Explicit compatibility upload for ready legacy observations."""
    result = []
    for o in observations:
        o.validate()
        fields = vars(o).copy()
        for key in ('xyz', 'normals', 'normal_valid', 'f5', 'feature_valid', 'pixels', 'camera_origin'):
            dtype = torch.bool if key in ('normal_valid','feature_valid') else torch.int64 if key == 'pixels' else torch.float64
            fields[key] = torch.as_tensor(fields[key], device=device, dtype=dtype).clone()
        result.append(TensorSurfaceObservation(**fields).validate_layout(fields['xyz'].device))
    return result


@torch.inference_mode()
def observations_from_tensor_packet(packet, camera):
    """Validated selected tensor packet -> same-frame observations, no readback."""
    if not isinstance(packet.get('points'), torch.Tensor):
        raise ValueError('selected tensor packet required')
    frame = packet_frame(packet)
    if (packet['camera'] != camera.camera or packet['episode'] != camera.episode
            or packet['source_frame'] != camera.timestep
            or not np.array_equal(packet['observation_ee_pose'], camera.tcp_pose)):
        raise ValueError('tensor packet/sensor identity or reference mismatch')
    device = packet['points'].device
    result = []
    matrix = packet['T_frame_from_camera'] if 'T_frame_from_camera' in packet else packet['T_EE_from_camera']
    for oid, rows in enumerate(object_rows(packet)):
        f5 = packet['geometric_features'][rows].clone()
        valid = packet['feature_field_valid'][rows].clone()
        normals = torch.where(valid[:, :1], f5[:, :3], 0.)
        result.append(TensorSurfaceObservation(camera.episode, camera.timestep, camera.camera, oid,
            packet['points'][rows].to(torch.float64).clone(), normals, valid[:, 0].clone(), f5, valid,
            packet['provenance']['source_index'][rows].clone(),
            torch.as_tensor(matrix[:3, 3].copy(), device=device), frame,
            tuple(camera.tcp_pose) if frame == 'ee' else None,
            ~packet['provenance']['surface_selected'][rows]).validate_layout(device))
    return result


@torch.inference_mode()
def consistency_keep(xyz, object_id, source_camera, views):
    """Exact original predicates; all-points safe gathers avoid inside nonzero."""
    if xyz.dtype != torch.float64 or xyz.device != views.device or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('float64 [N,3] consistency points required')
    if object_id not in (0, 1) or source_camera not in views.by_name:
        raise ValueError('consistency source identity')
    keep = torch.ones(len(xyz), device=xyz.device, dtype=torch.bool)
    if not len(xyz):
        return keep
    for name, c in views.by_name.items():
        if name == source_camera:
            continue
        m = c.camera_from_frame
        p = xyz @ m[:3, :3].T + m[:3, 3]
        k = c.intrinsic; h, w = c.depth.shape
        uv = torch.floor(p[:, :2]/p[:, 2, None].clamp_min(1e-12)
            * torch.stack((k[0, 0], k[1, 1]))+k[:2, 2]).to(torch.int64)
        inside = (p[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        u = uv[:, 0].clamp(0, w-1); v = uv[:, 1].clamp(0, h-1)
        depth_agrees = (c.depth[v, u]-p[:, 2]).abs() < .004
        other_only = c.mask[1-object_id, v, u] & ~c.mask[object_id, v, u]
        keep &= ~(inside & depth_agrees & other_only)
    return keep


@torch.inference_mode()
def consistent_views_tensor(observations, views):
    """Return compact observations and tensor-valued conflict audits on device."""
    result, audit = [], []
    for o in observations:
        views.check(o)
        keep = consistency_keep(o.xyz, o.object_id, o.camera, views)
        result.append(o.subset(keep))
        audit.append(dict(camera=o.camera, object_id=o.object_id, count=len(keep), unknown_conflict=(~keep).sum()))
    return result, audit


def numpy_observations(observations, *, include_structural=False):
    """One explicit packed selected-array readback for legacy Python records."""
    from graph.perception.tensor_packet import numpy_packet
    payload = numpy_packet({str(i): vars(o) for i, o in enumerate(observations)})
    result, flags = [], []
    for fields in payload.values():
        flags.append(fields.pop('structural'))
        result.append(SurfaceObservation(**fields).validate())
    return (result, flags) if include_structural else result

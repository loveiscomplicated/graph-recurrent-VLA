"""Current SF records and ordered greedy deduplication with resident tensors.

Per-point identity strings and dictionaries are materialized only by the explicit
numpy_records compatibility adapter. No observation arrays leave the device in
current_records_tensor or deduplicate_tensor.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch

from .tensor_fusion import TensorSurfaceObservation
from visual.runtime.coordinate_frames import check_frame


@dataclass(frozen=True)
class TensorRecords:
    episode: str
    timestep: int
    coordinate_frame: str
    reference_ee_pose: tuple | None
    camera_names: tuple
    xyz: torch.Tensor
    normals: torch.Tensor
    normal_valid: torch.Tensor
    f5: torch.Tensor
    feature_valid: torch.Tensor
    object_id: torch.Tensor
    structural: torch.Tensor
    camera_id: torch.Tensor
    pixels: torch.Tensor
    source_camera: torch.Tensor
    source_pixels: torch.Tensor
    source_offsets: torch.Tensor
    deduplicated: bool = False

    def __len__(self):
        return len(self.xyz)

    @property
    def device(self):
        return self.xyz.device


@torch.inference_mode()
def current_records_tensor(observations, *, coordinate_frame):
    """One current measurement per row; same-time, common-frame inputs only.

    Pass the current per-camera observations, including empty object groups, so
    identity/device/frame remain defined even for a frame with no visible points.
    Inputs are already validated tensor observations from packet/view processing.
    """
    frame = check_frame(coordinate_frame)
    if not observations:
        raise ValueError('empty frame requires identity-bearing empty observations')
    first = observations[0]
    device = first.xyz.device
    names = tuple(sorted({o.camera for o in observations}))
    for o in observations:
        if not isinstance(o, TensorSurfaceObservation):
            raise ValueError('tensor observations required')
        o.validate_layout(device)
        if (o.episode, o.timestep, o.coordinate_frame, o.reference_ee_pose) != (
                first.episode, first.timestep, frame, first.reference_ee_pose):
            raise ValueError('mixed record identity/frame/EE reference')
        if o.structural is None:
            raise ValueError('aligned structural flags required')
    if frame == 'ee':
        from visual.runtime.vision_packet import pose_matrix
        pose_matrix(first.reference_ee_pose)
    elif first.reference_ee_pose is not None:
        raise ValueError('World records must not declare an EE reference')
    def cat(name): return torch.cat([getattr(o, name) for o in observations])
    xyz = cat('xyz'); pixels = cat('pixels'); n = len(xyz)
    camera = torch.cat([torch.full((len(o.xyz),), names.index(o.camera),
                                  dtype=torch.int64, device=device) for o in observations])
    objects = torch.cat([torch.full((len(o.xyz),), o.object_id,
                                   dtype=torch.int64, device=device) for o in observations])
    return TensorRecords(first.episode, first.timestep, frame, first.reference_ee_pose, names,
        xyz, cat('normals'), cat('normal_valid'), cat('f5'), cat('feature_valid'), objects,
        cat('structural'), camera, pixels, camera, pixels,
        torch.arange(n+1, dtype=torch.int64, device=device))


def _order(records):
    # Stable tuple order: object, not structural, episode, time, camera, pixel.
    # Episode/time are common metadata. Do not stringify numeric pixel IDs.
    order = torch.argsort(records.pixels, stable=True)
    for values in (records.camera_id, ~records.structural, records.object_id):
        order = order[torch.argsort(values[order], stable=True)]
    return order


def _owners_cpu(xyz, objects, distance):
    # Explicit CPU Torch device path for contracts/CPU callers; no NumPy math.
    cells = torch.floor(xyz/distance).to(torch.int64)
    owners = torch.full((len(xyz),), -1, device=xyz.device, dtype=torch.int64)
    for i in range(len(xyz)):
        if int(owners[i]) >= 0: continue
        delta = xyz-xyz[i]
        norms = torch.sqrt((delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2])
        hit = (owners < 0) & (objects == objects[i]) & ((cells-cells[i]).abs() <= 1).all(1) & (norms < distance)
        owners[hit] = i
    return owners


@torch.inference_mode()
def deduplicate_tensor(records, distance=.003):
    """Same greedy representative rule, NOT transitive connected components.

    Process the earliest still-unassigned measurement as the next representative
    and claim all unassigned strict-radius neighbors. This is equivalent to the
    existing per-input-point loop selecting the earliest previously kept hit.
    XYZ/F5/validity are gathered from that real measurement, never averaged.
    """
    if records.deduplicated:
        raise ValueError('deduplication expects fresh one-source current records')
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('positive finite merge distance required')
    n = len(records); order = _order(records)
    xyz = records.xyz[order].contiguous(); objects = records.object_id[order].contiguous()
    if n and not bool((torch.isfinite(xyz) & ((xyz/distance).abs() < 2.**62)).all()):
        raise ValueError('finite XYZ within supported int64 grid required')
    if not n:
        owners = torch.empty(0, device=records.device, dtype=torch.int64)
    elif records.device.type == 'cuda':
        from . import tensor_records_kernels as kernels
        import triton
        cells = torch.empty_like(xyz, dtype=torch.int64)
        owners = torch.full((n,), -1, device=records.device, dtype=torch.int64)
        kernels.cells[(triton.cdiv(n*3, 256),)](xyz, cells, n, distance, 256, enable_fp_fusion=False)
        kernels.greedy[(1,)](xyz, objects, cells, owners, n, distance,
                            min(2048, triton.next_power_of_2(n)), enable_fp_fusion=False)
    elif records.device.type == 'cpu':
        owners = _owners_cpu(xyz, objects, distance)
    else:
        raise ValueError('explicit CPU or CUDA record device required')
    representatives = torch.nonzero(owners == torch.arange(n, device=records.device)).flatten()
    sources = torch.argsort(owners, stable=True)
    ends = torch.searchsorted(owners[sources], representatives, right=True)
    keep = order[representatives]; source_rows = order[sources]
    fields = {name: getattr(records, name)[keep] for name in (
        'xyz','normals','normal_valid','f5','feature_valid','object_id','structural','camera_id','pixels')}
    result = replace(records, **fields, source_camera=records.camera_id[source_rows],
        source_pixels=records.pixels[source_rows],
        source_offsets=torch.cat((torch.zeros(1, device=records.device, dtype=torch.int64), ends)),
        deduplicated=True)
    count = len(result)
    return result, dict(input_points=n, unique_points=count, duplicates=n-count,
        duplicate_fraction=(n-count)/n if n else 0.,
        representative='stable real measurement, no XYZ/F5 averaging')


def numpy_records(records):
    """Explicit packed readback into the existing Python record contract.

    For the default Graph this boundary is AFTER 3mm deduplication, immediately
    before the still-CPU final128-point FPS. Tensor callers need not invoke it.
    """
    from visual.runtime.tensor_packet import numpy_packet
    from .map import current_records
    from .contracts import SurfaceObservation
    names = ('xyz','normals','normal_valid','f5','feature_valid','object_id','structural',
             'camera_id','pixels','source_camera','source_pixels','source_offsets')
    arrays = numpy_packet({name: getattr(records, name) for name in names})
    made = {}
    for camera_id, camera in enumerate(records.camera_names):
        for oid in (0, 1):
            take = np.flatnonzero((arrays['camera_id'] == camera_id) & (arrays['object_id'] == oid))
            if not len(take): continue
            obs = SurfaceObservation(records.episode, records.timestep, camera, oid,
                arrays['xyz'][take], arrays['normals'][take], arrays['normal_valid'][take],
                arrays['f5'][take], arrays['feature_valid'][take], arrays['pixels'][take],
                np.zeros(3), records.coordinate_frame, records.reference_ee_pose)
            # current_records does not consume camera_origin; no camera geometry
            # is synthesized or emitted by this compatibility-only construction.
            rows = current_records([obs], fuse_views=False, coordinate_frame=records.coordinate_frame)
            for i, row in zip(take, rows): made[int(i)] = row
    result = []
    for i in range(len(records)):
        row = made[i]; row['structural'] = bool(arrays['structural'][i])
        a, b = arrays['source_offsets'][i:i+2]
        row['source_observation_ids'] = [(records.episode, records.timestep,
            records.camera_names[int(c)], int(p)) for c,p in
            zip(arrays['source_camera'][a:b], arrays['source_pixels'][a:b])]
        result.append(row)
    return result

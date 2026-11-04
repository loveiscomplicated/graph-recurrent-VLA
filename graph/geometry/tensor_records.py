"""Current SF records and ordered greedy deduplication with resident tensors.

Per-point identity strings and dictionaries are materialized only by the explicit
numpy_records compatibility adapter. No observation arrays leave the device in
current_records_tensor or deduplicate_tensor.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import numpy as np
import torch

from graph.geometry.tensor_fusion import TensorSurfaceObservation
from graph.perception.coordinate_frames import check_frame


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
        from graph.perception.vision_packet import pose_matrix
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
        import graph.geometry.tensor_records_kernels as kernels
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

    The default CUDA Graph does not call this adapter. Use it explicitly
    for legacy inspection or CPU file-writing after the resident graph path.
    """
    from graph.perception.tensor_packet import numpy_packet
    from graph.geometry.map import current_records
    from graph.geometry.contracts import SurfaceObservation
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


@lru_cache(maxsize=8)
def _identity_prefixes(camera_names, device):
    # Static identity metadata only. No current point, pose, mask or sensor cache.
    names = [str(name).encode('utf-8')+b':' for name in camera_names]
    width = max(map(len,names))
    prefix = torch.tensor([list(name)+[0]*(width-len(name)) for name in names],
                          dtype=torch.uint8,device=device)
    lengths = torch.tensor(list(map(len,names)),dtype=torch.int64,device=device)
    powers = torch.tensor([10**i for i in range(19)],dtype=torch.int64,device=device)
    return prefix,lengths,powers,width,(width+19+6)//7


def _identity_ranks(records):
    """Full persistent-ID string order; decimal pixel order differs from numeric."""
    n = len(records)
    if records.device.type == 'cpu':
        cameras = records.camera_id.tolist(); pixels = records.pixels.tolist()
        order = sorted(range(n),key=lambda i: records.camera_names[cameras[i]]+':'+str(pixels[i]))
        order = torch.tensor(order,dtype=torch.int64,device=records.device)
    else:
        import graph.geometry.tensor_records_kernels as kernels
        import triton
        prefix,lengths,powers,width,limbs = _identity_prefixes(records.camera_names,str(records.device))
        keys = torch.empty((n,limbs),dtype=torch.int64,device=records.device)
        kernels.identity_keys[(triton.cdiv(n*limbs,128),)](records.pixels,records.camera_id,
            prefix,lengths,powers,keys,n,width,limbs,128)
        order = torch.arange(n,device=records.device)
        for limb in range(limbs-1,-1,-1):
            order = order[torch.argsort(keys[order,limb],stable=True)]
    ranks = torch.empty(n,dtype=torch.int64,device=records.device)
    ranks[order] = torch.arange(n,device=records.device)
    return ranks


def _take_records(records, rows):
    """Gather representative fields and complete ragged provenance on device."""
    fields = {name:getattr(records,name)[rows] for name in (
        'xyz','normals','normal_valid','f5','feature_valid','object_id','structural','camera_id','pixels')}
    starts = records.source_offsets[rows]
    lengths = records.source_offsets[rows+1]-starts
    offsets = torch.cat((records.source_offsets.new_zeros(1),lengths.cumsum(0)))
    groups = torch.repeat_interleave(torch.arange(len(rows),device=records.device),lengths)
    sources = starts[groups]+torch.arange(len(groups),device=records.device)-offsets[groups]
    return replace(records,**fields,source_camera=records.source_camera[sources],
        source_pixels=records.source_pixels[sources],source_offsets=offsets)


@torch.inference_mode()
def budget_tensor(records, count=128):
    """Ordinary per-object final FPS, exactly mirroring budget(records,count).

    Keeps short clouds in original input order. Otherwise the seed and ties use
    persistent-ID STRING order, with float64 distances and no structural anchors.
    Array inputs/outputs stay on their explicit CPU or CUDA device. Dynamic object
    compaction/provenance allocation can synchronize; iterations do not read back.
    """
    if not isinstance(count,(int,np.integer)) or count < 0:
        raise ValueError('nonnegative integer point count required')
    check_frame(records.coordinate_frame)
    if records.coordinate_frame == 'ee':
        from graph.perception.vision_packet import pose_matrix
        pose_matrix(records.reference_ee_pose)
    elif records.reference_ee_pose is not None:
        raise ValueError('World records must not declare an EE reference')
    if records.device.type not in ('cpu','cuda'):
        raise ValueError('explicit CPU or CUDA record device required')
    if len(records) and not bool(torch.isfinite(records.xyz).all() & (records.pixels >= 0).all()):
        raise ValueError('finite XYZ and nonnegative source pixels required')
    rows = [torch.nonzero(records.object_id == oid).flatten() for oid in (0,1)]
    ranks = _identity_ranks(records) if count and any(len(r) > count for r in rows) else None
    selected = []
    for group in rows:
        if not count:
            selected.append(group[:0]);continue
        if len(group) <= count:
            selected.append(group);continue
        xyz = records.xyz[group].contiguous(); rank = ranks[group].contiguous()
        if records.device.type == 'cuda':
            import graph.geometry.tensor_records_kernels as kernels
            import triton
            width = min(2048,triton.next_power_of_2(len(group)))
            chosen = torch.empty(count,dtype=torch.int64,device=records.device)
            scratch = torch.empty(len(group) if len(group) > width else 0,
                                  dtype=torch.float64,device=records.device)
            kernels.final_fps[(1,)](xyz,rank,chosen,scratch,len(group),int(count),width,enable_fp_fusion=False)
        else:
            current = int(torch.argmin(rank)); chosen = []
            distance = torch.full((len(group),),torch.inf,dtype=torch.float64,device=records.device)
            for step in range(count):
                chosen.append(current)
                if step+1 < count:
                    delta = xyz-xyz[current]
                    d = (delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
                    distance = torch.minimum(distance,d); distance[chosen] = -1
                    candidates = distance == distance.max()
                    current = int(torch.argmin(torch.where(candidates,rank,torch.iinfo(torch.int64).max)))
            chosen = torch.tensor(chosen,dtype=torch.int64,device=records.device)
        selected.append(group[chosen])
    return _take_records(records,torch.cat(selected))

"""Native CUDA RGB-D observations; only small calibration metadata is host-side."""
from dataclasses import dataclass, fields
import torch
from graph.geometry.contracts import CameraObservation, validate_camera_metadata


@dataclass(frozen=True)
class TensorCameraObservation(CameraObservation):
    rgb: torch.Tensor
    depth_raw_mm: torch.Tensor

    def validate(self):
        if not self.episode or not self.camera or self.timestep<0:raise ValueError('observation identity')
        depth=self.depth_raw_mm;rgb=self.rgb
        if not isinstance(depth,torch.Tensor) or not depth.is_cuda or depth.dtype!=torch.int16 or depth.ndim!=2:
            raise ValueError('native CUDA signed-int16 depth required')
        if not isinstance(rgb,torch.Tensor) or rgb.device!=depth.device or rgb.dtype!=torch.uint8 or tuple(rgb.shape)!=(*depth.shape,3):
            raise ValueError('native CUDA uint8 RGB/depth alignment')
        validate_camera_metadata(self)
        return self

    @property
    def depth(self):
        from graph.geometry.tensor_fusion import depth_from_mm
        return depth_from_mm(self.depth_raw_mm)

    def camera_points(self):
        raise RuntimeError('dense XYZ disabled for tensor cameras; project selected mask pixels')

    def world_points(self):
        raise RuntimeError('dense XYZ disabled for tensor cameras; project selected mask pixels')


def tensor_camera(camera,device='cuda',*,clone=True):
    """Snapshot current image buffers on CUDA, preserving signed sensor depth."""
    values={f.name:getattr(camera,f.name) for f in fields(CameraObservation)}
    for name in ('rgb','depth_raw_mm'):
        x=torch.as_tensor(values[name],device=device)
        values[name]=x.clone() if clone else x
    for name in ('intrinsic','extrinsic_cv','cam2world_gl','tcp_pose'):
        values[name]=values[name].copy()
    return TensorCameraObservation(**values).validate()

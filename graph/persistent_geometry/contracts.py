"""Sensor-only contracts and causal, pixel-addressed observations.

Object actor poses, segmentation IDs and future observations are deliberately
absent from these input types. Evaluation and oracle adapters live separately.
"""
from dataclasses import dataclass
import numpy as np

from visual.runtime.vision_packet import pose_matrix, validate_transform


def transform(points, matrix):
    matrix = validate_transform(np.asarray(matrix, dtype=float))
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class CameraObservation:
    episode: str
    timestep: int
    camera: str
    timestamp: float
    rgb: np.ndarray
    depth_raw_mm: np.ndarray
    intrinsic: np.ndarray
    extrinsic_cv: np.ndarray
    cam2world_gl: np.ndarray
    tcp_pose: np.ndarray

    def validate(self):
        if not self.episode or not self.camera or self.timestep < 0:
            raise ValueError('observation identity')
        h, w = self.depth_raw_mm.shape
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError('RGB/depth alignment')
        if self.depth_raw_mm.dtype != np.int16:
            raise ValueError('original signed int16 millimetres required')
        k = np.asarray(self.intrinsic)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
                or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
                or k[0, 1] != 0 or k[1, 0] != 0):
            raise ValueError('CV intrinsic')
        cv = np.vstack((self.extrinsic_cv, [0, 0, 0, 1]))
        validate_transform(cv.astype(float))
        gl = validate_transform(np.asarray(self.cam2world_gl, float))
        # Independent conventions stored by the installed camera API.
        if not np.allclose(np.linalg.inv(cv), gl @ np.diag([1, -1, -1, 1]), atol=2e-5, rtol=0):
            raise ValueError('CV/GL camera calibration disagreement')
        pose_matrix(self.tcp_pose)
        if not np.isfinite(self.timestamp):
            raise ValueError('timestamp')
        return self

    @property
    def depth(self):
        # Preserve overflow and nonpositive values in the source; never unwrap
        # -32768 into fabricated positive depth.
        z = self.depth_raw_mm.astype(np.float64) / 1000
        return np.where(self.depth_raw_mm > 0, z, np.nan)

    @property
    def world_from_camera(self):
        # Match the existing recorded/live G0 CV calibration contract exactly.
        return np.linalg.inv(np.vstack((self.extrinsic_cv, [0, 0, 0, 1]))).astype(np.float32)

    def camera_points(self):
        v, u = np.indices(self.depth.shape)
        z = self.depth
        k = self.intrinsic
        return np.stack(((u + .5-k[0, 2])*z/k[0, 0],
                         (v + .5-k[1, 2])*z/k[1, 1], z), -1)

    def world_points(self):
        return transform(self.camera_points(), self.world_from_camera)


def synchronized(cameras):
    if not cameras or len({c.camera for c in cameras}) != len(cameras):
        raise ValueError('unique camera observations required')
    for c in cameras:
        c.validate()
    if len({(c.episode, c.timestep, c.timestamp) for c in cameras}) != 1:
        raise ValueError('cameras are not from the same physical timestep')
    if any(not np.array_equal(c.tcp_pose, cameras[0].tcp_pose) for c in cameras):
        raise ValueError('same-time robot/camera mismatch')


@dataclass(frozen=True)
class SurfaceObservation:
    episode: str
    timestep: int
    camera: str
    object_id: int
    xyz: np.ndarray                  # world metres
    normals: np.ndarray              # world unit vectors, zero when missing
    normal_valid: np.ndarray
    f5: np.ndarray                   # world normal, current variation/jump
    feature_valid: np.ndarray        # [N,3]
    pixels: np.ndarray               # current source raster indices, NOT IDs
    camera_origin: np.ndarray

    def validate(self):
        n = len(self.xyz)
        if (self.object_id not in (0, 1) or not self.episode or not self.camera
                or self.timestep < 0 or self.xyz.shape != (n, 3)
                or self.normals.shape != (n, 3) or self.normal_valid.shape != (n,)
                or self.f5.shape != (n, 5) or self.feature_valid.shape != (n, 3)
                or self.pixels.shape != (n,) or self.pixels.dtype.kind not in 'iu'
                or len(np.unique(self.pixels)) != n or (self.pixels < 0).any()
                or not np.isfinite(self.xyz).all() or not np.isfinite(self.normals).all()):
            raise ValueError('surface observation contract')
        if not np.array_equal(self.normal_valid, self.feature_valid[:, 0]):
            raise ValueError('normal validity disagrees')
        if n and not np.allclose(np.linalg.norm(self.normals[self.normal_valid], axis=1), 1, atol=1e-4):
            raise ValueError('nonunit measured normal')
        usable = self.feature_valid[:, [0, 0, 0, 1, 2]]
        if not np.isfinite(self.f5[usable]).all():
            raise ValueError('valid feature must be finite')
        return self

    def subset(self, selection):
        fields = vars(self).copy()
        for key in ('xyz', 'normals', 'normal_valid', 'f5', 'feature_valid', 'pixels'):
            fields[key] = fields[key][selection].copy()
        return SurfaceObservation(**fields)

    @property
    def sources(self):
        return [(self.episode, self.timestep, self.camera, int(p)) for p in self.pixels]


class CausalClock:
    def __init__(self):
        self.episode = None
        self.timestep = -1

    def advance(self, episode, timestep, observations):
        if episode != self.episode:
            if timestep != 0:
                raise ValueError('new episode must start at zero')
            self.episode, self.timestep = episode, -1
        if timestep != self.timestep + 1:
            raise ValueError('causal history must advance one frame at a time')
        for observation in observations:
            observation.validate()
            if observation.episode != episode or observation.timestep != timestep:
                raise ValueError('future/foreign/history observation')
        self.timestep = timestep

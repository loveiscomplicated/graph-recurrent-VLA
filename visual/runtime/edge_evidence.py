"""Observation-only adjacent-pixel evidence promoted unchanged from Task 02-C.

These are independent causes, never final structural labels or graph edges.
"""
from dataclasses import dataclass
import numpy as np
from visual.runtime.geometry import GeometryParameters
PARAMETERS = GeometryParameters()

@dataclass(frozen=True)
class PairEvidence:
    shape: tuple
    pixels: np.ndarray  # [P,2] source flat indices, right/down neighbour once
    supported: np.ndarray  # [P,2] unambiguous observed object endpoint
    mask_boundary: np.ndarray
    invalid_depth_boundary: np.ndarray  # XOR depth validity, not zero-depth jump
    overlap: np.ndarray  # [P,2], ambiguous tracker membership
    signed_depth_delta_m: np.ndarray  # z(second)-z(first), NaN unless both valid
    depth_jump: np.ndarray
    normal_angle_deg: np.ndarray  # NaN unless both normals available
    near_endpoint: np.ndarray  # local pair endpoint 0/1, -1 unless a depth jump
    far_endpoint: np.ndarray

    def endpoint_mask(self, selected, side=None):
        result = np.zeros(np.prod(self.shape), bool)
        for k in (0, 1):
            keep = selected & self.supported[:, k]
            if side is not None:
                keep &= side == k
            result[self.pixels[keep, k]] = True
        return result.reshape(self.shape)

    def candidates(self, threshold):
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('angle threshold must be finite and nonnegative')
        orientation = self.endpoint_mask(self.normal_angle_deg > threshold)
        depth = self.endpoint_mask(self.depth_jump)
        return dict(orientation=orientation, depth=depth, union=orientation | depth,
                    occluding=self.endpoint_mask(self.depth_jump, self.near_endpoint),
                    occluded=self.endpoint_mask(self.depth_jump, self.far_endpoint),
                    invalid=self.endpoint_mask(self.invalid_depth_boundary),
                    mask=self.endpoint_mask(self.mask_boundary))


def pair_evidence(geometry, object_id, normals, normal_valid, parameters=PARAMETERS):
    """Preserve all adjacent pairs touching the raw object mask, even invalids.

    Mask transition, invalid depth and overlap flags are independent. An external
    finite-depth neighbour is retained to determine jump direction. Its XYZ is
    never included as an object normal support or an object candidate point.
    """
    shape = geometry.valid_depth.shape
    indices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    mask = geometry.object_masks[object_id]
    support = geometry.observed_mask(object_id)
    parts = []
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        keep = mask[a] | mask[b]
        parts.append(np.column_stack((indices[a][keep], indices[b][keep])))
    pixels = np.concatenate(parts)
    valid = geometry.valid_depth.ravel()[pixels]
    both = valid.all(1)
    z = geometry.xyz[..., 2].ravel()[pixels]
    delta = np.full(len(pixels), np.nan, np.float32)
    delta[both] = z[both, 1]-z[both, 0]
    jump = np.zeros(len(pixels), bool)
    jump[both] = np.abs(delta[both]) > (parameters.depth_jump_abs_m
                    + parameters.depth_jump_rel*np.min(z[both], axis=1))
    nv = normal_valid.ravel()[pixels].all(1)
    angles = np.full(len(pixels), np.nan, np.float32)
    n = normals.reshape(-1, 3)[pixels[nv]]
    angles[nv] = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n[:, 0], n[:, 1]), axis=1),
                                      np.einsum('ij,ij->i', n[:, 0], n[:, 1])))
    near = np.full(len(pixels), -1, np.int8)
    near[jump] = np.argmin(z[jump], axis=1).astype(np.int8)
    far = np.where(jump, 1-near, -1).astype(np.int8)
    return PairEvidence(shape, pixels, support.ravel()[pixels],
                        np.logical_xor(mask.ravel()[pixels[:, 0]], mask.ravel()[pixels[:, 1]]),
                        valid[:, 0] ^ valid[:, 1], geometry.overlap_mask.ravel()[pixels],
                        delta, jump, angles, near, far)


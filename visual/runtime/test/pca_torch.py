"""Device-resident Torch PCA5 comparison, independent of the NumPy adapter.

Depth and tracker masks stay on the selected device. Float64 PCA preserves the
reference policy; CUDA measurements require an actual CUDA host. CPU Torch is
used for executable numerical contract tests, not claimed as a fast backend.
"""
from __future__ import annotations

import numpy as np
import torch

from visual.runtime.geometry import GeometryParameters
from visual.runtime.rgbd import PEG, BOX


class TorchPCA5:
    def __init__(self, intrinsic, shape, *, device='cuda', parameters=GeometryParameters(), backend='eigh'):
        k = np.asarray(intrinsic, float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1])):
            raise ValueError('invalid zero-skew intrinsic')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type == 'mps':
            raise ValueError('MPS cannot preserve the float64 PCA contract; use CPU or CUDA')
        self.shape, self.parameters = tuple(shape), parameters
        if backend not in ('eigh', 'triton', 'triton_fused'): raise ValueError('unknown Torch PCA backend')
        if backend.startswith('triton') and self.device.type!='cuda': raise ValueError('Triton backend requires CUDA')
        self.backend=backend
        h, w = self.shape
        if h < 1 or w < 1: raise ValueError('shape must be positive')
        self.ux = (torch.arange(w, dtype=torch.float64, device=self.device)+.5-k[0, 2])[None]
        self.vy = (torch.arange(h, dtype=torch.float64, device=self.device)+.5-k[1, 2])[:, None]
        self.fx, self.fy = float(k[0, 0]), float(k[1, 1])
        dv, du = torch.meshgrid(torch.arange(-2, 3, device=self.device), torch.arange(-2, 3, device=self.device), indexing='ij')
        self.dv, self.du = dv, du
        self.bits = (torch.ones(25, dtype=torch.int64, device=self.device) << torch.arange(25, device=self.device)).reshape(5, 5)

    def _jump(self, a, b):
        return (a-b).abs() > self.parameters.depth_jump_abs_m+self.parameters.depth_jump_rel*torch.minimum(a, b)

    def _normals(self, xyz, support):
        if self.backend.startswith('triton'):
            from visual.runtime.test.pca_triton import fused_normals
            return fused_normals(xyz,support,self.parameters)
        h, w = self.shape
        normals = torch.full_like(xyz, float('nan'))
        valid = torch.zeros_like(support)
        centres = support.nonzero()
        if not len(centres): return normals, valid
        v = centres[:, 0, None, None]+self.dv
        u = centres[:, 1, None, None]+self.du
        inside = (v >= 0) & (v < h) & (u >= 0) & (u < w)
        v, u = v.clamp(0, h-1), u.clamp(0, w-1)
        available = inside & support[v, u]
        points = xyz[v, u].to(torch.float64)
        z = torch.where(available, points[..., 2], 0)
        right_links = available[:, :, :-1] & available[:, :, 1:] & ~self._jump(z[:, :, :-1], z[:, :, 1:])
        down_links = available[:, :-1] & available[:, 1:] & ~self._jump(z[:, :-1], z[:, 1:])
        right = (right_links*self.bits[:, :-1]).sum((1, 2))
        down = (down_links*self.bits[:-1]).sum((1, 2))
        reached = torch.full((len(centres),), 1 << 12, dtype=torch.int64, device=self.device)
        # A simple path in a 25-vertex patch has <=24 edges. Fixed iterations
        # avoid a host .item()/synchronization at every flood-fill iteration.
        for _ in range(24):
            reached = reached | ((reached & right) << 1) | ((reached >> 1) & right) | ((reached & down) << 5) | ((reached >> 5) & down)
        component = (reached[:, None, None] & self.bits) != 0
        count = component.sum((1, 2))
        idx = (count >= 6).nonzero().flatten()
        centres, points, component, count = centres[idx], points[idx], component[idx], count[idx]
        delta = torch.where(component[..., None], points-xyz[centres[:, 0], centres[:, 1]][:, None, None], 0)
        mean = delta.sum((1, 2))/count[:, None]
        flat = delta.flatten(1, 2)
        covariance = flat.transpose(1, 2).bmm(flat)/count[:, None, None]-mean[:, :, None]*mean[:, None, :]
        values, basis = torch.linalg.eigh(covariance)
        threshold = torch.maximum(torch.full_like(values[:, 2], 1e-16), values[:, 2]*1e-8)
        good = (values[:, 1] > threshold) & (values[:, 1]-values[:, 0] > threshold) & torch.isfinite(values).all(1)
        centres, vectors = centres[good], basis[good, :, 0]
        vectors = vectors/torch.linalg.vector_norm(vectors, dim=1)[:, None]
        camera_dot = (vectors*xyz[centres[:, 0], centres[:, 1]]).sum(1)
        vectors = torch.where((camera_dot > 0)[:, None], -vectors, vectors)
        normals[centres[:, 0], centres[:, 1]] = vectors.to(torch.float32)
        valid[centres[:, 0], centres[:, 1]] = True
        return normals, valid

    def _features(self, xyz, valid_depth, mask, support, normals, normal_valid):
        shape, dev = self.shape, self.device
        z = torch.where(valid_depth, xyz[..., 2], 0)
        jump_m = torch.zeros(shape, dtype=torch.float32, device=dev)
        pair_valid = torch.zeros_like(support); discontinuity = torch.zeros_like(support)
        mask_boundary = torch.zeros_like(support); validity_boundary = torch.zeros_like(support)
        variation = torch.zeros_like(jump_m)
        count = torch.zeros(shape, dtype=torch.uint8, device=dev)
        safe_normals = torch.nan_to_num(normals)
        pairs = [((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))]
        for a, b in pairs:
            both_depth = valid_depth[a] & valid_depth[b]
            dz = (z[a]-z[b]).abs()
            jumps = both_depth & self._jump(z[a], z[b])
            both_normals = normal_valid[a] & normal_valid[b]
            n1, n2 = safe_normals[a], safe_normals[b]
            angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(n1, n2), dim=-1), (n1*n2).sum(-1))
            angle = torch.where(both_normals, angle, 0)
            for p, q in ((a, b), (b, a)):
                available = support[p] & both_depth
                pair_valid[p] |= available
                jump_m[p] = torch.maximum(jump_m[p], torch.where(available, dz, 0))
                discontinuity[p] |= support[p] & jumps
                mask_boundary[p] |= mask[p] & ~mask[q]
                validity_boundary[p] |= support[p] & ~valid_depth[q]
                variation[p] = torch.maximum(variation[p], angle)
                count[p] += both_normals.to(torch.uint8)
        variation_valid = normal_valid & (count > 0)
        image_boundary = torch.zeros_like(support)
        image_boundary[[0, -1], :] = mask[[0, -1], :]
        image_boundary[:, [0, -1]] = mask[:, [0, -1]]
        return dict(object_mask=mask, valid_points=support, normals=normals, normal_valid=normal_valid,
                    depth_jump_m=torch.where(pair_valid, jump_m, float('nan')), depth_pair_valid=pair_valid,
                    depth_discontinuity=discontinuity, mask_boundary=mask_boundary,
                    depth_validity_boundary=validity_boundary, image_boundary=image_boundary,
                    normal_variation_rad=torch.where(variation_valid, variation, float('nan')),
                    normal_variation_valid=variation_valid, normal_neighbor_count=count)

    @torch.inference_mode()
    def __call__(self, depth_m, masks):
        if depth_m.shape != self.shape or depth_m.device != self.device or not depth_m.is_floating_point():
            raise ValueError('depth must be floating (H,W) on the configured device')
        if set(masks)-{PEG, BOX}: raise ValueError('unknown object ID')
        masks = {i: masks.get(i, torch.zeros(self.shape, dtype=torch.bool, device=self.device)) for i in (PEG, BOX)}
        if any(m.dtype != torch.bool or m.shape != self.shape or m.device != self.device for m in masks.values()):
            raise ValueError('masks must be bool (H,W) on the configured device')
        depth_m=depth_m.contiguous()
        masks={i:m.contiguous() for i,m in masks.items()}
        valid_depth = torch.isfinite(depth_m) & (depth_m > 0)
        z = torch.where(valid_depth, depth_m, 0).to(torch.float64)
        xyz = torch.stack((self.ux*z/self.fx, self.vy*z/self.fy, z), dim=-1).to(torch.float32)
        xyz = torch.where(valid_depth[..., None], xyz, float('nan'))
        overlap = masks[PEG] & masks[BOX]
        objects = {}
        for oid, mask in masks.items():
            support = mask & valid_depth & ~overlap
            normals, normal_valid = self._normals(xyz, support)
            if self.backend=='triton_fused':
                from visual.runtime.test.pca_triton import fused_features
                objects[oid]=fused_features(xyz,valid_depth,mask,support,normals,normal_valid,self.parameters)
            else:
                objects[oid] = self._features(xyz, valid_depth, mask, support, normals, normal_valid)
        return dict(xyz=xyz, valid_depth=valid_depth, overlap_mask=overlap, objects=objects)

"""Depth-aware projection of actual convex robot link collision geometry.

Each convex collision shape is kept separate. A convex hull of the whole robot
would erase grasped objects and is forbidden here. No object GT enters this API.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull

from .contracts import transform


@dataclass(frozen=True)
class LinkSurface:
    link: str
    local_vertices: np.ndarray
    world_from_link: np.ndarray

    @property
    def world_vertices(self):
        return transform(self.local_vertices, self.world_from_link)


def robot_surfaces(env):
    """Installed SAPIEN geometry extraction, robot links only, same-time FK."""
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    result = []
    for link in env.unwrapped.agent.robot.get_links():
        entity = link._objs[0].entity
        matrix = entity.pose.to_transformation_matrix().astype(float)
        for component in entity.components:
            if isinstance(component, physx.PhysxRigidBaseComponent):
                for mesh in get_component_meshes(component):
                    if mesh is not None and len(mesh.vertices) >= 4:
                        result.append(LinkSurface(link.name, np.asarray(mesh.vertices).copy(), matrix.copy()))
    return result


def convex_depth(camera, surfaces, *, return_interior=False):
    """Ray/convex half-space clipping; depth is CV z in metres.

    Nonunit rays have camera z=1, so the ray parameter itself is metric depth.
    Entry intersections model the front surface, including moving finger pads.
    """
    h, w = camera.depth_raw_mm.shape
    v, u = np.indices((h, w))
    k = camera.intrinsic
    rays = np.stack(((u+.5-k[0, 2])/k[0, 0], (v+.5-k[1, 2])/k[1, 1], np.ones((h, w))), -1).reshape(-1, 3)
    wc = camera.world_from_camera
    direction = rays @ wc[:3, :3].T
    origin = wc[:3, 3]
    zbuffer = np.full(h*w, np.inf)
    owner = np.full(h*w, -1, np.int32)
    interior = np.zeros(h*w, bool)
    measured = camera.depth.ravel()
    for i, surface in enumerate(surfaces):
        vertices = surface.world_vertices
        vc = transform(vertices, np.linalg.inv(wc))
        if np.all(vc[:, 2] <= 0):
            continue
        if np.all(vc[:, 2] > 0):
            uv = vc[:, :2]/vc[:, 2, None] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2] - .5
            lo = np.maximum(np.floor(uv.min(0)).astype(int)-1, [0, 0])
            hi = np.minimum(np.ceil(uv.max(0)).astype(int)+1, [w-1, h-1])
            if np.any(hi < lo):
                continue
            vv, uu = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
            rows = (vv*w+uu).ravel()
        else:
            rows = np.arange(h*w)
        equations = ConvexHull(vertices).equations
        normals, offset = equations[:, :3], equations[:, 3]
        numerator = -(normals @ origin + offset)
        denominator = direction[rows] @ normals.T
        parallel = np.abs(denominator) < 1e-12
        crossing = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=~parallel)
        lower = np.max(np.where(denominator < -1e-12, crossing, -np.inf), axis=1)
        upper = np.min(np.where(denominator > 1e-12, crossing, np.inf), axis=1)
        outside_parallel = np.any(parallel & (numerator < 0), axis=1)
        entry = np.maximum(lower, 0)
        intersects = (upper >= entry) & (upper > 0) & ~outside_parallel
        # The full known collision volume is useful when a rendered visual
        # surface lies inside the link's collision hull. This is each actual
        # shape individually, never a hull or radius around the entire robot.
        interior[rows] |= (intersects & np.isfinite(measured[rows])
                           & (measured[rows] > entry+.0005)
                           & (measured[rows] < upper-.0005))
        hit = intersects & (entry < zbuffer[rows])
        zbuffer[rows[hit]] = entry[hit]
        owner[rows[hit]] = i
    result = (zbuffer.reshape(h, w), owner.reshape(h, w))
    return (*result, interior.reshape(h,w)) if return_interior else result


def classify_robot(camera, surfaces, *, tolerance_m=.0025, uncertainty_band_m=.006):
    predicted, owner, interior = convex_depth(camera, surfaces, return_interior=True)
    z = camera.depth
    valid = np.isfinite(z)
    difference = np.abs(z - predicted)
    confirmed = valid & ((np.isfinite(predicted) & (difference <= tolerance_m)) | interior)
    uncertain = valid & np.isfinite(predicted) & ~confirmed & (difference <= uncertainty_band_m)
    return dict(confirmed_robot=confirmed, uncertain=uncertain,
                candidate=valid & ~confirmed & ~uncertain,
                predicted_robot_depth=predicted, link_owner=owner,
                inside_known_link_volume=interior,
                method='per-shape collision CV-depth agreement or measured point inside actual link collision volume v2',
                tolerance_m=tolerance_m, uncertainty_band_m=uncertainty_band_m)

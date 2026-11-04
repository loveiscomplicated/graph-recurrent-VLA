"""Existing checked World-from-EE pose conversion used by runtime and viewer."""
import numpy as np

QUATERNION_NORM_ATOL = 1e-5


def pose_matrix(pose):
    """World-from-EE pose [x,y,z,qw,qx,qy,qz], with no identity fallback."""
    p = np.asarray(pose, np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('finite source observation TCP pose [7] required')
    norm = np.linalg.norm(p[3:])
    if abs(norm - 1) > QUATERNION_NORM_ATOL:
        raise ValueError('TCP wxyz quaternion is not unit')
    w, x, y, z = p[3:] / norm
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = p[:3]
    return t

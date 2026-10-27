"""Small deterministic SciPy/NumPy rigid tracker; no object model or GT input."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contracts import transform


@dataclass(frozen=True)
class Registration:
    pose: np.ndarray
    accepted: bool
    state: str
    residual_m: float
    inlier_ratio: float
    confidence: float
    translation_m: float
    rotation_rad: float
    rank: int


def rigid_fit(source, target):
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != b.shape or len(a) < 3:
        raise ValueError('three corresponding points required')
    ac, bc = a.mean(0), b.mean(0)
    u, s, vt = np.linalg.svd((a-ac).T @ (b-bc))
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = bc - rotation @ ac
    return matrix, int((s > max(s[0]*1e-5, 1e-12)).sum())


def register(local, current, previous_pose, *, max_distance=.012,
             min_inlier=.45, max_translation=.06, max_rotation=.6, iterations=20,
             local_normals=None, normal_valid=None):
    local, current = np.asarray(local, float), np.asarray(current, float)
    previous_pose = np.asarray(previous_pose, float)
    if min(len(local), len(current)) < 6:
        return Registration(previous_pose.copy(), False, 'INSUFFICIENT_SUPPORT',
                            0., 0., 0., 0., 0., 0)
    predicted = transform(local, previous_pose)
    plane = (local_normals is not None and normal_valid is not None and np.sum(normal_valid)>=6)
    # The second initialization permits reappearance translation, but cannot
    # import a GT pose or silently jump to an arbitrarily distant object.
    seeds = [previous_pose.copy()]
    shift = np.median(current, axis=0)-np.median(predicted, axis=0)
    if not plane and np.linalg.norm(shift) <= max_translation:
        moved = previous_pose.copy()
        moved[:3, 3] += shift
        seeds.append(moved)
    candidates = []
    for initial in seeds:
        pose = initial
        rank = 0
        for _ in range(iterations):
            world = transform(local, pose)
            distance, index = cKDTree(world).query(current)
            keep = distance < max_distance
            if keep.sum() < 6 or len(np.unique(index[keep])) < 4:
                break
            # Trim only correspondence residual, never object GT/target error.
            threshold = max(.0015, np.quantile(distance[keep], .85))
            keep &= distance <= threshold
            if keep.sum() < 6:
                break
            if plane:
                keep &= np.asarray(normal_valid)[index]
                if keep.sum()<6:
                    break
                p,q = world[index[keep]],current[keep]
                n = np.asarray(local_normals)[index[keep]] @ pose[:3,:3].T
                center=p.mean(0)
                scale=max(.005,float(np.sqrt(np.mean(np.sum((p-center)**2,axis=1)))))
                design=np.c_[n,np.cross((p-center)/scale,n)]
                residual=np.sum((q-p)*n,axis=1)
                u,s,vt=np.linalg.svd(design,full_matrices=False)
                observable=s>max(s[0]*.05,1e-8)
                rank=int(observable.sum())
                inverse=np.divide(1,s,out=np.zeros_like(s),where=observable)
                delta=(vt.T*inverse)@(u.T@residual)
                # Only identifiable rigid directions are updated. Tangential
                # translation on a partial plane is not inferred from a crop's
                # changing centroid or the arbitrary sample pattern.
                angle=delta[3:]/scale
                if np.linalg.norm(angle)>.15:
                    angle*=.15/np.linalg.norm(angle)
                correction=np.eye(4)
                correction[:3,:3]=Rotation.from_rotvec(angle).as_matrix()
                correction[:3,3]=center+delta[:3]-correction[:3,:3]@center
            else:
                correction, rank = rigid_fit(world[index[keep]], current[keep])
            pose = correction @ pose
            if np.linalg.norm(correction-np.eye(4)) < 1e-7:
                break
        distance, index = cKDTree(transform(local, pose)).query(current)
        good = distance < max_distance
        ratio = float(good.mean())
        residual = float(np.sqrt(np.mean(distance[good]**2))) if good.any() else float(max_distance)
        relative = pose @ np.linalg.inv(previous_pose)
        # Pose-frame origins can be away from world zero. Compare the map's
        # own origin, not the translation column of a world-origin correction.
        translation = float(np.linalg.norm(pose[:3, 3]-previous_pose[:3, 3]))
        angle = float(Rotation.from_matrix(relative[:3, :3]).magnitude())
        score = residual + (1-ratio)*max_distance + translation*.01 + angle*.0001
        candidates.append((score, pose, residual, ratio, translation, angle, rank))
    candidates.sort(key=lambda x: x[0])
    _, pose, residual, ratio, translation, angle, rank = candidates[0]
    state = 'TRACKED'
    if rank < (3 if plane else 2):
        state = 'SYMMETRY_AMBIGUITY'
    elif ratio < min_inlier or residual > max_distance*.65:
        state = 'REGISTRATION_FAILED'
    elif translation > max_translation or angle > max_rotation:
        state = 'POSE_JUMP'
    # Equal residual with materially different poses exposes ambiguous fits.
    if len(candidates) > 1:
        other = candidates[1]
        if (abs(other[2]-residual) < .0003 and abs(other[3]-ratio) < .05
                and np.linalg.norm(other[1][:3, 3]-pose[:3, 3]) > .012):
            state = 'SYMMETRY_AMBIGUITY'
    accepted = state == 'TRACKED'
    confidence = ratio*np.exp(-residual/.004) if accepted else 0.
    # Rank-2 surface geometry does not identify all rigid degrees robustly.
    if (plane and rank<6) or (not plane and rank==2):
        confidence *= .65
        if accepted:
            state = 'PARTIALLY_OBSERVABLE' if plane else 'TRACKED_PLANAR_UNCERTAINTY'
            confidence=min(confidence,.25)
    return Registration(pose if accepted else previous_pose.copy(), accepted, state,
                        residual, ratio, float(confidence), translation, angle, rank)

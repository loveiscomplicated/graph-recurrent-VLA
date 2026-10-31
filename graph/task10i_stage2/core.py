"""Frozen physical distance, provenance guards, and deterministic selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SEEDS = dict(split=1010202601, common=1010202602, random=1010202603,
             common_reserve=1010202604, random_reserve=1010202605, monte_carlo=1010202606)
WEIGHTS = np.array([.30, .20, .25, .15, .10], dtype=np.float64)
PROTOCOL = dict(
    version=1, precision='float64', rng='NumPy Generator PCG64', seeds=SEEDS,
    algorithm='greedy facility-location; minimize mean nearest-set distance on training220',
    tie_break='lexicographically smallest candidate ID on exact float64 score tie',
    formula='d = sum_g weight_g * group_distance_g; vector terms RMS, G3 mean(position_RMS, SO3_angle/scale)',
    groups=['G1 peg-minus-TCP world XYZ, m', 'G2 peg world SO(3) geodesic, rad',
            'G3 peg-minus-hole world XYZ and hole^-1*peg SO(3)',
            'G4 full peg length and full square thickness, m',
            'G5 mean of 8 projected OBB corners u,v divided by 128'],
    weights=WEIGHTS.tolist(), scaling='coordinate IQR; orientation pair-angle IQR; floors applied',
    floors=dict(g1_m=.01, g2_rad=float(np.deg2rad(10)), g3_position_m=.01,
                g3_angle_rad=float(np.deg2rad(10)), g4_m=[.01, .002], g5_image_fraction=4/128),
    floor_rationale='1cm translation/length, 2mm thickness, 10deg angle, 4px image resolution; predefined, not tuned',
    masks=dict(g1=[True]*3, g2='full rotation', g3=['XYZ','full relative rotation'],
               g4=[True,True], g5=[True,True]),
    excluded_from_objective=['TCP-peg scalar norm (duplicates XYZ)', 'yaw (duplicates SO3)',
        'hole radius (verified thickness/2 + fixed 3mm clearance)', 'image extent', 'boundary margin'],
    quaternion='world pose xyz,wxyz; normalize; SO3 angle=2*acos(abs(dot)); q/-q equivalent; pi reversal distinct',
    near_duplicate=dict(position_m=.001, angle_rad=float(np.deg2rad(1)), half_size_m=.0001,
                        robot_joint=.001, hole_radius_m=.0001),
    extremes='outside training253 coordinate p5/p95 on peg XYZ and size; diagnostic only',
    boundary_thresholds_px=[.1,1,4,8], monte_carlo_repeats=200,
    diagnostic_holdout='never passed to normalizer or greedy; evaluated after selection_identity freeze',
    expert_status='NOT_TESTED')


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rot(pose):
    return Rotation.from_quat(np.asarray(pose, dtype=np.float64)[[4,5,6,3]])


def angle(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(a @ b.T)
    # Snap roundoff at unity, retaining distinct rotations at physical resolution.
    dot[np.abs(dot-1) < 1e-14] = 1
    return 2*np.arccos(np.clip(dot, 0, 1))


def descriptor(state):
    p,t,h = [np.asarray(state[k], dtype=np.float64) for k in ('peg_pose','tcp_pose','hole_pose')]
    half = np.asarray(state['peg_half'], dtype=np.float64)
    for pose in (p,t,h):
        assert pose.shape == (7,) and np.isfinite(pose).all()
        assert abs(np.linalg.norm(pose[3:])-1) < 1e-5
    assert half.shape == (3,) and (half > 0).all()
    assert np.isclose(half[1], half[2], atol=1e-8, rtol=0)
    radius = float(np.asarray(state['hole_radius']).reshape(-1)[0])
    assert abs(radius-half[1]-.003) < 1e-7, 'hole clearance contract changed'
    rp,rh = rot(p),rot(h)
    local = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*half
    world = rp.apply(local)+p[:3]
    camera = world @ np.asarray(state['extrinsic_cv'],float)[:,:3].T + state['extrinsic_cv'][:,3]
    assert (camera[:,2] > .01).all()
    pix = camera @ np.asarray(state['intrinsic'],float).T
    uv = pix[:,:2]/pix[:,2:]
    lo,hi = uv.min(0),uv.max(0)
    out = dict(g1=p[:3]-t[:3], g2=rp.as_quat(), g3p=p[:3]-h[:3],
        g3q=(rh.inv()*rp).as_quat(), g4=2*half[:2], g5=uv.mean(0)/128,
        peg_xyz=p[:3], peg_yaw_rad=rp.as_euler('xyz')[2], hole_pose=h,
        tcp_peg_distance_m=np.linalg.norm(p[:3]-t[:3]), peg_hole_angle_rad=(rh.inv()*rp).magnitude(),
        hole_radius_m=radius, pixel_extent=hi-lo, corners_uv=uv,
        boundary_margin_px=float(min(*lo,*(128-hi))))
    out = {k: v.tolist() if isinstance(v,np.ndarray) else float(v) for k,v in out.items()}
    assert all(np.isfinite(np.asarray(v)).all() for v in out.values())
    return out


def partition(ids, nval=20, nhold=60):
    ids = sorted(ids); assert len(ids)==len(set(ids))
    order = np.random.Generator(np.random.PCG64(SEEDS['split'])).permutation(ids).tolist()
    return dict(validation=order[:nval], holdout=order[nval:nval+nhold], training=order[nval+nhold:],
                full_sampling_trace=order)


def guard(rows, allowed):
    ids = [r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(allowed):
        raise ValueError('Selection provenance guard: exact training/anchor membership required; holdout/validation forbidden')
    if any(r['role'] not in ('existing_train','training') for r in rows):
        raise ValueError('Holdout or validation data prohibited in fitting/selection')


def fit(rows, allowed):
    guard(rows, allowed)
    floors=PROTOCOL['floors']; result=dict(fit_ids=sorted(allowed), protocol_sha256=canonical_hash(PROTOCOL))
    for key,floor in [('g1',floors['g1_m']),('g3p',floors['g3_position_m']),
                      ('g4',floors['g4_m']),('g5',floors['g5_image_fraction'])]:
        a=np.array([r['descriptor'][key] for r in rows],dtype=np.float64)
        iqr=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0)
        result[key]=dict(median=np.median(a,axis=0).tolist(),iqr=iqr.tolist(),scale=np.maximum(iqr,floor).tolist())
    for key,floor in [('g2',floors['g2_rad']),('g3q',floors['g3_angle_rad'])]:
        a=np.array([r['descriptor'][key] for r in rows]); v=angle(a,a)[np.triu_indices(len(a),1)]
        iqr=float(np.quantile(v,.75)-np.quantile(v,.25))
        result[key]=dict(pair_angle_iqr=iqr,scale=max(iqr,floor))
    return result


def distances(a,b,scales):
    def vec(key):
        x=np.array([r['descriptor'][key] for r in a]); y=np.array([r['descriptor'][key] for r in b])
        return np.sqrt(np.mean(((x[:,None]-y[None,:])/scales[key]['scale'])**2,axis=-1))
    def ori(key):
        return angle([r['descriptor'][key] for r in a],[r['descriptor'][key] for r in b])/scales[key]['scale']
    groups=np.stack([vec('g1'),ori('g2'),.5*(vec('g3p')+ori('g3q')),vec('g4'),vec('g5')],axis=-1)
    return groups@WEIGHTS, groups


def random_order(ids, seed):
    return np.random.Generator(np.random.PCG64(seed)).permutation(sorted(ids)).tolist()


def greedy(rows, anchor_rows, scales, count=None):
    """Only accepts training demand points and existing training anchors."""
    guard(rows, [r['id'] for r in rows])
    if any(r['role']!='training' for r in rows) or any(r['role'] not in ('training','existing_train') for r in anchor_rows):
        raise ValueError('Non-training greedy demand or anchor')
    if not {r['id'] for r in rows+anchor_rows}.issubset(scales['fit_ids']):
        raise ValueError('Greedy IDs outside frozen fitting membership')
    assert len({r['id'] for r in anchor_rows})==len(anchor_rows)
    rows=sorted(rows,key=lambda r:r['id']);ids=[r['id'] for r in rows]
    matrix,_=distances(rows,rows,scales); init,_=distances(rows,anchor_rows,scales)
    nearest=init.min(1);used={r['id'] for r in anchor_rows}; trace=[]
    for _ in range(count if count is not None else len([i for i in ids if i not in used])):
        scores=np.minimum(nearest[:,None],matrix).mean(0)
        scores[[i for i,cid in enumerate(ids) if cid in used]]=np.inf
        j=int(np.argmin(scores)); assert np.isfinite(scores[j])
        before=float(nearest.mean());nearest=np.minimum(nearest,matrix[:,j])
        assert nearest.mean()<=before+1e-14
        trace.append(dict(id=ids[j],rank=len(trace)+1,objective_before=before,
                          objective_after=float(nearest.mean()),gain=before-float(nearest.mean())))
        used.add(ids[j])
    return trace


def freeze(path,obj):
    """Immutable, atomic write with byte-equivalent resume; divergent results fail."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=data: raise ValueError('Frozen artifact changed: '+str(path))
    else:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(data);temp.replace(path)
    assert json.loads(path.read_text())==obj
    return hashlib.sha256(data).hexdigest()


def require_frozen(root):
    root=Path(root); identity=json.loads((root/'selection_identity.json').read_text())
    for name, expected in identity['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Selection freeze hash failed: '+name)
    return identity

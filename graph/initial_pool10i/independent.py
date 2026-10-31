"""Independent GL camera/slab rays, without convex_depth/evaluate."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

def validate(root):
    rows=[];max_error=0.
    for f in sorted((root/'candidates').glob('*/record.json')):
        r=json.loads(f.read_text());d=np.load(f.parent/'initial.npz');ev=np.load(f.parent/'visibility_evidence.npz');v=json.loads((f.parent/'visibility.json').read_text())
        p=d['peg_pose'];half=d['peg_half'];R=Rotation.from_quat(np.r_[p[4:],p[3]]).as_matrix()
        wc=d['cam2world_gl'].astype(float)@np.diag([1,-1,-1,1]);cv=np.linalg.inv(wc);K=d['intrinsic'].astype(float)
        signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
        world=(signs*half)@R.T+p[:3];xyz=world@cv[:3,:3].T+cv[:3,3];uv=xyz[:,:2]/xyz[:,2,None]*[K[0,0],K[1,1]]+K[:2,2]
        inside=(xyz[:,2]>.01)&(uv[:,0]>=-.0001)&(uv[:,0]<=128.0001)&(uv[:,1]>=-.0001)&(uv[:,1]<=128.0001)
        assert bool(inside.all())==v['full_in_frame']
        y,x=np.indices((128,128));directions=np.stack(((x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones_like(x)),-1)
        origin=(wc[:3,3]-p[:3])@R;dirs=directions@wc[:3,:3].T@R;parallel=np.abs(dirs)<1e-12;safe=np.where(parallel,1,dirs)
        a=(-half-origin)/safe;b=(half-origin)/safe
        lower=np.where(parallel,-np.inf,np.minimum(a,b));upper=np.where(parallel,np.inf,np.maximum(a,b))
        enter=np.maximum(lower.max(-1),0);leave=upper.min(-1);outside=(parallel&((origin < -half)|(origin > half))).any(-1)
        hit=(leave>=enter)&(leave>0)&~outside&(enter>.01)
        assert np.array_equal(hit,ev['expected_mask']),(r['candidate_id'],'ray mask')
        if hit.any():
            err=float(np.max(abs(enter[hit]-ev['expected_depth'][hit])));max_error=max(max_error,err);assert err<.0001
        localx=(origin+enter[...,None]*dirs)[...,0]
        masks=dict(white_end=hit&(localx<=-half[0]+.0175),red_end=hit&(localx>=half[0]-.0175),grasp=hit&(abs(localx-v['grasp_local'][0])<=.00875))
        for k,m in masks.items():assert np.array_equal(m,ev[k]),(r['candidate_id'],k)
        dep=np.where(d['depth_raw_mm']>0,d['depth_raw_mm'].astype(float)/1000,np.nan);supported=hit&np.isfinite(dep)&(abs(dep-enter)<=.0015)
        verdict=bool(inside.all() and all(v['regions'][k]['in_frame'] and m.any() and supported[m].all() for k,m in masks.items()))
        assert verdict==v['requested_pass'],r['candidate_id']
        rows.append(r['candidate_id'])
    out=dict(status='PASS',attempts=len(rows),frustum_exact=len(rows),ray_masks_exact=len(rows),regions_exact=len(rows),primary_verdict_exact=len(rows),max_depth_error_m=max_error,
        depth_crosscheck_bound_m=.0001,gate_depth_tolerance_unchanged_m=.0015)
    (root/'independent_visibility_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

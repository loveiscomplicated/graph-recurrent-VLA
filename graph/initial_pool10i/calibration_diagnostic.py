"""Read-only independent calibrated OBB slab comparison: CV contract and GL sidecar."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def analyze(root):
    out={m:dict(rows=0,mask_differences=[],region_differences=[],verdict_differences=[],max_depth_error_m=0.) for m in ['cv','gl']}
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    for f in sorted((root/'candidates').glob('*/record.json')):
        rec=json.loads(f.read_text());d=np.load(f.parent/'initial.npz');ev=np.load(f.parent/'visibility_evidence.npz');v=json.loads((f.parent/'visibility.json').read_text());p=d['peg_pose'];half=d['peg_half'];K=d['intrinsic'].astype(float)
        R=Rotation.from_quat(np.r_[p[4:],p[3]]).as_matrix()
        for mode in out:
            wc=d['cam2world_gl'].astype(float)@np.diag([1,-1,-1,1]) if mode=='gl' else np.linalg.inv(np.vstack((d['extrinsic_cv'],[0,0,0,1]))).astype(np.float32)
            cv=np.linalg.inv(wc);world=(signs*half)@R.T+p[:3];xyz=world@cv[:3,:3].T+cv[:3,3];uv=xyz[:,:2]/xyz[:,2,None]*[K[0,0],K[1,1]]+K[:2,2]
            inside=(xyz[:,2]>.01)&(uv[:,0]>=-.0001)&(uv[:,0]<=128.0001)&(uv[:,1]>=-.0001)&(uv[:,1]<=128.0001)
            yy,xx=np.indices((128,128));rays=np.stack(((xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones_like(xx)),-1)
            origin=(wc[:3,3]-p[:3])@R;dirs=rays@wc[:3,:3].T@R;parallel=np.abs(dirs)<1e-12;safe=np.where(parallel,1,dirs)
            a=(-half-origin)/safe;b=(half-origin)/safe;lo=np.where(parallel,-np.inf,np.minimum(a,b));hi=np.where(parallel,np.inf,np.maximum(a,b))
            enter=np.maximum(lo.max(-1),0);leave=hi.min(-1);outside=(parallel&((origin < -half)|(origin > half))).any(-1);hit=(leave>=enter)&(leave>0)&~outside&(enter>.01)
            item=out[mode];item['rows']+=1;diff=hit^ev['expected_mask']
            if diff.any():
                pixels=[]
                for y,x in zip(*np.where(diff)):
                    pixels.append(dict(u=int(x),v=int(y),independent_hit=bool(hit[y,x]),reference_hit=bool(ev['expected_mask'][y,x]),slab_enter=float(enter[y,x]),slab_leave=float(leave[y,x]),gap_m=float(enter[y,x]-leave[y,x]),gt_peg=bool(d['peg_mask'][y,x]),required_reference=[k for k in ['white_end','red_end','grasp'] if ev[k][y,x]]))
                item['mask_differences'].append(dict(candidate_id=rec['candidate_id'],eligible=rec['eligible'],pixels=pixels))
            common=hit&ev['expected_mask'];err=float(np.max(abs(enter[common]-ev['expected_depth'][common]))) if common.any() else 0.;item['max_depth_error_m']=max(item['max_depth_error_m'],err)
            localx=(origin+enter[...,None]*dirs)[...,0];masks=dict(white_end=hit&(localx<=-half[0]+.0175),red_end=hit&(localx>=half[0]-.0175),grasp=hit&(abs(localx-v['grasp_local'][0])<=.00875))
            for k,m in masks.items():
                if not np.array_equal(m,ev[k]):item['region_differences'].append(dict(candidate_id=rec['candidate_id'],region=k,pixels=int((m^ev[k]).sum())))
            dep=np.where(d['depth_raw_mm']>0,d['depth_raw_mm'].astype(float)/1000,np.nan);support=hit&np.isfinite(dep)&(abs(dep-enter)<=.0015)
            verdict=bool(inside.all() and all(v['regions'][k]['in_frame'] and m.any() and support[m].all() for k,m in masks.items()))
            if verdict!=v['requested_pass']:item['verdict_differences'].append(dict(candidate_id=rec['candidate_id'],reference=v['requested_pass'],independent=verdict))
    return out
if __name__=='__main__':
    result=analyze(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

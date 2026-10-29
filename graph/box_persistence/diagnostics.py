"""Read-only boundary continuity, uncertainty/error and missing-frame strata."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from graph.integration_baseline.io import write_json,save_npz
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.metrics import sample_meshes
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix
from .media import load


def longest(mask):
    if not len(mask):return 0
    runs=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(runs==1);ends=np.flatnonzero(runs==-1)
    return int(max(ends-starts,default=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    uncertainty=[];continuity=[];strata=[]
    for ep in sorted((a.geometry/'geometry').glob('seed*')):
        seed=int(ep.name[4:]);source=a.capture/'collection'/ep.name;mesh=sample_meshes(source/'evaluation_meshes.npz')[1]
        prior_missing=False
        for f in sorted(p for p in ep.glob('f*') if p.is_dir()):
            t=int(f.name[1:]);ref=load(f/'evaluation_reference.npz');b0=load(f/'B0_native.npz');b0n=int((b0['object_id']==1).sum())
            visible=len(ref['main_visible']);label='GT_HIDDEN' if not visible else 'PREDICTED_CURRENT_MISSING' if not b0n else 'REAPPEARANCE_AFTER_MISSING' if prior_missing else 'CURRENT_PRESENT'
            prior_missing=not b0n
            for n in ('B0','B1','B2'):
                rr=load(f/(n+'_native.npz'));box=rr['object_id']==1;xyz=rr['world_xyz'][box]
                err=mesh['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(ref['actor_pose']))))[0]
                estimated=rr['estimated_only'][box]
                strata.append(dict(seed=seed,timestep=t,variant=n,stratum=label,native_nodes=len(xyz),current_visible_points=visible,
                    correct_retained_estimated_nodes_4mm=int(((err<.004)&estimated).sum()),wrong_retained_estimated_nodes_8mm=int(((err>.008)&estimated).sum()),
                    surface_mean_mm=float(err.mean()*1000) if len(err) else None,
                    note='evaluated interval frames only; missing and GT-hidden differ'))
                uncertainty.extend([seed,t,0 if n=='B0' else 1 if n=='B1' else 2,int(est),float(u),float(e),float(c)] for est,u,e,c in zip(estimated,rr['position_uncertainty'][box],err,rr['confidence'][box]))
                ids=list(rr['persistent_node_id'])
                for budget in ('native','cap128','matched_N'):
                    if budget=='native':selected=xyz
                    else:
                        names=json.loads((f/(n+'_'+budget+'_graph.json')).read_text())['node_ids']
                        ind=[ids.index(i) for i in names if rr['object_id'][ids.index(i)]==1];selected=rr['world_xyz'][ind]
                    rim=ref['rim'];covered=cKDTree(selected).query(rim)[0]<.004 if len(selected) else np.zeros(len(rim),bool)
                    for scope,key in [('current','main_visible'),('historical','historical_observed')]:
                        observed=cKDTree(ref[key]).query(rim)[0]<.004 if len(ref[key]) else np.zeros(len(rim),bool)
                        ratio=[]
                        for eligible,hit in zip(observed.reshape(8,64),covered.reshape(8,64)):
                            den=longest(eligible)
                            if den>=4:ratio.append(longest(eligible&hit)/den)
                        continuity.append(dict(seed=seed,timestep=t,variant=n,budget=budget,scope=scope,
                            supported_rim_segments=len(ratio),mean_longest_run_retention=float(np.mean(ratio)) if ratio else None,
                            definition='8 sampled square-rim sides; longest covered eligible contiguous run / longest observed eligible run, segments need4samples'))
    u=np.array(uncertainty,float).reshape(-1,7)
    save_npz(a.output/'uncertainty_vs_error.npz',seed=u[:,0].astype(int),timestep=u[:,1].astype(int),variant_index=u[:,2].astype(int),estimated=u[:,3].astype(bool),uncertainty_m=u[:,4],surface_error_m=u[:,5],confidence=u[:,6])
    bins=[]
    for n,index in [('B0',0),('B1',1),('B2',2)]:
        for lo,hi in ((0.,.0031),(.0031,.008),(.008,.02),(.02,1e9)):
            sel=u[(u[:,2]==index)&(u[:,4]>=lo)&(u[:,4]<hi)]
            bins.append(dict(variant=n,uncertainty_min_m=lo,uncertainty_max_m=hi,nodes=len(sel),
                mean_error_mm=float(sel[:,5].mean()*1000) if len(sel) else None,
                uncertainty_covers_error_fraction=float((sel[:,5]<=sel[:,4]).mean()) if len(sel) else None,
                independent_sample_warning='repeated nodes/frames, descriptive diagnostic not calibration guarantee'))
    csv_write(a.output/'uncertainty_bins.csv',bins);csv_write(a.output/'rim_continuity.csv',continuity);csv_write(a.output/'occlusion_reappearance_strata.csv',strata)
    write_json(a.output/'scope.json',dict(input_geometry_unchanged=True,GT_scope='evaluation only',point_metrics='native; repeated nodes not independent',
        rim_continuity='diagnostic supplementary metric, not a post-result selection rule',occlusion_scope='every5th and last frame; no dynamic contact recovery claim'))
if __name__=='__main__':main()

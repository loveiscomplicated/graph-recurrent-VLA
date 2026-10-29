"""Task10-C2: read-only observational matching; no model or simulator execution.
Run from repository root: python -m graph.plan_a_bc.learning_signal_audit
"""
import os
os.environ['WANDB_MODE']='disabled'
import csv, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.bc08_codec import controller_commands, ActionCodec

ROOT=Path.cwd(); R=ROOT/'experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot'; O=ROOT/'outputs/task10c2'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def cosine(a,b):
 d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
 return np.divide(np.sum(a*b,axis=1),d,out=np.full(len(a),np.nan),where=d>1e-10)

# Fixed before any action loading. Every threshold is reported, none selected by outcome.
LEVELS={'strict':dict(tcp_mm=3,q_rad=.02,angle_deg=3,qvel=.10,opening_mm=1,gripvel=.01,linear=.03,angular=.15,time=3,box_mm=5),
 'primary':dict(tcp_mm=5,q_rad=.04,angle_deg=5,qvel=.20,opening_mm=2,gripvel=.02,linear=.05,angular=.25,time=5,box_mm=10),
 'sensitivity':dict(tcp_mm=10,q_rad=.08,angle_deg=10,qvel=.40,opening_mm=4,gripvel=.04,linear=.10,angular=.50,time=10,box_mm=20)}
CONFIG=dict(parent_comparison_uuid=R.parent.name,levels=LEVELS,history_lags=[0,1,3,5],support_min=32,span_max_mm=250,centroid_bbox_max_mm=20,
 peg_bins_mm=[0,5,10,20,40,1000000],control_max_mm=5,different_min_mm=20,small_expert_delta_mm=1,
 phase='observed opening>=65mm=open, <=45mm=small, else transition; open t<30=early_open reach proxy; no action/GT/terminal-length phase',
 selection='all cross-episode radius candidates filtered by every group incl history; block subset: one per unordered episode pair and 10-step episode block, greedy robot score; episode-disjoint sensitivity separately',
 geometry='Graph node_input XYZ already TCP-relative; observed surface centroid, never GT pose',test12=False,training=False)

def geometry(split,ids,identities):
 rows=[];robots=[];histories=[];poses=[]
 for ep in ids:
  p=R/f'cache/ep{ep}/graphs.npz';h=sha(p);receipt=read(p.with_name('receipt.json'));assert h==receipt['sha256'];identities[str(p)]=h
  a=load(p,['node_input','node_type','node_ptr','robot76','timestep','presence2']);r=a['robot76'].astype(float);n=len(r);assert np.array_equal(a['timestep'],np.arange(n));robots.extend(r)
  lag=np.maximum(np.arange(n)[:,None]-np.array([0,1,3,5]),0);histories.extend(r[lag]);poses.extend(Rotation.from_quat(r[:,[28,29,30,27]]).as_matrix())
  for t in range(n):
   lo,hi=a['node_ptr'][t:t+2];typ=a['node_type'][lo:hi];xyz=a['node_input'][lo:hi,:3];row=dict(split=split,episode=ep,t=t,opening_mm=r[t,22]*1000)
   row['phase']='early_open' if r[t,22]>=.065 and t<30 else 'open' if r[t,22]>=.065 else 'small' if r[t,22]<=.045 else 'transition'
   for kind,name in [(3,'peg'),(4,'box')]:
    pp=xyz[typ==kind];count=len(pp);assert bool(count)==bool(a['presence2'][t,kind-3]);c=pp.mean(0) if count else np.full(3,np.nan);span=np.ptp(pp,axis=0) if count else np.full(3,np.nan);bc=(pp.min(0)+pp.max(0))/2 if count else np.full(3,np.nan)
    row.update({f'{name}_{axis}':float(v) for axis,v in zip('xyz',c)});row[f'{name}_support']=count;row[f'{name}_span_mm']=float(np.linalg.norm(span)*1000);row[f'{name}_center_gap_mm']=float(np.linalg.norm(c-bc)*1000)
    row[f'{name}_quality']=count>=32 and np.linalg.norm(span)<=.25 and np.linalg.norm(c-bc)<=.02
   rows.append(row)
 print('GEOMETRY',split,len(rows),flush=True)
 return pd.DataFrame(rows),np.array(robots),np.array(histories),np.array(poses)

def match(f,r,h,rot):
 pairs=cKDTree(r[:,24:27]).query_pairs(.010001,output_type='ndarray');i,j=pairs.T
 keep=(f.episode.values[i]!=f.episode.values[j])&(np.abs(f.t.values[i]-f.t.values[j])<=10)&(f.phase.values[i]==f.phase.values[j]);i,j=i[keep],j[keep]
 # Groupwise worst over four causal observations; qpos/qvel RMS arm only, gripper separately.
 def rms(s):return np.max(np.sqrt(np.mean((h[i,:,s]-h[j,:,s])**2,axis=-1)),axis=1)
 d=dict(q_rad=rms(slice(0,7)),qvel=rms(slice(9,16)),tcp_mm=np.max(np.linalg.norm(h[i,:,24:27]-h[j,:,24:27],axis=-1),axis=1)*1000,
 opening_mm=np.max(abs(h[i,:,22]-h[j,:,22]),axis=1)*1000,gripvel=np.max(abs(h[i,:,23]-h[j,:,23]),axis=1),linear=rms(slice(31,34)),angular=rms(slice(34,37)),time=abs(f.t.values[i]-f.t.values[j]))
 q=h[:,:,[28,29,30,27]];dot=np.abs(np.sum(q[i]*q[j],axis=-1));d['angle_deg']=np.max(2*np.arccos(np.clip(dot,0,1))*180/np.pi,axis=1)
 d['box_mm']=np.linalg.norm(f[['box_x','box_y','box_z']].values[j]-f[['box_x','box_y','box_z']].values[i],axis=1)*1000
 peg=f[['peg_x','peg_y','peg_z']].values[j]-f[['peg_x','peg_y','peg_z']].values[i];offset=np.linalg.norm(peg,axis=1)*1000
 out=[]
 for level,limits in LEVELS.items():
  ok=np.ones(len(i),bool)
  for k,v in limits.items():
   # Missing/poor box remains a separate uncontrolled stratum, never silently imputed.
   if k!='box_mm':ok &= d[k]<=v
  ix=np.flatnonzero(ok)
  if not len(ix):continue
  x=pd.DataFrame(dict(level=level,i=i[ix],j=j[ix],peg_offset_mm=offset[ix],**{k:v[ix] for k,v in d.items()}));x['score']=np.max(np.stack([d[k][ix]/v for k,v in limits.items() if k!='box_mm']),axis=0)
  x['peg_quality']=f.peg_quality.values[i[ix]]&f.peg_quality.values[j[ix]];x['box_controlled']=f.box_quality.values[i[ix]]&f.box_quality.values[j[ix]]&(d['box_mm'][ix]<=limits['box_mm'])
  x['bin']=pd.cut(x.peg_offset_mm,[-1e-6,5,10,20,40,np.inf],labels=['0-5','5-10','10-20','20-40','40+'],right=False).astype('object').fillna('missing').astype(str)
  x['pair_class']=np.where(~x.peg_quality,'poor_or_missing',np.where(x.peg_offset_mm<5,'control',np.where(x.peg_offset_mm>=20,'different','uncertain')))
  x['selected']=False;x['episode_disjoint']=False;blocks=set();eps=set();epairs=set()
  for idx in x.sort_values(['score','i','j']).index:
   rr=x.loc[idx];a,b=int(rr.i),int(rr.j);ea,eb=int(f.episode[a]),int(f.episode[b]);ba,bb=(ea,int(f.t[a])//10),(eb,int(f.t[b])//10);epair=tuple(sorted((ea,eb)))
   if not rr.peg_quality or not rr.box_controlled:continue
   if ba not in blocks and bb not in blocks and epair not in epairs:x.loc[idx,'selected']=True;blocks.update([ba,bb]);epairs.add(epair)
   if ea not in eps and eb not in eps:x.loc[idx,'episode_disjoint']=True;eps.update([ea,eb])
  for k in ['episode','t','phase','peg_support','box_support','peg_quality','box_quality','peg_span_mm','peg_center_gap_mm']:
   x[k+'_a']=f[k].values[i[ix]];x[k+'_b']=f[k].values[j[ix]]
  x['pair_id']=[f'{f.split.iloc[0]}:{level}:{a}:{b}' for a,b in zip(x.i,x.j)];x['split']=f.split.iloc[0];out.append(x)
 return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def actions(f,ids,split,identities):
 experts=[];preds={v:[] for v in ['E0','E1']};checks=[]
 manifest={(e['model'],e['episode']):e for e in read(ROOT/'outputs/task10c1/feature_manifest.json')['original_expert_unlabelled']}
 checkpoints={e['label']:e for e in read(ROOT/'outputs/task10c1/checkpoint_identity.json')}
 for ep in ids:
  p=R/f'cache/ep{ep}/graphs.npz';a=load(p,['action7','robot76']);experts.extend(a['action7'])
  for v in preds:
   c=checkpoints[v+'_last'];cp=Path(c['path']);assert sha(cp)==c['sha256'];identities[str(cp)]=c['sha256']
   p=R/f'final_offline/{v}/last/{split}/ep{ep}.npz';identities[str(p)]=sha(p);z=load(p);assert np.array_equal(z['target'],a['action7']);preds[v].extend(z['prediction'])
   feat=ROOT/f'outputs/task10c1/features/{v}_last/ep{ep}.npz';m=manifest[v+'_last',ep];assert sha(feat)==m['feature_sha256'] and m['checkpoint_sha256']==c['sha256'] and m['source_cache_sha256']==identities[str(R/f'cache/ep{ep}/graphs.npz')];identities[str(feat)]=m['feature_sha256'];ff=load(feat,['R0','action','timestep']);np.testing.assert_array_equal(ff['R0'],a['robot76']);np.testing.assert_array_equal(ff['timestep'],np.arange(len(a['action7'])));err=float(np.max(abs(ff['action']-z['prediction'])));assert err<3e-5;checks.append(dict(model=v,episode=ep,max_abs=err,tolerance=3e-5))
 return np.array(experts),{k:np.array(v) for k,v in preds.items()},checks

def evaluate(p,f,r,rot,expert,pred):
 ec=controller_commands(expert);pc={k:controller_commands(v) for k,v in pred.items()};i=p.i.values.astype(int);j=p.j.values.astype(int)
 # Each root-frame command rotated through its own base world and TCP inverse into homologous TCP axes.
 base=Rotation.from_quat(r[:,[67,68,69,66]]).as_matrix()
 transform=np.einsum('nji,njk->nik',rot,base)
 def trans(c):return np.einsum('nij,nj->ni',transform,c[:,:3])*1000
 e=trans(ec);delta=e[j]-e[i];mag=np.linalg.norm(delta,axis=1);geom=f[['peg_x','peg_y','peg_z']].values[j]-f[['peg_x','peg_y','peg_z']].values[i];geom*=1000
 er=p[['pair_id','split','level','pair_class','bin','selected','episode_disjoint','episode_a','episode_b','t_a','t_b','phase_a','box_controlled','peg_quality','peg_offset_mm']].copy();er['expert_delta_mm']=mag;er['peg_action_cosine']=cosine(geom,delta);er.loc[mag<1,'peg_action_cosine']=np.nan;er['peg_signed_gain']=np.divide(np.sum(geom*delta,axis=1),np.sum(geom**2,axis=1),out=np.full(len(i),np.nan),where=(np.linalg.norm(geom,axis=1)>=20)&(mag>=1));er['rotation_delta_rad']=np.linalg.norm(ec[j,3:6]-ec[i,3:6],axis=1);er['gripper_target_delta_mm']=abs(ec[j,6]-ec[i,6])*1000;er['small_expert_delta']=mag<1
 for k,axis in enumerate('xyz'):er['expert_d'+axis+'_mm']=delta[:,k]
 rows=[]
 for v,c in pc.items():
  a=trans(c);d=a[j]-a[i];m=np.linalg.norm(d,axis=1);error=np.linalg.norm(d-delta,axis=1);x=er.copy();x['model']=v;x['model_delta_mm']=m;x['delta_error_mm']=error;x['direction_cosine']=cosine(delta,d);x['signed_gain']=np.sum(d*delta,axis=1)/np.maximum(mag**2,1e-30);x['magnitude_ratio']=m/np.maximum(mag,1e-30);x.loc[mag<1,['direction_cosine','signed_gain','magnitude_ratio']]=np.nan;x['zero_improvement_mm']=mag-error;x['better_than_zero']=error<mag;x['frame_action_error_mm']=(np.linalg.norm(a[i]-e[i],axis=1)+np.linalg.norm(a[j]-e[j],axis=1))/2
  for k,axis in enumerate('xyz'):x['model_d'+axis+'_mm']=d[:,k]
  rows.append(x)
 return er,pd.concat(rows,ignore_index=True)

def main():
 global O
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=O);O=parser.parse_args().output.resolve()
 if (O/'run_identity.json').exists():raise FileExistsError('Use a fresh --output directory; completed audits are preserved')
 O.mkdir(parents=True,exist_ok=True);identities={};cfg=read(R/'config.json')
 with ExperimentLogger(team='integration',model='task10c2_learning_signal_audit',config=CONFIG,seed=42,job_type='evaluate',root=O/'experiments') as log:
  put(O/'run_identity.json',dict(uuid=log.run_id,**CONFIG));put(O/'matching_protocol.json',CONFIG)
  frames={};pairs={};data={}
  # All Train matching and coverage is frozen before validation or any actions are opened.
  for split in ['train','validation']:
   f,r,h,rot=geometry(split,cfg['splits'][split],identities);frames[split]=f;data[split]=(r,rot);p=match(f,r,h,rot);pairs[split]=p;p.to_csv(O/f'{split}_frozen_pairs.csv',index=False);print('MATCH',split,len(p),p.groupby('level').selected.sum().to_dict(),flush=True)
   if split=='train':put(O/'train_matching_freeze.json',dict(protocol_sha256=sha(O/'matching_protocol.json'),pairs_sha256=sha(O/'train_frozen_pairs.csv'),actions_accessed=False,validation_accessed=False))
  pp=pd.concat(pairs.values(),ignore_index=True);ff=pd.concat(frames.values(),ignore_index=True);ff.to_csv(O/'frame_geometry_quality.csv',index=False)
  coverage=[];episode=[]
  for split,p in pairs.items():
   f=frames[split]
   for level in LEVELS:
    s=p[p.level==level]
    for scope in ['all','selected','episode_disjoint']:
     z=s if scope=='all' else s[s[scope]]
     for cls in ['all','control','uncertain','different','poor_or_missing']:
      q=z if cls=='all' else z[z.pair_class==cls];indices=np.r_[q.i,q.j].astype(int);cnt=pd.Series(indices).value_counts();coverage.append(dict(split=split,level=level,scope=scope,pair_class=cls,pairs=len(q),episodes=len(set(f.episode.iloc[indices])),frames=len(cnt),frame_coverage=len(cnt)/len(f),max_frame_reuse=int(cnt.max()) if len(cnt) else 0,box_uncontrolled=int((~q.box_controlled).sum()),**{f'bin_{b}':int((q.bin==b).sum()) for b in ['0-5','5-10','10-20','20-40','40+','missing']}))
     for ep in cfg['splits'][split]:
      q=z[(z.episode_a==ep)|(z.episode_b==ep)];ix=np.r_[q.loc[q.episode_a==ep,'i'],q.loc[q.episode_b==ep,'j']].astype(int);episode.append(dict(split=split,level=level,scope=scope,episode=ep,pairs=len(q),unique_frames=len(set(ix)),total_frames=int((f.episode==ep).sum()),max_frame_reuse=int(pd.Series(ix).value_counts().max()) if len(ix) else 0))
  pd.DataFrame(coverage).to_csv(O/'geometry_pair_coverage.csv',index=False);pd.DataFrame(episode).to_csv(O/'episode_coverage.csv',index=False);pp.to_csv(O/'robot_state_matching.csv',index=False)
  ers=[];mrs=[];checks=[]
  codec=ActionCodec(read(R/'action_codec.json'));assert codec.method=='B'
  for split in pairs:
   f=frames[split];r,rot=data[split];e,pr,ch=actions(f,cfg['splits'][split],split,identities);checks.extend(ch);er,mr=evaluate(pairs[split],f,r,rot,e,pr);ers.append(er);mrs.append(mr)
  er=pd.concat(ers);mr=pd.concat(mrs);er.to_csv(O/'expert_conditional_action.csv',index=False);mr.to_csv(O/'model_conditional_action.csv',index=False)
  summaries=[]
  for scope in ['all','selected','episode_disjoint']:
   z=mr if scope=='all' else mr[mr[scope]]
   for keys,g in z.groupby(['split','level','pair_class','model']):
    summaries.append(dict(zip(['split','level','pair_class','model'],keys),scope=scope,n=len(g),episodes=len(set(g.episode_a)|set(g.episode_b)),small_expert=int(g.small_expert_delta.sum()),**{k:float(g[k].mean()) for k in ['expert_delta_mm','model_delta_mm','delta_error_mm','direction_cosine','signed_gain','magnitude_ratio','zero_improvement_mm','better_than_zero','peg_action_cosine','frame_action_error_mm']}))
  pd.DataFrame(summaries).to_csv(O/'conditional_response_comparison.csv',index=False)
  assert not set(cfg['splits']['train'])&set(cfg['splits']['validation'])
  unchanged=all(sha(p)==v for p,v in identities.items());assert unchanged
  put(O/'verification.json',dict(status='PASS',uuid=log.run_id,source_unchanged=unchanged,inputs=identities,prediction_readback=checks,protocol_sha256=sha(O/'matching_protocol.json'),train_freeze=read(O/'train_matching_freeze.json'),test12_opened=False,model_updates=0,model_inference=False,simulator_executed=False,vision_executed=False,backend='CPU NumPy/SciPy and installed ManiSkill controller preprocessing only; no physics/CUDA/MPS',python=platform.python_version(),code_sha256=sha(__file__),coordinate_contract='node XYZ in TCP axes; each command root->world->own TCP before delta; no achieved physical motion claimed'))
  for p in O.iterdir():
   if p.is_file() and not p.name.endswith('.tracking.json'):log.link_output(p,kind='metrics' if p.suffix=='.json' else 'reports')
  log.log(dict(train_candidate_pairs=len(pairs['train']),validation_candidate_pairs=len(pairs['validation']),model_updates=0));print('DONE',log.run_id,flush=True)
if __name__=='__main__':main()

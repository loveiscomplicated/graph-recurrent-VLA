"""Task10-I stage1 only. CPU PhysX / native CUDA renderer; never executes actions."""
import os
os.environ.update(WANDB_MODE='disabled',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys,json,hashlib,time,subprocess,shutil,traceback,csv,fcntl,pickle,random,inspect,platform,ast
from pathlib import Path
BASE=Path(os.environ.get('TASK10I_BASE','/workspace/task10i_stage1_20261031'))
OLD=Path('/workspace/task10g_f1_abc_20261031')
SRC=BASE/'source';AUDIT=BASE/'audit'
sys.path[:0]=[str(SRC),str(AUDIT)]
import numpy as np

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(p)
    fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,allow_nan=False).encode()).hexdigest()
def arr(x):return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()
def tree(x):
    if isinstance(x,dict):return {k:tree(v) for k,v in x.items()}
    return arr(x)
def flat(x,prefix=''):
    out={}
    if isinstance(x,dict):
        for k,v in x.items():out.update(flat(v,prefix+'/'+str(k)))
    else:out[prefix]=np.asarray(x)
    return out
def ahash(x):
    h=hashlib.sha256()
    for k,v in sorted(flat(x).items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()
def npz(p,**kw):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**kw);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def cmd(args,log):
    with Path(log).open('a') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
def init_env():
    from graph.geometry_reach10d.runtime import environment
    from graph.integration_baseline.direct_train import seed_all
    seed_all(42,1);return environment(OLD/'full')
def camera(d,cid='candidate'):
    from graph.persistent_geometry.contracts import CameraObservation
    return CameraObservation(cid,0,'base_camera',0.,d['rgb'],d['depth_raw_mm'],d['intrinsic'],d['extrinsic_cv'],d['cam2world_gl'],d['tcp_pose'])
def capture(env,seed):
    start=time.perf_counter();obs,_=env.reset(seed=int(seed));t=env.unwrapped
    reset_s=time.perf_counter()-start;start=time.perf_counter()
    sensor=obs['sensor_param']['base_camera'];sd=obs['sensor_data']['base_camera']
    d=dict(rgb=arr(sd['rgb'])[0],depth_raw_mm=arr(sd['depth'])[0,...,0],
        intrinsic=arr(sensor['intrinsic_cv'])[0],extrinsic_cv=arr(sensor['extrinsic_cv'])[0],cam2world_gl=arr(sensor['cam2world_gl'])[0],
        peg_pose=arr(t.peg.pose.raw_pose)[0],peg_half=arr(t.peg_half_sizes)[0],box_pose=arr(t.box.pose.raw_pose)[0],
        hole_pose=arr(t.box_hole_pose.raw_pose)[0],hole_radius=arr(t.box_hole_radii)[0],hole_offset=arr(t.box_hole_offsets.raw_pose)[0],
        tcp_pose=arr(t.agent.tcp.pose.raw_pose)[0],qpos=arr(t.agent.robot.get_qpos())[0],qvel=arr(t.agent.robot.get_qvel())[0],
        elapsed_steps=arr(t._elapsed_steps),qlimits=arr(t.agent.robot.get_qlimits())[0])
    d['segmentation']=arr(t._sensors['base_camera'].get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'])[0,...,0]
    ids={int(k):v.name for k,v in t.segmentation_id_map.items()};pegid=next(k for k,n in ids.items() if n.startswith('peg'))
    d['peg_mask']=d['segmentation']==pegid
    snap=dict(sim_state=tree(t.get_state_dict()),controller=tree(t.agent.get_controller_state()),
        main_rng=t._main_rng.get_state(),episode_rng=t._episode_rng.get_state(),
        batched_main_rng=[rng.get_state() for rng in t._batched_main_rng.rngs],batched_episode_rng=[rng.get_state() for rng in t._batched_episode_rng.rngs],
        main_seed=t._main_seed,episode_seed=t._episode_seed.copy(),torch_rng=__import__('torch').get_rng_state(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),elapsed_steps=d['elapsed_steps'],seed=int(seed))
    return d,snap,dict(reset_including_render_s=reset_s,readback_segmentation_s=time.perf_counter()-start)

def overlap(a,b):
    """Exact convex SAT; minimum separating translation in metres; zero if disjoint."""
    from scipy.spatial import ConvexHull
    if np.any(np.minimum(a.max(0),b.max(0))-np.maximum(a.min(0),b.min(0))<=0):return 0.
    ha,hb=ConvexHull(a),ConvexHull(b)
    def edges(v,h):
        pairs={tuple(sorted((int(s[i]),int(s[j])))) for s in h.simplices for i,j in ((0,1),(1,2),(2,0))}
        x=np.array([v[j]-v[i] for i,j in pairs]);return x/np.linalg.norm(x,axis=1)[:,None]
    ea,eb=edges(a,ha),edges(b,hb);cross=np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3)
    axes=np.concatenate([ha.equations[:,:3],hb.equations[:,:3],cross]);norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-9]/norm[norm>1e-9,None]
    pa=a@axes.T;pb=b@axes.T
    return max(0.,float(np.minimum(pa.max(0)-pb.min(0),pb.max(0)-pa.min(0)).min()))
def actor_shapes(actor):
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    result=[]
    for entity in actor._objs:
        T=entity.pose.to_transformation_matrix()
        for c in entity.components:
            if isinstance(c,physx.PhysxRigidBaseComponent):
                for mesh in get_component_meshes(c):
                    if mesh is not None:result.append(np.asarray(mesh.vertices)@T[:3,:3].T+T[:3,3])
    return result

def physical(env,d,snap):
    from graph.persistent_geometry.self_filter import robot_surfaces
    from visual.runtime.vision_packet import pose_matrix
    from geometry import SIGNS
    checks={k+'_finite':bool(np.isfinite(d[k]).all()) for k in ['peg_pose','box_pose','hole_pose','peg_half','qpos','qvel','tcp_pose','hole_radius']}
    checks['positive_shapes']=bool((d['peg_half']>0).all() and d['hole_radius']>0)
    checks['joint_limits']=bool((d['qpos']>=d['qlimits'][:,0]-1e-7).all() and (d['qpos']<=d['qlimits'][:,1]+1e-7).all())
    # The installed reset distribution is recorded, not a new proposal/workspace restriction.
    checks['native_reset_support']=bool(-.100001<=d['peg_pose'][0]<=.100001 and -.300001<=d['peg_pose'][1]<=.000001 and -.050001<=d['box_pose'][0]<=.050001 and .199999<=d['box_pose'][1]<=.400001)
    checks['clock_zero']=bool((d['elapsed_steps']==0).all())
    checks['sim_controller_finite']=all(np.isfinite(x).all() for x in flat({'sim':snap['sim_state'],'controller':snap['controller']}).values())
    T=pose_matrix(d['peg_pose']);pv=(SIGNS*d['peg_half'])@T[:3,:3].T+T[:3,3];bv=actor_shapes(env.unwrapped.box)
    checks['table_nonpenetration']=bool(pv[:,2].min()>=-1e-5 and all(v[:,2].min()>=-1e-5 for v in bv))
    pairs=[('peg_box',pv,b) for b in bv]
    for s in robot_surfaces(env):
        pairs.append(('robot_peg:'+s.link,s.world_vertices,pv))
        pairs.extend(('robot_box:'+s.link,s.world_vertices,b) for b in bv)
    contacts=[dict(pair=n,minimum_translation_m=overlap(a,b)) for n,a,b in pairs if np.all(np.minimum(a.max(0),b.max(0))-np.maximum(a.min(0),b.min(0))>0)]
    checks['no_severe_initial_interpenetration']=all(c['minimum_translation_m']<=.002 for c in contacts)
    camera(d).validate();checks['camera_calibration']=True
    checks['depth_rendering']=bool((d['depth_raw_mm']>0).any() and d['rgb'].shape==(128,128,3))
    return dict(status='PHYSICAL_VALID' if all(checks.values()) else 'PHYSICAL_INVALID',checks=checks,collision_evidence=contacts,
        contact_policy='No physics step. Per actual collision convex shape SAT; >2mm minimum translation is severe. Peg-table resting contact allowed. Table numerical allowance 10um. No expert success gate.')

def compare(a,b):
    return {k:dict(exact=bool(np.array_equal(a[k],b[k])),max_abs=float(np.max(np.abs(a[k].astype(float)-b[k].astype(float))))) for k in a}
def descriptor(d,r):
    from scipy.spatial.transform import Rotation as R
    def rot(p):return R.from_quat(np.r_[p[4:7],p[3]])
    pp,tp,hp=d['peg_pose'],d['tcp_pose'],d['hole_pose'];uv=np.asarray(r['corners_uv']);lo=uv.min(0);hi=uv.max(0)
    from visual.runtime.vision_packet import pose_matrix
    from geometry import projection
    T=pose_matrix(pp);axis=np.array([[-d['peg_half'][0],0,0],[d['peg_half'][0],0,0]])@T[:3,:3].T+T[:3,3]
    _,axis_uv,_=projection(camera(d),axis)
    out=dict(projected_longitudinal_axis_uv=axis_uv,peg_xyz=pp[:3],peg_quat_wxyz=pp[3:],peg_yaw_rad=rot(pp).as_euler('xyz')[2],peg_half=d['peg_half'],
        box_pose=d['box_pose'],hole_pose=hp,hole_radius=d['hole_radius'],hole_local_offset=d['hole_offset'],
        peg_minus_tcp_world=pp[:3]-tp[:3],peg_in_tcp=rot(tp).inv().apply(pp[:3]-tp[:3]),
        peg_orientation_in_tcp_xyzw=(rot(tp).inv()*rot(pp)).as_quat(),peg_minus_hole_world=pp[:3]-hp[:3],
        peg_orientation_in_hole_xyzw=(rot(hp).inv()*rot(pp)).as_quat(),peg_hole_angle_rad=(rot(hp).inv()*rot(pp)).magnitude(),
        peg_length_m=2*d['peg_half'][0],peg_thickness_m=2*d['peg_half'][1],box_outer_half=d['peg_half'][0],
        tcp_peg_distance_m=np.linalg.norm(pp[:3]-tp[:3]),tcp_hole_distance_m=np.linalg.norm(hp[:3]-tp[:3]),
        gripper_opening_m=d['qpos'][-2:].sum(),image_centroid_uv=uv.mean(0),bbox_uvxy=np.r_[lo,hi],pixel_extent=hi-lo,
        boundary_margin_px=min(lo[0],lo[1],128-hi[0],128-hi[1]))
    return clean(out)

def regress(root):
    from geometry import evaluate
    eps=read(AUDIT/'task10g_initial_visibility_inputs_v1/episodes.json');old=read(AUDIT/'results_v1/episodes.json')
    inp=AUDIT/'task10g_initial_visibility_inputs_v1';proof=read(inp/'extraction_verification.json')
    for k,v in proof['files'].items():assert sha(inp/k)==v,k
    import geometry
    for n,f in [('compute_grasp_info_by_obb',geometry.compute_grasp_info_by_obb),('build_grasp_pose',geometry.Panda.build_grasp_pose)]:assert inspect.getsource(f)==(inp/(n+'.py')).read_text()
    data=np.load(inp/'first_observations.npz');rows=[];env=init_env();render=[]
    try:
        seen=set()
        for i,ep in enumerate(eps):
            d={k:data[k][i] for k in ['rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl','tcp_pose','peg_pose','peg_half']}
            r,_=evaluate(camera(d,str(ep['episode_id'])),d['peg_pose'],d['peg_half'],d['tcp_pose'])
            assert r['requested_pass']==old[i]['requested_pass'] and ep['episode_id']==old[i]['episode_id']
            rows.append(dict(episode_id=ep['episode_id'],parent_seed=ep['parent_seed'],split=ep['split'],passed=r['requested_pass']))
            if ep['parent_seed'] not in seen:
                seen.add(ep['parent_seed']);actual,_,_=capture(env,ep['parent_seed']);checks={k:bool(np.array_equal(d[k],actual[k])) for k in d};assert all(checks.values()),(ep['parent_seed'],checks)
                render.append(dict(seed=ep['parent_seed'],checks=checks))
        counts={}
        for split,n,fail,np_,pf in [('train',272,190,96,63),('validation',48,25,16,9)]:
            rr=[r for r in rows if r['split']==split];parents={r['parent_seed']:r['passed'] for r in rr}
            assert len(rr)==n and sum(not r['passed'] for r in rr)==fail and len(parents)==np_ and sum(not p for p in parents.values())==pf
            ids=read(AUDIT/'results_v1/eligibility_ids.json')[split]['primary_geometry_depth_pass'];assert sorted(ids)==sorted(r['episode_id'] for r in rr if r['passed'])
            counts[split]=dict(logical=n,logical_fail=fail,parents=np_,parent_fail=pf)
        write(root/'visibility_regression.json',dict(status='PASS',exact_labels=320,exact_rendered_parents=len(render),counts=counts,rows=rows,render=render))
    finally:env.close()

def inventory(root):
    records=[];allints=set();seed_entries=[];skipped=[]
    def walk(x,path=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path+'/'+str(k))
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,path+'/'+str(i))
        elif isinstance(x,int) and not isinstance(x,bool):
            if 0<=x<2**31:allints.add(x)
            if 'seed' in path.lower():seed_entries.append((str(p),path,x))
    # All existing research JSON manifests, including retired runtime roots. No raw images/model loads.
    for p in Path('/workspace').rglob('*.json'):
        if BASE in p.parents or any(s in p.parts for s in ['.git','.hf_home','node_modules','site-packages']):continue
        if p.stat().st_size>30_000_000:skipped.append(str(p));continue
        try:x=read(p)
        except (ValueError,OSError):continue
        before=len(seed_entries);walk(x);records.append(dict(path=str(p),sha256=sha(p),seed_entries=len(seed_entries)-before))
    # Integer literals and literal range intervals in existing research Python files are conservative exclusions too.
    ranges=[]
    for base in [OLD/'source',Path('/workspace/graph-recurrent-VLA')]:
        for p in base.rglob('*.py'):
            try:node=ast.parse(p.read_text())
            except Exception:continue
            for v in ast.walk(node):
                if isinstance(v,ast.Constant) and isinstance(v.value,int) and 0<=v.value<2**31:allints.add(v.value)
                if isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='range' and 1<=len(v.args)<=3:
                    try:
                        args=[ast.literal_eval(a) for a in v.args]
                        if all(isinstance(x,int) for x in args):
                            rr=range(*args)
                            if rr:ranges.append([min(rr[0],rr[-1]),max(rr[0],rr[-1])])
                    except Exception:pass
    # Existing raw episode HDF5 attributes are represented by audited parent episodes and release manifests.
    for e in read(AUDIT/'task10g_initial_visibility_inputs_v1/episodes.json'):allints.add(e['parent_seed'])
    assert not skipped,skipped
    inv=dict(status='PASS',scope='All parseable /workspace JSON manifests + source integer literals/ranges; conservative integer exclusion, not an assertion all integers are reset seeds',files=records,
        seed_entries=seed_entries,conservative_integers=sorted(allints),literal_range_exclusions=ranges,oversized_skipped=skipped)
    write(root/'existing_parent_inventory.json',inv)
    rng=np.random.default_rng(10103110);seeds=[]
    while len(seeds)<1200:
        s=int(rng.integers(1_200_000_000,1_900_000_000))
        if s not in allints and s not in seeds and not any(a<=s<=b for a,b in ranges):seeds.append(s)
    write(root/'candidate_seed_manifest.json',dict(namespace='10i-stage1-random-candidates-v1',rng='NumPy PCG64(10103110)',inventory_sha256=sha(root/'existing_parent_inventory.json'),
        ordered_attempts=[dict(candidate_id=f'10i_s1_{i:04d}',attempt_index=i,reset_seed=s) for i,s in enumerate(seeds)],target=300,max_attempts=1200,
        stopping_rule='First 300 visibility + physical + replay + exact unique; or 1200 attempts. Never adapt proposal or seed order.',split='UNASSIGNED',blind_test=False))

def collect(root,limit=None,smoke=False):
    from geometry import evaluate,projection
    from visual.runtime.vision_packet import pose_matrix
    manifest=read(root/'candidate_seed_manifest.json');env=init_env();accepted=set();n=0
    try:
        for entry in manifest['ordered_attempts']:
            cid=entry['candidate_id'];dest=root/'candidates'/cid;commit=dest/'record.json'
            if commit.exists():
                r=read(commit)
                for k,h in r['files'].items():assert sha(dest/k)==h
                if r['eligible']:accepted.add(r['physical_state_sha256']);n+=1
                if n>= (3 if smoke else 300):break
                continue
            if limit is not None and entry['attempt_index']>=limit:break
            dest.mkdir(parents=True,exist_ok=True)
            d,snap,timing=capture(env,entry['reset_seed']);start=time.perf_counter();r,ev=evaluate(camera(d,cid),d['peg_pose'],d['peg_half'],d['tcp_pose']);timing['gate_s']=time.perf_counter()-start
            start=time.perf_counter();p=physical(env,d,snap);timing['physical_s']=time.perf_counter()-start
            desc=descriptor(d,r);ph=ahash({k:d[k] for k in ['peg_pose','peg_half','box_pose','hole_pose','hole_radius','qpos','qvel','tcp_pose']})
            # Every attempt gets a snapshot, evidence and readback; replay all visibility/physical passes.
            snapfile=dest/'snapshot.pkl'
            with snapfile.with_suffix('.tmp').open('wb') as f:pickle.dump(snap,f,protocol=5);f.flush();os.fsync(f.fileno())
            snapfile.with_suffix('.tmp').replace(snapfile)
            npz(dest/'initial.npz',**d);npz(dest/'visibility_evidence.npz',**ev)
            if smoke and os.getenv('TASK10I_CRASH')==str(entry['attempt_index']):os._exit(91)
            write(dest/'visibility.json',r);write(dest/'physical.json',p);write(dest/'descriptor.json',desc)
            replay=dict(status='NOT_REQUIRED',route='reset(seed) reconstructs shape; state dict + controller + RNG sidecar retained')
            eligible=False
            if r['requested_pass'] and p['status']=='PHYSICAL_VALID':
                # Reset from an unrelated state, then reconstruct native shape using same seed.
                env.reset(seed=42424242)
                dd,ss,_=capture(env,entry['reset_seed']);rr,_=evaluate(camera(dd,cid),dd['peg_pose'],dd['peg_half'],dd['tcp_pose'])
                c=compare(d,dd);sim=ahash(snap['sim_state'])==ahash(ss['sim_state']);ctrl=ahash(snap['controller'])==ahash(ss['controller'])
                ok=all(v['exact'] for v in c.values()) and sim and ctrl and rr['requested_pass']==r['requested_pass']
                replay=dict(status='REPLAY_PASS' if ok else 'REPLAY_FAIL',checks=c,sim_state_exact=sim,controller_exact=ctrl,visibility_exact=rr['requested_pass']==r['requested_pass'],route='independent reset(seed); native shape rebuild; exact saved-state readback',tolerance_used=False)
                eligible=ok and ph not in accepted
                if eligible:accepted.add(ph);n+=1
            write(dest/'replay.json',replay)
            # Freeze a per-candidate transaction only after every file has been written and read back.
            files={f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='record.json' and not f.name.endswith('.tmp')}
            for f,h in files.items():assert sha(dest/f)==h
            with np.load(dest/'initial.npz') as z:assert ahash({k:z[k] for k in z.files})==ahash(d)
            rec=dict(**entry,visibility='VISIBILITY_PASS' if r['requested_pass'] else 'VISIBILITY_FAIL',physical=p['status'],replay=replay['status'],eligible=eligible,
                exact_duplicate=ph in accepted and not eligible and replay['status']=='REPLAY_PASS',physical_state_sha256=ph,observation_sha256=ahash({k:d[k] for k in ['rgb','depth_raw_mm','segmentation','intrinsic','extrinsic_cv','cam2world_gl']}),
                reasons=r['reasons'],expert_success='NOT_TESTED',split='UNASSIGNED',files=files,timing=timing)
            write(commit,rec);write(root/'cursor.json',dict(next_attempt=entry['attempt_index']+1,eligible=n,status='RUNNING'))
            if entry['attempt_index']%20==0:print('ATTEMPT',entry['attempt_index']+1,'ELIGIBLE',n,flush=True)
            if n>=(3 if smoke else 300):break
    finally:env.close()

def replay_all(root):
    from geometry import evaluate
    env=init_env();rows=[]
    try:
        for f in sorted((root/'candidates').glob('*/record.json')):
            r=read(f)
            if not r['eligible']:continue
            d=dict(np.load(f.parent/'initial.npz'));dd,ss,_=capture(env,r['reset_seed']);checks=compare(d,dd)
            with (f.parent/'snapshot.pkl').open('rb') as h:snap=pickle.load(h)
            v,_=evaluate(camera(dd,r['candidate_id']),dd['peg_pose'],dd['peg_half'],dd['tcp_pose'])
            row=dict(candidate_id=r['candidate_id'],checks=checks,sim_state_exact=ahash(snap['sim_state'])==ahash(ss['sim_state']),controller_exact=ahash(snap['controller'])==ahash(ss['controller']),visibility=v['requested_pass'])
            row['passed']=all(v['exact'] for v in checks.values()) and row['sim_state_exact'] and row['controller_exact'] and row['visibility'];rows.append(row)
            assert row['passed'],row
            if len(rows)%40==0:print('INDEPENDENT_REPLAY',len(rows),flush=True)
        write(root/'candidate_replay_verification.json',dict(status='PASS',count=len(rows),all_fields_byte_exact=True,rows=rows,
            internal_state_limit='Registered actors/articulation plus controller readback exact. PhysX hidden caches/solver internals are not serialized; seed reconstruction is validated at t=0, not future trajectory identity.'))
    finally:env.close()

def rows(root):return [read(f) for f in sorted((root/'candidates').glob('*/record.json'))]
def csvwrite(p,rr):
    if not rr:p.write_text('candidate_id\n');return
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rr:w.writerow({k:json.dumps(clean(v)) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()})

def finalize(root):
    from collections import Counter
    rr=rows(root);eligible=[r for r in rr if r['eligible']];states=[];descs=[];metrics=[];export=[];snapshots=[]
    seen={};dups=[]
    for r in rr:
        cid=r['candidate_id'];dest=root/'candidates'/cid;d=dict(np.load(dest/'initial.npz'));v=read(dest/'visibility.json');desc=read(dest/'descriptor.json')
        if r['physical_state_sha256'] in seen:dups.append([seen[r['physical_state_sha256']],cid])
        else:seen[r['physical_state_sha256']]=cid
        states.append(d);descs.append(dict(candidate_id=cid,eligible=r['eligible'],**desc))
        expected=v['expected_in_frame_pixels'];gt=int(d['peg_mask'].sum())
        metrics.append(dict(candidate_id=cid,visibility=r['visibility'],boundary_margin_px=desc['boundary_margin_px'],bbox=desc['bbox_uvxy'],centroid=desc['image_centroid_uv'],
            expected_in_frame_surface_pixels=expected,gt_peg_pixels=gt,gt_peg_image_fraction=gt/16384,
            depth_supported_expected_surface_fraction=v['supported_fraction_in_frame'],required_regions=v['regions'],
            strict_depth=v['strict_100_pass'],strict_gt_required=all(d['peg_mask'][np.load(dest/'visibility_evidence.npz')[k]].all() for k in ['white_end','red_end','grasp']) and v['requested_pass']))
        snapshots.append(dict(candidate_id=cid,path=f'candidates/{cid}/snapshot.pkl',sha256=sha(dest/'snapshot.pkl'),initial=f'candidates/{cid}/initial.npz',initial_sha256=sha(dest/'initial.npz')))
        if r['eligible']:export.append(dict(candidate_id=cid,reset_seed=r['reset_seed'],split='UNASSIGNED',initial_npz=f'candidates/{cid}/initial.npz',snapshot=f'candidates/{cid}/snapshot.pkl',
            rgb=f'gallery/{cid}.png',gt_visibility_evidence=f'candidates/{cid}/visibility_evidence.npz',physical_state_sha256=r['physical_state_sha256'],descriptor=desc,
            source_sha256=sha(root/'source_identity.json'),runtime_sha256=sha(root/'runtime_identity.json'),expert_success='NOT_TESTED'))
    csvwrite(root/'candidate_attempts.csv',rr);csvwrite(root/'candidate_eligibility.csv',[{k:r[k] for k in ['candidate_id','reset_seed','attempt_index','visibility','physical','replay','eligible','exact_duplicate','expert_success','split']} for r in rr])
    csvwrite(root/'candidate_state_descriptors.csv',descs);csvwrite(root/'candidate_visibility_metrics.csv',metrics)
    npz(root/'candidate_initial_states.npz',candidate_id=np.array([r['candidate_id'] for r in rr]),**{k:np.stack([d[k] for d in states]) for k in states[0]})
    write(root/'candidate_snapshots_manifest.json',snapshots)
    # Near-duplicates are descriptive only, no removal: <=1mm peg+box/TCP translations, <=1deg rotations, <=0.1mm half-size.
    from scipy.spatial.transform import Rotation as R
    near=[]
    for i,a in enumerate(states):
        for j in range(i):
            b=states[j]
            if any(np.linalg.norm(a[k][:3]-b[k][:3])>.001 for k in ['peg_pose','box_pose','tcp_pose']):continue
            if np.max(abs(a['peg_half']-b['peg_half']))>.0001:continue
            angles=[]
            for k in ['peg_pose','box_pose','tcp_pose']:
                qa,qb=a[k],b[k];angles.append((R.from_quat(np.r_[qa[4:],qa[3]]).inv()*R.from_quat(np.r_[qb[4:],qb[3]])).magnitude())
            if max(angles)<=np.pi/180:near.append([rr[j]['candidate_id'],rr[i]['candidate_id']])
    seeds=[r['reset_seed'] for r in rr];inv=read(root/'existing_parent_inventory.json');overlap_seeds=set(seeds)&set(inv['conservative_integers'])
    assert len(seeds)==len(set(seeds)) and not overlap_seeds
    write(root/'duplicate_audit.json',dict(status='PASS',exact_pairs=dups,near_pairs=near,seed_overlap=sorted(overlap_seeds),candidate_id_unique=len({r['candidate_id'] for r in rr})==len(rr),near_rule='translation peg/box/TCP <=1mm, rotation <=1degree, half-size <=0.1mm; diagnostic only'))
    write(root/'eligible_candidate_pool.json',export)
    pool=dict(status='COMPLETED' if len(eligible)>=300 else 'INSUFFICIENT_POOL',attempted=len(rr),visibility_pass=sum(r['visibility']=='VISIBILITY_PASS' for r in rr),physical_valid=sum(r['physical']=='PHYSICAL_VALID' for r in rr),
        eligible_unique=len(eligible),remaining_to_300=max(0,300-len(eligible)),failure_reasons=dict(Counter(x for r in rr for x in r['reasons'])),replay_fail=sum(r['replay']=='REPLAY_FAIL' for r in rr),exact_duplicates=len(dups),near_duplicates=len(near),
        visibility_rate=sum(r['visibility']=='VISIBILITY_PASS' for r in rr)/len(rr),run_id=read(root/'run_identity.json')['run_id'],split='UNASSIGNED',expert_success='NOT_TESTED',stage2_started=False)
    write(root/'candidate_pool_manifest.json',pool)
    make_gallery(root,rr,states,descs,pool)
    report(root,rr,descs,pool)

def make_gallery(root,rr,states,descs,pool):
    from PIL import Image,ImageDraw
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g=root/'gallery';g.mkdir(exist_ok=True);cards=[];examples={}
    for r,d,desc in zip(rr,states,descs):
        cid=r['candidate_id'];v=read(root/'candidates'/cid/'visibility.json');im=Image.fromarray(d['rgb']).resize((384,384));draw=ImageDraw.Draw(im)
        uv=np.asarray(v['corners_uv'],dtype=float)*3
        from geometry import SIGNS,projection
        for i in range(8):
            for j in range(i):
                if sum(SIGNS[i]!=SIGNS[j])==1 and np.isfinite(uv[[i,j]]).all():draw.line([tuple(uv[i]),tuple(uv[j])],fill='yellow',width=2)
        guv=np.asarray(v['grasp_corners_uv'],dtype=float)*3
        for p in guv:
            if np.isfinite(p).all():draw.ellipse([p[0]-2,p[1]-2,p[0]+2,p[1]+2],fill='cyan')
        # Red/white axis endpoints from the real peg transform.
        from visual.runtime.vision_packet import pose_matrix
        T=pose_matrix(d['peg_pose']);pts=np.array([[-d['peg_half'][0],0,0],[d['peg_half'][0],0,0]])@T[:3,:3].T+T[:3,3]
        _,axis,_=projection(camera(d),pts)
        for p,c in zip(axis*3,['white','red']):
            if np.isfinite(p).all():draw.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4],fill=c)
        draw.rectangle([0,0,383,383],outline='lime' if r['eligible'] else 'red',width=3);draw.rectangle([0,0,383,22],fill='black');draw.text((4,4),cid+' '+r['visibility'],fill='white');im.save(g/(cid+'.png'))
        category='PASS' if r['eligible'] else ('CROP_FAIL' if any('out_of_frame' in s for s in r['reasons']) else 'DEPTH_FAIL')
        examples.setdefault(category,cid)
        for key,condition in [('GRASP_BAND_FAIL',any('grasp' in s for s in r['reasons'])),('PHYSICAL_INVALID',r['physical']=='PHYSICAL_INVALID'),('REPRODUCTION_FAIL',r['replay']=='REPLAY_FAIL')]:
            if condition:examples.setdefault(key,cid)
        cards.append(f'<article id="{cid}"><h3>{cid} / seed {r["reset_seed"]}</h3><img loading="lazy" src="gallery/{cid}.png"><pre>{html.escape(json.dumps(dict(verdict=r["visibility"],reasons=r["reasons"],physical=r["physical"],replay=r["replay"],extent=desc["pixel_extent"],margin=desc["boundary_margin_px"],peg=d["peg_pose"].tolist(),half=d["peg_half"].tolist()),indent=2))}</pre><a href="candidates/{cid}/record.json">record</a></article>')
    write(root/'example_index.json',{k:examples.get(k,'NOT_OBSERVED') for k in ['PASS','CROP_FAIL','GRASP_BAND_FAIL','DEPTH_FAIL','PHYSICAL_INVALID','REPRODUCTION_FAIL']})
    (root/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I all attempts</title><style>body{font:14px sans-serif;background:#eee}main{display:flex;flex-wrap:wrap}article{background:white;margin:8px;padding:12px;width:390px}pre{white-space:pre-wrap}img{width:384px}</style><h1>Task10-I Stage1 — all attempts</h1><p>Yellow OBB, white/red endpoints, cyan grasp band. No final split selection.</p><main>'+''.join(cards)+'</main>')
    specs=[('Peg x','peg_xyz',0),('Peg y','peg_xyz',1),('Peg z','peg_xyz',2),('Peg yaw rad','peg_yaw_rad',None),('Length m','peg_length_m',None),('Thickness m','peg_thickness_m',None),('Peg-hole x','peg_minus_hole_world',0),('Peg-hole y','peg_minus_hole_world',1),('Peg-hole z','peg_minus_hole_world',2),('TCP-peg distance','tcp_peg_distance_m',None),('Image center u','image_centroid_uv',0),('Boundary margin px','boundary_margin_px',None)]
    fig,axs=plt.subplots(4,3,figsize=(15,15))
    for ax,(label,key,index) in zip(axs.flat,specs):
        for flag,color in [(False,'tomato'),(True,'seagreen')]:
            vals=[d[key] if index is None else d[key][index] for d,r in zip(descs,rr) if (r['visibility']=='VISIBILITY_PASS')==flag]
            vals=[v for v in vals if v is not None and np.isfinite(v)]
            ax.hist(vals,bins=24,alpha=.55,label='PASS' if flag else 'FAIL',color=color)
        ax.set_title(label);ax.legend()
    fig.tight_layout();fig.savefig(root/'distributions.png');plt.close(fig)
    fig,ax=plt.subplots();ax.plot(np.arange(1,len(rr)+1),np.cumsum([r['visibility']=='VISIBILITY_PASS' for r in rr])/np.arange(1,len(rr)+1));ax.set(xlabel='Attempt',ylabel='Cumulative visibility pass rate',ylim=(0,1));fig.savefig(root/'pass_rate.png');plt.close(fig)

def report(root,rr,descs,pool):
    import html
    passed=[d for r,d in zip(rr,descs) if r['eligible']]
    ranges={}
    for k in ['peg_xyz','peg_yaw_rad','peg_length_m','peg_thickness_m','peg_minus_hole_world','tcp_peg_distance_m','boundary_margin_px']:
        a=np.array([d[k] for d in passed]);ranges[k]=dict(min=a.min(0).tolist(),max=a.max(0).tolist()) if len(a) else None
    write(root/'distribution_ranges.json',ranges)
    timing={k:dict(mean=float(np.mean([r['timing'][k] for r in rr])),p95=float(np.percentile([r['timing'][k] for r in rr],95))) for k in rr[0]['timing']}
    write(root/'timing_summary.json',timing)
    text=f'''# Task10-I / Stage 1 — Main-visible Initial State Candidate Pool

상태: **{pool['status']}**. ExperimentLogger UUID: `{pool['run_id']}`.
실행 Root: `{root}`. 이 결과는 최종 학습 데이터셋이 아닌 초기 상태 후보 풀이다.

## Q1. 기존 감사 재현

동일 geometry.py/protocol.json으로 320/320 Primary ID가 일치했다. Train FAIL 190/272, Validation FAIL 25/48; 고유 Parent FAIL 63/96, 9/16. 기존 112 Parent를 실제 CPU PhysX + CUDA Renderer로 다시 Reset하여 RGB/Depth/Camera/Peg/TCP exact 일치를 검사했다. [회귀 검증](visibility_regression.json), [Source](source_identity.json), [Protocol](visibility_protocol_identity.json).

## Q2–Q4. 신규 시도·고유 통과·실패 이유

신규 Random Reset {pool['attempted']}회, Visibility PASS {pool['visibility_pass']}개 ({pool['visibility_rate']:.2%}), 최종 유효 고유 Parent **{pool['eligible_unique']}개**. 부족량 {pool['remaining_to_300']}개. 최대 1,200개 사전 동결 Seed를 순서대로 사용하며 300개 도달 시 중단한다. 성능·공간 Descriptor로 Seed/배치를 수정하지 않았다.

실패 사유(중복 집계): `{json.dumps(pool['failure_reasons'],ensure_ascii=False)}`.

## Q5. 물리 유효성

Physical valid {pool['physical_valid']}/{pool['attempted']}. Finite pose/qpos/qvel/shape, joint limit, native reset workspace support, zero clock, calibration/depth/controller, actual collision mesh convex SAT 검사를 적용했다. 심각한 침투는 최소 분리 이동량 >2mm로 수집 전 고정했다. Table 정상 접촉은 허용하며 10µm 수치 오차만 허용한다. Reset 직후 PhysX contact 목록은 비어 있으므로 이를 무충돌 증거로 쓰지 않았다. Robot self-collision의 exhaustive 동역학 검증이나 Expert 성공을 주장하지 않는다. **EXPERT_SUCCESS=NOT_TESTED**, action 실행 0회.

## Q6. Replay

모든 Eligible은 다른 Seed로 Reset한 뒤 원래 Seed로 재초기화하여 모든 저장 geometry/robot/camera/RGB/depth/segmentation과 등록 simulator state, 초기 controller를 exact 비교한다. 별도 프로세스에서 전체 Eligible을 다시 재생하고 Visibility도 비교했다. [독립 Replay](candidate_replay_verification.json).

Snapshot에는 등록 Actor/Articulation state와 Controller, RNG, Clock을 저장했다. 지원 API의 get_state_dict에는 shape/randomized task assets와 PhysX hidden solver cache가 포함되지 않으므로 Snapshot 단독 완전 복원을 주장하지 않는다. 검증한 후속 경로는 **동일 Runtime/Source에서 reset(seed)로 shape를 재구성하고 initial.npz 및 snapshot을 readback 대조**하는 경로다. 미래 trajectory의 bitwise identity는 검증하지 않았다. 상태 미세 차이를 허용하는 tolerance는 Replay에 사용하지 않았다.

## Q7. Seed/중복

기존 /workspace 연구 JSON manifest, 기존 Source literal/range 및 감사 Parent를 조사하여 보수적으로 기존 정수·Seed namespace와 분리했다. 기존 Train/Validation/Final/10H/10D 파일의 실제 경로와 SHA는 [Inventory](existing_parent_inventory.json)에 있다. 후보 Seed·ID 중복 없음; 기존 조사 Seed 충돌 없음. Exact duplicate {pool['exact_duplicates']}, near duplicate {pool['near_duplicates']}. Near duplicate는 1mm translation, 1degree rotation, 0.1mm half-size 기준의 진단이며 제거하지 않았다. [동결 Seed](candidate_seed_manifest.json), [중복 감사](duplicate_audit.json).

## Q8. 상태 범위

Eligible 범위 (m, rad; world frame):

```json
{json.dumps(ranges,ensure_ascii=False,indent=2)}
```

Pose quaternion은 ManiSkill raw/GT38D와 동일한 `[x,y,z,qw,qx,qy,qz]`, descriptor의 명시적 `_xyzw` relative quaternion만 SciPy 순서다. RGB uint8, Depth signed int16 mm; nonpositive depth는 invalid로 보존한다. `depth_supported_expected_surface_fraction`은 프레임 내 예상 표면에 대한 depth 비율이며 `gt_peg_image_fraction`은 전체 128×128 중 GT peg 픽셀 비율이다. 전체 3D 표면 관측률과 혼동하지 않는다. GT pixel-perfect 진단은 Primary Gate에 사용하지 않았다.

![PASS/FAIL 분포](distributions.png)
![Reset 순서별 누적 통과율](pass_rate.png)

## Q9. 다음 단계 사용 가능성

[Eligible export](eligible_candidate_pool.json)는 {pool['eligible_unique']}개 상태·Descriptor·RGB-D·GT 증거·Snapshot·Source/Runtime hash를 연결한다. 모든 split은 UNASSIGNED다. 72개 보완 선택 대비 {pool['eligible_unique']/72:.2f}배 후보이며 Random/Coverage-aware 비교는 아직 수행하지 않았다. [전체 Gallery](gallery.html), [모든 시도](candidate_attempts.csv), [적격성](candidate_eligibility.csv), [State](candidate_initial_states.npz), [Descriptor](candidate_state_descriptors.csv), [Visibility](candidate_visibility_metrics.csv).

## Q10. 기존 통과 Parent와 결합 전 필요한 검증

다음 단계에서는 기존 33 Train/7 Validation 통과 Parent와 신규 후보 사이 물리 상태 중복·near duplicate를 비교하고, Parent 단위 split 누출을 방지해야 한다. 선택 방법·예산·Seed와 평가 분포를 먼저 동결하고, 선택 이후 Expert 수집 단계에서 실제 full-task 성공 및 RGB-D/GT/clock 정합성을 다시 검증해야 한다. 현재 Candidate 적격성은 Expert 성공 보증이 아니다. 새로운 Blind Test seed는 이 분석 Pool에 포함하지 않았다.

## 실행·검증·제한

기존 Task10-H 프로세스는 중단/변경하지 않았다. GPU 경합을 고려해 단일 Worker로 실행했다. CPU PhysX, CUDA SAPIEN Renderer를 지정 RTX3090/57987772에서 실행했다. macOS simulator, GPU PhysX, vision/policy inference는 실행하지 않았다. 장시간 실행은 tmux `task10i-stage1`의 독립 Python runner가 관리한다. 원자적 Candidate commit, SHA/readback, lock, frozen source/config, 유한 retry(자동 retry 0), Crash-resume fixture와 별도 Replay를 적용했다. [Integration](integration_test.json), [Artifact 검증](candidate_pool_verification.json), [Runtime](runtime_identity.json), [Timing](timing_summary.json).

공통 ExperimentLogger를 W&B disabled로 재사용했다. 기존 Dataset/Checkpoint/10-H는 읽기 전용이며 Git commit·외부 Publishing은 하지 않았다. Stage2/Stage3를 자동 시작하지 않는다. 관측되지 않은 실패 유형의 예시 이미지는 합성하지 않고 [example index](example_index.json)에 NOT_OBSERVED로 기록했다.
'''
    (root/'10i_stage1_main_visible_candidate_pool.md').write_text(text)
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    (root/'10i_stage1_main_visible_candidate_pool.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I Stage1</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif}img{max-width:100%}pre{white-space:pre-wrap;background:#eee;padding:16px}</style>'+body)


def verify(root):
    from independent import validate
    validate(root)
    rr=rows(root);pool=read(root/'candidate_pool_manifest.json');replay=read(root/'candidate_replay_verification.json')
    assert replay['status']=='PASS' and replay['count']==pool['eligible_unique']
    ids={r['candidate_id'] for r in rr if r['eligible']};assert ids=={r['candidate_id'] for r in read(root/'eligible_candidate_pool.json')}
    assert len(ids)==pool['eligible_unique'] and len(rr)==pool['attempted']
    for r in rr:
        for k,h in r['files'].items():assert sha(root/'candidates'/r['candidate_id']/k)==h
    for p in root.glob('*.json'):read(p)
    with np.load(root/'candidate_initial_states.npz') as z:assert len(z['candidate_id'])==len(rr)
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file() and not any(x in p.parts for x in ['experiments']) and p.name not in ['candidate_pool_verification.json','progress.json','run.lock'] and not p.name.endswith('.log')}
    write(root/'candidate_pool_verification.json',dict(status='PASS',attempts=len(rr),eligible=len(ids),files=files,readback=True))


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    from utils.experiment_logger import ExperimentLogger
    config=dict(target=300,max_attempts=1200,protocol_sha256=sha(AUDIT/'protocol.json'),worker_count=1,physical_severe_penetration_m=.002,table_numeric_m=.00001)
    sources={str(p):sha(p) for directory in [SRC,AUDIT,BASE/'code'] for p in directory.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.urdf'] and '__pycache__' not in p.parts}
    identity=dict(config=config,sources=sources,git_dirty_source=True)
    ip=root/'source_identity.json'
    if ip.exists():assert read(ip)==identity,'Source/config changed; cannot resume this run'
    else:write(ip,identity)
    with ExperimentLogger(team='graph',model='task10i_stage1_initial_state_pool',seed=10103110,job_type='evaluate',root=root/'experiments',config=config) as log:
        if not (root/'run_identity.json').exists():write(root/'run_identity.json',dict(run_id=log.run_id))
        else:log.record(resume_source_run_id=read(root/'run_identity.json')['run_id'])
        try:
            import importlib.metadata as md
            import mani_skill.envs.tasks.tabletop.peg_insertion_side as task_module
            runtime=dict(task_module_file=str(Path(task_module.__file__)),task_module_sha256=sha(Path(task_module.__file__)),python=sys.version,platform=platform.platform(),packages={k:md.version(k) for k in ['mani_skill','sapien','torch','numpy','scipy','trimesh']},instance=os.getenv('CONTAINER_ID'),simulation='physx_cpu',renderer='sapien_cuda',gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu','--format=csv'],text=True))
            if not (root/'runtime_identity.json').exists():write(root/'runtime_identity.json',runtime)
            else:
                prior=read(root/'runtime_identity.json')
                for key in ['python','platform','packages','instance','task_module_sha256']:assert prior[key]==runtime[key],('Runtime changed',key)
            write(root/'visibility_protocol_identity.json',dict(protocol=read(AUDIT/'protocol.json'),protocol_sha256=sha(AUDIT/'protocol.json'),geometry_sha256=sha(AUDIT/'geometry.py'),primary_unchanged=True))
            stages=['VISIBILITY_REGRESSION','SEED_MANIFEST_FREEZE','INTEGRATION','FULL_CANDIDATE_GENERATION','REPLAY_VERIFICATION','FINAL_REPORT','ARTIFACT_VERIFICATION']
            for stage in stages:
                marker=root/(stage+'.done.json')
                if marker.exists():continue
                write(root/'progress.json',dict(status='RUNNING',stage=stage,pid=os.getpid()))
                print('STAGE',stage,flush=True)
                if stage=='VISIBILITY_REGRESSION':cmd([sys.executable,__file__,'regress',str(root)],root/'regression.log')
                elif stage=='SEED_MANIFEST_FREEZE':inventory(root)
                elif stage=='INTEGRATION':integration(root)
                elif stage=='FULL_CANDIDATE_GENERATION':cmd([sys.executable,__file__,'collect',str(root)],root/'collection.log')
                elif stage=='REPLAY_VERIFICATION':cmd([sys.executable,__file__,'replay',str(root)],root/'replay.log')
                elif stage=='FINAL_REPORT':finalize(root)
                elif stage=='ARTIFACT_VERIFICATION':
                    assert all(sha(Path(p))==h for p,h in sources.items()),'Source changed during run'
                    verify(root)
                write(marker,dict(status='PASS',at=time.time()))
            pool=read(root/'candidate_pool_manifest.json');write(root/'progress.json',dict(status=pool['status'],stage='STOP_AFTER_STAGE1',eligible=pool['eligible_unique'],attempted=pool['attempted']))
            for name in ['candidate_pool_manifest.json','candidate_pool_verification.json','10i_stage1_main_visible_candidate_pool.md','10i_stage1_main_visible_candidate_pool.html','gallery.html']:
                log.link_output(root/name,kind='metrics' if name.endswith('.json') else 'reports')
            log.log(dict(eligible=pool['eligible_unique'],attempted=pool['attempted']))
        except BaseException:
            err=traceback.format_exc();(root/'failure_traceback.txt').write_text(err);write(root/'progress.json',dict(status='BLOCKED',traceback=err))
            if (root/'candidate_pool_manifest.json').exists():
                failed=read(root/'candidate_pool_manifest.json');failed['status']='BLOCKED';write(root/'candidate_pool_manifest.json',failed)
            raise


def integration(root):
    # Real fixture: native seeds disjoint from full seed list. Exit after two transactions,
    # restart same process entrypoint, prove committed bytes preserved and no duplicate generation.
    f=root/'integration_fixture';f.mkdir(exist_ok=True)
    full=read(root/'candidate_seed_manifest.json');sm=dict(full);sm['ordered_attempts']=[dict(candidate_id=f'10i_smoke_{i:04d}',attempt_index=i,reset_seed=1_990_010_000+i) for i in range(12)]
    write(f/'candidate_seed_manifest.json',sm)
    cmd([sys.executable,__file__,'smoke_partial',str(f)],root/'smoke.log')
    before={str(p.relative_to(f)):sha(p) for p in (f/'candidates').rglob('*') if p.is_file()}
    crash_env=os.environ.copy();crash_env['TASK10I_CRASH']='2'
    with (root/'smoke.log').open('a') as h:
        crash=subprocess.run([sys.executable,__file__,'smoke_resume',str(f)],stdout=h,stderr=subprocess.STDOUT,env=crash_env)
    assert crash.returncode==91,crash.returncode
    assert not (f/'candidates/10i_smoke_0002/record.json').exists()
    cmd([sys.executable,__file__,'smoke_resume',str(f)],root/'smoke.log')
    assert all(sha(f/k)==v for k,v in before.items())
    cmd([sys.executable,__file__,'replay',str(f)],root/'smoke.log')
    fr=rows(f);assert len(fr)>=2 and len({r['reset_seed'] for r in fr})==len(fr)
    # Generate fixture report/export through exactly the production finalizer and verify path.
    for name in ['run_identity.json','source_identity.json','runtime_identity.json','visibility_protocol_identity.json','existing_parent_inventory.json','visibility_regression.json']:shutil.copy2(root/name,f/name)
    write(f/'integration_test.json',dict(status='PASS',fixture=True));finalize(f);verify(f)
    write(root/'integration_test.json',dict(status='PASS',real_simulator=True,attempts=len(fr),eligible=sum(r['eligible'] for r in fr),transaction_hashes_preserved=len(before),resume=True,independent_process_replay=True,report_and_artifact_readback=True,forced_exit_mid_transaction=91,smoke_not_in_full_pool=True))

if __name__=='__main__':
    mode=sys.argv[1];root=Path(sys.argv[2])
    if mode=='run':run(root)
    elif mode=='regress':regress(root)
    elif mode=='collect':collect(root)
    elif mode=='replay':replay_all(root)
    elif mode=='smoke_partial':collect(root,limit=2,smoke=True)
    elif mode=='smoke_resume':collect(root,limit=12,smoke=True)
    else:raise ValueError(mode)

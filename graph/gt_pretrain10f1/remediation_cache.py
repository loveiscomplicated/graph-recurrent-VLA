"""Exact-input-only cache reuse; otherwise unchanged causal CUDA vision writer."""
import argparse,gc,shutil,time,subprocess
from pathlib import Path
import h5py,numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,npz,now
from graph.plan_a_bc.cache import make_vision,input_identity
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode
from graph.geometry_reach10d.optimizations import DetectionMemo
from utils.experiment_logger import ExperimentLogger

SENSOR_KEYS=['robot_state','robot_world_geometry','tcp_pose','timestep','timestamp']+[
 c+'/'+k for c in ['base_camera','hand_camera'] for k in ['rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl']]

def matching(old,new):
    with h5py.File(old) as a,h5py.File(new) as b:
        if len(a['actions'])!=len(b['actions']):return False,'length'
        for k in SENSOR_KEYS:
            if a[k].shape!=b[k].shape or a[k].dtype!=b[k].dtype or a[k][:].tobytes()!=b[k][:].tobytes():return False,k
    return True,None


def adopt(root,spec,source,old):
    folder=root/'cache'/f'ep{spec["episode_id"]}';raw=root/'raw'/str(spec['episode_id'])/'episode.h5'
    same,reason=matching(Path(old['raw_path']),raw)
    if not same:return None,reason
    oldgraph=source/'full_cache'/old['path'];assert sha(oldgraph)==old['sha256']
    assert old['identity']['graph_config']==read(root/'config.json')['geometry']
    assert old['identity']['frozen_geometry_protocol_sha256']==input_identity(root)['frozen_geometry_protocol_sha256']
    folder.mkdir(parents=True,exist_ok=False)
    with np.load(oldgraph,allow_pickle=False) as z,h5py.File(raw) as f:
        arrays={k:z[k] for k in z.files};arrays['episode_id']=np.int64(spec['episode_id']);arrays['action7']=f['actions'][:]
    npz(folder/'graphs.npz',arrays);shutil.copy2(oldgraph.with_name('gate_trace.json'),folder/'gate_trace.json')
    r=dict(old,identity=input_identity(root),episode_id=spec['episode_id'],seed=spec['parent'],split=spec['split'],
        path=str((folder/'graphs.npz').relative_to(root)),sha256=sha(folder/'graphs.npz'),raw_episode_sha256=sha(raw),
        reuse_proof=dict(status='PASS',all_current_sensor_arrays_exact=True,keys=SENSOR_KEYS,includes_terminal=True,
            old_raw=str(old['raw_path']),old_raw_sha256=old['raw_episode_sha256'],old_graph=str(oldgraph),old_graph_sha256=old['sha256'],
            algorithm='same frozen geometry/vision causal prefix; all T+1 sensor input bytes exact; episode/action metadata replaced only; no GT read'),
        completed_at=now(),readback='PASS')
    write(folder/'receipt.json',r,immutable=True);return r,None


def run(root,source,dino,sam):
    torch.set_num_threads(1);torch.manual_seed(42)
    specs=read(root/'supplement_plan.json')['episodes'];oldrows=read(source/'full_cache/corrective_cache_manifest.json')['episodes']
    oldc={x['episode_id']:x for x in read(source/'full_cache/cache_candidates.json')}
    def key(parent,kind,anchor=None,offset=None):return (parent,kind,anchor,tuple(offset[:2]) if offset else None)
    lookup={key(x['parent'],x['kind'],oldc[x['episode_id']].get('anchor'),oldc[x['episode_id']].get('offset_mm')):x for x in oldrows}
    pending=list(specs);done=[];failures=[];detector=tracker=memo=pool=None
    with ExperimentLogger(team='graph',model='Task10F1_GT_supplement_graph_cache',root=root/'experiments',job_type='prepare',
        config=dict(plan_sha256=sha(root/'supplement_plan.json'),GT_input=False,source=str(source))) as log:
        try:
            while pending:
                progress=False
                for spec in list(pending):
                    raw=root/'raw'/str(spec['episode_id']);rp=raw/'receipt.json'
                    if not rp.exists():continue
                    r=read(rp);assert r['approved']
                    for name,h in r['hashes'].items():assert sha(raw/name)==h
                    target=root/'cache'/f'ep{spec["episode_id"]}'/'receipt.json'
                    if target.exists():
                        cached=read(target);assert cached['identity']==input_identity(root) and sha(target.with_name('graphs.npz'))==cached['sha256']
                    else:
                        old=lookup[key(spec['parent'],spec['kind'],spec.get('anchor'),spec.get('offset_mm'))]
                        cached,reason=adopt(root,spec,source,old)
                        if cached is None:
                            # CUDA admission never controls the other job; wait if insufficient headroom.
                            if detector is None:
                                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])
                                if free<11500:continue
                                torch.cuda.set_per_process_memory_fraction(.44)
                                detector,tracker=make_vision('cuda',dino,sam)
                                identity=read(source/'cache_gate/corrective_cache_gate.json')['model_identity']
                                memo=DetectionMemo(detector,'Task10F1_GT_supplement',identity);pool=make_pool(root,4)
                            entry=dict(episode_id=spec['episode_id'],seed=spec['parent'],split=spec['split'],actions=r['actions'],path=str(raw/'episode.h5'),sha256=r['hashes']['episode.h5'])
                            cached=cache_episode(root,Path('/'),entry,memo,tracker,log,pool,4)
                            failures.append(dict(episode_id=spec['episode_id'],exact_cache_reuse=False,first_different_input=reason))
                            memo.cache.clear();gc.collect();torch.cuda.empty_cache()
                    done.append(cached);pending.remove(spec);progress=True
                    print('SUPPLEMENT_CACHE',len(done),'/',len(specs),spec['episode_id'],'reuse',bool(cached.get('reuse_proof')),flush=True)
                    write(root/'graph_cache_progress.json',dict(completed=len(done),total=len(specs),reused=sum('reuse_proof' in r for r in done),at=now()))
                if not progress:
                    for p in root.glob('collect_*.failure.json'):raise RuntimeError('collection failed '+str(p))
                    time.sleep(5)
            write(root/'graph_cache_manifest.json',dict(status='PASS',complete=True,episodes=done,exact_reuse_mismatches=failures),immutable=True)
        finally:
            if pool:pool.shutdown(wait=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    a=p.parse_args();run(a.root,a.source,a.dino,a.sam)

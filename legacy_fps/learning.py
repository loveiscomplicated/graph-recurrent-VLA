"""Opt-in equal-budget diagnostic E0/E1/E2 and existing 09-B frozen probes.

The default is the original 2000 update budget. --updates 100 is an explicitly
labelled pilot. No policy checkpoints, GRU, BC or actions are loaded. A different
GPU user blocks this stage; the original 09-C pipeline always takes precedence.
"""
import argparse,copy,json,os,subprocess,time,shutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json,save_npz,sha256
from graph.integration_baseline.geometry09b_probe import (csv_write,standardize,linear,mlp,absolute)
from graph.persistent_geometry.train import Dataset
from .training_adapter import train_one
from graph.persistent_geometry.probes import probe_features,metric_rows,decode_saved
from .benchmark import BASE,VARIANTS

OWN_GPU_PIDS={os.getpid()}

def initialize_gpu_identity():
    """Register the sole new context across container/host PID namespaces.

    Require zero contexts before initializing ours; reject concurrent contexts.
    NVIDIA reports a host PID not visible in this container's /proc namespace.
    """
    gpu_gate()
    if not torch.cuda.is_available():raise RuntimeError('BLOCKED_RESOURCE: CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.06)
    torch.empty(1,device='cuda');torch.cuda.synchronize()
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
    if len(pids)!=1:raise RuntimeError('BLOCKED_RESOURCE: cannot uniquely identify newly initialized CUDA context')
    OWN_GPU_PIDS.add(pids[0]);return dict(container_pid=os.getpid(),nvidia_host_pid=pids[0],before_contexts=0,after_contexts=1,allocator_fraction=.06)

def gpu_gate():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    others=[]
    for line in r.stdout.strip().splitlines():
        fields=line.split(',')
        if fields and fields[0].strip().isdigit() and int(fields[0]) not in OWN_GPU_PIDS:others.append(line)
    if others:raise RuntimeError('BLOCKED_RESOURCE: another GPU process is active; original jobs take precedence')
    return dict(at=time.time(),other_gpu_processes=0)

def probes(root,models,config):
    all_absolute=[];all_episode=[];all_pairs=[];manifest=[];seen=set()
    cfg=dict(linear=dict(alphas=[0.,.01,.1,1.,10.,100.]),
        mlp=dict(hidden=64,lr=.001,weight_decay=[.0001,.01],max_epochs=config['probe_epochs'],patience=min(150,config['probe_epochs']),min_delta=1e-8))
    for model in models:
        folder=root/model['directory']
        with np.load(folder/'frozen_features.npz',allow_pickle=False) as a:data={k:a[k] for k in a.files}
        train=data['train_fit_mask'];val=data['validation_selection_mask']
        for oid in (0,1):
            y=data['target6'][:,oid*3:oid*3+3].astype(float)
            ym=y[train].mean(0);ys=np.maximum(y[train].std(0),.001);yn=(y-ym)/ys
            for name in ('RawCentroidDirect','P0_robot76','P4_centroid','P5_geometry_summary','P1_readout'):
                label='Shared' if name=='P0_robot76' else model['variant'] if name in ('RawCentroidDirect','P4_centroid','P5_geometry_summary') else model['variant']+'_'+model['encoder']
                key=(label,name,oid)
                if key in seen:continue
                seen.add(key)
                if name=='RawCentroidDirect':
                    prediction=data['raw_geometry30'][:,oid*15:oid*15+3].astype(float)
                    identity=dict(model=label,variant=model['variant'],encoder='input_only',probe=name,decoder='unlearned',object_id=oid)
                    aa,ee,pp=metric_rows(data,y,prediction,identity)
                    all_absolute+=aa;all_episode+=ee;all_pairs+=pp
                    continue
                x=probe_features(data,name,oid).astype(float);xx,xm,xs=standardize(x,train)
                for decoder,fn in (('linear',linear),('mlp',mlp)):
                    pred,weights,selection=fn(xx,yn,train,val,cfg[decoder]);prediction=pred*ys+ym
                    target=root/'probes'/f'{label}_{name}_object{oid}_{decoder}';target.mkdir(parents=True)
                    save_npz(target/'decoder.npz',**weights,feature_mean=xm,feature_std=xs,target_mean=ym,target_std=ys)
                    restored=decode_saved(target/'decoder.npz',x);np.testing.assert_allclose(restored,prediction,atol=1e-10,rtol=1e-10)
                    save_npz(target/'predictions.npz',prediction=prediction,target=y,seed=data['seed'],timestep=data['timestep'],split=data['split'],condition=data['condition'])
                    write_json(target/'selection.json',selection)
                    identity=dict(model=label,variant=model['variant'],encoder=model['encoder'] if name=='P1_readout' else 'input_only',probe=name,decoder=decoder,object_id=oid)
                    aa,ee,pp=metric_rows(data,y,prediction,identity);all_absolute+=aa;all_episode+=ee;all_pairs+=pp
                    manifest.append(dict(**identity,directory=str(target.relative_to(root)),decoder_sha256=sha256(target/'decoder.npz'),train_rows=int(train.sum()),validation_rows=int(val.sum())))
        print('legacy frozen probes '+model['variant']+'/'+model['encoder'],flush=True)
    csv_write(root/'absolute_metrics.csv',all_absolute);csv_write(root/'per_episode_metrics.csv',all_episode);csv_write(root/'paired_delta_metrics.csv',all_pairs)
    write_json(root/'probe_manifest.json',dict(config=cfg,probes=manifest))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',type=Path,required=True);parser.add_argument('--paired',type=Path);parser.add_argument('--updates',type=int,default=2000);parser.add_argument('--probe-epochs',type=int,default=2000);parser.add_argument('--variants',nargs='*',default=VARIANTS+['L0_surface','L0_rawede']);args=parser.parse_args()
    geometry=json.loads((args.geometry/'geometry_manifest.json').read_text());frames=copy.deepcopy(geometry['frames'])
    for f in frames:f['folder']=str(args.geometry/f['folder'])
    if args.paired:
        pair=json.loads((args.paired/'geometry_manifest.json').read_text())
        for f in pair['frames']:f['folder']=str(args.paired/f['folder'])
        frames+=pair['frames']
    for p in [args.geometry,*([args.paired] if args.paired else [])]:
        if json.loads((p/'metadata.json').read_text())['status']!='finished':raise ValueError('finished geometry only')
    train={f['seed'] for f in frames if f['split']=='train'};val={f['seed'] for f in frames if f['split']=='validation'}
    if train&val:raise ValueError('episode split leakage')
    config=dict(task='09-C-Legacy-FPS',seed=42,updates=args.updates,probe_epochs=args.probe_epochs,batch=16,lr=.001,weight_decay=1e-5,
        validation_every=50,optimizer='AdamW',gradient_clip=1.,target_std_floor_m=.001,
        encoders=['E0','E1','E2'],variants=args.variants,geometry=str(args.geometry),paired=str(args.paired),
        equal_training_budget=True,pilot=args.updates<2000 or args.probe_epochs<2000,
        input_adapter='canonicalize robot anchors by existing node_type; identity on serialized graphs; original parameters/forward unchanged',
        feature_condition='FPS+F5 from actual source pixels; same21-feature contract as G',
        actual_reset_groups_independent=True,final_test_executed=False,prior09AB_states_not_new_test=True,
        checkpoint_selection='lowest validation normalized geometry loss; fixed interval50 and totalupdates',
        GPU_budget_fraction=.06,original_GPU_job_takes_precedence=True,no_policy_no_gru=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    initial_gpu=initialize_gpu_identity();torch.cuda.reset_peak_memory_stats()
    with ExperimentLogger(team='integration',config=config,seed=42,model='legacy_fps_equal_budget_geometry_encoders',job_type='train',root=BASE/'learning_runs') as logger:
        root=logger.directory;print('LEGACY_LEARNING_RUN='+str(root),flush=True);(BASE/'latest_learning.txt').write_text(str(root)+'\n')
        write_json(root/'split_manifest.json',dict(train_seeds=sorted(train),validation_seeds=sorted(val),frames=frames,final_test_seeds=[],reserved_seeds=[62091,62092]))
        write_json(root/'gpu_gate.json',initial_gpu);models=[]
        original_log=logger.log
        def guarded_log(*values,**keywords):
            gpu_gate()
            return original_log(*values,**keywords)
        logger.log=guarded_log
        for variant in args.variants:
            if shutil.disk_usage(BASE).free<3*1024**3:raise RuntimeError('BLOCKED_RESOURCE: protect original disk reserve3GiB')
            gpu_gate();data=Dataset(Path('/'),variant,frames)
            for encoder in ('E0','E1','E2'):
                gpu_gate();models.append(train_one(root,data,variant,encoder,config,logger))
                write_json(root/'model_manifest.json',dict(config=config,models=models))
            del data
        # Release GPU before the CPU-only frozen probes.
        torch.cuda.empty_cache()
        probes(root,models,config)
        write_json(root/'frozen_selection.json',dict(frozen_at=time.time(),final_data_used=False,models_sha256=sha256(root/'model_manifest.json'),probes_sha256=sha256(root/'probe_manifest.json')))
        for n in ('model_manifest.json','probe_manifest.json','absolute_metrics.csv','paired_delta_metrics.csv'):logger.link_output(root/n,kind='metrics')
if __name__=='__main__':main()

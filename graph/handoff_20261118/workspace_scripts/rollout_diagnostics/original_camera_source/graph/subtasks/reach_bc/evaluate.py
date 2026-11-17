"""Fixed paired starts, actual CPU closed-loop SR, reference, traces and videos."""
from utils.experiment_logger import ExperimentLogger
import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from graph.prototype.common import seed_all, versions, write_json
from graph.prototype.data import sha256, load_dataset
from graph.prototype.maniskill_gt import GTAdapter
from graph.subtasks.env import make_subtask_env, InvalidSubtaskReset
from .policy import ROOT, DATASET, LivePolicy, decode
from .prepare import SOURCE_ROOT
from .diagnostics import reference_action, snapshot, trace_summary, controller_evidence


def fingerprint(env):
    h=hashlib.sha256()
    def add(value):
        if isinstance(value,dict):
            for k in sorted(value):
                h.update(k.encode());add(value[k])
        else:
            a=np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
            h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    add(env.unwrapped.get_state_dict())
    add(GTAdapter()(env).as_dict())
    add(dict(reach_local=env.accessor.reach_local))
    return h.hexdigest()


def reset(env, start):
    return env.reset(**start['reset_kwargs'])


def seed_inventory(prior_path=None):
    """Read seed-valued metadata, never match coincidental numeric trace values."""
    paths=set(Path('outputs/subtasks').glob('**/segments/index.json'))
    paths.update(Path('outputs/subtasks').glob('**/evaluation_starts.json'))
    paths.update(Path('outputs/subtasks').glob('**/reports/collection.json'))
    paths.update(Path('outputs/graph').glob('**/evaluation/*.json'))
    paths.update(Path('outputs/graph').glob('**/diagnostics/*fresh*.json'))
    # Exclude this experiment's newly fixed manifest on repeated reads.
    found={}
    def collect(obj, result):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('seed','episode_seed','source_seed','eval_seed','new_configuration_seeds','new_seeds'):
                    vals=v if isinstance(v,list) else [v]
                    result.update(x for x in vals if isinstance(x,int))
                else:
                    collect(v,result)
        elif isinstance(obj,list):
            for x in obj:collect(x,result)
    for p in sorted(paths):
        result=set();collect(json.loads(p.read_text()),result)
        if result:found[str(p)]=sorted(result)
    prior=Path(prior_path) if prior_path is not None else ROOT/'reports/prior_seed_inventory.json'
    if prior.exists():
        found.update(json.loads(prior.read_text()))
    return found


def make_manifest(output=ROOT, dataset=DATASET, source_root=SOURCE_ROOT):
    if platform.system()!='Linux':raise RuntimeError('Linux benchmark only')
    output=Path(output);path=output/'evaluation/paired_starts.json'
    if path.exists():raise FileExistsError(path)
    data,_=load_dataset(dataset)
    seeds=list(range(12000,12050));inventory=seed_inventory(output/'reports/prior_seed_inventory.json')
    previous=set(x for vals in inventory.values() for x in vals)
    if previous & set(seeds):
        raise ValueError(f'fresh seed overlap {previous & set(seeds)}')
    index=Path(source_root)/'segments/index.json'
    env=make_subtask_env('reach',index=index)
    starts=[]
    try:
        for ep in data['episodes']:
            if ep['episode_id'] not in data['manifest']['splits']['test']:continue
            starts.append(dict(group='held_out',eval_seed=ep['seed'],source_episode_id=ep['episode_id'],
                               reset_kwargs=dict(seed=ep['seed'],options={'source_episode_id':ep['episode_id']})))
        starts.extend(dict(group='fresh',eval_seed=s,source_episode_id=None,reset_kwargs=dict(seed=s)) for s in seeds)
        for start in starts:
            start.update(subtask='reach',max_steps=env.budget,environment_config=env.dataset['environment'],
                         simulator_version=versions()['mani_skill'],simulator_backend='physx_cpu')
            try:
                reset(env,start);first=fingerprint(env)
                reset(env,start);second=fingerprint(env)
                if first!=second:raise ValueError('same-seed initial state differs')
                start.update(valid=True,initial_state_fingerprint=first,repeat_initial_state_fingerprint=second,
                             initial_metrics=dict(env.metrics),reset_valid=True)
            except InvalidSubtaskReset as exc:
                start.update(valid=False,reset_valid=False,initial_state_fingerprint=None,invalid_reason=str(exc))
            print('manifest',start['group'],start['eval_seed'],start['valid'],flush=True)
        evidence=controller_evidence(env)
    finally:env.close()
    report=dict(index=str(index.resolve()),index_sha256=sha256(index),dataset_sha256=sha256(dataset),
                splits=data['manifest']['splits'],starts=starts,versions=versions(),seed_inventory=inventory,
                fresh_seeds=seeds,seed_overlap=[],controller=evidence,
                fixed_before_policy_evaluation=True,success_predicate='unchanged SubtaskSpec reach',max_steps=130)
    write_json(path,report)
    write_json(output/'reports/controller_convention.json',evidence)
    return report


def wilson(successes, n):
    if not n:return None
    z=1.959963984540054;p=successes/n
    d=1+z*z/n;c=(p+z*z/(2*n))/d
    e=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-e),float(c+e)]


def summary(records):
    valid=[r for r in records if r['reset_valid']]
    successful=[r for r in valid if r['success']]
    return dict(attempted_resets=len(records),valid_resets=len(valid),invalid_resets=len(records)-len(valid),
                successful_episodes=len(successful),failed_episodes=sum(r['failure_reason'] not in (None,'timeout') for r in valid),
                timeout_episodes=sum(r['failure_reason']=='timeout' for r in valid),
                success_rate=len(successful)/len(valid) if valid else None,wilson_95=wilson(len(successful),len(valid)),
                mean_success_steps=float(np.mean([r['steps'] for r in successful])) if successful else None,
                mean_failure_steps=float(np.mean([r['steps'] for r in valid if not r['success']])) if len(valid)>len(successful) else None,
                failure_reasons=dict(Counter(r['failure_reason'] for r in valid if not r['success'])),
                diagnostic_cases=dict(Counter(r['diagnostics']['case'] for r in valid)),
                mean_minimum_distance_m=float(np.mean([r['diagnostics']['minimum_position_error_m'] for r in valid])) if valid else None)


def run_episode(env,start,predictor,mode='flat',writer=None):
    try:
        reset(env,start)
    except InvalidSubtaskReset as exc:
        return dict(start=start,reset_valid=False,success=False,steps=0,failure_reason=str(exc))
    actual=fingerprint(env)
    if actual!=start['initial_state_fingerprint']:
        raise ValueError(f'paired reset fingerprint mismatch: {start["eval_seed"]}')
    initial=snapshot(env);trace=[];latency=[];full_latency=[]
    def frame(step):
        if writer is None:return
        import cv2
        image=env.render()[0].cpu().numpy().copy()
        cv2.putText(image,f'{mode} seed={start["eval_seed"]} step={step}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(image,f'pos={env.metrics["position_error"]:.3f}m rot={np.rad2deg(env.metrics["rotation_error"]):.1f}deg',(12,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        writer.append_data(image)
    frame(0)
    while True:
        before=snapshot(env);fs=perf_counter()
        if mode=='reference':
            raw=reference_action(env);latency.append((perf_counter()-fs)*1000)
        else:
            features=predictor.features(env)
            ps=perf_counter();raw=predictor.model(*features)
            latency.append((perf_counter()-ps)*1000)
        full_latency.append((perf_counter()-fs)*1000)
        raw=raw.detach().cpu()
        applied=decode(raw).clone()
        if mode=='fixed_open':applied[:,6]=1.
        _,_,terminated,truncated,info=env.step(applied)
        row=dict(t=len(trace),before=before,after=snapshot(env),raw_action=raw[0].tolist(),applied_action=applied[0].tolist(),
                 clipped=bool((raw[:,:6].abs()>1).any() or (mode!='fixed_open' and (raw[:,6].abs()>1).any())),
                 rotation_norm_saturated=bool(applied[:,3:6].norm()>1),success=info['subtask_success'],
                 terminated=terminated,truncated=truncated,failure_reason=info['failure_reason'])
        trace.append(row)
        if len(trace)%2==0 or terminated or truncated:frame(len(trace))
        if terminated or truncated:break
    return dict(start=start,reset_valid=True,initial_state_fingerprint=actual,initial=initial,trace=trace,
                success=info['subtask_success'],steps=len(trace),failure_reason=info['failure_reason'],
                diagnostics=trace_summary(initial,trace,info['subtask_success']),
                forward_latency_ms=latency,feature_and_forward_latency_ms=full_latency)


def evaluate(output=ROOT,dataset=DATASET,mode='flat',repeat=False,video_seed=None):
    if platform.system()!='Linux':raise RuntimeError('Linux benchmark only')
    with ExperimentLogger(team='graph', config=dict(dataset=dataset, output=output, mode=mode, repeat=repeat, video_seed=video_seed), seed=0, model=mode, job_type='evaluate') as experiment:
        experiment.link_output(Path(output)/'checkpoints/best.pt', checkpoint=True)
        seed_all(0,4);output=Path(output)
        manifest_path=output/'evaluation/paired_starts.json'
        manifest=json.loads(manifest_path.read_text())
        if manifest['dataset_sha256']!=sha256(dataset) or manifest['index_sha256']!=sha256(manifest['index']):
            raise ValueError('manifest source changed')
        checkpoint=output/'checkpoints/best.pt'
        predictor=None if mode=='reference' else LivePolicy(checkpoint,dataset)
        experiment.record(parameters=predictor.ck['config'] if predictor else None, dataset_sha256=manifest['dataset_sha256'], evaluation_manifest=str(manifest_path.resolve()))
        label=mode+('_repeat' if repeat else '')
        result_path=output/f'evaluation/{label}.json'
        if video_seed is None and result_path.exists():raise FileExistsError(result_path)
        env=make_subtask_env('reach',index=manifest['index'],render_mode='rgb_array' if video_seed is not None else None)
        records=[]
        try:
            with torch.inference_mode():
                for start in manifest['starts']:
                    if video_seed is not None and start['eval_seed']!=video_seed:continue
                    if not start['valid']:
                        # Retry only the SAME fixed start; record the actual reset outcome.
                        pass
                    if video_seed is not None:
                        import imageio.v2 as imageio
                        path=output/f'videos/{mode}_{video_seed}.mp4'
                        if path.exists():raise FileExistsError(path)
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with imageio.get_writer(path,fps=10) as writer:
                            r=run_episode(env,start,predictor,mode,writer)
                        write_json(path.with_suffix('.json'),r)
                        experiment.link_output(path, kind='rollouts')
                        experiment.link_output(path.with_suffix('.json'), kind='episode_metadata')
                    else:r=run_episode(env,start,predictor,mode)
                    records.append(r)
                    experiment.log(summary(records), prefix='eval/partial')
                    print(mode,start['group'],start['eval_seed'],r['success'],r['steps'],r['failure_reason'],flush=True)
        finally:env.close()
        if video_seed is not None:
            if not records:raise ValueError('video seed not in manifest')
            return records
        report=dict(mode=mode,checkpoint=str(checkpoint) if predictor else None,
                    checkpoint_sha256=sha256(checkpoint) if predictor else None,manifest_sha256=sha256(manifest_path),
                    versions=versions(),backend='physx_cpu',device='cpu',records=records,
                    held_out=summary([r for r in records if r['start']['group']=='held_out']),
                    fresh=summary([r for r in records if r['start']['group']=='fresh']))
        lat=[v for r in records if r['reset_valid'] for v in r['forward_latency_ms']]
        full=[v for r in records if r['reset_valid'] for v in r['feature_and_forward_latency_ms']]
        report['latency']=dict(forward_mean_ms=float(np.mean(lat)),forward_p50_ms=float(np.median(lat)),
                               forward_p95_ms=float(np.percentile(lat,95)),feature_and_forward_mean_ms=float(np.mean(full))) if lat else None
        if repeat:
            old=json.loads((output/f'evaluation/{mode}.json').read_text())
            checks=[]
            for a,b in zip(old['records'],records):
                fields=('reset_valid','success','steps','failure_reason','initial_state_fingerprint','trace')
                checks.append(all(a.get(k)==b.get(k) for k in fields))
            if len(old['records'])!=len(records) or not all(checks):raise ValueError('policy repeat not exactly reproducible')
            report['repeat_exact']=True
        write_json(result_path,report)
        experiment.log(report, prefix='eval')
        experiment.link_output(result_path)
        print('SUMMARY',json.dumps({k:report[k] for k in ('held_out','fresh','latency')}),flush=True)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT);p.add_argument('--manifest',action='store_true')
    p.add_argument('--mode',choices=['flat','reference','fixed_open'],default='flat')
    p.add_argument('--repeat',action='store_true');p.add_argument('--video-seed',type=int)
    a=p.parse_args()
    if a.manifest:make_manifest(a.output,a.dataset,a.source_root)
    else:evaluate(a.output,a.dataset,a.mode,a.repeat,a.video_seed)

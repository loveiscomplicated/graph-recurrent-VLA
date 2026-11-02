"""Measured live module/E2E latency, calls and GPU peaks; no model execution."""
import argparse
from collections import defaultdict
import importlib.metadata
import os
from pathlib import Path
import subprocess
import numpy as np
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def summary(values):
    values=np.asarray(values,float)
    if not len(values):return dict(samples=0,mean_ms=None,p95_ms=None,total_ms=0.)
    assert np.isfinite(values).all() and (values>=0).all()
    return dict(samples=len(values),mean_ms=float(values.mean()),p95_ms=float(np.percentile(values,95)),total_ms=float(values.sum()))


def run(root,expected,baseline,challenger,candidate,criteria):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    entries=read(expected)['entries'];sources={'B0':baseline,candidate:challenger}
    groups=defaultdict(list);counts=defaultdict(lambda:defaultdict(int));peaks=defaultdict(float);inputs=[];episode_rows=[]
    for condition,source in sources.items():
        state=read(source/'pipeline_state.json');assert state['status']=='FROZEN_LIVE_CONDITION_COMPLETE' and state['condition']==condition
        for e in entries:
            folder=source/'perception'/str(e['episode_id']);receipt=read(folder/'receipt.json')
            assert receipt['frames']==e['actions']+1 and receipt['raw_sha256']==e['raw_sha256']
            assert all(sha(folder/p)==h for p,h in receipt['files'].items())
            timing=read(folder/'timing.json');assert [r['timestep'] for r in timing]==list(range(e['actions']+1))
            dino=read(folder/'detector_timing.json');sam=read(folder/'sam_timing.json')
            stages=read(folder/'sam_stage_timing.json');features=read(folder/'feature_total_timing.json')
            fwd=read(folder/'feature_calls.json') if (folder/'feature_calls.json').exists() else []
            policy=read(folder/'events.json') if (folder/'events.json').exists() else []
            handoff=read(folder/'baseline_handoff_events.json') if (folder/'baseline_handoff_events.json').exists() else []
            requests={r['timestep'] for r in policy if r['detection_requested']}
            accepted={r['timestep'] for r in policy if r['accepted']};handed={r['timestep'] for r in handoff if r['accepted']}
            buckets=('all_exploratory',e.get('stratum','internal'))
            for bucket in buckets:
                key=(bucket,condition);counts[key]['frames']+=len(timing);counts[key]['control_steps']+=e['actions'];counts[key]['episodes']+=1
                for metric in ('end_to_end_ms','baseline_and_input_ms','correction_ms','graph_ms'):
                    groups[key+(metric,)].extend(r[metric] for r in timing)
                for name,select in [('t0',lambda r:r['timestep']==0),('after_t0',lambda r:r['timestep']>0),
                    ('requested',lambda r:r['timestep'] in requests),('reanchor_accepted',lambda r:r['timestep'] in accepted),
                    ('baseline_handoff',lambda r:r['timestep'] in handed)]:
                    groups[key+('end_to_end_'+name,)].extend(r['end_to_end_ms'] for r in timing if select(r))
                    groups[key+('correction_'+name,)].extend(r['correction_ms'] for r in timing if select(r))
                for name,values in [
                    ('DINO_total', [r['total_ms'] for r in dino]),('DINO_model_forward',[r['forward_ms'] for r in dino]),
                    ('DINO_preprocess',[r['pre_ms'] for r in dino]),('DINO_postprocess',[r['post_ms'] for r in dino]),
                    ('SAM_core_single_frame',sam),('DINOv2_model_forward',[r['forward_ms'] for r in fwd if r['model_forward']]),
                    ('DINOv2_full_feature_request',[r['total_ms'] for r in features])]:groups[key+(name,)].extend(values)
                for kind in ('start','track','reset'):groups[key+('SAM_'+kind,)].extend(r['total_ms'] for r in stages if r['kind']==kind)
                counts[key]['DINO_calls']+=len(dino);counts[key]['SAM_core_calls']+=len(sam)
                counts[key]['DINOv2_forward_calls']+=sum(r['model_forward'] for r in fwd)
                counts[key]['DINOv2_feature_requests']+=len(features);counts[key]['accepted_reanchors']+=sum(r['accepted'] for r in policy)
                counts[key]['accepted_baseline_handoffs']+=sum(r['accepted'] for r in handoff)
                peaks[key]=max(peaks[key],max(r['peak_GPU_allocated_GiB'] for r in timing))
                peaks[key+('reserved',)]=max(peaks[key+('reserved',)],max(r['peak_GPU_reserved_GiB'] for r in timing))
            episode_rows.append(dict(episode=e['episode_id'],parent_id=e['parent_id'],stratum=e.get('stratum','internal'),condition=condition,
                **summary([r['end_to_end_ms'] for r in timing]),DINO_calls=len(dino),SAM_core_calls=len(sam),
                DINOv2_forward_calls=sum(r['model_forward'] for r in fwd),frames=len(timing),control_steps=e['actions']))
            inputs.append(dict(path=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json')))
    rows=[]
    for (bucket,condition,metric),values in sorted(groups.items()):
        row=dict(stratum=bucket,condition=condition,component=metric,**summary(values))
        n=counts[bucket,condition]
        row.update(episodes=n['episodes'],frames=n['frames'],control_steps=n['control_steps'],
            samples_per_episode=len(values)/n['episodes'],samples_per_control_step=len(values)/n['control_steps'])
        rows.append(row)
    gates=[];rules=read(criteria)['runtime']
    for bucket in sorted({r['stratum'] for r in rows}):
        lookup={(r['condition'],r['component']):r for r in rows if r['stratum']==bucket}
        b=lookup['B0','end_to_end_ms'];c=lookup[candidate,'end_to_end_ms']
        for field,limit in [('mean_ms',rules['mean_end_to_end_ratio_maximum']),('p95_ms',rules['p95_end_to_end_ratio_maximum'])]:
            ratio=c[field]/b[field];gates.append(dict(stratum=bucket,metric='end_to_end_'+field+'_ratio',value=ratio,limit=limit,status='PASS' if ratio<=limit else 'FAIL'))
        peak=peaks[bucket,candidate];limit=rules['peak_GPU_allocated_GiB_maximum']
        gates.append(dict(stratum=bucket,metric='peak_GPU_allocated_GiB',value=peak,limit=limit,status='PASS' if peak<=limit else 'FAIL'))
    csv_write(root/'latency_and_gpu_metrics.csv',rows);csv_write(root/'per_episode_latency.csv',episode_rows)
    write(root/'runtime_gate_results.json',gates)
    write(root/'calls_and_memory.json',[dict(stratum=k[0],condition=k[1],**v,
        peak_allocated_GiB=peaks[k],peak_reserved_GiB=peaks[k+('reserved',)]) for k,v in counts.items()])
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),criteria_sha256=sha(criteria),candidate=candidate,
        GPU=gpu,CPU_count=os.cpu_count(),packages={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')})
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_live_runtime_cost_report',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        write(root/'scope.json',dict(status='MEASURED_LIVE_RUNTIME_REPORT_COMPLETE',run_id=logger.run_id,**config,inputs=inputs,
            separate_process_models_and_memories=True,no_shared_cross_condition_cache_credit=True,
            module_times_not_additive=True,SAM_start_includes_loader_warmup_and_prompts=True,
            SAM_core_time_excludes_video_loading=True,DINOv2_forward_and_full_request_separate=True,
            end_to_end_includes_RGBD_HDF5_FK_vision_correction_Graph=True,research_serialization_excluded=True,
            normal_and_synthetic_stress_separate=True,twenty_Hz_closed_loop_verified=False,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','expected','baseline','challenger','criteria'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args();run(a.root,a.expected,a.baseline,a.challenger,a.candidate,a.criteria)

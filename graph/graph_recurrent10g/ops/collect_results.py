"""Read-only SSH receipt/audit after the existing tmux pipeline terminates.

This local helper never starts/restarts a research worker or invokes any agent.
It is deliberately outside the frozen remote research source snapshot.
"""
import argparse
import ast
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPORT = '10g_f1_initialized_graph_recurrent_abc.md'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def checked_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('artifact path escapes root: '+relative)
    resolved = (root/p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('artifact symlink escapes root: '+relative)
    return resolved


def boolean(value):
    if value not in ('True', 'False'): raise ValueError('invalid CSV boolean '+str(value))
    return value == 'True'


def verify_grid(rows, conditions, steps, seeds):
    actual = [(r['condition'], int(r['step']), int(r['seed'])) for r in rows]
    expected = set(itertools.product(conditions, steps, seeds))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('missing/duplicate/unexpected rollout combination')


PROBE = r'''
import json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]);state=json.loads((root/'pipeline_state.json').read_text());pid=state.get('pid')
p=Path('/proc')/str(pid);live=False
if (p/'cmdline').exists():
 cmd=(p/'cmdline').read_bytes()
 status=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
 live=status!='Z' and b'graph.graph_recurrent10g.orchestrator' in cmd and str(root).encode() in cmd
print('TASK10G_JSON:'+json.dumps(dict(status=state['status'],stage=state.get('current_stage'),
 pid=pid,orchestrator_live=live,observed_at=datetime.now(timezone.utc).isoformat())))
'''

REMOTE_AUDIT = r'''
import json,sys,csv,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);source=root.parent/'source';sys.path.insert(0,str(source))
from graph.graph_recurrent10g.orchestrator import verify_all_stages
from graph.plan_a_bc.common import sha,read
verify_all_stages(root)
identity=read(root/'source_identity.json')
for name,h in identity['hashes'].items():
 assert sha(Path(identity['source'])/name)==h,'research source changed '+name
for name,h in read(root/'original_preservation.json').items():assert sha(name)==h,'F1 artifact changed'
meta=read(root/'host_data/metadata.json');mask=np.load(root/'host_data/bc_mask.npy',mmap_mode='r')
schedule=read(root/'sampling_schedule.json');expected=[];all_scans=0;bc=0
for cursor,entry in enumerate(schedule,1):
 ix=[i for eid in entry['ids'] for i in meta['episodes'][str(eid)]['indices'][entry['start']:entry['start']+32]]
 n=int(mask[ix].sum());all_scans+=len(ix);bc+=n
 assert bool(n)==entry['supervised'],'schedule mask contract'
 if n:expected.append((entry['update'],cursor,n,len(ix)))
for c in 'ABC':
 with (root/f'train_{c}.csv').open() as f:rows=list(csv.DictReader(f))
 observed=[tuple(int(r[k]) for k in ('step','cursor','valid_BC_frames','scanned_frames')) for r in rows]
 assert observed==expected,'actual BC exposure mismatch '+c
for source in read(root/'dataset_identity.json')['sources']:
 assert sha(source['path'])==source['sha256'] and sha(source['graphs'])==source['graph_sha256'],'original data changed'
print('TASK10G_JSON:'+json.dumps(dict(status='PASS',frozen_stage_outputs_readback=True,
 source_unchanged=True,F1_original_hashes_preserved=True,original_raw_and_graph_hashes_preserved=True,
 actual_common_BC_exposure=True,optimizer_updates=len(expected),all_history_frame_exposures=all_scans,
 valid_BC_frame_exposures=bc,schedule_sha256=sha(root/'sampling_schedule.json'))))
'''


def remote_json(args, program, timeout=120):
    command = '/venv/main/bin/python - '+shlex.quote(args.remote_root)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-p', str(args.port),
        args.host, command], input=program, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        # SSH stderr contains host diagnostics, never dump environment/config secrets.
        raise ConnectionError('read-only SSH command failed: '+result.stderr[-1000:])
    records = [line.removeprefix('TASK10G_JSON:') for line in result.stdout.splitlines()
               if line.startswith('TASK10G_JSON:')]
    if len(records) != 1: raise ValueError('remote receipt missing or ambiguous')
    return json.loads(records[0])


def pull(args):
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    if not args.remote_root.startswith('/') or any(c in args.remote_root for c in '\n\r'):
        raise ValueError('absolute remote root required')
    transport = f'ssh -p {args.port} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3'
    cmd = ['rsync', '-az', '--checksum', '--safe-links', '--partial-dir=.rsync-partial',
        '--exclude=/host_data', '-e', transport, args.host+':'+args.remote_root.rstrip('/')+'/', str(root)+'/']
    # No --delete: preserve prior receipts and partial/failed local evidence.
    with (root.parent/'collection_transfer.log').open('a') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def tensor_hash(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def verify_local(root):
    import cv2
    import numpy as np
    import torch
    cfg=read(root/'config.json');budget=cfg['training']['updates'];state=read(root/'pipeline_state.json')
    if state['status']!='COMPLETED':raise ValueError('research pipeline not complete')
    if read(root/'COMPLETED')['completion_audit_sha256']!=sha(root/'completion_audit.json'):
        raise ValueError('completion marker changed')
    from graph.graph_recurrent10g.report import REQUIRED
    for name in REQUIRED+['pipeline_state.json','stage_manifest.json','completion_audit.json','artifact_index.json']:
        if not (root/name).is_file() or (root/name).stat().st_size==0:raise ValueError('required artifact '+name)
    indexed=read(root/'artifact_index.json')['files']
    for item in indexed:
        p=checked_path(root,item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('transferred artifact hash/readback '+item['path'])
    cal=read(root/'lambda_calibration.json');protocol=read(root/'training_protocol_frozen.json')
    if cal['lambda']!=protocol['lambda_geometry']:raise ValueError('lambda changed after freeze')
    valid=[r for r in cal['rows'] if min(r['BC_gnn_norm'],r['geometry_gnn_norm'])>1e-12
           and math.isfinite(r['BC_gnn_norm']) and math.isfinite(r['geometry_gnn_norm'])]
    raw=cfg['calibration']['fallback'] if cal['fallback_used'] else cfg['calibration']['ratio']*float(np.median([r['BC_gnn_norm'] for r in valid]))/float(np.median([r['geometry_gnn_norm'] for r in valid]))
    expected=float(np.clip(raw,*cfg['calibration']['clamp']))
    if not math.isclose(expected,cal['lambda'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('lambda median rule')
    train_ids=set(read(root/'action_codec.json')['fitted_episode_ids'])
    if any(not set(r['episode_ids'])<=train_ids for r in cal['rows']):raise ValueError('non-train calibration')
    if set(cfg['evaluation']['dev_seeds'])&set(cfg['evaluation']['final_seeds']):raise ValueError('Dev/Final overlap')
    initial=read(root/'initial_weight_identity.json')['hashes'];selected=read(root/'model_selection_frozen.json')['conditions']
    for name in ('gnn','readout','projection','gru','action_head'):
        if len({initial[c][name] for c in 'ABC'})!=1:raise ValueError('unequal initial '+name)
    schedule=read(root/'sampling_schedule.json');last_cursors=set();checkpoints={}
    for c in 'ABC':
        train=csv_rows(root/f'train_{c}.csv')
        if len(train)!=budget or [int(r['step']) for r in train]!=list(range(1,budget+1)):
            raise ValueError('training update budget/exposure ledger '+c)
        for row in train:
            for key in ('BC_loss','geometry_loss','gradient_norm','gnn_gradient_norm','state_head_gradient_norm'):
                if not math.isfinite(float(row[key])):raise ValueError('nonfinite training '+c+' '+key)
            if int(row['extra_graphs'])!=(2*cfg['geometry_pairs_per_update'] if c=='C' else 0):
                raise ValueError('geometry-only graph exposure '+c)
            if c=='A' and float(row['gnn_gradient_norm'])!=0:raise ValueError('frozen encoder gradient')
            if c in 'BC' and float(row['gnn_gradient_norm'])<=0:raise ValueError('joint gradient missing')
            if c=='C' and (float(row['state_head_gradient_norm'])<=0 or float(row['geometry_loss'])<=0):
                raise ValueError('C geometry route missing')
        ck=torch.load(root/f'train_{c}/last.pt',map_location='cpu',weights_only=False)
        if ck['condition']!=c or ck['progress']['step']!=budget or ck['progress']['cursor']!=len(schedule):
            raise ValueError('checkpoint budget/cursor '+c)
        for file,key in [('config.json','config_sha256'),('training_protocol_frozen.json','protocol_sha256'),
                ('dataset_identity.json','dataset_sha256'),('sampling_schedule.json','schedule_sha256')]:
            if ck[key]!=sha(root/file):raise ValueError('checkpoint identity '+file)
        if ck['normalizer']!=read(root/'normalizer.json') or ck['codec']!=read(root/'action_codec.json'):
            raise ValueError('checkpoint normalization '+c)
        if not all(k in ck for k in ('optimizer','scheduler','rng','initial')):raise ValueError('incomplete resume state')
        if ck['scheduler']['last_epoch']!=budget or not ck['optimizer']['state']:raise ValueError('optimizer/scheduler readback '+c)
        if not all(k in ck['rng'] for k in ('python','numpy','torch','cuda')):raise ValueError('incomplete RNG')
        enc={k.removeprefix('graph_encoder.'):v for k,v in ck['model'].items() if k.startswith('graph_encoder.')}
        if (tensor_hash(enc)==initial[c]['gnn'])!=(c=='A'):raise ValueError('frozen/joint update invariant '+c)
        best=root/f'train_{c}/best.pt';candidate=checked_path(root,selected[c]['path'])
        if sha(best)!=selected[c]['sha256'] or sha(candidate)!=selected[c]['sha256']:
            raise ValueError('selected Best checkpoint hash '+c)
        bestck=torch.load(best,map_location='cpu',weights_only=False)
        if bestck['progress']['step']!=selected[c]['step']:raise ValueError('selected Best checkpoint step '+c)
        for step in cfg['training']['checkpoint_steps']:
            p=root/f'train_{c}/step{step:06d}.pt'
            if sha(p)!=read(p.with_suffix('.json'))['sha256']:raise ValueError('scheduled checkpoint '+c)
        probe=torch.load(root/f'geometry_probe_{c}.pt',map_location='cpu',weights_only=False)
        if probe['updates']!=(3 if cfg['miniature'] else 500) or not probe['train_only']:
            raise ValueError('equal train-only probe budget '+c)
        if not all(torch.isfinite(t).all() for t in probe['head'].values()):raise ValueError('probe readback '+c)
        last_cursors.add(ck['progress']['cursor']);checkpoints[c]=dict(run_id=ck['run_id'],last_step=budget,
            last_sha256=sha(root/f'train_{c}/last.pt'),selected_step=selected[c]['step'],selected_sha256=sha(best))
    if len(last_cursors)!=1:raise ValueError('unequal final BC exposure')
    finals=csv_rows(root/'final_rollouts.csv');dev=csv_rows(root/'development_rollouts.csv')
    verify_grid(dev,'ABC',cfg['training']['checkpoint_steps'],cfg['evaluation']['dev_seeds'])
    # Each policy can select a different step; uniqueness is policy/parent.
    for c in 'ABC':
        verify_grid([r for r in finals if r['condition']==c],[c],[selected[c]['step']],cfg['evaluation']['final_seeds'])
    functional={r['condition']:r for r in csv_rows(root/'functional_success_comparison.csv')}
    video_frames=0;decoded_endpoint_frames=0
    for rows in (dev,finals):
        paired={}
        for row in rows:
            folder=checked_path(root,row['folder']);episode=read(folder/'episode.json');trace=read(folder/'trace.json')
            n=int(row['actions']);horizon=cfg['evaluation']['horizon']
            if not trace or n!=len(trace) or n>horizon:raise ValueError('episode length/readback')
            if [r['timestep'] for r in trace]!=list(range(n)):raise ValueError('timestep gap')
            if n<horizon and row['termination'] not in ('success','termination','time_limit'):
                raise ValueError('premature rollout stop')
            if sha(folder/'trace.json')!=row['trace_sha256'] or sha(folder/'rollout.mp4')!=row['video_sha256']:
                raise ValueError('rollout content hash')
            for metric,values in [('insertion',[r['success'] for r in trace]),('grasp',[r['grasp'] for r in trace]),
                                  ('alignment',[r['alignment'] for r in trace]),('strict_reach',[r['strict_reach'] for r in trace])]:
                if boolean(row[metric])!=any(values):raise ValueError('functional metric readback '+metric)
            streak=0;lift=False
            for record in trace:
                streak=streak+1 if record['lift_candidate'] else 0
                lift=lift or streak>=cfg['evaluation']['lift_consecutive_steps']
            if boolean(row['lift'])!=lift:raise ValueError('lift metric readback')
            cap=cv2.VideoCapture(str(folder/'rollout.mp4'))
            try:
                if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=n:raise ValueError('video count')
                ok,frame=cap.read()
                if not ok or frame.shape[:2]!=(448,768):raise ValueError('video first-frame decode')
                cap.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,_=cap.read()
                if not ok:raise ValueError('video last-frame decode')
            finally:cap.release()
            video_frames+=n
            decoded_endpoint_frames+=2
            key=int(row['seed']);pair=(row['initial_snapshot_sha256'],row['initial_input_sha256'])
            if key in paired and paired[key]!=pair:raise ValueError('paired initial state/input mismatch')
            paired[key]=pair
            with np.load(folder/'initial_student_input.npz',allow_pickle=False) as z:
                if set(z.files)!={'node_input','node_type','edge_index','edge_attr','edge_type','robot76'}:
                    raise ValueError('student input whitelist')
                if z['node_input'].shape[1]!=21 or z['edge_attr'].shape[1]!=4 or z['robot76'].shape!=(76,):
                    raise ValueError('graph/robot dimensions')
            snap=torch.load(folder/'initial_snapshot.pt',map_location='cpu',weights_only=False)
            if set(snap)!={'sim','clock','controllers','rng'}:raise ValueError('snapshot scope changed')
        if rows is finals:
            for c in 'ABC':
                subset=[r for r in rows if r['condition']==c]
                for metric in ('insertion','grasp','lift','alignment','strict_reach'):
                    if int(functional[c][metric])!=sum(boolean(r[metric]) for r in subset):raise ValueError('success aggregate '+c)
    # Recompute selection from Development only, including every registered tie-break.
    for c in 'ABC':
        ranks=[]
        for step in cfg['training']['checkpoint_steps']:
            rows=[r for r in dev if r['condition']==c and int(r['step'])==step]
            counts=[sum(boolean(r[k]) for r in rows) for k in ('insertion','lift','grasp','alignment')]
            error=float(np.mean([float(r['final_tcp_peg_distance_m']) for r in rows]))
            val=read(root/f'train_{c}/validation/{step:06d}.json')['BC_loss']
            ranks.append((tuple([-n for n in counts]+[error,val,step]),step))
        if min(ranks)[1]!=selected[c]['step']:raise ValueError('preregistered Development selection '+c)
    pairs=csv_rows(root/'paired_success_analysis.csv')
    from scipy.stats import binomtest
    for left,right in [('A','B'),('B','C'),('A','C')]:
        for metric in ('insertion','lift','grasp','alignment'):
            l={int(r['seed']):boolean(r[metric]) for r in finals if r['condition']==left}
            rr={int(r['seed']):boolean(r[metric]) for r in finals if r['condition']==right}
            gained=sum(not l[k] and rr[k] for k in l);lost=sum(l[k] and not rr[k] for k in l)
            row=next(r for r in pairs if r['contrast']==right+'-'+left and r['metric']==metric)
            p=float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.
            if (int(row['gained']),int(row['lost']),int(row['parents']))!=(gained,lost,len(l)) or not math.isclose(float(row['exact_discordant_p']),p):
                raise ValueError('paired success analysis readback')
    if not cfg['miniature']:
        phase=csv_rows(root/'phasewise_action_errors.csv')
        for c in 'ABC':
            for split in ('train','validation'):
                for name in ('reach','descend','close','lift','align','insert','expert_prefix','recovery'):
                    if not any(r['condition']==c and r['split']==split and r['phase']==name for r in phase):
                        raise ValueError('phase diagnostic coverage '+c+' '+split+' '+name)
    # Geometry coverage: matched probes, visibility strata, real5/10mm shifts.
    geom=csv_rows(root/'geometry_metrics.csv');deltas=csv_rows(root/'paired_displacement_metrics.csv')
    for c in 'ABC':
        for field in ('peg_position','hole_position','peg_tip_hole','peg_orientation','hole_orientation',
                      'peg_half_x','peg_half_y','peg_half_z','hole_radius'):
            for stratum in ('all','present','missing'):
                if not any(r['model']==c+'_matched_probe' and r['field']==field and r['stratum']==stratum for r in geom):
                    raise ValueError('geometry coverage '+c+' '+field)
        if not cfg['miniature']:
            for obj in ('peg','box'):
                magnitudes={round(float(r['actual_magnitude_mm'])) for r in deltas if r['model']==c+'_matched_probe' and r['object']==obj}
                if not {5,10}<=magnitudes:raise ValueError('5/10mm displacement coverage')
    return dict(status='PASS',at=now(),engineering=cfg['miniature'],primary_updates_per_condition=budget,
        indexed_files_readback=len(indexed),development_episodes=len(dev),final_episodes=len(finals),
        video_frame_count_total=video_frames,decoded_endpoint_frames=decoded_endpoint_frames,
        checkpoints=checkpoints,selection_recomputed=True,
        paired_snapshot_scope='sim/controller/task RNG plus initial graph; PhysX hidden solver state not certified',
        actual_main_success={c:{m:int(functional[c][m]) for m in ('grasp','lift','insertion')} for c in 'ABC'},
        no_training_or_model_updates=True)


def quantitative_summary(root):
    """Read existing CSVs only; no new policy, probe, selection or experiment."""
    import numpy as np
    def number(value):
        if value in (None, ''):return None
        value=float(value)
        return value if math.isfinite(value) else None
    def mean(rows,field):
        values=[number(r.get(field)) for r in rows];values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    def fmt(value):return '—' if value is None else f'{value:.4g}'
    text=['', '## 수치 요약과 해석 범위', '',
          '아래 표는 동결된 산출물의 추가 집계다. 정책·probe를 다시 학습하거나 checkpoint를 재선택하지 않았다.', '',
          '| 조건 | 마지막 Train BC | 마지막 Validation BC | 선택 step | 선택 Validation BC | C 평균 Geometry loss | 추가 Geometry graph |',
          '|---|---:|---:|---:|---:|---:|---:|']
    selection=read(root/'model_selection_frozen.json')['conditions']
    for c in 'ABC':
        train=csv_rows(root/f'train_{c}.csv');val=csv_rows(root/f'validation_{c}.csv')
        chosen=next(r for r in val if int(r['step'])==selection[c]['step'])
        text.append(f"| {c} | {fmt(float(train[-1]['BC_loss']))} | {fmt(float(val[-1]['BC_loss']))} | {selection[c]['step']} | {fmt(float(chosen['BC_loss']))} | {fmt(mean(train,'geometry_loss')) if c=='C' else '—'} | {sum(int(r['extra_graphs']) for r in train)} |")
    text += ['', 'Train 마지막 행은 공통 BC sampling schedule의 동일한 마지막 mini-batch다. 조건 간 fit 비교에는 동일 Validation 지표와 phase 진단도 함께 사용한다. Offline loss 감소는 Closed-loop 성공을 대신하지 않는다.', '',
             '| 조건 / 동일 train-only probe | Peg 위치 mm | Hole 위치 mm | Tip–Hole mm | Peg quaternion deg | Hole quaternion deg |',
             '|---|---:|---:|---:|---:|---:|']
    geom=csv_rows(root/'geometry_metrics.csv')
    for c in 'ABC':
        values=[next(r for r in geom if r['model']==c+'_matched_probe' and r['stratum']=='all' and r['field']==f)['mean']
                for f in ('peg_position','hole_position','peg_tip_hole','peg_orientation','hole_orientation')]
        text.append('| '+c+' | '+' | '.join(fmt(number(v)) for v in values)+' |')
    text += ['', 'Geometry 표는 이미 관찰했던 F1 Final의 진단이다. 새 Blind Closed-loop Final과 구별하며, quaternion 대칭의 식별 가능성을 보장하지 않는다.', '',
             '| 조건 | 물체 | 실제 이동 mm | valid pair | Parent | Δ error mm | Parent-macro error mm | cosine | signed gain | magnitude ratio | zero 대비 개선 mm |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    delta=csv_rows(root/'paired_displacement_metrics.csv')
    for c,obj,amp in itertools.product('ABC',('peg','box'),(5,10)):
        rows=[r for r in delta if r['model']==c+'_matched_probe' and r['object']==obj and round(float(r['actual_magnitude_mm']))==amp]
        parents={r['parent'] for r in rows}
        parent_mean=[mean([r for r in rows if r['parent']==parent],'delta_vector_error_mm') for parent in parents]
        macro=float(np.mean(parent_mean)) if parent_mean else None
        values=[mean(rows,'delta_vector_error_mm'),macro]+[mean(rows,f) for f in ('direction_cosine','signed_gain','magnitude_ratio','improvement_vs_zero_mm')]
        text.append(f'| {c} | {obj} | {amp} | {len(rows)} | {len(parents)} | '+' | '.join(fmt(v) for v in values)+' |')
    text += ['', 'Pair 수는 독립 표본 수가 아니다. cosine이 정의되지 않는 zero response는 평균에서 제외하고 원본 CSV의 빈 값으로 보존했다.', '',
             '| 조건 | 실제 Action 개입 eligible / 전체 | 독립 Parent | Controller response error mm | cosine | signed gain | zero 대비 개선 mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    action=csv_rows(root/'action_correction_metrics.csv')
    for c in 'ABC':
        allrows=[r for r in action if r.get('condition')==c];rows=[r for r in allrows if r.get('analysis_eligible')=='True']
        values=[mean(rows,f) for f in ('response_error_mm','direction_cosine','signed_gain','zero_response_improvement_mm')]
        text.append(f"| {c} | {len(rows)} / {len(allrows)} | {len({r['seed'] for r in rows})} | "+' | '.join(fmt(v) for v in values)+' |')
    text += ['', '이 표의 mm는 controller translation command의 변화다. 위 실제 물체 이동 mm와 같은 물리량으로 해석하지 않는다. 같은 Robot76·previous hidden·causal prefix와 baseline repeat/물리/Expert 유효성 검사를 통과한 행만 집계했다.', '',
             '| 조건 | Validation phase | frame | Controller translation error mm | Rotation component error rad | Direction reversal rate |',
             '|---|---|---:|---:|---:|---:|']
    phase=csv_rows(root/'phasewise_action_errors.csv')
    for r in phase:
        if r['split']!='validation':continue
        text.append(f"| {r['condition']} | {r['phase']} | {r['frames']} | {fmt(float(r['controller_translation_error_mm']))} | {fmt(float(r['controller_rotation_component_error_rad']))} | {fmt(float(r['direction_reversal_rate']))} |")
    text += ['', 'Phase 평균은 유효 BC frame 기준이며 branch가 같은 Parent를 공유할 수 있다. 이 frame 수를 독립 Parent 수로 간주하지 않는다. Reversal은 두 translation vector가 정의되고 dot<0인 frame의 비율이다.', '',
             '| 조건 | Final 초기 Action7 MAE | 조기 Close step 평균 | Saturation step 평균 | Object-missing step 평균 | 마지막 TCP–Peg mm | 마지막 Tip–Hole mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    failures=csv_rows(root/'failure_analysis.csv')
    for c in 'ABC':
        rr=[r for r in failures if r['condition']==c]
        values=[mean(rr,f) for f in ('initial_action_mae','premature_close','saturation','missing_frames')]
        values += [1000*mean(rr,f) for f in ('final_tcp_peg_distance_m','final_tip_hole_error_m')]
        text.append('| '+c+' | '+' | '.join(fmt(v) for v in values)+' |')
    text += ['', 'Hidden carry/reset 검사는 통과했지만 실제 실패 원인을 단독으로 배제하거나 확정하는 인과 실험은 아니다. Missing geometry, phase fit 오차와 누적 제어 오차는 별도 근거이며 Geometry loss의 개선만으로 Action Correction을 입증하지 않는다.', '',
             '| 조건 | End-to-end inference s | Policy s | CUDA encoder+readout s |',
             '|---|---:|---:|---:|']
    final=csv_rows(root/'final_rollouts.csv')
    for c in 'ABC':
        rr=[r for r in final if r['condition']==c]
        text.append('| '+c+' | '+' | '.join(fmt(mean(rr,f)) for f in ('inference_seconds','policy_seconds','gnn_encoder_readout_seconds'))+' |')
    seed=min(int(r['seed']) for r in final)
    text += ['', f'동일 첫 Final Parent{seed}의 영상(성과에 따른 선택 없음): '+', '.join(
        f"[{c}]({Path(next(r for r in final if r['condition']==c and int(r['seed'])==seed)['folder'])/'rollout.mp4'})" for c in 'ABC'), '']
    return '\n'.join(text)


def save_document(root, docs, audit):
    docs.mkdir(parents=True,exist_ok=True)
    original=(root/REPORT).read_text()+quantitative_summary(root)
    text=re.sub(r'\]\((?!https?://)([^)]+)\)',
        lambda m:']('+os.path.relpath(root/m.group(1),docs)+')',original)
    text+='\n\n## 로컬 전달 및 독립 검증\n\n'
    text+=f"원격 연구 프로세스 종료 후 SSH로 Task10-G 산출물을 회수했다. {audit['indexed_files_readback']}개 index 파일의 SHA/크기를 확인하고, "
    text+=f"Development{audit['development_episodes']}회와 Final{audit['final_episodes']}회의 trace 및 MP4 첫/마지막 프레임을 읽었다. "
    text+='공통 BC 노출, 선택 규칙, Best/Last 체크포인트와 독립 geometry probe도 별도로 검사했다. '
    text+='원격 원문 및 hash manifest는 그대로 보존했다. 원문의 외부 백업 미설정 설명은 원격 pipeline 자체의 설정이며, 이 절은 별도 SSH 수집 결과다. '
    text+='F1 원본 RGB-D/Graph/GT 데이터와 shared host mmap 전체는 이 전달의 범위에 포함하지 않았다.\n\n'
    receipt=os.path.relpath(root.parent/'independent_completion_audit.json',docs)
    text+=f'[독립 검증 기록]({receipt})\n'
    target=docs/(('10g_miniature_collector_fixture.md') if audit['engineering'] else REPORT)
    temp=target.with_suffix('.partial');temp.write_text(text);os.replace(temp,target)
    return target


def run(args):
    root=Path(args.output).resolve();parent=root.parent;parent.mkdir(parents=True,exist_ok=True)
    owner=(parent/('collector_'+root.name+'.lock')).open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=parent/('collector_'+root.name+'_status.json')
    if args.audit_local:
        audit=verify_local(root)
        if args.docs:
            document=save_document(root,Path(args.docs).resolve(),audit)
            audit.update(document=str(document),document_sha256=sha(document))
        write(parent/'independent_completion_audit.json',audit)
        print(json.dumps(audit,indent=2));return
    while True:
        try:
            state=remote_json(args,PROBE)
            write(status_path,dict(collector_status='WAITING_FOR_RESEARCH_PROCESS',**state,collector_pid=os.getpid()))
            if not state['orchestrator_live'] and state['status'] in ('COMPLETED','BLOCKED'):break
        except (ConnectionError,subprocess.TimeoutExpired,OSError,ValueError) as exc:
            write(status_path,dict(status='WAITING_FOR_SSH_OBSERVATION',at=now(),error=str(exc),collector_pid=os.getpid()))
        if args.once:raise RuntimeError('research process is not terminal; no collection/restart performed')
        time.sleep(args.poll_seconds)
    for attempt in range(1,4):
        try:
            write(status_path,dict(status='TRANSFERRING',at=now(),attempt=attempt,research_state=state['status']))
            pull(args);break
        except (subprocess.CalledProcessError,OSError) as exc:
            write(status_path,dict(status='TRANSFER_RETRY',at=now(),attempt=attempt,error=str(exc),research_process_untouched=True))
            if attempt==3:raise
            time.sleep(60)
    if state['status']!='COMPLETED':
        write(status_path,dict(status='RESEARCH_BLOCKED_ARTIFACTS_COLLECTED',at=now(),remote=state))
        raise RuntimeError('research failed; artifacts retained, no retry or worker restart')
    remote=remote_json(args,REMOTE_AUDIT,timeout=300);write(parent/'remote_completion_audit.json',remote)
    audit=verify_local(root)
    if args.docs:
        document=save_document(root,Path(args.docs).resolve(),audit)
        audit.update(document=str(document),document_sha256=sha(document),source_report_sha256=sha(root/REPORT))
    write(parent/'independent_completion_audit.json',audit)
    write(status_path,dict(status='DELIVERED_AND_AUDITED',at=now(),audit=str(parent/'independent_completion_audit.json'),
        report=audit.get('document'),research_complete=True))
    print(json.dumps(audit,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='root@104.233.209.157');p.add_argument('--port',type=int,default=13159)
    p.add_argument('--remote-root',default='/workspace/task10g_f1_abc_20261031/full');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--docs',type=Path);p.add_argument('--poll-seconds',type=float,default=60);p.add_argument('--once',action='store_true')
    p.add_argument('--audit-local',action='store_true')
    args=p.parse_args()
    try:run(args)
    except BaseException:
        root=Path(args.output).resolve();root.parent.mkdir(parents=True,exist_ok=True)
        write(root.parent/('collector_'+root.name+'_failure.json'),dict(status='FAILED',at=now(),traceback=traceback.format_exc(),
            research_process_untouched=True))
        raise


if __name__=='__main__':main()

"""Final local artifacts, static plots, relative-link/readback validation."""
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from graph.integration_baseline.direct_runtime import REPO,atomic,read,sha,finalize,now


DOC=REPO/'docs/visual-graph-integration/06c_direct_expert_bc_training.md'


def document(run):
    config=read(Path(run)/'config.yaml')
    return REPO/config['report_document'].format(run_id=Path(run).name,run_id_short=Path(run).name[:8]) if config.get('report_document') else DOC


def save_text(path,text):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid4().hex+'.partial')
    with tmp.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
    if tmp.read_text()!=text:raise ValueError('text readback failed')
    os.replace(tmp,path)


def status_document(run):
    """Initial report explicitly describes execution as pending/running."""
    run=Path(run)
    doc=document(run);config=read(run/'config.yaml')
    reuse_note=('수집 완료 CPU Dataset을 재사용하며 CUDA Vision/학습을 실행한다.'
                if config.get('execution_mode')=='existing_data_cuda' else '새 Expert Dataset을 수집한다.')
    callback_note=('오류 이벤트만 sandboxed Codex를 최대1회 호출하며 별도 repair tmux를 사용한다.'
                   if config.get('auto_callback_required',True) else
                   '사용자의 명시적 예외 승인으로 자동 Codex 콜백을 제외했다. 오류 Bundle을 보존하고 수동 조치를 기다린다.')
    text=f'''# Task06-C — Direct Expert Dataset + Graph–GNN–GRU/BC

Run UUID: `{run.name}`. **독립 tmux 실행 준비/시작 단계. 전체 Vision Dataset 및 학습 완료 아님.**

{reuse_note}

상태의 기준은 [status.json](../../experiments/{run.name}/status.json)이다.
최종 Dataset, Loss, Test 성능은 아직 산출되지 않았다. 정상 완료 시 Pipeline이
이 문서와 [정적 HTML](../../experiments/{run.name}/graph_gru_training.html)을 생성한다.

수집: normal/noise0/no hole shift, seed30000..30159, 250-step 독립 환경,
120 success/최대160 attempts, 성공96/12/12 episode split, seed42.
원본 Native Action7, 76D robot state, RGB-D 두 camera와 calibration을 기록한다.
기존 Expert·Vision·Raw EDE·Graph·GNN·GRU·BC regression MSE를 유지한다.
Object 결손은 기록하며 GT로 보완하지 않는다.

학습: {config['device']}, AdamW lr3e-4/weight_decay1e-5/betas(0.9,0.999)/eps1e-8,
CosineAnnealingLR(T_max50,eta_min0), max50 epochs/patience10/clip1,
연속 TBPTT32/batch2. Robot normalizer는 train만 fit한다. 기존 Graph feature view는
변경 없이 사용한다. Checkpoint 선택은 validation만, final test는 1회 offline 평가다.

Discord는 사용자의 후속 지시에 따라 비활성화했다. 로그·오류·복구 이력은 로컬에 남긴다.
{callback_note}
Closed-loop rollout, 외부 게시, git commit은 수행하지 않는다. 선택하지 않은 Backend는 미실행이다.

실행/중지/재개 명령과 tmux 생존 범위는 [README](../../graph/integration_baseline/README.md)를 따른다.
'''
    if doc.exists() and run.name not in doc.read_text():raise FileExistsError('another run owns Task06-C document')
    save_text(doc,text)
    save_text(run/'graph_gru_training.html',f'<!doctype html><meta charset="utf-8"><title>Task06-C | In progress</title><h1>Task06-C — In progress</h1><p>Run {run.name}. Dataset/training are not complete.</p><p><a href="status.json">Live local status</a> · <a href="config.yaml">Frozen config</a></p>')


class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href') and v:self.paths.append(v)


def verify_html(path):
    p=Links();p.feed(Path(path).read_text())
    for relative in p.paths:
        if relative.startswith(('http:','https:','data:','#')):continue
        target=Path(path).parent/relative
        if not target.is_file():raise ValueError('HTML missing link: '+relative)
        if target.suffix=='.json':read(target)
        if target.suffix=='.png':
            from PIL import Image
            with Image.open(target) as im:im.verify()
    return dict(status='PASS',relative_links=len(p.paths),browser_layout='NOT_EXECUTED')


def generate(run,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.experiment_artifacts import register_existing_output
    run=Path(run);m=read(run/'dataset_manifest.json');v=read(run/'dataset_validation.json')
    training=read(run/'training/completion.json');test=read(run/'training/test_metrics.json')
    history=training['history'];lengths=[e['actions'] for e in m['episodes']]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_mse'] for r in history],label='Train BC MSE')
    axes[0].plot(epochs,[r['val_mse'] for r in history],label='Validation BC MSE')
    axes[0].set(xlabel='Epoch',ylabel='Native Action7 MSE');axes[0].legend()
    axes[1].hist(lengths,bins=20);axes[1].set(xlabel='Actions until first success',ylabel='Success episodes')
    curve=run/'plots/loss_and_lengths.png';fig.savefig(curve,dpi=150);plt.close(fig)
    coverage=v['coverage'];fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    x=np.arange(len(coverage))
    for oid,name in enumerate(['Peg','Box']):
        ax.plot(x,[r['presence_counts'][oid]/r['observations'] for r in coverage],label=name)
    ax.set(xlabel='Dataset episode order',ylabel='Fraction of frames with object points',ylim=(-.05,1.05));ax.legend()
    presence=run/'plots/object_presence.png';fig.savefig(presence,dpi=150);plt.close(fig)
    graph=read(run/'vision_manifest.json')
    phase={};reasons={};all_points=[[],[]]
    for e in graph['episodes']:
        for frame in e['packets']:
            for oid,missing in enumerate(frame['missing']):
                reasons.setdefault(str(oid),{}).setdefault(missing['reason'],0)
                reasons[str(oid)][missing['reason']]+=1;all_points[oid].append(frame['object_counts'][oid])
    for e in coverage:
        for name,p in e['phase_counts'].items():
            entry=phase.setdefault(name,dict(frames=0,present=[0,0]))
            entry['frames']+=p['frames'];entry['present']=[a+b for a,b in zip(entry['present'],p['present'])]
    presence_summary=dict(episodes=coverage,box_never_observed_fraction=v['box_never_observed_fraction'],
        missing_reasons=reasons,phases=phase,
        point_counts={str(i):dict(min=min(p),median=float(np.median(p)),max=max(p)) for i,p in enumerate(all_points)})
    atomic(run/'vision_coverage.json',presence_summary,immutable=True)
    summary=dict(run_id=run.name,status='COMPLETE',collected_successes=m['successes'],attempts=m['attempts'],
        collection_success_yield_all_attempts=m['success_rate_all_attempts'],expert_success_rate_completed=m['success_rate_completed'],
        interrupted_attempts=len(m['interrupted_attempts']),actions=v['actions'],observations=v['observations'],
        length_distribution=dict(min=min(lengths),median=float(np.median(lengths)),max=max(lengths),mean=float(np.mean(lengths))),
        splits=read(run/'split_manifest.json')['splits'],alignment=v['alignment'],vision=presence_summary,
        model=config['model'],training=config['training'],result=training,test=test,
        tmux=read(run/'tmux.json'),discord=read(run/'discord_verification.json'),
        callback=read(run/'callback_verification.json'),errors=[str(p.relative_to(run)) for p in (run/'errors').glob('*/bundle.json')],
        recovery=read(run/'errors/callback_dispatch.json') if (run/'errors/callback_dispatch.json').exists() else None,
        collection_seconds=sum(e['elapsed_seconds'] for e in m['episodes']+m['failed_episodes']),
        vision_seconds=sum(e['elapsed_seconds'] for e in graph['episodes']),
        limitations=['BC loss is not manipulation success','predicted mask semantic accuracy is not GT-scored',
            f'executed model device: {config["device"]}; other devices NOT_EXECUTED','no closed-loop student rollout','no external artifact publishing',
            'no HPO, new architecture, expert change, dependency install or git commit'],time=now())
    atomic(run/'summary.json',summary,immutable=True)
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task06-C | {run.name}</title><style>body{{font:16px/1.6 system-ui;max-width:1120px;margin:40px auto;padding:0 20px;color:#21313c}}img{{max-width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:18px}}a{{color:#126d96}}</style>
<h1>Task06-C — Direct Expert BC</h1><p>Run {run.name} · COMPLETE · Offline evaluation only</p>
<p>Success {m['successes']}/{m['attempts']} ({m['success_rate_all_attempts']:.2%}).
Native Action7 / full Robot76 / causal predicted Graph. BC loss does not measure closed-loop success.</p>
<img src="plots/loss_and_lengths.png" alt="BC loss and episode lengths"><img src="plots/object_presence.png" alt="Object presence">
<p><a href="summary.json">Results</a> · <a href="dataset_manifest.json">Dataset</a> · <a href="split_manifest.json">Frozen split</a> ·
<a href="vision_coverage.json">Presence / missing reasons / phases / point counts</a> · <a href="training/metrics.json">Training curves</a> ·
<a href="training/test_metrics.json">Final offline test</a> · <a href="source_hashes.json">Source hashes</a> ·
<a href="verification.json">Readback</a> · <a href="training/best.json">Best checkpoint</a> · <a href="config.yaml">Config</a></p>
<h2>Measured results and limits</h2><pre>{html.escape(json.dumps({k:summary[k] for k in ['length_distribution','model','training','tmux','discord','callback','errors','recovery','collection_seconds','vision_seconds','limitations']},indent=2,ensure_ascii=False))}</pre>
<h2>Training and test</h2><pre>{html.escape(json.dumps(dict(best=training['best'],epochs=training['epochs'],training_seconds=training['elapsed_seconds'],memory=training['memory'],test=test),indent=2))}</pre></html>'''
    save_text(run/'graph_gru_training.html',body)
    atomic(run/'verification.json',dict(status='PENDING_LINK_READBACK'))
    report=verify_html(run/'graph_gru_training.html')
    report.update(dataset_alignment='PASS',training_gate=read(run/'training_gate.json'),complete_metrics=True,
        source_hashes=sha(run/'source_hashes.json'),test_evaluations=1,closed_loop='NOT_EXECUTED')
    atomic(run/'verification.json',report)
    verify_html(run/'graph_gru_training.html')
    doc=f'''# Task06-C — Direct Expert Dataset + Graph–GNN–GRU/BC

Run UUID: `{run.name}`. **전체 Dataset·BC 학습·최종 Offline Test 완료.**

[정적 HTML](../../experiments/{run.name}/graph_gru_training.html) · [전체 결과](../../experiments/{run.name}/summary.json) · [검증](../../experiments/{run.name}/verification.json).

실제 수집 성공 {m['successes']}/{m['attempts']} ({m['success_rate_all_attempts']:.2%}), 유효 {v['actions']} Actions / {v['observations']} Observations.
실패 Episode와 중단 시도는 Audit에 별도 보존했다. 성공 Episode만 {config['splits']} 고정 분할했고 Reset Group 중복은 없다.
Episode 길이 min/median/max = {min(lengths)}/{float(np.median(lengths))}/{max(lengths)}.
Native Action7·Controller·Observation 정합 PASS. Full Robot76·EE transform·원본 RGB-D/calibration은 동일 시점이다.

기존 Causal DINO/SAM2 → Shape-First/PCA5/F5 → EE → Raw EDE, keep_ratio0.5,
Variable N 및 Empty Object를 유지했다. Box 전구간 미관측 Episode 비율 {v['box_never_observed_fraction']:.2%}.
구간별 결손·회복·Point 수는 [Coverage](../../experiments/{run.name}/vision_coverage.json)에 기록했다.
GT Object/Hole/Goal은 Student 입력에 없다. 그래프 결손을 GT로 보완하지 않았다.

기존 Robot3/k4/M16/L2/d64/5d320/presence2/robot76/fused398/GRU144×1/Action7 구조다.
Train-only Robot Normalizer를 동결하고 기존 Graph feature view를 유지했다.
AdamW lr3e-4/weight_decay1e-5, CosineAnnealingLR T50/eta_min0,
TBPTT32/batch2/gradient clip1/max50/patience10이다. 연속 chunk hidden carry/detach,
다른 Episode hidden reset과 padding mask를 적용했다. 선택한 장치 `{config['device']}`에서 실행했다.

학습 {training['epochs']} Epoch, {training['elapsed_seconds']:.1f}s. Best Validation Checkpoint:
`{training['best']['path']}` (Epoch {training['best']['epoch']}).
최종 Test MSE {test['metrics']['mse']:.8g}, Translation {test['metrics']['translation_mse']:.8g},
Rotation {test['metrics']['rotation_mse']:.8g}, Gripper {test['metrics']['gripper_mse']:.8g}.
Test는 Validation으로 선택한 Checkpoint에 대해 1회 평가했다. **BC Loss와 Manipulation Success는 별개다.**

![Loss](../../experiments/{run.name}/plots/loss_and_lengths.png)
![Presence](../../experiments/{run.name}/plots/object_presence.png)

tmux `{summary['tmux']['session']}`에서 순차 Stage 실행. 전체 원본 로그는 `experiments/{run.name}/logs/`.
Discord는 후속 사용자 지시에 따라 비활성화했다. 실제 Codex sandbox fixture PASS 여부는 callback_verification.json,
오류·복구는 errors/와 events.jsonl에 보존했다. 처리량·Model/Optimizer Memory는 training/metrics.json에 있다.
Checkpoint는 Model/Optimizer/Scheduler/RNG/Normalizer/Split/Dataset/Config를 보존한다.

선택하지 않은 장치, Closed-loop Student Rollout/Success Rate, HPO, Architecture Search, 외부 게시·Git Commit은 미실행이다.
향후 Closed-loop 평가는 동일 Camera/Robot76/Action7 계약, Episode Reset/Hidden lifecycle,
별도 평가 Seed와 안전한 배포 전 검증이 필요하다. 과거 60/60 Expert 성공률은 이번 결과에 합산하지 않았다.
'''
    target_doc=document(run)
    if target_doc.exists() and run.name not in target_doc.read_text():raise ValueError('Task06-C document belongs to another run')
    save_text(target_doc,doc);save_text(run/'06c_direct_expert_bc_training.md',doc)
    for filename,kind in [('summary.json','metrics'),('verification.json','metrics'),('graph_gru_training.html','reports'),
        ('dataset_manifest.json','episode_metadata'),('split_manifest.json','episode_metadata'),('vision_coverage.json','metrics')]:
        register_existing_output(run,run/filename,kind=kind)

"""Evidence-driven automatic report; zero success remains a complete result."""
from pathlib import Path
import html
import json
import os
import shutil
import numpy as np
from graph.plan_a_bc.common import read,write,sha,now
from .analysis import read_csv

REPORT='10g_f1_initialized_graph_recurrent_abc.md'


def generate(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=read(root/'config.json');figs=root/'figures';figs.mkdir(exist_ok=True)
    functional=read_csv(root/'functional_success_comparison.csv');geometry=read_csv(root/'geometry_metrics.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for c in 'ABC':
        for index,name in enumerate(('train','validation')):
            rows=read_csv(root/f'{name}_{c}.csv')
            ax[index].plot([int(r['step']) for r in rows],[float(r['BC_loss']) for r in rows],label=c)
    for a,label in zip(ax,('Train BC','Validation BC')):a.set(title=label,xlabel='Optimizer updates',ylabel='Method B loss');a.legend()
    fig.tight_layout();fig.savefig(figs/'training_validation_curves.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
    for j,k in enumerate(('grasp','lift','insertion')):
        ax.bar(x+(j-1)*.22,[int(r[k])/int(r['parents']) for r in functional],.22,label=k)
    ax.set(xticks=x,xticklabels=list('ABC'),ylim=(0,1),ylabel='Functional success fraction');ax.legend()
    fig.tight_layout();fig.savefig(figs/'functional_success.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for j,field in enumerate(('peg_position','hole_position','peg_tip_hole')):
        rr=[next(r for r in geometry if r['model']==c+'_matched_probe' and r['field']==field and r['stratum']=='all') for c in 'ABC']
        ax.bar(x+(j-1)*.22,[float(r['mean']) for r in rr],.22,label=field)
    ax.set(xticks=x,xticklabels=list('ABC'),ylabel='F1 held-out mean error (mm)');ax.legend()
    fig.tight_layout();fig.savefig(figs/'geometry_errors.png');plt.close(fig)
    correction=read_csv(root/'action_correction_metrics.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    response=[]
    for c in 'ABC':
        rr=[r for r in correction if r.get('condition')==c and r.get('analysis_eligible')=='True']
        response.append(np.mean([float(r['response_error_mm']) for r in rr]) if rr else np.nan)
    ax.bar(list('ABC'),response);ax.set(ylabel='Controller translation response error (mm)',title='Actual object intervention; eligible pairs only')
    fig.tight_layout();fig.savefig(figs/'action_response.png');plt.close(fig)
    selected=read(root/'model_selection_frozen.json');benchmark=read(root/'concurrency_benchmark.json')
    contrasts=read_csv(root/'paired_success_analysis.csv')
    def positive_insertion(contrast):
        row=next(r for r in contrasts if r['contrast']==contrast and r['metric']=='insertion')
        return int(row['success_count_difference'])>0 and float(row['exact_discordant_p'])<=.05
    def action_improves(left,right):
        means={}
        for c in (left,right):
            rr=[r for r in correction if r.get('condition')==c and r.get('analysis_eligible')=='True']
            means[c]=float(np.mean([float(r['response_error_mm']) for r in rr])) if rr else None
        return all(v is not None for v in means.values()) and means[right]<means[left]
    assessment=dict(B_minus_A='SUPPORTED' if positive_insertion('B-A') else 'INCONCLUSIVE',
        C_minus_B='SUPPORTED' if positive_insertion('C-B') and action_improves('B','C') else 'INCONCLUSIVE',
        C_minus_A='SUPPORTED' if positive_insertion('C-A') and action_improves('A','C') else 'INCONCLUSIVE')
    if cfg['miniature']:assessment={k:'NOT_TESTED' for k in assessment}
    write(root/'hypothesis_assessment.json',dict(verdicts=assessment,
        rule='positive paired insertion count difference with exact discordant p<=.05; C contrasts additionally require lower eligible actual-action response error; single-seed evidence cannot be CONFIRMED',
        scope='descriptive support in this fixed experiment, not general algorithm superiority'))
    lines=['# Task10-G — F1-initialized Graph–Recurrent A/B/C', '',
        '**'+('ENGINEERING MINIATURE — 연구 결과로 사용하지 않음.' if cfg['miniature'] else '고정 A/B/C 학습과 Development 선택, 독립 Closed-loop Final 및 Geometry/Action 분석 완료.')+'**', '',
        '## 실제 실행과 Identity','',f'- Root: `{root}`',f'- Instance57987772 / RTX3090; CPU PhysX + CUDA renderer/vision/policy. GPU PhysX와 로컬 macOS CUDA는 미실행.',
        f'- F1 SHA256: `{cfg["f1_checkpoint_sha256"]}`',
        f'- Dataset manifest SHA256: `{cfg["dataset_manifest_sha256"]}`',
        f'- Optimizer updates: 각 {cfg["training"]["updates"]}; seed42 한 개. BC sampling schedule, train-only ActionCodec, batch2/sequence32 공통.',
        f'- 선택 병렬도: {benchmark["chosen_concurrency"]}; 실제 3 CUDA process 동시 benchmark를 실행했다. 전체 memory/throughput은 [기록](concurrency_benchmark.json)에 보존.',
        f'- 단일 geometry λ: {read(root/"lambda_calibration.json")["lambda"]:.9g}. Train-only gradient median의25% rule/clamp/fallback은 [calibration](lambda_calibration.json)에 보존.',
        '- 기존 ExperimentLogger의 실행 UUID와 로컬 기록을 재사용했다. Git commit, 외부 publishing, 새 유료 인스턴스 생성 없음.', '',
        '## Functional Closed-loop Final','',
        '| Condition | Parents | Grasp | Lift | Insertion | Selected step |','|---|---:|---:|---:|---:|---:|']
    for r in functional:
        c=r['condition'];lines.append(f'| {c} | {r["parents"]} | {r["grasp"]} | {r["lift"]} | {r["insertion"]} | {selected["conditions"][c]["step"]} |')
    lines += ['',f'Final seeds: `{cfg["evaluation"]["final_seeds"]}`. Zero offsets. 최대 {cfg["evaluation"]["horizon"]} Action, 공식 success/환경 종료에서 종료.',
        'Development는 고정 Validation Parent에서만 실행했다. Insertion→Lift→Grasp→Alignment→최종 TCP–Peg error→Validation BC→earliest step 순으로 선택하고 Final 전에 checkpoint SHA를 고정했다.',
        '각 조건은 독립 causal vision memory/GRU reset을 사용한다. 같은 seed의 초기 simulator/controller/RNG snapshot과 최초 graph/Robot76 hash가 정확히 일치했다. PhysX의 숨은 solver 상태 전체 직렬화나 모든 장치의 bit-exact trajectory를 주장하지 않는다.',
        '[전체 Final](final_rollouts.csv) · [선택 동결](model_selection_frozen.json) · [Paired 분석](paired_success_analysis.csv)', '',
        'Strict Reach는 기존 Task10D의 ordered approach/final pose/open-gripper/early-close/peg-motion/collision 계약으로 별도 계산했다. 실제 Insertion 성공보다 우선하지 않는다. End-to-end inference, 전체 policy, CUDA encoder+readout 시간은 각각 다른 열로 기록했다.',
        '![Functional success](figures/functional_success.png)', '',
        '## Geometry와 Action 반응','',
        'Geometry는 이미 관찰했던 F1 Final91000–91011의 재사용 진단이다. 새 Closed-loop Blind Final과 구별한다. A는 F1 head, B는 동결 F1 head 진단, C는 공동 학습 head를 기록했다. 주 비교는 세 encoder에 동일 F1 초기 head·동일 train-only500-update budget을 적용한 독립 probe다(Engineering miniature는3-update). B의 동결 head 오차만으로 representation 열화를 단정하지 않는다.',
        '실제 물체 Pair의 ±5/10mm Δ vector error/direction/signed gain/magnitude ratio/zero-response 차이를 기록했다. Action 분석은 기존 Task10D의 실제 Peg 개입과 Expert counterfactual 코드로 같은 Robot76, previous hidden, causal prefix를 고정한다. 물체 이동 mm와 controller command response mm는 별도 물리량이다. Baseline repeat/물리/Expert 계약을 통과한 Pair만 주 해석에 쓴다.',
        '[Geometry metrics](geometry_metrics.csv) · [Actual displacement](paired_displacement_metrics.csv) · [Action correction](action_correction_metrics.csv)', '',
        '![Geometry](figures/geometry_errors.png)', '', '![Action response](figures/action_response.png)', '',
        '## 학습과 실패 분석','',
        'A의 encoder/readout은 동결, B는 BC joint, C는 BC+Geometry joint다. C State Head는 Action/GRU 입력에 포함되지 않고 추론에서 실행하지 않는다. 동일 GNN 및 Projection/GRU/Action 초기 hash, lambda0 B/C gradient 일치, GT sidecar 차단 positive control, geometry gradient 전달과 실제 checkpoint 다음 update resume를 검증했다.',
        '정상 Expert prefix는 승인 nominal expert action/robot/timestep의 정확한 비교 후 BC를 활성화했다. 교란 구간은 실제 관측을 history에 포함하며 BC를 끈다. Recovery는 유효 expert labels만 사용했다. 같은 Parent의 branch를 독립 Parent로 세지 않는다.',
        '[Expert phase별 controller 오차](phasewise_action_errors.csv)와 [Final 실패 구간](failure_analysis.csv)에 초기 action fit, direction reversal, premature close, saturation, missing geometry, collision, error accumulation을 기록했다. Hidden carry/reset은 계약 시험과 trace로 검사했고 특정 실패 원인을 증명했다고 주장하지 않는다. Recovery 추가 rollout은 NOT_TESTED다.', '',
        '![Train/validation](figures/training_validation_curves.png)', '',
        '## 연구 질문 판정','',
        '| Question | Verdict | Evidence scope |','|---|---|---|',
        f'| B−A: 공동 BC 업데이트의 실제 제어 이점 | {assessment["B_minus_A"]} | paired outcome; 한 training seed/12 Parent |',
        f'| C−B: Geometry auxiliary 추가 효과 | {assessment["C_minus_B"]} | 성공률·geometry·action response를 함께 해석 |',
        f'| C−A: 전체 결합 효과 | {assessment["C_minus_A"]} | 제한된 paired 표본; 일반적 우월성 주장 없음 |','',
        '모든 functional metric이0이어도 사전 등록한 평가·진단을 끝냈다. Offline BC/geometry loss 감소는 Closed-loop success 또는 정확한 action correction의 증거를 대신하지 않는다. 추가 λ 탐색/학습/데이터 수집은 실행하지 않았다.', '',
        '## 운영·재현성과 보존','',
        '각 stage는 atomic manifest와 입력/출력 SHA, 실제 step/episode 수, 종료코드, traceback을 남긴다. Lock과 자식 process group 관리로 중복 실행을 막고, OOM은 동일 batch/objective로 마지막 검증 checkpoint에서만 제한적으로 재개한다. 정상 worker 산출물은 유지한다. tmux 프로세스는 SSH 종료와 독립적이며 인스턴스 삭제/재생성 복구는 보장하지 않는다.',
        '원본 F1 checkpoint/dataset/source archive의 SHA를 종료 시 재검사했다. Scratch는 인스턴스 파일시스템이며 삭제 시 보존되지 않을 수 있다. 자동 외부 백업/전달 연결은 설정하지 않았으므로 전송 성공을 주장하지 않는다. 보고서 full content는 실행 source의 docs/visual-graph-integration에도 복사한다.',
        '[Verification](verification.json) · [Artifact index](artifact_index.json) · [Completion audit](completion_audit.json)', '']
    # Preserve effect directions without awarding a hypothesis from tiny evidence.
    for r in read_csv(root/'paired_success_analysis.csv'):
        lines.append(f'- {r["contrast"]} / {r["metric"]}: gained{r["gained"]}, lost{r["lost"]}, difference{r["success_count_difference"]}/{r["parents"]}; exact discordant p={float(r["exact_discordant_p"]):.4g}.')
    text='\n'.join(lines);(root/REPORT).write_text(text)
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables'])
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-G</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui}img{max-width:100%}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style>'+body)
    docs=Path(__file__).resolve().parents[2]/'docs/visual-graph-integration';docs.mkdir(parents=True,exist_ok=True)
    # All artifact links in docs resolve to the full artifact root on this host.
    import re
    doc_text=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:']('+os.path.relpath(root/m.group(1),docs)+')',text)
    target=docs/(('miniature_' if cfg['miniature'] else '')+REPORT);target.write_text(doc_text)
    return dict(status='PASS',files=[REPORT,'report.html','hypothesis_assessment.json']+[str(p.relative_to(root)) for p in figs.glob('*.png')],docs_report=str(target))


REQUIRED=['training_protocol_frozen.json','concurrency_benchmark.json','initial_weight_identity.json','sampling_manifest.json',
    'lambda_calibration.json','checkpoint_manifest.json','development_rollouts.csv','model_selection_frozen.json','final_rollouts.csv',
    'functional_success_comparison.csv','paired_success_analysis.csv','phasewise_action_errors.csv','geometry_metrics.csv',
    'paired_displacement_metrics.csv','action_correction_metrics.csv','failure_analysis.csv','source_identity.json','dataset_identity.json',
    'resume_verification.json','worker_status.json',REPORT,'report.html']+[f'{s}_{c}.csv' for s in ('train','validation') for c in 'ABC']


def artifact_verify(root):
    from .verification import checkpoint_verification
    checkpoint_verification(root,write_manifest=False);cfg=read(root/'config.json');e=cfg['evaluation']
    for name in REQUIRED:
        if not (root/name).is_file() or not (root/name).stat().st_size:raise ValueError('required artifact missing '+name)
    for mode,expected in [('development',3*len(e['dev_seeds'])*len(cfg['training']['checkpoint_steps'])),('final',3*len(e['final_seeds']))]:
        rows=read_csv(root/(mode+'_rollouts.csv'))
        if len(rows)!=expected:raise ValueError('episode count incomplete '+mode)
        for r in rows:
            folder=root/r['folder'];trace=read(folder/'trace.json')
            if len(trace)!=int(r['actions']):raise ValueError('trace count')
            if sha(folder/'rollout.mp4')!=r['video_sha256'] or sha(folder/'trace.json')!=r['trace_sha256']:raise ValueError('rollout hash')
            import cv2
            cap=cv2.VideoCapture(str(folder/'rollout.mp4'));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if frames!=len(trace):raise ValueError('video frame readback')
    preservation=read(root/'original_preservation.json')
    for p,h in preservation.items():
        if sha(p)!=h:raise ValueError('F1 source artifact changed')
    # Source must remain byte-identical to the preflight snapshot through execution.
    source=read(root/'source_identity.json')
    for p,h in source['hashes'].items():
        if sha(Path(source['source'])/p)!=h:raise ValueError('executed source changed '+p)
    write(root/'verification.json',dict(status='PASS',required_artifacts=REQUIRED,
        rollouts_readback=True,original_hashes_preserved=True,source_unchanged=True,
        local_macOS_CUDA='NOT_EXECUTED',GPU_PhysX='NOT_EXECUTED',external_delivery='NOT_CONFIGURED'))
    excluded={'pipeline_state.json','stage_manifest.json','worker_status.json','artifact_index.json','completion_audit.json'}
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in excluded or p.suffix in ('.log','.lock','.partial') or 'experiments' in p.parts:continue
        files.append(dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size))
    write(root/'artifact_index.json',dict(at=now(),files=files,volatile_operational_files=sorted(excluded)))
    for r in files:
        if sha(root/r['path'])!=r['sha256']:raise ValueError('artifact readback '+r['path'])
    write(root/'completion_audit.json',dict(status='PASS',at=now(),engineering=cfg['miniature'],
        requirements=REQUIRED,artifacts_verified=len(files),actual_updates_per_condition=cfg['training']['updates'],
        actual_final_episodes=3*len(e['final_seeds']),hypotheses=read(root/'hypothesis_assessment.json')['verdicts'],
        completion='pipeline execution and artifact gates; not hypothesis success'))
    return dict(status='PASS',files=['verification.json','artifact_index.json','completion_audit.json'])

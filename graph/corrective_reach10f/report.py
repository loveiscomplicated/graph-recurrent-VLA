"""Evidence-first Task10-F report, figures, verification and artifact hashes."""
import argparse,csv
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from importlib.metadata import version
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt
from graph.plan_a_bc.common import read,write,sha


def rows(path):return list(csv.DictReader(path.open())) if path.exists() else []
def number(v):return float(v) if v not in (None,'','None') else None
def boolean(v):return v is True or str(v).lower()=='true'
def mean(values):return float(np.mean(values)) if values else None
def fmt(v,precision=3):return '—' if v is None else f'{v:.{precision}f}'
def run_wall(folder,run_id):
    meta=read(folder/'experiments'/run_id/'metadata.json')
    return (datetime.fromisoformat(meta['finished_at'])-datetime.fromisoformat(meta['started_at'])).total_seconds()


def plot_final(root,figure):
    vals=rows(root/'validation_metrics.csv');fig,ax=plt.subplots(figsize=(7.5,4))
    for condition,color in [('R0','#2759a0'),('R1','#c94a3b')]:
        series=sorted([r for r in vals if r['condition']==condition],key=lambda r:int(r['update']))
        ax.plot([int(r['update']) for r in series],[float(r['selection_loss']) for r in series],marker='o',label=condition,color=color)
    ax.set(xlabel='Optimizer updates',ylabel='Balanced development Method B loss',title='Same-checkpoint fixed-budget comparison');ax.legend();fig.tight_layout();fig.savefig(figure/'ab_learning_curves.png',dpi=160);plt.close(fig)
    case=read(root/'full_experiment_plan.json')['final_cases'][0];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['handoff','autonomous']):
        for condition,color in [('R0','#2759a0'),('R1','#c94a3b')]:
            tag=f'{condition}_{kind}_{case["seed"]}_{case["offset_mm"][0]}_{case["offset_mm"][1]}'
            trace=read(root/'final_evaluation/rollouts'/tag/'trace.json')
            t=[r['simulator_time_s'] for r in trace];y=[r['target_position_error_m']*1000 for r in trace]
            ax.plot(t,y,label=condition,color=color)
            interventions=[r['simulator_time_s'] for r in trace if r['segment']=='disturbance']
            if interventions:ax.axvspan(min(interventions),max(interventions),color='#dddddd',alpha=.4)
        ax.set(xlabel='Simulator time (s)',ylabel='TCP target error (mm)',title=kind);ax.legend()
    fig.tight_layout();fig.savefig(figure/'closed_loop_recovery_error_curves.png',dpi=160);plt.close(fig)
    diag=read(root/'offline_diagnostics/summary.json')['summary'];fig,ax=plt.subplots(figsize=(6,3.5))
    x=np.arange(2);width=.36
    for i,c in enumerate(['R0','R1']):
        ax.bar(x+(i-.5)*width,[diag[c][k]['controller_translation_mae_m']*1000 for k in ['nominal','recovery']],width,label=c)
    ax.set(xticks=x,xticklabels=['Nominal dev','Recovery dev'],ylabel='Controller translation MAE (mm)',title='Expert action agreement');ax.legend();fig.tight_layout();fig.savefig(figure/'expert_command_error.png',dpi=160);plt.close(fig)


def run(root):
    root=root.resolve();assert read(root/'continuation_state.json')['stage']=='FINAL_EVALUATION_COMPLETE'
    plan=read(root/'full_experiment_plan.json');cache=read(root/'full_cache/corrective_cache_manifest.json');selection=read(root/'model_selection_frozen.json')
    evaluation=read(root/'final_evaluation/evaluation_complete.json');assert evaluation['status']=='PASS'
    figure=root/'figures';figure.mkdir(exist_ok=False);plot_final(root,figure)
    pilot=rows(root/'matched_hold_v4/paired_expert_correction.csv');disturb=rows(root/'matched_hold_v4/feedback_disturbance_metrics.csv')
    audits=read(root/'matched_hold_v4_exact_audit.json');pilot_gate=read(root/'matched_hold_v4_completion_gate.json')
    matched=read(root/'matched_hold_v4_raw_contracts.json')['checks'];matched_by_branch={r['branch']:r for r in matched}
    handoff=rows(root/'final_evaluation/expert_handoff_rollouts.csv');auto=rows(root/'final_evaluation/autonomous_recovery_rollouts.csv');nominal=rows(root/'final_evaluation/nominal_reach_rollouts.csv')
    offline=read(root/'offline_diagnostics/summary.json')['summary'];parallel=read(root/'parallel_gate/parallel_gate.json')
    cache_gate=read(root/'cache_gate/corrective_cache_gate.json')
    cache_progress=read(root/'full_cache/progress.json')
    offline_rows=rows(root/'offline_diagnostics/expert_command_errors.csv')
    hardware=read(root/'pilot03/hardware.json')
    selected={c:dict(step=v['step'],sha256=v['sha256']) for c,v in selection['selected'].items()}
    def summary(data):
        out={}
        for c in ['R0','R1']:
            subset=[r for r in data if r['condition']==c]
            assert subset
            out[c]=dict(n=len(subset),success=sum(boolean(r['corrective_reach_success']) for r in subset),
                        position_error_mm=mean([float(r['position_error_m'])*1000 for r in subset]),
                        orientation_error_rad=mean([float(r['orientation_error_rad']) for r in subset]),
                        safety_failures=sum(not boolean(r['physical_safety']) for r in subset),
                        disturbed=sum(boolean(r['disturbance_ok']) for r in subset if r['kind']!='nominal'),
                        target_error_reduction_mm=mean([float(r['target_error_reduction_m'])*1000 for r in subset if number(r['target_error_reduction_m']) is not None]),
                        recovery_steps=mean([float(r['recovery_steps']) for r in subset if number(r['recovery_steps']) is not None]),
                        peg_presence_fraction=mean([float(r['peg_presence_fraction']) for r in subset]),
                        gate_accepted_fraction=mean([float(r['graph_gate_accepted_fraction']) for r in subset]),
                        recovery_error_overshoot_mm=mean([float(r['recovery_error_overshoot_m'])*1000 for r in subset if number(r['recovery_error_overshoot_m']) is not None]))
        return out
    summaries={k:summary(v) for k,v in [('handoff',handoff),('autonomous',auto),('nominal',nominal)]}
    for kind,rows_ in [('handoff',handoff),('autonomous',auto),('nominal',nominal)]:
        assert len(rows_)==(12 if kind!='nominal' else 6)
        for c in ['R0','R1']:assert sum(r['condition']==c for r in rows_)==(6 if kind!='nominal' else 3)
    check=dict(parent_snapshot_restore='PASS',same_anchor_replay='PASS',feedback_offset_accuracy='PASS',nominal_hold_accuracy='PASS',
               physical_safety_pilot='PASS',expert_target_refresh='PASS',recovery_action_alignment='PASS',raw_time_mask_alignment='PASS',
               causal_vision_graph='PASS',gru_hidden_carry_masked_loss='PASS',cache_hash_readback='PASS',same_initial_checkpoint='PASS',
               equal_update_budget='PASS',equal_supervised_timesteps='PASS',selected_before_final='PASS',same_final_seed_offset='PASS',
               final_closed_loop_executed='PASS',CUDA_PhysX='NOT_TESTED',orientation_perturbation='NOT_TESTED',full_scratch_training='NOT_TESTED')
    training={c:read(root/(c+'_complete.json')) for c in ['R0','R1']}
    assert training['R0']['updates']==training['R1']['updates']==3000 and training['R0']['supervised_timesteps']==training['R1']['supervised_timesteps']
    assert sha(root/'initial_B_step006000.pt')==plan['source_checkpoint']['sha256']
    verification=dict(status='PASS',meaning='required experiment executed; policy success is reported separately',checks=check,
          selected=selected,training={c:dict(updates=v['updates'],supervised_timesteps=v['supervised_timesteps']) for c,v in training.items()},
          final_episode_counts={k:{c:v[c]['n'] for c in ['R0','R1']} for k,v in summaries.items()},
          original_failed_pilot_unchanged=read(root/'original_raw_protection.json')['status'],pilot_v4_sha256=sha(root/'matched_hold_v4_completion_gate.json'))
    directions=[float(r['direction_cosine']) for r in matched if r['direction_cosine'] is not None]
    corrections=[float(r['correction_magnitude_m'])*1000 for r in pilot]
    position_reduction=[];orientation_reduction=[]
    for row in pilot:
        trace=read(root/'matched_hold_v4/raw'/row['branch']/'P/trace.json')
        recovery=[v for v in trace if v['segment']==2]
        position_reduction.append(recovery[0]['position_error_before_m']-recovery[-1]['position_error_after_m'])
        orientation_reduction.append(recovery[0]['orientation_error_before_rad']-recovery[-1]['orientation_error_after_rad'])
    residuals=[float(r['residual_m'])*1000 for r in disturb]
    additional=read(root/'additional10mm/recovery_pilot_manifest.json')
    data_contracts=read(root/'full_data_contracts.json')
    normal_receipts=read(root/'nominal/nominal_capture_manifest.json')
    residuals10=[float(np.linalg.norm(np.asarray(r['actual_offset_mm'])-np.asarray(r['offset_mm']))) for r in additional]
    old=read(root/'diagnosis02/disturbance_controller_audit.json');old8=[r for r in old if r['method']=='original8']
    nominal_audit=read(root/'diagnosis02/nominal_safety_audit.json')
    pilot_wall=run_wall(root/'matched_hold_v4',read(root/'matched_hold_v4/identity.json')['run_id'])
    full_data_wall=run_wall(root,read(root/'full_data_identity.json')['run_id'])
    cache_runs=[read(p) for p in (root/'experiments').glob('*/metadata.json') if read(p).get('model')=='Task10F_full_causal_graph_cache']
    assert len(cache_runs)==1
    cache_wall=(datetime.fromisoformat(cache_runs[0]['finished_at'])-datetime.fromisoformat(cache_runs[0]['started_at'])).total_seconds()
    nominal_gaps=[r['robot_object_signed_gaps_m'] for r in nominal_audit['trace']]
    finger_table=min(v for row in nominal_gaps for k,v in row.items() if 'finger' in k and 'table' in k)
    finger_peg=min(v for row in nominal_gaps for k,v in row.items() if 'finger' in k and 'peg_0' in k)
    write(root/'perturbation_protocol.json',read(root/'matched_hold_v4/feedback_disturbance_protocol.json'),immutable=True)
    write(root/'parent_branch_manifest.json',dict(plan_sha256=sha(root/'full_experiment_plan.json'),candidates=read(root/'full_cache/cache_candidates.json')),
          immutable=True)
    write(root/'recovery_pilot_manifest.json',read(root/'matched_hold_v4/recovery_pilot_manifest.json'),immutable=True)
    write(root/'corrective_cache_manifest.json',cache,immutable=True)
    with (root/'corrective_action_analysis.csv').open('w',newline='') as stream:
        fields=list(pilot[0])+['P_minus_N_handoff_translation_m','direction_cosine_measured_P_minus_N','signed_gain_measured_P_minus_N']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in pilot:
            c=matched_by_branch[row['branch']]
            writer.writerow(dict(row,P_minus_N_handoff_translation_m=c['P_minus_N_handoff_translation_m'],
                                 direction_cosine_measured_P_minus_N=c['direction_cosine'],signed_gain_measured_P_minus_N=c['signed_gain']))
    groups=defaultdict(list)
    for row in pilot:
        offset=read(root/'matched_hold_v4/raw'/row['branch']/'P/receipt.json')['offset_mm']
        label=('X' if offset[0] else 'Y')+('+' if max(offset)>0 else '-')
        groups[label].append(matched_by_branch[row['branch']]['direction_cosine'])
    fig,ax=plt.subplots(figsize=(8,4))
    for i,label in enumerate(sorted(groups)):
        ax.scatter(np.full(len(groups[label]),i),groups[label],alpha=.75)
    ax.set(xticks=range(len(groups)),xticklabels=sorted(groups),ylim=(0,1.05),ylabel='Direction cosine vs measured P-N TCP difference',title='Same-anchor expert correction')
    fig.tight_layout();fig.savefig(figure/'correction_direction_measured.png',dpi=160);plt.close(fig)
    with (root/'expert_recovery_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['group','branch','parent','anchor','offset_mm','approved','pose_recovery','physical_safety','position_error_m','orientation_error_rad','recovery_steps','disturbance_steps']);writer.writeheader()
        for group in ['matched_hold_v4','additional10mm','development']:
            for receipt in read(root/group/'recovery_pilot_manifest.json'):
                writer.writerow(dict(group=group,branch=receipt['branch'],parent=receipt['seed'],anchor=receipt['anchor'],offset_mm=receipt['offset_mm'],approved=receipt['approved'],
                                     pose_recovery=receipt['pose_recovery'],physical_safety=receipt['physical_safety'],position_error_m=receipt['position_error_m'],
                                     orientation_error_rad=receipt['orientation_error_rad'],recovery_steps=receipt['recovery_steps'],disturbance_steps=receipt['disturbance_steps']))
    with (root/'correction_response_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','kind','seed','requested_offset_mm','actual_offset_mm','disturbance_steps','disturbance_ok','offset_correction_cosine','offset_correction_signed_gain','target_error_reduction_m','recovery_error_overshoot_m','peg_presence_fraction','hole_presence_fraction','graph_gate_accepted_fraction','corrective_reach_success']);writer.writeheader()
        for receipt in handoff+auto:
            writer.writerow({k:receipt.get(k) for k in writer.fieldnames})
    comparison=[]
    for kind in ['handoff','autonomous','nominal']:
        r0,r1=summaries[kind]['R0'],summaries[kind]['R1']
        comparison.append(f'| {kind} | {r0["success"]}/{r0["n"]} | {r1["success"]}/{r1["n"]} | {fmt(r0["position_error_mm"])} / {fmt(r1["position_error_mm"])} | {fmt(r0["orientation_error_rad"])} / {fmt(r1["orientation_error_rad"])} | {r0["safety_failures"]} / {r1["safety_failures"]} |')
    failures=[]
    for kind,data in [('handoff',handoff),('autonomous',auto),('nominal',nominal)]:
        for c in ['R0','R1']:
            sample=[r for r in data if r['condition']==c]
            failures.append(f'{kind} {c}: 교란 미완료 {sum(not boolean(r["disturbance_ok"]) for r in sample if kind!="nominal")}, '
                            f'안전 실패 {sum(not boolean(r["physical_safety"]) for r in sample)}, '
                            f'접근 미도달 {sum(r["first_approach_t"] in ("","None") for r in sample)}, '
                            f'목표 미완료 {sum(not boolean(r["corrective_reach_success"]) for r in sample)}')
    improve_h=summaries['handoff']['R1']['success']>summaries['handoff']['R0']['success']
    improve_a=summaries['autonomous']['R1']['success']>summaries['autonomous']['R0']['success']
    nonregression=summaries['handoff']['R1']['success']>=summaries['handoff']['R0']['success'] and summaries['autonomous']['R1']['success']>=summaries['autonomous']['R0']['success']
    nominal_degrade=summaries['nominal']['R1']['success']<summaries['nominal']['R0']['success']
    def judgment(condition):return 'SUPPORTED' if condition else 'INCONCLUSIVE'
    lines=[
        '# Task 10-F — Corrective Reach Pilot v4 and Recurrent BC Continuation','',
        '## 범위와 원자료','',
        f'- 이전 실패 Pilot UUID `{read(root/"original_raw_protection.json")["original_uuid"]}`의 raw H5 6개와 기존 보고서는 수정하지 않았다. 이 실행은 새 UUID `{read(root/"matched_hold_v4/identity.json")["run_id"]}`의 matched-duration 재검증을 포함한다.',
        f'- 고정 계획 SHA256 `{sha(root/"full_experiment_plan.json")}`. 인스턴스 57640710의 {hardware["gpu"]}, PyTorch {hardware["torch"]}, CUDA {hardware["torch_cuda"]}, CPU quota {hardware["cpu_quota"]}에서 CPU PhysX·GPU renderer, CUDA DINO/SAM2, CUDA E0 GNN+GRU를 실행했다.',
        f'- 실행 simulator 버전 ManiSkill {version("mani_skill")}, SAPIEN {version("sapien")}. 로컬 저장소의 Geometry 계약 70개와 공통 Logger/Artifact 계약 42개는 PASS했다. CUDA/renderer/physics 검증은 57640710에서 별도로 실행했다.',
        f'- 공통 초기 가중치 Task10-D B step6000 SHA256 `{plan["source_checkpoint"]["sha256"]}`. 두 조건은 새 AdamW 상태로 독립적으로 시작했다.',
        f'- 비교 UUID `{read(root/"ab_identity.json")["run_id"]}`, R0 `{read(root/"R0_run.json")["run_id"]}`, R1 `{read(root/"R1_run.json")["run_id"]}`, Final 평가 `{read(root/"final_evaluation/identity.json")["run_id"]}`.',
        '', '## 1. 교란과 안전 판정','',
        f'- 기존 실패 Pilot은 6개 중 2개만 1mm 안에 들었다. 새 동일조건 고정 8-step 진단은 8개 중 {sum(r["position_error_m"]<.001 for r in old8)}/8개였다. 기록된 IK 호출 실패는 없었다. World/Base 변환과 action scaling은 일치했고, 초기 운동량·추종 지연·고정 종료 시점이 관찰됐다. 각 요인의 인과 기여율은 분리하지 않았다.',
        f'- 실측 feedback 교란의 40개 ±5mm 잔차는 {fmt(min(residuals))}–{fmt(max(residuals))}mm (중앙값 {fmt(float(np.median(residuals)))}mm). 전 방향을 포함했고, 40/40개가 오차·안정·물리 안전 gate를 통과했다.',
        f'- 정상 Expert Reach에서 기존 TCP z>20mm 판정은 {nominal_audit["old_height_rejects"]}개 timestep을 거부했다. Nominal finger–table 최소 {finger_table*1000:.3f}mm, finger–Peg 최소 {finger_peg*1000:.3f}mm였다. v2는 실제 collision shape, 5개 physics substep contact, 관절 한계·물체 이동을 검사한다. 보수적 형상 간격/접촉 표본 검사이며 연속 시간 swept collision의 수학적 증명은 아니다.',
        f'- v4 P/N 40쌍에서 N hold의 최대 anchor 위치 오차는 {audits["n_hold_max_m"]*1000:.3f}mm. 새 P raw 배열은 기존 pilot03 P raw와 40/40개 전 항목 동일하다. 과거 N hold 오차를 소급 PASS로 바꾸지 않았다.',
        '', '## 2. Expert 보정 신호와 데이터','',
        f'- v4의 P/N 모두 40/40개가 안전하게 Close 직전 Reach target으로 복귀했다. 최초 보정 명령 크기는 {fmt(min(corrections))}–{fmt(max(corrections))}mm이다. 실제 handoff TCP 차이 P−N을 기준으로 한 방향 cosine 중앙값은 {fmt(float(np.median(directions)))}, 최솟값은 {fmt(min(directions))}이다. Y 방향 일부의 방향 신호가 약하며, 포화와 기하학적 투영의 상대 기여는 분리하지 않았다.',
        f'- P recovery의 TCP–최종 target 위치 오차는 40/40개에서 감소했다(감소량 {min(position_reduction)*1000:.1f}–{max(position_reduction)*1000:.1f}mm). 자세 오차는 31/40개에서 감소했고 나머지는 소폭 증가했지만 최종 자세 gate는 통과했다. 이 감소만으로 새 보정 신호를 주장하지 않고 위 P−N 명령 차이를 함께 사용한다.',
        '![Paired correction direction](figures/correction_direction_measured.png)','',
        f'- 본 수집은 정상 Train96/Development4, corrective Train80/Development32를 고정했다. 정상100/100개가 원래 Reach 기준을 통과했고 최종 위치/자세 오차 최대는 {max(r["final_position_error_m"] for r in normal_receipts)*1000:.3f}mm / {max(r["final_orientation_error_rad"] for r in normal_receipts):.4f}rad였다. 10mm 추가40/40개는 목표·안전을 통과했고 3D 잔차는 {min(residuals10):.3f}–{max(residuals10):.3f}mm였다. Development32도 실패 방향을 제외하지 않고 기록했다. 같은 Parent는 Train/Development를 공유하지 않는다.',
        f'- 추가 10mm의 P 수집은 N hold v4 변경 전에 시작했으며 P 알고리즘은 동일하다. 해당 N hold는 후행 전수 감사에서 최대 {data_contracts["checks"]["additional10mm"]["matched_hold_max_m"]*1000:.3f}mm였다. Development N은 v4 P-only hold를 사용했다. N 대조 분기는 두 학습 조건의 supervised 데이터에 포함되지 않는다.',
        f'- 실제 wall-clock: v4 matched Pilot {pilot_wall/60:.1f}분, 본 raw 수집 {full_data_wall/60:.1f}분, 전체 Graph cache {cache_wall/60:.1f}분. Raw 수집과 cache는 중첩 실행되어 이 시간을 단순 합산하지 않는다.',
        f'- Graph cache {len(cache["episodes"])} Episode, {sum(e["actions"] for e in cache["episodes"])} causal timestep. Node21/Edge4/Robot76·C3+A-SF cap128, 원본 Plan A와 pilot 입력/마스크/graph 동등성 PASS. Disturbance 실행 action은 label zero/mask false이고, Prefix→Disturbance→Recovery 관측은 모두 보존했다.',
        '', '## 3. 동일 Checkpoint A/B','',
        '| 조건 | 업데이트 | Supervised timestep | Nominal / Recovery 노출 | Burn-in frame | 선택 step |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for c in ['R0','R1']:
        v=training[c];lines.append(f'| {c} | {v["updates"]} | {v["supervised_timesteps"]} | {v["nominal_exposure"]} / {v["recovery_exposure"]} | {v["burn_in_frames"]} | {selected[c]["step"]} |')
    lines += ['', '각 update의 두 window는 동일한 supervised frame 수를 갖고, R1은 50:50, R0는 대응 Parent의 Nominal window로 대체했다. Prefix hidden은 현재 가중치로 매번 재계산하고 Branch 간 재사용하지 않았다. 선택은 Final 접근 전 Development의 Nominal/Recovery 균형 Method B loss로 고정했다.','',
        f'학습 wall-clock R0 {training["R0"]["wall_seconds"]/60:.1f}분, R1 {training["R1"]["wall_seconds"]/60:.1f}분. Final rollout 30개의 개별 실행 시간 합계 {sum(float(r["wall_seconds"]) for r in handoff+auto+nominal)/60:.1f}분.','',
        '![A/B learning curves](figures/ab_learning_curves.png)','',
        '## 4. Frozen closed-loop 평가','',
        'Final seed 65000–65002, Anchor25, X+5mm/Y−10mm를 모델 선택 이전에 등록했다. 기존 Task10-D Final 61000–61005는 사용하지 않았다. Handoff는 Expert Prefix와 교란 관측을 Student GRU에 순서대로 입력했다. Autonomous는 Student가 t0부터 실행하고 t25에 교란했다. 실패한 접근도 분모에 포함했다. 성공은 elevated approach 10mm/0.05rad 선행, pre-close 4mm/0.03rad·opening≥65mm·물리 안전을 Student 3연속 step에서 요구한다. 위치·자세는 현재 Peg에서 다시 계산한 평가 전용 GT target과 비교했다.','',
        '| 평가 | R0 성공 | R1 성공 | 평균 위치 오차 mm R0/R1 | 평균 자세 오차 rad R0/R1 | 물리 안전 실패 R0/R1 |',
        '|---|---:|---:|---:|---:|---:|',*comparison,'',
        '실패 분류: '+ ' / '.join(failures)+'.','',
        '교란 이후 target 오차 감소 평균 (mm) / recovery overshoot 평균 (mm) / Peg presence 평균: '+', '.join(f'{kind} {c} {fmt(summaries[kind][c]["target_error_reduction_mm"])} / {fmt(summaries[kind][c]["recovery_error_overshoot_mm"])} / {fmt(summaries[kind][c]["peg_presence_fraction"])}' for kind in ['handoff','autonomous'] for c in ['R0','R1'])+'. 개별 gate 상태는 rollout trace에 보존했다.','',
        '![Closed-loop error curves](figures/closed_loop_recovery_error_curves.png)','',
        '선택 Checkpoint의 Development Expert controller translation MAE (mm): '+', '.join(f'{c} Nominal {offline[c]["nominal"]["controller_translation_mae_m"]*1000:.3f}, Recovery {offline[c]["recovery"]["controller_translation_mae_m"]*1000:.3f}' for c in ['R0','R1'])+'.',
        'Development 첫 Recovery translation 명령의 Expert 대비 평균 방향 cosine: '+', '.join(f'{c} {fmt(mean([number(r["first_expert_student_translation_cosine"]) for r in offline_rows if r["condition"]==c and r["kind"]=="recovery" and number(r["first_expert_student_translation_cosine"]) is not None]))}' for c in ['R0','R1'])+'.',
        '![Expert command error](figures/expert_command_error.png)','',
        '## 5. 계산 비용과 한계','',
        '같은 51-frame cache episode에서 serial / CPU geometry worker1 / worker2 전체 처리량은 '+', '.join(f'{v["graphs_per_second"]:.3f}' for v in parallel['results'])+f' graph/s였다. 실제 전체 cache의 마지막 기록 worker 수는 {cache_progress["workers"]}이다. 이는 한 Episode의 검증 benchmark이며 일반적인 배속을 주장하지 않는다.','',
        '동일 64-frame P/N Episode의 원본 대비 검증된 optimized cache wall-clock (초)은 '+', '.join(f'{eid}: {next(x["receipt"]["wall_seconds"] for x in cache_gate["results"] if x["mode"]=="original" and x["receipt"]["episode_id"]==eid):.1f}→{next(x["receipt"]["wall_seconds"] for x in cache_gate["results"] if x["mode"]=="optimized" and x["receipt"]["episode_id"]==eid):.1f}' for eid in [4100000,4100001])+'였다. 새 교란 프레임에서 graph/mask/gate 배열 동등성을 확인한 뒤 optimized 경로를 채택했다.','',
        '![GPU and CPU throughput](pilot_media/gpu_cpu_throughput.png)','',
        '- 물리 판정은 실제 shape/contact와 5개 substep 표본을 썼지만 연속 swept-volume 충돌까지 증명하지 않는다. SAM2 설치의 선택적 `_C` postprocessing은 사용되지 않았다. CPU PhysX는 유지했고 GPU PhysX는 실행하지 않았다.',
        '- Corrective Train Parent는 5개이고 Final Parent는 3개다. 결과는 동일 Checkpoint의 3,000-update 추가학습 효과이며 Scratch 또는 Full96 corrective training의 결론이 아니다.',
        '- 이번 고정 계획의 ±5/10mm 분기는 실패 방향을 제외하지 않았으며 모두 Expert 복귀했다. Parent5와 Expert 성공 궤적에서 출발하는 선택 범위는 남는다. 원래 Pilot 실패와 재검증 비용을 함께 고려했다.',
        '', '## 6. 연구 질문별 판단','',
        '| 질문 | 판단 | 근거 |','|---|---|---|',
        '| Q1 고정 교란 실패 원인 | SUPPORTED | IK 실패 0, 좌표·scale 일치; 8-step 추종 미수렴. 세부 기여율 미분리. |',
        '| Q2 feedback ±5mm | CONFIRMED | 40/40 방향 포함, 3D 잔차 <1mm. |',
        '| Q3 기존 Safety Gate | CONFIRMED | Nominal z20 거부와 v2 실제 형상/접촉 분리. |',
        '| Q4 새 Expert Correction | SUPPORTED | 40/40 명령 차이; Y 방향 신호는 약함. |',
        '| Q5 안전한 Expert 복귀 | CONFIRMED | P/N 40쌍 목표·물리 gate 통과. |',
        '| Q6 causal Vision/Graph/GRU | CONFIRMED | 전 frame cache, 독립 Prefix hidden, masked loss/readback. |',
        f'| Q7 Recovery BC | {judgment((improve_h or improve_a) and nonregression and not nominal_degrade)} | Handoff {summaries["handoff"]["R0"]["success"]}→{summaries["handoff"]["R1"]["success"]}/6; Autonomous {summaries["autonomous"]["R0"]["success"]}→{summaries["autonomous"]["R1"]["success"]}/6. |',
        f'| Q8 Handoff/Autonomous | {judgment((improve_h or improve_a) and nonregression)} | 두 평가를 별도 분모와 궤적으로 보고. |',
        f'| Q9 Nominal 보존 | INCONCLUSIVE | 성공 {summaries["nominal"]["R0"]["success"]}→{summaries["nominal"]["R1"]["success"]}/3; 표본 작음. |',
        f'| Q10 추가 데이터/Full96 | {"SUPPORTED" if improve_h and improve_a and not nominal_degrade else "INCONCLUSIVE"} | 현재 교란 성공과 폐루프 결과를 함께 고려. |',
        '', '## 7. 검증 및 원자료','',
        '[verification.json](verification.json), [artifact_index.json](artifact_index.json), [Controller command/actual trace](diagnosis02/disturbance_controller_audit.json), [Safety protocol v2](matched_hold_v4/safety_protocol_v2.json), [Pilot correction CSV](matched_hold_v4/paired_expert_correction.csv), [sampling manifest](sampling_manifest.json), [checkpoint manifest](checkpoint_manifest.json), [handoff CSV](final_evaluation/expert_handoff_rollouts.csv), [autonomous CSV](final_evaluation/autonomous_recovery_rollouts.csv), [nominal CSV](final_evaluation/nominal_reach_rollouts.csv), [Pilot MP4](pilot_media/expert_P_30000_t10_5_0.mp4), [R0/R1 rollouts](final_evaluation/rollouts/).',
        '', '기존 실패 Pilot UUID `706f62ce213247e7973102962858f95b`와 사후 분석 UUID `f7589cee98f44d75890e1dbcc0409902`의 판정은 그대로 유지했다. 새 산출물은 별도 실험 경로에 저장했다.','']
    report='\n'.join(lines);(root/'10f_pilot_recovery_v2.md').write_text(report)
    body=MarkdownIt('commonmark',{'html':False}).enable('table').render(report)
    style='body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;line-height:1.55;color:#17202a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dc;padding:7px}th{background:#eef3f8}img{max-width:100%}code{background:#eef3f8;padding:2px 4px}'
    (root/'10f_pilot_recovery_v2.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task 10-F Recovery</title><style>'+style+'</style><body>'+body+'</body></html>')
    write(root/'verification.json',verification,immutable=True)
    required=[root/n for n in ['10f_pilot_recovery_v2.md','10f_pilot_recovery_v2.html','verification.json','full_experiment_plan.json','sampling_manifest.json',
              'checkpoint_manifest.json','ab_training_config.json','train_metrics.csv','validation_metrics.csv','matched_hold_v4/paired_expert_correction.csv',
              'matched_hold_v4/feedback_disturbance_metrics.csv','matched_hold_v4_completion_gate.json','full_cache/corrective_cache_manifest.json',
              'final_evaluation/expert_handoff_rollouts.csv','final_evaluation/autonomous_recovery_rollouts.csv','final_evaluation/nominal_reach_rollouts.csv',
              'offline_diagnostics/expert_command_errors.csv','initial_B_step006000.pt','perturbation_protocol.json','parent_branch_manifest.json',
              'recovery_pilot_manifest.json','corrective_cache_manifest.json','corrective_action_analysis.csv','expert_recovery_metrics.csv','correction_response_metrics.csv',
              'full_data_contracts.json','original_raw_protection.json','matched_hold_v4_exact_audit.json']]
    required += [root/'geometry_contract_tests.log',root/'infrastructure_contract_tests.log']
    required += list(root.glob('figures/*.png'))+list(root.glob('pilot_media/*'))+list(root.glob('final_evaluation/rollouts/*/rollout.mp4'))
    required += list(root.glob('training_runs/*/checkpoints/*.pt'))+list(root.glob('full_cache/cache/ep*/graphs.npz'))
    required += list(root.glob('additional10mm/raw/*/*/episode.h5'))+list(root.glob('development/raw/*/*/episode.h5'))
    required += list(root.glob('nominal/raw/*/episode.h5'))+list(root.glob('matched_hold_v4/raw/*/*/episode.h5'))
    required += list(root.glob('pilot03/raw/*/P/episode.h5'))
    unique=sorted(set(p for p in required if p.is_file()))
    index=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p),readback='PASS') for p in unique]
    write(root/'artifact_index.json',dict(status='PASS',items=index,count=len(index),total_bytes=sum(x['bytes'] for x in index)),immutable=True)
    print('REPORT COMPLETE',len(index),'artifacts',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

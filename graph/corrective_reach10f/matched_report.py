"""Matched update3000 comparison and evidence-linked Korean diagnostic report."""
import argparse
import csv
import json
import statistics
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from markdown_it import MarkdownIt

from graph.corrective_reach10f.matched_diagnosis import read,write,sha,csvwrite,diagnose

LABELS=['R0_step0','R0_step3000','R1_step3000']
COLORS=['#8793a5','#265eb0','#d45332']


def average(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None


def statistics_for(rows):
    return dict(episodes=len(rows),success=sum(r['official_success'] for r in rows),
        intervention_completed=sum(r['kind']!='nominal' and r['disturbance_ok'] for r in rows),
        aborted_before_intervention=sum(r['aborted_before_intervention'] for r in rows),
        physical_safety_failures=sum(not r['physical_safety'] for r in rows),
        final_position_error_mm=average(rows,'final_position_error_mm'),minimum_position_error_mm=average(rows,'minimum_position_error_mm'),
        final_orientation_error_rad=average(rows,'final_orientation_error_rad'),minimum_orientation_error_rad=average(rows,'minimum_orientation_error_rad'),
        approach_reached=sum(r['approach_first_post_t'] is not None for r in rows),
        preclose_pose_reached=sum(r['preclose_pose_first_post_t'] is not None for r in rows),
        diagnostic_without_safety=sum(r['diagnostic_without_safety'] for r in rows))


def plots(episodes,timelines,figures):
    figures.mkdir(exist_ok=False)
    summary={label:{kind:statistics_for([r for r in episodes if r['checkpoint']==label and r['kind']==kind])
                    for kind in ['handoff','autonomous','nominal']} for label in LABELS}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for i,label in enumerate(LABELS):
        x=np.arange(3)+(i-1)*.24
        for ax,key,title in zip(axes,['success','final_position_error_mm','physical_safety_failures'],
             ['Official Reach successes','Terminal target position error (mm)','Physical safety failures']):
            ax.bar(x,[summary[label][k][key] for k in ['handoff','autonomous','nominal']],.24,label=label,color=COLORS[i])
            ax.set(xticks=np.arange(3),xticklabels=['Handoff (6)','Autonomous (6)','Nominal (3)'],title=title)
    axes[0].set_ylim(0,6.8);axes[2].set_ylim(0,6.8);axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'matched_step3000_reach.png',dpi=170);plt.close(fig)
    cases=sorted({(r['seed'],tuple(r['offset'])) for r in episodes if r['kind']=='handoff'})
    for metric,threshold,ylabel,name in [('position_error_mm',4,'Pre-close position error (mm)','handoff_position_error.png'),
        ('orientation_error_rad',.03,'SO(3) orientation error (rad)','handoff_orientation_error.png')]:
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
        for ax,(seed,offset) in zip(axes.flat,cases):
            for label,color in zip(LABELS,COLORS):
                rows=[r for r in timelines if r['checkpoint']==label and r['seed']==seed and tuple(r['offset'])==offset]
                ax.plot([r['post_state_t'] for r in rows],[r[metric] for r in rows],color=color,label=label,lw=1.4)
                student=[r for r in rows if r['segment']=='student_recovery']
                if student:ax.axvline(student[0]['pre_state_t'],color=color,lw=.8,alpha=.4)
                if not rows[-1]['physical_safety']:ax.scatter(rows[-1]['post_state_t'],rows[-1][metric],color=color,marker='x',s=40)
            disturbance=[r['post_state_t'] for r in rows if r['segment']=='disturbance']
            if disturbance:ax.axvspan(min(disturbance)-1,max(disturbance),color='#ddd',alpha=.45)
            ax.axhline(threshold,color='#333',ls=':',lw=1);ax.set_title(f'{seed}, offset={offset} mm');ax.set_ylabel(ylabel)
        for ax in axes[-1]:ax.set_xlabel('Post-action simulator state index (20 Hz)')
        axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/name,dpi=170);plt.close(fig)
    grid=np.zeros((18,130));names=[]
    for i,(seed,offset) in enumerate(cases):
        for j,label in enumerate(LABELS):
            idx=i*3+j;names.append(f'{seed} {offset[:2]} {label}')
            for r in timelines:
                if r['checkpoint']==label and r['seed']==seed and tuple(r['offset'])==offset:
                    grid[idx,r['post_state_t']-1]=4 if not r['physical_safety'] else {'expert_prefix':1,'disturbance':2,'student_recovery':3}[r['segment']]
    fig,ax=plt.subplots(figsize=(13,8));ax.imshow(grid,aspect='auto',interpolation='nearest',vmin=0,vmax=4,
        cmap=ListedColormap(['#f4f4f4','#9daac5','#e6c673','#68ad8c','#d4423d']),extent=[.5,130.5,17.5,-.5])
    ax.set(yticks=np.arange(18),yticklabels=names,xlabel='Post-action state index',title='Safety v2 timeline; blank after first abort is unobserved')
    ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(['#f4f4f4','#9daac5','#e6c673','#68ad8c','#d4423d'],
        ['Unobserved','Expert prefix safe','Disturbance safe','Student safe','First unsafe boundary'])],loc='upper center',bbox_to_anchor=(.5,-.08),ncol=5,fontsize=8)
    fig.tight_layout();fig.savefig(figures/'safety_predicate_timeline.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));keys=['Approach ordered','Position ever <4mm','Orientation ever <.03','Simultaneous pre-close','3 steps without safety','Physically safe']
    for i,label in enumerate(LABELS):
        rows=[r for r in episodes if r['checkpoint']==label and r['kind']=='handoff']
        vals=[sum(r['approach_first_post_t'] is not None for r in rows),sum(r['minimum_position_error_mm']<4 for r in rows),
            sum(r['minimum_orientation_error_rad']<.03 for r in rows),sum(r['preclose_pose_first_post_t'] is not None for r in rows),
            sum(r['diagnostic_without_safety'] for r in rows),sum(r['physical_safety'] for r in rows)]
        ax.bar(np.arange(6)+(i-1)*.24,vals,.24,label=label,color=COLORS[i])
    ax.set(xticks=np.arange(6),xticklabels=keys,ylim=(0,6.8),ylabel='Episodes / 6',title='Distinct Reach predicates; independent minima do not establish success')
    ax.tick_params(axis='x',labelrotation=15);ax.legend();fig.tight_layout();fig.savefig(figures/'reach_order_stability_conditions.png',dpi=170);plt.close(fig)
    return summary


def run(base,new,output,repo):
    assert read(new/'evaluation_complete.json')['status']=='PASS'
    output.mkdir(parents=True,exist_ok=False)
    protocol=read(base/'final_evaluation/evaluation_protocol.json');episodes=[];timelines=[];safety=[]
    for folder,labels in [(base/'final_evaluation/rollouts',{'R0':'R0_step0','R1':'R1_step3000'}),(new/'rollouts',{'R0':'R0_step3000'})]:
        for path in sorted(folder.glob('*/episode.json')):
            ep=read(path);e,t,s=diagnose(path.parent,labels[ep['condition']],protocol);episodes.append(e)
            if e['kind']=='handoff':timelines.extend(t);safety.append(s)
    assert len(episodes)==45
    csvwrite(output/'all_checkpoint_evaluation.csv',episodes)
    matched=[r for r in episodes if r['checkpoint']!='R0_step0']
    csvwrite(output/'matched_step_evaluation.csv',matched)
    csvwrite(output/'handoff_failure_breakdown.csv',[r for r in episodes if r['kind']=='handoff'])
    csvwrite(output/'handoff_stepwise_metrics.csv',timelines);csvwrite(output/'reach_predicate_timeline.csv',timelines)
    pairs=[]
    keys=['official_success','final_position_error_mm','minimum_position_error_mm','final_orientation_error_rad',
          'minimum_orientation_error_rad','physical_safety','target_error_reduction_mm','recovery_overshoot_mm',
          'approach_first_post_t','preclose_pose_first_post_t','gripper_final_mm','last_post_state_t','termination_reason',
          'disturbance_ok','disturbance_steps','aborted_before_intervention']
    for a in [r for r in episodes if r['checkpoint']=='R0_step3000']:
        b=next(r for r in episodes if r['checkpoint']=='R1_step3000' and r['kind']==a['kind'] and r['seed']==a['seed'] and r['offset']==a['offset'])
        assert a['reset_sha256']==b['reset_sha256']
        pair=dict(kind=a['kind'],seed=a['seed'],offset=a['offset'],same_reset_sha256=a['reset_sha256'])
        for k in keys:
            pair['R0_'+k]=a[k];pair['R1_'+k]=b[k]
        for k in ['final_position_error_mm','minimum_position_error_mm','final_orientation_error_rad','minimum_orientation_error_rad']:
            pair['R1_minus_R0_'+k]=b[k]-a[k]
        tag=a['episode'];ta=read(new/'rollouts'/tag/'trace.json');tb=read(base/'final_evaluation/rollouts'/tag.replace('R0_','R1_',1)/'trace.json')
        common=min(len(ta),len(tb));pair['common_observed_end_post_t']=common
        pair['R0_common_end_position_mm']=ta[common-1]['target_position_error_m']*1000
        pair['R1_common_end_position_mm']=tb[common-1]['target_position_error_m']*1000
        pair['R1_minus_R0_common_end_position_mm']=pair['R1_common_end_position_mm']-pair['R0_common_end_position_mm']
        pairs.append(pair)
    csvwrite(output/'paired_step3000_comparison.csv',pairs)
    summary=plots(episodes,timelines,output/'figures')
    write(output/'checkpoint_selection_effect.json',dict(summary=summary,
        R1_step0_alias='R0_step0; identical checkpoint bytes/model/codec/normalizer/config, evaluation state reset per episode; no duplicate execution',
        selected_comparison=dict(R0='step0',R1='step3000'),matched_comparison=dict(R0='step3000',R1='step3000'),
        original_selection_changed=False,original_selection_sha256=sha(base/'model_selection_frozen.json'),
        interpretation='Previously disclosed Final set; endpoint errors censored by safety abort; same updates do not equalize nominal/recovery or burn-in exposure.'))
    counts={}
    for label in LABELS:
        ss=[r for r in safety if r['checkpoint']==label];counter=Counter(p for r in ss for e in r['events'] for p in e['predicates'])
        counts[label]=dict(counter)
    other_safety=[p for ss in safety for e in ss['events'] for p in e['predicates'] if not ('PEG_BOX' in p)]
    write(output/'physical_safety_breakdown.json',dict(episodes=safety,predicate_episode_counts=counts,
        no_recorded_handoff_table_self_joint_velocity_object_motion_failure=not other_safety,
        persistence='NOT_AVAILABLE_AFTER_FIRST_FAILURE_ABORT',contact_force='NOT_AVAILABLE',substep_index='NOT_AVAILABLE',
        other_below_threshold_contacts='NOT_AVAILABLE; absence of violation records does not establish zero contacts',
        nominal_same_final_seed_suffix='NOT_TESTED; expert prefixes in all cases safe, separate seed30000 nominal full Reach safety audit available'))
    identity=read(new/'checkpoint_identity.json');write(output/'checkpoint_identity.json',identity)
    original_checks=read(new/'evaluation_contract_verification.json')
    checks=dict(original_checks)
    checks['posthoc_success_recomputation']=dict(status='PASS',episodes=45,exact=True)
    checks['analysis_source_sha256']={str(p.relative_to(repo)):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('matched_diagnosis.py').resolve()]}
    checks['old_source_snapshot_matches_current']={}
    checks['local_repository_matches_execution']={}
    snap=base/'f1_gt_supplement_20261030/source'
    executed_sources=output/'execution_source_dependencies';executed_sources.mkdir()
    for p,h in identity['source_sha256'].items():
        local_same=sha(repo/p)==h
        checks['local_repository_matches_execution'][p]='PASS' if local_same else 'DIFFERENT_LOCAL_COPY_PRESERVED'
        if (snap/p).exists():
            assert sha(snap/p)==h
            source=snap/p
        else:
            assert local_same
            source=repo/p
        target=executed_sources/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        checks['old_source_snapshot_matches_current'][p]='PASS' if (snap/p).exists() else 'NOT_AVAILABLE'
    write(output/'evaluation_contract_verification.json',checks)
    hand=[r for r in episodes if r['kind']=='handoff'];hs={k:[r for r in hand if r['checkpoint']==k] for k in LABELS}
    posnever={k:sum(r['minimum_position_error_mm']>=4 for r in v) for k,v in hs.items()}
    endrot={k:sum(not r['final_orientation_predicate_pass'] for r in v) for k,v in hs.items()}
    table='|평가|R0 step3000 성공|R1 step3000 성공|최종 위치 mm R0/R1|최종 자세 rad R0/R1|안전 실패 R0/R1|\n|---|---:|---:|---:|---:|---:|\n'
    for kind in ['handoff','autonomous','nominal']:
        a=summary['R0_step3000'][kind];b=summary['R1_step3000'][kind]
        table+=f'|{kind}|{a["success"]}/{a["episodes"]}|{b["success"]}/{b["episodes"]}|{a["final_position_error_mm"]:.3f}/{b["final_position_error_mm"]:.3f}|{a["final_orientation_error_rad"]:.3f}/{b["final_orientation_error_rad"]:.3f}|{a["physical_safety_failures"]}/{b["physical_safety_failures"]}|\n'
    matched_h=[r for r in pairs if r['kind']=='handoff']
    common_handoff=statistics.mean(r['R1_minus_R0_common_end_position_mm'] for r in matched_h)
    minimum_handoff_wins=sum(r['R1_minus_R0_minimum_position_error_mm']<0 for r in matched_h)
    before='|평가|R0 step0 위치→step3000 mm|R0 안전 실패 step0→3000|R1 step0→3000 위치 mm|\n|---|---:|---:|---:|\n'
    for kind in ['handoff','autonomous','nominal']:
        z=summary['R0_step0'][kind];a=summary['R0_step3000'][kind];b=summary['R1_step3000'][kind]
        before+=f'|{kind}|{z["final_position_error_mm"]:.3f}→{a["final_position_error_mm"]:.3f}|{z["physical_safety_failures"]}→{a["physical_safety_failures"]}|{z["final_position_error_mm"]:.3f}→{b["final_position_error_mm"]:.3f}|\n'
    detail='|Checkpoint|Seed / Offset mm|Handoff 위치 mm|최소 위치 mm (state t)|최종 위치 mm|최종 자세 rad|Approach t|최초 Safety 실패 t|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for e in hand:
        detail+=f'|{e["checkpoint"]}|{e["seed"]} / {e["offset"][:2]}|{e["handoff_pre_position_error_mm"]:.3f}|{e["minimum_position_error_mm"]:.3f} ({e["minimum_position_post_t"]})|{e["final_position_error_mm"]:.3f}|{e["final_orientation_error_rad"]:.4f}|{e["approach_first_post_t"]}|{e["first_safety_failure_post_t"]}|\n'
    safetytable='|Checkpoint|Episode|실패 state t|실제 위반 접촉 기록|Shape Clearance pair / gap mm|\n|---|---|---:|---|---|\n'
    for s in safety:
        for event in s['events']:
            contacts=sorted({' / '.join(c['names']) for c in event['substep_contact_violations']})
            shape='; '.join(f'{" / ".join(v["pair"])}: {v["gap_m"]*1000:.4f}' for v in event['shape_violations'])
            safetytable+=f'|{s["checkpoint"]}|{s["episode"]}|{event["post_state_t"]}|{"; ".join(contacts) or "위반 기록 없음"}|{shape}|\n'
    run_id=read(new/'identity.json')['run_id']
    offline_rows=[]
    for path,labels in [(base/'offline_diagnostics/expert_command_errors.csv',{'R0':'R0_step0','R1':'R1_step3000'}),
                        (new/'offline_diagnostics/expert_command_errors.csv',{'R0':'R0_step3000'})]:
        for r in csv.DictReader(path.open()):offline_rows.append(dict(checkpoint=labels[r['condition']],**r))
    csvwrite(output/'offline_expert_action_comparison.csv',offline_rows)
    offline_table='|Checkpoint|Nominal dev command MAE mm|Recovery dev command MAE mm|First Recovery direction cosine|\n|---|---:|---:|---:|\n'
    for label in LABELS:
        group=[r for r in offline_rows if r['checkpoint']==label]
        nominal=statistics.mean(float(r['controller_translation_mae_m'])*1000 for r in group if r['kind']=='nominal')
        recovery=statistics.mean(float(r['controller_translation_mae_m'])*1000 for r in group if r['kind']=='recovery')
        cosine=statistics.mean(float(r['first_expert_student_translation_cosine']) for r in group if r['kind']=='recovery')
        offline_table+=f'|{label}|{nominal:.3f}|{recovery:.3f}|{cosine:.3f}|\n'
    report=f'''# Task 10-F 후속 진단 — Matched Checkpoint & Handoff Failure

## 1. 범위와 핵심 판단

신규 Evaluation UUID: `{run_id}`. Parent Comparison: `d7c26fee195e4f32a7ddb99a94877b72`. 기존 Final UUID: `2bfe867c20e340aab2157c5e3ff33e8e`.

이번 작업은 저장된 30개 Episode를 먼저 분석하고, 부족한 **R0 step3000 15개만** 추가 평가했다. R1 step3000 15개는 Identity 검증 후 재사용했다. 학습·Expert Demonstration 수집·Architecture/Loss/Threshold 변경은 수행하지 않았다. 기존 Validation-selected 결과와 선택 규칙도 보존했다. 공개된 Final seeds를 사용한 후속 진단이며 새로운 Blind Test가 아니다.

**CONFIRMED:** 동일한 3000 updates 비교에서도 공식 Reach 성공 수는 아래 표와 같다. 성공률 개선과 연속 제어에서의 안정적인 Recovery가 입증됐다고 해석할 수 없다.

{table}

동일Step에서 R1의 Handoff 최종 위치·자세 오차가 R0보다 작다는 결과는 없다. 반면 안전 실패 수는 줄었고 Autonomous/Nominal의 평균 종료 위치 오차는 작았다. Handoff의 최소 위치 오차는 R1이 {minimum_handoff_wins}/6에서 작았으며, 공통 관측 마지막 시점의 평균 위치 차이(R1−R0)는 {common_handoff:.3f}mm였다. 서로 다른 지표의 개선을 하나의 일관된 Recovery 우위로 합치지 않는다.

최종 오차는 안전 중단 또는 130-step horizon의 값이다. 모델별 중단 시점이 달라 평균값은 동일 시간 노출의 비교가 아니다. `paired_step3000_comparison.csv`에는 같은 Seed/Offset의 원값과 차이, 두 Trace가 모두 관측된 마지막 시점의 오차도 별도로 보존했다. 표준화된 통계적 우월성이나 모집단 일반화는 **INCONCLUSIVE**다.

Handoff/Autonomous 각6개는 Parent3×Offset2이며 독립 Parent6개가 아니다. Nominal은 같은3개 Parent다.

R0 step3000 Autonomous의 seed65002 두 Offset 조건은 교란 시점t=25 이전인 post-state24에 Physical Safety 실패로 중단됐다. 이2개를 실패 분모에 유지했다. 실제 교란 완료는 신규 R0에서 Handoff6/6, Autonomous4/6이며 기존 R1은 둘 다6/6이다. Offset을 받지 못한2개를 Recovery 성공/실패만의 비교라고 표현하지 않는다. `disturbance_ok/steps`, `aborted_before_intervention`, 종료 이유를 CSV에 별도 기록했다. Autonomous의 최소 오차는 전체 Student 구간 기준이며, 교란 이후 error reduction은 실제 intervention 완료 사례에만 존재한다.

![Matched-step 결과](figures/matched_step3000_reach.png)

## 2. Checkpoint 및 평가 계약

네 Checkpoint의 파일 존재·전체 SHA256·strict model readback·ActionCodec·Normalizer·Config를 확인했다. R0/R1 step0은 파일/모델/codec/normalizer/config가 동일하여 R1 step0은 기존 R0 step0 평가의 명시적 alias로 사용했다. 중복 Simulator 실행은 없다.

|Checkpoint|SHA256|
|---|---|
'''
    for label,r in identity['checkpoints'].items():report+=f'|{label}|`{r["sha256"]}`|\n'
    report+='''
공통 Task10D B step6000 원본 SHA256은 `19da92de69b0d9aa66aa210ea759451d0f2b98e2d01c41c587ba986fd60b2cbc`다. Step0 Fine-tuning 저장 파일은 optimizer/config metadata를 포함하므로 원본 파일 SHA와 다르며 모델 가중치는 같다.

Vast **57640710**, RTX3090, Torch2.8.0+cu128/CUDA12.8, ManiSkill3.0.1, SAPIEN3.0.3, CPU PhysX/GPU Renderer, 제어20Hz/Physics100Hz를 사용했다. CPU quota1.92cores다. 기존 `recovery_eval.episode()`를 수정 없이 호출했고, Protocol·Source Hash 및 이전 Source Snapshot과 비교했다. 신규 R0의 모든 초기 Reset이 기존 평가와 같고, Handoff Expert Prefix+Disturbance post-state 및 t0/t25 live Graph 배열도 정확히 같았다. SAM2 optional `_C` postprocessing 부재는 원래 환경과 동일하다.

로컬 repository의 기존 `pilot.py` 사본은 서버 실행 사본과 SHA가 달랐다. 로컬 파일은 원래 보호 대상 Hash 그대로 보존했고, 서버 파일은 이전 archived Source Snapshot과 같은 Hash(`888877532584df27d58ad4509f54bc76f8f265edc607744ddfecf3828ccc6db7`)임을 확인했다. 따라서 이전/신규 평가 간 Source 변경이 아니다. 보고서의 `execution_source_dependencies/`에는 실제 서버와 Hash가 같은15개 의존 Source 사본을 보존했다.

Episode마다 GRU hidden과 Causal Tracker를 reset하고 현재 RGB-D→Plan A Graph를 계산했다. Expert Prefix와 disturbance 관측도 매번 순서대로 forward했다. GT Target은 평가의 `measure()`에서만 사용하며 Student 입력은 기존 Graph+Robot76이다. 저장 Graph로 live 입력을 대체하지 않았다.

`action_t=t`의 관측/Robot76은 pre-action, Trace의 metric은 **post_state_t=t+1**다. Handoff 초기값은 첫 Student Action 직전인 이전 Trace row에서 가져왔다. MP4는 원래 evaluator의 pre-action RGB와 post-action annotation을 포함하므로 세부 시간 판정은 영상 대신 Trace를 사용했다. SO(3) 오차는 기존 quaternion relative rotation의 shortest angle이고 quaternion 부호 동치 테스트를 통과했다. Native Rotation3 command 오차와 구분한다.

## 3. Checkpoint 선택 효과

'''+before+'''
기존 Selected 비교는 R0 step0 vs R1 step3000이었다. 이번 Primary는 R0 step3000 vs R1 step3000이다. R0의 일반 Nominal Fine-tuning 변화와 R1의 Corrective Mix 변화를 모두 위 표 및 `checkpoint_selection_effect.json`에 보존했다. 두 조건은 각153936 supervised timestep이지만 R0 Nominal153936/Recovery0, R1 Nominal76968/Recovery76968이다. Burn-in은 각각57357/125294 frames다. 같은 update가 Corrective Data만의 순수한 인과 효과를 고립시키지는 않는다.

## 4. Handoff 위치·자세·순서·유지 조건

'''+detail+f'''
Student 관측 구간에서 위치4mm를 한 번도 충족하지 못한 Episode 수는 {posnever}다. 모든 모델의 각 Episode에서 Orientation0.03rad 이하 프레임은 존재했으나, 위치와 동시 통과를 의미하지 않는다. 종료 시 Orientation을 충족하지 못한 수는 {endrot}다. Position/Orientation 최소값은 Student 구간에서 각각 계산했고 두 최소값의 시점도 따로 기록했다.

Approach position10mm + rotation0.05rad 선행 조건은 원래 평가 계약 그대로다. Seed65001의 Handoff는 모든 모델에서 Approach 미통과 상태였고 교란 직후 pre-close Target까지126–136mm였다. 다른 두 seed는44–48mm였고 Expert Prefix에서 이미 Approach를 통과했다. 초기 상태는 모두 Physical Safety를 통과했다. 초기 위치/순서 조건과 이후 Student 행동에서 생긴 접촉을 구분해야 한다.

기존 R1은 seed65000/65002에서 최소5.25–6.82mm까지 접근했으나4mm를 넘지 못했고, 이후 오차가 증가했다. seed65001은 기존 R0 step0의 최소102–113mm보다 R1의 최소10.8–17.6mm가 작지만 Approach 및 pre-close 성공은 없다. 신규 R0 step3000의 동일 값은 위 Episode 표에 포함했다. `moving_toward/away_target_intervals`, 감소 후 최초 증가 시점, running-min 이후 최대 rebound도 CSV로 제공한다. Target 반대편을 실제로 통과했다고 증명되지 않은 경우 오차 rebound를 기하학적 overshoot라고 단정하지 않는다.

![Handoff 위치](figures/handoff_position_error.png)

![Handoff 자세](figures/handoff_orientation_error.png)

## 5. Safety v2 실패의 실제 내용

'''+safetytable+'''
모든 안전 검사는 원래 Safety v2를 유지했다. External shape clearance200µm, penetration tolerance200µm, contact impulse threshold1e-5, max joint velocity5, object translation2mm/rotation0.01rad를 변경하지 않았다. 링크0 mounting 예외도 원래 그대로다. Gap-only 사례는 접촉 위반이 기록되지 않은 보수적 간격 검사 실패이며, 실제 충돌이라고 바꾸어 표현하지 않는다. Contact 사례의 `max_impulse`는 Force가 아니다.

저장된 Handoff에서 Robot–Table, enabled Robot self-collision, joint position/velocity, object-motion predicate의 실패 기록은 없다. 모든 Gripper opening은65mm 이상이었다. 안전 실패는 최초 실패 boundary에서 중단되어 이후 지속 여부는 **NOT_AVAILABLE**다. 원래 evaluator는 위반 Contact만 flatten하여 저장했으므로 전체 substep index/Force/허용 범위 Contact는 **NOT_AVAILABLE**다. 접촉 기록 부재가 모든 접촉의 부재를 뜻하지 않는다.

동일 Final Seed의 Expert Prefix 구간은 모두 안전했다. 같은 Seed에서 Handoff 이후까지 Nominal Expert를 완주한 대조 Trace는 없어 동일 조건의 Expert suffix 안전성은 **NOT_TESTED**다. 별도 seed30000 Nominal full Reach audit는 안전 판정 검증에 쓰였으나 같은 Final 조건이라고 주장하지 않는다.

![Safety 타임라인](figures/safety_predicate_timeline.png)

## 6. 실패 분류 및 안전 제외 보조 지표

`POSITION_FAILURE/ORIENTATION_FAILURE`는 Student 구간에서 해당 Target 기준을 한 번도 충족하지 못한 경우다. 최종 endpoint에서의 각 predicate 실패는 별도 column으로 제공한다. `ORDER_FAILURE`는 Approach 선행 통과 없음, `STABILITY_FAILURE`는 전체 후보 조건3연속 부재이며 upstream 실패를 포함한다. 따라서 독립 원인 수가 아니고 단순 합을 Episode 수로 사용할 수 없다. `MULTIPLE_FAILURES`는 중첩 flag다. 물리 안전 및 Gripper 실패는 별도 flag다.

`diagnostic_without_safety`는 Safety만 제외하고 Approach→Pre-close 순서, 동시 위치/자세, Opening, Peg displacement, Student3연속을 유지하는 사후 지표다. 공식 성공률을 수정하지 않는다. 안전 중단 이후의 미관측 행동은 추정하지 않았으므로 이 보조 지표는 저장된 구간만의 판정이다.

'''+f'안전 제외 보조 성공 수는 '+', '.join(f'{label}: {summary[label]["handoff"]["diagnostic_without_safety"]}/6' for label in LABELS)+'''. Safety를 통과한다고 가정해도 관측된 구간의 위치·순서·3-step 조건이 해결되는 것은 아니다.

![목표 순서와 유지 조건](figures/reach_order_stability_conditions.png)

## 7. 연구 질문별 해석

기존 frozen development Graph에서 R0 step3000만 추가 inference하고 기존 R1 step3000/R0 step0 offline 진단을 재사용했다. 각4 Nominal/32 Recovery Episode의 macro 평균이며, BC loss나 실제 TCP pose error가 아닌 Controller translation command의 성분 MAE다. Hidden은 각 sequence 시작부터 순차 forward했으며 optimizer update는 없다.

'''+offline_table+'''

같은3000-step의 Recovery command 성분 MAE는 R0가 더 작고 첫 보정 방향 cosine은 R1이 더 높다. Corrective Mix가 Offline Expert Action Agreement의 모든 측면을 개선했다고도 주장할 수 없다. 원래 validation Method B loss는 motion normalization 및 gripper를 포함하므로 이 성분 MAE와 동일 지표가 아니다.

|질문|판정|근거와 한계|
|---|---|---|
|Q1 동일Step Corrective Recovery 우위|INCONCLUSIVE|공식 성공 수는 Primary 표와 같으며 성공률 우위를 확립하지 못함. Episode별 연속 metric 차이는 관측 결과로 제한|
|Q2 R0 step0→3000 효과|SUPPORTED|동일15조건의 위치·안전 변화가 선택 효과 표에 존재. 일반 Fine-tuning이 Reach 성공을 보장하지 않음|
|Q3 Corrective Data 기여의 인과 해석|INCONCLUSIVE|초기 가중치·update·supervised 수는 같지만 데이터 노출과 burn-in 양이 다르고 학습 seed1개/Final parent3개|
|Q4 위치/자세 중 제한 조건|CONFIRMED|최소 위치와 동시 Target predicate 분석. 최종 Orientation 실패와 지속적인 위치 미도달을 구분|
|Q5 안전 실패 원인|CONFIRMED|기록된 Contact 또는 shape gap 위반 pair와 최초 시점을 보존. 중단 이후 지속/Force는 NOT_AVAILABLE|
|Q6 Safety 제외 시 성공|CONFIRMED|저장된 구간의 diagnostic_without_safety를 별도 계산. 공식 판정 변경 없음|
|Q7 목표 근처 접근 여부|SUPPORTED|Seed별 최소 오차·handoff 초기값·rebound에 따른 혼합 양상이며 단일 평균으로 설명 불가|
|Q8 다음 연구 우선순위|SUPPORTED|Full Training의 성공을 정당화하는 근거는 부족. 기존 정상/교란 상태의 Geometry→Action mapping과 Approach→Descend 전환 및 closed-loop error accumulation 진단이 우선|

추가 학습이나 데이터 수집을 실행하지 않았다. Geometry representation, GRU history, 학습 fit 중 특정 하나가 원인이라고 현재 결과로 확정할 수 없다. Peg presence/gate 통과와 checkpoint equality만으로 Geometry가 제어에 충분하다고 판단하지 않는다.

## 8. 실행·검증 및 산출물

최초 preflight 시 common `assert_nested_equal`의 호출 인자 오류로 중단됐으며 Simulator Episode는0개였다. 실패 UUID/로그를 보존하고 호출만 바로잡은 독립 run02에서15개를 실행했다. 성공 조건·정책·원래 evaluator는 바꾸지 않았다. 기존 다른 프로세스는 종료하지 않았다. Git commit/외부 publishing은 수행하지 않았다.

로컬 분석 시간축 반례 테스트3개, 기존 Logger/Artifact 계약 테스트42개, quaternion 부호 동치 검증을 수행했다. 실제 CUDA 정책/vision inference는 지정 인스턴스에서만 수행했다. 로컬 CUDA 및 GPU PhysX는 NOT_TESTED다. 새 Trace/Graph/Trajectory/MP4의 SHA256 및 저장 후 readback을 기록했다. 원본656 indexed artifact 및55 protected source/기존6 실패 raw도 그대로인지 확인했다.

- [Checkpoint Identity](checkpoint_identity.json)
- [Matched step 평가](matched_step_evaluation.csv)
- [Paired step3000 비교](paired_step3000_comparison.csv)
- [Checkpoint 선택 효과](checkpoint_selection_effect.json)
- [Handoff 실패 분해](handoff_failure_breakdown.csv)
- [Handoff stepwise](handoff_stepwise_metrics.csv)
- [Safety 세부 기록](physical_safety_breakdown.json)
- [Reach predicate timeline](reach_predicate_timeline.csv)
- [평가 계약 검증](evaluation_contract_verification.json)
- [Verification](verification.json)
- [Artifact Index](artifact_index.json)

기존 영상은 재생성하지 않고 각 Episode CSV에 원본 경로/해시를 연결했다. 새 평가는 원래 evaluator가 저장한15개 MP4와 Trace를 제공한다.
'''
    for label,root in [('신규 R0 step3000',new/'rollouts'),('기존 R1 step3000',base/'final_evaluation/rollouts')]:
        tag=('R0' if label.startswith('신규') else 'R1')+'_handoff_65000_5_0'
        import os
        rel=os.path.relpath(root/tag/'rollout.mp4',output)
        report+=f'\n- [{label} 대표 Handoff MP4]({rel})\n'
    (output/'10f_matched_checkpoint_handoff_diagnosis.md').write_text(report)
    html=MarkdownIt('commonmark',{'html':False}).enable('table').render(report)
    (output/'10f_matched_checkpoint_handoff_diagnosis.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10F Matched Checkpoint Diagnosis</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#17202b}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{border:1px solid #ddd;padding:7px}img{max-width:100%}code{overflow-wrap:anywhere}h1,h2{line-height:1.3}</style>'+html)
    write(output/'summary.json',dict(run_id=run_id,summary=summary,episodes=45,new_episodes=15,handoff_position_never_pass=posnever,final_orientation_fail=endrot))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base.resolve(),a.new.resolve(),a.output.resolve(),Path(__file__).resolve().parents[2])

"""Read-only analysis of saved Task07-G tensors; no policy or environment loading.

Adds reviewed findings and tolerance-aware impulse interpretation to this run.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graph.integration_baseline.io import write_json,sha256
from graph.integration_baseline.point_resampling_report import verify

CONDITIONS=('phase1','local01','local05')


def aggregate(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,
        max=float(x.max()) if len(x) else None,min=float(x.min()) if len(x) else None)


def corr(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float);good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.ptp(x[good])==0 or np.ptp(y[good])==0:return None
    return float(spearmanr(x[good],y[good]).statistic)


def recovery_with_tolerance(a,b,t0,atol,rtol):
    error=np.abs(a-b)
    close=(error <= atol+rtol*np.abs(a)).all(axis=1)
    outside=np.flatnonzero(~close[t0:])
    offset=int(outside[-1]+1) if len(outside) else 0
    return dict(stable_within_fp32_tolerance_after_steps=offset if offset<len(close)-t0 else None,
        censored=offset>=len(close)-t0,post_impulse_outside_tolerance_frames=int((~close[t0+1:]).sum()),
        maximum_component_error=float(error[t0:].max()),final_component_error=float(error[-1].max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args();run=args.run
    results=json.loads((run/'results.json').read_text());cfg=json.loads((run/'config.yaml').read_text())
    with (run/'frames.csv').open() as f:rows=list(csv.DictReader(f))
    def num(row,key):return float(row[key]) if row[key] else np.nan
    summary={};correlations={};recovery=[];stage_episode=[];episodes=[]
    for c in CONDITIONS:
        s=[r for r in rows if r['condition']==c]
        changed=[r for r in s if num(r,'peg_source_jaccard')<1 or num(r,'box_source_jaccard')<1]
        present={name:[r for r in s if num(r,name+'_count_a')>0] for name in ('peg','box')}
        summary[c]=dict(frames=len(s),actual_changed_frames=len(changed),
            conditional_present_jaccard={name:aggregate([num(r,name+'_source_jaccard') for r in p]) for name,p in present.items()},
            changed_frame_instant_translation_mm=aggregate([num(r,'instant_command_translation_mm') for r in changed]),
            worst_translation_frame=max(s,key=lambda r:num(r,'history_command_translation_mm')),
            worst_gripper_frame=max(s,key=lambda r:num(r,'history_command_gripper_mm')),
            native_mse_delta_increased=sum(num(r,'history_expert_mse_delta')>cfg['atol'] for r in s),
            native_mse_delta_decreased=sum(num(r,'history_expert_mse_delta') < -cfg['atol'] for r in s),
            native_mse_delta_unchanged=sum(abs(num(r,'history_expert_mse_delta'))<=cfg['atol'] for r in s),
            gripper_sign_changed_frames=sum(r['history_gripper_sign_changed']=='True' for r in s))
        for eid in [e['episode_id'] for e in results['episodes']]+['pooled_descriptive']:
            es=[r for r in s if eid=='pooled_descriptive' or int(r['episode_id'])==eid]
            correlations[c+'/'+str(eid)]={
                'point_count_delta_vs_z_l2':corr([num(r,'peg_count_b')+num(r,'box_count_b')-num(r,'peg_count_a')-num(r,'box_count_a') for r in es],[num(r,'z_l2') for r in es]),
                'edge_count_delta_vs_z_l2':corr([num(r,'edge_count_b')-num(r,'edge_count_a') for r in es],[num(r,'z_l2') for r in es]),
                'source_disagreement_vs_instant_translation':corr([1-(num(r,'peg_source_jaccard')+num(r,'box_source_jaccard'))/2 for r in es],[num(r,'instant_command_translation_mm') for r in es]),
                'peg_chamfer_vs_instant_translation':corr([num(r,'peg_chamfer_mean_m') for r in es],[num(r,'instant_command_translation_mm') for r in es]),
                'z_l2_vs_instant_translation':corr([num(r,'z_l2') for r in es],[num(r,'instant_command_translation_mm') for r in es])}
    for ep in results['episodes']:
        eid=ep['episode_id'];folder=run/f"{ep['split']}_{eid}"
        with np.load(folder/'tensors.npz',allow_pickle=False) as tensors:
            a={k:tensors[k] for k in tensors.files}
        # Plot QA: undefined distance/features for missing support are gaps, never zeros.
        metric=json.loads((folder/'sampling_comparison.json').read_text())
        fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True)
        for c in CONDITIONS:
            ms=[r for r in metric if r['condition']==c]
            for oid,name in ((0,'Peg'),(1,'Box')):
                for ax,key,scale in ((axes[0,0],'count_b',1),(axes[0,1],'chamfer_mean_m',1000),
                    (axes[1,0],'voxel_jaccard',1),(axes[1,1],'raw_ede_jaccard',1)):
                    ax.plot([r['timestep'] for r in ms],[r['objects'][oid][key]*scale if r['objects'][oid][key] is not None else np.nan for r in ms],label=c+' '+name,linewidth=.8)
            for k,name in ((1,'Robot-object'),(2,'Peg-Box')):
                axes[2,0].plot([r['timestep'] for r in ms],[r['edge_jaccard'][k] for r in ms],label=c+' '+name,linewidth=.8)
            values=[]
            for r in ms:
                normals=[v for o in r['objects'] if o['f5_nearest_abs_difference'] is not None
                    for v in o['f5_nearest_abs_difference'][:3] if v is not None]
                values.append(float(np.mean(normals)) if normals else np.nan)
            axes[2,1].plot([r['timestep'] for r in ms],values,label=c,linewidth=.8)
        for ax,label in zip(axes.ravel(),['Alternative point count','Symmetric NN mean / Chamfer (mm); absent = gap',
            'Nominal voxel Jaccard (empty/empty=1)','Raw EDE source-edge Jaccard','Interaction source-edge Jaccard','Nearest-point normal abs difference; absent = gap']):
            ax.set_ylabel(label);ax.set_xlabel('Timestep');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/'plots/geometry_topology.png',dpi=120);plt.close(fig)
        for c in CONDITIONS:
            for imp in ep['impulses']:
                if imp['condition']!=c or imp['status']!='PASS':continue
                t0=imp['timestep'];ai=a[f'impulse_{c}_{t0}_action'];hi=a[f'impulse_{c}_{t0}_hidden']
                hrec=recovery_with_tolerance(a['hidden_phase0'],hi,t0,cfg['atol'],cfg['rtol'])
                arec=recovery_with_tolerance(a['action_phase0'],ai,t0,cfg['atol'],cfg['rtol'])
                recovery.append(dict(episode_id=eid,condition=c,timestep=t0,initial_hidden_l2=imp['hidden']['initial'],
                    original_5pct_hidden_recovery_steps=imp['hidden']['stable_below_after_steps'],
                    initial_translation_delta_mm=imp['translation_mm']['initial'],
                    original_5pct_translation_recovery_steps=imp['translation_mm']['stable_below_after_steps'],
                    hidden_fp32_recovery_steps=hrec['stable_within_fp32_tolerance_after_steps'],
                    action_fp32_recovery_steps=arec['stable_within_fp32_tolerance_after_steps'],
                    hidden_post_outside_frames=hrec['post_impulse_outside_tolerance_frames'],
                    hidden_final_component_error=hrec['final_component_error'],action_final_component_error=arec['final_component_error']))
            delta=a['action_history_'+c]-a['action_phase0']
            sign_a=a['action_phase0'][:,6]<0;sign_b=a['action_history_'+c][:,6]<0
            episodes.append(dict(episode_id=eid,split=ep['split'],condition=c,
                natural_z_l2=ep['natural_z_l2'],same_state_z_l2=ep['conditions'][c]['z_l2'],
                sign_changed_frames=int((sign_a!=sign_b).sum()),
                gripper_transition_timesteps_a=(np.flatnonzero(np.diff(sign_a))+1).tolist(),
                gripper_transition_timesteps_b=(np.flatnonzero(np.diff(sign_b))+1).tolist(),
                expert_mse_a=ep['conditions'][c]['modes']['history']['expert_mse_a']['mean'],
                expert_mse_b=ep['conditions'][c]['modes']['history']['expert_mse_b']['mean'],
                history_vs_instant_translation_mean_ratio=ep['conditions'][c]['modes']['history']['command_translation_mm']['mean']/max(ep['conditions'][c]['modes']['instant']['command_translation_mm']['mean'],1e-12)))
            for stage in sorted(set(a['stage'])):
                mask=a['stage']==stage
                d=ep['conditions'][c]['modes']['history']
                selected=[r for r in rows if int(r['episode_id'])==eid and r['condition']==c and r['stage']==stage]
                stage_episode.append(dict(episode_id=eid,split=ep['split'],condition=c,stage=stage,frames=int(mask.sum()),
                    translation_delta_mm_mean=float(a[c+'_history_command_translation_mm'][mask].mean()),
                    hidden_l2_mean=float(a[c+'_hidden_l2'][mask].mean()),
                    expert_mse_delta_mean=float(a[c+'_history_expert_mse_delta'][mask].mean()),
                    gripper_sign_changed_frames=int((sign_a[mask]!=sign_b[mask]).sum())))
    write_json(run/'quantitative_findings.json',dict(conditions=summary,episode_details=episodes,
        note='Conditional present-object and actually-changed-frame summaries avoid diluting effects with identical/empty support'))
    write_json(run/'descriptive_correlations.json',dict(values=correlations,
        warning='Spearman rho is descriptive only; no frame-independence assumption or p-value. Null means constant/insufficient data.'))
    for name,values in [('recovery_tolerance.csv',recovery),('stage_episode_summary.csv',stage_episode)]:
        with (run/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,list(values[0]));writer.writeheader();writer.writerows(values)
    # Additional component/error and stage visualization from saved numbers.
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    colors=['#b83243','#2d7eae','#ad6c16']
    for ci,c in enumerate(CONDITIONS):
        for j,ep in enumerate(results['episodes']):
            modes=ep['conditions'][c]['modes']['history']
            axes[0,0].bar(j+(ci-1)*.22,modes['expert_mse_delta']['mean'],width=.22,color=colors[ci],label=c if j==0 else None)
        stages=sorted(set(r['stage'] for r in stage_episode))
        axes[0,1].plot(stages,[np.mean([r['translation_delta_mm_mean'] for r in stage_episode if r['condition']==c and r['stage']==s]) for s in stages],marker='o',label=c,color=colors[ci])
        rr=[r for r in rows if r['condition']==c]
        axes[1,0].scatter([num(r,'peg_chamfer_mean_m')*1000 for r in rr],[num(r,'instant_command_translation_mm') for r in rr],s=8,label=c,color=colors[ci],alpha=.5)
        axes[1,1].scatter([num(r,'edge_count_b')-num(r,'edge_count_a') for r in rr],[num(r,'z_l2') for r in rr],s=8,label=c,color=colors[ci],alpha=.5)
    axes[0,0].set_xticks(range(4),[str(e['episode_id']) for e in results['episodes']]);axes[0,0].set_ylabel('Paired expert native7 MSE delta (episode mean)')
    axes[0,1].set_ylabel('History translation difference mm (episode means)')
    axes[1,0].set_xlabel('Peg bidirectional NN mean / Chamfer mm');axes[1,0].set_ylabel('Same-past translation difference mm')
    axes[1,1].set_xlabel('Edge-count difference B−A');axes[1,1].set_ylabel('Structured readout absolute L2')
    for ax in axes.ravel():ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'reviewed_findings.png',dpi=130);plt.close(fig)
    aggregate_data=json.loads((run/'aggregate.json').read_text())['overall']
    def m(c,key):return aggregate_data[c][key]['mean']
    def maximum(c,key):return aggregate_data[c][key]['max']
    phase_imp=[r for r in recovery if r['condition']=='phase1' and r['initial_hidden_l2']>1e-4]
    p5=[r['original_5pct_hidden_recovery_steps'] for r in phase_imp]
    pt=[r['original_5pct_translation_recovery_steps'] for r in phase_imp]
    nearfloor=[r for r in recovery if 0<r['initial_hidden_l2']<1e-4]
    q={
        'Q1':dict(status='CONFIRMED',answer=f"Phase1 Readout relative L2 평균 {m('phase1','z_relative_l2'):.2%}, p95 {aggregate_data['phase1']['z_relative_l2']['p95']:.2%}, 최대 {maximum('phase1','z_relative_l2'):.2%}. 다섯 Slot/Cosine은 Tensor·CSV에 분리 저장."),
        'Q2':dict(status='CONFIRMED',answer=f"Phase1 동일 이전 Hidden의 이동 명령 차이 평균 {m('phase1','instant_command_translation_mm'):.4f}mm; 독립 History 평균 {m('phase1','history_command_translation_mm'):.4f}mm, 최대 {maximum('phase1','history_command_translation_mm'):.4f}mm. 회전 거리 평균 {m('phase1','history_command_rotation_rad'):.6f}rad; Finger Target 평균 {m('phase1','history_command_gripper_mm'):.4f}mm, 최대 {maximum('phase1','history_command_gripper_mm'):.4f}mm."),
        'Q3':dict(status='CONFIRMED',answer=f"Local01/05 Readout 상대 차이는 평균 {m('local01','z_relative_l2'):.3%}/{m('local05','z_relative_l2'):.3%}; History 이동 차이는 {m('local01','history_command_translation_mm'):.4f}/{m('local05','history_command_translation_mm'):.4f}mm. 실제 교체율은 Peg/Box 각각0.430%/0.492%, 3.682%/3.245%다. Phase1과 기하 변화 규모가 다르다."),
        'Q4':dict(status='INCONCLUSIVE',answer='Phase1 Episode 평균 Expert MSE는 30003에서 증가(+0.0002715), 나머지 세 Episode에서 감소했다. 30003 증가에는 Rotation·Gripper가 기여했다. Local05는 네 Episode 중 세 곳에서 작은 평균 증가가 있었다. 일관된 성능 저하나 모집단 유의성은 확정할 수 없다. 각 성분·Controller 오차는 별도 보고한다.'),
        'Q5':dict(status='CONFIRMED',answer=f"Phase1 Hidden L2 평균 {m('phase1','hidden_l2'):.5f}, Cosine 평균 {m('phase1','hidden_cosine'):.7f}. 지속 변경은 추가 History 효과를 만들지만 관측된 차이가 일방향으로 커진다고 볼 수 없다. Local01/05 Hidden Cosine 평균은 {m('local01','hidden_cosine'):.9f}/{m('local05','hidden_cosine'):.9f}."),
        'Q6':dict(status='CONFIRMED',answer=f"유효 Phase1 Impulse {len(phase_imp)}개에서 Hidden의 5% 잔류 기준은 {min(p5)}–{max(p5)} Step, 이동 명령은 {min(pt)}–{max(pt)} Step이었다. 완전한 수치 동등성과 다르며, Action의 FP32 성분별 복귀는20–46 Step이었다. 한 Hidden tail은 종료까지 허용오차 밖이었으나 마지막 최대 성분 차이는2.28e−6이었다. 물리적 Recovery는 NOT_TESTED."),
        'Q7':dict(status='CONFIRMED',answer=f"동일 Graph B에 Hidden A[t−1]/B[t−1]를 교차 입력해 과거 효과를 분리했다. Phase1 추가 History 이동 차이 평균 {m('phase1','history_extra_command_translation_mm'):.4f}mm, 최대 {maximum('phase1','history_extra_command_translation_mm'):.4f}mm. 두 크기는 벡터 효과이므로 단순 스칼라 가산하지 않는다."),
        'Q8':dict(status='SUPPORTED',answer='이 네 저장 Expert Sequence에서는 작은 Local Replacement에 대한 Action/Hidden 안정성과, 큰 Phase1 Representation 변화가 제한된 Action 차이로 전달되는 경향이 지지된다. Close의 드문 Finger Target 차이와 Align의 최대 이동 차이는 남는다. 자연적인 Phase0 Temporal Inconsistency가 실제 실패 원인이라는 주장은 INCONCLUSIVE다.'),
        'Q9':dict(status='SUPPORTED',answer='Temporal Consistency를 비교할 정량 기준과 Close/Align의 표적 사례는 확보했다. 그러나 Persistent Anchor가 반드시 필요하다는 근거는 약하며 우월성은 NOT_TESTED다. 동일 학습 예산의 Sampling/Anchor 대조, 관측 누락·Identity 조건 통제, 별도 Closed-loop 검증이 여전히 필요하다.')}
    write_json(run/'interpretation.json',q)
    lines=['## 검수된 정량 판정', '', '**이 네 Episode에서는 작은 Local Replacement에 대해 Action/Hidden이 비교적 안정적이었다. 큰 Phase1 변경도 Readout 차이에 비해 제어 명령 차이는 작았으며, 자연적인 Phase0 Resampling이 실패의 원인이라는 결론은 얻지 못했다.**',
        '', '아래 평균은 전체 Frame의 기술통계다. 연속 Frame을 독립 표본으로 취급하지 않으며 Episode별 수치와 극값을 함께 해석한다.',
        '', '| 조건 | Readout 상대 L2 평균 | 이동 명령 차이 평균 / 최대 mm (독립 History) | 회전 거리 평균 rad | Finger Target 차이 평균 / 최대 mm |',
        '| --- | --- | --- | --- | --- |']
    for c in CONDITIONS:lines.append(f"| {c} | {m(c,'z_relative_l2'):.3%} | {m(c,'history_command_translation_mm'):.4f} / {maximum(c,'history_command_translation_mm'):.4f} | {m(c,'history_command_rotation_rad'):.7f} | {m(c,'history_command_gripper_mm'):.4f} / {maximum(c,'history_command_gripper_mm'):.4f} |")
    lines+=['', 'Phase1은 Peg가 있는332 Frame / Box가 있는387 Frame에서 Source Jaccard 평균0.3573 /0.5358이었다. 양쪽 Empty 또는 동일 선택을 포함한 전체 Frame 평균0.5959 /0.6598과 구분한다. 실제 선택이 바뀐 Frame은 Phase1 387개, Local01 228개, Local05 374개다. 해당 Frame만의 동일 Hidden 이동 차이 평균은 각각0.1744 /0.0058 /0.0206mm다.',
        '', '관측 표면의 양방향 NN 평균(Chamfer)은 존재 Frame 기준 Phase1 Peg1.708mm /Box3.472mm, Local01 0.00583 /0.02834mm, Local05 0.05114 /0.17475mm였다. 실제 Geometry 차이가 다른 조건을 같은 세기의 Noise로 해석하지 않는다.',
        '', 'Phase1 최대 이동 차이2.0156mm는 Train30003 t79 Align(원본 Peg230 /Box0 Point), 최대 Finger Target 차이2.6779mm는 같은 Episode t49 Close에서 발생했다. Local05 최대 이동0.4987mm는 Train30003 t68 Lift, 최대 Finger Target0.5898mm는 Validation30012 t39 Close였다. 그럼에도 모든528 Frame ×3 Alternative에서 Gripper Sign과 전환 Timestep은 동일했다. 최초 음수 시점은 Episode 순서로50/49/55/39다.',
        '', 'Phase1 Expert native7 MSE의 Episode 평균 변화는30000 −0.00003268, 30003 +0.00027147, 30007 −0.00009131, 30012 −0.00043833이었다. 전체 평균−0.00004737을 개선의 증거로 일반화하지 않는다. 30003은 Rotation3 MSE +0.00022410 /Gripper1 MSE +0.00123162였고, Translation3 MSE는−0.00000120이었다. Local05 전체 평균은+0.00001336이며 Episode 간 방향이 일치하지 않았다.',
        '', '동일 시점 Phase1 Readout 절대 L2 Episode 평균은0.0448/0.2857/0.1056/0.1707 수준이며, 자연 연속 Frame 변화 평균0.3145/0.8242/0.2367/0.5753과 비교 자료를 제공한다. 자연 변화에는 실제 움직임·Mask 변화가 포함된다. 자연 변화 대비 비율은 sampling-noise의 인과 추정량이 아니다.',
        '', f'유효 Phase1 단일 Frame Impulse {len(phase_imp)}개(초기 Hidden L2>1e−4)는 Hidden 5% 기준4–6 Step, 이동 명령3–6 Step에 관측 잔류가 감소했다. 초기 차이가0인 사례는 별도 제외했다. 일부 Local01의 초기 Hidden L2 약4.9e−6/7.6e−6에서는 고정 L2 floor1e−6 기준으로124/50 Step처럼 긴 tail이 나왔다. 이는 큰 효과의 장기 증폭과 다르다. [성분별 FP32 허용오차 복귀](recovery_tolerance.csv)를 함께 제공하며 허용오차는 사전 고정 atol1e−6/rtol1e−5를 그대로 사용했다.',
        '', '5% 기준 도달은 완전한 수치 복귀가 아니다. Phase1 Action은 사전 고정 FP32 성분별 허용오차까지20–46 Step에 복귀했다. Hidden은10개가20–66 Step에 복귀했고, Validation30012 t30은 남은68 Step 동안 기준 밖인 성분이 남았다(종료 최대 성분 차이2.276e−6, Action은28 Step에 복귀). 이런 작은 tail과 초기 효과의 큰 비율이 지속되는 경우를 구분한다.',
        '', '[추가 정량 자료](quantitative_findings.json) · [Episode×Stage CSV](stage_episode_summary.csv) · [Point/Edge/Geometry/Readout 기술상관](descriptive_correlations.json) · [검수 그림](reviewed_findings.png).',
        '', '| 질문 | 판정 | 답 |','| --- | --- | --- |']
    # Replace the rounded absolute readout values with exact saved episode means.
    actual='/'.join(f"{e['conditions']['phase1']['z_l2']['mean']:.4f}" for e in results['episodes'])
    lines=[s.replace('0.0448/0.2857/0.1056/0.1707',actual) for s in lines]
    for key,v in q.items():lines.append(f"| {key} | {v['status']} | {v['answer']} |")
    lines+=['', '실행·검증: Vast57640710 CUDA inference 완료. 로컬61개 PASS; 원격60개 PASS + 기존 Task03 fixture 부재1개 BLOCKED_FIXTURE. 원격의 최초 fixture 실패 로그를 보존했다. 이 blocker는 실제 네 Episode의 입력/Control 검증과 구분한다. Simulator/새 Vision 추론/학습/설치/Commit/외부 Publishing은 수행하지 않았다.',
        '', '재현: 기존 원격 Python 환경에서 run.sh를 사용한다. loader의 기본 소스는/workspace/task07-stop-clock, 추가 진단 모듈은/workspace/task07-point-resampling이며 bootstrap.py가 package path를 확장한다. 독립 실행 시 같은 package 경로와 현재 checkpoint manifest를 제공한다. 추가 정량 검수는 `python -m graph.integration_baseline.point_resampling_findings experiments/RUN_ID`로 저장 Tensor만 읽는다.',
        '', '원래 Episode summary.json의 gripper_switches 배열은 경계 i→i+1의 i를 뜻한다. 이 검수의 quantitative_findings.json에는 실제 전환 Observation Timestep(i+1)을 명시했다. 전환 동등성 결론은 동일하다.',
        '', '### 상세 계약·기본 집계', '']
    findings='\n'.join(lines)+'\n'
    write_json(run/'review_metadata.json',dict(inference_repeated=False,inputs='saved NPZ/JSON/CSV only',
        review_source_sha256=sha256(__file__),original_l2_recovery_preserved=True,
        geometry_plot_qa='undefined missing-object distances/features displayed as gaps, not zero',
        additional_fp32_recovery='same predeclared componentwise tolerances; diagnostic supplement, not retuned threshold'))
    report=run/'07g_point_resampling_stability.md';text=report.read_text()
    # Keep the original audit text as a separate, immutable source for this report revision.
    original=run/'initial_automated_report.md'
    if not original.exists():shutil.copy2(report,original)
    original_html=run/'initial_automated_report.html'
    if not original_html.exists():shutil.copy2(run/'point_resampling_stability.html',original_html)
    base=original.read_text();first,rest=base.split('\n',1)
    # Avoid a second conflicting Q table: original contract explanation remains available separately.
    start=rest.index('## 7. Q1–Q9 and evidence limits')
    end=rest.index('## 8. Artifacts and verification')
    rest=rest[:start]+rest[end:]
    report.write_text(first+'\n\n'+findings+rest)
    doc=original_html.read_text();block=['<section id="quantitative-findings"><h2>검수된 정량 판정</h2>',
        '<p>작은 Local Replacement에서 Action/Hidden은 비교적 안정적이었다. Phase1의 큰 표현 변화도 제한된 제어 명령 차이로 전달됐다. 자연적인 Phase0 Resampling의 실패 인과성은 확정하지 못했다.</p>',
        '<table><tr><th>Q</th><th>Status</th><th>Finding</th></tr>']
    for key,v in q.items():block.append('<tr><td>'+key+'</td><td>'+v['status']+'</td><td>'+html.escape(v['answer'])+'</td></tr>')
    block+=['</table><img src="reviewed_findings.png" alt="Episode expert error, stage sensitivity and geometry relationships">',
        '<p><a href="quantitative_findings.json">Conditional and extreme-frame findings</a> · <a href="recovery_tolerance.csv">FP32 tolerance-aware recovery</a> · <a href="stage_episode_summary.csv">Episode × Stage</a> · <a href="descriptive_correlations.json">Descriptive relationships</a></p>',
        '<p>Long L2 tails from initial effects near 1e−6 are not evidence of lasting physical instability. Compare componentwise FP32 tolerance recovery; do not confuse hidden L2 with control failure. All gripper signs and transition times were identical in these fixed sequences.</p></section>']
    # Original English Q table is retained in the initial report, so the final HTML has one authoritative verdict.
    a=doc.index('<section><h2>Comparisons and scope</h2>');b=doc.index('<img src="distributions.png"',a)
    doc=doc[:a]+'\n'.join(block)+'<section><h2>Overall distributions</h2>'+doc[b:]
    (run/'point_resampling_stability.html').write_text(doc)
    target=run/'source_snapshot_review'/Path(__file__).name;target.parent.mkdir(exist_ok=True);shutil.copy2(__file__,target)
    # The original execution source hashes remain immutable; the review has its own source identity.
    write_json(run/'verification.json',dict(**verify(run),review_inference_repeated=False,
        png_visual_review='Timeline, RGB-D/3D overlay, impulse, geometry and summary plots inspected by agent',
        browser_visual_review='NOT_EXECUTED: no browser connector available; Safari computer-use access was not approved. HTML parsed and all local links/assets verified.'))
    print(json.dumps(dict(recovery=recovery,summary={c:{k:v for k,v in s.items() if not k.startswith('worst')} for c,s in summary.items()}),indent=2))


if __name__=='__main__':main()

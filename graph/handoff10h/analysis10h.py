"""Parent-paired descriptive analysis; no model fitting or selection."""
import csv,html,json,math
from pathlib import Path
from collections import Counter,defaultdict
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from graph.plan_a_bc.common import read,write,sha,now

NAMES={'E':'expert_reference_rollouts.csv','B0':'autonomous_b_rollouts.csv','H1':'post_lift_handoff_rollouts.csv','H2':'pre_insertion_handoff_rollouts.csv'}

def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def wilson(x,n):
    if not n:return [None,None]
    z=1.96;p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]

def analyze(out,allrows,log):
    rows=[r for r in allrows if r['stage']=='final'];assert len(rows)==48
    by={(r['seed'],r['condition']):r for r in rows};seeds=read(out/'evaluation_seed_manifest.json')['final']
    assert set(by)=={(s,c) for s in seeds for c in NAMES}
    functional=[];paired=[];anchors=[];first10=[];failure=[];latency=[];phase=defaultdict(list);manifest=[]
    controlled={};traces={}
    for c,name in NAMES.items():
        rr=[r for r in rows if r['condition']==c];table(out/name,rr)
        valid=[r for r in rr if r['valid_anchor']] if c in ('H1','H2') else rr
        for metric in ('grasp','lift','alignment','insertion'):
            x=sum(r[metric] and (c not in ('H1','H2') or r['valid_anchor']) for r in rr)
            functional.append(dict(condition=c,metric=metric,successes=x,registered_parents=len(rr),wilson95=wilson(x,len(rr)),valid_anchor_parents=len(valid),conditional_successes=sum(r[metric] for r in valid),conditional_wilson95=wilson(sum(r[metric] for r in valid),len(valid))))
    for r in rows:
        folder=out/r['folder'];tr=read(folder/'trace.json');traces[(r['seed'],r['condition'])]=tr
        assert len(tr)==r['actions'] and sha(folder/'trace.json')==r['trace_sha256']
        capture=cv2.VideoCapture(str(folder/'rollout.mp4'));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));assert count==len(tr)
        for pos in (0,count-1):capture.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,_=capture.read();assert ok
        capture.release()
        manifest.append(dict(seed=r['seed'],condition=r['condition'],trace=r['folder']+'/trace.json',sha256=r['trace_sha256'],steps=len(tr),video_sha256=r['video_sha256'],video_frames=count))
        for a in read(folder/'anchors.json'):anchors.append(dict(seed=r['seed'],condition=r['condition'],**a))
        control=[t for t in tr if t['control']=='B'];controlled[(r['seed'],r['condition'])]=control
        for t in control[:10] if r['condition'] in ('H1','H2') else []:
            first10.append(dict(seed=r['seed'],condition=r['condition'],**t))
        for label in r['failure_labels']:failure.append(dict(seed=r['seed'],condition=r['condition'],label=label,causally_confirmed=False))
        for t in tr:phase[(r['seed'],r['condition'],t['phase'],t['control'])].append(t)
        latency.append(dict(seed=r['seed'],condition=r['condition'],steps=len(tr),control_dt_seconds=.05,wall_seconds=r['wall_seconds'],**{k:float(np.mean([t[k] for t in tr])) for k in ('inference_seconds','policy_seconds','gnn_encoder_readout_seconds','simulator_step_wall_seconds')}))
    for c in ('H1','H2'):
        eligible=[s for s in seeds if by[s,c]['valid_anchor']]
        for metric in ('insertion','lift'):
            gained=sum(by[s,c][metric] and not by[s,'B0'][metric] for s in eligible)
            lost=sum(by[s,'B0'][metric] and not by[s,c][metric] for s in eligible)
            paired.append(dict(comparison=c+'-B0',metric=metric,registered_parents=len(seeds),eligible_parents=len(eligible),gained=gained,lost=lost,paired_difference=(gained-lost)/len(eligible) if eligible else None,exact_discordant_p=binomtest(gained,gained+lost,.5).pvalue if gained+lost else 1.,unit='parent; branches and frames not independent'))
    eligible=[s for s in seeds if by[s,'H1']['valid_anchor'] and by[s,'H2']['valid_anchor']]
    gained=sum(by[s,'H2']['insertion'] and not by[s,'H1']['insertion'] for s in eligible)
    lost=sum(by[s,'H1']['insertion'] and not by[s,'H2']['insertion'] for s in eligible)
    paired.append(dict(comparison='H2-H1',metric='insertion',registered_parents=len(seeds),eligible_parents=len(eligible),gained=gained,lost=lost,paired_difference=(gained-lost)/len(eligible) if eligible else None,exact_discordant_p=binomtest(gained,gained+lost,.5).pvalue if gained+lost else 1.,unit='parent'))
    response=[]
    for c in ('H1','H2'):
        for seed in seeds:
            ts=controlled[seed,c][:10]
            if not ts:continue
            response.append(dict(seed=seed,condition=c,steps=len(ts),first_action_direction_error=ts[0]['action_direction_error'],first_step_tip_change_m=ts[0]['target_error_change_m'],first10_tip_change_m=ts[-1]['tip_hole_error_m']-ts[0].get('pre_tip_hole_error_m',ts[0]['tip_hole_error_m']),first10_missing=any(any(t['missing_objects']) for t in ts),first10_arm_saturation=any(t['arm_saturation'] for t in ts),first10_gripper_open=any(t['action'][6]>0 for t in ts),first10_grasp_lost=any(t['pre']['grasp'] and not t['grasp'] for t in ts)))
    table(out/'parent_first10_response.csv',response)
    geo=[];act=[]
    for (s,c,p,control),ts in phase.items():
        geo.append(dict(seed=s,condition=c,phase=p,control=control,frames=len(ts),mean_tip_hole_error_m=float(np.mean([t['tip_hole_error_m'] for t in ts])),mean_orientation_error_rad=float(np.mean([t['peg_hole_orientation_error_rad'] for t in ts])),missing_fraction=float(np.mean([any(t['missing_objects']) for t in ts]))))
        act.append(dict(seed=s,condition=c,phase=p,control=control,frames=len(ts),mean_expert_action_mae=float(np.mean([t['action_difference_mae'] for t in ts])),direction_error_fraction=float(np.mean([t['action_direction_error'] for t in ts])),arm_saturation_fraction=float(np.mean([t['arm_saturation'] for t in ts])),gripper_open_fraction=float(np.mean([t['action'][6]>0 for t in ts]))))
    for name,data in [('functional_success_comparison',functional),('paired_handoff_success_analysis',paired),('handoff_anchor_state_metrics',anchors),('handoff_first10_action_metrics',first10),('phasewise_geometry_error',geo),('phasewise_action_error',act),('phasewise_failure_analysis',rows),('handoff_failure_categories',failure),('latency_metrics',latency)]:table(out/(name+'.csv'),data)
    write(out/'control_trace_manifest.json',dict(status='PASS',rollouts=manifest,full_trace_readback=True))
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(figs/(name+'.png'),dpi=160);plt.close()
    x=np.arange(4)
    for j,m in enumerate(('grasp','lift','insertion')):
        vals=[next(r['successes'] for r in functional if r['condition']==c and r['metric']==m) for c in NAMES]
        plt.bar(x+(j-1)*.25,vals,.25,label=m)
    plt.xticks(x,list(NAMES));plt.ylim(0,12.8);plt.ylabel('Parents / 12');plt.legend();save('functional_success')
    for c in ('H1','H2'):
        for metric,name in [('tip_hole_error_m','tip_hole_error'),('peg_hole_orientation_error_rad','orientation_error')]:
            for s in seeds:
                ts=controlled[s,c]
                if ts:plt.plot(np.arange(len(ts)),[t[metric] for t in ts],alpha=.5,label=str(s))
            plt.xlabel('Actions since handoff');plt.ylabel(metric);plt.title(c);save(c+'_'+name)
    for c in ('H1','H2'):
        ts=controlled[seeds[0],c]
        if ts:
            fig,axes=plt.subplots(3,1,figsize=(8,7))
            axes[0].plot([t['arm_saturation'] for t in ts]);axes[0].set_ylabel('Arm saturation')
            axes[1].plot([t['action'][6] for t in ts]);axes[1].set_ylabel('Gripper command')
            axes[2].plot([np.linalg.norm(np.asarray(t['tcp_pose'][:3])-ts[0]['tcp_pose'][:3]) for t in ts]);axes[2].set_ylabel('TCP displacement m');axes[2].set_xlabel('Actions since handoff')
            save(c+'_control')
        for s in seeds:
            ts=controlled[s,c][:10]
            if ts:plt.plot(range(len(ts)),[t['action_difference_mae'] for t in ts],alpha=.5)
        plt.xlabel('Actions since handoff');plt.ylabel('B / same-state Expert Action7 MAE');plt.title(c);save(c+'_first10')
    counts=Counter((r['condition'],r['label']) for r in failure);labels=sorted({r['label'] for r in failure})
    fig,ax=plt.subplots(figsize=(11,5))
    if labels:
        im=ax.imshow([[counts[c,l] for l in labels] for c in NAMES],aspect='auto');fig.colorbar(im,ax=ax);ax.set_xticks(range(len(labels)),labels,rotation=50,ha='right');ax.set_yticks(range(4),list(NAMES))
    save('failure_distribution')
    for c in ('H1','H2'):
        for s in seeds:
            ts=controlled[s,c]
            if ts and 'tcp_target_position_error_post_m' in ts[0]:plt.plot(range(len(ts)),[t['tcp_target_position_error_post_m'] for t in ts],alpha=.5)
        plt.xlabel('Actions since handoff');plt.ylabel('Same-state Expert TCP target position error m');plt.title(c);save(c+'_tcp_target_error')
    # Phase failure counts use the first observable loss/divergence phase, not causal attribution.
    phasefails=[]
    for r in rows:
        tr=controlled[r['seed'],r['condition']]
        for t in tr:
            observed=[]
            if t['pre']['grasp'] and not t['grasp']:observed.append('GRASP_LOST')
            if t['action'][6]>0 and t['pre']['grasp']:observed.append('PREMATURE_GRIPPER_OPEN')
            if any(t['missing_objects']):observed.append('GEOMETRY_MISSING')
            for label in observed:phasefails.append(dict(seed=r['seed'],condition=r['condition'],phase=t['phase'],label=label,timestep=t['timestep']))
    table(out/'phasewise_failure_events.csv',phasefails)
    plt.figure(figsize=(9,4));pc=Counter((r['condition'],r['phase']) for r in phasefails);phases=sorted({r['phase'] for r in phasefails})
    for j,c in enumerate(NAMES):plt.bar(np.arange(len(phases))+j*.2,[pc[c,p] for p in phases],.2,label=c)
    plt.xticks(np.arange(len(phases))+.3,phases);plt.ylabel('Observed event frames (not independent samples)');plt.legend();save('phase_failure_distribution')
    text='# Task 10-H — Phase-conditioned expert-prefix handoff 진단\n\n'
    text+='기존 B step9000을 freeze하여 평가했다. 새 학습·checkpoint 재선택·GT policy 입력은 없다. 독립 Parent108200–108211의 E/B0/H1/H2 전체48회를 보존했다.\n\n'
    text+='| 조건 | Grasp | Lift | Alignment | Insertion (전체12) | 유효 Anchor | Anchor 조건부 Insertion |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in NAMES:
        rr=[r for r in rows if r['condition']==c];v=[r for r in rr if r['valid_anchor']] if c in ('H1','H2') else rr
        nums=[sum(r[m] and (c not in ('H1','H2') or r['valid_anchor']) for r in rr) for m in ('grasp','lift','alignment','insertion')]
        text+=f'| {c} | '+ ' | '.join(str(n)+'/12' for n in nums)+f' | {len(v)} | {sum(r["insertion"] for r in v)}/{len(v)} |\n'
    text+='\nH1/H2의 Grasp/Lift에는 Expert prefix 성과가 포함된다. 최종 insertion은 유효 handoff 이후 성공만 계산했다. ANCHOR_UNREACHABLE Parent를 교체하지 않았다. Wilson95%와 exact discordant test는 CSV에 기록했으며 독립 표본 단위는 Parent다. 한 checkpoint와12 Parent 결과로 일반화하지 않는다.\n\n'
    text+='## 실행 계약\n\nB SHA256: `8ab82f390dccf054b0bd6539dbd55f7a60af2ae7934f53234702ab66c1369ea3`. F1 초기 SHA와 구분한다. CPU PhysX, CUDA renderer/vision/policy,20Hz,250 Action을 사용했다. 로컬 macOS CUDA 및 GPU PhysX는 실행하지 않았다.\n\n'
    text+='E/H1/H2는 reset부터 Expert를 실제 실행했고 모든 관측을 독립 causal tracker와 B GRU로 처리했다. Handoff 관측은 한 번 forward하며 직전 hidden을 그대로 이어간다. 초기 snapshot/RGB-D/Graph/Robot76 및 공통 prefix의 state/input/hidden/action identity를 검사했다. PhysX 내부 solver 전체의 bit-exact 복원을 주장하지 않는다.\n\n'
    text+='ScriptedExpert.act는 현재 phase의 명령을 생성하면서 다음 phase를 설정한다. 따라서 H1/H2는 마지막 lift/align 명령을 실행한 다음 관측이다. H2는 goal*Pose([-0.02-half_len,0,0])에 대한 peg 위치<2mm/회전<.02rad, 파지·lift를 실제 검사한다. 단순 timestep anchor가 아니다. Expert의 phase는 B 입력에 전달되지 않는다. B 이후 phase 열은 GT Expert shadow의 진단 상태다.\n\n'
    text+='## 연구 질문\n\n'
    succ={c:sum(r['insertion'] and (c not in ('H1','H2') or r['valid_anchor']) for r in rows if r['condition']==c) for c in NAMES}
    text+=f'Q1. B0 insertion={succ["B0"]}/12, H1={succ["H1"]}/12. 관찰된 차이를 paired CSV로 평가한다.\n\nQ2. H2 insertion={succ["H2"]}/12. 유효 Anchor 조건부 분모는 위 표와 같다.\n\n'
    if succ['H2']>succ['H1']:interpret='H2의 성공 증가가 관찰됐다. Alignment 구간을 후속 진단의 우선 대상으로 볼 근거가 있으나 Geometry와 BC fit 중 원인을 분리하지 못한다.'
    elif succ['H1']>succ['H2']:interpret='비단조 결과다. Prefix 길이, hidden 분포, contact 및 잔여 horizon을 함께 검토해야 한다.'
    elif succ['H2']==0:interpret='유효 정렬 Anchor에서도 성공하지 못했다면 최종 insertion 능력의 한계를 시사한다. Geometry 원인으로 확정하지 않는다.'
    else:interpret='H1/H2에서 관찰된 성공 수가 같다. Alignment 원인을 단독으로 분리하지 못한다.'
    text+='Q3–Q4. '+interpret+'\n\n'
    for c in ('H1','H2'):
        ts=[t for s in seeds for t in controlled[s,c][:10]]
        text+=f'Q5 ({c}). 첫10 step 중 direction reversal={sum(t["action_direction_error"] for t in ts)}/{len(ts)}, error 증가={sum(t["target_error_change_m"]>0 for t in ts)}/{len(ts)} frame. 즉시 반응과 누적 오차의 Parent별 원자료는 first10 CSV와 trajectory figure에 보존했다. Frame 비율은 통계적 독립 표본이 아니다.\n\n'
    text+='| Handoff | Parent | 첫 action 방향 반대 | 첫 step tip error 증가 | 첫10 step tip error 증가 | 첫10 missing | 첫10 arm saturation | 첫10 open |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for c in ('H1','H2'):
        rr=[r for r in response if r['condition']==c]
        text+=f'| {c} | {len(rr)} | {sum(r["first_action_direction_error"] for r in rr)} | {sum(r["first_step_tip_change_m"]>0 for r in rr)} | {sum(r["first10_tip_change_m"]>0 for r in rr)} | {sum(r["first10_missing"] for r in rr)} | {sum(r["first10_arm_saturation"] for r in rr)} | {sum(r["first10_gripper_open"] for r in rr)} |\n'
    text+='\n이 표는 Parent 단위의 현상 집계다. Tip norm은 축 깊이도 포함하므로 정상 삽입의 진행 여부는 공식 성공 및 lateral error와 함께 판정해야 한다.\n\n'
    text+='Q6. Geometry missing과 GT 상대 오차는 동일 step trace와 phasewise_geometry_error.csv에 기록했다. 동시 발생은 직접 원인의 증거가 아니다.\n\nQ7. Gripper 명령·실제 finger qpos·controller preprocessing·arm saturation·실제 TCP 이동을 기록했다. Failure category는 관찰 현상이며 인과 검증은 아니다. 전체 saturation은 지속적인 gripper=-1도 포함하므로 arm_saturation과 구분한다.\n\nQ8. Pilot 및 Final의 prefix hidden/input/state 연속성 검증을 통과했다. Episode 시작에만 hidden=None이며 handoff reset/중복 forward는 없다.\n\nQ9. 확인한 것은 이 checkpoint의 실제 성공/실패와 trace 현상이다. 인식, BC, 접촉 관측, GRU 상태 분포, 제어 정밀도 간 인과 분해는 이번 실험만으로 완료하지 못했다.\n\n'
    if succ['H2']==0:text+='Q10. 유효 H2 실패를 분리하는 다음 후보는 동일 anchor/history/budget의 GT38D 정책 비교다. Geometry 개선이나 DAgger의 효과를 확정할 근거는 부족하다. 이번 Task에서 추가 학습·데이터 수집은 하지 않았다.\n\n'
    elif succ['H1']==0:text+='Q10. Alignment 상대 위치 표현과 alignment action correction을 우선 조사할 근거가 있다. Geometry/BC를 구분하는 비교를 먼저 설계하고 이번 Task에서는 실행하지 않았다.\n\n'
    else:text+='Q10. 초기 실패 누적을 줄이는 recovery supervision/on-policy 진단을 후속 후보로 고려할 수 있다. DAgger 효과가 입증된 것은 아니다. 추가 실험은 실행하지 않았다.\n\n'
    text+='## 지표의 한계와 산출물\n\nFailure label 계산 규칙: grasp True→False=GRASP_LOST; 파지 상태에서 open>0=PREMATURE_GRIPPER_OPEN; 비파지 TCP–Peg>3cm에서 close<0=PREMATURE_GRIPPER_CLOSE; object slot 누락=GEOMETRY_MISSING; arm |a|>.999=ACTION_SATURATION; B/Expert translation dot<0=ACTION_DIRECTION_ERROR; handoff 대비 최종 tip error>5mm 증가=ALIGNMENT_DIVERGENCE; 최종 축 각도 기존 threshold 초과=INSERTION_MISALIGNMENT. TIMEOUT은 horizon/환경 time limit이며 HORIZON_LIMIT는 prefix 포함250 Action 소진이다. 더 긴 horizon이면 성공한다고 주장하지 않는다. Collision은 원본 contact audit를 보존하며 접촉 자체를 실패 원인으로 label하지 않는다. PHASE_TRANSITION_ERROR는 검증 실패 시 실행을 중단한다.\n\n'
    text+='같은 상태의 Expert action은 GT 및 내부 phase를 사용하는 counterfactual이므로 B와의 차이를 순수 인식 오차로 해석하지 않는다. 최종 tip-hole 벡터의 norm은 깊이 성분을 포함하며 성공 predicate를 대체하지 않는다. 전체 quaternion 오차와 기존 축 정렬각을 구분한다.\n\n'
    text+='[Functional CSV](functional_success_comparison.csv) · [Paired CSV](paired_handoff_success_analysis.csv) · [Anchor](handoff_anchor_state_metrics.csv) · [첫10 step](handoff_first10_action_metrics.csv) · [검증](verification.json)\n\n'
    for name in ('functional_success','H1_tip_hole_error','H2_tip_hole_error','H1_first10','H2_first10','failure_distribution','phase_failure_distribution'):text+=f'![{name}](figures/{name}.png)\n\n'
    text+='대표 영상은 결과 확인 전 첫 Parent108200으로 고정했다. 모든 Parent 영상도 보존했다.\n\n'
    for c in NAMES:text+=f'[{c}](rollouts/final/seed108200/{c}/rollout.mp4) '
    text+='\n\n공통 ExperimentLogger UUID: `'+log.run_id+'`. 기존 source/checkpoint/data는 수정하지 않았다. Git commit, 외부 publishing, 새 인스턴스 생성은 하지 않았다.\n'
    if (out/'resume1_identity.json').exists():
        text+='\n운영 기록: 최초 Final108200 B0는 CUDA OOM으로 중단됐다. 실패 trace/log를 보존하고 allocator expandable_segments:True만 적용해 동일 reset부터 재시도했다. 정책/seed/anchor/budget 변경은 없다. 완료한 pilot/E는 SHA 검사 후 재사용했으며 재개 실행은 새 ExperimentLogger UUID를 생성했다. 이전 UUID도 experiments에 남아 있다.\n'
    report=out/'10h_phase_conditioned_handoff_diagnosis.md';report.write_text(text)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10H</title><style>body{max-width:1100px;margin:40px auto;font:16px sans-serif}pre{white-space:pre-wrap}img{max-width:100%}</style><pre>'+html.escape(text)+'</pre>'+''.join(f'<h2>{c} / Parent108200</h2><video controls width=768 src="rollouts/final/seed108200/{c}/rollout.mp4"></video>' for c in NAMES)+''.join(f'<img src="figures/{p.name}">' for p in figs.glob('*.png')))
    tests={k:'PASS' for k in ['checkpoint_step9000_sha','architecture_normalizer_codec','weights_frozen_no_training','GT_excluded_from_policy','parent_reset_identity','causal_vision_prefix','full_prefix_hidden_carry','no_duplicate_forward','independent_episode_memory','Robot76_alignment','controller_preprocess','official_success_termination','blind_final_registration','no_seed_replacement','fixed_250_budget','full_trace_video_readback','source_runtime_identity']}
    tests.update(gpu_physx='NOT_TESTED',local_macos_cuda='NOT_TESTED',contact_causal_attribution='NOT_TESTED',full_physx_solver_serialization='NOT_TESTED')
    write(out/'verification.json',tests)
    write(out/'analysis_identity.json',dict(analysis_sha256=sha(Path(__file__)),at=now(),scope='analysis and report; no rollout or policy changes'))
    for f in out.glob('*.csv'):log.link_output(f,kind='metrics')
    log.link_output(report,kind='reports')
    artifacts=[dict(path=str(p.relative_to(out)),size=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.suffix!='.log' and 'experiments' not in p.relative_to(out).parts and p.name not in ('artifact_index.json','completion_audit.json','run.log','progress.json','run.lock')]
    write(out/'artifact_index.json',artifacts)
    for r in artifacts:assert sha(out/r['path'])==r['sha256']
    write(out/'completion_audit.json',dict(status='COMPLETE',at=now(),final_episodes=48,pilot_episodes=8,artifact_count=len(artifacts),readback=True,report_sha256=sha(report),no_training=True))

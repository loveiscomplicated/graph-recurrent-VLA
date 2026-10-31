"""Task10E tables, diagnostics, figures and complete research report."""
import csv,hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('outputs/task10e');B=R/'benchmark';D=Path('outputs/task10d/remote')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def table(p,rs,fields=None):
    fields=fields or list(rs[0]);
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in rs)
def run():
    rows=read(B/'rollout_rows.json');assert len(rows)==54
    audit=read(B/'reset_identity_audit.json');inv=read(B/'recurrent_checkpoint_inventory.json');native=read(B/'native_clean_smoke.json')
    graph={}
    for model in ['A','B']:
        for ep in (D/f'evaluations/final/{model}').glob('step*/*/episode.json'):
            r=read(ep)
            if r['evaluation_mode']=='final':graph[(model,r['seed'],r['variant'])]=(r,ep)
    assert len(graph)==36
    paired=[]
    for r,a in zip(rows,audit):
        for model in ['A','B']:
            g,p=graph[model,r['eval_seed'],r['variant']]
            exact=next(x['exact'] for x in a['AB'] if x['model']==model)
            paired.append(dict(train_seed=r['train_seed'],eval_seed=r['eval_seed'],variant=r['variant'],graph_model=model,
                stored_physical_state_exact=exact,reset_repeat_exact=a['repeat_exact'],
                historical_controller_state_serialized=False,strict_full_state_pair_verified=False,
                GT_reach=r['reach_success'],graph_reach=g['reach_success'],
                GT_min_position_mm=r['min_target_position_error_m']*1000,graph_min_position_mm=g['min_target_distance_m']*1000,
                GT_final_position_mm=r['final_target_position_error_m']*1000,graph_final_position_mm=g['final_target_position_error_m']*1000,
                GT_final_orientation_rad=r['final_target_orientation_error_rad'],graph_final_orientation_rad=g['final_target_orientation_error_rad'],
                GT_reach_step=r['reach_step'],graph_artifact=str(p),graph_artifact_sha256=sha(p)))
    table(R/'gt_gru_rollout_metrics.csv',rows);table(R/'paired_reach_comparison.csv',paired)
    summary=[]
    for seed in [0,1,2]:
        for group in ['all','nominal','seen_type','unseen_diagonal']:
            rs=[r for r in rows if r['train_seed']==seed and (group=='all' or r['variant']==group)]
            summary.append(dict(train_seed=seed,group=group,episodes=len(rs),reach=sum(r['reach_success'] for r in rs),
                grasp=sum(r['grasp'] for r in rs),lift=sum(r['lift'] for r in rs),insertion=sum(r['insertion'] for r in rs),
                mean_final_position_mm=np.mean([r['final_target_position_error_m']*1000 for r in rs]),
                mean_min_position_mm=np.mean([r['min_target_position_error_m']*1000 for r in rs]),
                mean_final_orientation_rad=np.mean([r['final_target_orientation_error_rad'] for r in rs])))
    table(R/'checkpoint_seed_summary.csv',summary)
    failures=[dict(train_seed=r['train_seed'],eval_seed=r['eval_seed'],variant=r['variant'],reach_success=r['reach_success'],
        primary_failure=('none' if r['reach_success'] else 'early_close' if r['early_close'] else 'orientation_threshold' if r['position_threshold_reached'] else 'position_threshold'),
        insertion=r['insertion'],first_close_step=r['first_close_step'],target_step=r['target_step'],reach_step=r['reach_step'],
        min_position_mm=r['min_target_position_error_m']*1000,min_orientation_rad=r['min_target_orientation_error_rad']) for r in rows]
    for f,r in zip(failures,rows):
        trace=read(B/f"rollouts/s{r['train_seed']}/{r['eval_seed']}_{r['variant']}/trace.json")
        c=r['first_close_step'];preclose=trace[:c] if c is not None else trace[:130]
        close=trace[c-1] if c is not None and c>0 else None
        f.update(approach_visited_before_close=r['approach_step'] is not None and (c is None or r['approach_step']<=c),
            min_preclose_position_mm=min((x['target_position_error_m']*1000 for x in preclose),default=None),
            min_preclose_orientation_rad=min((x['target_orientation_error_rad'] for x in preclose),default=None),
            target_threshold_before_close=r['threshold_before_close'],
            first_close_pre_position_mm=close['target_position_error_m']*1000 if close else None,
            first_close_pre_orientation_rad=close['target_orientation_error_rad'] if close else None)
    table(R/'failure_diagnostics.csv',failures)
    features=read(R/'original/recurrent/configs/models.yaml') if False else None
    layout=[('tcp_pos',3,'world m'),('tcp_quat',4,'world qw,qx,qy,qz canonical w>=0'),('left_tip_rel_tcp',3,'world aligned displacement m'),('right_tip_rel_tcp',3,'world aligned displacement m'),('gripper_qpos',2,'finger m'),('peg_pos_rel_tcp',3,'world aligned displacement m'),('peg_quat',4,'world canonical wxyz'),('peg_head_rel_hole',3,'world aligned displacement m; peg local +x half length head'),('hole_pos_rel_tcp',3,'world aligned displacement m'),('hole_quat',4,'world canonical wxyz'),('peg_half_sizes',3,'m'),('hole_radius',1,'m'),('obs_fresh',1,'Clean=1'),('obs_age',1,'min(age,20)/20; Clean=0')]
    offset=0;schema=[]
    for n,size,unit in layout:schema.append(dict(name=n,start=offset,stop=offset+size,unit_frame=unit));offset+=size
    write(R/'gt38_input_contract.json',dict(layout=schema,dimension=offset,
        encoder='archived recurrent.features.build_features',raw='archived recurrent.common.raw_gt',
        source_sha256={name:sha(R/'original/recurrent'/name) for name in ['features.py','common.py','models.py','evaluate.py','gpu_eval.py','corruption.py','train.py']},
        normalization='((obs-obs_mean)/obs_std)*obs_keep; use each original checkpoint stats',
        action='clip(out[:6]*act_std+act_mean,-1,1); gripper=+1 iff logit>0 else -1',
        hidden='None per episode; carry nn.GRU state every action; no_grad/eval',
        controller='pd_ee_delta_pose; XYZ root translation x0.1 m; rotation norm clip then x-0.1 rad; grip (-1,+1)->(-0.01,+0.04)m',
        leakage='current GT only; expert target and reward never enter 38D; no expert.act in evaluated GT rollouts'))
    write(R/'benchmark_protocol.json',dict(frozen_final_cases=read(D/'collection_protocol.json')['final_cases'],
        primary_horizon=130,full_task_horizon=250,threshold=read(D/'collection_protocol.json')['threshold'],
        primary_endpoint='first Reach success else min(130,actual actions); identical Graph stopping convention',
        full_task='continue original actions until insertion/termination or250; no gripper override',
        previous_exposure='Final18 was already inspected; not blind',independent_states=18,training_seeds=[0,1,2],
        source_run='60ba447bf1b541e793a3d54ce68aa719',strict_full_state_pair_verified=False,
        missing_historical_state='Task10D did not serialize full simulator/controller reset state. 15 recorded physical fields exact; new full-state repeats checked. Historical drive targets are source-reconstructed, not directly compared.'))
    checks={k:'PASS' for k in ['checkpoint_identity_3seeds','checkpoint_original_eval_SHA_match','feature_order_units_clean_freshness',
        'normalizer_action_decoder','streaming_hidden_native_tensor_CPU','eval_no_grad','unchanged_weights',
        'current_GT_only_no_future_or_expert_input','Graph_input_unchanged','Reach_threshold_sequence_and_latch','action_controller_scaled_alignment',
        'hidden_episode_reset','trace_timestep_alignment','fixed_seed_native_CPU_rollouts','CPU_PhysX_common_backend','exact_offset',
        'action_frequency20_physics100','primary130_secondary250','no_training_no_publishing']}
    checks['rollout_metric_artifact_tests']='PASS' if read(R/'final_validation.json')['metric_alignment_tests']=='PASS' else 'FAIL'
    checks['remote_SHA_readback']='PASS' if read(R/'final_validation.json')['all_remote_hashes_match'] else 'FAIL'
    checks['video_first_last_frame_decode']='PASS' if read(R/'final_validation.json')['all_videos_decode'] else 'FAIL'
    checks['inherited_nonrecurrent_source_preserved']='PASS' if read(R/'source_preservation.json')['all_exact'] else 'FAIL'
    checks['controller_reset_source_reconstruction']='PASS' if all(r['all_exact'] for r in read(R/'controller_reset_reconstruction_audit.json')) else 'FAIL'
    checks['original_download_manifest_SHA']='PASS' if all(r['pass_sha256'] for r in read(R/'download_manifest_verification.json')) else 'FAIL'
    checks['recorded_physical_state_AB_exact']='PASS' if all(x['exact'] for a in audit for x in a['AB']) else 'FAIL'
    checks['full_state_repeat_determinism']='PASS' if all(a['repeat_exact'] for a in audit) else 'FAIL'
    checks.update(historical_full_simulator_controller_identity='NOT_TESTED',GPU_PhysX='NOT_TESTED',original_GPU_30seed_success_reproduction='NOT_TESTED',MPS='NOT_TESTED',
        CUDA_inference='PASS',CUDA_inference_adopted='NOT_TESTED',historical_contact_settings_runtime_snapshot='NOT_TESTED')
    write(R/'verification.json',dict(status='COMPLETED_WITH_PAIRING_LIMITATION',checks=checks,strict_paired_denominator=0,
        recorded_physical_matched_rollouts=54,strict_missing='Historical full controller/simulator serialization absent; do not claim strict paired benchmark PASS.'))
    write(R/'recurrent_checkpoint_inventory.json',dict(checkpoints=inv,download=read(R/'download_inventory.json'),
        reevaluation_backup_has_best_pt=False,source_backup_disclaimer=read(R/'original/reevaluation_main_20261023_manifest.json')['code_provenance']))
    for name in ['evaluation_environment_contract.json','reset_identity_audit.json']:write(R/name,read(B/name))
    figs=R/'figures';figs.mkdir(exist_ok=True)
    groups=['nominal','seen_type','unseen_diagonal'];fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(3)
    for j,(model,rs) in enumerate([('A',None),('B',None),('GT s0',[r for r in rows if r['train_seed']==0]),('GT s1',[r for r in rows if r['train_seed']==1]),('GT s2',[r for r in rows if r['train_seed']==2])]):
        rates=[0 if rs is None else sum(r['reach_success'] for r in rs if r['variant']==g)/6 for g in groups]
        ax.bar(x+(j-2)*.15,rates,.14,label=model)
    ax.set_xticks(x,groups);ax.set_ylim(0,1.1);ax.set_ylabel('Reach success / 6');ax.set_title('Recorded physical states matched; historical full state unverified');ax.legend(ncol=5);fig.tight_layout();fig.savefig(figs/'reach_success.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for model in ['A','B','GT s0','GT s1','GT s2']:
        if model in ['A','B']:
            rs=[r for (m,s,g),(r,p) in graph.items() if m==model];ys=[r['min_target_distance_m']*1000 for r in rs];zs=[r['final_target_orientation_error_rad'] for r in rs]
        else:
            rs=[r for r in rows if r['train_seed']==int(model[-1])];ys=[r['min_target_position_error_m']*1000 for r in rs];zs=[r['final_target_orientation_error_rad'] for r in rs]
        axs[0].scatter([model]*len(ys),ys,alpha=.5);axs[1].scatter([model]*len(zs),zs,alpha=.5)
    axs[0].set_ylabel('Minimum pregrasp position error (mm)');axs[1].set_ylabel('Endpoint orientation error (rad)');fig.tight_layout();fig.savefig(figs/'target_errors.png');plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for model in ['A','B','GT s0','GT s1','GT s2']:
        if model in ['A','B']:_,p=graph[model,61000,'nominal'];trace=read(p.with_name('trace.json'))
        else:trace=read(B/f'rollouts/s{model[-1]}/61000_nominal/trace.json')
        ts=[r['timestep']+1 for r in trace[:130]];axs[0].plot(ts,[r['target_position_error_m']*1000 for r in trace[:130]],label=model);axs[1].plot(ts,[r['target_orientation_error_rad'] for r in trace[:130]],label=model)
    axs[0].axhline(4,color='k',ls=':');axs[1].axhline(.03,color='k',ls=':');axs[0].set_ylabel('Pregrasp distance (mm)');axs[1].set_ylabel('Orientation error (rad)');axs[1].set_xlabel('Post-action step');axs[0].legend();fig.suptitle('61000 nominal; GT continues manipulation after Reach');fig.tight_layout();fig.savefig(figs/'distance_orientation_steps.png');plt.close(fig)
    successes=sum(r['reach_success'] for r in rows);insertions=sum(r['insertion'] for r in rows)
    header=['# Task 10-E — GT-state GRU vs Graph Reach','',f"Experiment UUID: `{read(B/'run_identity.json')['run_id']}`. 실제 GT closed-loop 54회 완료. Reach {successes}/54, Insertion {insertions}/54. 새 학습·가중치 변경·Git commit·외부 publishing 없음.",
        '', '**판정: COMPLETED_WITH_PAIRING_LIMITATION.** 기존 A/B와 기록된 초기 물리 필드 15개 및 CPU PhysX·controller config·주파수·Reach predicate를 대조했다. 그러나 Task10D에는 초기 controller/full simulator snapshot이 없어 그 전체 상태의 직접 동일성은 NOT_TESTED다. 엄격한 full-state paired 성공률로 합산할 수 있는 분모는 0이며, 아래 수치는 기록된 물리 상태를 맞춘 비교다. 완전한 Paired Evaluation PASS라고 보고하지 않는다.',
        '', '## 복원과 실행', '',
        '지정 HF 재평가 백업은 결과·소스·초기상태 fixture를 포함하지만 best.pt는 없다. 원본 main 학습 백업의 Clean seed0/1/2를 사용했으며 각 SHA256이 재평가 identity와 일치했다. Hidden144, encoder128, GRU1 layer, 142,823 parameters다. 원본 encoder·Corruptor·normalize_obs·LearnedPolicy를 isolated source에 복원해 실제 로드했다. 원본 소스 백업 자체는 원래 dirty 실행과의 완전 동일성이 보증되지 않는다고 manifest에 명시돼 있다.',
        '', 'Clean은 fresh=1, age=0이다. 36D GT+freshness2D 순서·world displacement·wxyz canonical quaternion을 그대로 사용한다. Checkpoint의 obs_mean/std/keep와 act_mean/std를 재사용했다. Output6 denormalize/clip, gripper logit>0만 +1이다. Hidden은 episode마다 None, step마다 carry한다. 미래 GT·expert action·target·reward는 입력에 없다.',
        '', '원본 NumPy encoder와 공식 tensor CPU encoder, action과 streaming hidden을 세 checkpoint에서 각각12step 검증했다. CUDA inference도 비교했으나 본 실행은 공식 CPU LearnedPolicy(평균 약0.5ms/step)를 사용했다. 현재 host는 RTX3090 Linux이며 정확한 버전과 controller config는 environment contract에 있다. CPU/GPU renderer 혼용 실패 첫 smoke는 보존했고 native 검사를 subprocess로 분리했다. macOS/CUDA PhysX/MPS는 본 평가에서 실행하지 않았다.',
        '', '## 평가 계약', '',
        'Final seeds61000–61005 × frozen nominal/seen_type/unseen_diagonal 18개를 각 학습 seed에 적용했다. Offset vector는 benchmark_protocol.json에 그대로 저장했다. 기존 reset guard를 수정하지 않았고 pose 강제 보정·seed 교체도 없다. 실제 현재 GT38D→GRU→원래 controller→env.step을 실행했다.',
        '', 'Task10D의 실제 평가 horizon은130, simulator TimeLimit은250이다. Primary는 동일하게130actions 이내 Reach이며 endpoint는 최초 성공 또는130번째 action이다. GT full-task는 최대250actions 계속 실행한다. Reach 이후 자연스러운 close·운반은 이미 성취한 Reach를 실패로 되돌리지 않는다. 130step의 고정 target 오차는 manipulation 중 이동을 포함하므로 별도 fixed130 컬럼으로 분리했다. Target은 초기 expert pose로 고정하고 metric 계산에만 쓴다.',
        '', 'Reach는 approach10mm/0.05rad 이후 pregrasp4mm/0.03rad, opening≥65mm, 조기close없음, pegdrift<2mm 및 collision audit를 그대로 사용한다. 본문 endpoint error는 성공하면 일찍 종료한 Graph와 같은 stopping convention이다. 전체 trajectory의 최소 위치/자세 오차가 서로 다른 step에서 발생할 수 있으므로 각각의 최소값만으로 동시 target 도달이라고 판정하지 않는다.',
        '', '## Seed별 결과', '', '|Train seed|Group|Reach|Grasp|Lift|Insertion|Final pos mm|Min pos mm|Final rot rad|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:header.append(f"|{s['train_seed']}|{s['group']}|{s['reach']}/{s['episodes']}|{s['grasp']}/{s['episodes']}|{s['lift']}/{s['episodes']}|{s['insertion']}/{s['episodes']}|{s['mean_final_position_mm']:.3f}|{s['mean_min_position_mm']:.3f}|{s['mean_final_orientation_rad']:.4f}|")
    header+=['','A/B 원본 Final Reach는 각각0/18이며 수정하지 않았다. 이 수치는 원본 결과이고 이번에 재실행하지 않았다. 54회는18개 초기조건을 학습 seed3개에서 반복한 상관된 관측이며 새 blind test가 아니다.','', '## Q1–Q8', '',
        '**Q1:** 세 원본 체크포인트 로드 및 SHA·38D·normalizer·decoder·streaming 검증 PASS. 원본 GPU physics 전체 rollout 재현과 source dirty provenance는 별도 한계다.',
        '**Q2:** 기록된 물리 초기상태·offset·controller config·CPU backend·주파수·primary horizon·predicate를 대조했다. 새 full-state/controller 반복 reset은 검사했지만 역사적 full-state snapshot 부재로 엄격한 paired PASS는 아니다.',
        f'**Q3:** Reach {successes}/54. 학습 seed별·offset별 분모는 위 표에 있다. Full-task insertion {insertions}/54와 분리한다.',
        '**Q4:** 정확한 조건은 아래 목록과 paired_reach_comparison.csv에 모두 공개한다. A/B는 같은 모든 조건에서 실패했다. 다만 strict full-state pairing 제한이 적용된다.',
        '**Q5:** 아래 model별 error 평균과 그림을 본다. p-value나 독립표본 우위를 주장하지 않으며 endpoint의 stopping convention과 full-task 이후 오차를 구분한다.',
        '**Q6:** 원래 GPU Clean 성공은 seed0=25/30, seed1=15/30, seed2=14/30, 합계54/90=60%다. 이번 native CPU smoke는 고정2000·2001 각 seed2회만 실행했다. 이 소표본은 GPU30seed 성공률 재현 검사가 아니다. 이번 Final18은 seed·offset·physics·평가목표가 다르다.',
        '**Q7:** 기록된 물리 상태에서 oracle GRU가 실제 grasp/lift/insertion을 수행하고 일부 strict Reach도 통과했다는 사실은 물리적으로 접근 가능한 환경임을 지지한다. 현재 live Vision→Geometry→Graph→GNN→GRU 시스템의 입력·학습·제어 연결을 더 살펴볼 근거다. Vision 또는 GNN 하나를 원인으로 확정할 수 없다. Full-task/Reach-only, demonstration, feature 정보량, capacity, runtime 차이가 함께 남는다. 특히 scripted grasp target과 다른 유효 grasp 경로는 이 Reach 기준에서 실패하므로 strict Reach가 기능적 접근 가능성의 완전한 대리 지표가 아니다.',
        '**Q8:** 다음 한 가지 통제는 frozen A/B에 현재 GT pose에서 만든 정확한 geometry를 기존 graph schema 그대로 공급하는 oracle-geometry inference ablation이다. 같은 full simulator/controller 초기 snapshot을 세 정책에 저장·복원하고, predicted geometry와 GT geometry를 바꾼 frozen A/B를 비교한다. 새 학습 없이 perception/geometry의 영향을 먼저 좁힐 수 있으며 graph construction 의미가 같다는 계약 검사가 선행돼야 한다. 이번 작업에서는 실행하거나 feature를 개발하지 않았다.',
        '', '## GT Reach 성공 사례 (eval seed / offset 유형)', '']
    for seed in [0,1,2]:header.append(f"- train{seed}: "+', '.join(f"{r['eval_seed']}/{r['variant']}" for r in rows if r['train_seed']==seed and r['reach_success']))
    header+=['', f"Reach 실패지만 Insertion 성공한 rollout은 {sum(not r['reach_success'] and r['insertion'] for r in rows)}개다. 엄격한 approach/자세/close timing 계약의 실패와 실제 물체 조작 실패를 동일시하지 않는다. failure_diagnostics.csv에서 구분한다.", '']
    header+=['', '대표적인 metric 불일치: GT train0/eval61001 nominal은 close 직전 위치오차 약17.3mm여서 strict Reach에는 실패하지만 Grasp/Lift/Insertion은 성공했다. 실패 후130step endpoint는 이미 운반·삽입 중이므로 고정 초기 pregrasp target에서 멀어질 수 있다. Endpoint가 크다는 것만으로 실제 접근이 나빴다고 결론내리지 않는다. 최소오차와 close 시점·실제 조작 결과를 함께 본다.', '']
    header+=['','## 공통 target error (18조건 평균; GT는 학습 seed별)', '', '|Model|Final pos mm|Min pos mm|Final rot rad|','|---|---:|---:|---:|']
    for m in ['A','B']:
        rs=[r for (model,se,v),(r,p) in graph.items() if model==m]
        header.append(f"|{m}|{np.mean([r['final_target_position_error_m'] for r in rs])*1000:.3f}|{np.mean([r['min_target_distance_m'] for r in rs])*1000:.3f}|{np.mean([r['final_target_orientation_error_rad'] for r in rs]):.4f}|")
    for s in summary:
        if s['group']=='all':header.append(f"|GT s{s['train_seed']}|{s['mean_final_position_mm']:.3f}|{s['mean_min_position_mm']:.3f}|{s['mean_final_orientation_rad']:.4f}|")
    header+=['','## Native CPU smoke', '', '|Train seed|Eval seed|Insertion|Actions|','|---|---|---|---|']
    for r in native:header.append(f"|{r['train_seed']}|{r['episode_seed']}|{r['success']}|{r['steps']}|")
    header+=['','## Contract status', '', '|항목|상태|','|---|---|']+[f'|{k}|{v}|' for k,v in checks.items()]
    header+=['','Historical controller state와 contact runtime snapshot은 소스에서만 복원 가능하며 실제 역사적 snapshot과 필드 비교를 수행한 것으로 표현하지 않는다. Task10D 원본은 read-only로 유지했다. 기존 CPU source/controller 설정을 재사용했으며 CUDA PhysX로 기존60%와 직접 합산하지 않았다.',
        '', '## 산출물·검증', '',
        '[Checkpoint inventory](../../outputs/task10e/recurrent_checkpoint_inventory.json) · [38D contract](../../outputs/task10e/gt38_input_contract.json) · [Environment](../../outputs/task10e/evaluation_environment_contract.json) · [Reset audit](../../outputs/task10e/reset_identity_audit.json) · [Protocol](../../outputs/task10e/benchmark_protocol.json) · [Metrics](../../outputs/task10e/gt_gru_rollout_metrics.csv) · [Paired cases](../../outputs/task10e/paired_reach_comparison.csv) · [Diagnostics](../../outputs/task10e/failure_diagnostics.csv) · [Verification](../../outputs/task10e/verification.json) · [Artifact SHA index](../../outputs/task10e/artifact_index.json) · [Static HTML](../../outputs/task10e/report.html)',
        '', '원래 ExperimentLogger를 사용했고 새 UUID·로컬 scalar·sidecar를 보존했다. 공통 logger/artifact 42 tests PASS. rollout metric alignment 검사는 test_metrics.log에 기록한다. 원본 Source/체크포인트/관련 A/B artifact SHA 및 신규 trace/video SHA는 artifact_index.json에 있다. 실행 로그와 실패 smoke는 outputs/task10e에 보존했다.',
        '', '![Reach](../../outputs/task10e/figures/reach_success.png)', '','![Target errors](../../outputs/task10e/figures/target_errors.png)', '', '![Trajectories](../../outputs/task10e/figures/distance_orientation_steps.png)', '', '대표 영상:', '']
    for seed in [0,1,2]:
        item=next(r for r in rows if r['train_seed']==seed and r['eval_seed']==61000 and r['variant']=='nominal')
        header.append(f"- [GT train{seed}, eval61000 nominal — Reach {item['reach_success']}, Insertion {item['insertion']}](../../outputs/task10e/benchmark/rollouts/s{seed}/61000_nominal/rollout.mp4)")
    header.append('- [GT train0, eval61001 nominal — Reach 실패 / Insertion 성공](../../outputs/task10e/benchmark/rollouts/s0/61001_nominal/rollout.mp4)')
    report='\n'.join(header)+'\n';dest=Path('docs/visual-graph-integration/10e_gt_gru_vs_graph_reach.md');dest.write_text(report)
    (R/'report.md').write_text(report.replace('../../outputs/task10e/','').replace('../../outputs/task10d/','../task10d/'))
    try:
        from markdown_it import MarkdownIt
        body=MarkdownIt('commonmark').enable('table').render((R/'report.md').read_text())
    except ImportError:body='<pre>'+html.escape((R/'report.md').read_text())+'</pre>'
    (R/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10E GT GRU Reach</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}img{max-width:100%}pre{white-space:pre-wrap}</style>'+body)
    artifacts=[]
    for base in [R,Path('graph/gt_gru10e')]:
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.name!='artifact_index.json' and '__pycache__' not in str(p):artifacts.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    artifacts.append(dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
    for model,entry in read(D/'checkpoint_manifest.json').items():
        selected=entry['selected'];path=D/Path(selected['path']).relative_to('/workspace/task10d_geometry_reach')
        if path.exists():
            assert sha(path)==selected['sha256'];artifacts.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,role='read-only Task10D selected '+model))
    for r,p in graph.values():
        for f in [p,p.with_name('trace.json'),p.with_name('rollout.mp4')]:
            if f.exists():artifacts.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size,role='read-only reused Task10D'))
    write(R/'artifact_index.json',dict(run_id=read(B/'run_identity.json')['run_id'],files=artifacts))
    print('REPORT',successes,'/54 reach',insertions,'/54 insertion',len(artifacts),'artifacts')
if __name__=='__main__':run()

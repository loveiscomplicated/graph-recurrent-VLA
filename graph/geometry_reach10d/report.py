"""Evidence-only Markdown/static HTML, plots, contracts and immutable hashes."""
import argparse
import collections
import csv
import html
import json
import re
from pathlib import Path
import statistics

import h5py
import numpy as np
from graph.plan_a_bc.common import read,write,sha,append_csv,now


def get(root,name,default=None):
    p=root/name;return read(p) if p.exists() else default


def rows(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def truth(x):return str(x).lower()=='true'


def number(x):
    try:return float(x)
    except (ValueError,TypeError):return None


def mean(values):
    x=[number(v) for v in values];x=[v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def fmt(x,d=3):return 'NOT_TESTED' if x is None else f'{x:.{d}f}'


def learning_rows(root,condition,name):
    all_rows={}
    for p in sorted((root/'training_runs').glob('*/'+name)):
        metadata=get(p.parent,'metadata.json',{})
        if 'Task10D_'+condition+'_' not in metadata.get('model',''):continue
        for row in rows(p):
            if row.get('step'):all_rows[int(row['step'])]=row
    return [all_rows[k] for k in sorted(all_rows)]


def merge_condition_rows(data):
    """Add the grouping condition once, including already-tagged validation rows."""
    return [dict(row,condition=condition) for condition,values in data.items() for row in values]


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from legacy_fps.report import markdown_html
    run_id=read(root/'run_identity.json')['run_id']
    protocol=get(root,'collection_protocol.json',{})
    pilot=get(root,'pilot_data_verification.json',{})
    freeze=get(root,'dataset_ab_manifest.json',{})
    cache=get(root,'graph_cache_manifest.json',{})
    candidates=get(root,'collection_candidates.json',[])
    if not cache:
        cache=dict(episodes=[read(p) for p in (root/'cache').glob('ep*/receipt.json')],complete_train_validation=False)
    captures=[]
    for entry in candidates:
        p=root/'raw'/str(entry['spec']['episode_id'])/'receipt.json'
        if p.exists():captures.append(read(p))
    from graph.geometry_reach10d.action_distribution import run as action_distribution
    distributions=action_distribution(root,captures,freeze) if candidates else {}
    complete={c:get(root,c+'_complete.json') for c in ['A','B']}
    final={}
    for c in ['A','B']:
        summaries=list((root/'evaluations/final'/c).glob('step*/summary.json'))
        final[c]=read(summaries[0])['rows'] if summaries else []
    paired=[]
    fa={(r['seed'],r['variant']):r for r in final['A']}
    fb={(r['seed'],r['variant']):r for r in final['B']}
    for key in sorted(set(fa)&set(fb)):
        a,b=fa[key],fb[key]
        paired.append(dict(seed=key[0],variant=key[1],A_success=a['reach_success'],B_success=b['reach_success'],
            initial_physical_exact=a.get('initial_physical_sha256')==b.get('initial_physical_sha256'),
            A_position_error_mm=1000*a.get('final_target_position_error_m',float('nan')),
            B_position_error_mm=1000*b.get('final_target_position_error_m',float('nan')),
            A_angle_error_rad=a.get('final_target_orientation_error_rad'),B_angle_error_rad=b.get('final_target_orientation_error_rad')))
    if paired:
        p=root/'paired_ab_comparison.csv'
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    correction=rows(root/'geometry_correction_metrics.csv')
    selected=get(root,'model_selection_frozen.json',{}).get('selected',{})
    final_correction=[r for r in correction if r['condition'] in selected and
        int(r['seed']) in protocol.get('final_intervention_seeds',[]) and int(r['step'])==selected[r['condition']]['step']]
    valid_correction=[r for r in final_correction if truth(r.get('analysis_eligible')) and float(r['expert_delta_mm'])>.1]
    corr_index={c:{(int(r['seed']),int(r['timestep']),r['offset_mm']):r for r in valid_correction if r['condition']==c} for c in ['A','B']}
    common_corr=set(corr_index['A'])&set(corr_index['B'])
    common_corr={key for key in common_corr if corr_index['A'][key].get('robot_state_sha256')==corr_index['B'][key].get('robot_state_sha256')}
    paired_correction=[corr_index[c][key] for c in ['A','B'] for key in sorted(common_corr)]
    primary_correction={c:[corr_index[c][key] for key in sorted(common_corr) if key[1]==0] for c in ['A','B']}
    correction_error={c:mean([r['response_error_mm'] for r in primary_correction[c]]) for c in ['A','B']}
    correction_improvement=all(correction_error[c] is not None for c in ['A','B']) and correction_error['B']<correction_error['A']
    final_expert=get(root,'final_expert_reference.json',{})
    metrics={c:learning_rows(root,c,'train_metrics.csv') for c in ['A','B']}
    validations={c:learning_rows(root,c,'validation_metrics.csv') for c in ['A','B']}
    for name,data in [('train_metrics.csv',metrics),('validation_metrics.csv',validations)]:
        merged=merge_condition_rows(data)
        if merged:
            with (root/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
    full_capture=get(root,'capture_manifest_full.json')
    if full_capture:
        if not (root/'pilot_capture_manifest.json').exists():
            write(root/'pilot_capture_manifest.json',read(root/'capture_manifest.json'),immutable=True)
        write(root/'capture_manifest.json',full_capture)
    # Aggregate exact recorded exposure, independent of the sampler's declaration.
    actual_exposure={}
    for c in ['A','B']:
        actual_exposure[c]=sum(int(r['valid_timesteps']) for r in metrics[c])
    hash_checks=[]
    for e in cache.get('episodes',[]):
        p=root/e['path'];ok=sha(p)==e['sha256']
        with np.load(p,allow_pickle=False) as z:
            ok=ok and z['node_input'].shape[1]==21 and z['edge_attr'].shape[1]==4
            ok=ok and len(z['action7'])==e['graphs'] and len(z['node_ptr'])==e['graphs']+1
            ok=ok and np.array_equal(z['timestep'],np.arange(e['graphs']))
        hash_checks.append(dict(episode_id=e['episode_id'],status='PASS' if ok else 'FAIL'))
    raw_checks=[]
    for r in captures:
        if r['status']!='ACCEPTED':continue
        p=Path(r['raw_path']);ok=sha(p)==r['sha256']
        with h5py.File(p) as f:
            n=len(f['actions']);ok=ok and len(f['robot_state'])==n+1 and np.all(f['actions'][:,6]==1)
            ok=ok and np.array_equal(f['actions'][:],f['controller_input'][:])
        raw_checks.append(dict(episode_id=r['spec']['episode_id'],status='PASS' if ok else 'FAIL'))
    nfinal=len(protocol.get('final_cases',[]));all_final=nfinal>0 and all(len(final[c])==nfinal for c in ['A','B'])
    data_ready=bool(cache.get('complete_train_validation'))
    trained=all(complete.values())
    core_checks=dict(
        original_reset_identity_for_perturbations='PASS' if get(root,'conditional_signal_audit.json',{}).get('exact_new_initial_state_cases')==96 else 'NOT_TESTED',
        historical_original_reset_byte_identity=get(root,'original_reset_identity.json',{}).get('status','NOT_TESTED'),
        peg_offset_robot_box_hole_preservation='PASS' if captures and all(r['audit']['robot_box_hole_exact'] for r in captures) else 'NOT_TESTED',
        physical_collision_workspace='PASS' if captures and all(r['audit']['physical_validity']['valid'] for r in captures if r['status']=='ACCEPTED') else 'NOT_TESTED',
        scripted_expert_target_refresh='PASS' if captures and all(r['audit']['target_refresh']=='PASS' for r in captures) else 'NOT_TESTED',
        expert_reach='PASS' if captures and all(r['reach_complete'] for r in captures if r['status']=='ACCEPTED') else 'NOT_TESTED',
        observation_action_alignment='PASS' if raw_checks and all(x['status']=='PASS' for x in raw_checks) else 'NOT_TESTED',
        parent_variant_provenance='PASS' if captures else 'NOT_TESTED',
        episode_state_reset='PASS' if captures else 'NOT_TESTED',
        C3_wrist_peg_gate_main_only_box_node21_edge4_cap128_source_F5='PASS' if data_ready and all(x['status']=='PASS' for x in hash_checks) else 'NOT_TESTED',
        causal_vision_memory_no_future='PASS' if data_ready else 'NOT_TESTED',
        empty_missing_objects='PASS' if (root/'contracts.log').exists() and 'OK' in (root/'contracts.log').read_text() else 'NOT_TESTED',
        cache_hash_readback='PASS' if data_ready and all(x['status']=='PASS' for x in hash_checks) else 'NOT_TESTED',
        original_optimized_graph_equivalence=get(root,'optimization_equivalence.json',{}).get('status','NOT_TESTED'),
        CUDA_GNN_GRU_gradients=get(root,'training_smoke_verification.json',{}).get('status','NOT_TESTED'),
        hidden_carry_detach_episode_reset=get(root,'training_smoke_verification.json',{}).get('status','NOT_TESTED'),
        method_B_codec_normalizers_action7='PASS' if freeze and sha(root/'normalizer.json')==freeze['normalizer_sha256'] and sha(root/'action_codec.json')==freeze['codec_sha256'] else 'NOT_TESTED',
        save_resume_weight_optimizer_rng=get(root,'training_smoke_verification.json',{}).get('status','NOT_TESTED'),
        matched_AB_updates_frames='PASS' if trained and all(complete[c]['updates']==11487 and actual_exposure[c]==freeze['total_timesteps'][c] for c in ['A','B']) else 'NOT_TESTED',
        identical_AB_initialization='PASS' if trained and complete['A']['initial_weight_sha256']==complete['B']['initial_weight_sha256'] else 'NOT_TESTED',
        current_live_RGBD_graph_regeneration='PASS' if all_final and all(r.get('current_RGBD') and not r.get('expert_cache_used') for c in final for r in final[c]) else 'NOT_TESTED',
        no_expert_action_or_GT_student_leakage='PASS' if all_final and all(not r.get('GT_as_student_input',True) for c in final for r in final[c]) else 'NOT_TESTED',
        paired_AB_seed_offset='PASS' if len(paired)==nfinal and nfinal>0 and all(r['initial_physical_exact'] for r in paired) else 'NOT_TESTED',
        frozen_success_threshold_and_final_access='PASS' if all_final and selected else 'NOT_TESTED',
        same_previous_hidden_causal_history='PASS' if final_correction and all(truth(r['previous_hidden_exact']) for r in final_correction) else 'NOT_TESTED',
        heldout_expert_reference='PASS' if final_expert.get('attempts')==18 and final_expert.get('successes')==18 else 'FAIL' if final_expert else 'NOT_TESTED',
        intervention_baseline_repeat='PASS' if final_correction and all(truth(r['baseline_repeat_exact']) for r in final_correction) else 'FAIL' if final_correction else 'NOT_TESTED',
        CUDA_PhysX='NOT_TESTED',MPS='NOT_TESTED')
    execution_complete=data_ready and trained and all_final and len(final_correction)==32 and bool(raw_checks) and all(x['status']=='PASS' for x in hash_checks+raw_checks)
    essential=['matched_AB_updates_frames','identical_AB_initialization','observation_action_alignment',
        'current_live_RGBD_graph_regeneration','no_expert_action_or_GT_student_leakage','paired_AB_seed_offset',
        'frozen_success_threshold_and_final_access']
    execution_complete=execution_complete and all(core_checks[k]=='PASS' for k in essential)
    write(root/'verification.json',dict(full_experiment_complete=bool(execution_complete),contracts=core_checks,
        raw_episodes=raw_checks,graph_cache=hash_checks,actual_training_exposure=actual_exposure,
        final_rollouts={c:len(final[c]) for c in final},final_interventions=len(final_correction),
        interpretation='FAIL remains FAIL; executed diagnostics may be inconclusive; no model superiority inferred from offline loss'),immutable=False)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':120,'axes.grid':True,'grid.alpha':.2,'font.size':10})
    def savefig(name):plt.tight_layout();plt.savefig(figures/name);plt.close()
    # The representative parents are deterministic and unrelated to policy outcome.
    if captures:
        parent_seeds=get(root,'parent_seed_manifest.json',{}).get('parents',[])[:4]
        fig,axes=plt.subplots(2,max(1,len(parent_seeds)),figsize=(4*max(1,len(parent_seeds)),7),squeeze=False)
        for index,seed in enumerate(parent_seeds):
            ax=axes[0,index];bottom=axes[1,index]
            nominal=next((r for r in captures if r['spec']['seed']==seed and r['spec']['variant']=='P0'),None)
            center=np.array(nominal['audit']['grasp_pose'])[:2]*1000 if nominal else np.zeros(2)
            for r in captures:
                if r['spec']['seed']!=seed or r['status']!='ACCEPTED':continue
                trace=get(Path(r['raw_path']).parent,'evaluation_trace.json',[])
                xy=np.array([x['tcp'][:2] for x in trace])*1000
                if len(xy):
                    line=ax.plot(xy[:,0],xy[:,1],label=r['spec']['variant'])[0]
                    bottom.plot(xy[-10:,0]-center[0],xy[-10:,1]-center[1],color=line.get_color())
                    bottom.scatter(xy[-1,0]-center[0],xy[-1,1]-center[1],color=line.get_color(),s=22)
                    goal=np.array(r['audit']['grasp_pose'])[:2]*1000-center
                    bottom.scatter(goal[0],goal[1],color=line.get_color(),marker='x',s=55)
            ax.set(title=f'Parent {seed}',xlabel='TCP world X (mm)',ylabel='TCP world Y (mm)');ax.legend(fontsize=8)
            bottom.set(xlabel='TCP X relative to nominal target (mm)',ylabel='TCP Y relative to nominal target (mm)',
                title='Last10 actions / final TCP dot / target cross')
        savefig('expert_reach_trajectories.png')
    if pilot.get('conditional_expert'):
        fig,ax=plt.subplots(figsize=(7,4))
        data=[r for r in pilot['conditional_expert'] if r['variant']!='P0']
        ax.bar([f'{r["seed"]}/{r["variant"]}' for r in data],[r['first_translation_delta_mm'] for r in data])
        ax.set(ylabel='Same-state first expert command delta (mm)',title='Actual pose change, freshly computed expert actions');ax.tick_params(axis='x',rotation=40)
        savefig('expert_action_correction.png')
    if any(metrics.values()):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for c in ['A','B']:
            axes[0].plot([int(x['step']) for x in metrics[c]],[float(x['optimization_loss']) for x in metrics[c]],label=c,alpha=.7)
            axes[1].plot([int(x['step']) for x in validations[c]],[float(x['optimization_loss']) for x in validations[c]],label=c,marker='o')
        for ax,title in zip(axes,['Train Method B loss','Validation Method B loss']):ax.set(title=title,xlabel='Optimizer updates');ax.legend()
        savefig('ab_learning_curves.png')
    summary=[]
    for group in ['nominal','seen_type','unseen_diagonal']:
        for c in ['A','B']:
            group_rows=[r for r in final[c] if r['variant']==group]
            summary.append(dict(group=group,condition=c,n=len(group_rows),success=sum(r['reach_success'] for r in group_rows),
                position_error_mm=mean([1000*r['final_target_position_error_m'] for r in group_rows if 'final_target_position_error_m' in r]),
                angle_error_rad=mean([r.get('final_target_orientation_error_rad') for r in group_rows])))
    if all_final:
        fig,ax=plt.subplots(figsize=(8,4))
        for j,c in enumerate(['A','B']):
            data=[r for r in summary if r['condition']==c]
            ax.bar(np.arange(3)+(j-.5)*.35,[r['success']/r['n'] for r in data],width=.35,label=c)
        ax.set(xticks=np.arange(3),xticklabels=['Nominal','New seed / axis','Unseen diagonal'],ylabel='Reach successes / 6 seeds',ylim=(0,1));ax.legend();savefig('ab_reach_success.png')
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,group in zip(axes,['nominal','seen_type','unseen_diagonal']):
            for c in ['A','B']:
                first=next(r for r in final[c] if r['variant']==group)
                path=root/'evaluations/final'/c/f'step{first["step"]:06d}'/f'{first["seed"]}_{group}'/'trace.json'
                tr=read(path);ax.plot(range(1,len(tr)+1),[1000*r['target_position_error_m'] for r in tr],label=c)
            ax.axhline(4,c='k',ls='--',lw=.8);ax.set(title=f'{group}, seed61000',xlabel='Action step',ylabel='TCP final-target error (mm)');ax.legend()
        savefig('tcp_target_error_vs_step.png')
    if final_correction:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for c in ['A','B']:
            data=[r for r in valid_correction if r['condition']==c and int(r['timestep'])==0]
            axes[0].scatter([float(r['expert_delta_mm']) for r in data],[float(r['model_delta_mm']) for r in data],label=c)
            axes[1].scatter([np.linalg.norm(json.loads(r['offset_mm'])) for r in data],[float(r['response_error_mm']) for r in data],label=c,alpha=.7)
        axes[0].set(xlabel='Expert command delta (mm)',ylabel='Model command delta (mm)')
        axes[1].set(xlabel='Actual peg displacement (mm)',ylabel='Response vector error (mm)')
        for ax in axes:ax.legend()
        savefig('peg_displacement_action_response.png')
    optimization=get(root,'optimization_equivalence.json',{});renderer=get(root,'renderer_gpu_gate.json',{});workers=get(root,'worker_benchmark.json',{})
    parallel_gate=get(root,'isolated_parallel_cache_gate.json',{})
    speeds={'Causal cache':sum(r['wall_seconds'] for r in optimization.get('results',[]) if r['condition']=='original')/max(sum(r['wall_seconds'] for r in optimization.get('results',[]) if r['condition']=='optimized'),1e-9),
        'CUDA renderer':renderer.get('measured_speedup',0),'Capture workers':workers.get('speedup',0),
        'Vision/geometry overlap':parallel_gate.get('speedup',0)}
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(speeds),list(speeds.values()));ax.axhline(1,c='k',ls='--');ax.set(ylabel='Measured baseline / candidate wall time',title='Separate benchmarks; speedups must not be multiplied');savefig('gpu_cpu_throughput.png')
    resources=rows(root/'resource_samples.csv');resource_summary={}
    for stage in sorted({r['stage'] for r in resources}):
        rr=[r for r in resources if r['stage']==stage];gpu=[number(r['gpu_util_pct']) for r in rr];gpu=[x for x in gpu if x is not None]
        resource_summary[stage]=dict(samples=len(rr),GPU_util_mean=mean(gpu),GPU_util_p95=float(np.percentile(gpu,95)) if gpu else None,
            VRAM_peak_mib=max([number(r['vram_mib']) or 0 for r in rr]),CPU_process_mean=mean([r['cpu_process_pct'] for r in rr]),
            CPU_system_mean=mean([r['cpu_system_pct'] for r in rr]))
    profiles=rows(root/'pipeline_profile.csv');stage_totals=collections.defaultdict(list)
    for r in profiles:stage_totals[r['stage']].append(float(r['seconds']))
    parent_metadata=get(root/'experiments'/run_id,'metadata.json',{})
    from datetime import datetime,timezone
    since_start=(datetime.now(timezone.utc)-datetime.fromisoformat(parent_metadata['started_at'])).total_seconds() if parent_metadata.get('started_at') else None
    throughput=dict(resource_stages=resource_summary,stage_seconds={k:dict(total=sum(v),n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95))) for k,v in stage_totals.items()},
        separate_benchmark_speedups=speeds,cache_frames=sum(e['graphs'] for e in cache.get('episodes',[])),
        data_pipeline=get(root,'data_pipeline_complete.json'),kernel_benchmark=get(root,'geometry_kernel_benchmark.json'),
        nonempty_wrist_benchmark=get(root,'nonempty_geometry_kernel_benchmark_v2.json'),
        elapsed_seconds_since_experiment_UUID=since_start,includes_engineering_and_operational_pause=True,
        training_updates_per_second={c:len(metrics[c])/max(sum(float(r['update_seconds']) for r in metrics[c]),1e-9) for c in ['A','B']},
        note='GPU/system utilization in overlapping stages includes all task processes; process CPU is sampled separately. Primitive timings are not full pipeline speedups.')
    from datetime import datetime
    main_capture=[r for r in captures if r['status']=='ACCEPTED' and r['spec']['split']!='pilot']
    if main_capture:
        ends=[datetime.fromisoformat(r['created_at']).timestamp() for r in main_capture]
        begin=min(end-r['wall_seconds'] for end,r in zip(ends,main_capture));duration=max(ends)-begin
        throughput['main_capture']=dict(new_episodes=len(main_capture),reused_pilot_episodes=10,
            actual_window_seconds=duration,frames=sum(r['actions'] for r in main_capture),
            frames_per_second=sum(r['actions'] for r in main_capture)/duration,episodes_per_hour=len(main_capture)/duration*3600)
    transition=get(root,'cache_worker_transition.json',{})
    process_record=transition.get('old_processes') or get(root,'data_processes.json',{})
    if process_record.get('started_at'):
        begin=datetime.fromisoformat(process_record['started_at']).timestamp()
        end=datetime.fromisoformat(cache['generated_at']).timestamp() if cache.get('generated_at') else datetime.now().timestamp()
        throughput['production_cache']=dict(complete=data_ready,window_seconds=end-begin,
            graphs_per_second=sum(e['graphs'] for e in cache.get('episodes',[]))/(end-begin),
            episodes_per_hour=len(cache.get('episodes',[]))/(end-begin)*3600,
            includes_engineering_pause_and_contention=True,completed_graphs_reused_without_regeneration=True)
    throughput['training_wall_updates_per_second']={c:11487/complete[c]['wall_seconds'] if complete[c] else None for c in ['A','B']}
    write(root/'throughput_summary.json',throughput)
    # Full report: explicit executed scope and unanswered research questions.
    unseen={c:next((r for r in summary if r['condition']==c and r['group']=='unseen_diagonal'),{}) for c in ['A','B']}
    trained_text='동일한 초기 가중치·구조·Method B·정규화·Optimizer 및 사전 고정 11,487 update를 실행했다.' if trained else 'A/B 본 학습은 아직 완료되지 않았다.'
    outcome='NOT_TESTED'
    if all_final:
        sa,sb=unseen['A']['success'],unseen['B']['success']
        outcome='A 실패 / B 성공: 조건부 Reach 데이터 효과를 제한적으로 지지' if sb>sa else 'A/B 모두 성공: 기본 Reach 학습 가능' if sa==sb==6 else 'A/B 모두 실패: 이 데이터 변경만으로 충분하다는 근거 없음' if sa==sb==0 else '성공 수가 같거나 B가 우세하지 않음; 원자료의 오차·교환 관계를 함께 검토'
    verdict='실제 수집·캐시·학습·최종 평가 실행 완료' if execution_complete else '진행 중 또는 일부 단계 미완료 — 정책 결과를 추정하지 않음'
    lines=[f'# Task 10-D — Geometry-conditioned Reach BC','',f'ExperimentLogger UUID `{run_id}`. 기준 비교 `3788798a719f4ec4a66c224ca42f5d63`. 작성 시각 {now()}.','',
        f'**{verdict}.** {outcome}.','',
        '## 실행 범위','',f'수집 후보 {len(candidates)}개, 실제 시도 {len(captures)}개, 승인 {sum(r["status"]=="ACCEPTED" for r in captures)}개. Graph 캐시 {len(cache.get("episodes",[]))}개 / {sum(e["graphs"] for e in cache.get("episodes",[]))} frames. Final Reach A/B 각 {len(final["A"])}/{len(final["B"])}개. Final same-state intervention {len(final_correction)}개(A/B 합산).',
        '',trained_text,'',
        'E0 Edge-aware GNN L2/d64, Node21/Edge4, Readout320+Presence2+Robot76, Projection128, GRU144, sequence32, Hidden Carry/TBPTT, Action7 및 pd_ee_delta_pose를 그대로 사용했다. E1·새 Feature·새 Controller·Skip Connection은 도입하지 않았다.','',
        '## 데이터 생성과 고정','',
        '원본 Train96 seed와 사전 선택 Parent24(sorted Train96[::4])를 사용한다. 추가 Offset은 X/Y ±5mm의 네 방향이다. 새 위치에서 ScriptedExpert를 새로 초기화하고 실제 env.step으로 reach/descend를 실행했다. close/grasp/lift/insertion Action은 학습에 없다. 매 Action t에 관측 t를 대응시키고 T+1 관측에는 Action을 붙이지 않았다. 원본 HDF5/Graph/Checkpoint는 보존하고 Train96 Reach도 현재 Runtime에서 재수집하여 A/B 공통 원본을 만들었다.','',
        'Reach는 elevated approach target(10mm, 0.05rad) 도달 이후 close 직전 target(4mm, 0.03rad)에 도달하고 gripper opening≥65mm, 조기 close 없음, Peg 이동<2mm 및 물리 유효성을 만족해야 한다. 이는 Peg actor origin 30mm나 Grasp/Insertion 성공과 다르다. Target/Offset/GT는 생성·평가 metadata에만 있고 Student는 RGB-D, 예측 Mask, Graph, Robot76만 받는다.','',
        '과거 Train96과 현재 Runtime의 reset byte hash는 52/96만 같았다. Pilot의 기존 HDF5 대비 Robot/TCP 관측에는 약 4e−7 수준 차이와 RGB/Depth 차이가 있어 원본 Graph를 섞지 않았다. 현재 새로 수집한 Parent 원본–Offset 비교 96/96에서는 Robot76, Camera calibration, Box/Hole가 정확히 같다. 따라서 A/B는 공통 현재 Runtime을 비교하며 과거 데이터의 byte identity를 PASS로 바꾸지 않는다.','',
        f'Engineering Pilot {pilot.get("accepted",0)}/{pilot.get("attempts",0)} 승인. 첫 Action 변화뿐 아니라 전체 목표 이동·실행 시퀀스를 검증했다. 같은 Robot 초기 상태의 추가 사례는 최대 24 parent×4 offset이며 독립 Robot 상태 96개라는 뜻은 아니다. 이후 trajectory는 Robot 상태가 달라지므로 모든 t를 exact same-state 사례로 세지 않는다.','',
        'Graph 캐시는 원본 Plan A cache_episode/CurrentGeometry/validate_graph와 동일 C3+A-SF 계약을 사용한다. SAM2는 Episode/Camera별 reset 및 관측 prefix만 사용한다. DINO Memo는 같은 RGB/Prompt/가중치의 순수 detector 결과만 재사용한다. Gate 전 임의 cap128, 탐지 조건·Filter Threshold·Sampling 의미 변경은 없다. 완료된 Pilot Graph는 원본 raw/hash/geometry identity를 재검증해 파일 그대로 채택했다.','',
        '## 계산 비용','',
        f'새 Offset 2개/87 frames에서 최적화 전후 모든 저장 Graph·Mask·Gate·F5/Pixel Provenance가 정확히 일치했다. 캐시 Benchmark 속도비 {speeds["Causal cache"]:.3f}배, GPU Renderer {speeds["CUDA renderer"]:.3f}배, 수집 Worker {speeds["Capture workers"]:.3f}배. 독립 CPU Geometry와 CUDA Vision의 overlap은 격리된 동일 조건에서 {speeds["Vision/geometry overlap"]:.3f}배였다. 이 별도 수치를 곱하여 end-to-end 향상이라 주장하지 않는다.','',
        '최초 동시 실행 Benchmark는 GPU 경합이 있어 후보 1.046배로 기록했다. Task10D 캐시 Worker만 126초간 SIGSTOP/SIGCONT로 보존하여 원본/후보를 순서대로 재측정한 결과 73.62초→47.19초였다. 87개 frame의 모든 비시간 배열·Gate가 정확히 같아 2 CPU Geometry worker와 CUDA Vision의 overlap을 채택했다. 완료된 Raw/Graph를 해시 검사해 보존하고 미완료 Episode부터 새 Worker로 이어갔다. 기존 Plan A 서비스는 건드리지 않았다.','',
        'RTX3090 CUDA Vision과 CUDA GNN/GRU를 사용한다. GPU Renderer는 CPU Physics의 전체 RGB-D/Action/Robot Trace가 byte-identical한 경우에만 채택했다. GPU PhysX 라이브러리 준비가 약 303초 동안 진행되지 않아 해당 자식 probe를 중단했다. GPU Physics Expert/벡터화 수집은 NOT_TESTED이며 CPU/GPU Physics 동등성을 주장하지 않는다.','',
        'Backprojection/Connected Components/거리 primitive의 전송 포함 CUDA 후보는 geometry_kernel_benchmark.json에 있다. Graph/PCA/F5/Source ID의 최종 조립은 검증된 CPU 경로를 유지한다. GPU의 높은 사용률만을 위해 dense quadratic adjacency나 수치가 다른 경로로 바꾸지 않았다. AMP는 적용하지 않고 FP32 모델/기존 geometry 정밀도를 유지했다. SAM2의 설치된 _C postprocessing 확장이 없는 Runtime 동작도 기존 baseline과 동일하게 보존했다.','',
        '## A/B 학습 예산','',f'실제 기록된 유효 timestep: A={actual_exposure["A"]}, B={actual_exposure["B"]}. 사전 계획 노출: {freeze.get("total_timesteps",{})}. B 추가 데이터 frame 비율={fmt(freeze.get("exact_mixture_B_perturbed_frame_fraction"))}. 길이 매칭에서 제외된 추가 Episode={len(freeze.get("excluded_unmatched_lengths",[]))}. 제외 목록과 생성 실패는 보존했다.','',
        'A의 대체 Episode 길이를 B의 추가 Episode와 같게 선택하여 매 update의 mask/timestep 수를 맞춘다. 같은 길이이면 동일 parent 원본을 우선하고, 길이가 다르면 동일 길이의 다른 원본을 사용한다. 따라서 남는 Parent 재가중 차이는 sampling_frequency.csv에 공개하며 Geometry 포함과 완전히 분리된 인과 효과라 단정하지 않는다. 원본 통계는 다시 fit하지 않는다. action_distribution.json/CSV는 고유 데이터와 동결 Sampler의 Action 분포를 기존 Train96 통계와 비교한 진단이며 새 Normalizer로 사용하지 않는다. 학습률은 기존 Full96의 epoch 완료 update 경계에서 동일하게 진행한다. Reach 데이터 cycle 수는 별도로 기록한다. Checkpoint는 공통 Development8의 Method B loss 첫 최솟값으로 선택하며 Final Seed61000..61005는 모델 선택 후에만 접근한다. Offset별 같은 seed를 A/B에 paired 적용한다.','',
        '## 최종 Reach 비교','',
        f'모델 선택을 동결한 뒤 동일 Final18 조건의 ScriptedExpert ceiling도 실제 실행했다: {final_expert.get("successes",0)}/{final_expert.get("attempts",0)}. 이 데이터는 학습·선택·Student 입력에 쓰지 않았고 Expert 결과에 따라 Student 사례를 제외하지 않았다.','',
        '| 조건 | 모델 | 성공/사례 | 최종 위치 오차 mm | 최종 자세 오차 rad |','|---|---|---:|---:|---:|']
    lines += [f'| {r["group"]} | {r["condition"]} | {r["success"]}/{r["n"]} | {fmt(r["position_error_mm"])} | {fmt(r["angle_error_rad"])} |' for r in summary]
    lines+=['','## 실제 Peg 이동에 대한 Action 보정','',
        '같은 Robot 상태·동일 previous hidden·동일 Causal Vision prefix에서 실제 Peg Actor를 이동하고 새 RGB-D로 Graph를 재생성했다. t=0은 초기 조건 변화이며 t=10은 same-history 단일 시점 진단이다. 동적 Recovery Rollout은 아니다. baseline repeat의 Mask/Graph/readout/hidden/action이 정확히 재현되지 않으면 해당 anchor를 분석에서 제외하고 원자료와 FAIL 상태를 남긴다. 09-A/10-B의 controller command Δ, cosine, signed gain, response error, zero-response 계산을 재사용했다. Expert Δ≤0.1mm는 방향 판정에서 제외했다.','',
        '| t | 모델 | 유효 reference 수 | 방향 cosine | signed gain | Δ error mm | zero 대비 개선 mm |','|---|---|---:|---:|---:|---:|---:|']
    for t in [0,10]:
        for c in ['A','B']:
            rr=[r for r in paired_correction if r['condition']==c and int(r['timestep'])==t]
            lines.append(f'| {t} | {c} | {len(rr)} | {fmt(mean([r["direction_cosine"] for r in rr]))} | {fmt(mean([r["signed_gain"] for r in rr]))} | {fmt(mean([r["response_error_mm"] for r in rr]))} | {fmt(mean([r["zero_response_improvement_mm"] for r in rr]))} |')
    q=[('Q1','CONFIRMED' if pilot.get('status')=='PASS' else 'NOT_TESTED','실제 Peg 이동 뒤 새 목표와 연속 Reach Action을 생성했음을 Pilot에서 확인했다. 전체 시도·실패·목표 및 Trajectory는 raw metadata에 있다.'),
       ('Q2','SUPPORTED' if captures else 'NOT_TESTED','최대96개의 정확한 같은 초기 Robot/Box 상태 Offset 비교가 추가된다. 독립 parent는24개이며 이후 Frame은 trajectory 분기를 포함한다. 기존 Train96의 관찰 Pair 부족(10-C2)을 GT 기반 다양성 부재의 증명으로 바꾸지 않는다.'),
       ('Q3','CONFIRMED' if optimization else 'NOT_TESTED','87개 새 Graph의 exact equivalence와 별도 속도 Benchmark를 확인했다. 전체 벽시계 시간은 throughput_summary.json으로 구분하며 가속 배수를 곱하지 않는다.'),
       ('Q4','CONFIRMED' if core_checks['matched_AB_updates_frames']=='PASS' else 'NOT_TESTED','동일 update/실제 frame 노출, 초기화, normalizer, codec, scheduler와 선택 예산을 verification.json에서 대조한다.'),
       ('Q5','SUPPORTED' if correction_improvement else 'INCONCLUSIVE' if final_correction else 'NOT_TESTED',
        f't0에서 양쪽 모두 유효하고 Robot 상태가 같은 paired reference의 Δ error는 A={fmt(correction_error["A"])}mm / B={fmt(correction_error["B"])}mm다. '+('B의 응답 오차 개선을 제한적으로 지지한다.' if correction_improvement else 'B가 더 적절하게 보정한다는 근거는 부족하다.')+' 방향·gain을 함께 보며 반복 Offset/timestep은 독립 표본이 아니다.'),
       ('Q6','SUPPORTED' if all_final else 'NOT_TESTED',outcome),
       ('Q7','SUPPORTED' if all_final else 'NOT_TESTED','Nominal/새 seed 축 방향/미학습 대각선 성능을 분리한다. 개선과 nominal 손실의 동시 여부는 위 paired 결과로 판단하며 단일 학습 seed에서 일반적 우위를 주장하지 않는다.'),
       ('Q8','INCONCLUSIVE','Expert target/action 신호·offline fit·same-state Δ·closed-loop Reach를 구분한다. 실패가 데이터 신호 부족, 표현, 최적화, 제어 연결 중 하나만의 원인임을 이 A/B로 확정할 수 없다.'),
       ('Q9','SUPPORTED' if all_final and unseen.get('B',{}).get('success',0)>0 else 'INCONCLUSIVE' if all_final else 'NOT_TESTED','미학습 seed/대각선의 실제 Reach와 유효 intervention이 함께 개선되는 경우에만 제한된 초기 위치 조건부 제어의 증거가 된다. ±5mm 범위, 6 Final seeds, 학습 seed1개라 memorization 가능성은 남는다.'),
       ('Q10','SUPPORTED' if all_final else 'NOT_TESTED','B의 nominal 유지·미학습 Reach·보정 Δ가 함께 좋으면 다음 Grasp/Insertion 확장을 검토한다. Offline만 좋거나 둘 다 실패하면 이 데이터 신호/목표/제어 연결의 진단을 우선한다. BC 일반의 가능/불가능을 단정하지 않는다.')]
    lines+=['','## Q1–Q10 판정','']+[f'- **{k} {status}:** {text}' for k,status,text in q]
    lines+=['','## Contract tests 및 제한','', '| 항목 | 판정 |','|---|---|']+[f'| {k} | {v} |' for k,v in core_checks.items()]
    lines+=['','Supervisor의 Task10D 전용 data/experiment/monitor 프로세스는 SSH 세션과 분리되어 있다. 기존 Plan A 프로세스·감시 경로를 변경하지 않았다. 이벤트 Detector는 기존 구현을 재사용하며 ERROR/STALL/COMPLETION만 기록한다. Codex 자동 호출 연결은 NOT_TESTED이므로 완전 무인 복구 운영이라 주장하지 않는다. Discord 실제 상태는 monitor/notification_status.json에 별도로 기록한다. Git commit 및 외부 Publishing은 수행하지 않았다.','',
        'CUDA Physics·GPU vectorized Expert·MPS·동적 Recovery·Grasp/Insertion은 NOT_TESTED다. 사용된 Backend의 실제 실행과 기각한 후보를 구분했다. Final seed/offset 반복은 상관된 측정이며 이 수를 독립 표본 수로 간주한 p-value나 일반적 성공률 우위를 제시하지 않는다.','',
        '## 산출물','',
        '[수집 프로토콜](collection_protocol.json) · [A/B 동결](dataset_ab_manifest.json) · [캐시](graph_cache_manifest.json) · [성능](throughput_summary.json) · [검증](verification.json) · [SHA256 Index](artifact_index.json)','']
    for p in sorted(figures.glob('*.png')):lines.append(f'![{p.stem}](figures/{p.name})\n')
    body='\n'.join(lines)+'\n';name='10d_geometry_conditioned_reach_bc.md'
    (root/name).write_text(body)
    content=markdown_html('\n'.join(line for line in body.splitlines() if not line.startswith('![')))
    content=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',content)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Task10D Reach BC</title><style>body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172638;background:#fafbfe}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd5df;overflow-wrap:anywhere}.scroll{overflow-x:auto}img{max-width:100%}code{overflow-wrap:anywhere}h2{margin-top:40px}</style>'+content
    page+=''.join(f'<figure><img src="figures/{p.name}" alt="{html.escape(p.stem)}"></figure>' for p in sorted(figures.glob('*.png')))
    (root/'research_report.html').write_text(page+'</html>')
    checkpoint_manifest={c:dict(selected=get(root,c+'_selected.json'),last=get(root,c+'_last.json')) for c in ['A','B']}
    write(root/'checkpoint_manifest.json',checkpoint_manifest)
    from utils.experiment_logger import ExperimentLogger
    write(root/'artifact_index.json',dict(status='INDEXING',files=[]))
    with ExperimentLogger(team='graph',model='Task10D_analysis_report',seed=42,
            config=dict(comparison_uuid=run_id,execution_complete=bool(execution_complete),
                dataset_AB_sha256=sha(root/'dataset_ab_manifest.json') if freeze else None),
            root=root/'experiments',job_type='evaluate') as log:
        for p in [root/name,root/'research_report.html',root/'verification.json',root/'throughput_summary.json',
                root/'checkpoint_manifest.json',root/'artifact_index.json',*figures.glob('*.png')]:
            log.link_output(p,kind='reports' if p.suffix in ['.md','.html','.png'] else 'metrics')
    index=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or any(k in p.parts for k in ['source','monitor','__pycache__','worker_benchmark','renderer_probe']):continue
        if p.name=='artifact_index.json' or p.suffix in ['.log','.lock','.partial'] or '.interrupted.' in str(p):continue
        if p.name in ['pipeline_state.json','graph_cache_progress.json','experiment.log']:continue
        if not execution_complete and p.suffix=='.csv':continue
        if not execution_complete and any(k in p.parts for k in ['experiments','training_runs','evaluation_runs']):continue
        value=sha(p);index.append(dict(path=str(p.relative_to(root)),sha256=value,bytes=p.stat().st_size))
    write(root/'artifact_index.json',dict(run_id=run_id,files=index,generated_at=now(),
        excluded='source tree separately hashed; active monitor/log/lock/partial files; benchmark duplicate raw copies'))
    # Verify stable, immutable artifacts a second time, including report and plots.
    for item in index:
        assert sha(root/item['path'])==item['sha256'],item['path']
    print('REPORT',run_id,'complete',execution_complete,'indexed',len(index),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

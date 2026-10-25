"""Static, self-contained local report; verification never reruns optimization."""
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image

from graph.integration_baseline.graph import build_graph
from graph.integration_baseline.io import write_json, sha256


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def plots(out,summary,dataset,sequences):
    folder=out/'plots';folder.mkdir(exist_ok=True)
    frame=sequences[0][0];graph=build_graph(frame.packet,frame.robot_positions_ee)
    pos=graph.node_input[:,:3].numpy();edge=graph.edge_index.numpy();etype=graph.edge_type.numpy()
    fig=plt.figure(figsize=(13,5),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    for kind,color,label in [(0,'#456caa','Raw EDE'),(1,'#1b9277','Robot-object'),(2,'#dd7e20','Peg-box')]:
        selected=edge[:,etype==kind]
        if selected.size:ax.add_collection3d(Line3DCollection(pos[selected.T],colors=color,linewidths=.55,alpha=.6,label=label))
    for kind,color,label in [(3,'#bb3247','Peg'),(4,'#5266bb','Box')]:
        points=pos[graph.node_type.numpy()==kind]
        if len(points):ax.scatter(*points.T,s=6,c=color,label=label)
    ax.scatter(*pos[:3].T,c=['black','#00a885','#d253a4'],s=35)
    for i,name in enumerate(('TCP','Left','Right')):ax.text(*pos[i],name)
    ax.auto_scale_xyz(pos[:,0],pos[:,1],pos[:,2]);ax.set(xlabel='EE x (m)',ylabel='EE y (m)',zlabel='EE z (m)',title='REAL_ALIGNED: all point rows / all edges')
    ax.legend(fontsize=7,loc='best')
    ax2=fig.add_subplot(122)
    ax2.imshow(dataset.raw[frame.episode_id]['base_camera/rgb'][frame.timestep]);ax2.set_title('Same observation RGB (128×128)');ax2.axis('off')
    fig.savefig(folder/'graph_rgb.png',dpi=150);plt.close(fig)

    p=frame.packet
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    features=np.ma.masked_invalid(p['geometric_features'])
    axes[0].imshow(features,aspect='auto',interpolation='nearest',cmap='coolwarm');axes[0].set_title('Original F=5 (NaN masked)')
    axes[0].set_xticks(range(5),['nx','ny','nz','var rad','jump m'],rotation=40)
    for ax,key,label in [(axes[1],'feature_field_available','Availability'),(axes[2],'feature_field_valid','Validity')]:
        ax.imshow(p[key],aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='Greens');ax.set_title(label)
        ax.set_xticks(range(3),['normal','variation','depth'])
    axes[0].set_ylabel('Original object point row; no removal')
    fig.savefig(folder/'features_masks.png',dpi=150);plt.close(fig)

    diagnostic=json.loads((out/'replay_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,ep in zip(axes,diagnostic['episodes']):
        with np.load(out/f'diagnostic_ep{ep["episode_id"]:03d}.npz') as a:
            delta=a['absolute_error'];ax.plot(delta[:,:9].max(1),label='qpos');ax.plot(delta[:,9:18].max(1),label='qvel')
            ax.plot(delta[:,18:25].max(1),label='TCP pose components')
        ax.axhline(.001,color='red',ls='--',label='existing 1e-3 bound')
        ax.axvline(ep['first_failure_timestep'],color='gray',ls=':')
        ax.set(title=f'Episode {ep["episode_id"]}: full replay FAILED',xlabel='timestep',ylabel='max component error (mixed units)')
        ax.legend(fontsize=7)
    fig.savefig(folder/'replay_alignment.png',dpi=150);plt.close(fig)

    stages=summary['latency_memory']['summary']
    keys=['receiver_validation_copy','robot_object_selection','peg_box_global_selection','tensor_packing',
          'disjoint_union_pack_ms','gnn_ms','readout_ms','fusion_ms','gru_input_encoder_ms','gru_ms','action_head_ms']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ax.barh(keys,[stages[k]['mean'] for k in keys],color='#427ba5');ax.invert_yaxis()
    ax.set(xlabel='CPU wall ms, mean of 10 after 3 warmup',title='REAL_ALIGNED packet→action; upstream Vision / IO excluded')
    fig.savefig(folder/'latency.png',dpi=150);plt.close(fig)


def generate(out,summary,dataset,sequences):
    from graph.integration_baseline.smoke import REPORT
    frame_costs=[r['timings_ms'] for e in dataset.graph_manifest['episodes'] for r in e['packets']]
    data_costs=dict(replay_saved_repeat_seconds=[dict(episode_id=e['episode_id'],seconds=e['elapsed_seconds'])
                                                   for e in dataset.manifest['episodes']],
        replay_scope='last retained repeat per episode; after reset/initial restore through capture+checks+steps; excludes reset, disk compression, hashing and earlier attempts/repeats',
        offline_frame_ms={key:dict(count=len(frame_costs),sum=float(sum(r[key] for r in frame_costs)),
            mean=float(np.mean([r[key] for r in frame_costs])),median=float(np.median([r[key] for r in frame_costs])),
            p95=float(np.percentile([r[key] for r in frame_costs],95)),max=float(max(r[key] for r in frame_costs)))
                         for key in frame_costs[0]},
        offline_scope='mask_ms includes per-frame detector/prompt/SAM2 work; graph stages exclude packet compression and report IO; JPEG preparation and model load not separately timed',
        memory=dict(replay_peak='NOT_EXECUTED',offline_vision_peak='NOT_EXECUTED',
                    integration='latency_memory.json process high-water and logical tensor/parameter bytes'),
        end_to_end_latency_claim=False)
    write_json(out/'data_pipeline_costs.json',data_costs)
    plots(out,summary,dataset,sequences)
    run=summary['run_id'];actual=summary['real_forward_backward'];coverage=summary['replay_coverage']
    schema=summary['schema'];replay=dataset.manifest
    condition_rows=[[r['condition'],r['point_count'],r['edges'],r['object_present'],r['status']] for r in summary['stored_packet_conditions']]
    field_rows=[[s['name'],s['offset'],s['dimension'],', '.join(s['units']),s['frame'],s['source']] for s in schema['fields']]
    replay_rows=[[e['episode_id'],e['observations'],e['aligned_targets'],e['source_steps'],
                  e['failure']['timestep'] if e['failure'] else '-',e['max_errors']['qvel'],e['repeat_max_robot_error'],e['status']]
                 for e in replay['episodes']]
    lat=summary['latency_memory']['summary']['total_wall_ms']
    coverage_text=(f"완주 {coverage['complete_episodes']}/91 Episode. 검증된 Prefix {coverage['episodes']}개, "
                   f"관측 {coverage['observations']}개 / Target {coverage['aligned_frames']}개. "
                   f"전체 Replay Pilot Gate: {summary['full_replay_gate']}. 나머지 88개 Replay는 실행하지 않았다.")
    md=f'''# Task06 — Graph–GNN–GRU/BC 최소 통합 및 Expert Replay

Run UUID: `{run}`. **최소 실제 Prefix 경로 검증 완료, 전체 Replay Dataset은 BLOCKED_DATA.**

[Static HTML](../../experiments/{run}/graph_gnn_gru_baseline.html) · [검증](../../experiments/{run}/verification.json) · [전체 결과](../../experiments/{run}/summary.json) · [테스트 로그](../../experiments/{run}/tests.log) · [Source/Diff](../../experiments/{run}/implementation.diff)

## 실제 구현

Opt-in [패키지](../../graph/integration_baseline/README.md)에 Robot 3 Node, 원본 Point/F=5/Availability/Validity 보존, 기존 Raw EDE 수신, 물체별 Robot k-NN, exact global top-M, Attr4, disjoint-union Batch를 구현했다. GNN은 Node/Edge Type Embedding, 공유 Message MLP, destination mean과 residual MLP다. Readout은 `[TCP,Left,Right,mean(Peg),mean(Box)]` 그대로 5d이며 Empty Slot은 0이다.

`L=2,d=64,k=4,M=16,keep_ratio=0.5`는 임시 Smoke 설정이다. Node 입력에는 위치3+finite F5+available3+valid3만 들어간다. 원본 NaN/dtype/source/component/행은 보존하고 별도 finite view를 만든다. Consumer는 Delaunay/Strict Validation을 재계산하지 않는다.

Robot State는 **{schema['dimension']}차원**, Schema `{schema['id']}`다. qpos9/qvel9, Gripper position2/velocity2/opening1/opening velocity1, TCP·Left/Right Finger·Robot Base 각각 Pose7/linear velocity3/angular velocity3을 모두 포함한다. 실제 `get_proprioception()`의 모든 필드(qpos/qvel)를 사용하고 현재 Controller State는 빈 dict다. 중복을 제거하지 않았다. Gripper opening은 두 finger qpos의 합이며 pad 간 거리 추정이 아니다. Link 출처는 `agent.tcp`, `agent.finger1_link`, `agent.finger2_link`, `agent.robot.get_root()`이며 각각 `.pose.raw_pose`, `.linear_velocity`, `.angular_velocity`를 읽는다. Gripper 필드는 같은 qpos/qvel의 마지막 두 값과 그 합이다. Robot link kinematics는 같은 Live API를 추론에도 사용한다. Joint effort/acceleration/contact는 기존 Proprioception에 없는 항목이며 추정해 추가하지 않았다. [필드·순서·단위·출처](../../experiments/{run}/robot_state_schema.json).

기존 `StateNormalizer`의 mean/std 저장 구조와 `.001` std floor를 재사용한 schema-sized Adapter다. 확보한 Train Prefix만으로 Fit하고 Freeze했다. Validation/Test Episode는 수집하지 못했으며 이 Split의 통계/모델 성능을 검증했다고 주장하지 않는다. GNN 이후 `[5d, object_present2, normalized robot76]`={actual['fused_dimension']}차원을 기존 `RNNPolicy`(GRU hidden144/layer1/encoder128)에 전달한다. 기존 regression7 MSE를 재사용한다.

## Replay와 실제 Coverage

{coverage_text}

Source는 기존 `trajectory.state.pd_ee_delta_pose.physx_cpu.h5`: **91 Episode / 13,958 Action**, Frozen Split **73/9/9**를 재확인했다. 원본 Initial State Repair와 기존 source episode mapping을 재사용했다. Snapshot Restore 허용오차 `2e-5`, 순차 Replay `1e-3`은 기존 `prepare_dataset.py` 기준을 그대로 유지했다.

Episode 0/1/2의 순차 Replay는 각각 t=175/117/181에서 qvel 최대 오차 0.0010336712/0.0011354163/0.0011414587로 실패했다. State-only 재실행도 동일하고, 원본 추출 설정의 Torch threads=4/seed=0도 같은 실패 시점/오차였다. 반면 저장 Snapshot 복원 최대 오차는 약 3e-7이다. 따라서 RGB-D 추가나 이번 thread 수가 원인이라는 근거가 없고 정확한 drift 원인은 미확정이다. Snapshot Teleport로 순차 Replay 실패를 숨기거나 허용오차를 완화하지 않았다. [진단](../../experiments/{run}/replay_diagnostic.json) · [원본 Thread 설정 재검사](../../experiments/{run}/before_snapshot/replay_threads4_diagnostic.json).

각 실패 Frame 및 이후는 저장 입력에서 제외했다. 유효 Prefix의 마지막 관측도 다음 Target이 없는 행으로 남겨 보수적으로 Target 470개를 확보했다. 실제 입력에 쓰는 Prefix는 기존 bound 내에서 source Observation/Action과 대응하며 두 번의 동일 Replay에서 Robot State/Geometry 오차는 0이었다. 완전 Episode 정합 성공으로 해석할 수 없다. [Coverage/Alignment](../../experiments/{run}/replay_alignment_coverage.json).

RGB-D는 실제 CPU 환경 기본 Camera 128×128 두 개의 RGB, raw int16 mm depth(overflow/invalid 포함), floating metre view, CV Intrinsic/Extrinsic/OpenGL calibration을 기록했다. Raw mm 값은 보존하며 Camera-to-World CV는 Extrinsic의 역행렬이다. Robot-only 상태와 실제 rubber-pad collision centre를 같은 timestep에서 읽는다. Timestamp는 wall-clock이 아닌 `t/20Hz` simulation time이다. 원본 `a_t`를 재생성/정규화하지 않고 `o_t → a_t → step → o_(t+1)` 순서로 저장한다.

원본 TimeLimit은 100 step인데 Expert 저장 Episode는 그 이후도 계속된다. 실제/source truncated flag가 일치함을 검사하고 보존했다. Sequence Loader는 terminated/truncated 뒤를 가로지르는 sequence를 거부한다. 이번 실제 Smoke는 t=0..3이며 이 경계를 지나지 않는다. 후속 전체 학습 전 Timeout 의미의 별도 결정이 필요하다.

## Offline Vision Dataset

검증 Prefix의 **{summary['vision_frames']}/{coverage['observations']} 관측 전체**에 연속 Packet을 생성했다. RGB-only Grounding DINO와 forward SAM2를 기존 경로로 실행했다. 초기화는 현재 Prefix만 입력하고 현재 Frame 결정만 허용해, 기존 episode-wide late initialization의 미래 Box 참조 가능성을 제거했다. SAM2에도 매 시점 `num_frames=t+1`만 노출하고 한 Frame씩 전파한다. 설치 SAM2의 `min(num_frames,max_obj_ptrs_in_encoder)` 정규화 때문에 전체 영상 길이가 앞부분 1픽셀을 바꾼 첫 `vision_v1` 생성본은 거부·보존했다. `vision_v2`로 재생성하고 실제 RGB Prefix/미래내용변경/영상길이변경 간 앞부분 Mask exact 일치를 별도로 검증했다. [실제 인과성 검사](../../experiments/{run}/before_snapshot/actual_mask_causality.json) · [거부 기록](../../experiments/{run}/rejected_vision_v1_audit.json). 미래 Frame을 이용한 backfill·backward refinement·GT segmentation은 사용하지 않았다. 기존 SAM2 optional C extension이 없어 설치 Predictor가 hole-filling postprocess를 건너뛴 경고를 로그에 보존했다. Dependency 설치는 하지 않았다.

`RGB-D → predicted Mask → 기존 Shape-First/Surface-Second → 기존 PCA5/F5 → 같은 시점 EE → 기존 observed component → Raw EDE`를 재사용한다. Surface 추가량은 기존 Peg97/Box95이고 총 Point Budget/신규 Geometry Filtering은 없다. 공통 Producer가 실험 Adapter에 있으므로 opt-in Builder에서 기존 Helper를 직접 호출한다. 기존 Source를 이동/수정하지 않았다. [Dataset Manifest](../../experiments/{run}/dataset_manifest.json), [Join 검증](../../experiments/{run}/dataset_validation.json).

Missing/Invalid/Suffix는 Manifest에 구분한다. 전체 원본 91개 데이터셋이 완성된 것이 아니다. Source Hash+Episode+Timestep으로 RGB-D/State/Target/Graph를 Join하고 Transform·Raw Policy·Source Row·Split을 검사한다. GT Object/Hole/Goal, Reward/Success, Expert/Future Action은 모델 입력에서 제외한다.

## 검증 결과와 신뢰 구분

- **A — SYNTHETIC_CONTRACT_ONLY:** 작은 독립 brute-force oracle, tie/source/row, k/M overflow와 M=0, reverse/zero-distance, zero-degree/empty, +3/packed offset, GT/action 변경 비의존성, 실제 mask encoder 소비와 Readout 계약 PASS.
- **B — REAL_PACKET_WITH_SYNTHETIC_ROBOT:** Task05-E 기본 8 Packet + ratio1 stress 8 Packet의 GNN finite/Variable N/Mask/5d PASS. 기존 Packet의 Robot은 실제값이 없어 명시적 Synthetic Fixture를 사용했다. 실제 연속 시계열로 간주하지 않았다.
- **C — REAL_ALIGNED (검증 Prefix 한정):** 실제 Train Episode 2개 × 연속 4 Frame, Variable N/Packed mapping, Full Robot State와 Action7, 초기 nonzero Hidden의 Sequence↔Step max error `{actual['sequence_step_max_abs_error']:.3g}`, Episode/reset/terminated/truncated reset PASS.
- **D — REAL_ALIGNED (검증 Prefix 한정):** 기존 7D MSE `{actual['bc_loss']:.6g}`, finite GNN Node/Message/Update·Type Embedding·GRU·Head gradient 및 실제 Parameter Update PASS. **Optimizer Step은 정확히 1회**. Synthetic Backward 대체를 사용하지 않았다.
- **BLOCKED_DATA:** 전체 91개 순차 Replay와 완전 Dataset, Validation/Test 실데이터 Coverage.
- **NOT_EXECUTED:** CUDA/MPS, 장기 BC/HPO, Closed-loop Policy Rollout/Success Rate, 외부 게시/Commit/Production 변경.

[Forward/Backward 수치](../../experiments/{run}/real_forward_backward.json) · [Tensor/Gradient/Before/After NPZ](../../experiments/{run}/real_forward_backward.npz) · [저장 Packet 결과](../../experiments/{run}/stored_packet_forward.json).

사전 실행한 기존 GT Prototype 회귀 suite에는 자체 합성 overfit 단위 테스트(기존 모델 3종 × 80 step)가 포함돼 있었다. 이 결과는 [별도 Preflight Log](../../experiments/{run}/preflight_legacy_tests.log)에 보존하고, 신규 통합 모델의 실제 데이터 1-step Smoke와 구분한다. 최종 suite에서는 해당 overfit 검사를 반복하지 않았다. 새 Expert 장기 BC 학습은 실행하지 않았다.

![실제 관측 Graph](../../experiments/{run}/plots/graph_rgb.png)
![실제 RGB 기반 Mask](../../experiments/{run}/plots/actual_masks.png)
![Feature/Mask](../../experiments/{run}/plots/features_masks.png)
![Replay 오차](../../experiments/{run}/plots/replay_alignment.png)

## 비용과 한계

CPU packet→action(inference, B=1,T=1, warmup3+10회) wall mean **{lat['mean']:.3f} ms**, p95 **{lat['p95']:.3f} ms**, max **{lat['max']:.3f} ms**. Receiver/Robot k-NN/Global-M/Tensor packing/GNN/Readout/Fusion/Encoder/GRU/Head를 나눴다. 이 시간은 원본 RGB-D 수집·Mask·Offline Graph·IO를 제외한다. [단계별 Latency/Memory](../../experiments/{run}/latency_memory.json).

수집과 Offline Vision의 실제 처리시간은 별도 Manifest에 Frame/Episode 단위로 남겼고 [집계/측정 경계](../../experiments/{run}/data_pipeline_costs.json)에서 확인할 수 있다. Replay 시간은 저장한 마지막 반복의 reset/initial restore 이후 capture/check/step 구간이며 수집 전체 총시간이 아니다. SAM2 시간은 초기 검출/초기화가 포함된 Frame과 steady frame이 섞이며, RGB JPEG 파일 준비/모델 로딩/압축·IO는 별도로 계측하지 않은 overhead다. 수집/Offline Vision의 isolated peak memory도 NOT_EXECUTED다. 과거 Task05-E 수치를 합쳐 End-to-end 속도나 20Hz를 주장하지 않는다. 통합 RSS는 import/이전 backward를 포함하는 process high-water이며 isolated inference peak가 아니다.

![CPU stages](../../experiments/{run}/plots/latency.png)

후속 BC 전 조건: 전체 순차 Replay drift의 원인/재현 환경을 해결하고 기존 Gate로 다시 Pilot·91개 Coverage를 검증할 것, Timeout/Reset 의미를 확정할 것, Validation/Test까지 같은 Causal Mask/Schema/Manifest를 확보할 것, 관측 Mask/Depth 오염과 Empty Object 분포를 검토할 것. 현재 Mask/Graph의 물리 정확도나 BC 성공률은 검증하지 않았다. 최소 계약 통과는 학습 성능의 근거가 아니다.

재현 명령과 책임 구분은 [패키지 README](../../graph/integration_baseline/README.md)에 있다. 이전 Task03~05 Source/Artifact/Sidecar와 기존 GT/Recurrent Prototype은 보호 SHA-256 검사로 불변 확인했다. [보호 결과](../../experiments/{run}/source_protection.json), [Source Snapshot/Hashes](../../experiments/{run}/source_hashes.json), [Input Hashes](../../experiments/{run}/input_manifest.json).
'''
    REPORT.write_text(md)
    body=['<h1>Task06 · Graph → GNN → GRU → Action7</h1>',
        '<p class="status">Minimal REAL_ALIGNED prefix path verified · Full replay BLOCKED_DATA</p>',
        '<p>UUID '+run+' · CPU / pd_ee_delta_pose · exactly one optimizer step</p>',
        '<p><a href="../../docs/visual-graph-integration/06_graph_gnn_gru_baseline.md">결과 Markdown</a> · '
        '<a href="summary.json">Summary</a> · <a href="verification.json">Verification</a> · <a href="tests.log">Tests</a></p>',
        '<h2>Implementation & model contract</h2><p>3 actual robot nodes + original Peg/Box rows. Raw EDE + object-wise k=4 + exact global M=16. '
        'Attr4 = distance, destination−source unit direction. Mask-aware finite F5 view; original NaNs remain in packet.</p>',
        '<pre>Node14 + type4 → 2 Edge-MLP layers (d64)\n'
        '[h_TCP, h_Left, h_Right, mean(h_Peg), mean(h_Box)] = 320\n'
        '+ object_present2 + normalized_robot76 = 398\n'
        'RNNPolicy encoder128 → GRU(1,B,144) → regression Action7</pre>',
        '<p>Empty object slot = zero; robot stays present. Graph consumer never recomputes Delaunay or strict support/depth validation.</p>',
        table(['Actual packed encoder stage','Tensor shape'],
              [[('Node encoder' if i==0 else f'GNN layer {i-1}'),shape] for i,shape in enumerate(actual['trace']['layer_shapes'])]+
              [['Message / layer',actual['trace']['message_shape']],['Structured readout',actual['trace']['readout_shape']],
               ['Fused frames',actual['trace']['fused_shape']],['Action sequence',actual['action_shape']],['GRU hidden',actual['hidden_shape']]]),
        '<h2>Actual packet and edge types</h2><img src="plots/graph_rgb.png" alt="real graph and same RGB">',
        '<img src="plots/actual_masks.png" alt="actual RGB-only predicted masks, no simulator segmentation">',
        '<img src="plots/features_masks.png" alt="F5 and masks">',
        '<h2>Full Robot State</h2>'+table(['Field','Offset','Dimension','Units','Frame','Source'],field_rows),
        '<p>Available get_proprioception fields retained in full; controller state empty. Duplicate gripper and position information retained. '
        'TCP/left/right/base sources: agent.tcp / agent.finger1_link / agent.finger2_link / agent.robot.get_root(); '
        'each uses pose.raw_pose, linear_velocity and angular_velocity. Gripper fields use last two qpos/qvel and their sums. '
        'No task actor/hole/goal/reward/success or action enters the robot adapter. Train-only fit, then frozen.</p>',
        '<h2>Expert Replay · coverage & failures</h2><p>'+html.escape(coverage_text)+'</p>',
        table(['Episode','Valid obs','Targets','Source actions','Failure t','Prefix max qvel error','Repeat robot error','Status'],replay_rows),
        '<p>Initial-state repair preserved; restore bound 2e-5 / replay bound 1e-3. Failed frames and suffixes excluded; final valid observation unlabelled. '
        'State-only drift matches RGB-D; exact root cause unresolved. TimeLimit flags preserved and sequences cannot cross them.</p>',
        '<img src="plots/replay_alignment.png" alt="replay errors">',
        '<p><a href="replay_alignment_coverage.json">Replay manifest</a> · <a href="replay_diagnostic.json">Independent diagnostic</a> · '
        '<a href="dataset_manifest.json">Vision dataset manifest</a> · <a href="dataset_validation.json">Join checks</a></p>',
        '<h2>Observation/action alignment & causal masks</h2><pre>o_t → stored expert a_t → environment.step → o_(t+1)\n'
        'join: source SHA-256 + episode/reset identity + timestep\n'
        'RGB-D → prefix-causal GDINO/forward SAM2 → unchanged Shape-First/PCA5 → EE → Raw EDE</pre>',
        '<p>All '+str(summary['vision_frames'])+' verified prefix observations have packets. No simulator segmentation, GT point synthesis, future-frame initialization or stale fill. '
        'SAM2 optional C extension missing: its own hole-filling fallback warning is retained in vision.log.</p>',
        '<h2>Stored packet contracts — synthetic robot explicitly</h2>'+table(['Condition','Object N','Edges','Object presence','Status'],condition_rows),
        '<h2>Actual contiguous forward/backward</h2><p>REAL_ALIGNED train-prefix B=2,T=4; nonzero initial hidden; '
        'sequence/step max error '+str(actual['sequence_step_max_abs_error'])+'. Hidden episode/reset/termination contracts PASS. '
        'MSE '+str(actual['bc_loss'])+'; ONE Adam step and all required module updates.</p>',
        '<p>Separate preflight: the existing legacy GT-model unit test ran 3 synthetic overfit fixtures ×80 steps. '
        '<a href="preflight_legacy_tests.log">Legacy preflight log</a>. This is distinct from the new integration model and was not repeated in the final suite.</p>',
        table(['Module','Gradient L2','Parameter update L2'],[[k,v['l2_norm'],actual['parameter_updates_l2'][k]] for k,v in actual['gradients'].items()]),
        '<p><a href="real_forward_backward.json">Shapes, frame mapping & gradients</a> · '
        '<a href="real_forward_backward.npz">Actual tensors and parameters before/after</a></p>',
        '<h2>CPU latency / memory</h2><img src="plots/latency.png" alt="stage latency">',
        '<p>Packet→action only; upstream Vision, IO, replay and reports excluded. '+str(lat['mean'])+' ms mean. '
        'RSS high-water includes prior backward/imports; no isolated peak claim. <a href="latency_memory.json">Raw samples and boundaries</a>. '
        '<a href="data_pipeline_costs.json">Replay and offline Vision costs, separately scoped</a>.</p>',
        '<h2>Limits & unexecuted</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations']+summary['not_executed'])+'</ul>',
        '<p>Long BC, HPO, closed-loop rollout, production changes and publishing require a separate task. '
        'Complete replay drift, Timeout semantics and held-out coverage remain prerequisites.</p>',
        '<p><a href="source_hashes.json">Source hashes</a> · <a href="source_protection.json">Prior artifacts protected</a> · '
        '<a href="input_manifest.json">Input hashes</a> · <a href="implementation.diff">Implementation diff</a></p>']
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document+='<title>Task06 integration baseline</title><style>body{max-width:1200px;margin:32px auto;padding:0 22px;font:15px/1.65 system-ui;color:#203043}h1,h2{line-height:1.25}h2{margin-top:36px}a{color:#16669a}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}td,th{border:1px solid #d6dee6;padding:7px;text-align:left;vertical-align:top}th{background:#eaf1f7}img{display:block;width:100%;height:auto;margin:18px 0}pre{background:#edf3f6;padding:16px;overflow:auto}.status{padding:14px;background:#fff2d7;border-left:5px solid #b77b16}</style><body>'
    document+=''.join(body)+'</body></html>'
    (out/'graph_gnn_gru_baseline.html').write_text(document)


def verify(out,report):
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            self.targets.extend(v for k,v in attrs if k in ('href','src'))
    checked=dict(json=0,npz=0,png=0,relative_links=0)
    failures=[]
    for path in sorted(out.rglob('*')):
        if not path.is_file():continue
        if path.suffix=='.json':
            json.loads(path.read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)));checked['json']+=1
        elif path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as a:
                for key in a.files:
                    if a[key].dtype.hasobject:raise ValueError('object array')
            checked['npz']+=1
        elif path.suffix=='.png':
            with Image.open(path) as im:im.load()
            checked['png']+=1
    parser=Links();parser.feed((out/'graph_gnn_gru_baseline.html').read_text())
    entries=[(out/'graph_gnn_gru_baseline.html',link) for link in parser.targets]
    entries += [(report,url) for url in re.findall(r'\]\(([^)]+)\)',report.read_text())]
    # verification.json is being created here; its self-link is checked after write.
    for page,link in entries:
        if '://' in link or link.startswith('//'):
            failures.append('external reference '+link);continue
        target=(page.parent/link.split('#')[0]).resolve()
        if target.name=='verification.json' and target.parent==out.resolve():continue
        if not target.exists():failures.append(str(target))
        checked['relative_links']+=1
    summary=json.loads((out/'summary.json').read_text())
    from graph.integration_baseline.data import ReplayDataset
    data=ReplayDataset(summary['dataset_paths']['replay'],summary['dataset_paths']['vision'])
    validation=data.validation_report()
    # Include the three final, deliberately UNLABELLED observations in the
    # dataset readback. They must never be forced through the action loader.
    from graph.integration_baseline.io import read_packet
    from graph.vision_input import validate_packet
    packet_count=0
    unlabelled=0
    for episode in data.graph_manifest['episodes']:
        eid=episode['episode_id'];raw=data.raw[eid]
        times=[r['timestep'] for r in episode['packets']]
        if times!=list(range(len(raw['robot_state']))):
            failures.append(f'noncontiguous verified-prefix graph coverage episode {eid}')
        for row in episode['packets']:
            path=data.vision/row['path']
            if sha256(path)!=row['sha256']:raise AssertionError('all-packet hash mismatch')
            packet=read_packet(path);validate_packet(packet)
            t=row['timestep'];provenance=packet['observation_provenance']
            if (packet['episode']!=data.manifest['source_sha256']+f':traj_{eid}'
                    or packet['source_frame']!=t or packet['frame_idx']!=t
                    or packet['timestamp']!=float(raw['timestamp'][t])
                    or provenance['episode_id']!=eid or provenance['timestep']!=t
                    or provenance['replay_sha256']!=episode['replay_sha256']):
                raise AssertionError('all-observation source/timing join mismatch')
            np.testing.assert_array_equal(packet['observation_ee_pose'],raw['tcp_pose'][t].astype(np.float64))
            np.testing.assert_array_equal(packet['camera_to_world'],raw['base_camera/cam2world'][t].astype(np.float64))
            packet_count+=1
            if not row['has_action_target']:unlabelled+=1
    source=json.loads((out/'source_hashes.json').read_text())
    root=Path(__file__).resolve().parents[2]
    bad=[p for p,digest in source.items() if sha256(root/p)!=digest]
    if bad:failures.extend(bad)
    # Reconstruct the ZERO-initial-hidden BC prediction from saved pre-step
    # weights and tensors. The sequence/step comparison intentionally used a
    # separate nonzero initial hidden state; its predictions are not the loss.
    import torch
    from graph.integration_baseline.graph import PackedGraphs
    from graph.integration_baseline.model import IntegrationPolicy,bc_loss
    from graph.integration_baseline.robot import RobotNormalizer
    from graph.prototype.v2.state import StateNormalizer
    config=json.loads((out/'config.yaml').read_text())['provisional_smoke_config']
    with np.load(out/'real_forward_backward.npz',allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    ptr=torch.from_numpy(arrays['ptr'])
    owners=torch.repeat_interleave(torch.arange(len(ptr)-1),ptr[1:]-ptr[:-1])
    packed=PackedGraphs(*(torch.from_numpy(arrays[k]) for k in ('node_input','node_type','edge_index','edge_attr','edge_type')),
                        owners,ptr,torch.from_numpy(arrays['frame_map']))
    model=IntegrationPolicy(data.schema,d=config['d'],L=config['L'],gru_hidden=config['gru_hidden'],
                             gru_layers=config['gru_layers'],enc_hidden=config['enc_hidden']).eval()
    model.load_state_dict({k.removeprefix('before/'):torch.from_numpy(v) for k,v in arrays.items() if k.startswith('before/')})
    norm=RobotNormalizer(data.schema,StateNormalizer(torch.from_numpy(arrays['normalizer_mean']),
                                                    torch.from_numpy(arrays['normalizer_std'])),
                         summary['real_forward_backward']['normalizer']['episode_ids'])
    with torch.inference_mode():
        prediction,_=model(packed,torch.from_numpy(arrays['robot_state']),norm)
        target=torch.from_numpy(arrays['target']);mask=torch.from_numpy(arrays['target_mask'])
        recomputed_loss=float(bc_loss(prediction[mask],target[mask],'regression'))
    expected_loss=summary['real_forward_backward']['bc_loss']
    if not np.isclose(recomputed_loss,expected_loss,rtol=1e-6,atol=1e-7):
        raise AssertionError('saved-weight/tensor BC loss readback mismatch')
    for group in summary['real_forward_backward']['gradients']:
        gradients=[v for k,v in arrays.items() if k.startswith('gradient/'+group)]
        updated=[np.any(arrays['before/'+name]!=value) for name,value in
                 ((k.removeprefix('after/'),v) for k,v in arrays.items() if k.startswith('after/'+group))]
        if not gradients or not all(np.isfinite(g).all() for g in gradients) or not any(np.any(g!=0) for g in gradients) or not any(updated):
            raise AssertionError('gradient/parameter readback '+group)
    result=dict(status='PASS' if not failures else 'FAIL',checked=checked,failures=failures,
        aligned_target_readback=validation['validated_targets'],
        all_observation_packet_readback=packet_count,unlabelled_final_observations=unlabelled,
        saved_weight_tensor_bc_loss_readback=dict(status='PASS',value=recomputed_loss,expected=expected_loss,initial_hidden='zero'),
        saved_gradient_parameter_readback='PASS',
        byte_readback='JSON/NPZ at each write; Vision save_packet packet_digest at write; stored hashes rechecked',
        source_snapshot_current_match=not bad,prior_sources_artifacts='source_protection.json',
        browser='NOT_EXECUTED',png_visual_review='PENDING',
        gpu=dict(CUDA='NOT_EXECUTED',MPS='NOT_EXECUTED'),full_replay='BLOCKED_DATA',
        actual_forward_backward='REAL_ALIGNED_PREFIX_ONLY',stored_packet_forward='REAL_PACKET_WITH_SYNTHETIC_ROBOT',
        synthetic='construction/oracles/reset/empty/malformed-data fixtures only')
    write_json(out/'verification.json',result)
    if not (out/'verification.json').exists():raise AssertionError('verification self link')
    if failures:raise AssertionError('artifact/link failures: '+str(failures))

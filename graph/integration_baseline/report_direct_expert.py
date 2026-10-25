"""Task06-B local report/plots/readback, without simulator execution or publishing."""
import argparse
import csv
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graph.integration_baseline.io import sha256, write_json, save_npz, read_packet
from utils.experiment_artifacts import register_existing_output

DOC=Path('docs/visual-graph-integration/06b_recurrent_expert_collection_pilot.md')
ABORTED=Path('experiments/5a71640e4a9f4036b109bcaea7c17824')


def read(p): return json.loads(Path(p).read_text())


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def historical(out):
    prefix=out/'source_snapshot/origin_recurrent'
    rows=list(csv.DictReader((prefix/'docs/reports/recurrent_main_20261023/episodes.csv').open()))
    experts=[r for r in rows if r['model']=='expert']
    result=dict(source_ref=read(out/'pilot_manifest.json')['recurrent_ref'],
        expert_episodes=len(experts),successes=sum(r['success']=='True' for r in experts),
        within100=sum(int(r['steps'])<=100 for r in experts),
        first_success_step_range=[min(int(r['steps']) for r in experts),max(int(r['steps']) for r in experts)],
        conditions={name:dict(episodes=sum(r['condition']==name for r in experts),
            successes=sum(r['condition']==name and r['success']=='True' for r in experts),
            within100=sum(r['condition']==name and int(r['steps'])<=100 for r in experts)) for name in ('normal','hole_shift')},
        seeds=sorted(set(int(r['episode_seed']) for r in experts)),
        backend=sorted(set(r['sim_backend'] for r in experts)),
        fixture_sha256=experts[0]['initial_state_source'],
        current_runtime_reproduction=False,
        evidence='Git CSV/provenance only; private original evaluation JSONL/fixture not fetched or executed')
    write_json(out/'historical_expert_evidence.json',result)
    return result


def mask_diagnosis(out, vision):
    if not vision.get('packet_rows'):return dict(status='NOT_EXECUTED',rows=[])
    ep=next(e for e in read(out/'pilot_manifest.json')['episodes'] if e['episode_id']==vision['episode_id'])
    with np.load(out/ep['path'],allow_pickle=False) as a, np.load(out/'vision/predicted_masks.npz',allow_pickle=False) as z:
        masks=z['masks'];rows=[]
        for t,m in enumerate(masks):
            valid=np.isfinite(a['base_camera/depth'][t])&(a['base_camera/depth'][t]>0)
            overlap=m[0]&m[1]
            rows.append(dict(timestep=t,mask_pixels=m.sum((1,2)).tolist(),overlap_pixels=int(overlap.sum()),
                valid_nonoverlap_support=[int((m[o]&valid&~overlap).sum()) for o in (0,1)],
                invalid_depth_mask_pixels=[int((m[o]&~valid).sum()) for o in (0,1)]))
    empty=sum(r['valid_nonoverlap_support'][1]==0 for r in rows)
    full_overlap=sum(r['overlap_pixels']==r['mask_pixels'][1] for r in rows)
    result=dict(status='BLOCKED_BOX_OBSERVATION_COVERAGE' if empty else 'NONEMPTY_NOT_QUALITY_SCORED',rows=rows,
        basis=f'{empty}/{len(rows)} frames have zero Box support; {full_overlap}/{len(rows)} have fully overlapping Box masks',
        cause_scope='existing geometry observed_mask excludes overlap from both supports; semantic mask accuracy not GT-scored',
        new_mask_algorithm=False,gt_used=False)
    write_json(out/'mask_coverage_diagnosis.json',result)
    return result


def plots(out,m):
    folder=out/'plots';folder.mkdir(exist_ok=True)
    ep=m['episodes'][0]
    with np.load(out/ep['path'],allow_pickle=False) as z: a={k:z[k] for k in z.files}
    samples=sorted(set([0,len(a['robot_state'])//2,len(a['robot_state'])-1]))
    fig,axes=plt.subplots(4,len(samples),figsize=(10,9),layout='constrained')
    for col,t in enumerate(samples):
        for camera_idx,camera in enumerate(('base_camera','hand_camera')):
            axes[camera_idx*2,col].imshow(a[camera+'/rgb'][t]);axes[camera_idx*2,col].set_title(f'{camera} t={t}')
            d=a[camera+'/depth'][t];im=axes[camera_idx*2+1,col].imshow(np.ma.masked_where(d<=0,d),vmin=0,vmax=2,cmap='viridis')
            axes[camera_idx*2+1,col].set_title('Depth (m); nonpositive masked',fontsize=9)
            for row in (camera_idx*2,camera_idx*2+1):axes[row,col].axis('off')
    fig.colorbar(im,ax=axes[1::2].ravel().tolist(),label='metres',shrink=.5)
    fig.savefig(folder/'rgbd_samples.png',dpi=140);plt.close(fig)
    save_npz(out/'rgbd_robot_sample.npz',**{k:v[samples] for k,v in a.items() if len(v)==len(a['robot_state'])})
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    axes[0].bar([str(e['seed']) for e in m['episodes']],[e['actions'] for e in m['episodes']],color=['#2c8b76' if e['success'] else '#bb4e48' for e in m['episodes']])
    axes[0].axhline(100,color='#aa5a20',ls='--',label='100-step reference');axes[0].axhline(250,color='gray',ls=':',label='explicit horizon250')
    axes[0].set_ylabel('Applied actions');axes[0].legend()
    axes[1].plot(a['timestep'],a['robot_state'][:,:9]);axes[1].set_ylabel('qpos (rad / m)');axes[1].set_xlabel('Observation timestep (pre next action)')
    axes[2].plot(np.arange(len(a.get('actions',[]))),a.get('actions',np.empty((0,7))));axes[2].set_ylabel('Native action7');axes[2].set_xlabel('action t maps observation t to t+1')
    fig.savefig(folder/'episode_timeline.png',dpi=140);plt.close(fig)
    vision=read(out/'vision_coverage.json') if (out/'vision_coverage.json').exists() else {}
    if vision.get('packet_rows'):
        row=vision['packet_rows'][min(3,len(vision['packet_rows'])-1)];t=row['timestep'];p=read_packet(out/row['path'])
        with np.load(out/'vision/predicted_masks.npz',allow_pickle=False) as z:masks=z['masks'][t]
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        axes[0].imshow(a['base_camera/rgb'][t]);overlay=np.zeros((*masks.shape[1:],4))
        overlay[masks[0]]=[.2,.85,.6,.5];overlay[masks[1]]=[.9,.4,.2,.5];axes[0].imshow(overlay)
        axes[0].set_title(f'Predicted causal SAM2 masks t={t}');axes[0].axis('off')
        points=p['points'];edge=p['intra_object_edge_index'];objects=p['object_id']
        for ax,coords,label in [(axes[1],(0,1),'EE XY'),(axes[2],(0,2),'EE XZ')]:
            from matplotlib.ticker import MaxNLocator
            for src,dst in edge.T:
                ax.plot(points[[src,dst],coords[0]],points[[src,dst],coords[1]],color='#b0bac8',alpha=.15,lw=.4)
            ax.scatter(points[:,coords[0]],points[:,coords[1]],c=objects,cmap='coolwarm',s=5)
            from graph.integration_baseline.robot import geometry_in_ee
            robot=geometry_in_ee(a['robot_world_geometry'][t],a['tcp_pose'][t]);ax.scatter(robot[:,coords[0]],robot[:,coords[1]],c='black',marker='x',s=50)
            ax.set_title(label+' / Raw EDE');ax.set_aspect('equal')
            ax.xaxis.set_major_locator(MaxNLocator(2));ax.tick_params(labelsize=8)
        fig.savefig(folder/'predicted_mask_raw_ede.png',dpi=150);plt.close(fig)


def report(out):
    out=Path(out);run=out.name;m=read(out/'pilot_manifest.json');schema=read(out/'robot_state_schema.json')
    action=read(out/'expert_action_schema.json');runtime=read(out/'executed_conditions.json');past=historical(out)
    vision=read(out/'vision_coverage.json') if (out/'vision_coverage.json').exists() else dict(status='NOT_EXECUTED',packet_rows=[])
    forward=read(out/'sequence_forward.json') if (out/'sequence_forward.json').exists() else dict(status='NOT_EXECUTED')
    n=sum(e['actions'] for e in m['episodes']);obs=sum(e['observations'] for e in m['episodes'])
    aligned=sum(e['actions'] for e in m['episodes'] if e.get('alignment',{}).get('status')=='PASS')
    success=sum(e['success'] for e in m['episodes']);within=sum(e['success'] and e['first_success_step']<=100 for e in m['episodes'])
    graphs=vision.get('packet_rows',[]);both=sum(all(c>0 for c in r['object_counts']) for r in graphs)
    diagnosis=mask_diagnosis(out,vision)
    plots(out,m)
    costs=dict(episode_wall_seconds=[e['elapsed_seconds'] for e in m['episodes']],
        total_episode_seconds=sum(e['elapsed_seconds'] for e in m['episodes']),
        collection_scope='reset + expert + twice camera reads + independent H5 append + validation + NPZ compression; process/env startup excluded',
        npz_bytes=sum(e['compressed_npz_bytes'] for e in m['episodes']),
        h5_bytes=sum((out/e['path']).with_suffix('.h5').stat().st_size for e in m['episodes']),
        uncompressed_array_bytes=sum(e['uncompressed_array_bytes'] for e in m['episodes']),
        vision_seconds=vision.get('elapsed_seconds'),model_load_seconds=vision.get('model_load_seconds'),
        mask_total_seconds=sum(r['timings_ms']['mask_ms'] for r in graphs)/1000,
        packet_total_seconds=sum(r['timings_ms']['total_offline_graph_ms'] for r in graphs)/1000,
        packet_bytes=sum((out/r['path']).stat().st_size for r in graphs),
        sequence_forward=forward.get('elapsed_seconds'),process_maxrss_bytes=vision.get('process_maxrss_bytes'),
        unmeasured=['isolated collection/graph/inference peak RSS','GPU cost/memory','large-dataset loader throughput','full-episode mask/graph cost'])
    write_json(out/'data_pipeline_costs.json',costs)
    coverage=dict(episodes=len(m['episodes']),successes=success,within100=within,actions=n,observations=obs,
        action_aligned=aligned,robot_missing=0 if aligned==n else None,
        rgbd_observations=obs if aligned==n else None,vision_packet_frames=len(graphs),
        both_objects_nonempty=both,vision_processed=vision.get('frames_requested',0),
        vision_unprocessed=obs-vision.get('frames_requested',0),
        note='packet-valid and two-object observed coverage are separate; no full dataset split or quality claim')
    write_json(out/'alignment_coverage.json',coverage)
    aborted=dict(run_id=ABORTED.name,attempts=5,applied_actions_per_attempt=1,observations_saved_per_attempt=1,
        saved_actions_per_attempt=0,status='CENSORED_RECORDER_ERROR',cause='reset replaces controller; pre-reset hook lost',
        correction='install audit on post-reset controller',retry_authorization='user explicitly allowed same five seeds retry',
        eligible_for_bc=False,expert_task_success_rate='NOT_MEASURED',
        original_manifest_has_zero_action_counters='counter updated after failed audit; this annotation corrects applied count without rewriting original evidence')
    write_json(out/'aborted_attempts.json',aborted)
    q=[
      ('Q1','CONFIRMED','현재 recurrent/expert.py::ScriptedExpert는 origin/recurrent와 동일하다. 기존 진입점 recurrent.collect / recurrent.evaluate의 ExpertPolicy, 신규 opt-in collect_recurrent_expert가 원본을 호출한다.'),
      ('Q2','CONFIRMED' if aligned else 'PARTIALLY_CONFIRMED','원본 native 7D를 그대로 BC Target으로 보존한다. pd_ee_delta_pose Controller와 Task06 regression7 의미가 같다. Expert axis-angle 계산과 Controller XYZ Euler 적용은 다르지만 새 변환 없이 동일 Controller 명령을 모사하는 계약이다. 명목 Box 범위 초과 회전은 별도 계수한다.'),
      ('Q3','CONFIRMED' if success else 'BLOCKED',f'수정 후 normal Pilot {success}/{len(m["episodes"])} 성공, first-success steps '+str([e['first_success_step'] for e in m['episodes']])+f'. 100-step 이내 {within}/{len(m["episodes"])}. 과거 60/60 재현이 아니다.'),
      ('Q4','CONFIRMED' if aligned==n and n else 'BLOCKED',f'{n} actions / {obs} observations. 유효 Target {aligned}. Exact step/controller input, pre-action live state, same-time returned/fresh RGB-D, H5↔NPZ bytes와 episode/timestep identity를 검사했다.'),
      ('Q5','PARTIALLY_CONFIRMED',f'성공 Episode의 prefix {len(graphs)} Packet; 두 물체 nonempty {both}. 실제 Sequence Forward {forward["status"]}. Box가15/16 Frame에서 사라져 학습용 Vision Coverage는 BLOCKED다. ReplayDataset는 공식 H5/split 전용이며 신규 Reader가 필요하다.'),
      ('Q6','PARTIALLY_CONFIRMED','새 Observation과 Action을 같은 live 실행에서 만들므로 과거 Converted Trajectory의 state 재현을 필요로 하지 않는다. 과거 Replay Drift 원인 해결은 아니다. Mask/Depth 품질, GT expert 의존성, seed coverage와 새 dataset 계약이 남는다.'),
      ('Q7','NOT_TESTED','대규모 수집 전 split/reset group·action raw target·first success/tail·timeout boundary를 확정하고, 전체 Episode causal Vision coverage 및 graph 품질, loader/normalizer/hidden reset과 비용을 검증해야 한다. Full Dataset/BC 학습은 별도 승인 대상이다.')]
    write_json(out/'questions.json',[dict(question=a,status=b,answer=c) for a,b,c in q])
    stages=[('Expert source/config','PASS'),('Action raw→step→controller','PASS' if aligned else 'BLOCKED'),
            ('250-step direct pilot','PASS' if success else 'BLOCKED'),('T/T+1 / robot76 / camera','PASS' if aligned==n and n else 'BLOCKED'),
            ('Causal Vision prefix',vision['status']),('Actual sequence forward',forward['status']),
            ('Two-object Vision coverage',diagnosis['status']),
            ('Full-episode/all-episode Vision','NOT_EXECUTED'),('CUDA / MPS','NOT_EXECUTED'),
            ('Full Dataset / BC training / HPO / Policy rollout','NOT_EXECUTED')]
    rows=[(e['seed'],e['success'],e['first_success_step'],e.get('terminated'),e.get('truncated'),e['actions'],e['observations'],e['failure_reason'],e.get('nominal_box_exceeding_rows')) for e in m['episodes']]
    mdrows='\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)
    first=m['episodes'][0]
    body=f'''# Task06-B — Recurrent ScriptedExpert Direct Collection Pilot

Run UUID: `{run}`. **현재 Runtime 직접 수집 {success}/{len(m['episodes'])} 성공, {aligned}/{n} Action 정합.**

[Static HTML](../../experiments/{run}/recurrent_expert_collection_pilot.html) · [검증](../../experiments/{run}/verification.json) · [Pilot Manifest](../../experiments/{run}/pilot_manifest.json) · [Alignment/Coverage](../../experiments/{run}/alignment_coverage.json) · [비용](../../experiments/{run}/data_pipeline_costs.json) · [질문별 JSON](../../experiments/{run}/questions.json)

## 1. 기존 Source와 과거 기록

현재 branch `visual-graph`, HEAD `63c3f6c9fe0a4ac5b8e0096fc7adca20ca21e4f8`, 시작부터 dirty/untracked 작업이 있었다. origin/recurrent `{m['recurrent_ref']}`는 git show로만 조사했다. checkout/fetch/설치/commit/외부 upload는 하지 않았다.

Expert는 [recurrent/expert.py](../../recurrent/expert.py)::ScriptedExpert, SHA-256 `{m['expert_source_sha256']}`. origin/recurrent와 해당 파일 bytes가 같다. 기본 설정 max_trans=0.02m, max_rot=0.15rad, insert_speed=0.006m, gain=3.0을 그대로 사용한다. `reset()` 후 `act()`가 현재 live task를 읽는다. 기존 진입점은 recurrent.collect::run_episode, recurrent.evaluate::ExpertPolicy다. 기존 Collector는 noise=0.02/perturb_frac=0.5 기본값, 성공만 keep, T observations/T actions이므로 이번 계약에 그대로 쓰지 않았다. 신규 [opt-in Recorder](../../graph/integration_baseline/collect_recurrent_expert.py)는 noise=0, hole shift 없음, 실패 포함, T/T+1이며 원본 Expert를 수정하지 않는다.

`docs/reports/recurrent_main_20261023/`는 현재 working tree에 없고 origin/recurrent에 있다. [읽은 원본 보고서](../../experiments/{run}/source_snapshot/origin_recurrent/docs/reports/recurrent_main_20261023/README.md)와 [CSV 집계](../../experiments/{run}/historical_expert_evidence.json)를 보존했다. 과거 Expert **60/60 = normal30 + hole_shift30**, seed2000–2029를 조건별 재사용, horizon250, physx_cpu, obs state, corruption none. Hole shift는 transport 후10 step에 seed 짝홀 부호로 ±0.02m를 적용한다. 과거 100 이내 성공10/60, 성공 길이85–198이다. 250은 cap이며 실제 최대길이250이라는 뜻은 아니다. 원본 CPU fixture SHA `{past['fixture_sha256']}`는 로컬에 없으므로 이번에는 reset(seed)를 사용한다. 과거 report 평가 Torch2.14.1+cu130/Linux/Python3.11.17과 현재 host가 다르다. 원본 private JSONL/fixture는 내려받지 않았고 CSV·provenance 근거만 확인했다. 과거 정책11,340 Episode/84개 학습 실행은 이 Expert60회와 별도이며 이번에 학습 결과를 재현하지 않았다. Visual ScriptedPolicy5/5 및 공식 Converted91은 이번 수집·성공률에 합치지 않는다.

## 2. Expert 입력과 Student 입력 분리

Expert reset은 GT peg actor OBB, peg_half_sizes, 현재 TCP closing 축으로 grasp를 결정한다. act는 TCP world pose, peg world pose, goal_pose(peg 삽입 목표), peg half-length, 내부 phase/counter/lift_target을 사용한다. Goal은 hole task GT에 근거한다. qpos/qvel과 fingertip은 Expert의 직접 입력이 아니며, 현재 TCP 및 물리 Controller를 통해 간접 관계가 있다. RGB-D/과거 observation buffer/미래 frame은 읽지 않는다. 상태 기계는 reach→descend→close(8회)→lift→align(조건3회)→insert다. 첫 grasp/lift target과 counter만 과거 정보를 기억한다. 종료는 Expert 자체가 아닌 Collector의 info.success/terminated/truncated/horizon이다.

Student 입력은 같은 시점 predicted-mask RGB-D Graph와 robot-only76D다. GT peg/hole/goal, reward/success, Expert phase/action은 입력에 전달하지 않는다. phase는 `episodes/*.expert_diagnostics.json`에 진단 전용으로 분리했다. raw 저장파일도 GT object pose를 포함하지 않는다.

## 3. Action Schema

[Expert Action Schema](../../experiments/{run}/expert_action_schema.json)의 mode는 pd_ee_delta_pose, float32[7], translation3/rotation3/gripper1이다. native output→env.step tensor→실제 CombinedController entry가 exact 일치한다. Target adapter/clipping/z-score/gripper BCE는 추가하지 않았다. Task06의 기존 continuous regression7을 그대로 쓴다.

Translation은 root-frame normalized 값이며 Controller가 [-1,1] clip 후 ±0.1m로 scale한다. Rotation은 norm>1이면 norm1로 제한한 뒤 `rot_lower=-0.1`을 곱하고 **XYZ Euler** delta로 바꿔 root에서 left-compose한다. Expert는 relative quaternion의 shortest axis-angle를 계산해 부호 반전/0.1 나눗셈을 하지만 이것이 Euler와 수학적으로 동일하다고 주장하지 않는다. 원본 제어 알고리즘의 근사이며 이번 BC target 의미는 **실제로 받는 원본 Controller 명령**이다. Gripper -1→-0.01m, +1→0.04m mimic target. Native rotation 성분이 명목 Box ±1을 넘을 수 있다. 해당 행수는 표와 NPZ에 그대로 남겼으며 임의 정규화하지 않았다. 실제 scaled arm/gripper input을 별도로 저장하고 설치 Controller 수식과 검증했다. 물리 axis-angle target 자체를 다른 Controller에 이식하는 호환성은 NOT_TESTED다.

## 4. 환경과 수정 후 Pilot

PegInsertionSide-v1 / `{runtime['robot']}` / `{runtime['backend']}`. Control {runtime['control_freq']}Hz, Physics {runtime['physics_freq']}Hz, {runtime['sim_steps_per_control']} physics steps/control. [실행 조건](../../experiments/{run}/executed_conditions.json). shared default100은 그대로 두고 `TimeLimitWrapper(env.unwrapped,250)`만 적용했다. 첫 Task success 또는 terminated/truncated에서 즉시 중지하며 post-success tail은0이다. First success는 Action count 기준1-based, action index는0-based다. 이번 normal 5개는 과거 두 조건60개 평가와 다르며 전체 성공률을 확정하지 않는다.

| Seed | Success | First success step | Terminated | Truncated | Actions | Observations | Failure | native Box 초과 행 |
|---|---|---|---|---|---|---|---|---|
{mdrows}

최초 시도 UUID `{ABORTED.name}`는 5개 seed마다 첫 Action 후 Recorder hook 누락으로 중단됐다. reset이 Controller를 교체해 pre-reset hook이 없어졌으며 **Expert task 실패율이 아닌 관측 중단/censored attempt**다. 각1 action 적용/1 pre-observation 보존/0 action row 저장. 원본 counter는 audit 뒤 갱신되어0으로 남았고 [정정 annotation](../../experiments/{run}/aborted_attempts.json)에 실제 적용 수를 기록했다. 원본 실패 Artifact/Source/log는 보존했다. 사용자가 같은5 seed 재실행을 명시 승인한 후 현재 Pilot을 수행했다. 이 중단5회를 숨기거나 60/60 재현으로 표현하지 않는다.

## 5. 동시 기록 및 Robot76D

계약: `o[t] → expert.act() → env.step(exact a[t]) → o[t+1]`. 모든 Episode의 T actions/T+1 observations를 보존하며 마지막 observation에는 Target이 없다. 각 reset 이후 실제 Controller에 감사 hook을 설치한다. command entry에서 읽은 robot state/geometry와 저장한 pre-action state/geometry가 exact 일치하고 elapsed_steps=t다. reset/step 반환 RGB-D와 같은 물리 시점에서 독립 get_obs를 호출한 RGB-D가 exact 일치한다. capture 전후 elapsed counter가 같고 env.step 외 physics stepping/reset/teleport가 없다. 같은 시점 hand/base calibration, image와 robot 상태를 함께 저장한다. Timestamp는 t/control_freq로 산출한 simulation relative seconds이며 별도 PhysX clock 직접 측정이 아니다.

RGB uint8[128,128,3], depth raw int16[128,128]mm, float metre view, intrinsic_cv, extrinsic_cv(world→camera), inverse CV cam2world, cam2world_gl를 두 Camera 모두 기록한다. 설치 shader는 position*1000을 int16로 cast하므로 precision/range 손실이 있고 nonpositive/-32768 count를 보존한다. raw만으로 모든 positive overflow를 식별·복구할 수 있다고 주장하지 않는다. 원본 depth를 보존하며 nonpositive는 downstream invalid다.

Robot Schema [필드/순서/차원/단위/frame](../../experiments/{run}/robot_state_schema.json)는 Task06 저장 Schema와 **dict exact,76D**다. qpos9/qvel9(rad,m/rad/s,m/s), gripper pos2/vel2/opening1/openingvel1, TCP·left/right finger·base 각pose7/world linear3/angular3다. Pose xyz+wxyz, velocities world; gripper/joint는 joint frame이다. [기존 robot_observation](../../graph/integration_baseline/robot.py)의 observable accessor를 그대로 사용한다. 실제 fingertip은 finger collision pad center local offsets (0,±0.00758,0.04525)m를 link pose로 변환한다. 각 frame의 link pose에서 independently reconstruction 및 EE transform 일관성을 검사했다. 없는 effort/acceleration/contact는 생성하지 않았다. Controller state는 빈 dict이며 필드 변경은 거부한다. Nonfinite0.

## 6. Alignment와 Provenance Gate

{aligned}/{n} Target에 대해 action bytes3경로, actual scaling, pre-state/geometry, timestep/episode/reset seed, 카메라·depth conversion/calibration, finite76과 boundary를 검사했다. RGB-D/Robot/action은 final NPZ와 **실행 중 append된 별도 HDF5**를 모든 dataset별 dtype/shape/bytes로 다시 비교했다. HDF5는 NPZ를 읽어 복제한 파일이 아니다. [독립 Reader](../../graph/integration_baseline/validate_direct_expert.py)는 source NPZ/H5 hash와 같은 gate를 다시 검사한다. Metadata에는 source hash/ref/config/runtime/backend/schema가 있다. 이전 Converted Replay gate2e-5/1e-3은 이 새로운 direct recording에 적용하지 않는다. Timestamp equality와 반복 camera read는 같은 live state의 동시 기록 근거이며 외부 센서 지연 측정은 아니다.

## 7. 실제 Vision Graph와 Coverage

성공 seed{vision.get('episode_id')}에서 **t0..{max([r['timestep'] for r in graphs],default=-1)} prefix만** 처리했다. Status `{vision['status']}`, Packet {len(graphs)}/{vision.get('frames_requested',0)}, 두 물체 nonempty {both}/{len(graphs)}. 전체관측{obs} 중 미처리{obs-vision.get('frames_requested',0)}는 Missing/NOT_EXECUTED이며 성공적인 full-episode Vision dataset으로 표현하지 않는다. [Coverage/실패/시간](../../experiments/{run}/vision_coverage.json).

**학습용 두 물체 관측 Coverage는 BLOCKED다.** [Mask/Depth/Overlap 진단](../../experiments/{run}/mask_coverage_diagnosis.json)에서 Box mask가15/16 Frame 전체에서 Peg mask와 겹친다. 기존 `OrganizedGeometry.observed_mask`가 ambiguous overlap을 양쪽 support에서 제외하여 Box0 Point가 된다. t11만 nonoverlap valid support2/selected2 Point다. 이 관측된 결손은 depth invalid보다 overlapping predicted masks로 설명된다. 어떤 mask가 의미적으로 틀렸는지 GT 점수로 확정하지 않았으며, Box GT 복원/새 mask 알고리즘/수정은 하지 않았다. Forward t0..3은 실제 결손 Box presence0 상태를 통과한 것이므로 양쪽 물체를 충분히 보는 Sequence 검증으로 해석할 수 없다.

Overlay를 직접 확인하면 Peg로 분류된 Mask가 Robot도 포함하는 것으로 보인다. 이는 RGB와 예측 Mask의 시각적 관찰이며 GT segmentation 기반 정량 정확도 평가는 아니다. Raw Dataset은 정합됐더라도 현재 Mask 결과를 올바른 Peg/Box geometry로 간주해 학습을 시작할 근거가 부족하다.

Task06 `causal_masks`와 `packet_from_frame`을 수정 없이 재사용했다. 현재 Prefix RGB-only DINO 초기화, current-frame decision, SAM2 num_frames=t+1, one-frame forward/no backward/no backfill, GT segmentation 없음. Sampling Shape-First, keep_ratio0.5, optimized PCA5/F5, availability/validity, same-time EE, observed component, intra Raw EDE/validation none, source row/index와 variable N을 보존했다. 빈 물체는 presence/missing으로 남기며 GT로 복원하지 않는다. 이전 인과성 실험은 Task06 기록이며 이번에는 그 검증된 코드 경로를 호출했다. 새 prefix/future perturbation 실험을 반복하지 않았다. Mask/Graph 물리 정확도 점수는 GT와 새로 비교하지 않았다.

## 8. 기존 Graph–GNN–GRU/BC

기존 build_graph/pack_graphs/collate_sequences/IntegrationPolicy/EpisodeStream을 그대로 사용했다. Robot node3, F5+availability3+validity3, Raw EDE+3, robot-object k4, peg-box exact globalM16, object presence, L2/d64, readout5d, robot76, GRU144/layer1, regression7이다. [실제 Forward](../../experiments/{run}/sequence_forward.json) `{forward['status']}`. Timesteps {forward.get('timesteps')}, output {forward.get('output_shape')}, node counts {forward.get('node_counts')}, hidden carry max error {forward.get('hidden_carry_max_error')}. Random initialized 기존 모델의 finite forward만 확인했으며 예측 성능을 뜻하지 않는다. 정규화는 과거 Task06 train-prefix에서 이미 frozen된 mean/std를 재사용하고 hash를 기록했다. 새 Pilot을 임의 train/test split하거나 새로운 학습 통계에 포함하지 않았다. Reset 검사는 실제 frame tensor에서 explicit identity/flag를 주입한 lifecycle contract이며 실제 Policy rollout은 아니다. Backward0/Optimizer0.

기존 ReplayDataset는 source H5·공식 split·replay gate에 묶여 있어 이 자료를 직접 읽지 못한다. 신규 DirectExpertReader는 독립 진단용이며 production loader/reset 정책을 변경하지 않는다. 실학습용 direct dataset loader와 source-kind별 provenance validation은 추가 구현이 필요하다.

## 9. 전체 Dataset 방향과 남는 결정

이번 5개는 feasibility pilot이며 성공률/seed/shape/camera/phase coverage와 최종 split 비율을 결정할 표본이 아니다. 원본 Expert default Collection의 action noise/perturbation을 쓸지 별도 결정하고, noisy label과 executed command가 다르면 둘을 분리해야 한다. 이 Pilot에서는 noise0로 exact 동일하다. Raw native Action7를 계속 쓸지와 명목 Box 초과 convention을 manifest에 고정해야 하며 자동 clipping은 금지한다.

새 dataset은 실제 reset 시작→첫 success/실제 termination까지 Transition과 final observation을 보존하는 방향이 맞다. 관리상 truncation은 physical reset과 별도 metadata로 저장하고 실제 reset/terminated에서 hidden을 reset해야 한다. Pilot은 horizon250 전에 성공한 경우 실제 timeout boundary를 실행하지 않았으므로 250 timeout 시료의 lifecycle은 NOT_TESTED다. timeout 이후 수집이 필요해지면 physical episode ID와 administrative segment ID를 분리하고 기존 loader/collate/stream의 truncated reset 정책을 별도 승인·검증해야 한다. 이번에는 수정하지 않았다.

공식91의73/9/9 ID는 재사용하지 않는다. 새 dataset UUID+reset seed/geometry 초기상태 hash/perturbation family로 reset group을 정의하고 episode/group 단위 별도 split manifest를 만들 것을 제안한다. 동일 초기상태 계열의 train/test 중복과 frame random split을 피하고 최종 비율은 Pilot만으로 정하지 않는다. Train split 확정 후에만 normalizer fit/freeze, validation/test로 통계를 갱신하지 않는다.

Full 수집 전 필요한 것은 opt-in Collector의 interruption/exception partial-finalization, stable controller lifecycle, 독립 ledger/hash/atomic manifest, 전체 Episode causal mask/graph 및 missing 정책, phase별 coverage/shape 난이도, frame/action/EE calibration gate, group split/normalizer/sequence loader와 IO/memory 검증이다. Graph 알고리즘이나 Expert 제어 개선으로 확대하지 않는다.

## 10. 실측 비용과 규모 추정

[실측 JSON](../../experiments/{run}/data_pipeline_costs.json): 수집 Episode 합 {costs['total_episode_seconds']:.3f}s, NPZ {costs['npz_bytes']:,} bytes, 독립 H5 {costs['h5_bytes']:,} bytes, raw array {costs['uncompressed_array_bytes']:,} bytes. 수집 시간은 reset/Expert/twice-camera-read/H5/NPZ/검증을 포함하며 env startup은 제외한다. 중복 H5는 진단 비용이며 full dataset 저장 기본값으로 처방하지 않는다.

Vision prefix wall {costs['vision_seconds']}s(loading {costs['model_load_seconds']}s), packet 내 mask 합 {costs['mask_total_seconds']:.3f}s, Shape-First/F5/Raw EDE 합 {costs['packet_total_seconds']:.3f}s, serialized packet {costs['packet_bytes']:,} bytes. Mask 타이밍은 초기 검출과 steady frame이 섞인다. SAM2 JPEG 준비/압축/IO·모델 load overhead가 있어 합산 core 시간과 wall은 다르다. CPU high-water RSS {costs['process_maxrss_bytes']} bytes는 process imports/model load를 포함하며 stage peak가 아니다. GPU 비용은 미실행.

후속 예산은 `예정 Action수/관측수 × 이번 per-row 저장량/처리량`으로 조건부 추정할 수 있으나 prefix의 작은 N과 SAM2 horizon cache·가림·후기 시점을 전체 Episode 비용으로 실측처럼 외삽하면 안 된다. 예: RGB 두 camera만 98,304 bytes/observation(128×128×3×2); raw depth두개65,536 bytes/observation; 기록된 float depth/calibration/robot/audit와 압축비를 추가해야 한다. Sequence loading은 ragged N/edge 수에 좌우되며 위 Forward4frame 외 대규모 throughput/CPU-GPU memory는 NOT_TESTED다. 장기수집·학습을 실행하지 않았다.

## 11. 검증·미실행·보존

새 lifecycle/abort rejection tests, raw readback의 corruption negative tests와 기존 integration regression을 실행했다. 로그와 [verification](../../experiments/{run}/verification.json)에서 실제 결과를 확인한다. 공통 Logger/Publisher 소스를 변경하지 않았고 ExperimentLogger 및 register_existing_output의 로컬 manifest 등록만 재사용했다. W&B disabled. CUDA/MPS/다른OS, 과거60개 재실행/hole-shift, Full Dataset, 장기 BC/HPO/Closed-loop Policy rollout, external upload/commit/production default 변경은 NOT_EXECUTED다. 현재 CPU physics와 sapien_cpu host readback만 실행했다(렌더러 Vulkan/Metal device는 Apple GPU일 수 있음).

최종 **31 tests PASS**(새8개, 기존23개). 첫 test 실행은 TestCase의 `run` 이름을 Path attribute로 가린 테스트 코드 오류로 중단됐고 [원본 실패 로그](../../experiments/{run}/test_initial_failure.log)를 보존했다. `run_dir`로 고친 뒤 재실행했다. 이는 Recorder/Expert 실패와 별개다. 실제 Causal Mask 실행은 설치 SAM2 `_C` optional extension 부재로 hole filling을 건너뛴 warning이 있으며 로그에 보존했다. FakeTracker suffix-invariance 단위 테스트와 실제 DINO/SAM2 실행 결과를 구분한다.

최초 실패와 수정 후 실행 로그, 재현명령, Source snapshot/hash·implementation diff, PNG/JSON/NPZ/H5를 보존했다. 정적 HTML은 CDN 없이 생성하고 링크·JSON·NPZ·PNG를 실제 readback한다. 기존 source/artifact 보호 manifest는 작업 시작SHA와 비교하며 Task03–06A를 덮어쓰지 않는다.

PNG는 직접 열어 검토·수정했다. CUA에서 Browser 연결은 없었고 native Chrome 접근도 `Computer Use permissions are not granted`로 거부되어 실제 Browser layout 검증은 NOT_EXECUTED다. 권한 우회나 추가 설정 변경을 하지 않았다.

## 12. 질문별 답

'''+ '\n\n'.join(f'**{a} — {b}.** {c}' for a,b,c in q)+f'''

## 13. 재현

새 collection은 새 UUID를 만들고 기존 결과를 덮어쓰지 않는다. [명령](../../experiments/{run}/reproduction_commands.txt)을 사용한다. 사전 보호 manifest는 이번 세션 전용이므로 다른 작업환경에서는 먼저 그 환경의 시작SHA를 생성한다. 보고서/reader만 재실행하는 것은 Simulator rollout을 추가하지 않는다. **이번 Pilot 종료 후 중단하며 Full Dataset 생성과 BC 학습은 사용자 별도 승인 대상이다.**
'''
    DOC.write_text(body)
    artifact_links=' · '.join(f'<a href="{p}">{html.escape(p)}</a>' for p in ['pilot_manifest.json','expert_action_schema.json','robot_state_schema.json','alignment_coverage.json','vision_coverage.json','sequence_forward.json','data_pipeline_costs.json','verification.json','test_log.txt','reproduction_commands.txt','implementation.diff','rgbd_robot_sample.npz'])
    h=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task06-B {run}</title><style>body{{font:16px/1.65 system-ui,sans-serif;margin:36px auto;padding:0 24px;max-width:1160px;background:#f6f8fc;color:#172a43}}h1{{font-size:32px}}h2{{margin-top:40px}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #dce3ed;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde4ee;text-align:left;overflow-wrap:anywhere}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f8;padding:16px}}a{{color:#2162ab}}.flow{{padding:20px;background:#e9f4f0;font-family:monospace;font-size:17px}}</style><h1>Task06-B · Direct ScriptedExpert Pilot</h1><p>UUID {run} · <a href="../../docs/visual-graph-integration/06b_recurrent_expert_collection_pilot.md">Markdown 보고서</a></p><section><h2>실제 결과</h2><p>수정 후 {success}/{len(m['episodes'])} 성공 · ≤100 step {within}/{len(m['episodes'])} · {n} actions / {obs} observations · {aligned} aligned targets</p>{table(['Seed','Success','First success','Terminated','Truncated','Actions','Observations','Failure','Native Box exceed'],rows)}<p>최초 5회는 각 첫 action 뒤 Recorder 오류로 중단. 사용자 승인 후 같은 seed 재실행. 원본 기록은 <a href="../{ABORTED.name}/pilot_manifest.json">별도 UUID</a>에 보존. 과거60/60 재현이 아니다.</p><img src="plots/episode_timeline.png"></section><section><h2>단계별 Gate</h2>{table(['Stage','Status'],stages)}</section><section><h2>Action7 · 실제 적용 명령</h2><p>native float32[7] → exact env.step → exact CombinedController. Translation normalized ±1→±0.1m/root. Rotation norm clipping then -0.1 scaling → XYZ Euler root left-composition. Gripper -1→-0.01m/+1→0.04m. Axis-angle를 Euler와 같다고 해석하지 않는다. Target 변환0.</p>{table(['Seed','Minimum7','Maximum7'],[(e['seed'],e['action_min'],e['action_max']) for e in m['episodes']])}</section><section><h2>같은 시점 RGB-D / Robot</h2><p>두 camera128×128 RGB uint8, raw depth int16 mm(negative/overflow 보존), CV intrinsic/extrinsic 및 inverse cam2world. 마지막 Observation에 Action을 붙이지 않는다.</p><img src="plots/rgbd_samples.png"><div class="flow">o[t] RGB-D + robot76 + pads → expert a[t] → actual controller → env.step → o[t+1]</div><p>Pre-action state/geometry exact, elapsed_steps t, returned observation와 independent live camera read exact, 독립 H5 ↔ NPZ all-dataset byte readback PASS. Timestamp=t/20Hz.</p>{table(['Field','Offset','Dim','Units','Frame'],[(f['name'],f['offset'],f['dimension'],f['units'],f['frame']) for f in schema['fields']])}</section><section><h2>Vision Graph 및 Sequence</h2><p>Prefix packet {len(graphs)}/{vision.get('frames_requested',0)}; both objects nonempty {both}; 전체미처리 {obs-vision.get('frames_requested',0)}. GT object/hole 정보는 Student 입력에 없다. 기존 causal DINO/SAM2 → Shape-First0.5 → PCA5/F5 → same-time EE → Raw EDE. GNN L2/d64/k4/M16, readout5d, robot76, GRU144×1, Regression7.</p>'''
    if (out/'plots/predicted_mask_raw_ede.png').exists():h+='<img src="plots/predicted_mask_raw_ede.png">'
    h+=table(['Episode','Valid RGB-D','Missing robot','Vision valid','Vision not executed'],[(e['seed'],e['observations'],0 if e.get('alignment') else 'unknown',sum(r['episode_id']==e['seed'] for r in [dict(episode_id=vision.get('episode_id')) for _ in graphs]),e['observations']-(len(graphs) if vision.get('episode_id')==e['seed'] else 0)) for e in m['episodes']])
    h+=f'<pre>{html.escape(json.dumps(forward,ensure_ascii=False,indent=2))}</pre></section><section><h2>실측 비용</h2><pre>{html.escape(json.dumps(costs,ensure_ascii=False,indent=2))}</pre><p>Full dataset/후기 mask throughput/GPU peak는 측정하지 않았다.</p></section><section><h2>질문별 결론</h2>'+table(['Question','Status','Answer'],q)+f'</section><section><h2>Artifacts / Reproduction</h2><p>{artifact_links}</p><p><a href="../../docs/visual-graph-integration/06b_recurrent_expert_collection_pilot.md">Markdown 보고서로 돌아가기</a></p></section></html>'
    (out/'recurrent_expert_collection_pilot.html').write_text(h)
    write_json(out/'summary.json',dict(run_id=run,coverage=coverage,stages=stages,questions=q,costs=costs))


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href'):self.links.append(v)


def verify(out):
    out=Path(out);protected=read(out/'before_snapshot/protected.json')
    if not (out/'verification.json').exists():
        write_json(out/'verification.json',dict(status='PENDING'))
    changed=[p for p,h in protected.items() if not Path(p).is_file() or sha256(p)!=h]
    errors=[];counts=dict(json=0,npz=0,png=0,h5=0,links=0)
    from PIL import Image
    import h5py
    for p in out.rglob('*'):
        if not p.is_file() or p.name.endswith('.tracking.json'):continue
        try:
            if p.suffix=='.json':read(p);counts['json']+=1
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as a:
                    for k in a.files:a[k]
                counts['npz']+=1
            elif p.suffix=='.png':
                with Image.open(p) as im:im.load()
                counts['png']+=1
            elif p.suffix=='.h5':
                with h5py.File(p,'r') as f:
                    def load(k,v):
                        if isinstance(v,h5py.Dataset):v[()]
                    f.visititems(load)
                counts['h5']+=1
        except Exception as exc:errors.append(dict(path=str(p),type=type(exc).__name__))
    targets=[out/'recurrent_expert_collection_pilot.html',DOC]
    for p in targets:
        if p.suffix=='.html':
            parser=Links();parser.feed(p.read_text());links=parser.links
        else:links=re.findall(r'\]\(([^)]+)\)',p.read_text())
        for link in links:
            if link.startswith(('http:','https:','#')):continue
            target=p.parent/link.split('#')[0];counts['links']+=1
            if not target.exists():errors.append(dict(path=str(p),missing=link))
    source=out/'final_source_snapshot';source.mkdir(exist_ok=True)
    files=[*Path('graph/integration_baseline').rglob('*.py'),*Path('visual/runtime').rglob('*.py'),
           *Path('recurrent').rglob('*.py'),*Path('envs/maniskill').glob('*.py'),Path('graph/vision_input.py')]
    hashes={}
    diffs=[]
    for p in files:
        target=source/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(p)]=sha256(p)
        if p.name in ('collect_recurrent_expert.py','validate_direct_expert.py','report_direct_expert.py','test_direct_expert.py'):
            diffs.append(subprocess.run(['git','diff','--no-index','--','/dev/null',str(p)],capture_output=True).stdout.decode())
    write_json(out/'final_source_hashes.json',hashes)
    (out/'implementation.diff').write_text('\n'.join(diffs))
    value=dict(status='PASS' if not errors and not changed else 'FAIL',counts=counts,errors=errors,
               protected_files=len(protected),preexisting_changed=changed,
               browser_layout='NOT_EXECUTED: no browser connection; native Chrome Computer Use permissions not granted',source_snapshot='final_source_snapshot',
               scope='local artifact/protection/readback only; not Expert population success or BC performance')
    write_json(out/'verification.json',value)
    assert read(out/'verification.json') == value
    for p in out.rglob('*'):
        if p.is_file() and not p.name.endswith('.tracking.json') and not any(x in p.parts for x in ('before_snapshot','source_snapshot','final_source_snapshot','installed_source')) and p.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
            register_existing_output(out,p,kind='trajectory' if p.suffix in ('.npz','.h5') else 'metrics' if p.suffix=='.json' else 'reports')
    print(json.dumps(value,ensure_ascii=False),flush=True)
    assert value['status']=='PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.verify:verify(args.run)
    else:report(args.run)

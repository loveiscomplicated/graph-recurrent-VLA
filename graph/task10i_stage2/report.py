"""Static measured figures and raw/annotated evidence gallery."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

from .core import PROTOCOL,require_frozen
from graph.initial_pool10i.mirror.verify import read

CSS='body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 24px;color:#203040;background:#f8fafc}table{border-collapse:collapse;background:white}td,th{padding:8px 12px;border:1px solid #ccd5df}img.plot{max-width:100%} .cards{display:flex;flex-wrap:wrap;gap:12px}.card{width:340px;padding:12px;background:white;border:1px solid #ccd5df;border-radius:8px} .card img{width:150px;image-rendering:pixelated}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1459a6}h2{margin-top:40px}'


def table(rows,keys):
    def fmt(v):return f'{v:.6g}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows)


def generate(out,byid,states,sets,split,summary,runid):
    require_frozen(out);out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    cand=[r for r in byid.values() if r['kind']=='new'];old=sets['existing33']
    figure_names=[]
    def save(name,title):
        plt.title(title);plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=150);plt.close();figure_names.append((name,title))
    def xy(name,title,key,axes=(0,1)):
        plt.figure(figsize=(7,5))
        for label,rs,color,marker in [('Candidate300',cand,'#c1c8ce','.'),('Existing33',old,'#171d29','x'),
                ('R',sets['R'],'#3978b4','o'),('D',sets['D'],'#df7a20','+')]:
            a=np.array([r['descriptor'][key] for r in rs]);plt.scatter(a[:,axes[0]],a[:,axes[1]],label=label,c=color,marker=marker,s=24,alpha=.65)
        plt.xlabel(key+' '+str(axes[0]));plt.ylabel(key+' '+str(axes[1]));plt.legend(fontsize=8);save(name,title)
    xy('peg_xy','Peg world XY (m)','peg_xyz')
    plt.figure(figsize=(7,4))
    for label,rs in [('Candidate300',cand),('Existing33',old),('R',sets['R']),('D',sets['D'])]:
        plt.hist([r['descriptor']['peg_yaw_rad'] for r in rs],bins=np.linspace(-np.pi,np.pi,25),density=True,histtype='step',label=label)
    plt.legend();plt.xlabel('Peg yaw rad, periodic [-pi,pi]');save('peg_yaw','Peg orientation distribution')
    xy('peg_shape','Peg full length/thickness (m)','g4')
    xy('peg_hole','Peg minus hole world XY (m)','g3p')
    xy('tcp_peg','Peg minus TCP world XY (m)','g1')
    xy('main_center','Mean OBB-corner center (image fraction)','g5')
    for cohort,key,name in [('selection','selection','coverage_selection'),('holdout','holdout','coverage_holdout')]:
        rows=summary[key];plt.figure(figsize=(7,4));x=np.arange(len(rows));w=.22
        for j,metric in enumerate(['mean','p90','p95']):plt.bar(x+(j-1)*w,[r[metric] for r in rows],w,label=metric)
        plt.xticks(x,[r['dataset'] for r in rows]);plt.ylabel('Frozen distance');plt.legend();save(name,cohort+' nearest-set coverage')
    with (out/'nearest_set_distances.csv').open() as f:nearest=list(csv.DictReader(f))
    plt.figure(figsize=(7,4))
    for cohort in ['selection','holdout']:
        for name in ['R','D']:
            a=np.sort([float(r['distance']) for r in nearest if r['cohort']==cohort and r['dataset']==name])
            plt.plot(a,np.arange(1,len(a)+1)/len(a),label=f'{cohort} {name}')
    plt.xlabel('Nearest-set distance');plt.ylabel('ECDF');plt.legend();save('nearest_ecdf','Nearest distance distributions')
    with (out/'random_monte_carlo_metrics.csv').open() as f:mc=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,cohort in zip(axs,['selection','holdout']):
        ax.hist([float(r['mean']) for r in mc if r['cohort']==cohort],bins=22,color='#b6c6d7')
        for name,color in [('R','#3978b4'),('D','#df7a20')]:ax.axvline(summary['monte_carlo'][cohort]['datasets'][name]['score'],color=color,label=name)
        ax.set_title(cohort);ax.set_xlabel('Mean nearest distance');ax.legend()
    save('random_mc','Conditional Random19 Monte Carlo (200 repeats)')
    plt.figure(figsize=(7,4))
    for name,rs in [('Candidate300',cand),('R',sets['R']),('D',sets['D'])]:
        a=np.sort([r['descriptor']['boundary_margin_px'] for r in rs]);plt.step(a,np.arange(1,len(a)+1)/len(a),label=name)
    plt.xlabel('Boundary margin px');plt.ylabel('ECDF');plt.legend();save('boundary_margin','Boundary margin distribution')
    sr={r['id'] for r in sets['R']};sd={r['id'] for r in sets['D']}
    plt.figure(figsize=(7,5))
    for label,ids,color in [('Shared',sr&sd,'#64748b'),('R-exclusive',sr-sd,'#3978b4'),('D-exclusive',sd-sr,'#df7a20')]:
        a=np.array([byid[i]['descriptor']['peg_xyz'] for i in sorted(ids)])
        if len(a):plt.scatter(a[:,0],a[:,1],label=f'{label} ({len(a)})',color=color,alpha=.8)
    plt.xlabel('Peg X m');plt.ylabel('Peg Y m');plt.legend();save('selection_overlap','Selected parents: shared and exclusive')
    gallery(out,byid,states,sets,split)
    s=summary;o=s['overlap'];sel={r['dataset']:r for r in s['selection']};held={r['dataset']:r for r in s['holdout']}
    gain=lambda rows:100*(rows['R']['mean']-rows['D']['mean'])/rows['R']['mean']
    existing_gain=lambda rows,name:100*(rows['existing33']['mean']-rows[name]['mean'])/rows['existing33']['mean']
    boundary={r['population']:r for r in s['boundary']}
    plan=read(out/'stage3_demonstration_plan.json')
    preserved=read(out/'existing_episode_preservation_manifest.json')
    text=f'''# Task10-I Stage 2 — Diversity-aware Initial State Dataset Selection

상태: **COMPLETED**. ExperimentLogger UUID `{runid}`. 본 결과는 **선정 계획**이며 신규 Expert action/trajectory가 있는 학습 데이터셋이 아니다. Stage 3는 자동 실행하지 않았다.

Engineering 이력: 첫 `/workspace/task10i_stage2_20261031/run`은 분포 summary의 `maximum` 출력 키 중복 오류로 BLOCKED를 보존했다. 수정은 분포 CSV 출력에만 적용했고 seed/weight/scale/selection은 변경하지 않았다. 회귀 테스트에 분포 출력 계약을 추가한 뒤 별도 Source/Output Root에서 전 단계를 재검증했다. 이전 선정 ID/순서와의 동등성은 별도 완료 감사로 확인한다.

## 1. 입력과 Source Identity — CONFIRMED

Vast.ai instance **57987772**에서 CPU NumPy/SciPy로 실행했다. Canonical `/workspace/task10i_stage1_20261031/verified_pool_v2`는 읽기 전용이다. Stage 1.5 `handoff/stage2_input_receipt.json`과 그 Artifact Index 전체를 검증했고, **8,514개 전체 파일**을 원본 path/size/SHA256과 다시 대조했다. Stage 1의 8,506 indexed files와 8개의 기존 bookkeeping 파일을 구분했다. Missing/Mismatch/Extra/Unreadable은 모두 0이다.

926행 aggregate와 300개 eligible initial NPZ, Descriptor, RGB-D, Visibility evidence를 실제로 읽고 Stage 1.5 fingerprint와 대조했다. 기존 Primary Gate, 1.5mm depth / 0.0001px frustum tolerance, Physical/Replayed PASS는 변경하지 않았다. 기존 JSON의 역사적 macOS 절대경로는 provenance이며 새 로더는 실제 remote pool 상대경로로 읽는다.

입력 [Identity](stage2_input_identity.json), [Receipt 검증](stage2_input_receipt_verified.json), [300개 Readback](input_candidate_readback.json), [Source 보존](source_preservation_audit.json), [실행 Source](execution_source_identity.json).

## 2. 기존 Train33 / Validation7 — CONFIRMED

Task10-G final release manifest, 실제 host metadata, dataset identity, 승인 Eligibility IDs, 320개 원본 HDF5 및 GT sidecar SHA를 대조했다. episode→parent seed→split→branch를 실제 자료로 연결했다. 기존 감사 NPZ에는 Hole GT가 없으므로 원본 GT sidecar의 `raw_hole_pose`, `raw_hole_radius`에서 복원했다. Peg/TCP/RGB-D/calibration을 저장 첫 프레임과 exact 대조하고, Parent 안 모든 branch의 초기 Peg/Hole/TCP/qpos/qvel/camera/image를 exact 비교했다. Box actor pose는 기존 raw GT에 없으며 추정하지 않았다. 거리 계산에는 실제 Hole frame을 사용한다.

Train 33개는 **82 logical episodes**, Validation 7개는 **23 logical episodes**다. 원본 272/48 중 190/25 visibility-failed episodes는 Stage 3 계획의 제거 inventory로만 기록했고 원본은 수정하지 않았다. Existing Final/Test는 사용하지 않았다. [Parent inventory](existing_valid_parent_inventory.json), [보존/제거 episode inventory](existing_episode_preservation_manifest.json), [원본 SHA 목록](existing_source_file_inventory.json).

## 3. 신규300 분할과 무작위 추출 — CONFIRMED

정렬된 ID에 PCG64 seed `{PROTOCOL['seeds']['split']}`를 적용한 uniform permutation으로 Validation Cohort20 / Diagnostic Holdout60 / Training220을 먼저 동결했다. Validation 앞9는 Primary, 뒤11는 Reserve다. Validation20 전체와 Holdout60은 Train/scale/greedy에서 제외했다. Source snapshot은 그대로 UNASSIGNED이며 Stage2 manifest에만 새 역할을 부여했다.

Training220에서 Common44와 잔여 Random19를 독립 uniform permutation으로 비복원 추출했다. 최종 신규63의 분포는 uniform subset63과 동등하다: 각 63-subset을 만드는 44/19 분할 수가 동일하고 각 순서의 확률도 같다. 가장 좋은 Monte Carlo 결과로 R을 바꾸지 않았다. [분할/전체 trace](candidate_pool_split_manifest.json), [RNG](random_seed_manifest.json), [추출 trace](sampling_trace.json).

## 4. Feature와 거리 계약 — CONFIRMED

GT는 Dataset construction/분석 전용이다. Student 입력·모델은 바꾸지 않았다. 모든 pose는 world xyz,wxyz, m; quaternion 정규화 후 sign-invariant SO(3) angle `2 acos(abs(dot(q1,q2)))`을 사용한다. 180° reversal은 π 거리이며 Red/White 끝을 합치지 않는다. Yaw는 [-π,π] 주기 진단이며 objective에 중복 추가하지 않는다.

| Group | 정의 | Weight |
| --- | --- | ---: |
| G1 | world peg−TCP XYZ / coordinate scale의 RMS | .30 |
| G2 | Peg world SO(3) geodesic / angular scale | .20 |
| G3 | 평균(position RMS, hole⁻¹·peg SO(3) / scale) | .25 |
| G4 | Full length, square thickness / scale의 RMS | .15 |
| G5 | 8개 OBB corner 평균 u,v /128 / scale의 RMS | .10 |

최종 거리 `d=Σ weight_g·distance_g`. Group 차원 수로 가중치가 커지지 않도록 vector RMS를 쓴다. G3 안 position/orientation은 각 1/2이다. 원점 이동에 무관한 world displacement이며 TCP local frame으로 혼합하지 않는다. G5는 mask centroid/actor center와 다른 명시적 OBB-corner 평균이다.

Scale은 기존 Train33+Train220 **253개만** fit한다. Vector는 coordinate IQR, rotation은 253개 내 upper-triangle pair-angle IQR이다. Floor는 위치1cm, length1cm, thickness2mm, angle10°, image4px이다. Nearly constant axis에도 이 floor를 사용한다. 실제 clearance가 thickness/2+3mm임을 모든 state에서 검사했으므로 hole radius를 독립 가중하지 않는다. scalar TCP distance·Yaw·image extent·boundary margin도 중복 또는 품질 진단으로만 둔다. 이것은 사전 정의 heuristic이며 검증된 최적 weight가 아니다.

[Schema](feature_schema.json), [Scale 실제 값과 fit IDs](feature_normalization.json), [Distance](distance_protocol_frozen.json), [Selection](selection_protocol_frozen.json). Holdout positive-control 차단·q/−q·wraparound·방향 반전·degenerate floor·miniature E2E는 [테스트 로그](contract_tests.log)를 따른다.

## 5. R/D/Validation 선정 — CONFIRMED

R=기존33+Common44+Random19=**96**. D=동일 기존33+동일 Common44+Facility-location19=**96**. Shared Validation=기존7+Random9=**16**. D는 Training220의 `J(S)=mean_x min_s d(x,s)` 감소량을 매번 최대화했다. Pure farthest-point는 사용하지 않았다. Float64와 ID 정렬 tie-break로 전체 greedy continuation을 독립 두 번 실행하여 exact 일치했다.

실제 R/D 교집합 **{o['shared']}**, R-exclusive **{o['R_exclusive']}**, D-exclusive **{o['D_exclusive']}**. 추가19 중 **{len(o['tail_overlap'])}개**가 겹친다. R tail을 D 후보에서 배제하지 않았다. 신규 Expert 수집 고유 Parent union은 Validation9 포함 **{o['new_collection_union']}개**다(상한91).

[R manifest](dataset_R_parent_manifest.json), [D manifest](dataset_D_parent_manifest.json), [동일 Validation](shared_validation_parent_manifest.json), [Hash freeze](selection_identity.json), [전체 greedy 순서](coverage_selection_order.json), [공통/차이 감사](selection_overlap_audit.json).

## 6. Coverage 분석

### Training Candidate220

{table(s['selection'],['dataset','n','mean','median','p90','p95','maximum'])}

기존33 대비 mean 감소는 R **{existing_gain(sel,'R'):.2f}%**, D **{existing_gain(sel,'D'):.2f}%**다. D의 R 대비 mean 감소는 **{gain(sel):.2f}%**다. Greedy 자체 objective에서의 개선은 독립 일반화 증거가 아니다. p90/p95/max는 표의 실제 값으로 별도 해석한다.

### 독립 Diagnostic Holdout60

{table(s['holdout'],['dataset','n','mean','median','p90','p95','maximum'])}

Holdout에서 D의 R 대비 mean 감소는 **{gain(held):.2f}%**다. 판정은 **{'SUPPORTED' if gain(held)>0 else 'INCONCLUSIVE'}**: 이 60개 초기 상태의 근접 Coverage에 한정한다. Tail 개선의 유무는 mean과 별개다. Holdout은 최종 Policy blind test가 아니며 어떤 결과도 seed/weight/algorithm 재선정에 사용하지 않았다.

[Training metrics](random_vs_diversity_coverage.csv), [Holdout metrics](coverage_holdout_metrics.csv), [Group별 nearest 및 total-nearest에 대한 group 값](feature_group_coverage.csv), [개별 nearest parent](nearest_set_distances.csv).

## 7. Conditional Random Monte Carlo 200회

Common44를 고정하고 잔여 후보에서 Random19를 200회 추출했다. 200개 독립 학습 실험이 아니다. Percentile은 random mean scores 중 해당 score 이하의 비율이며 낮을수록 더 나은 coverage다.

{table([dict(cohort=k,**{kk:vv for kk,vv in v.items() if kk!='datasets'},R_percentile=v['datasets']['R']['percentile_lower_is_better'],D_percentile=v['datasets']['D']['percentile_lower_is_better']) for k,v in s['monte_carlo'].items()],['cohort','mean','median','variance','p5','p95','R_percentile','D_percentile'])}

[200회 모든 metrics](random_monte_carlo_metrics.csv), [sample IDs trace](random_monte_carlo_sampling_trace.json). 동결 R을 유지했다.

## 8. 분포 왜곡·경계·극단 상태 감사

Candidate300은 native random reset을 고정 Main visibility/physical/replay 조건으로 제한한 경험 분포다. R/D는 기존33을 포함하므로 300 신규만의 분포와 완전히 같을 필요는 없다. R/D 전체와 각 추가19, Common44를 따로 비교했다. Peg XYZ, periodic yaw/SO3 pair angle, length/thickness/radius, peg−hole, peg−TCP, image center/extent, margin을 기술 통계·KS statistic·Wasserstein distance로 보존했다. 중복 pair/부분집합의 의존성을 감안해 유의성 p-value를 주장하지 않는다.

{table([boundary[n] for n in ['candidate300','existing33','R_tail19','D_tail19','R','D']],['population','n','minimum','mean','median','fraction_below_0.1px','fraction_below_1px','fraction_below_4px','fraction_below_8px'])}

Training253의 Peg XYZ/size p5/p95 밖인 축이 하나라도 있는 Parent를 극단 위치·크기 diagnostic으로 미리 정의했다. Candidate300 **{s['extreme']['candidate300']['fraction']:.1%}**, R tail19 **{s['extreme']['R_tail19']['fraction']:.1%}**, D tail19 **{s['extreme']['D_tail19']['fraction']:.1%}**, R 전체 **{s['extreme']['R']['fraction']:.1%}**, D 전체 **{s['extreme']['D']['fraction']:.1%}**다. D tail만과 전체를 혼동하지 않는다. Margin은 objective에 포함하지 않았다.

Coverage 개선만으로 ‘분포 왜곡 없음’을 확정하지 않는다. **INCONCLUSIVE**: 이 감사는 측정한 주변 분포·경계·SO3 통계 범위의 집중도를 보여주며, 통계적 동등성이나 고차원 분포 보존을 입증하지 않는다. 분포 차이가 정책 성능을 악화하는지는 **NOT_TESTED**다. [분포 summary](initial_distribution_summary.csv), [shift](distribution_shift_metrics.csv), [boundary](boundary_margin_comparison.csv), [extremes](extreme_distribution_audit.json).

## 9. 누출·중복 — CONFIRMED

Train/Validation exact ID overlap0, Train/Holdout0, Validation/Holdout0. 기존 split 재할당0, Final/Test 사용0, candidate ID/seed 충돌0. 신규 vs 기존33/7, R vs D, 각각 Train vs Validation/Holdout의 Peg/Hole/TCP pose·shape·qpos/qvel을 실제 비교했다. R/D 공통 Parent의 exact overlap은 의도된 것이다. Near 기준은 위치1mm, orientation1°, half-size/radius0.1mm, robot qpos0.001이다. 서로 다른 Split의 동일 초기 물리 상태는 차단한다. Near만으로 GT 누출을 단정하거나 후보를 제거하지 않는다.

{table([dict(comparison=k,**v) for k,v in s['duplicates'].items()],['comparison','pairs','exact_same_parent','exact_distinct_parent','near_distinct_parent'])}

[실제 거리·차원 CSV](near_duplicate_audit.csv), [duplicate summary](physical_duplicate_summary.json). CSV는 exact/near 전체와 각 left의 가장 가까운 physical threshold-ratio 상대를 기록한다. 전체 pair 수를 독립 표본 수로 해석하지 않는다.

## 10. Reserve와 Stage 3 Handoff

Validation11, Common176, R157, D157의 순서를 동결했다. Common/R은 별도 seed의 uniform order, D는 같은 greedy trajectory의 20번째 이후 continuation이다. Reserve는 역할 간 겹칠 수 있으며 초기 선정이나 미래 성공을 의미하지 않는다. Validation은 자기 Cohort만 사용한다. Common은 현재 R/D union에 없는 ID를 함께 대체한다. R/D tail은 자기 dataset에 없는 상대 dataset ID를 공유할 수 있다. 실패 Parent는 모든 queue에서 제외하고 skip 사유를 기록한다. 충돌 처리 순서는 Validation→Common→R→D, 각 initial slot rank/ID 순이다. 원래 실패 상태를 삭제하지 않고 모든 attempted/failed/replacement/final ID와 coverage 변화를 새 버전으로 추적한다. Exhaustion은 차단하며 임의 seed를 보충하지 않는다. 현재 실제 대체는 **0회**다.

[reserve protocol](stage3_collection_reserve_protocol.json), [Validation11](validation_reserve.csv), [Common](common_reserve.csv), [R](random_reserve.csv), [D](coverage_reserve.csv), [고유 수집 union](stage3_parent_collection_union.json), [Stage3 input receipt](stage3_input_receipt.json).

## 11. Logical Episode 구성과 Stage 3 검증 과제

{table([dict(split=k,**v) for k,v in plan['structure'].items()],['split','original_logical','preserved_logical','new_parent_count','logical_deficit','target_logical'])}

기존 full-task/corrective source 및 branch별 비율과 모든 실패 parent의 실제 branch inventory는 [수집 계획](stage3_demonstration_plan.json)에 있다. 목표272/48과 preserved82/23의 차이는190/25 logical episodes다. 새 Parent63/9를 선정한 사실만으로 그 수가 복원되지 않는다. 실패한 기존 Train63/Validation9 parent를 ID 정렬하여 신규 slot에 대응한 **Parent별 branch 수·유형 template**도 저장했다. Common44는 같은 앞44 template, R/D tail은 같은 뒤19 template를 각자 계승한다. Tail overlap이 서로 다른 template를 받으면 수집 union과 dataset별 포함을 별도 기록한다. 이 계획의 logical deficit는 각각190/190/25이며 성공한 episode가 있다는 뜻은 아니다. 모든 신규 Parent의 Full-task 성공 확인이 선행되어야 한다.

Collector에서는 prefix/disturbance supervision=false, recovery=true다. Task10G BC는 승인 nominal의 actions/robot/timestep과 exact 일치한 prefix만 재활성화하고 disturbance는 항상 false다. Stage3는 이 원본 계약을 확인해야 한다. Parent마다 동일한 임의 branch 수를 만들지 않았다.

Stage3는 실제 Expert 성공, Full-task/Corrective lineage, copied-prefix·supervision mask·action/RGB-D/GT clock 정합, reset(seed)+snapshot readback, 실패 attrition과 실제 collection 후 coverage를 검증해야 한다. 고정33/7이 무효가 되는 경우 silent replacement하지 않고 차단한다. 새 Final Evaluation은 이300과 기존 Final/Test에 없는 별도 seed가 필요하다.

## 12. 연구 질문과 해석 범위

- Q1/Q2: 기존33과 신규300의 실제 위치·방향·크기·상대관계 범위는 [분포 summary](initial_distribution_summary.csv)와 아래 그림에 있다. 신규 후보가 기존 nearest distance를 줄일 수 있는 미대표 영역을 제공함은 measured selection/holdout baseline로 확인한다.
- Q3/Q4: 고정 R/D의 기존33 대비 변화와 독립 Holdout 차이는 6절의 수치 범위에서만 해석한다.
- Q5/Q6: 분포 동일성은 INCONCLUSIVE다. 극단/작은 margin 비율과 orientation 통계를 별도 제시했고 margin 최적화는 하지 않았다.
- Q7: ID·seed·physical state·split 감사 범위에서 계획 identity는 CONFIRMED다.
- Q8: 고정 reserve와 충돌/실패 기록 규칙은 CONFIRMED다. 실제 대체 후 성공/coverage는 NOT_TESTED다.

**NOT_TESTED:** 실제 Expert 성공률·action 품질·정책 성공률·성능상 악영향. Simulator reset/action, GPU PhysX/CUDA/MPS inference, DINO/SAM2, Graph recache, GNN/GRU 학습·closed-loop 평가는 실행하지 않았다. Linux CPU 저장자료 분석만 수행했고 로컬 simulator도 실행하지 않았다. 기존 Task10-H source/process/checkpoint를 수정하지 않았다. 공통 ExperimentLogger를 disabled transport로 재사용했고 외부 publishing·Git commit·새 인스턴스 생성은 하지 않았다.

## 13. 그림과 이미지 근거

[전체 Gallery](gallery.html)는 원본 Main RGB와 별도로 그린 Peg OBB overlay, Descriptor, Visibility evidence, original snapshot/source를 연결한다. Stage1 annotated preview를 raw로 사용하지 않았다.

'''
    for name,title in figure_names:text+=f'### {title}\n\n![{title}](figures/{name}.png)\n\n'
    text+='## 14. 최종 검증\n\n[Artifact Index](artifact_index.json), [Final Verification](final_verification.json), [Stage3 Receipt](stage3_input_receipt.json). 선정·Config hash를 다시 읽고 모든 JSON/CSV/PNG와 SHA/size를 검증했다. 단계별 atomic proof와 progress를 보존했다. 완료된 실행은 재선정 없이 readback만 수행한다. 미완료 resume는 동일 source/protocol에서 deterministic stage를 재검증하며 frozen 결과가 달라지면 차단한다.\n'
    (out/'10i_stage2_diversity_aware_dataset_selection.md').write_text(text)
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        body='<pre>'+html.escape(text)+'</pre>'
    body=body.replace('<img ','<img class="plot" ')
    (out/'10i_stage2_diversity_aware_dataset_selection.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task10-I Stage2</title><style>'+CSS+'</style>'+body+'</html>')


def gallery(out,byid,states,sets,split):
    asset=out/'gallery';asset.mkdir(exist_ok=True)
    for cid,r in byid.items():
        raw=Image.fromarray(states[cid]['rgb']);raw.save(asset/(cid+'_raw.png'))
        im=raw.copy();draw=ImageDraw.Draw(im);uv=np.array(r['descriptor']['corners_uv'])
        for i in range(8):
            for j in range(i+1,8):
                if (i^j) in (1,2,4):draw.line([tuple(uv[i]),tuple(uv[j])],fill=(0,255,80),width=1)
        im.save(asset/(cid+'_obb.png'))
    sr={r['id'] for r in sets['R']};sd={r['id'] for r in sets['D']}
    common={r['parent_id'] for r in read_csv(out/'common_random44.csv')}
    categories=[('Existing Train33',[r['id'] for r in sets['existing33']]),
        ('Existing Validation7',[i for i,r in byid.items() if r['role']=='existing_validation']),
        ('Common Random44',sorted(common)),('R-exclusive',sorted(sr-sd)),('D-exclusive',sorted(sd-sr)),
        ('Shared R/D tail overlap',sorted((sr&sd)-common-{r['id'] for r in sets['existing33']})),
        ('New Validation9',split['validation'][:9]),('Reserve: validation',split['validation'][9:]),
        ('Reserve: training candidates',sorted(set(split['training'])-(sr|sd)))]
    body='<h1>Task10-I Stage2 evidence gallery</h1><p>左/Left: original raw RGB. Right: new OBB annotation. Both 128×128; no Stage1 annotated preview is used as raw. Reserve role queues are linked in the report. Expert NOT_TESTED for every new parent.</p><p><a href="10i_stage2_diversity_aware_dataset_selection.html">Report</a></p>'
    for label,ids in categories:
        body+=f'<h2>{html.escape(label)} ({len(ids)})</h2><div class="cards">'
        for cid in ids:
            r=byid[cid];s=r['source'];d=r['descriptor']
            if r['kind']=='new':
                e=s['export'];refs=f'<a href="{s["pool"]}/{e["initial_npz"]}">Original RGB-D NPZ</a> · <a href="{s["pool"]}/{e["gt_visibility_evidence"]}">Visibility evidence</a> · <a href="{s["pool"]}/{e["snapshot"]}">Snapshot (not standalone restore)</a>'
            else:refs=f'<a href="{s["first_observations"]}">Original first observations NPZ</a> · <a href="{s["visibility_evidence"]}">Visibility evidence</a>'
            body+=f'<article class="card"><b>{cid} / seed {r["seed"]}</b><div><img src="gallery/{cid}_raw.png" alt="raw RGB"><img src="gallery/{cid}_obb.png" alt="OBB overlay"></div><p>Visibility PASS; margin {d["boundary_margin_px"]:.4f}px</p>{refs}<details><summary>Spatial descriptors / source identity</summary><pre>{html.escape(json.dumps(r,indent=2))}</pre></details></article>'
        body+='</div>'
    (out/'gallery.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage2 Parent Gallery</title><style>'+CSS+'</style>'+body+'</html>')


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

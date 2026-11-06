"""Regenerate the measured report; unexecuted experiments stay N/A."""
import json
from pathlib import Path


def main():
    root=Path('outputs/graph/v3');a=json.loads((root/'audit/results.json').read_text())
    comparison=root/'comparison/results.json'
    c=json.loads(comparison.read_text()) if comparison.exists() else {'models':{}}
    text=['''# Graph Prototype v3 — Surface Interaction Graph 검증

이번 작업은 Graph 팀 내부의 독립적인 설계 후보 검증이다. v1/v2, shared
simulator/GT accessor, Visual/Recurrent 코드는 변경하지 않았다. 코드 변경 전에
실제 episode replay로 수집한 근거는 `outputs/graph/v3/audit/phase_a_evidence.json`,
`phase_b_evidence.json`에 보존했다. 기존 시각화는
`outputs/graph/v2/figures/goals_{initial,acquired,transport,alignment}.png`에 있었다.
동일 episode 29 / prefix 0,56,72,77을 live CPU simulator에서 재생했다.

## A. 기존 Graph 진단 결과

### 실제 입력과 MPNN

- Node 131개: TCP 1, 좌/우 rubber-pad collision 중심 2, peg 표면 64,
  hollow box의 노출된 collision 표면 64. Pad 중심은 실제 접촉 표면점과 다르다.
- Node feature 14차원: type one-hot 5, world-axis TCP-relative 위치 3,
  object-local 위치 3, world normal 3. Hole local 좌표는 offset hole frame이
  아니라 **box frame**이다. Robot local/normal은 0이다.
- Context 81차원: gripper/qpos, EE/root 위치·rotation, peg/box/hole 상대 위치와
  rotation, 크기/radius/offset. v2는 동일한 qvel 및 현재 goal geometry 31차원을
  action head에 추가한다. 미래 action/waypoint/phase는 입력하지 않는다.
- Local edge: 각 물체에서 k=8 Euclidean kNN + 각 점의 가장 가까운 earlier-index
  point를 잇는 spanning tree. 거리 cutoff나 surface geodesic 제약은 없다.
- Finger–peg: pad마다 peg 전체에서 nearest k=4. 거리/기능 영역 제약 없음.
- Peg–hole: **peg의 모든 점에서 hole의 nearest k=4**, 거리 제약 없이 항상 존재.
- 모든 relation에 reverse edge 생성 후 `(src,dst,type)`별 coalesce. Self-loop 없음.
- Edge feature 8차원: `position[dst]-position[src]` 3, 그 norm(m) 1,
  relation one-hot 4. Reverse edge의 상대 벡터 부호도 올바르다.
- 3개 residual message layer. 하나의 공유 MLP가 `(h_src,h_dst,edge_attr)`를 입력받고,
  destination에 index_add 후 **전체 incoming edge의 평균**을 사용한다. Type은
  one-hot으로 모델에 전달되지만 relation별 MLP/평균/가중치 분리는 없다.
- Readout은 TCP/left/right embedding을 개별 concat하고 peg/hole 각각 mean pooling,
  context를 concat한다. Robot node를 전체 131개 node 평균에 섞어 희석하지 않는다.
- Train split으로만 node/context/shared state를 표준화한다. Type one-hot은 그대로,
  node 좌표/local/normal은 모든 train node를 합친 통계로 표준화한다. Edge 상대
  벡터·길이는 raw metre이고 type도 그대로다. Normalization 오류는 확인되지 않았다.
- World transform은 wxyz rotation + translation이다. Hole에는 box pose를 적용하며,
  normal에는 translation을 적용하지 않는다. Geometry consistency를 검사했으며,
  정책 자체의 SE(3) equivariance를 주장하지 않는다.

### 실제 directed edge 통계

아래 count에는 양방향이 포함된다. 네 상태에서 surface sampling 및 local edge
topology는 동일하다. Mean/min/max는 metre이다.

| State / prefix | Relation | Count | Mean | Min | Max |
|---|---|---:|---:|---:|---:|''']
    for stage,s in a['states'].items():
        for name,e in s['variants']['original']['graph']['edges'].items():
            text.append(f'| {stage} / {s["prefix"]} | {name} | {e["count"]} | {e["mean_m"]:.6f} | {e["min_m"]:.6f} | {e["max_m"]:.6f} |')
    text.append('''
Local relation을 물체별로 나누면 episode 29에서 peg–peg 618개
(mean 0.024087, min 0.001555, max 0.210053m), hole–hole 654개
(mean 0.067705, min 0.011550, max 0.210660m)이다.
이 긴 intra-object 연결 일부는 연결성을 보장하는 earlier-index tree에서 발생한다.
따라서 기존의 “local” edge가 항상 물리적 이웃이나 surface-adjacent 관계는 아니다.
이번 B 비교에서는 이 local topology를 유지해 cross relation 변경과 분리했다.

| State | Degree min / max | Incoming cross 비율 최대(hole node) | Peg–Hole >10cm | Connected components |
|---|---:|---:|---:|---:|''')
    for stage,s in a['states'].items():
        d=s['variants']['original']['graph'];degrees=d['incoming_degree'][0]
        text.append(f'| {stage} | {min(degrees)} / {max(degrees)} | {d["hole_cross_fraction_max"]:.1%} | {d["cross_over_10cm"]} / 512 | {len(d["components"][0])} |')
    text.append('''
전체 relation 중 512/1804=28.4%가 peg–hole이다. Initial hole 일부 node에서는
incoming 평균의 88.7%를 cross relation이 차지한다. 이것은 **message 수에 따른
혼합 비중**이지 학습된 message norm/attention/정보 기여도의 측정은 아니다.
정확한 node별 incoming/outgoing degree와 node type별 총합, 중복·self-loop·component
검사는 `outputs/graph/v3/audit/results.json`에 있다.

### 시각화와 실제 edge 대조

기존 `v2/visualize_goals.py`의 두 번째 panel은 실제 `GraphBuilder().edge_index`를
읽는다. 첫 번째 panel의 검정 TCP→grasp 및 teal head→hole 선은 별도 goal overlay다.
두 번째 panel의 다수 긴 teal 선은 **실제 PEG_HOLE message-passing edge**다.
그러므로 overlay와 혼동한 시각화만의 문제는 아니었다.

새 시각화는 `edge_rows(graph, selected)`에서만 actual edge를 읽는다. Solid segment는
reverse pair를 한 번 표시하고, 옆 `.edges.json`에는 directed edge를 전부 export한다.
Type별 색/CLI 필터/개수/endpoint 확대를 제공한다. Dashed goal overlay와 insertion
axis arrow는 message edge가 아니라는 label을 붙였다. 두 카메라의 각 축에는 동일한
metre 범위와 1:1:1 aspect를 적용했다. 모든 PNG를 생성하고 대표 그림을 직접 확인했다.

## B. 발견한 문제와 geometry 검증

**확인된 설계 불일치**: 원거리 목표와 가까운 물리적 interaction이 동일 PEG_HOLE
type으로 처리된다. 모든 peg point의 kNN이 멀리서 hole 일부 node에 집중된다.
Head↔entry/inner 및 fingertip↔grasp 영역이 graph construction에 명시되지 않는다.
기존 graph는 shape+generic proximity/global relation을 표현하지만, 의도한 기능별
robot–surface interaction을 충분히 구분하지 않는다.

**확인되지 않은 가설**: 위 edge가 local geometry를 실제로 희석해 action/rollout을
악화시켰다는 인과 관계, 입구 point 3개의 보편적인 부족, 특정 GNN 구조의 우월성.
Incoming count만으로 이 가설들을 확정하지 않는다.

**확인된 geometry 오류는 없음**: live peg collision mesh 1개와 hollow box collision
primitive 4개의 bounds를 adapter와 비교했고, Panda pad collision mesh 중심도
비교했다. Episode 29 네 pose와 별도 randomized seed 0,1,2를 검증했다.
Primitive union의 노출 경계에 대해 normal 방향 안쪽은 solid, 바깥쪽은 empty라는
검사 및 hole opening 보존, local/world transform 검사를 통과했다. Randomized
크기·radius·hole offset이 반영된다. 원본 peg는 물리적으로 하나의 직육면체이며
head/tail은 +X/-X 기능 구분이다. Visual 색으로 나뉜 두 부분을 별도 collision shape로
오인하지 않았다. Live 검증은 CPU 단일 환경에 한정된다.

Sampling은 면적 균등이 아니다. Peg 6개 / hole 16개 rectangle patch를 round-robin으로
커버하며 patch마다 UV를 고정 seed로 생성한다. 기존 peg cap +X(head) 11,
-X(tail) 11, side 42개이고 tail-half side 22개다. Hole은 outer 16, inner 16,
negative-X entry face 16, positive-X exit face 16개다. Entry face 전체가 “rim 바로
주변”은 아니다. Aperture에서 2cm 이내인 entry point는 episode 29에서 3개였다.
2cm는 분석/샘플링 band이며 물리적 collision clearance가 아니다.

표면 boundary 검사는 추가 24개 dataset pose에서도 통과했다. 전체 collision mesh
topology를 모든 dataset row마다 재추출한 것은 아니다. Contact point/force나 실제
surface clearance를 입력하거나 검증하지 않았다.

## C. 독립적인 수정 후보

### B: 동일 geometry + interaction-aware connectivity

131개 node, node feature 14, context 81, v2 shared state 31 및 기존 readout/action
head/BC를 유지한다. Object/region one-hot node feature는 추가하지 않는다. Region은
edge eligibility와 시각화 metadata로 사용한다. 이 제한으로 geometry 정보 증가와
node 수 증가를 첫 비교에서 피했다.

- Peg head region: local X ≥ half_length/2(leading quarter 및 cap).
- Grasp region: local X < 0(tail-half side/cap). 실제 contact/grasp label이 아니다.
- Hole: entry(-X end face), inner wall, outer wall, exit(+X end face).
- Local geometry relation은 기존 kNN/tree를 그대로 보존한다.
- Fingertip↔tail-half region 및 head↔entry/inner에 nearest cross_k=4 후보를 만들고
  point distance ≤0.05m인 edge만 유지한다. Interaction radius와 별도 threshold의
  최소값으로 cutoff를 정할 수 있다. Surface normals는 기존 node feature에 남지만
  normal alignment/clearance 조건은 아직 추가하지 않는다.
- GLOBAL_GOAL type을 추가해 2개 pair(4 directed edge)를 항상 유지한다:
  TCP↔tail-cap 중심에 가장 가까운 기존 surface point,
  head-cap 대표점↔entry aperture 중심에 가장 가까운 entry surface point.
  대표점은 local geometry에서 선택하므로 common rigid transform에 따라 바뀌지 않는다.
- 이는 **소수 surface 대표점을 통한 global proxy**다. 새로운 object-pose node나
  정확한 target-pose node를 구현한 완전한 hierarchical graph는 아니다. 실제
  grasp/target insertion geometry는 기존 context/shared state에 계속 포함된다.
- Feature는 9차원(dst-src xyz, distance, relation one-hot 5). Shared message MLP의
  input만 1차원 늘어난다. Parameter 증가는 384개다. Relation-specific aggregation은
  변경하지 않아 여전히 incoming 평균으로 혼합한다.

Radius 밖에서 interaction이 0개여도 local spanning tree와 global proxy로 graph는
연결된다. 네 지정 상태에서는 head↔hole local interaction이 모두 0개다. Dataset
snapshot의 episode 29 timestep 85에서는 126개, 90에서는 134개, 125에서는 168개가
활성화된다. `alignment` phase의 시작(t=77)은 최소 cross point 거리가 11.1cm이므로
5cm interaction이 없는 것이 정상이다. Phase label을 contact truth로 해석하면 안 된다.

### C: B + region sampling 후보 (성능 미검증)

Peg sampling은 유지한다. Hole outer/inner/entry/exit weights 1:3:3:1로 64개를 배분해
10/22/22/10개를 사용한다. 각 entry patch의 broad point 하나씩은 남기고 나머지는
aperture 주변 2cm band에 배치한다. 모든 점은 기존 exposed collision face 위에 있다.
Episode 29에서 aperture-near entry point가 3→20개가 된다. 이 수치가 성능 향상을
의미하지는 않는다. 전체 표면 coverage와 task-relevant 영역 사이의 별도 후보다.

Sampling 변경 효과를 graph 구조 효과로 오인하지 않도록 C와 같은 node/context
정보를 flat vector로 받는 `flat_sampling` 실행 경로도 제공한다. C/Flat-C는 각각
2 optimizer step, validation 실행 및 checkpoint 복원 smoke만 수행했다. Test set
action loss/rollout은 실행하지 않았다. C/Flat-C 학습 성능은 N/A다.

| State | A original edge | B interaction edge | C sampling edge | B/C components |
|---|---:|---:|---:|---:|''')
    for stage,s in a['states'].items():
        ds=s['variants']
        text.append(f'| {stage} | {ds["original"]["graph"]["edges_total"]} | {ds["interaction"]["graph"]["edges_total"]} | {ds["sampling"]["graph"]["edges_total"]} | 1 / 1 |')
    text.append('''
Config/CLI에서 local k, cross k, interaction radius, interaction distance threshold,
sampling region weight 및 entry band를 조절할 수 있다. Audit에는 peg/hole point
count 옵션도 있다. Paired full training은 원래 64+64 dataset 설정을 검사한다.
다른 count로 학습하려면 해당 count로 dataset과 config를 새로 준비해야 한다.
Test 성능을 본 threshold/region/seed 재선택은 하지 않았다.

## D. 구조적 검증 및 실험 결과

### CPU 비용

20회 측정의 median. Graph construction과 model forward를 별도로 측정했다.
Network timing에는 encoder, 동일 action head와 shared-state concat이 포함된다.
Normalization/GT extraction/샘플링 생성 및 simulator step은 제외한다. Benchmark는
dataset의 최초 1/32개 row를 사용하므로 episode 29 edge 수와 다를 수 있다.
최종 benchmark는 학습과 rollout 종료 후 다시 측정했다. Host 부하와 warmup 등에
영향을 받는 CPU latency이므로 절대 성능 보장은 아니다.

| Variant | Parameters | Batch | Edges | Graph ms | Network ms | Graph tensors bytes |
|---|---:|---:|---:|---:|---:|---:|''')
    for variant,bench in a['benchmarks'].items():
        for batch,b in bench['batches'].items():
            text.append(f'| {variant} | {bench["parameters"]} | {batch} | {b["edges"]} | {b["construction"]["median_ms"]:.3f} | {b["inference"]["median_ms"]:.3f} | {b["graph_tensor_bytes"]} |')
    text.append(f'\nAudit process peak RSS: {a["process_peak_rss_bytes"]:,} bytes. Dataset/plots/models를 포함한 process peak이며, 모델 단독 peak memory 차이가 아니다.\n')
    text.append('''
B는 batch별 Python loop를 사용하고 원본 builder에서 local edge를 재사용하므로
graph 생성이 느리다. Edge 감소가 end-to-end 속도 향상을 보장하지 않는다.
Candidate construction은 아직 최적화하지 않았으며 graph tensor memory와
parameter 수를 개별적으로 보고한다.

### 동일 budget BC와 held-out 평가

91 expert episodes、73/9/9 episode split、action representation、v2 goal bypass、
159(embedding 128+shared state 31)→128→7 head, seed 0, AdamW lr=0.001 /
weight_decay=1e-5, cosine schedule, batch 32, 30 epoch, 10,590 optimizer steps를
공통 조건으로 사용한다. A는 저장된 v2 checkpoint를 재사용하고 B를 새로 학습했다.
Validation 최소 loss의 checkpoint를 선택하며 test는 설정 변경에 이용하지 않는다.
Parameter 차원이 바뀐 message MLP의 초기 weight는 A와 완전히 같지 않다.
단일 seed 비교의 한계가 있다.

Translation/rotation error는 normalized command 성분의 RMSE이며 실현된 TCP pose
오차가 아니다. Physical command RMSE도 JSON에 기록한다. Closed loop는 같은 CPU
backend / original reset state / converted expert prefix / horizon 250으로 비교한다.

| Metric | Original Surface A | Interaction Surface B | Sampling C / Flat-C |
|---|---:|---:|---:|''')
    models=c['models']
    for label,key in [('Action MSE','action_mse'),('Translation command RMSE (normalized)','translation_rmse'),('Rotation command RMSE (normalized)','rotation_rmse'),('Translation physical command RMSE (m)','translation_command_rmse_m'),('Rotation physical command component RMSE (rad)','rotation_command_euler_rmse_rad')]:
        values=[f'{models[v]["offline"][key]:.8f}' if v in models else 'N/A' for v in ('original','interaction')]
        text.append(f'| {label} | {values[0]} | {values[1]} | N/A |')
    for label,stage in [('Full rollout SR','full'),('Alignment-start SR','alignment')]:
        values=[]
        for variant in ('original','interaction'):
            result=models.get(variant,{}).get('settings',{}).get(stage)
            values.append(f'{result["successes"]}/{result["denominator"]}' if result else 'N/A')
        text.append(f'| {label} | {values[0]} | {values[1]} | N/A |')
    for variant in ('original','interaction'):
        if variant in models: text.append(f'\n{variant} validation-selected checkpoint epoch: {models[variant]["checkpoint_epoch"]}.')
    if all(v in models for v in ('original','interaction')):
        old,new=models['original']['offline']['action_mse'],models['interaction']['offline']['action_mse']
        text.append(f'\nB의 held-out Action MSE는 A 대비 {(new/old-1)*100:+.1f}% 변화했다. 단일 seed의 offline 결과이며 graph 구조가 closed-loop 성공을 개선한다는 결론과 구분한다.')
        if all(stage in models['interaction']['settings'] for stage in ('full','alignment')):
            text.append('\n이번 B 후보의 성능 향상은 확인되지 않았다. Full rollout은 양쪽 모두 0/9이며 alignment-start는 A 1/9, B 0/9였다. 구조적 관계 구분이 더 명시적이라는 결과와 성능 향상이 없다는 결과를 함께 유지한다. 이 작은 차이로 B가 본질적으로 열등하다고 단정하지 않는다.')
    train=root/'experiments/interaction/training.json'
    if train.exists():
        t=json.loads(train.read_text());text.append(f'\nB training: {t["training_seconds"]:.1f}s, {t["optimizer_steps"]} optimizer steps, best validation MSE {t["best_validation_mse"]:.8f}.')
    text.append('''
B는 원거리 cross-surface 집중과 기능 영역의 미구분을 완화하고 global goal 경로를
유지한다. Surface normal 기반 contact 추정이나 정확한 clearance 표현을 완성한 것은
아니다. Action loss 개선과 closed-loop 성공을 구분한다. 9 episode의 alignment
성공 횟수만으로 구조의 우열을 단정할 수 없고 full-task 실패를 graph만의 문제로
결론 내리지 않는다. C의 성능 효과는 미평가다.

### Tests와 보존 확인

Root simulator/GT 8, v1 graph/data/model 12, v2 16, v3 6으로 총 42 tests를 실행했다.
Shape/index/relative vector/normal과 rigid pose 변환, collision boundary/opening,
region/radius 조건, global-only connectivity, single-vs-batch edge 일치, determinism,
forward/backward 전체 parameter의 finite gradient, checkpoint round-trip, actual
filtered edge와 PNG 부속 JSON의 완전 일치를 확인했다. C/Flat-C smoke checkpoint도
실제로 다시 로드해 train split 입력으로 forward를 검사했다. 추가 24 pose에서
component=1, 중복 edge=0, self-loop=0이었다. Feature/edge는 미래 action이나 phase를
사용하지 않는다.
생성한 48개 PNG의 directed edge export도 저장된 실제 tensor와 48/48 일치했다.

첫 test의 float32 rigid-transform 오차에 대한 exact zero 비교가 지나치게 엄격해
geometry 허용 오차 1e-6m로 고친 후 재실행했다. 기존 test의 relative import를
유지하기 위해 package module 이름으로 실행했다. Geometry 구현 수정이 필요한
실패는 아니었다.
''')
    preservation=a.get('preservation',{})
    text.append(f'\n기존 checkpoint/result/dataset 파일 {len(preservation)}개, SHA256 일치 {sum(preservation.values())}/{len(preservation)}.\n')
    text.append('''
실행 host: macOS Apple M4 Pro, Python 3.11 / ManiSkill 3.0.1 / SAPIEN 3.0.3 /
Torch 2.14.1. Simulation은 PhysX CPU, renderer는 SAPIEN CPU/MoltenVK, Torch는 CPU다.
CUDA는 불가하며 MPS training/inference, CUDA/vectorized simulation은 미실행(N/A).
API/primitive 판단에는 installed task source와 live collision meshes를 사용했다.

## E. 연구 방향에 대한 답변

1. **기존 graph는 interaction을 일부만 명시했다.** Robot/object type과 relative
   geometry는 있으나 grasp/head/entry/inner 관계가 미구분이고 원거리 goal과 근거리
   interaction을 같은 dense relation으로 처리했다.
2. **Surface representation을 연구 후보로 유지할 근거는 있다.** Collision surface,
   opening, normal을 올바르게 표현하고 근거리 region별 관계를 만들 수 있다.
   Center/Flat보다 action 성능이 우수하다는 근거와는 구분해야 한다.
3. **기능 영역은 construction/진단에서 구분할 가치가 있다.** B는 node feature를
   늘리지 않고 eligibility에 region을 사용했다. Task-specific tail-half/head-quarter
   정의의 일반화와 region feature의 학습 효과는 검증하지 않았다.
4. **Global과 local interaction 분리는 의미적으로 타당한 검증 후보다.** 원거리
   목표를 제거하지 않고 근거리 surface 연결을 제한할 수 있다. 이번 대표점 proxy는
   근사이며 true object-pose node가 필요한지는 prototype 근거에 따라 결정한다.
5. **다음에는 controller/BC의 closed-loop 실패 요인을 먼저 분리하길 권한다.** B의
   구조 검증과 action 평가를 기록한 후 GNN/controller를 동시에 바꾸지 않고 검증한다.
   Relation별 aggregation이나 normal 조건을 늘리기 전에 expert 이후 오차 누적,
   grasp 유지, action distribution 원인을 독립적으로 조사한다. Surface 고유 성능
   주장에는 matched Flat, 복수 seed, 충분한 rollout이 필요하다.

## 재현 명령·성과물·미완료

```bash
.venv/bin/python -m graph.prototype.v3.graph_audit --figures
.venv/bin/python -m graph.prototype.v3.train --output outputs/graph/v3/experiments/interaction_new --epochs 30
.venv/bin/python -m graph.prototype.v3.compare --checkpoint outputs/graph/v3/experiments/interaction_new/best.pt --rollouts --output outputs/graph/v3/comparison/new.json
.venv/bin/python -m unittest discover -v
.venv/bin/python -m unittest graph.prototype.tests.test_geometry graph.prototype.tests.test_data graph.prototype.tests.test_models graph.prototype.v2.tests.test_v2 graph.prototype.v3.tests.test_v3 -v
.venv/bin/python -m graph.prototype.v3.visualization --state acquired --variant interaction --edge-types 2 --zoom --output outputs/graph/v3/visualizations/finger_zoom.png
.venv/bin/python -m graph.prototype.v3.visualization --state initial --variant original --edge-types 3 --output outputs/graph/v3/visualizations/original_cross.png
.venv/bin/python -m graph.prototype.v3.train --variant sampling --output outputs/graph/v3/experiments/sampling_new
.venv/bin/python -m graph.prototype.v3.train --variant flat_sampling --output outputs/graph/v3/experiments/flat_sampling_new
.venv/bin/python -m graph.prototype.v3.report
```

Full training은 기존 checkpoint를 덮어쓰지 않는다. `new` 출력 이름은 변경 가능하다.
Report generator는 이번 `comparison/results.json`을 읽고 미실행 칸은 N/A로 작성한다.
재실험할 때는 report가 읽는 비교 결과 경로도 확인해야 한다.

- 코드: `graph/prototype/v3/{graph_audit,geometry_audit,edge_diagnostics,interaction_graph,visualization,train,compare,report}.py`, README, tests.
- 수치/geometry snapshot: `outputs/graph/v3/audit/`.
- 4 state × 3 variant × 4 mode × 2 camera의 48 PNG, edge JSON, edge-length/degree histogram: `outputs/graph/v3/visualizations/`.
- Full B checkpoint/history: `outputs/graph/v3/experiments/interaction/`.
- Offline 지표 및 per-step rollout trace: `outputs/graph/v3/comparison/results.json`.
- 미완료: C와 matched Flat-C의 full-budget 학습/rollout, 복수 seed, true object-pose
  node 비교, normal alignment/실제 clearance, relation별 aggregation, contact truth
  평가, GPU backend/vectorization. 이들의 성능은 N/A이며 우위를 주장하지 않는다.

[기존 graph의 actual cross edge](../outputs/graph/v3/visualizations/initial_original_interaction.png) ·
[B의 global relation과 diagnostic goal overlay](../outputs/graph/v3/visualizations/initial_interaction_global.png) ·
[C의 collision과 region points](../outputs/graph/v3/visualizations/alignment_sampling_geometry.png) ·
[hole opening 정면 비교](../outputs/graph/v3/visualizations/hole_aperture.png) ·
[edge histogram](../outputs/graph/v3/visualizations/original_edge_lengths.png) ·
[degree histogram](../outputs/graph/v3/visualizations/degree_histogram.png)
''')
    Path('docs/graph_prototype_v3.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__': main()

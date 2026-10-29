# Task10-B 추가 진단 — 실제 Geometry 개입과 짧은 feedback 분기

기존 Phase1/2의 Expert-state 오차 확인은 원인 분해의 완료가 아니다. 이 추가 진단은 그 공백을 메우기 위해 기존09-A/09-B의 물리 개입·causal memory·response metric과08-F의 짧은 분기 구조를 현재 E0/E1에 연결했다. 모델/Geometry/Codec을 변경하거나 재학습하지 않았다. 기존 보고서와 원자료는 보존했다. 최종 연구 판단은 `causal_findings.md`를 따른다.

## 실험 identity와 범위

- Parent comparison: `3788798a719f4ec4a66c224ca42f5d63`.
- 이전 Phase1/2 UUID: `35877452d6d94d9691964fc8a8490858`.
- Reset/Prefix guard UUID: `b659dc3c41494ea89abeee09ca421c59`.
- Geometry collection UUID: `bf2ab695001a487494a732727ece1be8`.
- Geometry analysis UUID: `8a7c11f84a8a43c7b6c67783db71c096`.
- Branch driver UUID (k25 guard로 failed; k0 원자료 보존): `4647218956e044dbb90852f1a1361f5c`.

새 결과는 `/workspace/task10b_extension_20261029`에만 저장했다. frozen source는 별도의 기존 Task10-B source copy를 재사용했고 원본 체크포인트·Graph cache·rollout·Task10-A source/process는 수정하지 않았다. 신규 진단은 Train30003만 사용하며 Test12를 seed/threshold/가설/모델 선택에 사용하지 않았다. Task10-A 관리 상태/복구 코드 확인 중 읽힌 Test reset audit는 진단 설정이나 결과 계산에 사용하지 않았다. 새 Test cache/prediction/rollout은 열지 않았다.

## Reset와 동일 상태 유효성

Train30003의 reset hash는 `27fd426300899a6a44b55aa04aab72e373cee7e61881762fd03d467d274d9f62`로 원본과 정확히 일치했다. Pose 복원이나 hash guard 우회는 없었다. 원본 Expert action prefix를 두 번 독립 reset에서49 actions까지 재생한 qpos/qvel/TCP/물체 pose/velocity 등이 모두 정확히 일치했다.

원래 HDF5와는 Robot76 최대 0.00010915613, TCP pose 최대 6.3627958e-06의 차이가 있다. 따라서 원본 Expert 입력과 byte-identical하다고 주장하지 않는다. 인과 개입은 **동일 current-runtime physical state 안**에서 로봇·카메라·previous hidden·causal vision history를 고정한 비교다. 원본 상태 근접성과 paired-state 동일성을 별개로 기록했다.

## Geometry 개입

시점은 사전 고정한 t=0,10,25,40,48이다. 각 시점 baseline, Peg world X/Y ±5/10mm, Box X+10mm negative control, baseline repeat를 수집했다. 총55조건이며 물체를 teleport한 뒤 physics step 없이 새 RGB-D를 렌더했다. SAT 정적 collision/near-contact 검사이며 teleport 후 cached PhysX contact를 새 contact force로 해석하지 않았다. 동적 안정성 실험은 아니다. 또한 다른 물체·Robot76·보정의 exact 검사를 통과한 물리 조건은 55/55다.

실제 causal RGB prefix를 sequential tracker로 처리하고, current frame 직전의 tracker memory를09-B `memory/restore`로 복원했다. 네 Best/Last policy의 hidden은 각자의 같은 history에서 얻은 값을 고정했다. Student 입력은 원래 whitelist의 RGB-D·예측 mask·Robot76뿐이다. GT pose는 개입 정의/Expert reference/평가, segmentation ID는 source purity 평가에서만 사용했다. 저장 NPZ 전체를 sensor input으로 넘기려던 최초 시도는 whitelist가 차단했으며, sensor-only key adapter로 수정한 뒤 계산했다. whitelist/geometry forward는 바꾸지 않았다.

Baseline repeat에서는 mask·Graph·Readout·GRU input/hidden·Native Action을 exact 비교했다. 각 baseline의 원래 forward와 adapter도 action/hidden exact, forward repeat exact였다. 모든 frame의 current source timestamp를 검사했고 미래 frame이나 Expert Action은 Student input에 넣지 않았다.

## Expert counterfactual 검증

09-A `expert_reference`를 그대로 사용해 같은 ScriptedExpert phase/counter와 새 OBB target으로 다시 계산했다. 목표 translation 이동량·방향/자세와 독립 servo 계산이 기준을 만족한 경우에만 유효 reference로 사용했다. 이전 원본 action을 임의의 perturbation 정답으로 넣지 않았다. 원래 HDF5 action과의 작은 차이는 별도 수치로 남기며, 동일 current-runtime reference의 유효성과 혼동하지 않는다.

| t | Phase | baseline target 재계산 차 m | baseline reference | 유효 물리 조건 |
|---|---|---:|---|---:|
| 0 | reach | 0 | VALID_STAGE_CONDITIONED | 11/11 |
| 10 | reach | 1.058e-06 | VALID_STAGE_CONDITIONED | 11/11 |
| 25 | reach | 1.729e-06 | VALID_STAGE_CONDITIONED | 11/11 |
| 40 | descend | 3.144e-06 | INVALID_EXPERT_REFERENCE | 11/11 |
| 48 | descend | 3.874e-06 | INVALID_EXPERT_REFERENCE | 11/11 |


후반 reference가 무효면 direction/gain 유효 집계에서 제외하되 원자료와 사유를 보존했다. Threshold를 완화하거나 target을 임의로 교정하지 않았다. Reference Δ≤0.1mm는09-A의 NON_DISCRIMINATIVE_REFERENCE 기준이다. 과거09-A 입력 purity≥0.5, 이동방향으로 centroid response≥1mm 기준을 C3의 양 camera source provenance에 적용했다. Point-centroid 변화는 전체 물체 pose 복원의 정확도와 같지 않으며 sampling/visibility 영향도 포함한다.

## Geometry → Readout → Action 결과

유효성 분류(모델 중복 전)는 {'VISIBLE_VALID': 46, 'GEOMETRY_UNSUPPORTED_RESPONSE': 4}이다. `GEOMETRY_UNSUPPORTED_RESPONSE`는 source purity는 통과했지만 기준 centroid 반응이 부족한 경우로 따로 남겼다. 아래는 physical·visible/purity/centroid·Expert discriminative reference를 모두 통과한 Peg 개입만 기술적으로 집계했다. 실패 조건과 box control/repeat도 CSV/JSON에는 남아 있다. 같은 시점의 ±offset, 같은 graph를 쓰는4모델을 독립 Episode로 취급하지 않는다.

| 모델 | t | 유효 Peg 개입 | Readout 상대L2 | Expert Δ mm | Model Δ mm | 방향 cosine | Signed gain | Zero보다 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_last | 0 | 7 | 0.1742 | 0.396 | 0.4041 | 0.2476 | 0.6803 | 2/7 |
| E0_last | 10 | 7 | 0.1327 | 0.5703 | 0.1931 | 0.1389 | -0.1308 | 4/7 |
| E0_last | 25 | 8 | 0.09204 | 1.249 | 0.1501 | 0.3612 | 0.02891 | 5/8 |
| E0_last | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_last | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_validation_best | 0 | 7 | 0.1747 | 0.396 | 0.4028 | 0.2477 | 0.6776 | 2/7 |
| E0_validation_best | 10 | 7 | 0.1331 | 0.5703 | 0.1916 | 0.1508 | -0.1284 | 4/7 |
| E0_validation_best | 25 | 8 | 0.09202 | 1.249 | 0.1505 | 0.364 | 0.02913 | 5/8 |
| E0_validation_best | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_validation_best | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_last | 0 | 7 | 0.04892 | 0.396 | 0.1443 | 0.3298 | 0.3302 | 3/7 |
| E1_last | 10 | 7 | 0.03584 | 0.5703 | 0.06382 | 0.2122 | -0.002062 | 4/7 |
| E1_last | 25 | 8 | 0.0357 | 1.249 | 0.06782 | 0.3741 | 0.03141 | 7/8 |
| E1_last | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_last | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_validation_best | 0 | 7 | 0.04895 | 0.396 | 0.145 | 0.3228 | 0.3332 | 3/7 |
| E1_validation_best | 10 | 7 | 0.03587 | 0.5703 | 0.06406 | 0.2147 | -0.001276 | 4/7 |
| E1_validation_best | 25 | 8 | 0.03572 | 1.249 | 0.06758 | 0.3733 | 0.03117 | 7/8 |
| E1_validation_best | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_validation_best | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |


Readout 변화만으로 유용한 정보 전달을 입증하지 않는다. Raw Geometry가 유효하게 변하고 Readout/GRU가 변해도 Action 방향·gain·zero-response 대비 오차를 함께 봐야 한다. 이 실험은 encoder/GRU/head 전체의 반응을 측정하며 개별 모듈의 독립 원인 기여를 분리하지 않는다. Frozen probe를 학습하지 않았으므로 정보 소멸 여부는 확정할 수 없다. E0/E1은245,447/194,759 parameters로 구조와 용량이 달라 message passing 단독 인과 효과를 주장할 수 없다.

## 별도 검증한 Current-OBB Reference

기존09-A 기준의 후반 무효 판정은 그대로 유지했다. 추가로 current live state에서 fresh OBB target을 만든 baseline과 perturbation을 같은 phase/counter로 비교하는 별도 reference를 사전 정의·검증했다. 원래 demonstration action과 같은 reference라고 주장하지 않는다. Threshold를 완화하지 않았으며 아래 결과는 기존 primary 표와 섞지 않았다. Model/Graph/Action 파일은 그대로 읽어 새 reference와만 비교했다. source physical state는 원래 수집 before/after와 정확히 일치했다.

| 모델 | t | 유효 개입 | Expert Δ mm | Model Δ mm | 방향 cosine | Signed gain | Zero보다 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0_last | 0 | 7 | 0.396 | 0.4041 | 0.2476 | 0.6803 | 2/7 |
| E0_last | 10 | 7 | 0.5703 | 0.1931 | 0.1389 | -0.1308 | 4/7 |
| E0_last | 25 | 8 | 1.249 | 0.1501 | 0.3612 | 0.02891 | 5/8 |
| E0_last | 40 | 7 | 2.206 | 0.0423 | 0.1376 | -0.0009844 | 4/7 |
| E0_last | 48 | 7 | 10.79 | 0.186 | 0.2916 | 0.002254 | 4/7 |
| E0_validation_best | 0 | 7 | 0.396 | 0.4028 | 0.2477 | 0.6776 | 2/7 |
| E0_validation_best | 10 | 7 | 0.5703 | 0.1916 | 0.1508 | -0.1284 | 4/7 |
| E0_validation_best | 25 | 8 | 1.249 | 0.1505 | 0.364 | 0.02913 | 5/8 |
| E0_validation_best | 40 | 7 | 2.206 | 0.04244 | 0.1518 | -0.0009463 | 4/7 |
| E0_validation_best | 48 | 7 | 10.79 | 0.1846 | 0.2915 | 0.002192 | 4/7 |
| E1_last | 0 | 7 | 0.396 | 0.1443 | 0.3298 | 0.3302 | 3/7 |
| E1_last | 10 | 7 | 0.5703 | 0.06382 | 0.2122 | -0.002062 | 4/7 |
| E1_last | 25 | 8 | 1.249 | 0.06782 | 0.3741 | 0.03141 | 7/8 |
| E1_last | 40 | 7 | 2.206 | 0.01497 | 0.572 | 0.002799 | 6/7 |
| E1_last | 48 | 7 | 10.79 | 0.0581 | 0.3618 | 0.0009949 | 5/7 |
| E1_validation_best | 0 | 7 | 0.396 | 0.145 | 0.3228 | 0.3332 | 3/7 |
| E1_validation_best | 10 | 7 | 0.5703 | 0.06406 | 0.2147 | -0.001276 | 4/7 |
| E1_validation_best | 25 | 8 | 1.249 | 0.06758 | 0.3733 | 0.03117 | 7/8 |
| E1_validation_best | 40 | 7 | 2.206 | 0.01497 | 0.5767 | 0.00278 | 6/7 |
| E1_validation_best | 48 | 7 | 10.79 | 0.05785 | 0.3621 | 0.0009779 | 5/7 |

![Current OBB response](geometry_action_response_fresh_reference.png)

Baseline command 차이는 `reference_definition_comparison.json`, 정의·검증은 `fresh_reference_protocol.json`과 `fresh_expert_reference.json`에 보존했다. 이 reference도 학습 target이나 Test12 fitting에는 쓰지 않았다.


## 10-action feedback 분기

08-F의 첫 행동×suffix 구조를 재사용했다. Train30003, Last E0/E1, k=0에서 E/E·EM·ME·MM 각10 actions(모델별4개, 총8개)을 완료했다. 사전등록한 k=25는 RGB strict input guard가 차단해 policy branch action0개이며 BLOCKED_OBSERVATION_IDENTITY로 남겼다. 기준을 완화하지 않았다. 10 actions는20Hz에서0.5초다. E/E는 원래08-F reference replay처럼 첫 input 검사 뒤 policy 추론 없이 원본 Expert action을 적용한다. E suffix도 새 feedback Expert가 아니라 고정된 원본 Expert 명령이다. 이 제어 조건을 실제 scripted feedback recovery와 혼동하지 않는다.

분기 시작의 reset hash·물리상태·Robot76·RGB-D/보정·current Graph·previous hidden·prediction은 정확히 일치했다. 고정된 합법적 causal prefix의 tracker/hidden을 복원했고, suffix의 실제 새로운 관측에서 hidden을 갱신했다. 임의 hidden 교환·다른 시점 history 이식은 없다. E/E와 EM의 첫 post-state, ME와 MM의 첫 post-state도 정확히 일치했다.

초기 model-versus-expert 행동의 영향은 Model suffix에서 ||MM−EM||, Expert suffix에서 ||ME−EE||로 측정했다. 벡터는 TCP−Peg 상대 위치이며 시점 n은 분기 후 적용 action 수다. n=1 두 값이 같다는 계약을 확인한다. 비율의 분모가 작으면 mm 절대값을 우선 해석한다. EE가 피드백 Expert가 아니므로 “Expert가 보정해주면 회복한다”는 실험과는 다르다.

| 모델 | k | n | Expert suffix impulse mm | Model suffix impulse mm | 비율 | MM–EE mm |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0 | 1 | 2.116 | 2.116 | 1 | 2.116 |
| E0 | 0 | 3 | 2.587 | 5.791 | 2.239 | 4.194 |
| E0 | 0 | 5 | 2.59 | 9.877 | 3.814 | 5.42 |
| E0 | 0 | 10 | 2.583 | 16.86 | 6.528 | 16.73 |
| E1 | 0 | 1 | 2.214 | 2.214 | 1 | 2.214 |
| E1 | 0 | 3 | 2.528 | 3.738 | 1.478 | 4.293 |
| E1 | 0 | 5 | 2.521 | 8.112 | 3.218 | 6.941 |
| E1 | 0 | 10 | 2.514 | 14.93 | 5.939 | 20.87 |


## 실행 환경과 산출물

Linux Python3.12 / Torch2.14.1+cu130 / NumPy2.5.4. CPU PhysX와 SAPIEN host image readback을 실행했다. 이 호스트의 NVIDIA renderer 초기화에는 CUDA device가 보여야 했고 CUDA context를 초기화했다. Policy와 DINO/SAM2 inference는 CPU에서 실행해 Task10-A의 GPU 추론을 점유하지 않았다. CUDA PhysX/MPS는 실행하지 않았다. 설치된 SAM2의 optional `_C` 후처리 누락 warning은 보존했으며 패키지/모델/geometry filter는 변경하지 않았다.

원자료는 physical_collection의 RGB-D/보정/robot/평가 segmentation, geometry_inference의 실제 Graph/예측 mask/source records/Readout/GRU/Action, branches의 시작 input/step trace/physics states/MP4다. `geometry_sensitivity.csv`, `graph_to_action_response.csv`, `geometry_validity_audit.json`, `branch_metrics.csv`, `error_amplification_analysis.json`, `prefix_intervention_analysis.json`과 대조할 수 있다. 소규모 분기 영상은 새 실제 branch에만 생성했고 기존 rollout 영상을 재생성하지 않았다.

![Geometry response](geometry_action_response.png)
![Graph to action](graph_readout_action_response.png)

![Impulse divergence](divergence_curves.png)

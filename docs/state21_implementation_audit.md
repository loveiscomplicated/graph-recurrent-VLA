# Object21D 상태 추정 구현 감사

## 기존 구현과 입력 확인

2026-11-16 KST, `graph` 브랜치 `2b6c40e` 및 작업 디렉터리의 기존 변경을
조사했다. 기존 Vision, TAPNext++, 표면 그래프 실험과 리더보드 수치 의존성은
별도 경로에 있다.

- [4,102 프레임 리더보드](experiment_leaderboard.md#detected-resume-transformer)
  1위는 detector-only 대응 Transformer + 조기 종료 재개 GNN이다. 그래프 본체는
  `graph/gated_mpnn.py`의 `VertexEdgeGatedMPNN`이다. Peg/target별
  `EdgeGatedMPNNLayer`와 별도 cross layer, confidence attention pool,
  두 포즈·크기 헤드, 상대 끝점 residual을 쓴다. Edge geometry는 메시지와
  sigmoid gate 양쪽에 전달한다.
- [리더보드의 체크포인트 표](experiment_leaderboard.md#detected-resume-transformer)는
  seed 0/1/2의 best epoch 48/36/64, 실제 경로와 SHA-256을 기록한다.
  로컬에 표의 best 파일은 없으며 [보존 백업](detected_corner_gnn.md#published-checkpoint-backup)에
  원본이 등록되어 있다. 초기 레시피는 hidden 64, layer 2, edge embedding 16,
  dropout 0, 관측 provenance 추가 feature 1, AdamW lr 0.0003,
  weight decay 0.01, batch 96, 최대 gradient norm 1이다. Fresh 40-epoch
  모델의 optimizer와 RNG에서 재개하며 최대 100 epoch, patience 10을 적용했다.
- [TAPNext++](tapnextpp.md)는 BGR과 초기 픽셀 쿼리로 시작하며 프레임마다
  원본 이미지 xy, visibility 및 다음 tracker state를 반환한다. Surface replay는
  물체 마스크별 트래커 상태를 유지하고, 가시성과 depth/mask로 지지된 쿼리만
  OpenCV extrinsic으로 월드 좌표에 올린다. 쿼리 번호는 해당 point stream의
  인덱스다. `TensorResult`의 corner는 Peg(0)/Box(1) object ID, UV, world와
  camera XYZ를 제공한다. 고정 semantic ID와 검출 confidence는 제공하지 않는다.
- 작업 시작 때 존재하던 `data_collection/surface_graph_frames.py`의 캐시는
  `frame_indices`, `node_offsets`, `uv`, `xyz_world`, `object_type`,
  `is_corner`, `track_id`, `tcp_xyz_world`, `object21`을 보존한다.
  `track_id=-1`인 corner의 배열 인덱스는 semantic ID가 아니다.
  `graph/build_graph/surface_graph.py`는 모든 점을 같은 영상 평면에
  triangulate하며 cross edge를 포함할 수 있다. 리더보드의 기존 코너 그래프는
  complete graph다. 새 모델은 공통 물체별 Delaunay를 사용하고 기존
  메시지 층, pooling과 출력 규약을 재사용한다.
- `graph/utils/object21_targets.py:object21_from_gt`는 world axes와 wxyz를
  유지하면서 TCP-relative 위치와 Peg +X 끝점−Hole 벡터를 만든다.
  `graph/gated_mpnn.py:assemble_state38`은 Robot15, Object21, Freshness2
  순서로 연결한다.
- 원본 파일럿 manifest는 seed, trajectory 경로, 원본 프레임 순서와 해시를
  기록한다. 표면 캐시는 원본 numeric frame index를 저장하며 wall-clock
  timestamp는 없다. 저장된 관측 TCP와 원본 GT의 TCP는 캐시 생성 시 대조된다.
  저장된 표면 실험은 train 3개, validation 1개 에피소드다. 독립 test 및
  확인된 semantic ID가 필요한 평가는 별도 데이터가 필요하다.
- 기존 테스트는 `tests/test_gated_mpnn.py`, `tests/test_detected_corner_gnn*.py`,
  `tests/test_fixed_id_gnn*.py`, `tests/test_leaderboard_reproduction.py`에
  있다. 기존 Mamba-1 PyTorch 구현은 `graph/temporal/insertion.py`,
  causal Transformer block은 `graph/temporal/causal_transformer.py`에 있다.

## 변경 범위

Graph 모델 및 builder, 데이터 adapter와 split 검사, 팀 YAML, 학습 및 평가,
단위·통합 테스트를 추가한다. 가중치 호환성이 달라지는 신규 입력·그래프와
기존 리더보드 체크포인트를 구분한다. 전체 re-training과 기존 CUDA 리더보드
수치 재현은 이번 워크플로의 선행 조건이 아니다.

## 입력과 모델

실데이터 표면점은 기존 [TAPNext++ 표면 replay](tapnextpp.md#프레임-0부터-표면점-추적과-3d-꼭짓점-복원)
로 생성한 캐시를 사용한다. 어댑터는 raw manifest, replay archive, 선택된 원본
frame의 SHA-256을 확인한다. 원본 RGB-D 레코드에 저장된 관측 TCP와 GT에서 복원한
Object21D를 캐시의 같은 원본 프레임 번호와 대조한다. Box 검출 및 외부 표면점의
object ID 1은 Hole 조립체 쪽 관측이다. TAPNext++ query ID는 표면점의 계보를
확인하는 데 사용한다. Semantic ID와 confidence는 이 캐시에서 모두 미확인이다.

다른 TAPNext++ 관측은 `data.kind: observations`와 `data.manifest`로 읽을 수 있다.
JSON 최상위는 `format: state21_observations`, `coordinates: world_m`,
`provenance`, `episodes`다. `provenance`에 `tracker: TAPNext++`,
`tracker_sha256`, `tracker_training_or_selection_parents`를 기록한다.
각 episode에 `episode_id`, `parent_id`, `split`, `frames`를 둔다.
Frame에는 `frame_index`, `target_frame_index`, `source_frame_sha256`,
`tcp_world_m`, `object21`, `nodes`가 필요하다. 노드에는 `object_id`
(Peg 0, Hole 조립체 1), `is_vertex`, `visible`, `source` (`observed`,
`tracker`, `missing`)가 필요하다. 위치 `position`은 지원되는 노드의 월드
xyz이고 `confidence`는 선택적인 [0, 1] 값이다. `identity_verified: true`인
semantic vertex만 해당 `vertex_id`를 B1/B2 결합에 사용한다. 꼭짓점 동일성 판단은
관측을 생산하는 tracker/adapter가 제공한다. 새 manifest의 원본 레코드 해시는
producer가 제공하며 로컬 native replay 경로는 원본 파일까지 대조한다.

모든 모델은 같은 builder에서 13개 node feature와 7개 edge feature를 받는다.
Node feature에는 고정 물리 스케일로 변환한 world-axis TCP-relative xyz,
물체·노드 유형 one-hot, 실제 가시 여부, confidence 값/유효 여부, provenance
one-hot이 들어간다. 가려진 좌표 자체는 버리고 missing slot과 batch padding을
구분한다. Tracker 추정 좌표는 spatial geometry에 쓸 수 있으며 B1/B2의 시간
attention key는 직접 관측한 검증 ID로 제한한다. 확인된 동일 물체 내 ID가 중복되면
입력을 거부한다. 배열 순서와 근접성은 동일 ID의 근거에서 제외한다.

Edge는 Peg와 Hole 조립체의 월드 좌표를 따로 triangulate한다. Affine rank 3에서
3D Delaunay, rank 2에서 평면 투영 후 2D Delaunay, rank 1에서 선 위 인접
점을 사용한다. 중복 좌표는 triangulation 전에 합치고 기존 노드에 이웃을
되돌린다. 유일 위치가 하나 이하거나 Qhull 실패 시 고립 노드를 보존하고
방법을 기록한다. Edge는 양방향이며 destination−source 벡터, 거리, 세
node-pair 유형 one-hot을 전달한다. 기본 80 mm를 넘는 링크를 버린다. 저장
캐시에 분할 마스크가 없어 짧은 가림 경계를 가로지르는 링크는 판별할 수 없다.
Delaunay edge는 물체 표면 연결의 보증으로 취급하지 않는다.

| 모델 | `model.name` | 처리 순서 |
| --- | --- | --- |
| A | `spatial` | 현재 프레임 spatial GNN, 물체별 pooling |
| B1 | `temporal_first` | 검증 vertex별 causal attention, 현재 프레임 spatial GNN |
| B2 | `spatial_first` | 프레임별 spatial GNN, 검증 vertex별 causal attention |
| C1 | `object_transformer` | 프레임별 spatial GNN과 물체별 pooling, 물체별 causal Transformer |
| C2 | `object_mamba` | 프레임별 spatial GNN과 물체별 pooling, 물체별 Mamba-1 |

기존 `EdgeGatedMPNNLayer`와 `ConfidenceAttentionPool`을 사용한다. Edge 기하는
가중치를 결정하는 sigmoid gate와 message MLP 양쪽에 들어간다. B1·B2는 동일한
모듈과 readout을 사용하고 적용 순서가 다르다. C1·C2는 같은 shape head로
Peg 반쪽 크기 3개와 Hole 반지름을 예측한다. Auxiliary loss를 끈 경우에도 모든
모델은 공통 Object21D loss를 사용한다. Positional loss는 물리 스케일을
고정하고 quaternion loss는 q와 −q를 같은 회전으로 처리한다. Shape target에는
pose 항이 없다. 일정 크기의 validation 집합에서 shape memory를 판정할 수 없다.

`model(frames)["object21"]`은 현재 프레임의 `[B,21]`을 반환한다. 입력
`frames`는 시간순 `GraphBatch` 목록이고, frame graph는 관측 `Frame`만
받는다. Target, action 및 robot state는 model forward에 전달되지 않는다.
Output은 TCP 상대 위치, world wxyz, Peg +X 끝점−Hole과 크기의 기존 규약을
따른다. 필요 시 기존 `assemble_state38`에 별도로 연결한다.

공정 비교에서 A는 현재 프레임, B1/B2/C1/C2는 `data.history`까지의 저장
프레임을 사용한다. B1/B2/C1은 episode 경계와 마스크를 적용하고 원본 frame
index 간격을 attention time feature에 남긴다. C2의 `step(frame, state)`로
전체 episode stream을 처리할 수 있다. 호출자가 cache를 보유하며 episode ID가
변한 배치 행만 초기화한다. 학습에서 `tbptt_steps` 간격으로 상태를 detach한다.
공통 평가의 C2는 매 window에서 초기화한 상태로 실행한다. Streaming에서는
cache를 계속 전달하면 window 비교보다 긴 이력이 남는다.

## 공유 TAPNext++ State36 캐시 학습

`data.kind: tapnext_state36`는 공유 NPZ `tapnext_state36.npz`와
같은 원본 manifest를 연결한다. 그래프에는 TAPNext++가 가시로 판단하고 현재
깊이가 유효한 픽셀을 camera calibration으로 월드에 올린 좌표와 관측 TCP가
들어간다. 깊이 표본은 해당 픽셀에서 측정된 표면이며, 물체 마스크로 소유권을
검증하는 단계는 없다. B1·B2가 사용하는 꼭짓점 ID는 동일 에피소드 내부의
고정 TAPNext++ 쿼리 번호다. State36의 0-based 열 `15:36`을 Object21D
정답으로 사용한다.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/shared_tapnext_smoke.yaml
```

정렬된 reset seed를 seed 0으로 섞어 **train 80 / validation 10 / test 10**
그룹으로 고정했다. 총 240/30/30 에피소드와 49,719/6,293/6,383 프레임이다.
같은 reset seed의 세 정책은 모두 같은 split에 속한다. 전체 manifest에서
split 간 원본 프레임 해시와 초기 fingerprint 중복이 없는지 검사했다.

CPU 스모크는 train 3개 seed와 validation 2개 seed의 모든 정책에서
첫 24프레임을 사용했다. 다섯 모델 모두 같은 train 216개와 validation
144개 프레임, seed 0, 1 epoch, 15 optimizer step을 사용했다. Test는
분할 감사에 사용하고 스모크 학습과 validation에서는 보류했다.
B1/B2/C1/C2의 시간 모듈 가중치도 갱신됐다. 같은 실제 관측을 고정하고
State36의 36개 열을 모두 변경했을 때, 저장 체크포인트 다섯 개의 예측이
각각 완전히 일치했다.

| 모델 | Train Object21D loss | Validation Object21D loss |
| --- | ---: | ---: |
| A | 5.770 | 4.579 |
| B1 | 6.213 | 4.702 |
| B2 | 6.206 | 4.881 |
| C1 | 4.721 | 4.117 |
| C2 | 5.996 | 4.486 |

공통 Object21D loss는 이 스모크의 수치 확인용이다.
[실행·분할·GT 변경 검증](../outputs/state_estimation/shared_tapnext_smoke/verification.json)

### 파일럿과 Expert 복귀 데이터의 CUDA 스모크

`data.sources`에 두 공용 NPZ와 각 원본 manifest를 지정한다. 기존 분할
receipt와 복귀 NPZ의 seed 배정을 함께 확인한다. 두 원본 전체에서 초기 장면
fingerprint와 공유 raw frame hash가 split을 넘지 않는 것도 검사한다.
전체 train/validation/test는 reset seed 80/10/10개, 에피소드 320/40/40개다.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/pilot_recovery_cuda_smoke.yaml
```

이 설정은 train 3개와 validation 2개 reset seed를 사용한다. 파일럿은
에피소드마다 첫 24프레임을, 복귀 데이터는 Expert 복귀 직전 12프레임과
복귀 프레임부터 12프레임을 사용했다. 다섯 모델 모두 같은 train 288프레임
(파일럿 216, 복귀 72)과 validation 192프레임(파일럿 144, 복귀 48)으로
RTX 3090에서 FP32 1 epoch와 19 optimizer step을 완료했다.
Test는 전체 분할 감사에 포함하고 스모크 학습·검증에서는 보류했다.

| 모델 | Train Object21D loss | Validation Object21D loss |
| --- | ---: | ---: |
| A | 5.691 | 4.595 |
| B1 | 6.039 | 4.762 |
| B2 | 5.642 | 4.638 |
| C1 | 4.565 | 4.143 |
| C2 | 5.739 | 4.251 |

다섯 체크포인트에서 데이터 SHA-256, 선택된 원본 프레임과 epoch 종료 시
shuffle generator 상태가 일치했다. 네 시간 모듈은 학습으로 갱신됐다.
GPU에서 다시 로드한 모든 체크포인트가 실제 복귀 관측에 유한한 Object21D를
출력했다. [CUDA 실행·분할 검증](../outputs/state_estimation/pilot_recovery_cuda_smoke/verification.json)
에는 실행 ID, 체크포인트 경로 및 해시가 있다.

## Fixed-Hole curriculum

`graph/configs/state_estimation/fixed_hole_curriculum.yaml`은 기존 파일럿과 복귀 캐시를
합친 400개 에피소드의 reset seed 분할을 재사용한다. 기존 분할의 학습 320개와 검증
40개를 그래프로 한 번 변환한다. 보류한 테스트 40개는 준비한 학습 그래프에 포함하지 않는다.
양쪽 원본 캐시, raw manifest와 프레임 파일, 기존 split receipt가 있는 환경에서
`phase: prepare`로 돌린다. 다른 GPU에서는 준비된 그래프 디렉터리와 split receipt를
복사해 같은 설정의 `phase: train`으로 학습한다.

| 단계 | 학습 및 검증 데이터에 추가하는 프레임 |
| --- | --- |
| `expert` | Expert 전체 |
| `post_grasp_perturbation` | 전환 시 실제로 Peg를 쥔 Expert→Random 에피소드의 시작부터 Random action 12개까지 |
| `extended_perturbation` | 모든 Expert→Random의 전체 프레임과 순수 Random 에피소드당 처음 24프레임 |
| `expert_recovery` | Expert→Random→Expert 전체 |

모델별 첫 단계는 새 가중치로 시작한다. 각 후속 단계는 직전 검증으로 선택한 가중치를
불러오고 AdamW를 새로 시작한다. 단계당 1 epoch가 기본값이다. `status.json`은 단계별
학습·검증 프레임 수와 완료한 체크포인트를 기록한다. Hole 이동은 해당 관측 캐시를 준비한
뒤 추가할 수 있다.

학습이 끝나면 `graph/configs/state_estimation/fixed_hole_holdout.yaml`을 루트
`main.py`로 실행한다. 두 원본 캐시와 테스트 원본 프레임, 고정 분할 및 그래프
manifest, 다섯 모델의 `status.json`과 마지막 단계 체크포인트가 필요하다.
40개 테스트 에피소드의 전체 프레임에서 Peg 끝점−Hole 상대 벡터의 L2 오차를
mm로 계산한다. 복귀 지표는 그중 Expert→Random→Expert 10개에서 집계한다.
저장된 `recovery_step`은 첫 Expert 행동 직전 관측이며 직후 오차는
`recovery_step + 1`에서 잰다. 안정화 시간은 첫 Expert 행동 이후부터
오차가 10 mm 이내인 관측이 5프레임 연속으로 확인되는 시점까지의 프레임 수다.
20 mm 기준도 같이 기록한다. 각 시간 중앙값은 안정화가 관측된 에피소드만
집계하며 전체 10개 중 도달 개수를 함께 기록한다.

`graph/configs/state_estimation/fixed_hole_spatial_first_resume.yaml`은
B2의 최종 고정 Hole 단계 체크포인트에서 가중치, AdamW, RNG를 복원한다.
기존 그래프 manifest와 reset seed 분할로 학습·검증 윈도우를 다시 구성한다.
`epochs: 10`은 최종 단계의 누적 epoch 수다. 기존 1 epoch에서 시작하면
2~10 epoch를 실행하며, 결과와 체크포인트는 별도 실행 디렉터리에 저장한다.

`graph/configs/state_estimation/fixed_hole_spatial_first_early_stopping.yaml`은
10 epoch의 `last.pt`에서 가중치, AdamW와 RNG를 복원한다. 직전 최저 검증
Object21D loss를 기준으로, 0.001을 넘는 개선이 10 epoch 연속 없으면 종료한다.
최대 100 epoch까지 실행한다. `last.pt`에 최저 검증 loss의 가중치와 조기 종료
상태를 보존하며, `training.json`에 실제 종료 epoch와 종료 여부를 기록한다.

## 실행과 재현

Python 3.11, PyTorch, NumPy, SciPy, PyYAML이 필요하다. 테스트는 pytest,
coverage 확인은 pytest-cov를 사용한다. CPU를 기본으로 지정한다.
`training.amp: true`는 CPU 또는 BF16을 지원하는 CUDA에 적용한다.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/compare.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/audit_tapnext.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/tapnext_smoke.yaml
```

모델별 YAML은 공통 `compare.yaml`을 가리킨다. 공정 비교는 데이터, 필터, window,
seed, AdamW, epoch, clip, 평가 지표를 공유한다. 체크포인트는 validation의
공통 Object21D objective로 선택한다. `last.pt`에 선택된 model, 마지막
model, optimizer 및 RNG가 함께 있다. 재개할 때 같은 설정에서 `resume`에
경로를 지정하고 `training.epochs`를 늘린다. 단일 seed를 지정하려면
`seeds: [0]`처럼 목록을 설정한다.

평가는 해당 데이터/모델 YAML 구성을 유지하고 `phase: evaluate`,
`evaluation.checkpoint`, `evaluation.split`을 설정한다. 예를 들어 실데이터
평가는 smoke YAML을 복사하고 phase를 변경한다. 원본 manifest가 다른 위치에
있으면 동일 SHA-256의 경로를 `data.source_manifest`에 지정할 수 있다. `phase: stress`는 같은
checkpoint에 10개 합성 조건을 적용한다. 상세 accuracy와 phase별 mean/median/P90/P95,
drift 주입량별 error 및 재관측 시간별 error는 JSON/CSV에 기록된다. Latency는
batch 1의 동기화된 forward이며 그래프 생성 및 장치 이동을 제외한다. 처리한 노드와
방향성 edge 수로 입력 규모를 기록한다. CUDA는 peak PyTorch 할당량을 함께 기록한다.
기존 `ExperimentLogger`가 config, seed, Git commit/dirty, run ID, dataset/checkpoint
SHA-256 및 output sidecar를 관리한다.

분할 감사는 부모·reset fingerprint·원본 frame hash의 split 중복과 명시된 tracker
학습·선택 parent와의 held-out 중복을 실패로 처리한다. 샘플의 원본 frame index와
target index를 비교하고 윈도는 episode 내에서만 생성한다. TCP 궤적을 16점으로
재표본화한 평균 거리 2 mm 이하의 근접 후보도 별도로 기록한다. 이 진단이 감지한
결과는 확인 필요 표시와 함께 저장된다. 데이터에서 정규화 통계를 계산하지 않는다.

```bash
WANDB_MODE=disabled python -m pytest -q tests/test_state21_*.py \
  -o junit_family=xunit1 --junitxml=outputs/state_estimation/tdd_verification.xml \
  --cov=graph.state_estimation --cov=data_collection.state21 \
  --cov=data_collection.state21_batches --cov=data_collection.state21_tapnext \
  --cov=data_collection.state21_synthetic --cov=training.state21 \
  --cov=training.state21_loss --cov=evaluation.state21 --cov-branch

python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -q
```

## TDD 기록

| 순서 | 먼저 확인한 실패 | 통과시킨 변경 |
| --- | --- | --- |
| 공통 계약 | 관측·builder 모듈이 구현되기 전 `test_state21_data.py` 수집 실패 | Object21D codec, 관측·GT 분리, 물체별 Delaunay와 split 감사 |
| A | `SpatialEstimator` import 실패 | Gate와 message 양쪽의 edge 반응 및 빈 graph 처리 |
| B1 | 클래스 부재, 첫 구현에서 episode 연결 시 위치 인코딩으로 인해 예측 불일치 | 검증 ID 결합과 episode suffix 기준 위치 인코딩 |
| B2 | `SpatialFirstEstimator` import 실패 | 역사 프레임별 spatial pass 후 동일 vertex attention |
| C1 | `ObjectTransformerEstimator` import 실패 | 물체별 temporal attention, static size auxiliary loss |
| C2 | `ObjectMambaEstimator` import 실패 | ReferenceMamba의 독립 cache, episode reset, TBPTT |
| 통합 | 학습 모듈 부재. 첫 과적합 실행에서 A/B1/B2/C2가 기존 오차 기준에 미달 | 전체 workflow와 동일한 tiny 학습 스케줄 적용. 검증 assertion은 유지 |

각 모델의 단위 테스트가 통과한 뒤 다음 모델을 구현했다. CPU에서는 다섯 모델의
유한한 gradient, BF16 train/eval, batch 1 및 다중 배치, optimizer step,
dropout 포함 epoch 재개의 마지막 weights 및 best weights 일치를 확인했다.
C2의 reference scan과 40-step 결과를 비교했다. CPU 과적합은 두 크기와
여러 자세가 포함된 6 frame을 사용했다. Loss 항별 값은
[JUnit 결과](../outputs/state_estimation/tdd_verification.xml)에 저장했다.

## 검증 결과

macOS Apple M4 Pro, Python 3.11.18, PyTorch 2.14.1. TAPNext++의 저장
캐시로 train 73개 및 validation 27개 프레임에서 seed 0, CPU 1 epoch를
실행했다. 검증 추론은 CPU 스레드 1개, batch 1, history 최대 4 저장
프레임으로 측정했다. A는 항상 현재 프레임만 사용한다.
실데이터 측정은 저장 캐시부터 시작하며 TAPNext++ 추론 비용을 포함하지 않는다.

| 모델 | 구현 | 단위 | 통합 | Split 누출 방지 테스트 | Tiny overfit | TAPNext++ CPU smoke | 파라미터 | CPU mean / P95 / P99 ms |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| A | 완료 | PASS | PASS | PASS | PASS | PASS | 185,047 | 0.661 / 0.734 / 0.891 |
| B1 | 완료 | PASS | PASS | PASS | PASS | PASS | 218,647 | 0.729 / 0.809 / 0.836 |
| B2 | 완료 | PASS | PASS | PASS | PASS | PASS | 218,647 | 1.968 / 2.293 / 2.344 |
| C1 | 완료 | PASS | PASS | PASS | PASS | PASS | 222,683 | 2.328 / 2.763 / 2.987 |
| C2 | 완료 | PASS | PASS | PASS | PASS | PASS | 221,851 | 2.831 / 3.259 / 3.385 |

[실데이터와 합성 실행 인덱스](../outputs/state_estimation/verification_index.json)는
checkpoint selection, training time, latency, 오류 quantile 및 window별
지표의 원본 JSON을 연결한다. [합성 3-seed 비교](../outputs/state_estimation/synthetic_seed_comparison.json)는
모델별 독립 test와 10개 스트레스 조건을 seed 0/1/2로 실행한 결과다.
장시간 완전 가림 동안의 Hole 위치 오차는 모델별 프레임 오차와 seed 간
편차를 남겼다. 이는 경험적 편차이며 모델의 확률적 불확실성 추정값은 없다.

실데이터 검사에서는 동일 부모·reset fingerprint·원본 해시의 split 중복과
TCP 유사도 후보가 없었다. 독립 test, 검증 semantic ID, tracker 학습/선택
에피소드 목록이 이 캐시에 없다. B1/B2의 실데이터 smoke는 ID가 없는 관측을
안전하게 처리하는 경로까지 검증했다. B1/B2의 ID 기반 시간 결합과 C1/C2의
다양한 크기 학습은 합성 조건에서 검증했다. 실제 모델 간 성능 개선은
별도 독립 test에서 확인해야 한다.

신규 테스트 66개 통과, CUDA 테스트 5개는 이 호스트에서 CUDA가 없어
실행하지 않았다. 다섯 모델의 MPS FP32 합성 train/eval smoke를 실행했다.
CUDA 수치, fused Mamba, CUDA peak memory, 정식 실데이터 학습은
실행하지 않았다. 전체 저장소 검증은 613개 통과, 26개 skip, 기존 로거
테스트 1개 실패였다. `imageio.plugins`를 먼저 import하면 기존 테스트의
전역 `importlib.import_module` mock과 `assertWarns`가 충돌하는 문제를
별도 프로세스에서 재현했다. 기존 Logger/Artifact 계약 테스트는 unittest로
분리 실행해 36개 통과했다.

신규 builder, 모델과 데이터 모듈의 branch coverage는 93.5% (230/246)였다.
[coverage 원본](../outputs/state_estimation/coverage.json)에 실행 분기를
기록했다. 기존 리더보드 모델·학습 코드와 수치 의존성은 변경하지 않았다.

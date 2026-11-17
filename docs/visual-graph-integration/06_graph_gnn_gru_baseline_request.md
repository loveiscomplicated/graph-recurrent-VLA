# Task 06 — Minimal Graph–GNN–GRU/BC Integration Baseline

## 0. 요청: 통합 구현 전 계획 수립

Graph-Recurrent VLA 팀 프로젝트에서 Vision 기반 기하학적 Graph의 기본 표현과 Graph Readout 설계를 논의했다.

이제 목표는 다음 전체 경로가 실제로 동작하는 **최소 통합 기준선(Minimal Working Baseline)** 을 만드는 것이다.

`Vision Geometry → Graph Construction → Edge-aware GNN → Structured Graph Readout → Robot State Fusion → GRU → BC Action`

단, 이번 요청에서는 **기존 구현과 데이터 경로를 확인하고 최소 구현·검증 계획만 작성하라.**

구현, 학습, GPU Benchmark, Rollout은 별도 승인 전까지 실행하지 않는다.

이번 작업은 새로운 Graph 알고리즘이나 모델을 탐색하는 연구가 아니다.

이미 합의한 표현을 실제 학습 파이프라인에 연결할 수 있도록 인터페이스를 완성하는 작업이다.

---

## 1. 연구 목적과 설계 원칙

우리 Graph의 목적은 다음과 같다.

**RGB-D 관측에서 물체와 로봇의 기하학적 형상·공간 관계를 추출하고, 이를 이미지 전체보다 낮은 비용의 구조적 표현으로 GNN에 전달한다.**

Graph가 로봇의 전체 제어 상태를 표현할 필요는 없다.

Graph와 Robot State는 분리한다.

### Graph에 포함할 정보

- 물체 표면의 Point 위치
- 물체 Point의 기하학적 Feature
- TCP와 양쪽 Fingertip의 기하학적 위치
- 물체 내부의 국소 연결
- Robot–Object 상대 관계
- Peg–Box 상대 관계

### Graph에 넣지 않을 정보

- 전체 Joint State
- Robot Velocity
- Gripper Control State
- Graph 표현에 불필요한 Proprioception
- GT Object Pose/Hole 위치
- RGB Appearance Feature
- Action Label 또는 미래 정보

위에서 제외한 Robot State는 필요할 경우 **Graph와 별도 모델 입력**으로 전달한다.

Robot State와 Graph Representation은 GNN 이후에 결합한다.

현재 프로젝트는 Expert Demonstration 기반 BC 연구다. Goal Graph, GCSL, 새로운 Pretraining Objective 등을 추가하지 않는다.

---

## 2. 재사용할 Vision 결과

기존 실험을 정확성 근거로 사용한다.

| Task | UUID | 재사용 목적 |
|---|---|---|
| Task03-K | `9185355a32b8433abc70f1738f983377` | Shape-First Sampling |
| Task04 | `9f90293b227e4dbfaaa6290afad9cc3b` | F=5 및 Vision–Graph Feature/Edge 계약 |
| Task05-B | `9f36f7c89dbd4097a8be9358138d1537` | Extended Delaunay |
| Task05-E | `0f107550c9b64f1a84677a18e988cb51` | Raw EDE 결과와 Receiver 계약 |

Task05-C/D의 Exact 및 GPU 최적화 이력도 확인하되, 이번 기준선에서 무거운 Strict Edge Validation이나 GPU B2를 다시 도입하지 않는다.

### Vision 기본 정책

- Shape-First + Surface-Second
- Variable Point N
- 현재 잠정 `keep_ratio=0.5`
- 원본 관측 3D Point 사용
- EE 좌표계의 Point Position
- 기존 Continuous Geometric Feature F=5
- Raw Extended Delaunay Intra-object Edge

`keep_ratio=0.5`는 현재 사용 중인 잠정 설정이지 최종 성능 최적값이 아니다.

기존 Sampling, PCA5, Feature Extraction, EE Transformation, Raw EDE 로직은 변경하지 않는다.

저장 Vision Packet을 우선 이용하고, 새 Vision 실험을 수행하지 않는다.

---

## 3. 전체 Graph 구조

### 3.1. Robot Node

정확히 세 개의 Robot Geometry Node를 사용한다.

1. TCP
2. Left Fingertip
3. Right Fingertip

기존 수신 계약의 Robot Node 3개와 Point Index `+3` 규칙을 유지한다.

모든 위치는 동일한 EE 좌표계로 표현한다.

실제 TCP/Fingertip 위치는 같은 시점의 유효한 Robot Observation에서 가져와야 한다.

Robot Node를 만들기 위해 필요한 실제 위치 정보가 기존 데이터에 없다면 그 사실을 명확하게 보고하라.

Synthetic Robot Geometry는 계약 테스트에만 사용할 수 있으며, 이를 실제 Observation 통합 성공으로 보고하지 않는다.

### 3.2. Object Node

기존 Peg와 Box의 선택된 표면 Point를 사용한다.

- Point 추가·삭제 없음
- Source Index 보존
- Variable N 유지
- Object Identity 유지
- 원본 Point Geometry 및 Feature 불변

### 3.3. Intra-object Edge

Task05-E의 **Raw EDE**를 그대로 사용한다.

Vision이 제공한 Intra-object Edge를 Graph에서 새로 생성하지 않는다.

기존 Edge Index를 Robot 3개 Node에 맞춰 offset한 뒤 전체 Graph에 결합한다.

Strict EDE의 Supercover·Mask·Depth 경로 검증은 재실행하지 않는다.

### 3.4. Robot–Object Edge

우리가 합의한 방식은 **Robot Node마다, 물체별로 가장 가까운 k개 Point를 연결하는 것**이다.

예를 들어 TCP는 Peg에서 k개, Box에서 k개를 각각 선택한다.

Left/Right Fingertip에도 동일한 규칙을 적용한다.

거리 기준은 EE 좌표계의 3D Euclidean Distance다.

- 실제 물체 Point가 k개보다 적으면 가능한 Point만 사용
- 물체가 비어 있으면 해당 물체로 향하는 Edge 없음
- 양방향 Directed Edge 생성
- 추가 Contact Classifier 없음
- Normal/Depth/Mask 기반 새로운 필터 없음

`k`는 아직 결정하지 않은 하이퍼파라미터다.

동일 거리 tie-break는 Source Index 및 Point Row를 사용해 deterministic하게 정의할 계획을 제출하라.

### 3.5. Peg–Box Edge

**두 물체 사이의 전체 Point 쌍에서 가장 가까운 M개 쌍**을 선택한다.

- EE 기준 3D Euclidean Distance
- Global nearest M pairs
- 양방향 Directed Edge
- 중복 연결 제거
- 한 물체라도 비어 있으면 연결 없음

`M=0`은 후속 ablation에서 Peg–Box 직접 Edge가 없는 조건으로 허용한다.

`M` 역시 최종적으로 실험으로 결정한다.

Point 쌍 탐색의 시간·메모리 복잡도를 확인하고, 기존 Variable N 범위에서 실행 가능하도록 최소 구현을 설계하라.

근사 최근접 탐색, 임의 거리 cutoff 또는 접촉 판정은 이번에 도입하지 않는다.

---

## 4. Node Feature Schema

Node Geometry와 Geometric Feature를 분리한다.

### 공통 Node Position

`node_pos_ee: [N_total, 3]`

### Object Geometric Feature

기존 Task04의 5차원 Feature를 사용한다.

`[normal_x, normal_y, normal_z, normal_variation_rad, depth_jump_m]`

주의:

- Normal은 EE 좌표계
- Normal Variation은 radian
- Depth Jump는 기존 Camera-z 기반 측정값
- 원본 Feature 정의와 단위 변경 금지

### Feature Availability / Validity

기존 Task04의 두 Mask를 유지한다.

- `field_available`
- `field_valid`

Robot Node에 Object Surface Normal이 없다는 사실을 유효한 영벡터 Normal로 해석하지 않는다.

GNN이 NaN을 직접 연산하지 않도록 Finite Input Representation을 만들되, **원본 Feature 값과 두 Mask의 의미를 유지하고 네트워크에서 Mask를 실제 소비**하도록 설계하라.

### Node Type

다음 다섯 종류를 구분한다.

- TCP
- Left Fingertip
- Right Fingertip
- Peg Point
- Box Point

최소 Node Type Embedding을 검토한다.

Robot의 Proprioception을 Node Type이나 Geometric Feature에 추가하지 않는다.

---

## 5. Edge Feature Schema

모든 Edge에 공통 4차원 Geometric Feature를 사용한다.

`edge_attr = [distance, direction_x, direction_y, direction_z]`

Directed Edge `i→j`에 대해:

\[
\Delta p_{ij}=p_j^{EE}-p_i^{EE}
\]

\[
d_{ij}=\|\Delta p_{ij}\|_2
\]

\[
u_{ij}=
\begin{cases}
\Delta p_{ij}/d_{ij}, & d_{ij}>0\\
(0,0,0), & d_{ij}=0
\end{cases}
\]

`edge_attr = [d_ij, u_ij]`

Reverse Edge에서는 거리는 동일하고 방향만 반대다.

이 Feature는 아래 세 종류의 Edge에 모두 부여한다.

1. Intra-object
2. Robot–Object
3. Peg–Box

`edge_type`은 별도 범주형 값이다.

새로운 Pairwise Contact Probability, Relative Pose Estimator, GT Relation Feature는 추가하지 않는다.

---

## 6. 최소 GNN Encoder

처음에는 **Edge-aware MLP Message Passing**을 사용한다.

Graph Attention, Graph Transformer, EGNN 및 다른 복잡한 Encoder를 도입하지 않는다.

### 6.1. Node Encoder

Node Position, F=5, Availability/Validity Mask, Node Type을 이용해 Node Embedding을 생성한다.

Node Embedding 크기는 `d`다.

실제 입력 정규화 및 Mask 처리 위치를 기존 코드에 맞춰 설계하되, 원본 데이터의 의미를 변경하지 않는다.

### 6.2. Message MLP

Directed Edge `j→i`에 대해:

\[
m_{j\rightarrow i}
=
\operatorname{MLP}_{msg}
([h_i,h_j,e_{ji},t_{ji}])
\]

여기서:

- `h_i`, `h_j`: Node Embedding
- `e_ji`: 4D Geometry
- `t_ji`: Edge Type Embedding

첫 기준선에서는 세 Edge Type이 하나의 Message MLP를 공유한다.

Edge Type별 별도의 복잡한 Network는 만들지 않는다.

### 6.3. Message Aggregation

첫 기준선은 Mean Aggregation이다.

\[
\bar m_i=
\operatorname{Mean}_{j\in\mathcal N(i)}
m_{j\rightarrow i}
\]

수신 메시지가 없는 Node도 NaN 없이 처리해야 한다.

Graph Readout의 Mean과 Message Passing의 Mean Aggregation은 별개의 연산이다.

### 6.4. Node Update

Residual MLP Update를 사용한다.

\[
h_i^{(l+1)}
=
h_i^{(l)}
+
\operatorname{MLP}_{upd}
([h_i^{(l)},\bar m_i])
\]

Message Passing Layer 수는 `L`이다.

이번 최소 기준선에서 `L`, `d`의 임시 실행값은 필요할 수 있지만, 해당 숫자를 최종 아키텍처로 확정하지 않는다.

---

## 7. 합의된 Structured Graph Readout

이 Readout 구조는 이번 기준선에서 변경하지 않는다.

\[
\boxed{
z_G=
[h_{\mathrm{TCP}},
h_L,
h_R,
h_{\mathrm{Peg}},
h_{\mathrm{Box}}]
}
\]

Readout 규칙:

- TCP: 개별 Node Embedding
- Left Fingertip: 개별 Node Embedding
- Right Fingertip: 개별 Node Embedding
- Peg: Peg Point Embedding의 Mean
- Box: Box Point Embedding의 Mean

모든 Node를 하나로 합치는 Global Pooling을 사용하지 않는다.

각 Node Embedding이 `d`차원이면 `z_G`는 `5d`차원이다.

### Empty Object 처리

해당 물체의 Point가 없는 경우:

- Mean 분모 0 방지
- 빈 물체의 Readout은 정의된 영벡터 사용
- 물체 관측 여부를 별도 Mask로 전달
- Robot Readout은 유지
- 해당 물체와 연결된 Interaction Edge 없음

기존 자료의 Frame18/56처럼 Box가 관측되지 않는 사례도 테스트해야 한다.

이번에는 미구현 상태인 Stale Geometry 정책을 추가하지 않는다.

---

## 8. Robot State는 Graph 외부에서 결합

전체 모델을 다음과 같이 구성한다.

`Graph → GNN → Structured Readout`

`Robot Proprioception → 별도 State Input`

두 표현을 결합해 Recurrent Controller에 전달한다.

가장 단순한 후보는 Concatenation이다.

`[z_G, robot_state] → GRU → BC Action Head`

Robot State의 실제 필드와 차원은 기존 Expert Demonstration Loader 및 BC 구현을 조사해 확인하라.

필요한 State를 추측해 새로 생성하지 않는다.

기존 State Normalization과 Action Convention을 최대한 재사용한다.

Robot State를 Graph Node Feature로 복사하지 않는다.

복잡한 State Attention/Fusion Network를 새로 설계하지 않는다.

---

## 9. GRU와 BC Action Head

기존 Recurrent / BC Prototype 구현이 있다면 우선 재사용한다.

다음을 조사하라.

- 기존 GRU 입력/출력 규약
- Hidden State 크기·Layer 수
- Hidden State 초기화
- Episode 시작/종료 시 Reset
- Action Target Dimension
- Action Scaling 및 Normalization
- Gripper Action의 표현
- BC Loss와 데이터 Loader
- Checkpoint 및 평가 방식

기존 Prototype과 새 Graph Encoder가 맞지 않는 부분만 최소한으로 연결한다.

Action Head의 물리적 출력 의미를 바꾸거나 새로운 BC Loss를 도입하지 않는다.

Sequence 입력과 단일 Step 추론을 모두 지원할 수 있는 최소 Interface를 설계하라.

**GRU Hidden State는 실제 시간에 따른 Observation Sequence 사이에서 유지한다.**

한 Observation에 대한 반복적 Refinement를 Recurrent 동작으로 간주하지 않는다.

---

## 10. 실제 데이터와 시간 정합 조사 — 중요

이번 계획의 핵심 조사 항목이다.

기존 Task03/04/05 저장 Vision Packet은 일부 고정 Frame에 대한 Geometry 평가 결과다.

이것만으로 실제 BC 학습에 필요한 충분한 시계열 데이터가 확보됐다고 가정하지 않는다.

Repository에서 다음을 확인하라.

1. 실제 Expert Demonstration Episode 수와 경로
2. Train/Validation Episode Split의 현재 구현 여부
3. Observation과 Expert Action의 시간 정합
4. Robot TCP/Fingertip 실제 관측값의 제공 여부
5. Graph 생성에 필요한 RGB-D/Mask/Point 자료의 Episode Coverage
6. 기존 Robot Proprioception Field 및 Action Schema
7. 기존 GT 기반 Graph Prototype과 실제 Vision Packet의 차이
8. Variable N Graph를 시간순으로 공급하는 기존 Loader/Collate 지원 여부

Expert Action은 BC Target으로 사용 가능하지만, 미래 Action 또는 GT Object Pose를 현재 Vision Graph 입력에 누출해서는 안 된다.

만약 전체 Demonstration에 필요한 Vision Graph가 없다면, 최소 Forward 테스트에 쓸 저장 자료와 실제 학습에 필요한 추가 데이터 준비를 명확하게 분리하라.

이 단계에서 누락된 Vision Graph를 생성하기 위해 전체 Episode를 재처리하지 않는다.

---

## 11. Minimal Working Baseline의 완료 기준

후속 구현 승인 후 다음 동작을 먼저 검증할 계획을 작성한다.

### A. Graph Construction Contract

- Robot 3개 Node 존재
- 기존 Object Point 행 순서 보존
- Raw EDE Intra-object Edge 보존
- Robot–Object 물체별 k-NN
- Peg–Box 최근접 M쌍
- 모든 Edge의 4D Feature
- Node/Edge Type 일치
- Directed Edge 양방향과 Index 검증
- GT-independent Geometry Selection

### B. GNN Forward

- 실제 저장 Vision Packet으로 Graph Tensor 구성
- 실제 Robot Geometry가 존재하는 경우 사용
- Variable Point N 처리
- Empty Object 처리
- Feature Validity Mask 반영
- 모든 Node Embedding Finite
- Message Passing 정상 동작
- Structured Readout 크기 `5d`

### C. GRU/BC Forward

- Graph Embedding과 별도 Robot State 결합
- Episode Sequence 입력
- GRU Hidden State 유지·Reset
- 기존 BC Action Convention에 맞는 출력
- Action Tensor Shape 및 Finite 검사

### D. Minimal Backward Smoke Test

- 작은 실제 학습 배치가 준비된 경우 Loss 계산
- GNN/GRU/Action Head Gradient Flow 확인
- Optimizer Step 실행 가능 여부 확인
- Graph Batch 간 Index 충돌 없음
- 학습 Label과 Observation 시점 정합

데이터가 준비되지 않았다면 Synthetic Tensor로 연산 계약만 검증하고 실제 BC 통합 성공으로 주장하지 않는다.

**Smoke Test PASS는 모델이 조작을 학습했다는 뜻이 아니다.**

---

## 12. 하이퍼파라미터 — 실험으로 결정

다음 네 가지를 최종값으로 확정하지 않는다.

| Hyperparameter | 의미 |
|---|---|
| `L` | Message Passing Layer 수 |
| `d` | Hidden Dimension |
| `k` | Robot Node별·물체별 최근접 Point 수 |
| `M` | Peg–Box 최근접 Pair 수 |

최소 Forward/Backward Smoke Test를 위해 임시 Config 값을 제안할 수 있다.

다만:

- `provisional_smoke_config`로 표시
- 사용 이유와 예상 Graph 크기 기록
- 학습 성능상 최적값이라고 주장하지 않음
- 기존 논의 없이 최종값 채택 금지

### 후속 HPO/Ablation 계획

최소 기준선이 검증된 다음의 별도 실험으로 설계한다.

**Phase 1 — Graph Connectivity**

우선 `k`, `M` 비교.

GNN의 나머지 설정을 임시 고정한다.

`M=0`을 포함해 Peg–Box 직접 Edge의 효과를 확인한다.

**Phase 2 — GNN Capacity**

연결 설정을 고정하고 `L`, `d`를 비교한다.

**Phase 3 — Interaction Check**

남은 소수의 후보 조합에서 `k,M,L,d`의 상호작용을 검토한다.

현재 단계에서 임의의 Grid Search를 실행하지 않는다.

실험 후보 범위와 실제 Training Budget은 이후 별도 승인으로 정한다.

---

## 13. 향후 학습·평가의 공정성

다음 비교는 후속 연구 대상으로 유지한다.

- Flat Representation vs Graph Representation
- Feed-forward vs GRU
- Raw EDE vs Strict EDE — 필요 시
- Peg–Box Edge 없음 vs 있음
- GT-based Geometry vs Estimated Vision Geometry — 입력 출처 구분

비교 조건에서는 가능한 한 동일한 Expert Demonstration, Episode Split, Action Target, State Input, Training Budget 및 Evaluation Seed를 사용한다.

Graph와 Flat의 정보량이 다르면 그 사실을 기록하고, Graph의 효과라고 과장하지 않는다.

평가 지표:

- Validation BC Loss
- Action Prediction Error
- Closed-loop Manipulation Success Rate
- Collision/Failure
- Inference Latency
- GPU Memory / Parameters

현재 요청에서는 어떠한 Training이나 Closed-loop Rollout도 실행하지 않는다.

---

## 14. 최소 구현 구조 조사

Repository의 실제 코드를 확인하고 기존 구현을 최대한 재사용하라.

특히 다음 영역을 조사한다.

- `visual/runtime/`
- `graph/vision_input.py`
- `graph/prototype/`
- `graph/prototype/v4/`
- 기존 Recurrent/BC 관련 코드
- Expert Demonstration Loader
- 현재 Evaluation/ExperimentLogger

다음 요소에 대해 재사용 가능한 함수·파일과 미구현 부분을 구분하라.

1. Robot 3개 Node의 실제 데이터 연결
2. Robot–Object k-NN
3. Peg–Box M-pair
4. 4D Edge Feature 계산
5. Node/Edge Tensor Schema
6. Variable-N Graph Batching
7. Edge-aware MLP GNN
8. Structured Readout
9. Robot State Fusion
10. GRU/BC Forward와 학습 데이터 정합

새로운 Graph Library 또는 Dependency 설치를 전제로 설계하지 않는다.

PyTorch 기본 연산으로 충분한지 우선 검토하라.

기존 Graph/GRU/BC 구현을 불필요하게 전면 재작성하지 않는다.

Production Default를 변경하지 않는다.

---

## 15. 후속 구현·검증 계획의 필수 산출물

이번에는 계획만 작성한다.

계획 Markdown:

`docs/visual-graph-integration/06_graph_gnn_gru_baseline_plan.md`

계획에는 다음 네 가지를 반드시 포함한다.

### A. 현재 구현 상태

- 실제 재사용 가능한 모듈
- Graph/Robot State/Action의 현재 인터페이스
- 확인된 데이터 경로
- 부족한 입력·미구현 모듈
- 예상 Blocker

### B. 최소 통합 설계

- Graph Construction 책임 분리
- Node/Edge Schema
- GNN Message Passing
- Structured Readout
- GRU/BC Integration
- Variable N 및 Empty Object 처리
- Robot State 별도 입력

### C. 단계적 구현·테스트

- Synthetic 계약 테스트
- 실제 저장 Vision Packet Forward
- 최소 Sequence/Backward Smoke
- Data/Action Alignment Gate
- Latency/Memory 측정 경계
- 실패 시 중단 기준

### D. 후속 학습·Ablation 경계

- `L,d,k,M` 실험 가능 Config 구조
- Minimal Baseline 검증 이후 학습 계획
- 공정한 비교를 위한 Episode Split
- GT/Estimated Geometry 구분
- Action Loss와 실제 Rollout 성능 구분

---

## 16. Markdown·정적 HTML 보고서 계획 — 필수

후속 구현이 승인되면 결과 Markdown:

`docs/visual-graph-integration/06_graph_gnn_gru_baseline.md`

새 ExperimentLogger UUID:

`experiments/<run_id>/graph_gnn_gru_baseline.html`

HTML에는 최소한 다음을 포함하도록 계획한다.

- 실제 Graph Node/Edge Schema
- Robot–Object 및 Peg–Box 연결 시각화
- Raw EDE와 Interaction Edge 결합 예시
- Object별 Point/Edge 수
- 4D Edge Feature와 Edge Type 검사
- Feature Validity Mask 및 Empty Object 사례
- GNN Layer별 Tensor Shape
- Structured Readout의 다섯 Embedding
- Graph + Robot State Fusion
- GRU Sequence/Hidden State
- BC Action Forward/Backward Smoke 결과
- 실제 데이터와 Synthetic 테스트 구분
- CPU/GPU 단계별 Latency 및 Memory
- 미구현·미검증 항목과 다음 Ablation 대상

결과 Markdown과 HTML을 상대경로로 양방향 연결한다.

동일 UUID에 JSON/NPZ/PNG, Test Log, Verification, Source Hash, Config, Source Snapshot 및 Implementation Diff를 보존하도록 계획한다.

HTML은 외부 CDN 없이 작동하는 간단한 정적 문서로 작성한다.

Plan-only 단계에서는 새 UUID, 결과 HTML, Forward Output 또는 Benchmark Artifact를 생성하지 않는다.

---

## 17. 금지 및 종료

**이번 요청에서는 계획만 제출한다.**

승인 없이 다음을 실행하지 않는다.

- 신규 Graph/GNN/GRU 코드 구현
- Expert Demonstration 학습
- Hyperparameter Search
- 새로운 Loss 설계
- GNN Architecture 변경 실험
- Vision Sampling/Feature/Edge 정책 변경
- Strict Pixel Validation 복원
- GT 기반 Graph 입력 누출
- 전체 Episode Vision 재처리
- 추가 Simulator/Rollout
- 새로운 GPU Kernel 개발
- Dependency 설치
- 외부 업로드
- Git Commit
- Production Default 변경

기존 dirty/untracked 작업, Task03~05 실험 및 모든 Source Artifact/Sidecar를 보호한다.

### 최종 제출

다음 질문에 답하는 **구현 계획 Markdown만 작성하라.**

1. 현재 Repository에서 Graph→GNN→GRU/BC 중 어디까지 구현되어 있는가?
2. 최소 Working Baseline에 꼭 필요한 변경 파일은 무엇인가?
3. 실제 Robot Geometry와 Proprioception, Expert Action을 어디에서 얻을 수 있는가?
4. 저장 Vision Packet만으로 어떤 Forward/Backward 테스트까지 가능한가?
5. 실제 BC 학습을 시작하기 위해 부족한 데이터·인터페이스는 무엇인가?
6. `L,d,k,M`을 추후 공정하게 실험할 수 있도록 어떻게 구성할 것인가?

**계획 제출 후 구현 승인 전까지 중단하라.**

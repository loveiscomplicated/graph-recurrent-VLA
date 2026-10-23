# Graph-Recurrent VLA — Vision–Graph Interface 설계 v1

**상태:** 주요 설계 합의 완료 / 세부 구현 및 성능 검증 전

## 1. 전체 구조와 역할 분리

현재 시스템은 다음 흐름을 기준으로 설계합니다.

```text
RGB-D + Camera State
          │
          ▼
        Vision
  ├─ Segmentation / Tracking
  ├─ Geometric Feature Extraction
  ├─ Structural / Surface Sampling
  ├─ EE-relative Coordinate Conversion
  └─ Intra-object Edge Construction
          │
          ▼
   Vision Output
  ├─ Point Coordinates
  ├─ Object IDs
  ├─ Geometric Features
  ├─ Intra-object Edges
  └─ Observation Status
          │
          ▼
        Graph ◄── Robot Kinematics
  ├─ Point Nodes
  ├─ TCP / Fingertip Nodes
  ├─ Intra-object Edges
  └─ Inter-object Interaction Edges
          │
          ▼
         GNN
          │
          ▼
    Recurrent (GRU)
          │
          ▼
        Action
```

### Vision의 책임

Vision은 GNN이 실제로 사용할 point node를 완성해 전달합니다.

- RGB-D에서 물체 segmentation 및 tracking
- Structural point와 Surface point 추출
- Point sampling 및 개수 관리
- EE 기준 상대 좌표 변환
- 물체 내부의 연결 관계 계산
- 관측 유효 여부 및 과거 관측 정보 관리

**Vision에서 sampling까지 완료합니다.** Graph 팀이 point cloud를 다시 sampling하거나 내부 연결 관계를 처음부터 재구성하는 것은 기본 설계가 아닙니다.

또한 물체의 전체 3D reconstruction, 크기 추정, 6D pose fitting을 필수 과정으로 요구하지 않습니다.

### Graph의 책임

Graph는 Vision이 전달한 geometry를 로봇 상태와 결합합니다.

- Vision point node 수용
- TCP, Left fingertip, Right fingertip node 추가
- Vision의 물체 내부 edge 사용
- Gripper–Peg, Peg–Box 등 물체 간 interaction edge 생성
- 최종 GNN 입력 Graph 구성

물체 간 edge의 연결 규칙은 Graph 팀이 결정합니다. 고정 연결과 거리 기반 연결 중 무엇이 좋은지는 아직 미결정입니다.

---

## 2. Vision → Graph 데이터 인터페이스

기본적으로 PyTorch Tensor를 사용하며, PyTorch Geometric(PyG)에 의존하지 않습니다.

| 필드 | Shape / Type | 의미 |
|---|---|---|
| `points` | `[N, 3]`, float | EE 기준 point 좌표 |
| `object_id` | `[N]`, int | 각 point의 물체 소속 |
| `geometric_features` | `[N, F]`, float | 기하학적 특징 |
| `intra_object_edge_index` | `[2, E]`, int | 물체 내부 연결 관계 |
| `valid` | 물체별 bool | 현재 관측의 유효 여부 |
| `age_steps` | 물체별 int | 마지막 유효 관측 이후 경과 step |
| `observation_ee_pose` | 물체별 pose | 마지막 유효 관측 당시 EE pose |

여기서 `N`은 **Peg와 Box의 point를 합친 실제 개수**이며, `E`는 Vision이 계산한 물체 내부 edge 수입니다.

`observation_ee_pose`와 현재 EE pose를 이용해 로봇의 상대 움직임 `ΔEE`를 구할 수 있습니다. 어느 모듈에서 계산할지는 구현 시 정하면 됩니다.

### 2.1. Point 좌표

모든 point는 EE 좌표계를 기준으로 표현합니다.

\[
p_i^{EE}=T_{EE\leftarrow W}\,p_i^W
\]

Vision은 카메라 관측을 해당 시점의 EE frame으로 변환해 전달합니다.

Robot node 역시 동일한 좌표계에 있어야 합니다.

### 2.2. Object ID

각 point가 어느 물체에 속하는지 나타냅니다.

예를 들어 다음과 같은 값을 사용할 수 있습니다.

```python
PEG = 0
BOX = 1
```

이는 실제 RGB 색상과 무관한 물체 소속 정보입니다.

### 2.3. Geometric Features

세 가지 방식을 모두 구현하고 config로 선택합니다.

```yaml
geometric_features:
  mode: onehot
  # onehot | probability | continuous
```

| Mode | 표현 | 상태 |
|---|---|---|
| `onehot` | Corner / Edge / Surface 분류 | 기본값 |
| `probability` | 각 유형에 대한 확률 | 비교 후보 |
| `continuous` | Curvature, normal variation 등 | 비교 후보 |

기본값은 `onehot`이지만, GT 대비 기하학적 특징 추출 정확도를 평가한 이후 적절성을 판단합니다.

실제 RGB 색상 및 무늬 정보는 GNN의 기본 node feature에 포함하지 않습니다. 단, Vision이 특징 후보를 검출하는 단계에서 RGB를 보조적으로 사용하는 것은 허용합니다.

### 2.4. Intra-object Edge

물체 내부 연결은 Vision이 계산해 전달합니다.

```python
intra_object_edge_index.shape == (2, E)
```

각 index는 Peg와 Box를 합친 `points` 배열을 기준으로 합니다.

예를 들어 Peg point가 40개, Box point가 100개라면 전체 index는 `0~139`입니다. Vision에서 계산한 내부 edge도 이 index를 그대로 사용합니다.

기본 연결은 k-NN을 고려하되, RGB-D의 표면 불연속성 정보를 이용해 부적절한 연결을 제한합니다.

**물체 내부 edge는 Vision이 담당하고, 서로 다른 물체 사이의 interaction edge는 Graph가 담당합니다.**

---

## 3. Point Sampling 및 Node Budget

Point node 수는 config에서 조절합니다.

```yaml
pointcloud:
  num_points: 256
  structural_ratio: 0.25
  sampling: fps
```

위 수치는 예시이며 최종값이 아닙니다.

- `num_points`: Peg와 Box를 합친 point node 수의 상한
- `structural_ratio`: Structural point의 최대 할당 비율
- `sampling`: 공간적으로 고른 point sampling 방식

Structural point를 먼저 추출하고 공간적으로 고르게 sampling합니다. 남은 budget은 Surface point로 채웁니다.

Surface point는 FPS(Farthest Point Sampling)를 기본으로 사용합니다.

Peg와 Box에는 각각 최소 node 수를 보장합니다. 단, 실제 관측된 점이 부족하면 해당 물체의 node 수가 최소값보다 적을 수 있습니다.

남은 budget은 두 물체의 관측 geometry에 따라 배분합니다.

### Node 수가 부족한 경우

관측된 point 수가 `num_points`보다 적으면 실제로 존재하는 point만 전달합니다.

- 관측되지 않은 point를 생성하지 않음
- 동일한 point를 복제하지 않음
- 기본적으로 padding하지 않음
- Graph는 가변 node 수를 허용

따라서 `num_points=256`일 때 최대 256개의 point node가 생성되며, 여기에 Robot node 3개가 추가됩니다.

**전체 Graph의 최대 node 수는 259개입니다.**

---

## 4. Robot Node

Robot node는 Graph 모듈에서 로봇 kinematics를 이용해 생성합니다.

현재 구성은 다음 세 개로 확정합니다.

1. TCP
2. Left fingertip
3. Right fingertip

Robot node는 Vision의 point budget에 포함되지 않습니다.

Robot node와 Vision point node는 같은 EE 좌표계를 사용합니다.

추가적인 joint 또는 link node는 현재 설계에 포함하지 않습니다.

---

## 5. 관측 실패 및 Recurrent Memory 대응

이 부분은 전체 인터페이스에서 특히 중요한 설계 결정입니다.

**물체가 완전히 가려져 관측이 실패하더라도 마지막으로 유효했던 geometry를 유지하고, 해당 정보가 오래된 관측임을 명시적으로 전달합니다.**

목적은 Recurrent Controller가 과거 geometry와 로봇의 움직임을 이용해 관측 소실 상황에 대응할 수 있도록 하는 것입니다.

### 5.1. 정상 관측

Peg와 Box가 정상적으로 관측될 때는 매 step 새로운 point node와 intra-object edge를 생성합니다.

```text
valid = True
age_steps = 0
```

Graph는 최신 geometry를 사용해 interaction edge를 구성합니다.

### 5.2. 물체가 완전히 가려진 경우

예를 들어 Peg가 로봇 손에 가려져 추적이 끊기면 다음과 같이 처리합니다.

| 데이터 | 처리 방식 |
|---|---|
| Peg point coordinates | 마지막 유효 관측값 유지 |
| Peg intra-object edges | 마지막 유효 연결 유지 |
| Peg `valid` | `False` |
| Peg `age_steps` | 매 step 증가 |
| Peg observation EE pose | 마지막 관측 당시 값 유지 |
| Box geometry | 정상 관측된다면 계속 갱신 |
| TCP / Fingertips | 현재 로봇 상태로 계속 갱신 |

여기서 중요한 것은 **마지막 관측의 EE 기준 좌표를 현재 좌표인 것처럼 수정하지 않는다는 점**입니다.

가령 Peg가 5 step 전에 관측됐다면, 그 point들은 5 step 전의 EE frame에서 계산된 좌표입니다.

현재 EE pose와 마지막 관측 당시 EE pose를 함께 사용하면 로봇이 그동안 얼마나 움직였는지 알 수 있습니다.

하지만 Peg 자체가 함께 움직였는지, 손 안에서 미끄러졌는지까지 Vision이 임의로 추정하지는 않습니다.

### 5.3. Stale Graph의 Edge 처리

관측이 끊긴 물체의 Graph는 다음 규칙을 따릅니다.

**물체 내부 edge**

마지막으로 유효했던 연결 관계를 유지합니다.

이는 이전에 관측한 물체 내부 geometry를 계속 전달하기 위한 것입니다.

**물체 간 edge**

Stale point를 이용해 새로운 거리 기반 edge를 생성하지 않습니다.

예를 들어 Peg는 5 step 전 좌표이고 Box는 현재 좌표라면, 두 좌표를 직접 비교해 새로운 Peg–Box 근접 관계를 판단해서는 안 됩니다.

고정형 interaction edge를 채택할 경우의 처리 방식은 별도로 결정해야 합니다.

### 5.4. 관측이 복구된 경우

새로운 관측이 유효해지면 해당 물체의 point cloud와 내부 edge를 갱신합니다.

```text
valid = True
age_steps = 0
```

새로운 Graph는 다시 현재 EE 좌표계의 geometry를 사용합니다.

### 5.5. Recurrent Controller의 역할

GRU는 다음 정보를 활용할 수 있습니다.

- 이전 시점의 Graph embedding
- 현재 유효한 geometry
- 마지막 관측 geometry
- `valid`, `age_steps`
- 관측 이후 로봇의 움직임

이를 통해 관측이 일시적으로 끊겨도 이전 정보를 활용해 action을 이어가는 능력을 학습하는 것이 목표입니다.

다만 **이 정보들을 전달한다고 GRU가 자동으로 관측 소실에 강해지는 것은 아닙니다.**

정상 관측뿐 아니라 occlusion과 observation dropout이 포함된 조건에서 학습 및 평가가 필요합니다.

---

## 6. Vision–Graph–Recurrent 경계에서 주의할 점

### 6.1. Stale geometry와 현재 geometry의 구분

Vision이 전달하는 Graph에는 관측 시점이 서로 다른 물체가 함께 존재할 수 있습니다.

따라서 Graph와 Recurrent 모듈은 다음 정보를 구분할 수 있어야 합니다.

- 현재 관측한 geometry
- 마지막으로 관측했던 geometry
- 마지막 관측 이후 로봇이 움직인 정도

이 정보가 없다면 과거 geometry를 현재 위치로 잘못 해석할 수 있습니다.

### 6.2. Stale node와 현재 Robot node의 혼합

Robot node는 항상 현재 상태로 갱신하지만, stale point는 과거 좌표를 유지합니다.

따라서 stale point와 현재 robot node 사이에서 새롭게 계산한 거리 역시 현재의 실제 물리적 거리를 의미하지 않을 수 있습니다.

이 관계를 사용하는 경우 관측 시점 차이를 반드시 고려해야 합니다.

### 6.3. Geometry와 Recurrent Memory의 책임

Vision은 관측되지 않은 물체의 현재 pose를 예측하거나 완전한 형상을 복원할 필요가 없습니다.

과거 정보를 바탕으로 다음 action을 결정하는 역할은 Recurrent Controller에 둡니다.

단, GRU의 hidden state가 실제로 object state를 기억하거나 추적하는지는 검증 대상입니다.

---

## 7. 남은 구현 및 검증 사항

주요 인터페이스 구조는 합의됐지만, 다음 사항은 구현과 실험을 통해 결정해야 합니다.

| 항목 | 현재 상태 |
|---|---|
| 전체 point 수 | Configurable, 최적값 미정 |
| Structural point 비율 | Configurable, 최적값 미정 |
| 물체별 최소 node 수 | 원칙 합의, 구체적 값 미정 |
| Geometric feature | 3가지 모드 구현, 기본 one-hot |
| Intra-object k-NN | 기본 후보, k값 미정 |
| RGB-D 경계 기반 edge filtering | 설계 합의, 실제 정확도 미검증 |
| 물체 간 edge | Graph 담당, 연결 규칙 미정 |
| Stale geometry 유지 | 합의 완료 |
| Stale intra-object edge 유지 | 합의 완료 |
| Stale node의 신규 거리 기반 edge 제한 | 합의 완료 |
| `valid`, `age_steps`, `ΔEE` 활용 | 합의 완료, Recurrent 성능 미검증 |
| 부분 관측 시 갱신 기준 | 추가 구체화 필요 |

실제 구현에서는 `valid=False`일 때 새로운 geometry를 만들지 않고 마지막 관측값을 유지해야 합니다. 또한 물체별 point 수가 변화하므로, 내부 edge index가 최종 concatenated point 배열을 정확히 참조하는지도 확인해야 합니다.

---

## 8. 최종 합의

현재 Vision–Graph 인터페이스는 다음 원칙으로 정리됩니다.

**Vision**

관측된 geometry를 바탕으로 Structural/Surface point를 선택하고, 물체 내부의 기하학적 연결까지 계산합니다. 관측이 실패하면 마지막으로 유효했던 geometry와 관측 상태 정보를 전달합니다.

**Graph**

Vision의 point node와 내부 edge를 받아 TCP 및 양쪽 fingertip을 추가하고, 물체 간 interaction edge를 구성합니다. 오래된 관측과 최신 관측을 혼동하지 않도록 처리합니다.

**Recurrent**

현재 Graph 정보와 과거 memory, 관측 유효 여부, 로봇 움직임을 활용해 부분 관측과 일시적인 관측 소실에 대응하도록 학습합니다.

이 인터페이스의 목적은 Vision에서 완전한 물체 상태를 복원하는 것이 아니라, **부분 관측된 geometry를 GNN+GRU가 활용할 수 있는 형태로 일관되게 전달하는 것**입니다.

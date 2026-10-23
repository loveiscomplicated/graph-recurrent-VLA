# Graph-Recurrent VLA: Point Cloud Representation 개선 방향

## 1. 현재 Point Cloud 방식의 문제

현재 Vision 모듈은 SAM2로 물체를 segmentation한 뒤, RGB-D에서 관측된 표면을 point cloud로 변환하고 고정된 수의 점을 sampling합니다.

이 방식은 물체의 표면 정보를 추출할 수 있다는 장점이 있지만, **GNN이 manipulation에 필요한 geometry를 충분히 활용할 수 있도록 구성되어 있는지는 의문**입니다.

### 1.1. Random sampling으로 구조적 정보가 불안정함

현재 point cloud는 매 프레임 sampling되는 위치가 달라집니다. GNN은 노드 순서 변화 자체에는 대응할 수 있지만, 꼭짓점이나 모서리처럼 중요한 기하학적 특징이 sampling 과정에서 누락될 수 있습니다.

또한 Box의 넓은 평면에 많은 점이 배정되는 반면, 실제 삽입에 중요한 hole 경계는 충분히 표현되지 않을 수 있습니다.

### 1.2. Point 수에 비해 정보가 중복될 가능성

기존 방식은 물체별 1,024개 point를 사용합니다.

하지만 단순한 평면이나 직육면체를 표현하는 데 이만큼의 point가 반드시 필요한지는 확인되지 않았습니다. 불필요한 node와 edge는 GNN의 연산 비용을 증가시킬 수 있습니다.

**Point 수 자체보다 task-relevant geometry를 얼마나 보존하는지가 중요합니다.**

### 1.3. 표면의 연결 관계가 고려되지 않음

3D 공간에서 가까운 두 점이 반드시 실제 물체 표면에서도 인접한 것은 아닙니다.

예를 들어 컵 손잡이의 구멍을 사이에 둔 두 점은 유클리드 거리가 가까워도 표면상으로는 직접 연결되지 않습니다. 단순 k-NN을 사용하면 이런 점들 사이에 부적절한 edge가 생길 수 있습니다.

다양한 물체 형상으로 확장하려면 관측된 표면의 불연속성도 고려할 필요가 있습니다.

### 1.4. 현재 관측된 표면과 전체 물체 형상을 구분해야 함

RGB-D에서 생성한 point cloud는 현재 카메라에 보이는 표면만 포함합니다. Grasp 이후에는 로봇 손에 의해 상당 부분이 가려질 수 있습니다.

이 자체는 잘못이 아닙니다. 우리가 원하는 것은 완전한 3D reconstruction이 아니라 **manipulation에 도움이 되는 부분 관측 geometry**입니다.

따라서 가려진 형상을 억지로 복원하기보다, 관측된 geometry를 잘 보존하고 시간적 정보의 활용은 GNN+GRU에서 학습하는 방향을 택합니다.

---

## 2. 합의한 개선 설계

### 2.1. Vision의 역할

Vision은 RGB-D에서 물체의 실제 관측 geometry를 추출하고 Graph 구성에 필요한 정보를 전달합니다.

반드시 물체의 6D pose, 크기 또는 전체 형상을 추정할 필요는 없습니다.

기본적으로 다음 정보를 사용합니다.

- Peg와 Box의 관측 point cloud
- 기하학적 특징점 및 특징 정보
- 물체 소속 정보(Peg / Box)
- RGB-D 기반 표면 경계 및 불연속성 정보
- 로봇의 TCP, left/right fingertip 위치

모든 위치 정보는 EE 기준 좌표계로 통일합니다. TCP와 fingertip 위치는 Vision에서 추정하는 것이 아니라 로봇의 kinematics로 얻습니다.

기존 E4의 pose fitting은 실제 Graph 입력의 필수 단계에서는 제외하되, 비교 실험과 분석용으로 유지할 수 있습니다.

### 2.2. Structural Point + Surface Point

모든 Graph node는 기본적으로 3D point로 표현합니다.

다만 point의 역할을 구분합니다.

| 종류 | 의미 |
|---|---|
| Structural Point | 꼭짓점, 모서리, 기하학적으로 특징적인 위치 |
| Surface Point | 나머지 관측 표면을 표현하는 점 |
| Robot Node | TCP, Left fingertip, Right fingertip |

Structural point는 특정 물체의 알려진 형상이나 GT pose를 이용해 생성하지 않습니다. **실제로 관측된 point 중에서 기하학적으로 특징적인 점을 선택**합니다.

Peg에서는 관측 가능한 꼭짓점과 모서리를, Box에서는 특히 hole 입구의 경계를 보존하는 것이 목적입니다.

또한 다음 원칙을 적용합니다.

- 관측되지 않은 꼭짓점은 추정해서 채우지 않습니다.
- 같은 종류의 Structural point는 homogeneous node로 취급합니다.
- 동일한 물리적 꼭짓점에 고유 ID를 부여하거나 프레임마다 추적할 필요는 없습니다.
- 평면과 연속적인 곡면은 모두 Surface로 취급합니다.
- Edge와 Corner는 단순히 곡률이 높은 점이 아니라 실제 기하학적 불연속성을 중심으로 정의합니다.

따라서 다양한 물체 형상에서도 동일한 표현을 사용할 여지를 남겨둡니다.

### 2.3. RGB-D 기반 기하학적 특징 추출

우선 활용할 라이브러리 후보는 PCL의 **`OrganizedEdgeFromRGBNormals`**입니다.

초기 실험은 geometry 중심 표현을 검증합니다. RGB 이미지는 모서리나 경계 후보를 찾는 데 보조적으로 쓸 수 있지만, 색상이나 무늬만으로 실제 기하학적 edge라고 단정하지 않습니다. 첫 geometry 중심 기준선의 결과를 얻은 뒤 RGB 색상·무늬 feature를 추가하는 비교도 열어둡니다.

경계는 다음 두 종류로 구분합니다.

| 경계 종류 | 활용 방식 |
|---|---|
| 실제 기하학적 경계 | Structural point 후보 선정 |
| 가림 또는 관측 중단으로 생긴 경계 | 잘못된 Graph edge 연결 방지 |

특히 로봇 팔에 의해 가려지는 영역에서 발생하는 관측 경계를 실제 물체의 모서리로 잘못 해석하지 않도록 해야 합니다.

동일한 RGB-D 경계 검출 결과를 **노드 추출과 엣지 구성에 재사용**하는 것이 목표입니다.

다만 해당 PCL 라이브러리가 Structural node 선정이나 Graph edge filtering까지 직접 수행하는 것은 아닙니다. 이 부분은 추가 구현과 검증이 필요합니다.

### 2.4. Node Sampling

전체 point node 수는 config에서 조절할 수 있도록 합니다.

```yaml
pointcloud:
  num_points: 256
  structural_ratio: 0.25
  sampling: fps
```

위 숫자는 예시이며 최종값은 아닙니다.

`num_points`는 **Peg와 Box를 합친 point node 수의 상한**입니다.

`structural_ratio`는 Structural point가 사용할 수 있는 최대 비율입니다. 실제 검출된 Structural point가 적다면 남은 자리는 Surface point로 채웁니다.

Sampling 방식은 다음과 같습니다.

1. 관측된 geometry에서 Structural point 후보를 추출합니다.
2. Structural point를 공간적으로 고르게 sampling합니다.
3. 나머지 node budget을 Surface point로 채웁니다.
4. Surface point도 FPS(Farthest Point Sampling)를 이용해 공간적으로 고르게 선택합니다.

또한 Peg와 Box에 각각 최소 node 수를 보장하고, 남은 budget은 두 물체의 geometry에 따라 배분합니다.

최소 node 수 역시 충분한 관측점이 존재할 때만 보장합니다. 실제로 관측되지 않은 point를 임의로 생성하지 않습니다.

### 2.5. 가변 Node 수와 Padding

`num_points`는 실제 node 수를 강제로 맞추는 값이 아니라 상한입니다.

관측점이 부족하면 더 적은 node를 사용합니다.

기본적으로 padding을 사용하지 않고 가변 node 수를 허용합니다. GNN 구현이나 GPU 최적화 과정에서 필요한 경우 padding을 추가로 고려할 수 있습니다.

TCP와 left/right fingertip은 point budget에 포함되지 않으며, 항상 별도의 Robot node로 존재합니다.

따라서 `num_points=256`이라면 최대 Graph node 수는 259개입니다.

### 2.6. Graph Edge 구성

Vision은 각 point의 물체 소속 정보와 기하학적 경계 정보를 전달합니다.

Graph 모듈은 이를 활용해 node 간 연결을 구성합니다.

현재 기본 후보는 다음과 같습니다.

- **물체 내부:** k-NN을 기본으로 하되 RGB-D 표면 불연속성을 고려해 부적절한 연결을 제한
- **물체 간:** 우선 Gripper–Peg, Peg–Box 관계를 고려

물체 간 edge의 구체적인 연결 방식은 Graph 모듈에서 결정합니다.

고정적으로 연결할지, 거리 조건에 따라 동적으로 연결할지는 아직 정하지 않았습니다. 동일한 학습 조건에서 비교해야 합니다.

MeshGraphNets의 mesh edge와 proximity edge 구분은 참고할 만하지만, 현재는 복잡한 mesh topology를 복원하기보다 **잘못된 공간적 연결을 제한하는 가벼운 방식**을 먼저 검토합니다.

---

## 3. 실험을 통해 결정할 사항

여기까지는 설계 방향이고, 실제 성능이 검증된 것은 아닙니다.

특히 다음 사항은 아직 확정하지 않습니다.

| 미결정 사항 | 비교 후보 | 판단 기준 |
|---|---|---|
| Node type 표현 | One-hot / 분류 확률 / 연속적인 기하 feature | GT 대비 분류 정확도, 위치 오차, score calibration |
| Point 수 | 64 / 128 / 256 / 512 등 | Geometry 보존, GNN 성능, 연산량 |
| Structural 비율 | Config별 비교 | 중요한 특징 보존율과 downstream 성능 |
| 특징 검출 알고리즘 | RGB-D 경계, normal, 국소 연속성 등 | Noisy RGB-D에서의 정확도와 안정성 |
| 물체 간 edge | 고정 연결 / 거리 기반 연결 | 동일한 GNN 조건에서의 성능 |
| Structural node 유지 효과 | Structural + Surface / 일반 Surface | 실제 manipulation 성능 |

특히 Node type 표현은 중요한 미결정 사항입니다.

One-hot으로 단순하게 `corner / edge / surface`를 나타낼 수도 있고, 각 유형에 대한 확률을 전달할 수도 있습니다. 또는 normal variation, curvature, surface discontinuity처럼 연속적인 기하 feature 자체를 전달할 수도 있습니다.

분류 정확도가 낮다고 연속 score가 반드시 더 좋은 것은 아닙니다. 해당 score가 실제 불확실성을 얼마나 잘 반영하는지도 확인해야 합니다.

**따라서 특징 추출 정확도를 먼저 평가하고 Node type 표현을 선택합니다.**

---

## 4. 검증 방향

검증은 크게 두 단계로 나눕니다.

### Geometry 추출 검증

우선 ManiSkill의 GT geometry를 기준으로 실제 RGB-D에서 기하학적 특징을 얼마나 잘 추출하는지 확인합니다.

주요 평가 대상은 다음과 같습니다.

- GT의 실제 corner·edge와 검출된 point 사이의 거리
- Corner / Edge / Surface 분류 정확도
- 가림 경계를 실제 모서리로 잘못 검출하는 비율
- Sampling 이후 중요한 geometry가 얼마나 보존되는지
- RGB-D 표면 경계 정보를 이용한 edge filtering의 정확도
- Point cloud 생성 및 특징 추출에 소요되는 시간

여기서 GT 전체 형상을 기준으로 보이지 않는 꼭짓점까지 검출하도록 요구해서는 안 됩니다. **평가는 현재 관측 가능한 geometry를 기준으로 해야 합니다.**

### GNN 기반 검증

Geometry 자체의 품질을 확인한 다음에는 실제 Graph representation이 제어에 도움이 되는지 검증합니다.

현재 E5의 rule-based policy는 명시적인 peg/hole pose를 사용하므로, 새로운 point cloud representation의 효과를 직접 평가하기에는 적절하지 않습니다.

따라서 기존 rule-based 결과는 참고 baseline으로 유지하고, 최종적인 판단에는 **동일한 조건에서 학습한 GNN 기반 BC policy의 성능**을 사용합니다.

우선 기존 point cloud 방식과 개선된 Structural + Surface 방식을 비교합니다.

평가에서는 action imitation error뿐 아니라 실제 closed-loop rollout 성공률, grasp 및 alignment 성능, occlusion 대응, inference latency 등을 확인합니다.

이후 필요할 경우 GRU의 temporal memory가 partial observation을 얼마나 보완하는지도 별도로 검증합니다.

또한 E6에서 사용한 GT 기반 `grasped` 판단과 rule-based relevance score는 전체 Graph-Recurrent VLA에 반드시 전달해야 하는 정보로 취급하지 않습니다. 실제 Graph 입력의 일부로 채택할지 여부는 별개의 문제입니다.

---

## 5. 현재 설계의 중심 방향

우리가 만들려는 것은 완전한 물체 형상을 복원하는 Vision 모듈이 아닙니다.

**RGB-D에서 관측 가능한 geometry를 추출하되, manipulation에 중요한 기하학적 특징을 우선 보존하는 Point Cloud Representation**입니다.

Vision에서 모든 정보를 명시적으로 계산해 제공하기보다는, GNN이 구조적 특징과 표면 정보, 로봇과 물체 사이의 관계를 활용하도록 합니다.

따라서 현재 개선 방향은 다음과 같이 요약할 수 있습니다.

**기존 방식**

RGB-D → Segmentation → Point Cloud → Random Sampling → Graph

**개선 방식**

RGB-D → Segmentation + Geometric Feature Extraction → Structural Point 우선 배정 + Surface FPS → Graph

Graph 구성 과정에서는 물체 소속 및 RGB-D 경계 정보를 활용해 부적절한 연결을 제한합니다.

이 설계가 실제로 더 좋은지는 아직 검증되지 않았습니다. 또한 구조적 특징이 명시적으로 제공되지 않아도 GNN이 기존 point cloud로 충분한 정보를 학습할 가능성은 남아 있습니다.

따라서 최종 목표는 더 복잡한 Vision 모듈을 만드는 것이 아니라, **가능한 한 간결한 geometry representation으로 manipulation 성능을 높일 수 있는지 확인하는 것**입니다.

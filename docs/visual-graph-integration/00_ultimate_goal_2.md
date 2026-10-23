# 6. 설계 의도 및 구현 시 추가로 알아야 할 사항

## 6.1. 우리가 만들려는 것은 무엇인가?

이번 개선의 목적은 point cloud에서 더 정확한 물체 pose를 추정하는 것이 아닙니다.

**GNN이 물체의 기하학적 구조와 로봇–물체 간 관계를 학습하기에 적합한 입력을 만드는 것**입니다.

기존 Vision v1에서는 point cloud를 이용해 peg와 hole의 pose를 추정하고, 이 pose를 rule-based controller에 전달했습니다. 하지만 Graph-Recurrent VLA에서는 반드시 이 과정을 거칠 필요가 없습니다.

우리가 생각하는 구조는 다음과 같습니다.

```text
RGB-D
  ↓
Object Detection / Segmentation / Tracking
  ↓
Geometric Feature Extraction
  ↓
Structural + Surface Point Sampling
  ↓
EE-relative Point Cloud
  ↓
Interaction Graph ← Robot State (TCP, Fingertips)
  ↓
GNN
  ↓
GRU
  ↓
Action
```

이 구조에서는 GNN이 partial point cloud로부터 manipulation에 필요한 정보를 직접 추출하도록 학습합니다.

예를 들어 peg의 정확한 중심이나 길이를 명시적으로 계산하지 않더라도, GNN이 peg 표면과 fingertip, hole 주변 표면 사이의 관계를 이용해 적절한 action을 학습할 가능성이 있습니다.

물론 이것이 실제로 가능한지는 아직 검증되지 않았습니다.

**우리가 검증하려는 것은 완전한 3D reconstruction 없이도 관측된 geometry만으로 정밀 manipulation을 수행할 수 있는가입니다.**

## 6.2. RGB와 geometry feature의 초기 검증 방향

첫 통합 실험에서는 색상이나 무늬에 기대지 않고 geometry 정보로 manipulation이 가능한지 확인하기 위해, point node에 EE 기준 3D 좌표와 물체 소속 정보를 제공하고 Structural feature를 검토합니다. Corner, Edge, Surface 구분이나 국소 곡률, normal 변화량은 가능한 geometry 표현 후보이며, 효과는 실험으로 판단합니다.

RGB는 물체 검출·분할과 기하학적 경계 후보 검출에 사용할 수 있습니다. 다만 색상이나 무늬만으로 표면의 실제 기하학적 edge라고 판단하지 않도록 RGB 단서와 3D geometry를 구분해 평가합니다.

| 정보 | 초기 geometry 중심 실험 |
|---|---|
| EE 기준 3D 좌표 | 사용 |
| Peg / Box 소속 정보 | graph 구성에 사용 |
| Corner / Edge / Surface 관련 정보 | 후보로 검토 |
| Curvature, normal variation 등 | 후보로 검토 |
| RGB 색상·무늬 feature | 첫 실험에서는 사용하지 않고, 이후 비교 대상으로 열어둠 |

색상과 무늬가 달라져도 geometry 관계로 manipulation할 수 있는지는 평가로 확인해야 합니다. RGB feature를 추가하는 것이 항상 나쁘다고 전제하지 않으며, geometry 중심 기준선의 결과를 확보한 뒤 필요에 따라 같은 조건에서 비교합니다. 시각화용 물체 색상은 모델 입력 feature와 별개입니다.

## 6.3. Structural node는 고정된 물리적 점이 아니다

Structural node를 도입하는 이유는 매번 동일한 물리적 점을 추적하기 위해서가 아닙니다.

**Random sampling으로 인해 꼭짓점이나 모서리 같은 중요한 geometry가 누락되는 것을 방지하려는 것**입니다.

예를 들어 한 프레임에서 peg의 꼭짓점 A, B, C가 관측되고 다음 프레임에서 B, C, D가 관측됐다고 해보겠습니다.

두 번째 프레임에서 꼭짓점 B를 반드시 이전 프레임과 같은 node index에 배치할 필요는 없습니다.

같은 종류의 Structural node는 homogeneous하게 취급하고, GNN이 노드 순서에 의존하지 않도록 구성합니다.

따라서 다음 원칙을 따릅니다.

- Structural node마다 고유한 물리적 ID를 부여하지 않습니다.
- 이전 프레임과 현재 프레임의 Structural point를 일대일 대응시키지 않습니다.
- 가려진 Structural point의 좌표를 예측하거나 강제로 유지하지 않습니다.
- 현재 관측된 point 중에서 기하학적으로 특징적인 점을 선택합니다.

여기서 주의할 점은 **GNN의 permutation invariance/equivariance와 관측 geometry의 시간적 안정성은 다른 문제**라는 것입니다.

노드 순서가 달라지는 것은 적절한 GNN 구조에서 처리할 수 있지만, 중요한 표면이 가려져 point 자체가 사라지는 문제까지 해결되는 것은 아닙니다.

이런 부분 관측의 시간적 변화는 이후 GRU가 어느 정도 보완할 수 있는지 검증합니다.

별도의 node correspondence나 3D reconstruction은 현재 필수 요구사항이 아닙니다.

## 6.4. 점·선·면이라는 표현의 정확한 의미

현재 설계에서 Corner, Edge, Surface는 서로 다른 기하학적 객체를 Graph node로 만든다는 의미가 아닙니다.

**모든 point node는 3D 좌표를 갖는 점이고, 그 점이 어떤 기하학적 특성을 가지는지 구분하는 것**입니다.

예를 들어 직육면체의 모서리는 하나의 line node로 표현하지 않습니다. 실제 관측된 모서리 위의 점 여러 개를 Structural point로 선택합니다.

곡면 역시 하나의 surface node를 생성하는 것이 아니라, 곡면 위의 관측점들을 surface point로 사용합니다.

기하학적 분류의 기본 의미는 다음과 같습니다.

| 구분 | 의미 |
|---|---|
| Corner | 여러 구조적 경계가 만나는 위치 |
| Edge | 표면 방향이 불연속적으로 변하는 경계 |
| Surface | 평면 또는 연속적으로 변하는 곡면 |

구 형태의 물체는 날카로운 기하학적 경계가 없다면 대부분 Surface로 표현될 수 있습니다.

여기서 곡률이 높다고 반드시 Edge로 판단해서는 안 됩니다. 작은 구의 표면에서도 normal은 빠르게 변하지만 연속적입니다.

따라서 국소적인 normal 변화량뿐 아니라 **표면의 연속성 및 불연속성**을 함께 고려할 필요가 있습니다.

다만 이를 최종적으로 어떤 feature로 표현할지는 아직 결정하지 않았습니다.

1. Corner / Edge / Surface의 one-hot
2. 각 유형에 대한 분류 확률
3. Curvature, normal variation, discontinuity 등 연속적인 기하 feature

이 세 가지는 실험 이후 결정합니다.

특히 세 번째는 점·선·면 분류 자체를 거치지 않고 기하학적 정보를 직접 제공한다는 점에서 앞의 두 방식과 다릅니다.

## 6.5. PCL은 어디까지 활용하려는가?

현재 활용하기로 한 후보는 PCL의 `OrganizedEdgeFromRGBNormals`입니다.

다만 PCL을 이용해 Graph 전체를 자동 생성하려는 것은 아닙니다.

**PCL은 RGB-D에서 기하학적 경계 정보를 얻기 위한 도구이고, 그 결과를 어떻게 sampling하고 Graph로 연결할지는 우리가 결정합니다.**

활용 목적은 두 가지입니다.

**첫째, Structural point 후보 추출**

실제 표면의 불연속성을 나타내는 점들을 찾아 Structural point 후보로 사용합니다.

**둘째, 잘못된 Graph edge 연결 방지**

3D 거리가 가까워도 서로 다른 표면에 속하는 점들을 무조건 연결하지 않도록 경계 정보를 활용합니다.

여기서 특히 가림 경계에 주의해야 합니다.

로봇 팔 때문에 물체의 일부가 사라졌다고 해서 그 경계를 물체의 실제 모서리로 분류해서는 안 됩니다.

반면 해당 경계는 잘못된 point 간 연결을 방지하는 데 활용할 수 있습니다.

PCL의 edge detection 결과는 이 두 용도로 활용할 수 있는 후보 정보입니다. **기하학적 경계와 가림 경계를 실제 환경에서 정확히 분리할 수 있는지, 이를 이용해 잘못된 k-NN edge를 얼마나 제거할 수 있는지는 검증이 필요합니다.**

또한 이 방법만으로 컵 손잡이 같은 물체의 완전한 surface topology를 복원할 수 있는 것은 아닙니다. 현재 목표는 완벽한 mesh connectivity가 아니라 명백히 부적절한 연결을 줄이는 것입니다.

## 6.6. Vision과 Graph가 담당하는 부분

모듈 간 책임은 다음처럼 구분합니다.

| 구성 요소 | 담당 역할 |
|---|---|
| Vision | 관측 geometry, Structural point 후보, 경계 정보, 물체 소속 정보 추출 |
| Robot state / Kinematics | TCP, fingertip 위치 및 자세 제공 |
| Graph | Node feature 구성, node sampling 결과를 이용한 edge 생성 및 연결 제한 |
| GNN | 기하학적 관계를 embedding으로 학습 |
| GRU / Controller | 과거 관측과 현재 관계를 활용해 action 생성 |

물체 소속 정보인 `object_id`는 실제 RGB 색상값과 다릅니다.

예를 들어 Peg와 Box를 구분하기 위한 categorical 정보는 사용할 수 있습니다. 이것은 segmentation 결과로 얻은 물체의 소속 정보이며, 실제 물체가 빨간색인지 파란색인지를 나타내는 정보가 아닙니다.

또한 모든 좌표는 EE frame 기준으로 통일합니다.

다만 실제 구현에서는 EE frame과 TCP frame이 정확히 같은 좌표계인지 확인해야 합니다. 서로 다르다면 TCP node의 위치가 반드시 원점이 되는 것은 아닙니다.

**제외하려는 것은 물체의 명시적인 6D pose fitting이지, 카메라 좌표 변환이나 로봇 kinematics가 아닙니다.**

기존 pose estimator는 평가와 baseline 비교에 사용할 수 있지만 새로운 Graph 경로의 필수 입력으로 요구하지 않습니다.

E6에서 GT를 이용해 판단했던 `grasped` 상태나 rule-based relevance score 역시 전체 Graph-Recurrent VLA의 필수 입력으로 확정된 것이 아닙니다.

## 6.7. Node 개수와 Sampling에서 주의할 점

`num_points`는 Peg와 Box를 합친 point node 수의 **상한**입니다.

TCP, left/right fingertip의 세 Robot node는 별도로 추가합니다.

또한 Structural point는 우선 배정하지만, `structural_ratio`는 반드시 채워야 하는 비율이 아니라 최대 할당량입니다.

예를 들어 Structural point가 충분히 검출되지 않았다면 이를 억지로 생성하지 않고 남은 budget을 Surface point에 배정합니다.

Peg와 Box에 각각 최소 node 수를 보장한다는 원칙도 실제 관측점이 충분할 때만 적용됩니다.

Structural point와 Surface point 모두 공간적으로 고르게 sampling하는 방향으로 정했습니다.

다만 구체적인 node budget, 물체별 최소 node 수, Structural point 비율은 실험으로 선택해야 합니다.

또한 **공간적으로 고르게 sampling하는 것과 시간적으로 같은 point를 유지하는 것은 다릅니다.**

FPS는 전자를 위한 방법입니다. FPS를 사용한다고 point identity가 자동으로 유지되는 것은 아니며, 현재 설계에서 동일 point의 temporal identity는 요구하지 않습니다.

## 6.8. 이번 설계가 아직 보장하지 않는 것

새로운 representation을 사용하더라도 다음 문제가 자동으로 해결되는 것은 아닙니다.

**첫째, 관측되지 않은 geometry는 여전히 알 수 없습니다.**

GNN+GRU가 학습한 prior나 관측 history를 이용해 보완할 가능성은 있지만, 완전히 보이지 않는 물체의 상태를 항상 정확하게 추론할 수 있는 것은 아닙니다.

**둘째, Structural point가 반드시 GNN 성능을 높이는 것은 아닙니다.**

기존 Surface point만으로도 GNN이 필요한 geometry를 충분히 학습할 수 있습니다. Structural point가 도움이 되는지는 실험 결과로 판단해야 합니다.

**셋째, perception 오류는 남습니다.**

SAM2가 손을 Peg로 잘못 segmentation한다면, 잘못된 point cloud가 GNN에 전달될 수 있습니다.

**넷째, 기하학적 구조를 더 잘 표현한다고 정밀 삽입이 자동으로 성공하지는 않습니다.**

마지막 정렬 및 삽입 단계에서는 controller의 오차 보정 능력도 영향을 미칩니다.

따라서 Geometry 추출 성능과 실제 manipulation 성공률을 별도로 측정해야 합니다.

---

# 7. 구현 및 검증을 위한 작업 기준

현재 설계에서 **확정된 사항과 미검증 사항을 혼동하지 않는 것**이 중요합니다.

### 현재 합의된 사항

- Geometry 중심 표현을 먼저 검증하고 RGB 색상·무늬 feature는 후속 비교 대상으로 둠
- Structural + Surface point 방식 사용
- 실제 관측된 point만 Graph에 포함
- Structural point의 고유 identity 추적은 하지 않음
- 전체 point budget과 Structural 비율은 config로 관리
- Structural / Surface 모두 공간적으로 고르게 sampling
- Peg와 Box를 합친 point budget 사용
- Peg와 Box에 각각 최소 node budget 보장
- Padding은 기본적으로 사용하지 않음
- Robot node는 TCP, left/right fingertip을 별도로 유지
- 물체의 명시적 6D pose estimation은 필수 Graph 입력에서 제외

### 아직 검증해야 하는 사항

- PCL 기반 경계 검출이 noisy RGB-D에서 충분히 정확한가?
- 실제 물체 edge와 로봇 팔에 의한 가림 경계를 구분할 수 있는가?
- Structural point를 추가했을 때 중요한 geometry가 더 잘 보존되는가?
- One-hot, 분류 확률, 연속적인 기하 feature 중 무엇이 적절한가?
- Node 수와 Structural point 비율을 얼마나 줄일 수 있는가?
- RGB-D 경계 정보가 잘못된 Graph edge를 실제로 줄이는가?
- 물체 간 edge는 고정 연결과 거리 기반 연결 중 무엇이 좋은가?
- 개선된 point cloud가 GNN의 closed-loop manipulation 성능을 높이는가?
- 추가 경계 검출 및 Graph 생성 비용이 실제 제어 주기에 들어오는가?

### 검증에서 반드시 구분할 것

**Geometry 추출 검증과 GNN 성능 검증은 목적이 다릅니다.**

Geometry 검증은 GT를 이용해 실제 관측에서 추출한 특징이 얼마나 정확한지 확인하는 작업입니다.

GNN 검증은 그 특징이 실제 action 학습과 closed-loop manipulation에 도움이 되는지 확인하는 작업입니다.

특히 GNN에서는 기존 Random Surface와 새로운 Structural + Surface를 비교하되, 가능하다면 **동일한 FPS와 node budget을 사용하고 Structural 정보만 다른 조건**도 함께 비교해야 합니다.

그래야 단순히 random sampling을 FPS로 교체한 효과와 Structural feature의 효과를 구분할 수 있습니다.

기존 E5의 rule-based policy는 명시적 pose를 사용하는 참고 baseline으로 남겨두고, 새로운 Graph representation의 최종 효과는 BC 기반 GNN의 rollout에서 확인합니다.

---

# 8. 팀이 공유해야 할 최종 설계 원칙

이번 개선에서 모든 geometry를 명시적으로 복원하거나 해석하려는 것은 아닙니다.

Vision이 모든 물체의 정확한 pose와 형상을 계산해 전달하는 방식도 아니고, 반대로 아무런 구조 없이 random point만 전달하는 방식도 아닙니다.

**실제로 관측된 geometry에서 manipulation에 유용할 가능성이 높은 구조적 정보를 보존하고, 그 관계를 해석하는 역할은 GNN에 맡기는 것**이 현재 방향입니다.

이 구조에서 Structural point는 GNN을 위한 기하학적 단서이며, 물체의 완전한 형상을 복원했다는 뜻이 아닙니다.

Point sampling, 기하학적 feature, Graph connectivity를 어떻게 구성하는지가 실제 manipulation 성능에 미치는 영향은 앞으로 검증해야 할 연구 질문입니다.

아직 실험으로 확인하지 않은 개선 효과를 설계의 확정된 장점처럼 취급하지 않습니다.

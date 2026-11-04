# Task10-I — 512 관측 정보가 Graph에서 사라지는 구간 특정

상태: **COMPLETE — 단계 분리, 구조 영역, 예산 대조, edge 참조 대조 실행 완료.**

512의 추가 관측 정보는 실제로 존재한다. 현재 경로에서 주요 손실은 **Box의
Shape-first 후보 구성**, **Peg/Box의 최종 FPS128 압축**, **압축된 점만 참조하는
surface-edge witness 검사**에서 발생한다. 이번 자료에서는 로봇/중첩 필터,
교차뷰 검사, Node21 포장이 주 손실 구간이라는 근거가 없다.

512에서만 새로 관측한 공통 GT 표면 표본 중 최종 점이 보존하는 비율은
Peg **36.94%**, Box **32.84%**다. 다만 최종 경계와 Box 입구 rim coverage는128보다
좋다. 전체 coverage 정체를 모든 구조 정보의 개선 부재로 해석하면 안 된다.

## 비교 조건과 지표

[선행 정확도 평가](10i_current_graph_128_512_accuracy_20261104.md)의 실제 Terra/SAM2
예측을 동결하고 CurrentGeometry를 재생했다. 4개 기존 Train Parent
0197/0275/0305/0531, t0..20, native128/512의 **168개 Graph 조건**이다.
초기 약1초 prefix이며, 전체 에피소드·후기 가림 tracking·새 Parent 일반화 평가가 아니다.

두 해상도가 공통으로 사용하는 native1024 현재 관측 GT depth 표본에 대해,
**올바른 actor source를 가진 점이 엄격한4mm 이내에 있는 비율**을 coverage로 쓴다.
Peg는 Main/Wrist, Box는 현재 경로대로 Main만 평가한다. GT mask를512로 늘리지 않는다.
아래 전체 coverage는 동일84개 관측의 평균이다. 원래 실험은 Parent마다21관측으로
동일하므로 Parent-macro와 같다. 물리적 균일 표면적 대신 카메라 pixel density와
중복이 들어간 관측 표면 지표다.

양끝은 Peg local X의 각 바깥10% 길이 구간이다. 물리적 끝면 중심 pose 오차가 아니다.
Box rim은 기존 mesh/aperture reference의 음의X 접근 입구 중 native1024 Main 관측
표면4mm 이내에 실제 support가 있는 표본만 평가하며 **54관측에서 정의**된다.
없는 rim support를 coverage0으로 채우지 않는다. 경계는 GT 투영의1pixel boundary다.

## 단계별 손실량

| 실제 경로 | Peg128 | Peg512 | Box128 | Box512 |
| --- | ---: | ---: | ---: | ---: |
| Raw mask 유효 RGB-D 관측점 | 94.02% | 99.85% | 68.41% | 95.79% |
| 로봇/중첩 필터 후 | 94.02% | 99.85% | 68.41% | 95.79% |
| Wrist C3 gate 후 | 93.64% | 99.84% | 68.41% | 95.79% |
| Camera별 Shape-first 후 | 83.41% | 85.60% | 32.05% | 44.84% |
| 교차뷰 검사 후 | 83.41% | 85.60% | 32.05% | 44.84% |
| 3mm dedup 후 | 80.57% | 80.21% | 32.05% | 44.57% |
| 최종 FPS ≤128/object | 46.45% | 36.54% | 31.41% | 30.22% |

512의 인접 단계 순감소량은 Peg에서 gate→Shape-first **14.24pp**,
dedup **5.39pp**, 최종 FPS **43.67pp**다. Box는 Shape-first **50.95pp**,
dedup **0.27pp**, 최종 FPS **14.35pp**다. 교차뷰 coverage 감소는 두 물체 모두0이다.
이는 해당 frozen prefix에 대한 관측이며 다른 상태에서도 필터 손실이0이라는 보장은 아니다.
단계 간 lost/recovered 표본을 별도로 기록했다. canonical float64 재구성과 packet
float32의 차이로 Shape-first에서 극소수의 reference recovery가 있어, net loss와
제거된 point 수를 동일시하지 않는다.

![단계별 전체·추가 관측·구조 영역 보존](../../outputs/task10i/current_graph_128_512_20261104_v1/information_loss_localization.png)

512에서만 얻은 표본은 `raw512 covered AND NOT raw128 covered`로 고정한다.
각 단계에서 그 집합의 보존율을 측정했다. 이는 전체 coverage와 다른 조건부 분모다.
Peg84관측, Box82관측에서 추가 표본이 있으며, 분모0 관측은 undefined로 남긴다.

| 512 추가 관측 표본 보존율 | Peg | Box |
| --- | ---: | ---: |
| Gate 후 | 99.91% | 100.00% |
| Shape-first 후 | 79.70% | 50.41% |
| Dedup 후 | 76.45% | 50.07% |
| FPS128 후 | 36.94% | 32.84% |
| 같은 pool의 FPS256 대조 | 66.74% | 47.61% |

## 손실 구간1: Shape-first의 후보 영역과 고정 surface 수

[shape_first_cuda](../../visual/runtime/cuda_sampling.py)는 PCA orientation/silhouette
후보 C를 voxel 대표 R로 압축하고 절반 간격 thinning 후 priority P를 만든다.
그 뒤 surface FPS로 Peg97/Box95점을 추가한다. 앞단 출력 N은 가변이지만 넓은
표면을 보충하는 surface 수는 해상도와 관계없이 고정이다.

기존 실제 surface source S를 고정한 상태에서 C+S, R+S, P+S를 비교했다.
P+S는 실제 packet source 집합과 정확히 같다. 새 sampling 정책을 독립 재실행한
실험이 아니며 point 수가 다른 진단 집합이다.

| 512 Shape-first 분기 대조 | Peg 전체 | Box 전체 | Box 관측 rim |
| --- | ---: | ---: | ---: |
| Gate dense | 99.84% | 95.79% | 97.22% |
| C + 고정 S | 86.85% | 47.53% | 92.32% |
| R + 고정 S | 86.62% | 46.92% | 92.05% |
| P + 고정 S = 실제 선택 | 85.60% | 44.84% | 84.83% |

Box 전체 표면의 큰 감소는 voxel/thinning만으로 설명되지 않는다.
**Dense interior를 후보 영역과 고정95개 surface로 대표하는 시점**에 이미
95.79→47.53%로 떨어진다. Voxel/thinning의 추가 감소는47.53→44.84%다.
반면 입구 rim에는 thinning 전후92.05→84.83%의 추가 손실이 있어 구조 영역별
확인도 필요하다. C/R/P의 의미와 손실을 단순 합산해 독립 원인의 비중으로 해석하지 않는다.

## 손실 구간2: 최종 FPS가 앞단 priority와 구조 영역을 보장하지 않음

[CurrentGeometry](../../graph/plan_a_bc/geometry.py)는 가변 pool 뒤에
`budget_tensor(records,128)`을 호출한다.
[budget_tensor](../../graph/persistent_geometry/tensor_records.py)는 XYZ의 FPS와
source identity 동점 규칙을 사용하며 **structural/priority anchor를 보존하지 않는다**.
앞단에서 P로 지정했어도 최종 점으로 남는 보장은 없다.

| 512 dedup pool → 최종 점, 관측당 평균 | Priority P | Surface S |
| --- | ---: | ---: |
| Peg | 239.54 → 62.73 | 157.79 → 65.27 |
| Box | 166.10 → 58.83 | 90.30 → 64.05 |
| Peg 전체 pooled 잔존율 | 26.19% | 41.37% |
| Box 전체 pooled 잔존율 | 35.42% | 70.93% |

이 차이는 P가 모두 올바른 GT 구조점이라는 뜻은 아니다. 다만 **앞단 priority라는
선택 의도가 최종 선택에서 유지되지 않는다**는 직접적인 source 추적 결과다.

| 512 구조 영역 coverage | Gate | Shape-first | Dedup | FPS128 |
| --- | ---: | ---: | ---: | ---: |
| Peg −X 끝 구간 | 99.96% | 83.50% | 78.06% | 37.59% |
| Peg +X 끝 구간 | 97.27% | 84.25% | 80.24% | 36.06% |
| Box 관측 입구 rim ·54관측 | 97.22% | 84.83% | 80.92% | 23.50% |

두 끝과 rim 모두 최종 압축의 손실이 크다. Box rim은80.92→23.50%, **57.43pp** 감소한다.
512 Peg의 camera별 평균 대표점은 dedup Main263.33/Wrist133.99에서 최종
Main88.68/Wrist39.32로 바뀐다.128 최종 Main72.85/Wrist55.15와도 다르다.
현재 FPS에는 view quota가 없으므로 해상도 변경이 최종 view 배분도 바꾼다.

### 예산만 늘리면 회복되는가

같은 dedup pool과 기존 FPS를 재사용한 **point-set 대조**다. Graph builder/validator의
128 초과 계약을 변경하거나 새 Graph interface를 검증한 결과가 아니다.
Matched N은 물체·물리 프레임마다 두 pool 수의 최솟값으로 GT 없이 정했다.

| 512 point-set 대조 | Peg 평균 N / 전체 coverage | Box 평균 N / 전체 coverage | Box rim |
| --- | ---: | ---: | ---: |
| 현재 FPS128 | 128.0 / 36.54% | 122.9 / 30.22% | 23.50% |
| FPS256 | 256.0 / 68.74% | 221.2 / 42.57% | 59.78% |
| Matched N | 279.5 / 71.66% | 115.9 / 24.57% | 23.84% |
| Native variable N ·dedup 전부 | 397.3 / 80.21% | 256.4 / 44.57% | 80.92% |
| GT-correct pool만 FPS128 ·privileged | 128.0 / 38.06% | 122.5 / 31.88% | 24.95% |

128의 Matched N coverage는 Peg80.57%, Box32.05%다. 같은 N에서도512의 전체
coverage가 낮으므로 단순한 점 수 부족만으로 해상도 격차가 설명되지는 않는다.
후보 구성·view 배분·geometry FPS 목적과 관측 coverage 목적의 차이가 함께 작용한다.
FP 제거만으로도 회복되지 않는다. 앞선 보고서의 **희소 배경 오점 확대** 근거는
유효하지만 전부를 설명하는 원인은 아니다. GT-filter 대조는 배포 가능한 방법이 아니다.

## 손실 구간3: 최종 희소점만으로 edge witness를 검사함

[surface_edges_tensor](../../graph/persistent_geometry/tensor_graph.py)는 kNN 후보 중
20mm 이내 edge를 고르고 normal 규칙을 적용한다. 이후 edge의25/50/75% 위치가
모두 같은 물체 점의4mm 이내에 있어야 한다. **이 참조점은 압축 전 관측점이 아니라
최종 Graph의 희소 XYZ다.** Depth free-space 검사는 그 뒤 별도로 수행한다.

512 Peg에서 관측당 평균 후보429.46개 중 normal로88.83개, witness로329.83개,
free-space로0.38개가 제거되고 무방향10.42개가 남는다. Normal 통과 후보의 약96.83%가
witness에서 제거된다. 따라서 최종 FPS 압축이 node뿐 아니라 연속 표면 판단의
참조 정보까지 줄일 가능성을 대조 실험했다.

기존 함수 AST를 재사용해 **witness 거리의 참조 XYZ/object 두 읽기만** 바꿨다.
최종 node/normal/validity/kNN/threshold/depth free-space는 고정하고, witness에서만
동일 시각의 pre-FPS dedup pool을 참조한다. GT 없이 수행하며 production 코드는
변경하지 않았다. 참조를 원래 최종 점으로 주면336개 물체 조건 모두 edge가 정확히
일치하고, pool 참조에서는 모든 기존 edge가 유지되는지 검사했다.

| 해상도·물체 | 원래 무방향 intra edge | Pool 참조 edge | 원래 intra 고립 node | Pool 참조 고립 node |
| --- | ---: | ---: | ---: | ---: |
| 128 Peg | 64.18 | 199.04 | 78.63 /128 | 23.73 /128 |
| 512 Peg | 10.42 | 150.88 | 117.19 /128 | 26.73 /128 |
| 128 Box | 7.05 | 7.05 | 108.01 /111.01 | 108.01 /111.01 |
| 512 Box | 65.38 | 91.24 | 102.19 /122.88 | 80.36 /122.88 |

512 Peg의 witness 참조를 바꾸면 edge가약14.5배 늘고 intra 고립점이117→27개로
줄어든다. **연결 판단에 필요한 현재 관측 context를 압축하면서 연결선도 대량 제거하는
구간**임을 분리 확인했다. 두 해상도 모두 영향을 받으므로 이 변경만으로512가128보다
좋아졌다는 결론은 아니다. GT edge topology가 없고 pool에도 오점이 있어 늘어난
edge의 정답률이나 controller 이득은 확인하지 않았다. Robot/object간 edge도 존재하므로
intra 고립을 전체 Graph에서 완전히 고립됐다는 뜻으로 쓰지 않는다.

## 주 손실 구간에서 제외하거나 범위를 좁힌 항목

- **Cross-view와 dedup을 분리했다.** 이 prefix의 교차뷰 coverage 손실은0이며,
  Peg의5.39pp 손실은 dedup 단계다.512에서 correct source를 포함하지만 wrong source를
  대표로 고른 merge group은 Peg5개·Box5개로 전체 감소를 주도하지 않는다.
- **선택 후 Node21 포장은 추가로 점을 버리지 않았다.** 모든168개 Graph에서 선택된
  XYZ, 유효 F5 값, validity가 정확히 유지되고 추가 node drop은0이다.
- **Normal availability와 정확도를 구분한다.**512 Peg의 valid normal 비율은 dense
  99.97%→dedup98.65%→최종96.72%, Box는99.32→93.44→90.27%다.
  Valid 값이 있다는 검사이며 GT normal 오차 평가가 아니다. Depth-pair validity는100%다.
- **선명한 RGB 자체는 현재 node feature가 아니다.** 현재 Node21에는 XYZ/F5와
  validity 등이 들어가고 RGB appearance 및 priority 역할은 직접 feature로 들어가지
  않는다.512의 이점은 mask/depth/geometry를 거쳐 간접 전달된다. 따라서 포장이 정확해도
  RGB의 모든 세부 정보가 Graph에 전달됐다는 의미는 아니다.

## 전체 coverage에 가려진 실제 개선

| 최종 FPS128 영역 coverage | 128 | 512 |
| --- | ---: | ---: |
| Peg 투영 경계 | 43.65% | 49.50% |
| Box 투영 경계 | 20.94% | 37.78% |
| Box 관측 입구 rim ·동일54관측 | 12.59% | 23.50% |
| Peg −X 끝 구간 | 48.82% | 37.59% |
| Peg +X 끝 구간 | 40.58% | 36.06% |

512는 경계와 Box rim의 최종 보존에 이득이 있지만 Peg 양끝·전체 표면에서는
이득을 충분히 유지하지 못한다. 향후 비교는 전체 coverage와 함께 끝/rim, FP,
intra 연결 보존을 병행해야 한다. 이번 결과로 새 quota·voxel·threshold·예산을
채택하지 않았다. 수정 후보의 우선 검증 대상은 **최종 선택의 구조 보존**, **압축과
edge witness 참조의 분리**, **Box surface 후보 보충**이다.

## 실행·보존·재현 자료

공통 `ExperimentLogger`를 사용한 별도 세 실행이다. 각 실행은 같은168개 조건을
재생한 것이므로504개의 독립 표본으로 세지 않는다.

| 실행 | Run ID | 확인 |
| --- | --- | --- |
| 세부 정보 손실 분리 | `c71c06b602f74bef978811ed4a198038` | 168 Graph의5 tensor 정확 일치, CPU FPS source 일치, Node 포장 일치 |
| Priority/edge funnel | `baa2cf28eb9f4962978435e297e4bc61` | 168 Graph와336 per-object edge 정확 일치 |
| Edge witness 참조 대조 | `0260477cbdaa4c9884819a52df7d2b9c` | 168 Graph와336 same-reference 대조 정확 일치 |

서버58576054의 RTX3090 CUDA, 기존 Torch2.14.1+cu130/Triton 환경에서 실행했다.
입력은 기존 native CPU-rendered 자료를 GPU에 업로드한 것이다. 추가 Terra/API,
SAM2/DINO/GNN forward는0이며 simulator/render/physics와 정책은 실행하지 않았다.
macOS MPS/CUDA도 실행하지 않았다. 공통 logger/publisher interface는 변경하지 않았고
게시·학습·production sampling/Graph 변경은 하지 않았다.

원래 frozen source1,134개 hash 불변, inference receipt SHA256
`217c4d25ff4d7f532cc2dd80d1961075cb46c4957cb45281b01aca9ff9e7f6f2` 불변을
재확인했다. 연구 helper는 기존 snapshot에 별도 추가하고 자체 SHA를 기록했다.
로컬 의미 검증7개 통과, helper compile 및 문서 링크를 확인했다. 읽어온 stage
CSV4,704행에서 요약 평균616개를 독립 재계산해 일치했고168개 포장 증명과336개
edge 참조 대조 증명을 다시 확인했다. 두 결과 archive의 원격/로컬 SHA도 일치했다.

원격 root: `/workspace/task10i_current_graph_128_512_20261104_v1/`.
Reference ledger12개는 원격 `information_loss_v1/reference_ledgers/`에 보존하며
아래 verification에 크기/hash를 기록했다. 로컬 결과 archive에는 대용량 ledger와
실험 logger 원자료 디렉터리를 제외하고 요약·CSV·증명을 읽어왔다.

- [세부 단계·구조 영역·예산 요약](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/summary.json), [전체 stage CSV](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/stage_metrics.csv)
- [인접 단계 lost/recovered](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/transition_summary.json), [512 추가 관측 보존](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/gain_summary.json)
- [재생·Node 포장 일치](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/replay_and_encoding.json), [feature availability](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_v1/feature_validity_summary.json)
- [Priority 잔존·edge funnel](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_edges_v1/summary.json), [CSV](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_edges_v1/structural_and_edges.csv)
- [Edge 참조 대조 결과](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_edge_reference_v1/summary.json), [CSV](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/information_loss_edge_reference_v1/edge_reference_control.csv)
- [동결 source·receipt·ledger 보존 검증](../../outputs/task10i/current_graph_128_512_20261104_v1/information_loss_verification_v1.json), [로컬 artifact hash](../../outputs/task10i/current_graph_128_512_20261104_v1/information_loss_artifact_hashes_v1.json)
- [세부 분석 helper](../../graph/experiments/graph_accuracy_128_512_10i_20261104_v1/localize_information_loss.py), [priority/edge helper](../../graph/experiments/graph_accuracy_128_512_10i_20261104_v1/probe_structural_and_edges.py), [기존 edge 함수 참조 대조](../../graph/experiments/graph_accuracy_128_512_10i_20261104_v1/edge_reference_control.py)

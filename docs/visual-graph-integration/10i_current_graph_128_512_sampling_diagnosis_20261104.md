# Task10-I — 512의 Graph 이득이 사라지는 단계 진단

상태: **COMPLETED — frozen mask의 단계별 관측·수치 대조 완료.**

후속 [정보 손실 위치 특정 보고서](10i_512_information_loss_localization_20261104.md)는
교차뷰/dedup을 분리하고 양끝·rim, priority 잔존율, 예산 대조와 edge witness 참조를
추가 검증했다. 아래 합친 단계의 한계와 미분리 항목은 후속 결과를 함께 읽는다.

[현재 코드의128/512 정확도 파일럿](10i_current_graph_128_512_accuracy_20261104.md)에서
저장한 mask를 재사용했다. 고해상도의 추가 관측 정보는 앞단에 실제로 있지만,
현재 Shape-first·교차뷰 검사·3mm dedup·최종128점 FPS를 거치면서 그 이득이 사라진다.
특히 최종 FPS는 희소한 배경 오점을 상대적으로 많이 남기며, 오점을 없앤 진단
대조군에서도 Peg coverage 격차가 남는다. 따라서 배경 오점 한 가지로 모든 차이를
설명하거나 해상도 자체의 일반적 열등함을 결론내릴 수는 없다.

## 동일 예측의 단계별 재생

4 Parent, t0..20,128/512의168개 Graph를 기존 source_v5와 frozen mask로 재생했다.
관측 wrapper는 입력과 결과를 보존하며 단계별 배열을 채점용으로 읽기만 한다.
모든 프레임에서 저장된 Node21·Node type·Edge index·Edge4·Edge type의5개 tensor가
정확히 일치했다. Detector/API/SAM2/GNN forward는 각각0회이며 CUDA geometry 재생은
168회다. 정책 실행, simulator/render/physics step, 학습과 게시를 수행하지 않았다.

공통 참조는 선행 파일럿과 동일한 native1024 현재 관측 GT depth 표면이다.
올바른 actor source point만4mm credit을 준다. 아래 coverage는84개 물리 관측의
동일 분모 평균이다. Source purity/FP 비율은 전체 point를 합친 pooled 값이므로
선행 보고서의 precision Parent-macro와 분모가 다르다.

## 추가 관측 정보는 앞단에서 확인된다

| 단계 | Peg128 coverage | Peg512 coverage | Box128 coverage | Box512 coverage |
| --- | ---: | ---: | ---: | ---: |
| Raw mask의 유효 관측점 | 94.02% | 99.85% | 68.41% | 95.79% |
| 로봇/중첩 필터·C3 gate 후 전체 관측점 | 93.64% | 99.84% | 68.41% | 95.79% |
| Camera별 Shape-first 선택 후 | 83.41% | 85.60% | 32.05% | 44.84% |
| 교차뷰 검사·3mm dedup 후 pool | 80.57% | 80.21% | 32.05% | 44.57% |
| 최종 FPS ≤128/object | 46.45% | 36.54% | 31.41% | 30.22% |

512의 raw 관측 support는 두 물체 모두 넓다. Shape-first 이후에도512의 coverage가
높지만, Peg는 교차뷰 검사·dedup 후 거의 같은 수준이 되고 마지막128점 압축에서
차이가 커진다. Box는 마지막 압축 전까지 남아 있던512의 이득이 최종 FPS에서 사라진다.
현재 관측 wrapper는 교차뷰 검사와dedup을 합친 전후를 측정했으므로, 두 연산 중
어느 쪽이 개별적으로 얼마나 기여했는지는 이 표에서 분리하지 않는다.

![동일 frozen mask의 단계별 coverage·오점 비율](../../outputs/task10i/current_graph_128_512_20261104_v1/sampling_stage_diagnosis.png)

## 작은 mask 오류가 최종 point에서 큰 비중을 차지한다

| 512 단계 | Peg FP 비율 | Box FP 비율 |
| --- | ---: | ---: |
| 필터·C3 gate 후 유효 mask point | 0.0465% | 0.4858% |
| Shape-first packet | 0.8788% | 5.0822% |
| 교차뷰 검사·3mm dedup pool | 1.7438% | 5.1214% |
| 최종 FPS128 | 4.0923% | 7.9345% |

예를 들어512 Peg의 pool은 정상32,793점·오점582점이다. 최종 FPS에는 정상10,312점과
오점440점이 남는다. 정상점의31.4%를 남길 때 오점은75.6%를 남겼다. 이는 최종
선택에서 오점의 상대 비중이 커졌다는 직접적인 관측이다.

[budget_tensor](../../graph/persistent_geometry/tensor_records.py)의 최종 FPS는
source identity로 초기점·동점을 결정하고, 이미 선택한 점에서 가장 멀리 떨어진
후보를 반복해서 고른다. Actor 의미나 mask의 올바름은 이 단계 입력이 아니다.
흩어진 배경점도 공간적으로 먼 점이면 선택될 수 있고, 같은128점 예산을 소모한다.

실제로 최종512 Peg 오점440개의 관측 GT 표면 최단거리 중앙값은25.26mm,
233개는20mm를 넘는다. Box 오점819개는 중앙값169.39mm이고454개가100mm를 넘는다.
Box 오점의 최대 거리는 관측 GT 표면 기준18.61m다. 한 사례의 Box actor 중심 기준
거리도 약18.7m였다. 이는 보존 depth로 재구성된 배경 point의 거리이며, 배경의 실제
물리 형상이나 signed-depth overflow의 원인을 새로 검증했다는 의미는 아니다.

즉 높은 pixel precision과 높은 최종 point precision은 다른 조건이다. 대부분의
픽셀이 맞아도 일부 배경 픽셀이3D에서 멀리 떨어지면 geometry-only 선택이 그 점을
유지할 수 있다. 최종 오점 증가는 추가 정보의 부족보다 선택 과정의 오점 가중 문제를
함께 보여준다.

## 고해상도에서 sampler가 보는 물리적 척도와 후보 조성이 달라진다

현재 [shape_first_cuda](../../visual/runtime/cuda_sampling.py)는 orientation·silhouette
후보를 voxel에서 추린 뒤 절반 간격으로 남기고, 그 priority에 surface FPS를 추가한다.
Voxel 크기는 depth/focal length에 비례한다. 마지막 Graph 예산은 해상도와 관계없이
물체당 최대128점이다.

| 실제 관측 평균 | 128 | 512 |
| --- | ---: | ---: |
| Main Peg voxel 크기 | 6.331mm | 1.585mm |
| Main Peg priority point | 57.96 | 265.51 |
| Main Peg 추가 surface point | 97 | 97 |
| Wrist Peg priority point | 88.14 | 454.90 |
| Wrist Peg 추가 surface point | 97 | 97 |
| Main Box priority point · nonempty82관측 | 32.37 | 183.11 |

해상도가 올라가면 훨씬 많은 priority point가 후보 pool에 들어오지만 추가 surface
point 수와 최종 예산은 그대로다. 따라서 입력 해상도만 바꿔도 최종 FPS가 선택하는
후보들의 공간적 구성은 달라진다. PCA5의 픽셀 단위 이웃도 같은 물리 크기의 이웃이
아니다. 이 코드상 변화와 실측 수량은 확인했지만 voxel·normal stencil·priority 각각의
독립적인 인과 효과는 아직 분리 실험하지 않았다.

FPS가 최적화하는 것은 선택점 사이의 공간적 분산이며, 평가 지표는 관측 GT depth
표본 중4mm 이내인 비율이다. 두 목적은 같지 않다. 공통 참조를 사용했어도 coverage는
uniform mesh surface area가 아니며 카메라별 pixel density와 중복을 포함한다.
그래서 같은128점을 쓴다는 사실만으로 높은 해상도가 coverage를 단조 증가시킬 수는 없다.

## 오점 제거만으로 해결되는지 확인한 privileged 진단 대조군

각각의 같은 dedup pool에서 GT label이 틀린 대표점만 제거하고 **기존 FPS128**을
다시 실행했다. GT는 이 사후 대조군에서만 선택에 쓰였고 실제 Graph/추론에 적용하지
않았다. 이 대조군은 `ORACLE_SOURCE_FILTER_FPS128`이며 배포 가능한 필터가 아니다.

| Coverage | 원래128 | 원래512 | 512의 GT source-filter 대조군 |
| --- | ---: | ---: | ---: |
| Peg | 46.45% | 36.54% | 38.06% |
| Box | 31.41% | 30.22% | 31.88% |

512 Peg의 오점을 완전히 배제해도 coverage는38.06%로,128의46.45%까지 회복되지 않는다.
배경 오점은 문제의 일부이며, 남는 차이는 이미 구성된 후보 pool·같은128점 예산·현재
FPS와 coverage 목적의 차이를 함께 검토해야 한다. 오점 제거 대조군은 새로운 FPS
경로를 만들기 때문에 이 변화량을 독립 원인의 가산적 비중으로 해석하지 않는다.

## 해석과 실행 범위

### 기존 가변 N 로직과 현재 최종 상한의 구분

후속 코드 검색에서 가변 point 수를 정하는 로직이 여러 종류임을 확인했다.

1. **현재 사용 중인 앞단 Shape-first는 가변 N이다.**
   [shape_first_cuda](../../visual/runtime/cuda_sampling.py)는 voxel·thinning 후 남은
   priority `P` 전체에 Peg97/Box95개 이하의 surface point를 추가한다. `len(P)`가
   장면마다 달라지므로 출력도 가변이다. CPU 원형은
   [select_object](../../visual/runtime/test/run_variable_rate_shape_first.py)이며
   [Task03-J 보고서](03j_variable_rate_shape_first.md)가 기존 검증을 기록한다.
2. **면적 기반 Adaptive Total Budget도 구현돼 있다.**
   [AreaBudget/requested_budget](../../visual/runtime/test/run_adaptive_node_budget.py)는
   첫 양의 유효 관측 pixel 수를 reference로 고정하고 현재 support/reference 비율에
   따라 Peg+Box 합계 요청을 최대256점 안에서 정한다. 현재 Graph의 물체별128점과
   같은 정책이 아니다. `AreaBudget`과 `requested_budget`의 호출은 해당 평가 실행기와
   테스트에서만 발견했으며 현재 rollout 경로에는 연결돼 있지 않다.
   [Task03-AB 결과](03ab_adaptive_node_budget.md)는 계산 절감과 geometry 보존 손실을
   함께 보고하므로 이미 채택된 품질 개선 정책으로 해석하지 않는다.
3. **현재 최종 Graph에는 별도의128/object 상한이 있다.**
   [CurrentGeometry._build_prepared](../../graph/plan_a_bc/geometry.py)는 위의 가변
   pool을 만든 뒤 `budget_tensor(records,128)` 또는 `budget(records,128)`로 압축한다.
   CUDA `build_graph_tensor`와 `validate_graph`도128 초과를 거부한다. 따라서 이번
   비교는 앞단 가변 N을 사용했지만 그 pool 전체를 Graph에 전달하는 실험은 아니다.
   정확한 최종 수는 `min(pool_N,128)`이며 부족한 관측을128개로 padding하지 않는다.

실제로 이번512 Peg의 dedup pool은 프레임별281~474점으로 가변이었지만 최종은
모든84관측에서128점이었다.128 입력의 Peg pool도204~339점으로 가변이었다.
일반 CPU `build_graph(point_cap=None)`는 전체 가변 N을 유지하는 API를 제공한다.
그러나 주 live의 앞선 budget 호출과 CUDA builder/validator 상한을 그대로 둔 채
이 API나 config 값 하나만 바꿔서는 같은 동작을 얻을 수 없다. 이번 후속 확인은
코드·기존 산출물 검색이며 새 모델/Graph 실행이나 production 변경은 하지 않았다.

현재 근거는 **원본 관측 정보의 부족보다 후단 point 선택·압축이 고해상도 이득을
보존하지 못한다**는 것이다. 다음 검증의 대상은 배경 depth 이상점의 처리와, 같은
point 예산에서의 해상도별 후보·공간 배분이다. 고정된 새 threshold, camera quota,
voxel 또는 예산을 production에 채택한 결과는 아니다. 전체 episode와 후기 가림의
tracking, GT edge 또는 physical end/hole pose, controller success는 평가하지 않았다.

새 분석 코드는 [diagnose_sampling.py](../../graph/experiments/graph_accuracy_128_512_10i_20261104_v1/diagnose_sampling.py)에 있다.
공통 ExperimentLogger를 W&B disabled로 사용했고 run ID는
`956b4c54b3e54deb9c771444e02a7a3d`다. 분석 helper는 inference source snapshot에 별도
파일로 추가하고 자체SHA를 기록했다. 원래1,134개 source Hash와 동결 inference receipt는
유지됐다. 로컬 readback은2,352개 stage row의 point/TP 합과84관측 분모를 재계산했다.
CUDA geometry·point 선택만 원격에서 실행했다. macOS MPS/CUDA 및 simulator/renderer는
이번에 실행하지 않았다. 기존 자료·예측·production 코드는 변경하지 않았다.

## 산출물

원격 root는 `/workspace/task10i_current_graph_128_512_20261104_v1/sampling_diagnosis_v1/`다.

- [단계별 요약](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/summary.json)
- [2,352개 stage metric 행](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/stage_metrics.csv)
- [Sampler 후보·voxel·priority 집계](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/sampling_summary.json), [원자료](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/sampling_details.csv)
- [오점 거리 집계](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/fp_distance_summary.json)
- [168프레임5 tensor 재생 동일성](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/replay_equivalence.json), [보존·실행 검증](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/sampling_diagnosis_v1/verification.json)
- [결과 archive](../../outputs/task10i/current_graph_128_512_20261104_v1/sampling_diagnosis_results.tar.gz)

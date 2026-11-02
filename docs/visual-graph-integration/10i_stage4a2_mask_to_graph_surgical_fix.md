# Task 10-I Stage 4A-2 — Mask→Graph surgical diagnosis

2026-11-02. **판정: `ROOT_CAUSE_CONFIRMED_NO_PATCH`.** 지정된 네 실패 프레임에서
최초 손실 경로를 확인했다. Box는 서로 충돌하는 Main Peg/Box Mask의 **Overlap 제거**,
Peg의 B0 대비 끝부분 Coverage 악화는 **최종 cap128 선택**에서 발생한다.
현재 입력만으로 안전한 소유권 판정이나 보존량 변경을 정당화할 수 없어 Patch는 만들지 않았다.
`PAUSED_SCOPE_REVIEW`를 유지한다. Production GO 또는 전체 데이터셋 개선 판정은 아니다.

## 입력·재현 확인

실제 Instance **58576054**의 보존 자료를 읽었다. 네 프레임 모두 Branch는 `full_task`이며
정확한 Episode는 `10i_stage3_10i_s1_0355_full_task`와
`10i_stage3_10i_s1_0645_full_task`, Parent는 각각 `10i_s1_0355`, `10i_s1_0645`다.
`nominal` Branch나 같은 Parent의 다른 시점으로 대체하지 않았다.

| 자료 | 실제 원격 경로 |
|---|---|
| 저장 RGB-D·Robot State | `/workspace/task10i_stage3_20261031_v3/run/parents/{parent}/full_task/episode.h5` |
| 동결 B0/후보 Mask | `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/run/development/perception/{episode}/{condition}.npz` |
| 기존 Graph·Provenance·지표 | 위 Root의 `run/paired_graph/episodes/{episode}/` |
| 기존 EXACT_GT | `/workspace/task10i_stage4a2_newdev_reference_20261101_v1/run/references/{episode}/` |
| 실행한 원본 Source | `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source/` |
| 새 독립 Output Root | `/workspace/task10i_stage4a2_mask_to_graph_surgical_20261102_v1/` |

B0 복사 Mask는 원래 `newdev_A_20261101_v1/run/development`의 B0 파일과 SHA가 같다.
Raw H5·기존 GT 파일·Mask Receipt·Reference Receipt의 SHA를 검증했다. 네 시점 모두
저장 `exact_frame=true`다. GT를 새로 Render하거나 보간하지 않았다.

B0 원본 Source/Config Manifest SHA256은
`cd2cf5a6eccab2a3eb8a9e68bbb7c34809ffab21c9a70b1eec408db34cadf04d`다.
후보 Coordinator Source/Config Manifest SHA256은
`beaf98d64b7fe4399ec2fe92e390641ea1ddcac1f4f2bb06cadcafdd39a96ba9`,
후보 Development Manifest는
`8844d2bc2f779acfec48bd62fd2b16d8f51dc15586e8d2b786dd1920412b5de5`,
기존 Paired Evaluator Manifest는
`105a35018b486130acae9eb4310870a709ae1b6db8a1f6120d222b369ec1b5ee`다.
각각 689/689/156개 Manifest 항목을 실제 파일과 시작·종료 시 대조했다.
이들은 중복을 포함한 항목 수다. B0의 원래 Config SHA, 각 Mask/Graph SHA,
Raw/GT 경로·SHA 및 동결 후보 Source/Config 보존 검사는
[전체 결과 JSON](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/local_regression_results.json)에 있다.

**재현 장벽 PASS:** 4 Frame × 2 조건의 Graph 8개 모두 저장본과 5개 Tensor가 byte-equal이다.
대표점 World XYZ, Object ID, Source Provenance, Wrist Gate도 같다. 이후 Observer를
연결한 8개 완주 추적에서도 5개 Tensor가 byte-equal이었다. 저장 Scalar의 Correct Point,
Surface Coverage, Peg Ends Coverage와 사후 재계산값도 일치했다.
Graph의 실제 Node 정렬까지 확인해 Source ID를 최종 Node Index에 연결했다.

## 실제 실행 순서와 단위

반환되는 `C3/SF` Graph 경로는 다음과 같다.

1. `CurrentGeometry.from_h5`: 저장 RGB-D/Calibration/Robot State와 URDF FK를 연결한다.
2. `PlanA.prepare`: 카메라별 `classify_robot` 후 `filtered_masks`를 호출한다.
   후자는 `overlap | confirmed_robot | uncertain`의 **OR 합집합을 한 번에** 지운다.
3. Wrist Gate가 Clean Mask의 유효 Depth Support를 검사하고, C3에서 승인된 Wrist Peg만 사용한다.
   Wrist Box는 사용하지 않는다. Main은 Clean Mask를 유지한다.
4. 카메라별 `make_packet → packet_from_frame`: Organized Geometry의 유효 Depth Support를 만든 뒤
   PCA/Silhouette Voxel 대표화·0.5 Thinning·추가 Surface FPS(Peg97/Box95)를 수행한다.
5. 선택된 Packet을 World 좌표로 변환한 뒤 `consistent_views`와 `current_records(fuse_views=False)`를 거친다.
6. SF Point Pool에 3mm Dedup을 적용한다. 대표 측정값은 평균내지 않고 병합 Source ID를 보존한다.
7. `CurrentGeometry.build`가 `budget(pool,128)`을 호출한다. Object별, 두 Camera가 공유하는
   전역 Spatial FPS이며 기본 `anchors=False`다.
8. `PlanA.graph → build_graph(point_cap=None)`가 Graph21/Edge4를 조립한다.
   이 단계는 추가 cap128을 적용하지 않는다.

Source: [Prepare/Represent](../../graph/plan_a_geometry/core.py),
[실제 cap 호출](../../graph/plan_a_bc/geometry.py),
[Overlap/Identity](../../graph/persistent_geometry/fusion.py),
[Shape-first](../../visual/runtime/test/run_variable_rate_shape_first.py),
[최종 FPS/Graph 조립](../../graph/persistent_geometry/graph.py).

CSV의 `valid_depth`, `after_confirmed_robot`, `after_uncertain_robot`, `after_overlap`은
원인별 수치를 분리하는 **진단용 분해**다. Production이 이 순서로 여러 Filter를
실행했다는 뜻이 아니다. 실제 합집합 출력은 `filtered_mask_union`, 실제 C3 입력 Mask는
`geometry_mask_C3`, 유효 Depth 이후 Sampling 입력은 `shape_input`이다.
`stage_semantics`가 실제 Forward/진단 분해/병렬 후보 부분집합을 구분한다.

유효 Pixel 한 개는 원래 Depth 측정 Point 한 개와 연결된다. Shape-first에서는 일부 Source를
**선택하지 않으며**, 모든 미선택 Pixel이 대표점으로 병합되는 것은 아니다.
3mm Dedup은 여러 선택 Source를 대표점 하나에 연결한다. Pixel 수, 대표점 수,
병합된 고유 Source 수를 별도 열로 저장했다. GT Actor 분류와 End-band는 Builder 반환 후에만 붙였다.
물체별 Point 수에는 Graph의 이름이 고정된 Robot Node 3개를 포함하지 않는다.

## Box 0355 — t66 / t85

TP620/597은 Geometry Filter **이전의 선택된 현재 SAM2 출력**에서 측정한 값이다.
모든 내부 SAM 후보나 Logit을 뜻하지 않는다. Geometry 입력 Clean Mask의 TP가 아니다.

아래 각 Cell은 `정상 Box / Robot / Background / Other-object` 수다.
Sampling 이전에는 Pixel/원래 측정 Source, 이후에는 실제 대표점 수다.

| 시점·조건 | 저장 Mask | 유효 Depth | Robot·Uncertain 제외 후 | Overlap 제외 후 | Sampling→Dedup→cap→최종 Graph |
|---|---|---|---|---|---|
| t66 B0 | 605/1772/0/0 | 605/1772/0/0 | 605/4/0/0 | 0/4/0/0 | 0/4/0/0 |
| t66 BC | 620/0/2/1 | 620/0/2/1 | 620/0/2/1 | **0/0/2/0** | **0/0/2/0** |
| t85 B0 | 586/1467/1/0 | 586/1467/0/0 | 586/1/0/0 | 0/1/0/0 | 0/1/0/0 |
| t85 BC | 597/7/0/1 | 597/7/0/1 | 597/0/0/1 | **0/0/0/1** | **0/0/0/1** |

최초 TP 소실은 [`filtered_masks`, L40–45](../../graph/persistent_geometry/fusion.py)다.
Main Peg Mask 자체가 Box를 가리킨다. t66 Main Peg Mask의 621 Pixel은 실제 Peg1/Box620이고,
t85의 602 Pixel은 Peg0/Robot5/Box597이다. 후보 Box의 정상 Pixel **전부**가 이 잘못된
Peg Mask와 겹친다. 함수는 둘 중 어느 물체인지 결정하지 않고 두 Slot에서 모두 지운다.

t66은 Box TP620과 Other-object1이 Overlap에서 제거되고 Background2가 남는다.
t85는 Robot7이 Confirmed Robot에서 제거된 뒤 Box TP597이 Overlap에서 제거되고,
겹치지 않은 Other-object1이 남는다. 두 경우 Uncertain Robot은 정상 Box를 제거하지 않았다.
Shape-first 입력에 정상 Box는 이미 **0개**다. Wrist Box는 비활성이고,
남은 1–2점은 Cross-view 검사·Dedup·cap128에서 더 줄지 않는다.
Wrist Peg/Main Box Cross-view Conflict, 잘못된 Dedup 대표점, cap 경쟁,
Graph Presence 조립 누락이 이 두 Box 실패의 최초 원인은 아니다.

최종 Box Typed Node는 존재하지만 정상 Source Fraction과 정상 표면 Coverage는 모두 **0**이다.
따라서 단순한 Node 존재와 정상 Box Presence를 구분해야 한다. B0도 이미 같은 충돌의 영향을 받았다.
후보가 수백 Pixel을 복구했어도 충돌 제거 후에는 오점만 남는 이유가 이 경로다.

## Peg 0645 — t6 / t86

두 조건의 Main Peg Mask는 동일하다. 후보가 추가한 Wrist는 t6 TP1369/GT1386, FP1이며,
t86 TP9057/GT9064, FP0, IoU **0.999227714…**로 기존 0.999228과 일치한다.
t86 Wrist가 복구한 표면에는 **평가용 End-band Source가 0개**다.
좋은 2D Mask IoU가 Main에서 관측한 끝부분의 3D Support를 보충한다는 뜻은 아니다.

| 단계 | t6 B0 Main/Wrist | t6 BC Main/Wrist | t86 B0 Main/Wrist | t86 BC Main/Wrist |
|---|---:|---:|---:|---:|
| Filter/Gate 후 정상 Source | 842/0 | 842/1369 | 106/0 | 106/9057 |
| Shape-first 후 정상 대표점 | 197/0 | 197/188 | 106/0 | 106/218 |
| Dedup 후 정상 대표점 | 197/0 | 195/154 | 106/0 | 104/149 |
| cap128 후 정상 대표점 | **124/0** | **75/48** | **106/0** | **51/69** |
| cap128 후 전체 대표점 | 128/0 | 79/49 | 114/0 | 59/69 |

정상 End-band 대표점 수는 다음과 같다. L/R은 GT Peg의 가장 긴 Mesh 축에서
최솟값 쪽 10% / 최댓값 쪽 10%이며 화면의 좌우를 뜻하지 않는다.

| 후보 단계 | t6 Main L/R | t6 Wrist L/R | t86 Main L/R | t86 Wrist L/R |
|---|---:|---:|---:|---:|
| Filter/Gate 후 Source | 148/45 | 89/135 | 82/0 | 0/0 |
| Shape-first 후 | 34/19 | 10/22 | 82/0 | 0/0 |
| Dedup 후 | 33/19 | 8/18 | 82/0 | 0/0 |
| cap128 후 | **13/11** | **3/5** | **35/0** | **0/0** |

Shape-first 자체가 Dense Pixel을 희소화하므로 절대 Source 수의 첫 감소는 여기서 발생한다.
그러나 **후보의 aggregate End Coverage가 B0보다 낮아지는 최초 단계는 두 프레임 모두 cap128**이다.
t6의 Dedup은 일부 끝 대표점을 병합하지만 Source ID는 버리지 않았고, 이 단계의 후보 Coverage는
여전히 B0보다 높다. t86은 Dedup까지 Main 끝 82점과 Coverage1.0을 유지한다.

| End Coverage | t6 B0 | t6 BC | t86 B0 | t86 BC |
|---|---:|---:|---:|---:|
| Shape-first 후 | .453488 | .637209 | 1.000000 | 1.000000 |
| Dedup 후 | .453488 | .620930 | 1.000000 | 1.000000 |
| cap128·최종 Graph | **.313953** | **.239535** | **1.000000** | **.426829** |
| 최종 L Coverage | .298319 | .214286 | 1.000000 | .426829 |
| 최종 R Coverage | .333333 | .270833 | N/A | N/A |

t6 평가 분모는 L238/R192로 총430개의 현재 관측 GT Depth 표본이다.
t86은 L82/R0이다. R의 N/A는 0% 또는 100%로 바꾸지 않았다.
지표는 정상 Source **대표점**으로부터 4mm 미만에 있는 관측 GT 표본의 비율이다.

실제 cap 경로는 [`CurrentGeometry.build`](../../graph/plan_a_bc/geometry.py)의
`budget(pools['SF'],128)` → [`select(anchors=False)`](../../graph/plan_a_geometry/core.py) →
[`farthest_rows`](../../graph/persistent_geometry/graph.py)다. Camera별 보존량이나 End-band 기준은 없다.
t6 후보의 354개 Dedup 대표점, t86의 261개가 같은 Object 예산128을 두고 경쟁한다.
t86 B0는 Main114개라 cap에서 줄지 않지만, 후보는 Main112/Wrist149에서 Main59/Wrist69로 바뀐다.
Main 끝 82개 중 47개가 선택되지 않아 35/82=.426829가 된다. Final Graph 조립에서 추가 손실은 없다.

| 최종 Peg 지표 | t6 B0 | t6 BC | t86 B0 | t86 BC |
|---|---:|---:|---:|---:|
| 정상/전체 대표점 | 124/128 | 123/128 | 106/114 | 120/128 |
| 정상 대표점 비율 | .968750 | .960938 | .929825 | .937500 |
| 병합된 고유 Source의 정상 비율 | 124/128=.968750 | 140/145=.965517 | 106/114=.929825 | 153/161=.950311 |
| Robot/Background/Other-object | 0/4/0 | 0/5/0 | 1/6/1 | 1/6/1 |
| 정상 표면 Coverage | .234004 | .198210 | .133479 | .755289 |

대표점 비율은 병합된 Pixel 수로 가중하지 않는다. 병합된 **고유 Source**를 분모로 한
`correct_source_fraction`도 JSON에 별도로 제공한다. t86은 전체 표면 Coverage가 크게 증가하면서
끝 Coverage가 감소한다. 한 지표로 다른 지표의 손상을 가리지 않았다.

## Patch 판단과 남은 범위

**수정한 Production/연구 Source는 0개, 실행한 Patch도 0개**다.
`minimal_patch/`를 만들지 않았고 정상 대조 프레임을 추가 처리하지 않았다.
Patch 전후 개선이나 정상 대조 통과를 주장하지 않는다.

Box의 Overlap 제거는 실제 원인이지만, 이 함수에는 충돌한 두 Mask 중 올바른 Semantic Owner를
확정할 정보가 없다. 이번 예에서 작은 Mask나 Box Slot을 무조건 우대하면 GT의 답을 규칙으로
대신하는 검증되지 않은 변경이 된다. 두 물체에 모두 남겨도 다른 Actor 오염을 만들 수 있다.
Main Peg Mask의 Upstream 원인을 조사하거나 Prompt/Memory를 바꾸는 일은 이번 범위 밖이다.

Peg cap은 입력을 잘못 읽은 버그가 아니라 현재 전역 FPS 정책의 실제 비용이다.
Main 전량 보존, Camera Quota, 구조점 우대 등의 대안은 서로 다른 관측 표면을 교환한다.
예를 들어 t86 Main 끝 82개 중 Shape-first 구조점은 12개, Surface점은 70개이고,
Wrist 구조점121개에는 End-band가 없다. 단순히 기존 구조점 우대 옵션을 켜는 것이
안전한 수정이라는 근거도 없다. 어떤 대안도 실행·Sweep하지 않았다.

후속 검사가 승인된다면 필요한 정보는 (1) 충돌 Mask의 GT-free 소유권 신뢰 근거,
(2) 끝 GT를 입력하지 않는 Camera별 표면 보존 계약이다. 이 보고서는 그 규칙을 정하거나
추가 후보·벤치마크·전수 평가를 시작하지 않는다. 네 프레임 밖의 일반화는 미확인이다.

## 계산·보존·산출물

DINO/DINOv2/SAM2 Forward, SAM2 Sequence 재추론, Simulator Rollout, GT Render,
전체 Dataset Graph 처리, 정책 학습은 **각각 0회**다. 저장 입력의 원본 Graph 재현은 8회,
완주 Observer 추적은 8회다. 초기 Observer의 비유효 Depth에 대한 평가 처리 오류로
같은 대상의 부분 추적 3회가 추가되었다. 이는 평가 코드에서 발생했으며 원본 Builder나
Mask를 바꾸지 않았다. 오류 로그를 보존하고 Coverage 계산에서 비유효 XYZ만 제외했다.
Pixel 수에는 비유효 Source를 그대로 기록했다. 대상 프레임은 늘리지 않았다.

Instance의 기존 Python3.12 환경에서 **CPU Geometry/Graph와 사후 NumPy/SciPy 평가만** 실행했다.
CUDA Model Forward, GPU PhysX, SAPIEN Render, macOS Simulator/Geometry 실행은 하지 않았다.
Import 시 SAPIEN Vulkan ICD 경고가 기록됐지만 Simulator나 Renderer를 생성하지 않았다.

공통 `ExperimentLogger`를 W&B disabled로 사용했고 기존 UUID/Project/Group 규칙을 유지했다.
HF/W&B 게시나 새 Instance 생성은 하지 않았다. 종료 시 동결 후보5개 파일과 Launcher5개
Stop Stub·보존 Original의 SHA가 그대로였고, Task 모델/Runner/Waiter 실행 프로세스는 0개였다.
기존 Dataset, Checkpoint, Source 및 중지 보고서를 덮어쓰지 않았다.

필수 산출물은 새 로컬 Output Root에 보존했다.

- [단계별 CSV](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/mask_to_graph_trace.csv):
  648행, 4 Frame × 2 조건. Source/Removal/대표점/최종 Node 연결 포함.
  Source `b:<pixel>`/`h:<pixel>`은 각각 Main/Wrist이며 Episode·Timestep은 해당 행에서 상속한다.
  `u=pixel%128`, `v=pixel//128`. 병합 Group의 첫 ID가 대표 측정 Source다.
  최종 Node Index가 없는 Source는 전달되지 않았다. 큰 필드 때문에 Python CSV Reader에서는
  `csv.field_size_limit(sys.maxsize)`가 필요하다.
- [최초 손실 Attribution](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/first_loss_attribution.json).
- [원본 비교·재현·보존 결과](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/local_regression_results.json).
- [4 Frame 비교 PNG](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/case_comparison.png).
  Patch는 미실시이므로 B0/동결 후보를 비교하고 그 상태를 그림에 명시했다.

실행 Script와 작은 로그는 같은 Root에 있다. 원격 임시 그림 입력은 배포 산출물에서 제외했다.
대규모 HTML, MP4, 새 Experiment Gallery, 전체 재캐시를 만들지 않았다.
기존 Internal/Final 평가·추가 후보 탐색·정책 학습은 중지 상태로 남겨두고 이 작업을 종료한다.

![네 프레임 비교](../../outputs/task10i/stage4a2/mask_to_graph_surgical_20261102_v1/case_comparison.png)

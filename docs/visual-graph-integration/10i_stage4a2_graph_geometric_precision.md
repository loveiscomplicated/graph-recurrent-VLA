# Task 10-I Stage 4A-2 — Graph Geometric Precision, 128 vs 256

**판정: `PARTIAL_PRECISION_MEASURED`. 기존 `PAUSED_SCOPE_REVIEW` 유지.**

동일 Graph-only 추정기의 Peg 축 위치 오차는 추정 가능한 5프레임 평균 **19.73→16.74mm**,
축 각도 오차는 **8.98→8.01°**였다. 그러나 위치 오차 <3mm는 두 Budget 모두 **1/6프레임**이다.
필수 t86은 Coverage가 크게 증가해도 축 위치 오차가 **20.75→21.15mm**로 악화됐다.
양쪽 물리적 End-face Center Pair, Hole 입구 및 삽입 상대 정렬의 유효 오차는 확보하지 못했다.
**256개로 3mm 상대 삽입 정렬이 개선됐다는 결론은 내릴 수 없다.**

## 입력·프레임·재현

기존 Instance **58576054**, Python3.12.15/Torch2.14.1+cu130에서 CPU만 실행했다.
`BC_WB_HANDOFF` 저장 Mask, 원본 RGB-D/Calibration/Robot, Shape-first, 3mm Dedup,
Original Global FPS를 사용했다. Budget은 기존 코드처럼 **물체별 최대 수**다.
B0·Structural FPS·Camera Quota는 실행하지 않았다.

필수 `0645/t6,t86`에, 추정 오차를 보기 전 다음 4개를 고정했다.
`0023/t6` 초기 Reach, `0029/t67` Alignment 중간, `0120/t80` Insertion 첫 시점,
`0355/t85` 선행 보고서의 Box 관측 부족 사례다. 모두 Stage3 `full_task`이며 추가 Parent는 서로 다르다.
프레임 교체·추가는 없었다. Parent 표기는 `10i_stage3_10i_s1_<번호>_full_task`의 축약이다.

6프레임의 기존 B128 XYZ/Object/Source Group/Gate 및 Node21/Edge4의 다섯 Tensor를
저장 Receipt·Hash와 byte 단위로 대조했다. 필수 두 프레임의 B256도 이전 선택 Peg Records와
다섯 Tensor Hash에 정확히 일치했다. 두 Budget은 같은 순서·Feature·Source의 Dedup Pool을 사용한다.
현재 프레임 Provenance를 검사했고, Validator는 독립 Wrapper에서 마지막 `point cap` 검사만
메모리상 제외했다. 나머지 Node/Edge/Finite/Source 검사는 유지했으며 Production 파일은 변경하지 않았다.

`0029/t67`은 보존 Segmentation의 `exact_frame=False`였다. 최초 검증 실행은 여기서 중단됐고
GT 오차 계산은 시작하지 않았다. 이후 프레임·추정 규칙을 유지한 채 누락 Reference 처리만 분리했다.
이 프레임은 원본 Sidecar/physics Pose·Mesh 계약으로 위치·각도를 평가하되,
오점 비율·관측 GT Surface/Rim Support는 **`GT_UNSUPPORTED`/N/A**다. 부정확한 Segmentation으로
수치를 채우거나 새 GT를 생성하지 않았다.

## 실제 GT 기하 계약

설치된 `peg_insertion_side.py`, 원본 Sidecar, 보존 collision Mesh 및 actor Pose를 프레임별 대조했다.
설치 `Pose`의 순수 행렬 변환도 실행하여 저장 Pose가 world `[x,y,z,qw,qx,qy,qz]`, metres임을 확인했다.
Simulator 생성·Reset·Render는 하지 않았다.

- Peg 원점은 collision box 중심이다. Half-size `[L,r,r]`, 긴 축 local X,
  실제 끝단 Face Center는 local `[-L,0,0]`, `[+L,0,0]`이며 삽입 head는 **+X 끝**이다.
- `raw_hole_pose = box.pose * box_hole_offsets`는 **통로 중앙**, 입구가 아니다.
  Box local X half-depth도 `L`이다. 접근 측 입구는 `hole.p − L·R_hole[:,0]`,
  반대쪽 입구는 `hole.p + L·R_hole[:,0]`이다. 입구 평면의 법선과 삽입 축은 X이다.
- Hole은 원형이 아닌 사각 통로이며, `raw_hole_radius`는 사각 opening half-width `r+0.003m`다.
  Box 중심과 Hole의 Y/Z offset은 서로 다르며 원본 Mesh의 네 내벽 경계로 확인했다.
- 필수 Parent0645의 Peg는 **204.210 ×48.864 ×48.864mm**, Hole half-width **27.432mm**다.
  Hole offset은 Box local `[약0,+2.429,−6.082]mm`다. 따라서 중앙 Hole Pose를 입구로
  오해하면 local X 방향 **102.105mm**의 기준점 오류가 생긴다.
- 실행 코드의 성공 검사는 통로 중앙 기준 head의 `x≥−15mm`, `|y|,|z|≤hole_halfwidth`다.
  Source docstring의 색/성공 서술과 일부 차이가 있어 실제 offset·`evaluate()`를 기준으로 삼았다.
  이 조건을 3D Pose Error <3mm와 동일시하지 않는다. 실제 성공·충돌·동역학은 미평가다.

## 실행 전 고정한 추정기

[계약 JSON](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/geometry_estimator_contract.json)에
전체 Parameter와 Failure 조건을 보존했다. 추정 함수 입력은 **선택된 Graph Tensor와 측정 TCP Pose뿐**이다.
Node XYZ는 TCP 좌표에서 world로 변환한다. Feature는 Object Type, 관측 Normal/Validity,
Depth-jump/Validity이며, RGB 색·GT·Dense Depth·미선택 Pool·미래 프레임은 추정에 넣지 않았다.
모든 추정값을 저장한 뒤 GT scoring을 시작했다.

Peg는 이웃거리 Median/MAD로 고립점만 제거한 PCA 최대축을 사용한다.
최소12점, 고유값비≥4, 관측 길이≥40mm가 필요하다. 축 부호는 world 성분으로 결정하고 GT로 교정하지 않는다.
횡방향 중심은 두 PCA 좌표의 q02/q98 중점이다. 위치 오차는 **GT Peg 중심을 지나는 횡단 평면과
추정 직선의 교점**으로 계산한다. Skew line 최단거리로 오차를 과소평가하지 않는다.
각도는 부호 동치 `acos(|dot|)`이다.

양 끝 관측 envelope는 축 투영 q01/q99다. 물리적 Face Center로 인정하려면 각 끝의 5mm band에
축 정렬 Normal≥3점, 축 방향 표준편차≤2mm, 두 횡방향 span≥6mm가 필요하다.
양 끝이 모두 조건을 만족할 때만 Unordered Pair로 오차를 계산하도록 고정했다.

Hole은 Normal 기반 plane consensus와 Delaunay empty-gap 후보 하나를 사용했다.
충분한 평면 Support, convex hull 내부 여유, ≥8 Rim Depth-jump Point와 ≥6/8 방향의 경계 증거,
유일 후보를 요구한다. Box 중심이나 GT Hole 크기·Offset은 사용하지 않는다.
보수적인 이 절차의 실패가 다른 추정기의 실패를 의미하지 않는다.

## Peg 정밀도

| Parent/t, Phase | 축 위치 mm B128→B256 | 축 각도 ° B128→B256 | 판정 |
|---|---:|---:|---|
| 0645/t6, reach | **2.552→2.178** | 1.561→1.127 | `GEOMETRY_ESTIMATED` |
| 0645/t86, align | **20.751→21.149** | 10.415→10.906 | `HIGH_ERROR` |
| 0023/t6, reach | 13.838→13.175 | 8.200→6.396 | `HIGH_ERROR` |
| 0029/t67, align | 27.661→18.743 | 9.052→8.719 | `HIGH_ERROR`; Segmentation GT 미지원 |
| 0120/t80, insert | 33.871→28.433 | 15.694→12.922 | `HIGH_ERROR` |
| 0355/t85, insert | N/A→N/A | N/A→N/A | `AMBIGUOUS_GEOMETRY` |

`HIGH_ERROR`는 위치≥5mm 또는 각도≥5°이며 평균에서 제외하지 않았다.
0355는 두 Budget 모두 Peg83점, PCA 고유값비 **2.451<4**여서 abstain했다.

| Estimability/Precision | B128 | B256 |
|---|---:|---:|
| Peg 축 추정 가능 | 5/6 (83.3%) | 5/6 (83.3%) |
| 그중 `GEOMETRY_ESTIMATED` / `HIGH_ERROR` | 1 / 4 | 1 / 4 |
| 위치 <1mm / 1–<3mm / 3–<5mm / ≥5mm | 0 / 1 / 0 / 4 | 0 / 1 / 0 / 4 |
| 위치 미추정 | 1 | 1 |
| 추정 가능 사례 위치 평균 / 최대 mm | 19.734 / 33.871 | 16.736 / 28.433 |
| 추정 가능 사례 각도 평균 / 최대 ° | 8.985 / 15.694 | 8.014 / 12.922 |
| 한쪽 qualified Face Center | 5/6 | 5/6 |
| 양쪽 물리적 End Pair | **0/6** | **0/6** |
| Hole Center·Axis·Plane Normal | **0/6** | **0/6** |
| 삽입 Tip–Hole 상대 Pose | **0/6** | **0/6** |

한쪽 qualified Face는 항상 추정 envelope의 한쪽에만 있었다. 반대쪽은 축 정렬 Normal Support가0이다.
고정 Pair 평가 규칙상 한쪽 결과에 GT 의미 Label이나 짝을 부여하지 않았고,
물리적 End Error·Maximum End Error는 N/A로 보존했다. 두 End의 관측 band/face Support는 CSV·Audit에 있다.
Graph21에 Red/White 의미 Feature가 없어 모든 상대 항목은 **`TIP_IDENTITY_UNRESOLVED`**다.

관측 envelope와 실제 양 끝의 Unordered 최대거리도 보조 값으로만 남겼다.
t6은 **4.835→5.608mm**, t86은 **79.965→80.614mm**다.
이는 보이지 않는 물리적 끝을 관측했다고 주장하는 End Center Error가 아니다.
Surface·End-band Coverage가 증가해도 투영 범위나 중심축 편향이 자동으로 해결되지는 않았다.

## 관측 Support·오염·Coverage

GT Coverage는 이전 실험과 같은 **정상 대표점에서 <4mm 이내인 현재 관측 GT Depth 표본 비율**이다.
전체 Mesh 표면이나 보이지 않는 뒷면의 Coverage가 아니다. 오염률은 대표점 기준이며 병합 Source로 가중하지 않는다.

| Parent/t | Peg 수 128→256 | Peg 오염률 % | Peg Coverage | Box 수 | Box 오염률 % | Box Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 0645/6 | 128→256 | 3.91→1.95 | .1982→.3919 | 128→147 | 0→0 | .2602→.2988 |
| 0645/86 | 128→256 | 6.25→3.13 | .7553→.9862 | 119→119 | 2.52→2.52 | .9748→.9748 |
| 0023/6 | 128→256 | 2.34→1.17 | .4876→.8203 | 128→139 | 3.91→3.60 | .2617→.2851 |
| 0029/67 | 128→204 | N/A | N/A | 128→148 | N/A | N/A |
| 0120/80 | 128→250 | 2.34→1.20 | .5974→.7870 | 70→70 | 1.43→1.43 | .8415→.8415 |
| 0355/85 | 83→83 | 0→0 | .3240→.3240 | 1→1 | 100→100 | 0→0 |

| Parent/t | 정상 Peg End-face ±X Point 수 B128→B256 | 정상 Hole 접근 Rim Point 수 | 관측 GT Rim 표본 | Hole 판정 |
|---|---|---:|---:|---|
| 0645/6 | (10,4)→(12,5) | 3→3 | 8 | `INSUFFICIENT_OBSERVED_SUPPORT` |
| 0645/86 | (33,0)→(77,0) | 0→0 | 0 | `INSUFFICIENT_OBSERVED_SUPPORT` |
| 0023/6 | (21,4)→(40,5) | 7→7 | 21 | `INSUFFICIENT_OBSERVED_SUPPORT` |
| 0029/67 | N/A | N/A | N/A | Estimator Support 부족; GT Segmentation 미지원 |
| 0120/80 | (48,0)→(82,0) | 0→0 | 0 | `INSUFFICIENT_OBSERVED_SUPPORT` |
| 0355/85 | (0,0)→(0,0) | 0→0 | 1 | `INSUFFICIENT_OBSERVED_SUPPORT` |

End-face는 GT local 끝 평면에서4mm 이내, Rim은 접근 평면과 사각 aperture 경계 각각4mm 이내인
정상 Source 대표점이다. **추정 완료 후의 보조 진단**이며 Estimator에 넣지 않았다.
Rim Point 수만으로 완전한 개구부 방향·중심을 결정할 수 있다고 주장하지 않는다.
두 Budget의 Hole 후보는 모두0개다. 수치적 `ESTIMATOR_FAILURE`는0건이다.

t86/0120은 GT-visible 접근 Rim이0이므로 실제 관측 부족 증거가 있다.
t6/0023에는 일부 Rim 정보가 있지만 Graph의 정상 Rim은3/7점이고 고정 절차의 폐쇄 경계 조건을
충족하지 않았다. 이는 관측 희소성과 보수적인 추정 절차의 결합이다.
Peg의 `HIGH_ERROR`를 정보의 원천적 부재로 해석하지 않는다. 한쪽 표면·부분 길이의 PCA 편향,
오염, 단순 중심 규칙의 한계가 분리되지 않았으며 대체 Estimator·Sweep은 실행하지 않았다.

## 검증·실행 경계

공통 `ExperimentLogger`를 W&B disabled로 사용했다. 완료 UUID는
`3052fd906cbb491cb8388272229cf7ba`; 최초 Reference 검증 실패 UUID도 같은 Root에 보존했다.
Source/Config/Raw/Mask/GT/Freeze/Stop Stub/보존 Original 등 **767개 보호 파일 Hash**가 유지됐다.
독립 Readback은 저장 추정값의 mm/deg 재계산, 설치 Pose 행렬, 기존 필수 프레임 Coverage·오염률,
12조건 Graph Tensor 재생 Hash, CSV12행과 PNG Decode를 검증했다.

초기 Reference 검증 실패 시 native B1283회·Budget Graph6회,
완료 측정 시6회·12회, 그림 QA/Hash Readback 시6회·12회였다.
총 native B12815회·Budget Graph30회이나 **고유 측정 조건은6 Frame×2 Budget=12개**다.
그림 Readback은 추정기를 재실행하지 않았다. 계약·저장 추정값·CSV Hash는 바뀌지 않았다.
먼 비정상 Point가 t86 그림을 압축해 전체 범위 Inset과 Peg XY 확대를 추가했다. 최종 PNG는 macOS에서 저장된 Point·추정값만으로
다시 배치했으며 Geometry/Estimator를 실행하지 않았다.

DINO/DINOv2/SAM2 Forward, Simulator Reset/Render/Rollout, 새 GT 생성, GNN/GRU,
학습·전체 재캐시·새 Sampling·새 Dataset·HF/W&B 게시는 **각각0회**다.
CUDA/MPS·GPU PhysX·SAPIEN Render·macOS Geometry/Simulator는 실행하지 않았다.
SAPIEN import의 Vulkan ICD 경고는 있었지만 Env/Renderer는 생성하지 않았다.
종료 검사에서 Task Stage4A2 모델/Runner/Waiter 프로세스는0개였다.
Production Source·Dataset·Checkpoint와 기존 중지 상태를 보존했다.

## 산출물

독립 Root: `outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/`.

- [고정 추정 계약](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/geometry_estimator_contract.json)
- [12조건 mm/deg·실패·Threshold CSV](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/geometric_precision_128_256.csv)
- [Point·Ends/Rim·부분 관측 Audit](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/geometry_observability_audit.json)
- [동일성·보존·독립 검증](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/verification.json)
- [GT scoring 전 저장 추정값](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/estimates_before_gt_scoring.json)
- [독립 실행 코드](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/run_probe.py), [Readback/그림 QA](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/finish_probe.py)

![동일 world 좌표의128/256 비교. GT는 평가·표시에만 사용. N/E는 미추정이며 빨간 envelope 끝은 물리적 양 끝을 의미하지 않는다.](../../outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/geometric_precision_comparison.png)

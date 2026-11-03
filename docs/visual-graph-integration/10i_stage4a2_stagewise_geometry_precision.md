# Task 10-I Stage 4A-2 — Stagewise Geometric Information Loss Attribution

**`LOCAL_STAGEWISE_ATTRIBUTION_MEASURED` · `PAUSED_SCOPE_REVIEW` 유지.**

고정한 6프레임에서 H1/H2/H3의 증거가 함께 관측됐다. **t86의 약21mm 축 오차는 최종 FPS 이전에도 존재**했다.
Hole은 t86·0120에서 센서 접근 Rim이0개였고, t6·0023에서는 Shape-First의 Rim 손실,
0355에서는 예측 Overlap Filter의 정상 Box 제거가 확인됐다. 모든 단계의 Hole 추정은 실패했다.
관측 Point의 존재나 Coverage만으로 정확한 Pose 복원 가능성을 입증하지는 못했다.

## 범위·재현·입력 계약

기존 Stage3 `full_task`의 **0645/t6,t86; 0023/t6; 0029/t67; 0120/t80; 0355/t85**만 사용했다.
Instance58576054의 동결 `BC_WB_HANDOFF`, 원본 RGB-D/Calibration/Robot, 기존 Geometry를 읽었다.
B0·다른 Mask·새 Sampling은 실행하지 않았다.

기존 [Geometric Precision](10i_stage4a2_graph_geometric_precision.md)의 추정 Source·Contract·산출물 Hash를 검사하고,
12개 P2/P3 Graph의 다섯 Tensor Hash와 **전체 추정값·상태를 정확히 재현**한 뒤 P0/P1을 실행했다.
별도 Node21 Adapter의 좌표·Feature·Type도 P2/P3 실제 Tensor와 byte 단위로 일치했다.
P0는 기존 `extract_sampling_evidence(estimator='pca5')`의 관측 Normal/Depth-jump/Validity를 사용한다.
Overlap·오점을 포함한 유효 Depth Point를 모두 유지하고, 원래 무효 Feature는 무효 Flag와 함께 처리했다.
P0/P1은 추정 입력 Tensor만 연결하며 새로운 Edge·Sampling을 만들지 않았다. 비교 불가능 단계는 없었다.

**P0=Raw 예측 Mask+유효 Depth, P1=최종 cap 직전 Pool, P2=Original FPS128, P3=Original FPS256**이다.
추정은 이 네 단계뿐이며 Filtered/ShapeFirst/CrossView는 같은 Forward의 보조 계측이다.
실제 순서는 다음과 같다.

1. `PlanA.prepare`: Robot/Uncertain/Overlap의 **OR** 제거 → Wrist Gate. Gate 내부에도 dense Cross-view 검사가 있다.
2. `PlanA.represent`: Camera별 Shape-First Packet → `consistent_views` → `current_records(fuse_views=False)`.
3. 구조점 우선의 실제 측정 대표점 3mm Dedup → 기존 Global FPS → Node21/Edge4 Graph.

보조 CSV의 Filtered는 Wrist Gate와 기존 Wrist Box 비활성화까지 포함한다.
Filter 원인의 disjoint 재배분이나 대체 실행은 하지 않았다. 함수 경로·행 번호와 실제 Source Group은 JSON에 있다.

GT는 모든 추정값 저장 **이후** Actor/관측 Support·오차·그림에만 사용했다.
`GT_VISIBLE_SENSOR_CEILING`은 별도 Actor 표본 감사이며 추정기에 전달하지 않았다.
**0029/t67의 `exact_frame=False`를 유지**했다. Pose 오차는 원본 Sidecar로 평가하지만 Actor/Surface/Rim 통계와
센서 Ceiling은 N/A다. 미지원 Ceiling을0개 관측으로 처리하지 않았다.

## 동일 추정기의 실제 결과

단위는 **축 위치 mm / 부호 동치 축 각도 deg**다. N/E는 미추정이며0이 아니다.

| Parent/t | P0 Raw | P1 Pre-budget | P2 Graph128 | P3 Graph256 |
|---|---:|---:|---:|---:|
| 0645/6 | **7.725 / 3.094** | **3.061 / 2.451** | **2.552 / 1.561** | **2.178 / 1.127** |
| 0645/86 | N/E | **21.175 / 10.899** | **20.751 / 10.415** | **21.149 / 10.906** |
| 0023/6 | 9.326 / 3.970 | 13.195 / 6.383 | 13.838 / 8.200 | 13.175 / 6.396 |
| 0029/67 | 21.295 / 8.857 | 18.743 / 8.719 | 27.661 / 9.052 | 18.743 / 8.719 |
| 0120/80 | N/E | 28.433 / 12.922 | 33.871 / 15.694 | 28.433 / 12.922 |
| 0355/85 | N/E | N/E | N/E | N/E |

| 지표 | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 축 추정 가능 / 전체6 | 3/6 | 5/6 | 5/6 | 5/6 |
| 축 위치 <3mm / 전체6 | 0/6 | 0/6 | 1/6 | 1/6 |
| 한쪽 qualified End-face / 전체6 | 3/6 | 5/6 | 5/6 | 5/6 |
| 양쪽 물리적 End Pair / 전체6 | 0/6 | 0/6 | 0/6 | 0/6 |
| Hole / 전체6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 상대 정렬 / 전체6 | 0/6 | 0/6 | 0/6 | 0/6 |

Raw의 t86·0120·0355는 `AMBIGUOUS_GEOMETRY`로 abstain했다.
PCA 고유값비는 각각 **3.302 / 1.794 / 2.686**, 고정 기준은≥4다.
0355는 P1/P2/P3에서도2.451이다. 추정 성공을 위해 축 선택·Threshold를 조정하지 않았다.

t6은 비교적 정상적인 최종 Graph 대조지만 **Raw에서도 <3mm인 사례는 아니었다**.
표면 Coverage가 높은 Raw보다 희소화된 P2/P3에서 고정 추정기가 더 정확했다.
반대로0023은 Raw9.326mm보다 P1이13.195mm로 나빠졌다.
이는 Point 수/Coverage와 PCA 정확도의 단조 관계를 부정하며, Sampling을 개선안으로 채택하는 결과는 아니다.

모든 물리적 End Pair·Hole 오차는 N/A다. 양쪽 Face 조건을 만족하지 못했고 Hole 후보도 전부0개였다.
Graph Feature에 삽입 끝 의미가 없어 상대 항목은 **`TIP_IDENTITY_UNRESOLVED`**다.
관측 Envelope·각 끝 band/normal support·부분 Face 결과는 추정 JSON과 CSV에 보존했다.

## t86: 높은 Coverage와 약21mm 축 오차가 공존한 이유

센서 GT Peg9170점 중 nearest physical face가 **Z+인 표본8949개(97.59%)**,
local Z>0인 표본은99.60%였다. Local X 관측 범위는 **−103.03…+31.48mm**이며
물리적 +X 끝은+102.11mm다. +X End-band 표본은0개였다.
이는 **한쪽 표면·부분 길이 관측**의 증거이며, 축을 다른 방법으로 복원할 수 없다는 증명은 아니다.

| t86 Peg 단계 | 전체 / 정상 Point | Surface Coverage | End-band Coverage |
|---|---:|---:|---:|
| P0 | 9189 / 9163 | 1.0000 | 1.0000 |
| Filtered | 9171 / 9163 | 1.0000 | 1.0000 |
| ShapeFirst·CrossView | 332 / 324 | .9927 | 1.0000 |
| P1 | 261 / 253 | .9890 | 1.0000 |
| P2 | 128 / 120 | .7553 | .4268 |
| P3 | 256 / 248 | .9862 | 1.0000 |

P0의 오염률은0.283%여도 고정 PCA는 추정을 거부했다. P1에서 이미21.175mm였고,
FPS 효과는 P1 대비 P2 **−0.424mm**, P3 **−0.026mm**였다.
P2→P3 Coverage 회복을 축 위치 정밀도 개선으로 해석할 수 없다.
**H1의 부분 관측과 H3의 고정 추정 한계가 함께 지지되며, H2 최종 Budget만으로 이 오차를 설명할 수 없다.**

## 확인된 표면 손실

t6의 Peg Surface Coverage는 P0 **.9915** → ShapeFirst **.5696** → P1 **.5512** →
P2 **.1982** / P3 **.3919**였다. End-band Coverage도 **.9814→.6372→.6209→.2395/.4349**로 줄었다.
Shape-First·Dedup 대표점·최종 Budget에서의 관측 표면 손실은 실제 존재한다.
그러나 같은 경로에서 고정 추정기의 축 위치는7.725→3.061→2.552/2.178mm로 개선됐다.
**측정된 표면 보존 손실과 Pose 추정 오차의 원인을 구분해야 한다.**

Cross-view 전후 Point/Source 탈락은 여섯 프레임 모두0개였다.
Dedup은 Peg 대표점 수를 t6 **390→354**, t86 **332→261**, 0023 **311→260**,
0120 **314→250**, 0355 **157→83**으로 줄였다. 병합된 Source ID는 전부 남아 있었다.
대표점 기준 Coverage는 일부 감소했으나 이를 Source Pixel의 삭제나 평균된 가상 Point 생성으로 보고하지 않았다.

## Hole: 센서·Mask·Filter·Shape-First의 서로 다른 기여

Rim은 GT 접근 입구 평면 및 사각 opening 경계 각각4mm 이내의 정상 관측 Point다.
표의 단계 수는 대표 측정 Point 수이며 GT Source 고유 수와 병합 Group은 별도로 기록했다.

| Parent/t | 센서 GT Rim Main/Wrist | P0→Filtered→ShapeFirst→P1→P2→P3 정상 Rim 수 | 확인된 기여 |
|---|---:|---|---|
| 0645/6 | **8 / 0** | **8→8→3→3→3→3** | 센서 Rim은3/8 방향에만 존재; Shape-First에서5점 손실 |
| 0645/86 | **0 / 0** | **0→0→0→0→0→0** | `SENSOR_NOT_OBSERVABLE` |
| 0023/6 | **21 / 55** | **21→21→7→7→7→7** | 예측 Mask에55 Rim Source 부재; Shape-First에서14점 손실 |
| 0029/67 | N/A | N/A | `GT_UNSUPPORTED` |
| 0120/80 | **0 / 0** | **0→0→0→0→0→0** | `SENSOR_NOT_OBSERVABLE` |
| 0355/85 | **1 / 2** | **1→0→0→0→0→0** | 센서 자체 희소; Mask에서2점, Filter에서남은1점 손실 |

0023은 두 카메라 전체 Rim Reference76개에 대한4mm Coverage가 센서1.0→P0 **.8158**→
ShapeFirst/P1/P3 **.2237**이었다. 단순55 Source 탈락과 Coverage55/76 손실을 동일시하지 않는다.
Camera 간 가까운 표면 표본이 있기 때문이다. P0의 정상 Rim21점은6/8 방향에 분포했지만 고정 Hole 절차는
여전히 추정을 거부했다. 경계 일부의 Point 수·방향 수만으로 완전한 opening·유일 후보를 입증하지 못한다.

0355는 P0 Box605점 중 **정상597점**이 모두 예측 Peg/Box Overlap에 있었다.
Filtered에는 **오점1개만** 남았다. 정상597 Source와 Rim1 Source 제거를 실제 Mask·Source로 확인했다.
따라서 이 프레임의 최종 Box 부재를 FPS 탓으로 돌릴 수 없다.
동시에 Peg End-band는 센서109점→P0 **0점**이므로 `PREDICTED_MASK_MISSING_SUPPORT`도 확인된다.

이전 계약과의 비교를 위해 기본 Box Coverage 분모는 **Main GT 표본**으로 유지했고,
`all_camera_sensor_*` 열에 Wrist까지 포함한 센서 Ceiling과 Coverage를 별도 제공했다.
Main만 보고 센서 전체의 Hole 가시성을 판단하지 않았다. 오점 Robot/Background/Other-object,
Camera별 Point 수, 정상 수, Surface/Ends/Rim Coverage는 CSV288행에 모두 있다.

## 판정의 한계·보존·산출물

H1은 t86·0120의 Hole/미관측 Peg +X 끝에서 직접 지지된다. H2는 Mask/Filter/Shape-First/
대표점 Dedup/최종 Budget의 구체적인 Source·Coverage 손실로 지지된다.
H3는 남은 Point에 대해 고정 PCA의 거부·큰 오차가 지속된다는 관측이다.
**다른 추정기에서도 불가능한지, 추정기만 바꾸면 해결되는지는 이번 실험으로 확정하지 않았다.**

원본 Peg local X/+X head, 통로 중앙 `raw_hole_pose`, 입구 `hole.p−L·R_hole[:,0]`,
사각 half-width 차이3mm 계약을 유지했다. 3mm 계측 오차를 Insertion Success와 동일시하지 않았다.

공통 `ExperimentLogger` UUID **502e63aedbf24f4391fc19ab2edc6f1a**, W&B disabled다.
보호 파일 **789개**의 Hash가 유지됐다. Native Geometry Trace6회, Graph256 조립6회,
고유 추정 조건 **6×4=24개**만 실행했다. Source Readback은6프레임의 저장 RGB-D/Calibration 재독출이며
추정기·Graph·Sampling 재실행은0회다. 독립 Readback은 기존12조건 Tensor/전체추정 재현,
18개 유효 축 오차 재계산, Source 소유·탈락/병합 보존, CSV24/288행과 PNG를 검증했다.

Instance의 CPU Geometry/NumPy만 실행했다. macOS는 저장 배열의 그림·수치 검증만 수행했다.
CUDA/MPS, Simulator/Renderer, DINO/DINOv2/SAM2 Forward, 새 GT/Frame/Dataset,
GNN/GRU·학습·전체재캐시·새Sampling·Internal/Final 재개·게시를 실행하지 않았다.
SAPIEN import의 Vulkan ICD 경고는 있었지만 Env를 생성하지 않았다. 종료 시 Task 모델/Runner/Waiter는0개였다.
Production Source·기존 Config·Dataset·Checkpoint·Stop Stub과 `PAUSED_SCOPE_REVIEW`를 보존했다.
해결책은 구현하지 않았다.

독립 Root: `outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/`.

- [단계별 정상/오점·Camera·Ends/Rim Support](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/stagewise_geometry_support.csv)
- [24조건 정밀도·추정 상태](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/stagewise_precision_comparison.csv)
- [최초 Source 손실·단계/함수·복수 원인](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/first_information_loss_attribution.json)
- [보존·재현·No GT Leakage 검증](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/verification.json)
- [독립 CPU 실행 코드](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/run_stagewise.py)

![0645/t6·t86의 동일 좌표·축척에서 P0/P1/P2/P3 비교. 빨간 envelope는 양쪽 물리적 End 추정 성공을 뜻하지 않는다.](../../outputs/task10i/stage4a2/stagewise_geometry_precision_20261102_v1/stagewise_geometry_comparison.png)

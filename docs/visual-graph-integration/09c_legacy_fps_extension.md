# Task 09-C Extension — Legacy FPS Point Cloud / Geometry·Graph·GNN Baseline

2026-10-27 · independent feasibility diagnostic · 100-update pilot + source compatibility revalidation

2026-10-28 게시: [비공개 고정 Revision ZIP](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/fbaac6691194048043932bffb86cc96678f80bcf/integration/8c41299e9d3e4989bbf32ab1b894dc49/artifacts/reports/3d0e12d6e7f02632ee61589a9e814e43fdcf6df3742549fa6685f00150adf4aa/report-3d0e12d6e7f02632ee61589a9e814e43fdcf6df3742549fa6685f00150adf4aa.zip) · [다운로드·재현 안내와 검증](../legacy_fps/publication.md#download). 아래 실험 수치·실행 당시 상태는 원본 기록이며, Git 구현·HF 게시만 후속 수행했다. 저장소 외부 Artifact 링크는 다운로드 안내로 연결한다. ZIP 안의 보고서는 선택된 근거 파일에 대한 상대 경로를 제공한다.

## 1. 연구 질문과 실행 범위

초기 Vision 팀의 `erosion → SOR → FPS` 경로를 Source Pixel provenance와 함께 독립 재현하고, 동일한 현재 RGB-D/예측 마스크에서 Shape-First G0–G4와 Legacy L0–L4를 비교했다. 주요 실험 UUID는 `8c41299e9d3e4989bbf32ab1b894dc49`, 후속 소스 재검증 UUID는 `bd465b9d57ed4fd6843e87c950896edc`다.

Legacy 경로는 파일럿의 표면 거리와 source 순도에서 개선 사례를 보였다. 경계 보존과 누적 coverage는 나빠진 조건이 있었고, Wrist map의 Ghost/Drift도 남았다. **FPS가 전체적으로 우수하거나 후속 Policy에 도입할 준비가 됐다는 결론은 얻지 못했다.**

Vast.ai57640710 / RTX3090의 기존 환경에서 저장 관측 CPU 기하 분석과 CUDA 진단 Encoder를 실행했다. 새 simulator를 생성하지 않았다. 소스 호환성 재검증은 같은 원격 호스트의 CPU에서 실행했다. GRU·Action Head·BC 학습·Policy rollout은 수행하지 않았다.

Train8/Validation4 Episode의 t=0과 Seed30002/30125의 t=0..15를 합친 **42프레임**, 네 Episode의 t=0 이동·repeat 분기 **20개**, Seed30002의 전체 **106프레임**을 사용했다. 연속 재생은 초기 파일럿과 일부 겹치며 별도 독립 반복 실험이 아니다. 진단 Encoder36개와 Frozen Probe Decoder244개를 실행했다. 이번 확장에는 독립 Final Test가 없다.

참고한 [기존09-C 보고서](09c_persistent_geometry_graph_benchmark.md)의 Final2EP/258프레임·2000-update 결과와 이 파일럿의 수치를 직접 순위 비교하지 않는다. 아래 G/L 비교는 이번 확장 안에서 같은 입력·분할·예산을 사용한 결과다.

[실행 구성/프레임 목록](../legacy_fps/publication.md#download) · [Capture/Frozen identity](../legacy_fps/publication.md#download) · [진단 학습 구성](../legacy_fps/publication.md#download)

## 2. 실제 Sensor/Camera와 입력 Identity

기존 완료 Capture의 `base_camera`(Main), `hand_camera`(Wrist)를 사용했다. RGB는 [128,128,3] uint8, raw depth는 [128,128] signed int16 mm다. 0 이하·overflow depth는 유효 깊이로 복원하지 않았다. 각 observation의 Episode/Timestep/Timestamp/Robot 상태를 일치시키고 Intrinsic CV, Extrinsic CV, cam2world GL의 정합을 검사했다. Wrist는 매 프레임 저장된 이동 Extrinsic을 사용했다.

G/L 양쪽에 같은 RGB-D, predicted Peg/Box mask, calibration, 물체 ID와 로봇 상태를 전달했다. GT segmentation/actor pose/hole geometry는 평가 sidecar와 진단 target 전용이다. GT Hole centre나 grasp target을 Point Sampling 또는 GNN feature에 추가하지 않았다.

과거 Vision v0는512×512, 현재 입력은128×128이다. 과거 시각화와 현재 Graph 수치를 직접 성능 비교하지 않았다. 현재 Capture와 원본 HDF5의 `actual_reset` 불일치도 보존했다. 제공된 과거 Screenshot의 RGB-D/Runtime hash가 없어 **정확한 Screenshot 재현은 입증하지 못했다.** Seed30002/t0/Reach/Main의 새 동일 관측 비교로 한정한다.

[Seed30002 Camera audit](../legacy_fps/publication.md#download) · [t0 실제 RGB/예측 마스크](../legacy_fps/publication.md#download)

## 3. Legacy Source와 원래 Point 정책 재현

원형은 `visual/scripts/pointcloud.py`이며 Step5/Step8 JSON과 v0 보고서·시각화로 대조했다. 단순 FPS만 사용한 경로가 아니다.

| 순서 | 원래 정책 / 값 |
| --- | --- |
| 1 | 3×3 mask erosion, 2 iterations |
| 2 | Erosion 후 유효 depth point가50개 미만이면 원래 mask로 fallback |
| 3 | CV camera 좌표 역투영, pixel centre `(u+0.5, v+0.5)` |
| 4 | Point가2048개를 넘으면 RNG random presampling |
| 5 | SOR k=16, std ratio=2.0 |
| 6 | Optional small-cluster removal, 기본OFF |
| 7 | Object당 최대1024개 FPS |
| 8 | 원래 camera-to-world transform |

Source SHA256은 `191f0f8932d5be580890a50e412bc3ae1a474f85b674275b6d072d52c8a9de13`다. 원래 backprojection/SOR/cluster/FPS/world helper를 재사용했다. 선택 전후 XYZ의 **정확한 float32 바이트 row join**으로 Pixel ID를 유지하며 nearest-neighbour로 UV를 추측하지 않는다. 중복 XYZ 때문에 출처가 모호하면 거부한다. 실제 입력과 계약 테스트에서 XYZ·선택 순서·RNG 상태·변환을 대조했다.

원래 seed0의 순차 Episode/Frame/Peg→Box RNG와 이번 seed42의 독립 프레임 RNG 정책은 구분한다. 이번 seed는 parent episode/timestep/camera/object의 canonical JSON hash로 만들고 같은 물리 분기의 Parent RNG 그룹을 공유한다. `LEGACY_EXACT_GIVEN_INPUT_AND_RNG`는 같은 입력과 RNG 상태에서 원본 함수와 같다는 의미이며, 과거 전체 실행의 선택 픽셀까지 같다는 주장은 아니다.

[원형 Source Snapshot](../legacy_fps/publication.md#download) · [Legacy Step5 기록](../../visual/results/step5/pointcloud.json) · [Step8 기록](../../visual/results/step8/demo_conf_ep001_pointcloud.json) · [계약 테스트](../legacy_fps/publication.md#download)

## 4. G0–G4 / L0–L4와 Feature·Budget 계약

| 비교 | Robot filter | Temporal map | Camera |
| --- | --- | --- | --- |
| G0 ↔ L0 | 없음 | 없음 | Main |
| G1 ↔ L1 | 공통 filter | 없음 | Main |
| G2 ↔ L2 | 공통 filter | 공통 persistent map | Main |
| G3 ↔ L3 | 공통 filter | 없음 | Main+Wrist |
| G4 ↔ L4 | 공통 filter | 공통 persistent map | Main+Wrist |

G0는 keep_ratio0.5 / PCA5·F5 / Raw EDE / native variable-N을 유지했다. 원본 packet과14-feature graph는 별도로 저장했다. L0는 원래 전처리·최대1024/Object의 **No-edge Point Cloud**다. GNN용 L0 graph에서도 robot/object 상호작용을 포함해 Edge0을 사용했다. L1–L4는 별도 기록한 공통3D surface/k4/M16 Edge schema를 사용한다.

Native Legacy는 XYZ 표현이다. Map/GNN에 사용한 조건은 실제 선택 Pixel에서 기존 dense PCA5/F5를 gather한 **FPS+F5 증강 조건**이다. 현재 정의할 수 없는 값은 invalid로 두고, 과거 depth jump/normal variation은 현재 관측 feature로 재사용하지 않았다. 원래 FPS가 F5를 생성했다고 표기하지 않는다.

Native surface/map 크기, geometry-only matched-N, GNN cap128/Object는 서로 다른 조건이다. 지원 점이 부족하면 채우거나 보간하지 않는다. 공통 GNN 입력은21-feature, robot3개와 실제 지원 object node로 구성한다. 옛 BC checkpoint를 새 분포에 넣어 공정한 순위로 해석하지 않았다.

## 5. Seed30002/t0 동일 화면 비교

같은 World 좌표계·시점·축 범위·point marker 크기를 사용했다. G0 Points-only를 필수 Negative Control로 포함하고, G0 Raw EDE, Legacy Points-only, 동일 Legacy 점의3D surface edge, matched-N 및 평가 전용 GT overlay를 나눴다.

| 표현 | Budget | Peg N | Box N | Box 거리 mm | Box 가시 coverage | Box 경계 coverage |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | native_variable_N | 0 | 148 | 16.07 | 20.5% | 30.1% |
| L0 | native_variable_N | 0 | 442 | 2.07 | 65.7% | 10.5% |
| G0 | matched_native_minN_posthoc | 0 | 148 | 16.07 | 20.5% | 30.1% |
| L0 | matched_native_minN_posthoc | 0 | 148 | 2.08 | 22.0% | 7.2% |

Peg의 predicted mask는 비어 있어 G0/L0 모두0개다. SAM2가 완전히 누락한 Peg를 FPS만으로 복구할 수 없다. Empty 객체의 표면 거리는 미정의이며 좋은 정확도0으로 기록하지 않는다.

Box는 같은148점 진단에서도 거리 개선이 남았지만 Boundary Coverage는30.1%→7.2%로 감소했다. Edge를 숨겨도 G0의 sparse selection과 Legacy의 내부 표면 밀도 차이가 남는다. 시각 차이를 Edge clutter만으로 설명할 수는 없다.

![Seed30002 t0 동일 시점 Point/Edge/Matched-N 비교](../legacy_fps/publication.md#download)

[Same-view 영상](../legacy_fps/publication.md#download) · [Matched-N 영상](../legacy_fps/publication.md#download)

## 6. Native Representation 기하 비교

아래는12개 Episode의 t=0을 같은 가중치로 평균했다. 거리·순도는 nonempty에서만 정의되며 거리 정의 EP 수를 표시한다. N과 coverage는 Empty도 포함한다. Surface 거리는 collision mesh에서 물체당30000개 면적 가중 표본을 만든 뒤 내부 면을 제외한 외부 surface reference까지의 거리다. Exact analytic mesh distance는 아니다. Coverage threshold는4mm다.

| 표현 | 물체 | 평균 N | 거리 평균 mm | 가시 coverage | 경계 coverage | Source 순도 | 거리 정의 EP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | Peg | 152.2 | 1.76 | 69.9% | 68.4% | 96.8% | 11 |
| G0 | Box | 163.8 | 148.63 | 22.5% | 35.3% | 89.5% | 12 |
| L0 | Peg | 445.4 | 0.80 | 64.5% | 5.5% | 100.0% | 11 |
| L0 | Box | 484.3 | 5.94 | 66.2% | 17.0% | 96.6% | 12 |
| G1 | Peg | 152.2 | 1.76 | 69.9% | 68.4% | 96.8% | 11 |
| G1 | Box | 149.4 | 136.94 | 23.2% | 35.9% | 98.1% | 12 |
| L1 | Peg | 445.4 | 0.80 | 64.5% | 5.5% | 100.0% | 11 |
| L1 | Box | 414.7 | 2.03 | 59.9% | 3.5% | 100.0% | 12 |
| G2 | Peg | 148.9 | 1.76 | 69.7% | 68.4% | 96.8% | 11 |
| G2 | Box | 149.4 | 136.94 | 23.2% | 35.9% | 98.1% | 12 |
| L2 | Peg | 194.2 | 0.77 | 49.4% | 0.4% | 100.0% | 11 |
| L2 | Box | 414.7 | 2.03 | 59.9% | 3.5% | 100.0% | 12 |
| G3 | Peg | 197.2 | 1.77 | 66.1% | 71.6% | 95.6% | 12 |
| G3 | Box | 170.5 | 115.15 | 22.2% | 37.3% | 94.6% | 12 |
| L3 | Peg | 289.6 | 0.78 | 49.1% | 8.2% | 100.0% | 12 |
| L3 | Box | 461.7 | 5.70 | 57.5% | 3.6% | 98.8% | 12 |
| G4 | Peg | 197.2 | 1.77 | 66.1% | 71.6% | 95.6% | 12 |
| G4 | Box | 170.5 | 115.15 | 22.2% | 37.3% | 94.6% | 12 |
| L4 | Peg | 289.6 | 0.78 | 49.1% | 8.2% | 100.0% | 12 |
| L4 | Box | 461.7 | 5.70 | 57.5% | 3.6% | 98.8% | 12 |

G0 Box의 Episode별 평균 거리의 중앙값은19.83mm, 최대는706.31mm(seed30086)였다. L0는 중앙값2.07mm, 최대34.42mm(seed30050)였다. 따라서148.63→5.94mm의 평균 차이에는 큰 오염 사례의 영향이 있다. Point를 모두 합친 pooled 통계나 모든 장면에서 같은 크기의 개선이라는 뜻은 아니다.

Legacy는 Peg의 거리와 순도를 개선했으나 native 가시 coverage69.9%→64.5%, 경계68.4%→5.5%로 줄었다. Box는 가시 표면 밀도를 늘렸지만 경계 coverage35.3%→17.0%로 감소했다. 표면 거리 하나로 형상 보존 전체를 판정하면 안 된다.

Gi/Li 쌍 안에서는 같은 센서 범위를 사용한다. Main 전용과 Main+Wrist의 own-visible 분모는 다르므로 이 표의 coverage만으로 camera 효과를 비교하지 않는다.

[Native/cap/matched 전체 수치](../legacy_fps/publication.md#download) · [t0 평균·정의 분모](../legacy_fps/publication.md#download)

## 7. Matched Point Budget과 실제 FPS 동작 빈도

| 표현 | 물체 | 평균 matched N | 거리 평균 mm | 가시 coverage | 경계 coverage |
| --- | --- | --- | --- | --- | --- |
| G0 | Peg | 152.2 | 1.76 | 69.9% | 68.4% |
| G0 | Box | 163.8 | 148.63 | 22.5% | 35.3% |
| L0 | Peg | 152.2 | 0.79 | 62.4% | 3.3% |
| L0 | Box | 163.8 | 6.49 | 24.3% | 7.7% |

Matched-N은 각 물체의 실제 native 지원 수 중 min(Gi,Li)를 택하고 공통 identity-seeded farthest subset을 선택한 **사후 진단**이다. Shape-First structural/surface의 원래 선택 규칙을 새 Native 설정으로 바꾼 결과가 아니다. 가상·보간 Point는0개다. 같은 N에서도 Legacy의 거리 개선 사례는 남지만 Peg/Box의 경계 손실은 더 뚜렷하다.

현재 Main84 Object-frame 중68개가 nonempty였다. 그중67개는 SOR 후1024개 이하라 최종 FPS가 그대로 반환했다. Main에서 Final FPS와 random presampling이 실제 점 수를 줄인 사례는 각각1개다. Filtered/Unfiltered를 포함한 모든 Camera 조건252개 중 Final FPS 동작은6개였다.

**이번 결과를 FPS 알고리즘 자체의 우위라고 부를 수 없다.** 대부분의 현재 해상도 사례는 erosion/SOR를 포함한 전체 정책과 점 밀도 차이다. Erosion/SOR/FPS 각각을 분리한 factorial ablation은 실행하지 않았다.

[실제 전처리 동작·실패 catalog](../legacy_fps/publication.md#download) · [Pixel별 제거 감사 집계](../legacy_fps/publication.md#download)

## 8. Robot Self-filter와 연속 전처리 손실

기존09-C의 개별 robot collision shape depth/interior filter를 재사용했다. 큰 robot hull이나 추가 제거 반경을 새로 설계하지 않았다. G/L의 confirmed/uncertain 픽셀 판정이 정확히 같음을 검사하고 같은 clean mask를 각 Point 생성기에 전달했다. GT robot/actor ID는 평가에만 썼다.

| Camera / 물체 | Robot recall | 정상 표면 보존 | Robot 오염 전→후 | Mask 순도 전→후 |
| --- | --- | --- | --- | --- |
| base_camera / Peg | 99.6% | 100.0% | 0.0% → 0.0% | 99.0% → 99.0% |
| base_camera / Box | 99.6% | 100.0% | 11.3% → 0.2% | 87.7% → 98.7% |
| hand_camera / Peg | 94.2% | 100.0% | 0.0% → 0.0% | 98.1% → 98.1% |
| hand_camera / Box | 94.2% | 100.0% | 96.5% → 37.9% | 2.4% → 2.5% |

표는42프레임의 정의 가능한 비율을 프레임별 평균한 값으로,12EP t0 기하 표와 가중치가 다르다. 두 Episode의 연속 구간이 포함돼 독립 Episode 평균으로 해석하지 않는다. 정상 표면 보존100%는 이 파일럿의 전체 GT object pixel 기준이다. Finger20mm 근접 오제거 지표는 정의 가능한 분모가 없어 **NOT_TESTED**이며 접촉 근처 안전성을 보장하지 않는다.

Wrist Box는 제거 후에도 robot 오염37.9%, mask 순도2.5%로 실패가 남았다. Mask 픽셀 순도와 FPS 후 node source 순도는 다른 단계의 지표다. Filter 뒤에 erosion/SOR를 이어 적용할 때도 올바른 경계가 사라질 수 있다.

전처리 감사에서 Erosion 때문에 유효 입력이 완전히 비게 된 사례는0개였다. SOR가 제거한 올바른 surface pixel은3676개였고, 정의한4mm Hole rim reference에 해당하는 제거는0개였다. Filtered/Unfiltered·Camera 중복을 포함한 계수다. **SOR rim 제거0은 전체 경계 보존의 증거가 아니다.** Erosion에서 먼저 사라진 경계와 FPS 후 coverage를 함께 봐야 한다.

[필터 전후와 분모](../legacy_fps/publication.md#download) · [실제 t0 filter tensor](../legacy_fps/publication.md#download)

## 9. Persistent Node Lifecycle와 Pose Tracking

동일 `PersistentMap`/registration/fusion Source Snapshot을 사용했다. SurfaceObservation에 Episode/Timestep/Camera/Object, world XYZ, normal/F5 validity, 실제 Source Pixel과 camera origin을 전달했다. Camera XYZ는 원래 Point artifact에 별도 보존한다. Source mask의 soft assignment confidence는 없어 availability=false로 기록하며 SAM2 확률을 만들어 넣지 않는다.

Map은 persistent ID, object-local surface, first/last seen, supporting source ID, confidence/uncertainty와 observed/estimated 상태를 유지한다. 재관측은 update, 신규 연결 표면은 add, tracking/identity 실패는 reject·confidence 감소·contradiction 제거로 처리한다. 원래 merge3mm 규칙을 사용하며 매 프레임1024개를 강제로 추가하지 않는다. Native map budget과 GNN cap128은 독립이다. GT pose·미래 관측으로 practical map을 보정하지 않았다.

아래는 Seed30002 전체106프레임을 causal replay하고, 동일14개 평가 시점에서 비교한 Native 결과다. Point count/ghost/누적 coverage는 물체당14행, pose는 정의 가능한 n을 표시한다. 공통 누적 reference는 두 센서에서 지금까지 관측한 GT surface의 합집합이다. Pose 이동 오차는 관측으로 초기화한 map frame에 대한 GT 상대 운동 기준이며 actor origin 절대 복원 오차와 다르다.

| 표현 | 물체 | 평균 Native N | 공통 누적 coverage | Ghost 수 | Pose 이동오차 mm | Local drift mm | Pose 정의 n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G2 | Peg | 0.0 | 0.0% | 0.0 | — | — | 0 |
| G2 | Box | 605.9 | 61.9% | 3.3 | 0.58 | 0.27 | 14 |
| L2 | Peg | 0.0 | 0.0% | 0.0 | — | — | 0 |
| L2 | Box | 566.2 | 60.2% | 0.0 | 0.11 | 0.12 | 14 |
| G4 | Peg | 808.9 | 60.7% | 255.9 | 58.72 | 47.96 | 14 |
| G4 | Box | 647.9 | 34.8% | 289.6 | 10.55 | 11.43 | 14 |
| L4 | Peg | 680.2 | 53.5% | 140.5 | 31.87 | 29.29 | 14 |
| L4 | Box | 610.8 | 33.1% | 299.6 | 13.98 | 12.59 | 14 |

Main-only G2/L2는 이 Episode에서 Peg map이 없었다. 따라서 낮은 평균 pose 오차를 Peg tracking 성공으로 해석할 수 없다. Box L2는 G2보다 Ghost와 drift가 줄었지만 누적 coverage61.9%→60.2%로 감소했다.

L4 Peg는 G4보다 Ghost255.9→140.5, pose 이동오차58.72→31.87mm로 줄었으나 정확한 추적이라고 보기 어렵다. Box는 L4의 Ghost299.6과 pose13.98mm로 G4의289.6/10.55mm보다 나빴다. 물체를 합친 평균만으로 FPS Map의 일관된 개선을 주장하지 않는다.

[L 시간축 기하 원자료](../legacy_fps/publication.md#download) · [동일 Capture G reference 원자료](../legacy_fps/publication.md#download) · [완료된 G 파일의 Source identity](../legacy_fps/publication.md#download)

## 10. 가림·신규 표면·Ghost 사례

현재 Sensor 관측과 과거 estimated surface를 색으로 구분했다. t65에는 Main G1/L1 관측이 비었지만 G2/L2의 Box map이 유지됐다. 가림 중 유지가 현재 물체 pose의 정확성을 보장하지는 않는다. Peg처럼 한 번도 확보하지 못한 표면은 Map도 만들어 내지 않는다.

Map lifecycle의 추가/갱신 이벤트와 GT 기준 `new_correct_previously_unseen_nodes`를 구분했다. **t0 이후 평가한13개 시점에서 G2/L2/G4/L4의 해당 신규 표면 지표는 양성0개**였다. 전체106개 모든 시점에 신규 표면이 없었다는 결론은 아니다. 기하 지표는 interval10과 고정25/65/마지막 시점에서 계산했고, t0 초기화·기존 표면의 새 node·sampling으로 놓친 이벤트를 분리해야 한다. 올바른 신규 표면 통합 우위는 이번 파일럿에서 미확인이다.

![t65 평가 전용 Ghost 표시: 검은 X는 GT surface에서8mm 초과인 estimated node](../legacy_fps/publication.md#download)

[전체106프레임 persistent 비교 영상](../legacy_fps/publication.md#download) · [신규 node 추가 영상](../legacy_fps/publication.md#download) · [Ghost/실패 영상](../legacy_fps/publication.md#download) · [시각 사례 수치](../legacy_fps/publication.md#download)

## 11. Main+Wrist Fusion 기여

동일한 object association/consistent-view 정책으로 두 센서를 합쳤다. 다른 camera의 mask 누락 자체를 negative evidence로 쓰지 않고, 같은 측정 표면에 다른 identity가 있으면 unknown으로 거부한다. 겹치는 surface는 공통 merge 규칙으로 통합한다.

아래는 같은14개 시간축 시점에서 같은 두-camera reference를 사용한 L 계열 결과다. 현재 visible은 물체가 GT 두 센서 합집합에서 보일 때 정의되며 Peg14행/Box11행이다. 누적 coverage는 모두14행이다. Source 순도는 nonempty일 때만 정의돼 L1 Box는9행, 나머지 표의 nonempty 물체는14행이며 L1/L2 Peg는 미정의다.

| 표현 | 물체 | 공통 현재 visible coverage | 공통 누적 coverage | Source 순도 |
| --- | --- | --- | --- | --- |
| L1 | Peg | 0.0% | 0.0% | — |
| L1 | Box | 36.6% | 12.1% | 99.1% |
| L3 | Peg | 87.2% | 36.7% | 100.0% |
| L3 | Box | 36.6% | 12.1% | 61.4% |
| L2 | Peg | 0.0% | 0.0% | — |
| L2 | Box | 74.0% | 60.2% | 99.8% |
| L4 | Peg | 89.6% | 53.5% | 100.0% |
| L4 | Box | 31.5% | 33.1% | 97.5% |

L1→L3의 Peg 현재 visible coverage0%→87.2%는 Wrist의 관측 기여다. Box coverage36.6%는 같았고 L3 source 순도61.4%로 잘못된 camera 입력이 남았다. L2→L4는 Peg 누적 coverage를 추가했지만 Box 누적 coverage60.2%→33.1%로 악화됐다. 동일성/좌표 정합 통과와 올바른 물체 형상 복원은 구분해야 한다.

[Main/Wrist 전체 영상](../legacy_fps/publication.md#download) · [t0 Fusion conflict audit](../legacy_fps/publication.md#download)

## 12. 같은 FPS Point의 Edge 유무·정책

아래는12EP t0에서 물체-Episode24행 평균이다. Intra edge는 무방향 수다. L0_surface와 L0_rawede는 L0의 **동일 cap128 node tensor**를 사용했다. G0는 자기 Shape-First node를 사용하므로 G0와 FPS Edge 수 차이를 Edge 정책 단독 효과로 해석하지 않는다.

| 표현 / cap128 | Intra edge | 표면 미지원 edge | Free-space crossing | Aperture crossing | Components |
| --- | --- | --- | --- | --- | --- |
| G0 | 250.71 | 33.88 | 24.08 | 5.58 | 10.04 |
| L0 | 0.00 | 0.00 | 0.00 | 0.00 | 122.67 |
| L0_surface | 260.17 | 4.75 | 6.25 | 0.00 | 47.17 |
| L0_rawede | 333.58 | 48.04 | 21.67 | 7.46 | 2.25 |

L0는 Edge가 없어 crossing0이며, geometry가 완전하다는 뜻은 아니다. 3D surface gate는 Raw EDE보다 허위 bridge/crossing을 줄였지만 components47.2로 단절 비용이 있었다. Raw EDE의 낮은 components2.25도 잘못된 연결을 포함할 수 있다. Hole rim coverage와 graph connectivity를 별도로 저장했다.

FPS Raw EDE는 실제 현재 Main Source Pixel/Observed Component에서 원래 helper로 재계산했다. Camera/과거 Timestep이 섞인 Map에는 단일 raster UV Raw EDE를 적용하지 않았다. G/L source·feature row join 및 same-point Edge 조건84건을 실제 tensor로 검사했다.

[Edge 길이/교차/분모 전체](../legacy_fps/publication.md#download) · [t0 FPS Raw EDE node/edge tensor](../legacy_fps/publication.md#download)

## 13. 동일 예산 GNN·Frozen Probe 구성

E0는 공통 Edge-aware GNN, E1은 message passing을 제거한 Point-set encoding/pooling, E2는 같은 입력 XYZ의 geometry summary를 readout에 보존한 후보다.12개 표현(G0–G4/L0–L4/L0_surface/L0_rawede)×3 Encoder=36개를 새로 진단 학습했다.

| 항목 | 실제 실행 조건 |
| --- | --- |
| Encoder 입력 | 동일21-feature, cap128/Object, 실제 지원 N 유지 |
| Objective | TCP-relative actor position6 + 현재 visible surface summary30 |
| Optimizer | AdamW, lr0.001, weight decay1e-5, batch16 |
| Seed / Update | seed42, 100 updates/후보 |
| 선택 | 고정 validation checkpoint의 최소 Geometry loss |
| Split | Train8EP / Validation4EP, branch는 parent split 유지 |
| 실제 fit row | Train19 / Validation15; timestep interval5, repeat·physical-invalid 제외 |
| Frozen Probe | 기존09-B Ridge/Small MLP, 동일 정규화·target·split·decoder 설정 |
| Probe 예산 | Small MLP 최대100 epoch/후보, hidden64 |

총62개 저장 Graph/후보에는 non-fit row와 repeat도 포함하며, 독립 표본62개라는 의미가 아니다. 정규화는 Train만 사용했다. GT actor pose는 진단 label이며 입력 feature가 아니다. Training fit만으로 표현을 판정하지 않고 독립 Episode Validation과 ±10mm paired response를 분리했다.

정식2000-update/2000-Probe-epoch 확장과 독립 final62091/62092 적용은 **NOT_EXECUTED**다. 기존09-C 최종 보고서의19개 모델/Final 결과를 이번 FPS 후보와 같은 학습 조건의 비교값으로 가져오지 않았다.

[학습·Checkpoint identity](../legacy_fps/publication.md#download) · [Split/실제 Frame 목록](../legacy_fps/publication.md#download) · [Probe244개 구성/선택](../legacy_fps/publication.md#download)

## 14. Geometry Readout Probe 결과

아래는 Frozen Readout의 Ridge(P1) 결과다. 위치 열은 Validation baseline의 Peg/Box mean error를 같은 가중치로 평균했다. Paired 열은 Validation의 physical-valid·target-moved 분기를 합친 평균이며 위치 열과 분모가 다르다. 방향은 zero response에서 미정의일 수 있어 원 CSV의 정의 행을 함께 읽는다.

| 표현 | Encoder | Validation 위치 mm | Paired Δ 오차 mm | 방향 cosine | Signed gain |
| --- | --- | --- | --- | --- | --- |
| G0 | E0 | 61.85 | 11.09 | 0.579 | 0.307 |
| G0 | E1 | 79.17 | 12.54 | -0.004 | -0.026 |
| G0 | E2 | 58.26 | 14.46 | 0.482 | 0.183 |
| L0 | E0 | 68.54 | 16.12 | 0.316 | 0.247 |
| L0 | E1 | 85.47 | 13.29 | 0.192 | 0.160 |
| L0 | E2 | 51.48 | 16.84 | 0.736 | 0.812 |
| G1 | E0 | 70.13 | 13.59 | 0.065 | 0.176 |
| G1 | E1 | 64.04 | 11.30 | 0.288 | 0.088 |
| G1 | E2 | 76.42 | 23.38 | 0.141 | 0.113 |
| L1 | E0 | 79.07 | 9.99 | 0.378 | 0.093 |
| L1 | E1 | 86.59 | 10.06 | 0.374 | 0.275 |
| L1 | E2 | 32.97 | 7.33 | 0.801 | 0.944 |
| G2 | E0 | 81.86 | 9.73 | 0.289 | 0.148 |
| G2 | E1 | 92.48 | 8.94 | 0.676 | 0.240 |
| G2 | E2 | 78.36 | 15.07 | 0.139 | 0.102 |
| L2 | E0 | 70.78 | 8.48 | 0.419 | 0.216 |
| L2 | E1 | 67.32 | 8.99 | 0.511 | 0.119 |
| L2 | E2 | 30.49 | 6.60 | 0.854 | 0.772 |
| G3 | E0 | 70.03 | 23.04 | 0.371 | 0.171 |
| G3 | E1 | 54.83 | 14.85 | 0.113 | -0.081 |
| G3 | E2 | 85.40 | 12.67 | 0.237 | 0.040 |
| L3 | E0 | 63.34 | 11.11 | -0.053 | -0.014 |
| L3 | E1 | 92.87 | 10.63 | 0.280 | 0.226 |
| L3 | E2 | 46.88 | 7.17 | 0.483 | 0.411 |
| G4 | E0 | 72.14 | 10.58 | 0.536 | 0.369 |
| G4 | E1 | 81.11 | 9.07 | 0.572 | 0.208 |
| G4 | E2 | 51.54 | 9.25 | 0.513 | 0.384 |
| L4 | E0 | 66.79 | 9.94 | 0.083 | 0.083 |
| L4 | E1 | 83.12 | 11.20 | 0.526 | 0.336 |
| L4 | E2 | 65.65 | 8.59 | 0.459 | 0.337 |
| L0_surface | E0 | 62.41 | 23.94 | 0.510 | 1.038 |
| L0_surface | E1 | 85.47 | 13.29 | 0.192 | 0.160 |
| L0_surface | E2 | 42.77 | 14.43 | 0.594 | 0.689 |
| L0_rawede | E0 | 63.87 | 15.94 | 0.313 | -0.035 |
| L0_rawede | E1 | 85.47 | 13.29 | 0.192 | 0.160 |
| L0_rawede | E2 | 49.05 | 12.54 | 0.442 | 0.309 |

L0+E0는 G0+E0보다 위치61.85→68.54mm, paired Δ11.09→16.12mm로 나빴다. 더 정확해 보이는 Raw surface가 그대로 GNN readout 우위로 이어지지 않았다. 같은 FPS 점에서 E0의3D edge는 위치68.54→62.41mm로 줄였지만 paired Δ는16.12→23.94mm로 커졌다. 목적 지표에 따라 결론이 달라진다.

L0/L0_surface/L0_rawede의 E1 readout과 선택 파라미터 hash는 정확히 같았다. Edge만 바꿔도 no-message 결과가 바뀌지 않는 Negative Control이다. E2가 좋은 조건도 있으나100-update/소수 Episode·한 seed 결과이며 최종 우승 구조를 선정하지 않는다.

[전체 Absolute Probe/정의 분모](../legacy_fps/publication.md#download) · [Paired Δ/방향/gain](../legacy_fps/publication.md#download) · [Episode별 Probe](../legacy_fps/publication.md#download)

## 15. 실패 사례·Negative Control·소스 호환성

| 항목 | 실제 결과 / 한계 |
| --- | --- |
| Empty predicted Peg | Seed30002/t0에서 G0/L0 모두0; FPS 단독 복구 불가 |
| Erosion fallback | Main1건; 유효 입력의 완전 소실0건. 경계 coverage 손실은 남음 |
| SOR | 올바른 surface pixel3676개도 제거; 무조건 GT 보존 정책 아님 |
| FPS 외곽 outlier 선호 | 정의한 outlier fraction 증가 사례0; FPS 활성 표본이 적음 |
| Robot contamination | 공통 filter로 감소; Wrist Box mask 실패 남음 |
| Persistent Ghost | G4/L4 모두 큰 Ghost; Box L4는 일부 지표 악화 |
| Raw EDE raster misuse | 실제 현재 Pixel/Component만 허용; past/foreign camera 거부 |
| Same point / no-message | Edge 세 조건의 E1 checkpoint/readout exact |
| GNN 정보 보존 | Raw surface 거리 개선에도 L0+E0 Probe 우위 없음 |
| GT/Future leakage | Practical 입력 whitelist, source timestep/feature validity 검사 통과 |

초기 학습 감사는 원래 positional robot readout 때문에 전체 노드 순열에서 실패했다. 확장에는 node_type으로 로봇3행을 정렬하는 입력 Adapter를 두었다. 정상 저장 순서에서는 항등 변환이며 구조·초기 파라미터·정상 입력 출력을 보존했다. 실패 UUID/Import·직렬화·PID namespace 설정 오류는 로그로 남겼고 성능 순위에 쓰지 않았다.

이후 원본 작업의 Encoder에도 같은 목적의 수정이 들어왔다. 후속 독립 스냅샷에서 현재 source 계약54개, 기존36 Checkpoint의2232개 실제 Graph Readout/Decoder exact, 전체 노드 순열108건을 재검증했다. CPU↔이전 CUDA 차이는 atol2e-6 + rtol2e-5 내였다. Backend 간 bitwise 동일성은 주장하지 않는다.

[실패 Attempt readback](../legacy_fps/publication.md#download) · [후속 Source 재검증](../legacy_fps/publication.md#download) · [정상 입력 exact/순열 결과](../legacy_fps/publication.md#download)

## 16. 계산 비용·Memory

CPU 분석은 nice19, BLAS/OMP1 thread로 실행했다. 다음은 Seed30002/t0 고정 입력의 warmup 제외5회 평균과 단일 Graph GNN inference다. Map은 매회 fresh 초기화이며 steady-state tracking 속도가 아니다. GNN은 Encoder별12개 표현 평균이고 각 Graph의 실제 N이 다를 수 있다.

| 측정 단계 | 평균 ms | 범위 |
| --- | --- | --- |
| Robot filter | 21.84 | Seed30002/t0 Main |
| Legacy XYZ 추출 | 2.99 | 동일 입력, F5 없음 |
| Legacy XYZ + F5 | 9.21 | 동일 source evidence |
| Map 초기화 | 49.11 | fresh map |
| Graph 구성 | 78.21 | cap128/object |
| E0 inference | 2.59 | 12개 표현, 단일 Graph |
| E1 inference | 1.33 | 12개 표현, 단일 Graph |
| E2 inference | 2.96 | 12개 표현, 단일 Graph |

학습 Source 기록의 최대 CUDA allocator peak는134,085,632bytes(약127.9MiB), process RSS high-water는1,618,960,384bytes다. Context·다른 프로세스를 포함한 global GPU memory와 다르다. 모델별 학습 시간 합은약80.70s이며 CPU Probe/입출력·실패 Attempt까지 합한 전체 wall이 아니다.

전체 pipeline·Graph+metric·pure 단계 시간을 구분해 저장했다. GPU 진단은 다른 GPU process가 없는 검사와6% allocator cap에서만 시작했고, NVIDIA host PID와 container PID 차이를 처리했다. 확인한 확장 GPU 단계와 해당 원본 학습 시간은 겹치지 않았다. 실시간20Hz 보장이나 속도 최적화는 수행하지 않았다.

[CPU 단계 비용](../legacy_fps/publication.md#download) · [실제 GNN inference](../legacy_fps/publication.md#download) · [GPU 실행 시간 근거](../legacy_fps/publication.md#download)

## 17. 최종 연구 질문 Q1–Q12

| 질문 | 판정 | 실측 해석 |
| --- | --- | --- |
| Q1 Legacy가 Surface를 더 정확히 보존하는가? | SUPPORTED | 표면 거리·순도 개선 사례가 있으나 경계/coverage 악화가 공존한다. |
| Q2 시각 차이의 원인은 무엇인가? | INCONCLUSIVE | 점 수·사후 matched-N·Edge를 분리했다. 대부분 Final FPS는 pass-through이며 단계별 독립 ablation은 없다. |
| Q3 같은 N에서도 유리한가? | SUPPORTED | 거리 개선은 남지만 Boundary 손실도 커진다. Native sampler 단독 순위는 아니다. |
| Q4 Robot filter 후 보존이 개선되는가? | INCONCLUSIVE | Main 오염은 줄었으나 작은 Peg/finger 근접 분모와 Wrist Box 실패가 남는다. |
| Q5 가림 유지·새 표면 통합에 도움이 되는가? | INCONCLUSIVE | 유지/추가 경로는 구현했다. 현재 표본에서 일관된 correct-coverage/new-surface 우위는 미확인이다. |
| Q6 움직일 때 Ghost/Drift는 어떤가? | INCONCLUSIVE | L4 Peg 개선과 Box 악화가 함께 있다. 한 연속 Episode로 일반화하지 않는다. |
| Q7 Wrist로 coverage가 개선되는가? | SUPPORTED | Main에서 누락한 Peg 관측은 추가했다. Box coverage/순도·Map 품질은 악화할 수 있다. |
| Q8 Edge-aware가 no-message보다 좋은가? | INCONCLUSIVE | 절대/paired 목표에 따라 달라지고100-update 파일럿으로 확정 순위는 부족하다. |
| Q9 Raw geometry가 Readout에서 손실되는가? | INCONCLUSIVE | 입력과 Readout의 차이는 측정했으나 Probe 일반화·학습 부족과 손실 원인을 완전히 분리하지 못했다. |
| Q10 G0–G4 대비 개선/악화 조건을 구분하는가? | SUPPORTED | 거리·경계·순도·Ghost·Probe별로 다른 방향의 효과를 확인했다. |
| Q11 후속 Policy 도입 근거가 충분한가? | INCONCLUSIVE | 충분한 동일 진단 예산·다수 연속 Episode·독립 Final 적용이 필요하다. Policy는 실행하지 않았다. |
| Q12 기존 Source/Process/Artifact를 보호했는가? | INCONCLUSIVE / 후속 구간 CONFIRMED | 과거 전역 Source 불변은 외부 변경으로 성립하지 않았다. 이 확장은 원본을 쓰거나 제어하지 않았고, 후속 기준7590개/기존9892개 불변과 PID/starttime 동일을 확인했다. |

## 18. 실행·미실행 항목과 Stage 판정

| Stage | 상태 | 근거 |
| --- | --- | --- |
| 0 Audit/isolation | PARTIAL | 원본 Source 외부 변경 이력 보존; 현재 검증 revision의 호환성·새 보호 구간은 PASS |
| 1 Legacy Adapter | PASS | 원형 helper 재사용, Source Pixel/RNG/XYZ/transform 재현 |
| 2 Static geometry | PARTIAL |42프레임·12EP t0, Native/matched/points-only/edge 파일럿 |
| 3 Filter/persistent | PARTIAL | 공통 map·tracking 연결, 한 Episode106프레임, drift/경계 한계 |
| 4 Wrist fusion | PARTIAL | 실제 두 sensor 실행, Peg 관측 추가와 Box 실패 |
| 5 GNN/probe | PARTIAL |36개100-update/244 Probe 실행; 정식 예산·독립 Final 미실행 |
| 6 Report/integration | PASS | 실패·원자료·영상·Guard/Manifest·Source 재검증 보존 |

실행: 저장 RGB-D/예측 마스크의 Linux CPU 기하 분석, RTX3090 진단 GNN, CPU Frozen Probe, Native/cap/matched 평가, actual tensor·Source join·Map/Camera/Edge 감사, PNG/MP4 decode·대표 contact sheet 검수, 현재 source의 CPU 호환성 재검증.

미실행: 이번 확장의 신규 simulator capture, CUDA PhysX/vectorized simulator, 이번 Mac simulator/MPS 학습, 독립 Final Test, 다수 학습 seed, late-time branch의 causal map fork, Policy/GRU/Action Head/BC/closed-loop 성공률, 정식2000-update 확장, 외부 실물 센서. 새 dependency 설치·Git commit·외부 publishing은 수행하지 않았다.

이 Markdown은 기존 Artifact를 정리한 보고서다. 문서 작성 과정에서 학습/평가를 다시 실행하거나 기존09-C 결과를 변경하지 않았다.

## 19. Source·Artifact 보호, 통합 계약과 재현 자료

별도 `/workspace/task09c_legacy_fps/` Source/Run directory를 사용했다. 공유 코드·Production Shape-First 기본값·기존 G0–G4 정의·UUID·checkpoint·Frozen Expert Dataset/Split·중단된 Full A/B/C를 변경하지 않았다. 공통 `ExperimentLogger`의 UUID/project/group 규칙과 disabled local logging을 재사용했다. 기존 작업에 signal·pause·restart·동적 후보 주입을 하지 않았다.

초기 보호 대상7589개는 한 감사 시점에 모두 같았으나, 병렬 원본 작업의 `encoder.py`/`tests/test_contracts.py`가 변경돼 다음 전역 감사는7587개 불변/2개 변경으로 PARTIAL/FAIL을 보존했다. 원본 자체의2000-update 후 순열 감사 오류와 이후 재개는 시간별 읽기 전용 관찰이다. 확장이 원본을 수정·재시작한 것이 아니다.

후속 검사에는 추가로 `branch_eval.py`/`evaluate.py`/`geometry_audit.py`의 handoff·상대 경로 수정과 새 `handoff.py`를 포함해 **변경5개+추가1개**를 고정 스냅샷으로 확보했다. Encoder는 canonical 입력 호출 외의 AST가 같았고 Geometry 함수7개는 동일했다. Handoff CLI는 이 확장에서 호출하지 않았다. 이 기준의 원본7590파일과 기존 확장9892파일은 재검증 시작/종료 사이 불변이었다.

새 Manifest는 Common Source817개와 Extension hash를 요구한다. `legacy_fps.integration_guard.load_candidate`는 기본 비활성화이고 명시적 opt-in 후 Source 검사를 거친다. stale Encoder hash negative control을 실제 거부했다. 현재 Native21-feature 진단 Encoder에 기존36개 확장 진단 Checkpoint를 strict load할 수 있으며 정상 입력 출력이 같다. 변경 중인 다른 revision이나 옛14-feature BC까지 자동 호환을 보장하지 않는다. 통합 Patch는 원본 실행에 적용하지 않았다.

| 자료 | Experiment UUID |
| --- | --- |
| Main static/prefix geometry | `8c41299e9d3e4989bbf32ab1b894dc49` |
| Paired t0 | `71c3e498b85b4a49ab543374b1ff5d0e` |
| Full temporal | `41385af9132c4ff6b84bd944400711c3` |
| Diagnostic learning/probes | `b220c3fdced649adab3edee0c7d5378b` |
| Latency | `ff8379e4ae494841a61b4b9986666bed` |
| Current-source revalidation | `bd465b9d57ed4fd6843e87c950896edc` |

원자료는 Repository의 형제 디렉터리 `../../../task09c_legacy_fps/`에 보존했다. 로컬9892개 파일/1,600,253,183bytes의 원격 SHA 대조와 후속1107개 파일·HTML 링크 검사를 통과했다. Repository만 별도로 복제하면 이 외부 Artifact 경로는 함께 제공해야 한다. 기존 Run identity와 원격 provenance 절대 경로를 로컬 경로인 것처럼 바꾸지 않았다.

최초 bundle 검증은3331 JSON/4944 NPZ/155 PNG/10 MP4를 실제 readback/decode했다. Source/F5 join50220행, historical feature10025행, same-point Edge84건과 Geometry2520행을 재검사했다. 초기 계약54개 및 모델별 CUDA 순열 감사를 보존했다. 후속은 CPU에서54계약·2232 정상 Graph exact·108 전체 순열 검사와 Source Hash Guard를 확인했다. 최초 전체 보호 PARTIAL을 소급해 PASS로 바꾸지 않으며 **현재 검증 revision 호환성·후속 실행 구간 보호 PASS**를 별도 기록한다.

[최초 verification](../legacy_fps/publication.md#download) · [최초 전역 보호 감사](../legacy_fps/publication.md#download) · [소스 변경 후 verification](../legacy_fps/publication.md#download) · [새 구간 원본 보호](../legacy_fps/publication.md#download) · [기존 Artifact 보호](../legacy_fps/publication.md#download) · [현재 Source transition](../legacy_fps/publication.md#download)

[현재 Integration Manifest](../legacy_fps/publication.md#download) · [Hash Guard 검사](../legacy_fps/publication.md#download) · [기하 함수 AST 감사](../legacy_fps/publication.md#download) · [미적용 Implementation diff](../legacy_fps/publication.md#download) · [정적 HTML 원보고서](../legacy_fps/publication.md#download) · [후속 호환성 보고서](../legacy_fps/publication.md#download)

[최초 로컬 전송 검증](../legacy_fps/publication.md#download) · [후속 로컬 전송 검증](../legacy_fps/publication.md#download) · [10개 영상 decode/contact-sheet 감사](../legacy_fps/publication.md#download) · [시각 검수 기록](../legacy_fps/publication.md#download)

이 문서의 수치는 해당 저장 Run에서 집계했다. 파일럿의 기하 이점, GNN 정보 보존과 후속 Policy 도입 판단은 각각 구분한다.

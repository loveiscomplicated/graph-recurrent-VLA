# Task 09-C — Persistent Geometry / Multi-view / Graph·GNN Benchmark

2026-10-27 · independent feasibility diagnostic

## 1. 연구 질문과 실행 범위

Robot self-filter, object-local persistent surfels, 관측 기반 강체 추적, 신규 표면 통합, Main/Wrist 융합과 G0–G4/O1–O3 공통 Graph/GNN 진단을 구현하고 실제 실행했다. 최종 비교 UUID는 `e7b6f9ac66c848308e28da0716bbce0c`다.

로봇 제거와 Peg의 현재 다중 시점 관측에는 개선이 있었다. 시간 누적이 정확한 현재 형상을 보장하지는 않았다. Peg의 추적 drift와 Box의 잘못된 Wrist 마스크가 남았고, G4는 현재 관측 G3보다 Peg 평균 표면 오차가 컸다. 표현 전체의 단일 우승 후보를 선정할 근거는 부족하다.

Vast 57640710의 RTX 3090에서 perception과 진단 GNN을 실행했다. CPU PhysX + SAPIEN으로 수집했고, 최종 CPU 배열/기하 평가는 사용자 측정에 따라 M4 Pro 1개 작업자로 분리했다. CUDA PhysX와 Mac의 이번 simulator 실행은 미실행이다. GRU·Action Head·BC 학습·정책 rollout은 실행하지 않았다.

Train 8EP/981프레임, Validation 4EP/505프레임, 독립 final 2EP/258프레임이다. 반복 프레임은 독립 표본으로 해석하지 않는다. ±10mm 이동 분기는 parent episode와 같은 split에 묶었다. physical-invalid와 same-pose repeat는 학습 및 주 평가에서 제외했다. 최종 capture는 선택 동결(2026-10-27 12:13:13.086244 UTC) 이후 시작했다. 최종 평가는 기존 가중치·정규화·decoder를 그대로 적용했다.

Pilot은 개발 seed와 겹치며 별도 독립 반복 실험이 아니다. Pilot/validation 관찰은 구현 진단에 사용됐으므로 최종 일반화 평가는 동결 후 새로 수집한62091/62092로 한정한다. 기존 FrozenTest12와 debug seed30003은 최종 성능 비교에 쓰지 않았다.

[동결·실제 reset group 분리 감사](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/91dfd8218a8947859f3d3f8c57b3a929/independent_final_gate.json) · [에피소드·프레임 수](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/split_episode_summary.csv) · [전체 수치](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/benchmark_summary.json)

## 2. 실제 Sensor/Camera 지원

설치된 task/mount와 실제 observation에서 `base_camera`(Main), `hand_camera`(Wrist)를 확인했다. 두 센서는 같은 observation의 episode/timestep을 공유한다. RGB는 [128,128,3] uint8, raw depth는 [128,128] signed int16 mm다. 0 이하·overflow/비유한 depth는 유효 측정에서 제외한다. Intrinsic CV [3,3], world-to-camera CV [3,4], camera-to-world GL [4,4]를 저장하고 상호 정합을 검사한다. 각 카메라의 DINO/SAM2 tracker 상태는 분리했다.

원본 HDF5에는 두 카메라가 있지만 시간별 actor GT sidecar가 없어, 평가용 live expert capture를 별도로 만들었다. 기존 BC dataset에는 합치지 않았다. Wrist mount transform은 현재 robot link pose로 매 프레임 갱신한다. 독립 XML-URDF FK와 실제 robot link pose의 6 step 비교에서 최대 행렬 원소 차이는 7.691259e-7이었다.

native packed GL position과 CV depth 재구성의 평균 차이는 Main 0.594mm/Wrist 0.565mm였다. +0.5 pixel centre convention이 offset 0보다 일치했다. 이는 같은 시뮬레이터의 API 표현·양자화 비교이며 외부 측량 정확도 검증은 아니다. depth invalidity는 카메라/에피소드별 CSV로 제공한다.

[실제 센서 계약](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/8b742730936f4a52abde23ccfe25b3c6/collection/seed62091/sensor_contract.json) · [Depth invalidity](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/depth_invalidity_summary.csv)

## 3. 원본 G0 Baseline Identity

기존 Main Shape-First/F5/Raw EDE의 native packet과 14-feature graph를 그대로 별도 저장했다. 원격 preflight에서 09-B의 9개 저장 상태를 packet/graph와 standalone frozen Full-C encoder readout까지 정확히 재현했다(최대 오차 0). 새 capture의 original-G0 helper도 6개 상태에서 exact였다. 이전 09-B의 저장 decoder 120개/720행 예측은 float32에서 exact 재현했다. 정책/GRU는 불러 실행하지 않았다.

새 GNN 비교에는 별도 21-feature schema와 cap128/object를 사용한다. 변경 schema에 옛 BC checkpoint를 넣어 새 성능 순위를 만들지 않았다. 3개 robot node + 최대 256개 object node이며, 지원 point가 없으면 채워 넣지 않는다. 정확히 같은 수를 쓰는 geometry-only matched-count 분석도 별도로 저장했다. final 258프레임 중 양쪽 물체가 모두 nonempty인 matched 비교는 85프레임뿐이다.

CPU host를 옮긴 뒤 첫 20프레임을 감사했다. 모든 변형의 native node ID/좌표는 정합했고, G1–G4/O1–O3의 선택 node 집합과 semantic edge 집합은 같았다. 최대 정합 feature 차이는 약 1.19e-7이었다. **G0 Raw EDE는 semantic edge 차이가 누적 556개 있었다.** NumPy/SciPy 환경의 삼각분할 차이가 원인일 가능성이 있지만 확정하지 않았다. 원격 baseline exact 재현과 cross-host G0 동등성을 혼동하면 안 된다. 최종 19개 모델에는 모두 동일한 로컬 생성 평가 그래프를 제공했으나, G0의 train/eval host 차이는 순위 해석의 혼란 요인이다.

[Cross-host semantic 감사](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/prefix20_semantic_equivalence.json) · [19개 모델 식별자](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/99908cb9d49c4b1a868c2e2d86c2701f/model_manifest.json)

## 4. Robot Self-filter 방법과 성능

현재 qpos/FK와 알려진 개별 robot collision shape를 이용한다. depth projection과 각 shape 내부 판정을 결합하고 confirmed robot/uncertain/object 후보를 분리했다. 전체 로봇을 둘러싼 hull이나 큰 제거 반경은 사용하지 않았다. GT actor segmentation은 제거 입력에 전달하지 않고 평가에만 쓴다.

| 카메라 / 물체 | robot recall | 정상 표면 보존 | robot 오염 전→후 | 순도 전→후 | finger 20mm 근접 오제거 |
| --- | --- | --- | --- | --- | --- |
| base_camera / Peg | 97.4% | 99.3% | 1.0% → 0.0% | 97.8% → 98.9% | 10.9% |
| base_camera / Box | 97.4% | 100.0% | 2.4% → 0.0% | 96.1% → 98.4% | — |
| hand_camera / Peg | 94.2% | 100.0% | 0.0% → 0.0% | 99.3% → 99.3% | 10.6% |
| hand_camera / Box | 94.2% | 100.0% | 99.2% → 3.1% | 0.0% → 0.0% | — |

표는 final 258프레임의 정의 가능한 비율을 프레임별 평균한 값이다. 예를 들어 Main Peg 보존은 GT Peg가 유효 depth를 가진 214프레임, 순도는 마스크가 nonempty인 150프레임에서 정의된다. 모든 분모·defined_n은 CSV에 남겼다. Wrist Box 순도는 제거 후에도 0%인 실패 조건이 남았다. robot을 덜 남겼다는 사실이 Box를 올바르게 인식했다는 뜻은 아니다.

Finger centre에서 20mm 이내인 실제 GT object pixel의 confirmed removal은 Peg에서 약 10.9%/10.6%였다. 전체 보존율만으로 접촉 근처가 안전하다고 판단할 수 없다. 이는 기하적 근접 지표이며 실제 접촉력 검증이 아니다. 초기 367프레임 paired filter revision에서는 동일 입력 Main robot recall이 51.9%→98.3%, Peg 보존은 99.55%로 같았다. 초기 파일의 `contact_near`는 depth-proximity proxy였으므로 최종 finger 지표와 합치지 않는다.

[필터 전체 수치/분모](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/robot_filter_summary.csv) · [필터 전후 3D](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/seed62091/robot_filter_before_after.png)

## 5. Persistent Node Lifecycle

Map은 object-local XYZ/normal, persistent_node_id, confidence, first/last seen, camera/pixel source ID, observation_count, current/estimated 상태와 uncertainty를 보존한다. 영구 ID는 frame pixel ID와 분리했다. 마지막 물체 pose로 world 좌표를 복원하고 현재 EE frame으로 변환한다. 관측 초기 frame이 local origin이며 GT actor origin으로 초기화하지 않았다.

동일 표면은 local voxel/nearest-neighbour 및 normal 일치로 갱신하고, 신규 연결 표면은 추가한다. 가림만으로 삭제하지 않는다. FREE-SPACE 모순, identity 불일치와 등록 실패는 confidence 감소·reject·removed 상태로 기록한다. `CURRENTLY_OBSERVED`, `PERSISTENT_ESTIMATED`, `LOW_CONFIDENCE`, `REMOVED`를 구분한다. episode가 바뀌면 map/clock/ID를 초기화한다. 실제 이전 map을 다음 episode t0에 재사용한 결과는 fresh 생성과 exact였고 이전 ID 누출은 0이었다.

과거 depth-jump/normal-variation은 현재 유효 F5로 재사용하지 않았다. 현재 측정 normal과 과거 pose-dependent normal의 validity를 따로 둔다. status-off ablation은 추가 7 feature를 0으로 만들지만 원래 F5 availability에도 관측 여부가 남아 있으므로 모든 시간 정보를 제거한 실험은 아니다.

[ID 시간 변화](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/seed62091/persistent_ids.png) · [실제 reset 감사](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/actual_reset_inference.json)

## 6. Object Pose Tracking Error

NumPy/SciPy observable point-to-plane registration을 구현했다. SVD rank/방향별 관측 가능성, inlier/residual, pose jump/identity gate를 기록한다. 좁은 평면이나 대칭 표면의 미관측 방향은 확정하지 않는다. 초기 point-to-point 경로의 과신 실패를 보존하고 부분 관측 경로로 수정했다. GT로 practical pose를 복구하지 않았다. gripper close나 TCP 이동만으로 물체 부착을 가정하지 않았다.

| 변형 | 물체 | pose 이동 오차 mm | 회전 오차 ° | 기존 node 이동 갱신 오차 mm | 기록 uncertainty mm |
| --- | --- | --- | --- | --- | --- |
| G2 | Peg | 76.79 | 4.91 | 2.54 | 127.72 |
| G2 | Box | 0.54 | 0.53 | 0.09 | 158.62 |
| G4 | Peg | 60.68 | 8.54 | 2.30 | 104.23 |
| G4 | Box | 0.54 | 0.53 | 0.09 | 212.69 |
| O1 | Peg | 33.70 | 10.39 | 1.34 | 18.52 |
| O1 | Box | 0.48 | 0.51 | 0.11 | 162.10 |
| O2 | Peg | 0.00 | 0.00 | 0.04 | 12.27 |
| O2 | Box | 0.00 | 0.00 | 0.00 | 21.62 |
| O3 | Peg | 0.00 | 0.00 | 0.04 | 9.82 |
| O3 | Box | 0.00 | 0.00 | 0.01 | 22.91 |

수치는 cap 그래프에 대응하는 프레임 평균이며, motion-update는 공통으로 남은 ID에 대해 정의한다. Box의 pose 오차가 작아도 마스크에서 온 배경 표면은 큰 표면 오차를 만들 수 있다. Peg practical pose 오차는 60–77mm로 크다. confidence/uncertainty는 추정 상태 지표이고 통계적으로 보정된 확률이나 신뢰구간이 아니다. Oracle pose 오차 0은 GT motion을 넣은 정의상 결과다.
| 변형 | 물체 | low confidence 거리 mm | medium 거리 mm | high 거리 mm |
| --- | --- | --- | --- | --- |
| G2 | Peg | 39.88 (n=258) | 1.25 (n=48) | 0.65 (n=25) |
| G2 | Box | 11.08 (n=258) | 7.90 (n=4) | — (n=0) |
| G4 | Peg | 18.64 (n=253) | 5.18 (n=70) | 0.75 (n=56) |
| G4 | Box | 19.74 (n=258) | 10.99 (n=4) | — (n=0) |

신뢰도 bin은 low[0,0.25), medium[0.25,0.6), high[0.6,1]이다. 거리는 해당 bin이 존재하는 프레임의 평균이고 n은 그 프레임 수다. high bin의 작은 오차가 전체 map의 정확성을 뜻하지 않는다. Box는 high bin 자체가 없었다.

실제 visible 이동 예(seed62091/t25, Peg +10mm world X)의 공통 과거 ID 평균 변위는 G2 0.078mm/G4 1.819mm/O3 10.007mm였다. 같은 t24까지의 history에서 분기했으며 실용 추적의 과소 갱신을 보여 준다. [실제 이동 전후 node](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/actual_motion_before_after.png)를 제공한다. 이 한 예를 전체 이동 성능 평균으로 대신하지 않았다.

## 7. 신규 표면 통합 정확도

Depth validity, self-filter, object identity, normal/local support와 registration state를 확인한 source pixel만 map에 추가했다. GT mesh로 보지 못한 표면을 생성하지 않았다. 평가에서는 과거 GT-visible reference에 없던 실제 표면과 새 node를 비교했다.

Final의 신규 표면 양성 metric row는 전체 738개다. 이 수에는 t0 초기화와 frame-only 변형이 포함된다. **t>0의 practical G2/G4 양성 row는 92개**다. 이는 event/row 수이며 고유한 새 표면 개수나 무오염 성공률이 아니다. 새 correct node, duplicate-neighbour fraction, source purity, ghost를 함께 기록했다. 실측에는 올바른 신규 표면과 잘못된 마스크에서 생긴 추가 표면이 공존했다. 노드 수 증가만으로 개선을 판정하지 않았다.

[신규 표면 사례](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/actual_new_surface_cases.csv) · [추가/유지/오차 시계열](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/seed62091/temporal_geometry.png)

[실제 새 node 추가 전후](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/actual_new_surface_before_after.png)는 t0 이후 G4의 newly-correct-unseen 지표가 처음 양수인 frame을 선택했다. 새 node 전체가 correct라는 의미는 아니다.

## 8. 가림 중 Map 유지·오류

Final에서 GT segmentation상 두 카메라 모두에 보이지 않는 물체·프레임은 119개였다. 이 구간 practical map을 현재 관측으로 잘못 표시한 사례는 0개다. 마지막 pose를 유지하면서 관측 중단 age와 uncertainty가 증가했다. 정확한 현재 pose를 알아냈다는 증거는 아니다.

동일 robot/calibration/history에서 완전히 가려진 Box를 10mm 옮겨 rerender한 valid pair 6개는 RGB-D가 동일했다. practical G0–G4 tensor는 최대 차이 0으로 같았고 O3의 과거 node만 약 10mm 이동했다. 물체 이동을 모르는 결과가 이 조건의 센서 정보와 일치한다. 물리 actor pose/render 개입이며 step 없이 수행했으므로 동적 접촉력 결과로 해석하지 않는다.

재관측 case 17개의 회복 지연은 native-map mean mesh distance≤4mm, own-current-visible coverage≥0.5를 동시에 만족하는 최초 frame으로 정의했다. practical G2는 5구간 중3구간, G4는4구간 중2구간에서 기준을 충족했다. 회복하지 못한 구간도 CSV에 남겼다. 보이는 Peg의 point가 없었던 사례는 SAM 누락이며 완전 가림으로 분류하지 않았다.

[가림 사례](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/full_occlusion_cases.csv) · [가림 중 이동](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/hidden_motion_cases.csv) · [재관측 회복](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/reappearance_recovery.csv)

## 9. Wrist Camera 기여

Main-only G1→현재 두 카메라 G3에서 final Peg의 공통 visible coverage는 24.8%→82.3%, 평균 표면 거리는 1.26→1.13mm였다. Wrist가 Main에서 놓친 Peg 표면을 실제로 제공했다. 반면 Box coverage는 31.2%→31.2%로 거의 같았고 평균 표면 거리는 27.4→148.6mm로 악화됐다. Wrist Box의 잘못된 마스크가 주 실패 경로다. 따라서 multi-view 자체를 무조건 개선으로 판정하지 않았다.

Main에서 GT Peg와 유효 depth가 보이는214프레임 중64프레임은 G0/G1 Peg 입력이 비었다. G3는 두 센서 기준258개 visible 프레임에서 Peg empty가0이었다. Box의 own-sensor-visible empty는 G1 8/139, G3 7/139였다. 공통 두 카메라 reference 기준 empty와 입력 센서 자체에서 visible인 empty를 구분한다.

카메라 간 모순 identity는 unknown으로 분리했고, 겹치는 표면은 중복 fusion한다. Unknown conflict 비율과 native map의 spatial component/고립 node 수도 공개한다. 이 connectivity는 interval5의 k8/radius20mm 공간 이웃 지표이고, normal/witness/free-space를 쓰는 실제 graph edge 정책과 다르다.

[Fusion unknown](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/fusion_unknown_summary.csv) · [Native connectivity](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/native_map_connectivity_summary.csv)

## 10. G0–G4 Representation 비교

G0=원본 Main, G1=filter+현재 Main, G2=filter+Main map, G3=filter+현재 Main/Wrist, G4=filter+Main/Wrist map이다. 아래는 final의 **cap128/object** 기준이며 표면 거리는 sampled external collision-union surface까지의 거리다. 내부 component face는 제외했다. 4mm coverage 분모는 동일한 현재 두 카메라 GT-visible reference의 합집합이다. 두 카메라에서 각각 동시에 보이는 표면만을 뜻하지 않는다. ghost는 과거 추정 node 중 현재 GT 표면에서 8mm 초과인 node다.

| 표현 | 물체 | 평균 N | 거리 평균 mm | 프레임 p95 평균 mm | 공통 visible coverage | source 순도 | ghost 수 | 거리 정의 프레임 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | Peg | 72.6 | 1.36 | 2.98 | 24.8% | 95.7% | 0.0 | 150 |
| G0 | Box | 57.6 | 32.55 | 32.81 | 30.8% | 95.2% | 0.0 | 131 |
| G1 | Peg | 72.1 | 1.26 | 2.27 | 24.8% | 96.7% | 0.0 | 150 |
| G1 | Box | 57.5 | 27.41 | 7.37 | 31.2% | 97.7% | 0.0 | 131 |
| G2 | Peg | 128.0 | 39.74 | 48.79 | 40.0% | 96.2% | 39.5 | 258 |
| G2 | Box | 128.0 | 11.14 | 3.66 | 16.0% | 98.8% | 1.1 | 258 |
| G3 | Peg | 128.0 | 1.13 | 2.45 | 82.3% | 97.0% | 0.0 | 258 |
| G3 | Box | 58.0 | 148.64 | 134.71 | 31.2% | 68.9% | 0.0 | 185 |
| G4 | Peg | 128.0 | 17.58 | 59.43 | 40.7% | 89.6% | 34.8 | 258 |
| G4 | Box | 128.0 | 19.63 | 112.03 | 15.9% | 95.9% | 4.7 | 258 |
| O1 | Peg | 128.0 | 4.01 | 16.21 | 44.1% | 100.0% | 17.7 | 258 |
| O1 | Box | 128.0 | 1.91 | 3.51 | 13.8% | 100.0% | 0.0 | 258 |
| O2 | Peg | 128.0 | 1.48 | 7.61 | 51.6% | 90.8% | 6.7 | 258 |
| O2 | Box | 128.0 | 176.65 | 126.16 | 15.6% | 94.6% | 6.3 | 258 |
| O3 | Peg | 128.0 | 0.69 | 1.12 | 48.8% | 100.0% | 0.0 | 258 |
| O3 | Box | 128.0 | 1.92 | 3.61 | 14.1% | 100.0% | 0.0 | 258 |

각 열은 프레임별 통계의 평균이다. p95 열은 모든 point를 합친 pooled p95가 아니다. nonempty에서만 거리/순도가 정의되므로 defined_n을 함께 읽어야 한다. missing 입력의 coverage는 0이다. native variable-N 및 matched-count 결과는 같은 CSV에서 분리한다. cap은 memory 크기와 별개이므로 O3도 넓은 누적 표면에서 128개를 고르면 현재 visible coverage가 낮아질 수 있다.

G3는 Peg 현재 형상에서 가장 좋은 실용 후보이나 Box의 큰 오류 때문에 전체 우승 표현으로 추천할 수 없다. G2/G4는 누적 coverage와 ghost/drift의 교환관계를 보인다. Point budget이나 probe 점수만으로 map 정확성을 대신하지 않는다.

[전체 native/cap/matched 수치](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/geometry_summary.csv) · [비교 그림](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/independent_geometry_comparison.png)

## 11. Estimated vs Oracle 비교

O1은 GT-visible segmentation + estimated motion, O2는 predicted mask + GT relative motion, O3는 둘 다 GT인 진단군이다. 모두 지금까지 관측한 표면만 이동시키며 full mesh를 node로 넣지 않는다. practical과 oracle 결과를 합산하지 않았다.

Peg G4→O1→O3 평균 표면 거리는 17.58→4.01→0.69mm였다. segmentation과 motion 모두 개선 여지가 있다. Box는 G4 19.63mm, O1 1.91mm, O3 1.92mm로 segmentation의 영향이 크다. O2 Box는 176.65mm였다. GT motion을 사용해도 잘못된 관측 표면이 정답 표면으로 바뀌지 않는다. 이 평균은 먼 배경 point에 민감하므로 순도·p95·native 결과를 함께 확인한다. 실제 lifecycle에서 purge 후 반복 초기화는 관찰되지 않았으며, 이를 관측된 실패 원인으로 단정하지 않았다.

Oracle은 측정된 진단 상한 조건이고, 사용 가능한 production 성능이나 해결된 tracking으로 표현하지 않는다.

## 12. Graph Node/Edge 품질

새 schema는 `task09c_persistent_surface_v1`이며 원래14 feature에 상태/신뢰도/age/uncertainty/pose confidence 7개를 더한다. 새 surface edge는 3D k8/radius20mm, normal/tangency, 중간 surface witness와 camera free-space 검사를 쓴다. Robot–object k4, Peg–Box M16 helper는 재사용했다. 원본 Raw EDE와 별도 policy다. G0→G1의 GNN 비교에는 filter와 edge policy 변화가 함께 있으므로 순수 filter 인과 ablation으로 읽으면 안 된다. 순수 제거 효과는 동일 mask의 filter 지표로 판단한다.

Final Box, cap128, interval5 53개 frame의 edge audit:

| 표현 | 무방향 surface edge | component | GT unsupported | free-space crossing | aperture crossing | 관측 rim coverage |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | 137.2 | 4.4 | 27.2 | 8.25 | 2.85 | 32.0% |
| G1 | 20.4 | 48.0 | 1.2 | 0.09 | 0.00 | 33.2% |
| G2 | 17.7 | 110.4 | 2.2 | 0.04 | 0.32 | 16.4% |
| G3 | 20.5 | 48.4 | 1.5 | 0.00 | 0.00 | 32.5% |
| G4 | 12.9 | 115.1 | 3.3 | 0.04 | 0.00 | 16.0% |
| O1 | 41.1 | 89.8 | 4.6 | 0.09 | 0.45 | 13.1% |
| O2 | 9.8 | 118.3 | 1.4 | 0.00 | 0.00 | 16.3% |
| O3 | 43.8 | 88.8 | 3.3 | 0.15 | 0.00 | 17.6% |

부적절한 연결을 줄이는 대신 graph가 심하게 분리됐다. G1의 aperture crossing 0과 rim coverage 33.2%는 hole 형상을 충분히 보존했다는 뜻이 아니다. G4는 component 약115개, rim coverage16.0%다. GT aperture/mesh는 사후 감사에만 쓰며 hole goal이나 전용 sampling feature를 입력하지 않았다.

[Edge/rim 전체 감사](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/boundary_summary.csv) · [Raw EDE와 새 edge](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/seed62091/raw_ede_vs_surface_edges.png) · [Aperture 3D](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/seed62091/aperture_boundary.png)

## 13. Geometry Probe 결과

기존09-B의 ridge(6개 alpha)와 small MLP(hidden64, weight decay 2조건, 최대2000/patience150)를 재사용했다. Train-only scaler, Validation-only decoder 선택 후 동결했다. P0=robot76, P1=readout, P2=readout+robot76, P3=object slots, P4=입력 centroid, P5=입력 geometry summary다. Constant와 RawCentroidDirect는 비학습 대조군이다. 아래 absolute는 final baseline 258프레임, Δ는 물체를 실제 옮긴 valid pair 중 해당 표현 입력 센서에서 baseline/branch 모두 GT-visible인 조건이다(Peg n8/Box n6). P0는 object-sensor stratum을 정의하지 않고 ALL_VALID_MOVED(Peg n8/Box n12)를 사용한다.

| 표현 | Probe/linear | absolute Peg mm | absolute Box mm | Δ Peg mm | Δ Box mm |
| --- | --- | --- | --- | --- | --- |
| G0 | P0_robot76 | 46.27 | 135.28 | 10.00 | 10.00 |
| G0 | P4_centroid | 22.32 | 140.24 | 5.75 | 16.30 |
| G0 | RawCentroidDirect | 61.64 | 317.97 | 3.75 | 99.42 |
| G0 | P5_geometry_summary | 23.38 | 131.89 | 5.13 | 23.89 |
| G0 | P3_object_slots E1 | 27.55 | 119.41 | 6.66 | 15.80 |
| G0 | P1_readout E1 | 21.87 | 124.75 | 6.39 | 15.14 |
| G0 | P2_readout_robot76 E1 | 26.67 | 128.65 | 7.30 | 14.90 |
| G1 | P0_robot76 | 46.27 | 135.28 | 10.00 | 10.00 |
| G1 | P4_centroid | 22.09 | 139.76 | 5.77 | 16.61 |
| G1 | RawCentroidDirect | 61.81 | 316.51 | 3.56 | 100.13 |
| G1 | P5_geometry_summary | 22.97 | 135.64 | 5.07 | 16.20 |
| G1 | P3_object_slots E1 | 29.06 | 113.70 | 6.82 | 14.66 |
| G1 | P1_readout E1 | 23.02 | 120.51 | 6.36 | 14.73 |
| G1 | P2_readout_robot76 E1 | 27.31 | 133.04 | 7.12 | 13.52 |
| G3 | P0_robot76 | 46.27 | 135.28 | 10.00 | 10.00 |
| G3 | P4_centroid | 15.77 | 143.73 | 6.06 | 16.01 |
| G3 | RawCentroidDirect | 32.55 | 327.76 | 4.34 | 100.47 |
| G3 | P5_geometry_summary | 17.45 | 149.36 | 4.83 | 26.16 |
| G3 | P3_object_slots E1 | 12.48 | 172.16 | 5.87 | 19.63 |
| G3 | P1_readout E1 | 11.40 | 183.38 | 5.67 | 19.89 |
| G3 | P2_readout_robot76 E1 | 24.33 | 127.38 | 5.76 | 12.88 |
| G4 | P0_robot76 | 46.27 | 135.28 | 10.00 | 10.00 |
| G4 | P4_centroid | 30.23 | 109.35 | 8.41 | 21.39 |
| G4 | RawCentroidDirect | 51.01 | 90.12 | 5.73 | 99.88 |
| G4 | P5_geometry_summary | 28.91 | 59.93 | 7.51 | 75.27 |
| G4 | P3_object_slots E1 | 21.63 | 33.72 | 8.55 | 11.76 |
| G4 | P1_readout E1 | 20.75 | 31.80 | 8.50 | 17.20 |
| G4 | P2_readout_robot76 E1 | 35.59 | 47.98 | 9.92 | 14.92 |

동일 robot 개입에서 P0와 Constant의 predicted Δ는 0이고 오류는 약10mm다. Raw centroid는 Peg에서 3.56–5.73mm로 readout보다 좋은 경우가 많다. 그러나 Box visible-pair에서는 약99–100mm까지 악화돼 segmentation/현재 point 선택에 민감했다. 완전 가림의 0-response와 보이는 입력에서의 0-response는 분리해야 한다.

모든 Ridge/MLP, P0–P5, Constant, RawCentroidDirect의 absolute mean/median/p90/p95/max, 축별 MAE/R², paired Δ/방향 cosine/gain/축별 오차, unmoved-target control 및 episode별 결과를 CSV로 제공한다. R²는 final 두 episode의 좁은 축 분산에서 크게 음수가 될 수 있다. low error만으로 object 정보의 인과적 활용이나 action 성능을 주장하지 않는다.

[Absolute 전체](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/91dfd8218a8947859f3d3f8c57b3a929/absolute_metrics.csv) · [개별 paired Δ](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/91dfd8218a8947859f3d3f8c57b3a929/paired_delta_metrics.csv) · [Visible/hidden/support strata](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/paired_probe_summary.csv) · [에피소드별](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/sources/91dfd8218a8947859f3d3f8c57b3a929/per_episode_metrics.csv) · [Probe 대조 그림](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/probe_controls_absolute_delta.png)

## 14. E0/E1/E2 GNN 비교

19개 모델을 모두 scratch에서 2000 update, batch16, seed42, AdamW lr0.001/wd1e-5, AMP/TF32 off, deterministic 조건으로 학습했다. 매50 update의 validation loss 최솟값을 선택하고 총 budget은 줄이지 않았다. 위치6과 current-visible surface summary30(물체당 XYZ mean/min/max/bbox centre/span15)의 train-normalized MSE를 각각0.5로 합했다. GT target은 감독 학습·평가에만 쓰며 encoder 입력에는 없다. 최종 data로 refit/selection하지 않았다.

E0는 기존 message passing 구조를 새 schema에 opt-in 적용했다. E1은 같은 node encoder/pooling에서 message passing을 제거한다. E2는 E0 readout에 같은 입력 XYZ의 고정 summary30을 연결한다. 총 parameter는 E0 79,428/E1 28,740/E2 81,348, readout은 320/320/350이다. 동일 update 비교이며 parameter 수가 같은 실험은 아니다.

P1 Ridge의 final 결과(absolute=baseline, Δ=own-sensor-visible moved pair):

| 표현 | Encoder | absolute Peg mm | absolute Box mm | Δ Peg mm | Δ Box mm | 선택 step |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | E0 | 34.05 | 127.25 | 12.58 | 21.18 | 700 |
| G0 | E1 | 21.87 | 124.75 | 6.39 | 15.14 | 1850 |
| G0 | E2 | 33.42 | 121.99 | 8.35 | 21.21 | 650 |
| G1 | E0 | 31.52 | 127.19 | 13.21 | 14.50 | 1150 |
| G1 | E1 | 23.02 | 120.51 | 6.36 | 14.73 | 1950 |
| G1 | E2 | 24.44 | 135.09 | 8.55 | 10.76 | 1950 |
| G2 | E0 | 31.78 | 106.77 | 11.50 | 10.67 | 200 |
| G2 | E1 | 24.33 | 46.88 | 9.40 | 20.77 | 2000 |
| G2 | E2 | 27.01 | 85.20 | 7.39 | 18.33 | 1800 |
| G3 | E0 | 20.36 | 172.84 | 7.87 | 29.60 | 1950 |
| G3 | E1 | 11.40 | 183.38 | 5.67 | 19.89 | 1850 |
| G3 | E2 | 14.00 | 194.95 | 8.07 | 31.35 | 1300 |
| G4 | E0 | 33.18 | 89.98 | 15.60 | 23.52 | 1950 |
| G4 | E1 | 20.75 | 31.80 | 8.50 | 17.20 | 1750 |
| G4 | E2 | 20.80 | 83.50 | 8.41 | 13.31 | 1900 |
| O1 | E0 | 25.65 | 68.31 | 12.95 | 12.04 | 850 |
| O2 | E0 | 41.02 | 77.03 | 10.02 | 28.50 | 1450 |
| O3 | E0 | 20.28 | 69.42 | 11.42 | 12.46 | 1450 |
| G4_status_off | E0 | 23.45 | 101.50 | 15.83 | 25.52 | 1750 |

같은 diagnostic head의 final visible-summary15 MAE(mm):

| 표현 | Encoder | Peg | Box |
| --- | --- | --- | --- |
| G0 | E0 | 16.16 | 50.55 |
| G0 | E1 | 15.57 | 43.09 |
| G0 | E2 | 18.66 | 48.64 |
| G1 | E0 | 16.30 | 46.39 |
| G1 | E1 | 16.96 | 41.99 |
| G1 | E2 | 16.15 | 47.09 |
| G2 | E0 | 12.57 | 56.84 |
| G2 | E1 | 13.64 | 45.03 |
| G2 | E2 | 16.83 | 57.33 |
| G3 | E0 | 11.83 | 51.63 |
| G3 | E1 | 12.81 | 40.19 |
| G3 | E2 | 12.35 | 47.50 |
| G4 | E0 | 13.49 | 62.49 |
| G4 | E1 | 11.71 | 46.61 |
| G4 | E2 | 13.69 | 55.92 |
| O1 | E0 | 12.57 | 54.42 |
| O2 | E0 | 13.75 | 42.76 |
| O3 | E0 | 13.04 | 50.24 |
| G4_status_off | E0 | 13.69 | 49.42 |

E1은 여러 조건의 위치/Δ/형상 지표에서 E0보다 좋았다. E2의 geometry skip은 일부 이점이 있지만 일관된 우승은 아니다. G4 status-off와 full의 차이도 제한적이며 단일 seed/작은 물리 episode 수로 유의한 일반화 우위를 확정할 수 없다. Surface head는 current-visible 요약을 복원하는 감독 target이며 전체 unobserved mesh 복원 성능이 아니다.

[GNN/Δ 그림](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/gnn_absolute_and_visible_delta.png) · [Slot/readout 진단](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/object_slots_and_readout.png) · [Validation episode별](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/figures/validation_per_episode.png) · [형상 decoder CSV](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/surface_decoder_metrics.csv)

## 15. Stage별 실패 사례와 부정 실험

초기 P2P tracking의 과신, Wrist Box 오분류, Peg drift, ghost, sparse/disconnected edge를 실패 결과로 보존했다. 첫 full G0/E0 학습은 2000 update 후 전체-node permutation 감사에서 실패했다. 기존 readout이 첫3행 robot slot을 가정했고 이전 fixture는 object만 섞었다. 새 encoder에서 node type으로 robot slot을 정렬하는 opt-in adapter를 추가하고 전체 node permutation fixture로 바꿨다. canonical graph에서는 forward가 bitwise 같았고 parameter identity도 같았다. CUDA permutation 최대 차이는7.152557e-7이었다. 19개를 같은 budget으로 처음부터 다시 학습했고 재시도 G0/E0 selected hash는 첫 시도와 같았다. 실패 checkpoint를 warm-start/추가 선택에 사용하지 않았다.

보고서 생성 중 empty-edge plot과 FFmpeg thread 한도로 실패한 시도도 보존했다. plot의 빈 edge 처리와 encoder/decoder thread1을 적용한 뒤 영상 전 frame decode를 재검증했다. 기하/모델 결과는 이 수정으로 재계산하지 않았다.

| 부정 실험 | 증거 종류 | 결과/제약 |
| --- | --- | --- |
| 1 완전 가림 | 실제 RGB-D/GT visibility | 119 object-frames; 현재 관측 오표시0; 정확 pose 보장은 실패/미확정 |
| 2 가림 중 이동 | 실제 actor 개입/rerender | 6 hidden pair; practical 입력/graph 동일, oracle만10mm 이동 |
| 3 잘못된 mask | 실제 predicted mask | 255 robot-contaminated mask; Wrist Box 오염 해결 안 됨 |
| 4 재관측 충돌 | 실제 reject 및 별도 fixture | 55 rejection event/17 reappearance case; 자동 회복 일반화 미확정 |
| 5 신규 표면 | 실제 RGB-D+사후 mesh | correct addition 관찰; t0 제외 통계 별도, 오염도 남음 |
| 6 episode boundary | 실제 map 재사용/새 EP 입력 | fresh state exact, old ID leak0 |
| 7 wrong alignment | 실제 calibration의 통제 주입 | CV/GL 일관성 유지한50mm fault를 mount anchor로 탐지 |
| 8 wrong history | 실제 map 복사본의 통제 주입 | 미래 source event 거부; 원본 미수정 |

Synthetic contract PASS는 실제 센서 성능 PASS와 구분했다. 38개 현재 contract와 15개 기존09-A/B test가 통과했다. 27개 요구 contract는 fixture/실제 검증 범위를 catalog에 매핑했다. 실제 large disagreement 회복의 보편적 성공, 동적 contact, unseen class로의 identity generalization은 검증하지 않았다.

[27개 계약 catalog](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/contract_catalog.json) · [현재 test log](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/current_contracts.log) · [기존 test log](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/prior_contracts.log) · [8개 부정 실험 원자료](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/negative_cases/independent/negative_experiments.json) · [permutation 수정 검증](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/encoder_permutation_fix_verification.json)

## 16. 계산 비용 및 Memory

동일 첫20프레임/동일 평가 코드에서 원격4개 동시 작업자는7.78s/frame, M4 Pro1개 작업자는0.89s/frame이었다. 이 구간의 per-frame wall은 로컬이 약8.7배 빨랐다. 전체 pipeline 속도 향상 비율은 측정하지 않았다. 원격 cgroup CPU limit은192000/100000≈1.92 core로 표시16core와 다르다. 로컬 profile은 graph 구성31%, geometry metric25%, transform/validation 약30만 회를 기록했다.

최종258프레임의 로컬 geometry mean1.446s/frame, p951.908s/frame, stage wall377.916s였다. 다른 데이터/길이이므로0.89s와 직접 같은 조건으로 비교하지 않는다. 배열17,177개 파일의 SHA를 확인해 원격 frozen evaluation에 전달했다. 입력 수신·압축·출력 전송 시간은 이 worker frame 시간에 포함되지 않는다. Mac swap 약5.2–5.6GB 조건에서1개 worker만 사용했다.

| 단계 | wall s |
| --- | --- |
| smoke_boundary_audit | 5.1 |
| development_capture | 1072.9 |
| development_geometry | 4883.1 |
| development_branches | 625.4 |
| development_boundary_audit | 72.7 |
| diagnostic_training | 818.2 |
| frozen_probes | 247.9 |
| independent_capture | 168.8 |
| independent_geometry | 377.9 |
| independent_branches | 58.9 |
| independent_boundary_audit | 5.2 |
| frozen_final_application | 23.9 |

위 wall은 각 ExperimentLogger 시작/종료 차이이며 worker CPU time·GPU kernel time·end-to-end 전송 시간을 구분한다. 19개 GNN 학습은 약818s, probe 약248s였으며 실패·pilot·진단·보고서 비용은 별도다. Torch CUDA peak allocated는 모델별 약118MB였고 process RSS high-water는 모델 실행 이력의 영향을 받는다. 전 GPU 사용량과 task PID-tree RSS의10초 sampled monitor도 제공한다. 다른 프로세스의 사용량이 포함될 수 있는 global GPU memory를 전부 본 작업에 귀속하지 않았다. 원격 cgroup memory 약42.7GiB와 host128GiB도 구분했다.

10초 샘플의 owned RSS 합 최대는 development geometry5.23GB, diagnostic training1.76GB, final capture3.14GB, frozen final apply1.38GB였다. global GPU memory 관측 최대는 final capture14,603MiB, training512MiB, final apply410MiB였다. sampling이므로 순간 최대값을 놓칠 수 있다. 로컬로 옮긴 CPU 단계의 원격 RSS는 대기 프로세스만 측정한 값이며 Mac worker 메모리로 해석하지 않는다.

[사용자 CPU profile 원본](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/cpu_profile_summary.json) · [에피소드별 worker 시간](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/summary/compute_costs.csv) · [자원 monitor 목록](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/resource_monitor_index.json)

## 17. 다음 연구 후보와 최종 질문 Q1–Q14

다음 소규모 후보는 G1+E1을 기준으로 두고, Wrist object identity/마스크 유효성을 먼저 개선한 G3+E1을 비교하는 것이다. robot 근접 false removal과 Box contamination을 먼저 줄이고, 같은 host G0 edge 재현 및 다수 독립 episode/학습 seed에서 검증해야 한다. G4는 pose observability, contamination rejection, historical confidence와 graph fragmentation 해결 전 정책 성능 향상을 예측할 근거가 부족하다. 이는 다음 실험 후보이며 현재 production/공통 인터페이스를 정하는 결정이 아니다.

| 질문 | 판정 | 실측 해석 |
| --- | --- | --- |
| Q1 | CONFIRMED | Robot 제거와 Main mask 순도 개선을 측정했다. 근접 Peg 오제거와 Wrist Box 오인식은 남았다. |
| Q2 | INCONCLUSIVE | 가림 중 node/불확실성 상태 유지는 구현·측정했다. 올바른 현재 geometry의 지속 보장은 ghost/drift 때문에 성립하지 않는다. |
| Q3 | INCONCLUSIVE | 관측 가능한 방향의 갱신은 수행하나 Peg pose 오차60–77mm, 완전 가림 이동은 식별 불가다. |
| Q4 | SUPPORTED | 실제 새 correct surface addition은 확인했다. 오염·중복 없는 일반적 통합은 입증하지 못했다. |
| Q5 | SUPPORTED | Peg current coverage24.8%→82.3%; Box는 개선 없이 오차 증가. 물체 조건부 결과다. |
| Q6 | INCONCLUSIVE | Peg current geometry는 G3가 유리하다. Box contamination과 persistence drift로 전체 우승은 없다. |
| Q7 | CONFIRMED | 단순 과거 유지가 ghost를 만들었다. G2 Peg39.5/G4 Peg34.8 cap node 평균이며 해결된 문제가 아니다. |
| Q8 | CONFIRMED | O1/O2/O3 차이를 실측했다. GT mask/motion을 사용할 때만 얻는 격차다. |
| Q9 | SUPPORTED | E1이 여러 조건에서 E0보다 좋았다. E2 skip과 status의 우위는 보편적이지 않다. |
| Q10 | SUPPORTED | 위치/Δ/visible shape를 모두 측정했다. Peg raw centroid가 readout보다 나은 조건과 Box 실패가 함께 존재한다. |
| Q11 | SUPPORTED | 입력 누락/오염의 직접 증거가 강하다. 새 encoder 비교는 구조 한계의 진단 증거지만 옛 BC/GRU 원인의 인과 분해는 아니다. |
| Q12 | INCONCLUSIVE | 빈 공간 crossing은 줄었지만 graph fragmentation과 낮은 aperture rim coverage가 남아 일반적 형상 개선을 확정할 수 없다. |
| Q13 | CONFIRMED | filter·Peg current multiview 개선은 실측; accurate persistent pose/robust new surface는 부분적; O1–O3는 oracle 조건이다. |
| Q14 | SUPPORTED | 추가 진단은 G1+E1 기준과 object-wise mask 검증을 강화한 G3+E1 비교가 타당하다. 현 G4의 정책 투입 근거는 부족하다. |

## 18. 실행/미실행 항목과 단계 판정

| Stage | 상태 | 근거 |
| --- | --- | --- |
| 0 Audit/baseline | PASS | 원격9개 baseline/6개 helper exact; camera/FK/frozen identity |
| 1 Self-filter | PARTIAL | Main robot 제거 개선; 접촉 근처 오제거와 Wrist 오인식 |
| 2 Persistent map | PARTIAL | lifecycle/causality/reset 구현; Peg drift/ghost |
| 3 Wrist fusion | PARTIAL | Peg coverage 개선; Box contamination |
| 4 Graph construction | PARTIAL | schema/budget/감사 통과; 단절·rim 보존 한계 |
| 5 GNN/probes | PASS | 19개 equal-update 진단과 frozen independent apply 완료; 우수 성능 보장 아님 |
| 6 Final analysis | PASS | 실패/Oracle/host 차이/원자료·영상·보호 감사 포함 |

실행: Linux CPU PhysX/SAPIEN capture, RTX3090 DINO/SAM2·진단 GNN, Mac CPU 배열 기하 평가, frozen probe, native/cap/matched 기하·edge audit, 실제 8 negative의 관찰/통제 주입 및 synthetic contract, 저장 결과 readback.

미실행: CUDA PhysX/vectorized simulator, 이번 Mac simulator/GPU, GRU/action head/BC/policy rollout, closed-loop 성공률, 동적 hidden-move contact 실험, 외부 실물 센서/물체, 여러 학습 seed 통계 검정, 20Hz 최적화. 2026-10-27 실험 실행 중에는 dependency 설치·Git commit·HF/W&B 외부 publishing을 하지 않았다. 이후 로컬 커밋과 HF 비공개 게시 기록은 §20에 구분했다. 다른 작업을 kill/pause/suspend하지 않았다. CPU 단계를 분리하는 동안 CUDA 학습은 원격에 유지했다.

## 19. Source/Artifact 보호 감사 및 재현 자료

공통 `utils.experiment_logger.ExperimentLogger`를 재사용해 단계별 UUID와 config/metadata/manifest를 보존했다. W&B disabled와 offline model cache로 실행했고 publishing utility의 별도 구현을 추가하지 않았다. 완성된 원격 comparison(`/workspace/task09c/comparison_runs/e7b6f9ac66c848308e28da0716bbce0c`)에는 완료된 실험 원자료, per-frame input/mask/filter/map/tracker/provenance/graph, GT sidecar, GNN/probe checkpoint와 frozen feature, source snapshot, dependency/deployment identity를 보존했다. `path_remap.json`으로 원격 provenance prefix를 bundle의 `sources/<uuid>`로 해석한다. 원래 source manifest의 절대 경로를 새 경로인 것처럼 바꾸지 않았다.

로컬 전달본은 보고서·영상·모든 GNN/probe 가중치·지표·source snapshot과 최종 독립 CPU 4개 run의 원자료를 포함한다. 23,893개 파일/1,201,058,577 bytes를 SHA 검증했다. 개발·pilot의 per-frame 원자료143,648개 파일은 원격 완성본에 보존하며 로컬에 모두 복제하지 않았다. [로컬 전달 범위](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/local_bundle_transfer.json), [원격 원자료 전송시점 해시 목록](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/transfer_file_manifest.json), [최종 로컬 파일 해시](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/final_file_hashes.json)를 제공한다. 전송시점 목록은 이후 완성한 보고서/audit 파일의 최신 해시를 뜻하지 않는다. 원격30개 완료 run은 전체 readback을 통과했다. 모델 feature와 실제 입력 graph/target의 byte hash도 개발29,374행/최종5,472행에서 일치했다. 로컬 독립 평가 graph/model 연결은 다시 검증했다.

기존 Full/Single checkpoint, normalizer, frozen dataset/splits, 중지된 Full status ERROR,09-A/B artifacts는 protection audit로 확인했다. 초기 local HEAD는 `bce1ae7860db4c3d458b9efc252131660336f703`, 작업 중 사용자가 별도로 커밋한 완료 당시 기준 HEAD는 `58e645b4eba1bc5a06c827bb4082508122de467a`다. 새 package가 그 사용자 commit에 포함됐으므로 그 시점의 잔여 diff만으로 최초 구현 전체를 복원할 수 없다. 초기 HEAD 대비 새 package 전체 diff와 완료 당시 사용자 HEAD 대비 잔여 diff를 함께 남겼다. visualization/publishing 쪽 동시 사용자 변경은 본 작업으로 주장하거나 되돌리지 않았다.

초기 pilot 중 execution-time snapshot이 없는 결과는 NOT_AVAILABLE로 표시했다. 뒤늦게 source를 복사해 당시 기록인 것처럼 주장하지 않는다. 이후 run별 snapshot과 동결 release/overlay hash가 재현 기준이다. source bundle은 .git 없는 배포 환경이라 자동 git metadata가 null인 곳이 있으며, 별도 deployment identity로 보완했다. 모델 선택/학습 예산을 바꾼 metadata는 없다.

[최종 verification](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/verification.json) · [원본 frozen 보호](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/original_frozen_protection.json) · [로컬 보호·동시 사용자 변경](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/local_protection_audit.json) · [원격 전체 readback](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/source_readback_verification.json) · [Source 경로 매핑](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/path_remap.json) · [Implementation diff](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/audits/implementation_initial.diff) · [Dependency identity](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/dependency_identity.json) · [요구사항별 완료 감사](../../experiments/e7b6f9ac66c848308e28da0716bbce0c/requirement_audit.json)

이 문서의 모든 수치는 저장된 source run에서 집계한 값이다. 향후 최적화·설계는 이번 feasibility 결과와 분리한다.

## 20. Task09-C 비공개 게시 — 2026-10-28 KST

Task09-C 본체의 CPU handoff·자원 기록, 전체 node permutation 대응, 기하 집계·영상·readback 보완과 README·테스트를 로컬 구현 커밋 `a1ade094f1a84af4ce2cee5dbe502a54bec72ebf` (`fix(graph): finalize Task09-C evaluation and CPU handoff`)으로 확정했다. 완료 당시 `audits/package_source/`의 35개 파일과 현재 package가 일치했다. 보고서 bundle의 `publication_commit`에는 이 SHA를 기록했으며 실험 당시 Git metadata·UUID·W&B project/group은 유지했다.

기존 비공개 Model·Dataset 저장소에 아래 세 묶음을 게시했다. 파일 수·용량은 config와 재현성 manifest를 포함한 실제 게시 selection 기준이다. 세 receipt 모두 `uploaded`이며 skipped 파일은 0개다. 저장소 생성·공개 범위 변경·Space 생성은 하지 않았다.

| 묶음 / 원본 UUID | 고정 revision의 HF 경로 | 게시 파일 수 | 게시 bytes |
| --- | --- | ---: | ---: |
| 비교 보고서·소스·지표 / `e7b6f9ac66c848308e28da0716bbce0c` | [Dataset: `fbec145d2343689b410c389ec32b423342c87288`](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/fbec145d2343689b410c389ec32b423342c87288/integration/e7b6f9ac66c848308e28da0716bbce0c) | 10 | 75,138,085 |
| 선택 GNN / `99908cb9d49c4b1a868c2e2d86c2701f` | [Model: `41e190a85eba22daa0a961b33532201ea75ac6f6`](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/41e190a85eba22daa0a961b33532201ea75ac6f6/integration/99908cb9d49c4b1a868c2e2d86c2701f) | 83 | 11,011,267 |
| Probe decoder·선택 기록 / `790dbcb6cc3f42d3b09f52087c6f03c8` | [Model: `475c6a001e0cb1c902c2aa032401d58cdda54a1e`](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/475c6a001e0cb1c902c2aa032401d58cdda54a1e/integration/790dbcb6cc3f42d3b09f52087c6f03c8) | 603 | 12,099,909 |
| 합계 | 기존 비공개 저장소 2개, 게시 commit 3개 | **696** | **98,249,261** |

GNN은 `model_manifest.json`의 19개 `selected.pt`만 선택했다. Probe는 `probe_manifest.json`의 324개 `decoder.npz`를 원래 Probe UUID의 `kind="checkpoint"`로 명시 등록했다. 그중 24개는 다른 선택 항목과 바이트가 같으므로 공통 게시기의 content hash 중복 제거 후 원격 decoder 파일은 300개다. 324개 원본 선택 항목은 모두 원격 파일에 연결된다. 전체 NPZ 분류 규칙은 바꾸지 않았다. 가중치와 내부 정규화 배열은 원본 그대로이며 실패·smoke 가중치는 포함하지 않았다.

공통 `tools.publish_experiment`와 `utils.experiment_artifacts`만 사용했다. 원본 실험 디렉터리를 보존하고 무시되는 `outputs/task09c_publication/runs/<기존 UUID>/`에 게시용 복사본을 만들었다. Metadata의 `outputs`·`artifact_kinds`만 재구성해 비교 보고서의 원격 prefix 등록 6개를 해결했다. 원본 config·metadata·manifest는 ZIP의 `original_records/<UUID>/`에 별도로 보존했다. 새 실험 UUID나 학습·평가 API·uploader는 만들지 않았다.

### 보고서 ZIP 다운로드와 압축 해제

HF 계정에 저장소 접근 권한이 있는 상태에서 [보고서 ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/fbec145d2343689b410c389ec32b423342c87288/integration/e7b6f9ac66c848308e28da0716bbce0c/artifacts/reports/950728853c2e32c97e188773136e846c80988ce1e19a329e99ca2be90e08215a/report-950728853c2e32c97e188773136e846c80988ce1e19a329e99ca2be90e08215a.zip?download=true)를 사용한다. ZIP 크기는 **69,485,757 bytes**, SHA-256은 `950728853c2e32c97e188773136e846c80988ce1e19a329e99ca2be90e08215a`다.

다운로드 폴더에서 새 폴더로 풀고 `index.html`을 브라우저로 연다. macOS에서는 다음처럼 실행할 수 있다.

```bash
unzip report-950728853c2e32c97e188773136e846c80988ce1e19a329e99ca2be90e08215a.zip -d task09c_report
open task09c_report/index.html
```

ZIP에는 원래 실험 보고서 HTML·Markdown, 상대 링크의 그림·영상·CSV·JSON, 검증 자료와 게시 범위 README를 포함했다. 본 저장소 문서의 이 게시 후 추가 내용은 원래 보고서와 구분된다. 보고서 selection은 933개 파일이며 공통 bundler의 `bundle_manifest.json`·config·metrics를 더해 ZIP 최상위 inventory는 936개다.

실행 시점의 소스와 게시 구현 소스는 각각 `source_archives/execution-sources.zip`, `source_archives/implementation-a1ade094f1a84af4ce2cee5dbe502a54bec72ebf.zip`에 보관했다. 실행 소스는 원래 UUID별 snapshot·deployment overlay와 완료 package를 보존하며, 뒤늦게 만든 소스를 실행 당시 snapshot으로 표시하지 않는다. 파일별 원본 UUID·SHA·ZIP member와 Model 게시 경로는 `publication/file_inventory.json`, `publication/weight_inventory.json`, `publication/model-inventory-<UUID>.json`에서 확인한다. 가중치는 Model 저장소에 별도 게시했다.

대용량 per-frame RGB-D·지도·그래프 원자료, frozen feature·prediction 배열과 Legacy FPS 확장은 이번 게시에서 제외했다. 보존된 원본 manifest·보고서가 가리키는 원자료 경로를 ZIP에 모두 포함했다는 뜻은 아니다. HTML·Markdown의 실제 상대 링크는 ZIP 안에서 모두 해석된다.

### 게시 검증 결과와 실행 한계

| 검사 | 결과 |
| --- | --- |
| 현재 Task09-C contract / 기존 09-A/B | 별도 process에서 **38/38**, **15/15 PASS** |
| 공통 Logger·artifact 테스트 | **41/42 PASS**. `test_reject_running_run_symlink_and_outside_organization` 1개 실패는 기존 HEAD `58e645b4eba1bc5a06c827bb4082508122de467a`를 새 임시 폴더에 추출해도 재현됐다. 기존 구현의 namespace override 허용과 테스트 기대가 다르며, 공통 코드는 변경하지 않았다. 이번 목적지는 정확한 기본 조직 저장소 2개로 별도 검증했다. |
| 가중치 선택·해시·정규화 readback | GNN 19개·Probe 324개가 원본 manifest 및 `frozen_selection.json`과 일치. Probe NPZ 전체를 `allow_pickle=False`로 읽었다. 제외 가중치·원자료가 실제 selection에 없음. |
| 새 임시 폴더 ZIP 추출 | 상대 링크 **137개**, PNG **27개**, MP4 **75+58 전체 프레임** 검사 PASS |
| 별도 소스 ZIP | 실행 소스 **2,674개**·게시 구현 **369개**의 파일별 SHA 일치. 빈 파일 **33+10개**도 보존·검증 |
| dry-run과 실제 게시 selection | 보고서 `reports metrics`, GNN·Probe `checkpoint metrics`로 세 묶음 검사 후 같은 파일·해시 목록을 잠그고 `--private` 게시 |
| 원격 고정 revision | 선택 **696개**의 경로·크기와 LFS SHA-256 또는 Git blob SHA-1 일치. ZIP 전체·대표 가중치·config·metrics·manifest **14개**를 강제 재다운로드해 SHA-256 대조 PASS |
| 원본 보존 | 게시 전후 원본 파일 **23,916개 / 1,218,969,674 bytes**의 목록·SHA 불변. §19의 전송 당시 집계와 구분되는 게시 시점 전체 감사 |

게시용 receipt·선택 목록·잠금 해시·검증 기록은 로컬 `outputs/task09c_publication/`에 유지한다. 이번 작업은 macOS의 CPU contract·파일/영상 readback과 HF 게시만 실행했다. Simulator·CUDA·PhysX를 새로 실행하지 않았고 재학습·W&B sync·GitHub push도 하지 않았다. Legacy FPS와 진행 중인 결과 뷰어 변경은 두 로컬 커밋 범위에 포함하지 않았다.

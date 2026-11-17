# Task04 — Vision Geometric Feature & Intra-object Edge

Run UUID: `9f90293b227e4dbfaaa6290afad9cc3b`. Feature 계약 **PASS**, Edge 계약 **PASS**. 고정된 8개 frame × 2개 ratio의 CPU offline 검증이며 독립 episode 16개가 아니다.

[정적 HTML](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/vision_feature_edge.html) · [JSON](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/vision_feature_edge.json) · [NPZ](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/vision_feature_edge.npz) · [검증](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/verification.json) · [테스트 로그](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/tests.log) · [Source/input hash](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/source_hashes.json) · [Implementation diff](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/implementation.diff)

## Feature와 수신 계약

Task03-K 저장 point/source/행 순서와 provenance를 유지했다. 기존 optimized PCA5 evidence에서 EE normal 3성분, normal variation(rad), camera-z depth jump(m)를 gather한 F=5다. Availability와 measurement validity는 분리했다. Invalid는 NaN/false이고 point 제거·재선택은 없다. `onehot`/`probability`는 unsupported error다. RGB/선택 근거를 GNN x에 자동 추가하지 않는다.

총 11095개 행(동일 frame의 두 ratio 포함) 중 normal 유효 10483개다. Silhouette-only/Surface-only/중복 후보, 부분 유효 측정과 빈 관측을 구분해 JSON/HTML에 저장했다. Frame18/56의 빈 Box와 frame27/66의 복구도 그대로 전달한다.

8개 source frame의 deterministic extraction은 각각 **1회**, 두 ratio가 공유했다. AB cache의 support/candidate/validity는 exact, float Feature는 사전 고정 tolerance로 비교했다. Float64 reference와 float32 저장 오차는 별도로 검증했다. 저장 허용오차는 성분별 `max(1e-6m,4 ULP)`이며 결과를 본 뒤 확대하지 않았다. 원본 float32 calibration의 미세한 비직교성에 따른 거리 변형은 float64 연산 오차와 분리했다. Near-tie의 EE64/EE32 ordering 차이는 진단하며 실제 topology는 camera64 기준으로 고정했다.

Unversioned v1과 opt-in v2, 전체 packet deep-copy/Feature/provenance 보존을 검증했다. Synthetic robot 3개가 실제 공급된 테스트에서만 +3 remap했다. 실제 robot 좌표 수집이나 GNN 입력 encoder 완성은 검증하지 않았다.

## Edge와 한계

전체 observed support X의 기존 right/down PairEvidence와 depth gate로 component를 만들고, 동일 object/component의 최종 point에 Euclidean k-NN(k=8)을 적용했다. Self-loop/directed duplicate는 0이며 양방향 합집합, index 범위, object/component 소속, E≤16N을 검사했다. 관측상 다른 component를 잇는 Edge는 0이다.

Component/degree/isolated node/길이 mean·p95·max와 가장 긴 Edge의 source/UV/camera·EE XYZ를 보존했다. Hole/얇은 표면 shortcut 의심 사례는 긴 연결 overlay와 3D 그림에서 검토할 진단이며 GT로 Edge를 재선택하지 않았다. 같은 component라는 사실은 물리 표면 연결성이나 physical crease를 보장하지 않는다. Radius/normal veto/geodesic 등 후속 정책은 추가하지 않았다.

![Edge degree/length](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/edge_distributions.png)

Frame0에서는 N=950→1694 (×1.78)일 때 k-NN 평균이 17.984→45.494ms (×2.53)로 증가했다. Extraction 포함 가산 subtotal 평균은 52.060→79.809ms다. 두 frame0 조건 모두 최대 subtotal이 50ms를 넘었다.

Frame0 Box의 가장 긴 연결은 **263.469mm**, camera-z **8.431m**, source pixel (251,206)↔(259,206) (ratio .5)이다. 관측 이미지의 hole 주변 위치를 잇는 의심 사례다. 실제 Box 앞면의 표면 연결이라는 판정은 하지 않는다. 같은 component 안의 관측 depth 오염/긴 연결이 남을 수 있음을 보여준다. 개선 정책은 추가하지 않고 최종 채택 판단을 보류한다. Full-extent 3D는 최대 30.726m finite point까지 포함하며 아래 확대 그림은 표시 범위만 좁힌 진단이다.

Float64 reference의 최대 성분 오차는 2.78e-17m, float32 저장 오차는 1.2e-06m이고, 허용오차 대비 최대 비율은 0.121였다. Normal norm 최대 오차는 1.03e-07다. Raw calibration의 비직교성까지 분리한 거리 검증을 통과했다. 이 검사들은 physical Edge 정확도 PASS가 아니다.

![Hole 주변 긴 연결 확대](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/f0000_r0.5_shortcut_detail.png)
![Ratio1 긴 연결 확대](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/f0000_r1_shortcut_detail.png)

## CPU 시간과 메모리

Extraction은 frame당 1회만 허용된 입력 관문을 우선해 별도 실측했다. **3회 warmup + 5회 계측은 evidence 이후 단계**에 적용했다. Component는 각 frame-repeat에서 두 ratio가 공유한다. Offline subtotal은 측정한 component+ratio 처리에 같은 frame의 단회 extraction 시간을 더한 값이며 연속 실시간 loop latency가 아니다. I/O/그림/압축/로깅/진단은 core 시간에서 제외했다.

| frame/ratio | N | E | k-NN mean ms | evidence 이후 subtotal mean ms | extraction 포함 가산 subtotal mean ms | fresh worker peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f0000_r0.5 | 950 | 8486 | 17.984 | 21.051 | 52.060 | 152.1 |
| f0000_r1 | 1694 | 15138 | 45.494 | 48.799 | 79.809 | 154.0 |
| f0018_r0.5 | 453 | 4112 | 9.591 | 11.913 | 26.251 | 149.5 |
| f0018_r1 | 807 | 7304 | 25.083 | 27.612 | 41.950 | 150.1 |
| f0027_r0.5 | 539 | 4888 | 9.593 | 11.573 | 25.309 | 149.3 |
| f0027_r1 | 883 | 7952 | 22.807 | 24.951 | 38.688 | 150.2 |
| f0045_r0.5 | 514 | 4490 | 8.161 | 10.149 | 23.177 | 149.0 |
| f0045_r1 | 830 | 7318 | 17.776 | 19.912 | 32.940 | 149.8 |
| f0056_r0.5 | 371 | 3220 | 6.495 | 8.215 | 19.034 | 148.6 |
| f0056_r1 | 636 | 5518 | 15.410 | 17.263 | 28.082 | 149.1 |
| f0066_r0.5 | 424 | 3666 | 6.418 | 8.305 | 18.021 | 149.5 |
| f0066_r1 | 630 | 5516 | 12.722 | 14.687 | 24.404 | 148.9 |
| f0080_r0.5 | 477 | 4128 | 5.819 | 7.509 | 18.354 | 148.5 |
| f0080_r1 | 747 | 6704 | 10.749 | 12.609 | 23.454 | 149.1 |
| f0120_r0.5 | 446 | 3774 | 5.585 | 7.364 | 20.115 | 148.8 |
| f0120_r1 | 694 | 6168 | 10.320 | 12.195 | 24.946 | 149.3 |

고정 k에도 정확한 거리 계산은 O(Σn_c²), 행별 전체 정렬은 O(Σn_c² log n_c)다. 큰 N에서 비용이 증가하며 component별 분포에도 영향을 받는다. 최대 N 조건과 다른 조건의 실측 비교는 HTML/JSON에 있다. 50ms 초과 조건은 그대로 표시한다. N/k를 줄이거나 GPU 최적화를 수행하지 않았다. RSS는 fresh CPU worker의 baseline과 native allocation을 포함한 peak로 분리했고 packet/evidence/receiver 배열 byte도 별도 저장했다.

![N/E와 validity](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/validity_variable_ne.png)
![CPU 비용](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/latency.png)
![메모리](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/memory.png)
![좌표 변환 검증](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/camera_ee_verification.png)

## 검증 범위와 남은 판단

PNG decode/JSON·NPZ readback/상대 링크/hash 보존은 verification.json에 기록한다. 실제 브라우저: **NOT_EXECUTED**. PNG 시각 검토: **PASS_FOR_INSPECTED_PNGS**. 실행하지 않은 검사를 PASS로 기록하지 않는다.

수치 계산 완료 후 빈 Box의 빈 3D Edge collection을 그리는 Matplotlib 오류가 발생했다. `failure.json`과 최초 계산 source를 보존하고 **같은 UUID에서 저장 수치만으로 보고서를 복구**했다. 새 sampling/extraction/Edge 계산/benchmark를 실행하지 않았다. `numerical_results_before_report_recovery.json`, `computation_source_snapshot/`, `source_snapshot/`와 verification의 recovery 항목을 구분한다.

현재 repository 계약으로 진행하도록 명시적으로 승인받았다. 별도 첨부 `SHARED_TECHNICAL_FOUNDATION.md`는 읽지 않았으며 새로 만들지 않았다. 공통 ExperimentLogger/local artifact 등록을 재사용했다. 기존 dirty/untracked 파일과 입력 artifact를 보호했고 Sampling/PCA5/threshold/ratio/Surface FPS/GT GraphBuilder를 변경하지 않았다.

Simulator/renderer, 추가 episode, CUDA/MPS/PCL, GNN/GRU/BC 학습·rollout, stale geometry, robot/inter-object 정책, PyG adapter, dependency 설치, 외부 게시와 commit은 수행하지 않았다. Feature/Edge 계약 PASS는 최종 GNN encoder·BC 성공·20Hz 통합·최종 Edge 정책 채택을 뜻하지 않는다. 결과 검토 후 다음 변경을 결정한다.

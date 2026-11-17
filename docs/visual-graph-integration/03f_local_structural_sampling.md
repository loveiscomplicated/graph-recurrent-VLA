# Task03-F — Generalizable Local Geometry Preservation

전체 HTML·NPZ·실행 소스와 원래 run 기록은 [HF 후속 게시 인덱스](../../visual/results/publications/sampling_followups.md)에 있다. Git에 포함한 핵심 그림·작은 수치는 아래 미리보기 링크로 확인할 수 있다.

실행 UUID: `1f7ad262dd2e4b2fb0ac805a4eda5278`. ep000 Frame 0 S3의 저장 Global λ=0과 Local UV spatial stratification control을 비교했다.

제한된 예산에서 후보의 국소 공간 보존과 실제 구조·Surface의 trade-off를 검증한다. 후보 수를 균형 있게 나누는 UV 분할이며 structural component detection, geometry-aware grouping, density-invariant sampling이 아니다. Grouping·quota·group별 초기화가 함께 달라지므로 효과를 quota 하나에 귀속하지 않는다.

## 실행 조건과 사전 고정

- num_points=256, structural_ratio=.25, threshold=20°, seed=0. C/X 및 원본 XYZ·pixel·source 불변. P 각 32, Peg F=129, Box F=127. 실제 Global FPS 호출은 0회다.
- 최대 8 leaf, split 최소 4후보, 최대 capacity leaf부터 UV 유효 span 축/균형 cut, capacity-capped round robin과 기존 Camera XY FPS를 사용했다. Surface는 원본 3D FPS다.
- 8/4는 이번 K=32 control 설정이다. 일반 budget의 성능을 보장하지 않는다. 규칙·평가 구역·7개 fixture·N/A 정의를 config/source snapshot에 고정한 뒤 테스트·선택을 수행했다.
- GT pose·rim·face-pair·ROI는 실제 선택 저장 후에만 평가에 사용했다. Task-Relevance Scoring은 구현하지 않았다.

## GT-free 공간 보존

| 대상 | 방식 | C→P 3D p50 / p95 / max mm | UV p95 px | UV 점유 구역 선택 / 후보 구역 | UV 구역 동등 평균 p95 mm | Surface mean / p95 mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| peg | global | 2.469 / 6.662 / 8.198 | 6.708 | 8/8 | 5.616 | 5.284 / 8.609 |
| peg | local | 2.891 / 8.904 / 15.126 | 7.273 | 8/8 | 7.738 | 5.277 / 8.626 |
| box | global | 8.843 / 29.779 / 46.052 | 8.081 | 9/9 | 22.589 | 11.580 / 19.077 |
| box | local | 7.200 / 35.093 / 96.389 | 6.083 | 9/9 | 28.031 | 11.676 / 19.561 |

peg: C→P p95 6.662→8.904mm, max 8.198→15.126mm; UV 후보 점유 구역 중 P=0은 0→0개다.
box: C→P p95 29.779→35.093mm, max 46.052→96.389mm; UV 후보 점유 구역 중 P=0은 0→0개다.

전체 p95는 후보 밀도에 가중된다. 독립 UV 4×4 및 XYZ 최대 2×2×2 구역의 count·밀도·선택률·p95/max·동등 구역 통계와 raw/normalized 거리·ECDF는 JSON/NPZ에 저장했다. C에는 false response가 섞이므로 지표 변화를 구조 검출 정확도 변화로 치환하지 않는다. 최댓값은 관측 C 위의 미피복 위치이며 연속 빈 공간의 크기를 복원한 값이 아니다.

## Peg seam / Box rim 보조 평가

같은 point/reference pair의 2px AND 3mm matcher를 재사용했다. Peg observed 462(GT visible 464), Box 103. nodes는 실제 matching node 수, refs는 unique reference 수다. 구간 복수 귀속은 ambiguous이며 Peg 물리적 간격은 N/A다.

| 대상 | 방식 | C nodes / refs | P nodes / refs | S nodes / refs | F nodes / refs | Surface 추가 refs | Negative union P / S / F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| peg | global | 692 / 462 | 30 / 160 | 0 / 0 | 30 / 160 | 0 | 2 / 90 / 92 |
| peg | local | 692 / 462 | 31 / 140 | 0 / 0 | 31 / 140 | 0 | 1 / 90 / 91 |
| box | global | 127 / 91 | 3 / 9 | 1 / 3 | 4 / 12 | 3 | 14 / 61 / 75 |
| box | local | 127 / 91 | 1 / 3 | 0 / 0 | 1 / 3 | 0 | 15 / 63 / 78 |

| 대상 | 후보 C / support X | Leaf capacity (ID 순서) | Quota | Global / Local exact feature rows |
| --- | ---: | --- | --- | ---: |
| peg | 703 / 11780 | [87, 89, 90, 90, 87, 87, 75, 98] | [4, 4, 4, 4, 4, 4, 4, 4] | 26 / 14 |
| box | 599 / 11665 | [76, 74, 75, 71, 77, 78, 73, 75] | [4, 4, 4, 4, 4, 4, 4, 4] | 45 / 40 |

| 대상 | 구간 | observed | C refs | Global P / F refs | Local P / F refs | Global / Local F nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| peg | ambiguous | 2 | 2 | 1 / 1 | 2 / 2 | 1 / 3 |
| peg | face_0_2 | 91 | 91 | 44 / 44 | 24 / 24 | 7 / 5 |
| peg | face_0_5 | 106 | 106 | 49 / 49 | 48 / 48 | 7 / 8 |
| peg | face_2_5 | 263 | 263 | 66 / 66 | 66 / 66 | 15 / 15 |
| box | rim_axis1_+1 | 25 | 24 | 6 / 6 | 0 / 0 | 2 / 0 |
| box | rim_axis1_-1 | 26 | 26 | 0 / 0 | 0 / 0 | 0 / 0 |
| box | rim_axis2_+1 | 26 | 15 | 0 / 3 | 3 / 3 | 1 / 1 |
| box | rim_axis2_-1 | 26 | 26 | 3 / 3 | 0 / 0 | 1 / 0 |

`rim_axis1_-1`의 기존 0/26은 평가 전용 실패 사례다. X/C의 지원과 P/S/F의 보존 reference·matching node·source 목록을 별도 저장했다. 이 구간 개선을 필수 성공 조건으로 삼거나 이를 보고 재선택하지 않았다.

## Synthetic sampler stress test

3종 총 7개 고정 fixture, K=32, 같은 C·anchor·Camera XY λ=0·기존 FPS다. Synthetic에서는 P만 비교하며 detector/normal/tracker/Surface를 만들지 않았다. 구조/noise/reference annotation은 평가 전용이다. 투영 충돌은 생성 ID가 작은 한 점만 남기고 flat 순으로 정렬하며 refill하지 않았다. 실제 camera XYZ를 보존했다.

| Fixture | 생성 / 충돌 전 / 관측 C | Global / Local C→P p95 mm | Global / Local 정규화 p95 | Global / Local reference→P p95 mm | Global / Local noise P |
| --- | ---: | ---: | ---: | ---: | ---: |
| line_uniform | 128 / 128 / 128 | 3.750 / 5.625 | 0.016 / 0.024 | 3.750 / 5.625 | 0 / 0 |
| line_dense | 256 / 256 / 216 | 3.750 / 7.500 | 0.016 / 0.031 | 3.750 / 7.500 | 0 / 0 |
| separated_clean | 192 / 192 / 188 | 3.750 / 5.625 | 0.018 / 0.028 | 3.984 / 6.094 | 0 / 0 |
| separated_noise | 200 / 200 / 196 | 3.750 / 5.625 | 0.016 / 0.023 | 4.219 / 6.094 | 1 / 1 |
| camera_near | 192 / 192 / 188 | 3.750 / 5.625 | 0.018 / 0.028 | 3.984 / 6.094 | 0 / 0 |
| camera_far | 192 / 192 / 154 | 3.750 / 6.281 | 0.018 / 0.031 | 3.832 / 6.094 | 0 / 0 |
| camera_occluded | 192 / 64 / 60 | 0.469 / 0.469 | 0.016 / 0.016 | 0.586 / 0.586 | 0 / 0 |

| Fixture | 구조 | 길이 m | Global / Local 선택 수 | Global / Local reference p95 mm |
| --- | --- | ---: | ---: | ---: |
| line_uniform | 0 | 0.240 | 32 / 32 | 3.750 / 5.625 |
| line_dense | 0 | 0.240 | 32 / 32 | 3.750 / 7.500 |
| separated_clean | 0 | 0.030 | 5 / 8 | 3.398 / 2.461 |
| separated_clean | 1 | 0.060 | 8 / 12 | 4.453 / 4.453 |
| separated_clean | 2 | 0.120 | 19 / 12 | 3.281 / 7.969 |
| separated_noise | 0 | 0.030 | 5 / 8 | 3.398 / 2.461 |
| separated_noise | 1 | 0.060 | 11 / 11 | 2.578 / 5.391 |
| separated_noise | 2 | 0.120 | 15 / 12 | 5.156 / 7.641 |
| camera_near | 0 | 0.030 | 5 / 8 | 3.398 / 2.461 |
| camera_near | 1 | 0.060 | 8 / 12 | 4.453 / 4.453 |
| camera_near | 2 | 0.120 | 19 / 12 | 3.281 / 7.969 |
| camera_far | 0 | 0.030 | 5 / 8 | 3.398 / 2.227 |
| camera_far | 1 | 0.060 | 11 / 12 | 2.578 / 3.516 |
| camera_far | 2 | 0.120 | 16 / 12 | 4.219 / 7.969 |
| camera_occluded | 0 | 0.030 | 32 / 32 | 0.586 / 0.586 |
| camera_occluded | 1 | 0.060 | 0 / 0 | N/A / N/A |
| camera_occluded | 2 | 0.120 | 0 / 0 | N/A / N/A |

| Fixture | 외곽 띠 후보 수 | Global / Local 외곽 띠 P | 가림 경계 후보 / Global P / Local P |
| --- | ---: | ---: | ---: |
| line_uniform | 26 | 7 / 6 | 0 / 0 / 0 |
| line_dense | 61 | 8 / 9 | 0 / 0 / 0 |
| separated_clean | 124 | 24 / 20 | 0 / 0 / 0 |
| separated_noise | 132 | 21 / 21 | 0 / 0 / 0 |
| camera_near | 124 | 24 / 20 | 0 / 0 / 0 |
| camera_far | 94 | 21 / 20 | 0 / 0 / 0 |
| camera_occluded | 12 | 8 / 6 | 0 / 0 / 0 |

가림 fixture에서는 짧은 선분 하나만 관측에 남고, 지정한 가림 경계 1px 띠에는 후보가 없다. 이 사례의 동률과 경계 점유 0은 복잡한 부분 가림 또는 경계 noise에 대한 강건성 증거가 아니다. Fixture를 결과에 맞게 바꾸지 않았다.


## 최종 질문별 판단

1. 공간 누락: peg: C→P p95 6.662→8.904mm, max 8.198→15.126mm; UV 후보 점유 구역 중 P=0은 0→0개다. box: C→P p95 29.779→35.093mm, max 46.052→96.389mm; UV 후보 점유 구역 중 P=0은 0→0개다. Box pixel p95는 8.081→6.083px로 좋아졌지만 3D tail과 실제 rim 보존은 악화됐다. 독립 UV 구역의 점유 개수도 같아 점유율만으로 차이를 설명할 수 없다. Group quota 충족은 성능 근거가 아니다.
2. 밀도·길이 편향: 균일→조밀 직선의 reference p95는 Global 3.750→3.750mm, Local 5.625→7.500mm다. 길이 비 1:2:4 clean 구조의 선택 수는 Global [5, 8, 19], Local [8, 12, 12]다. 후보 수 기반 분할과 균등 quota는 실제 길이나 면적을 균등 보존하는 규칙이 아니다.
3. Noise: 8개 noise 후보의 선택 수는 Global 1/32, Local 1/32다. 실제 세 구조 선택 수는 clean→noise에서 Global [5, 8, 19]→[5, 11, 15], Local [8, 12, 12]→[8, 11, 12]다. 이 fixture에서는 noise에 더 많은 예산을 쓰는 현상은 관찰되지 않았다. Noise 추가에 따른 구조별 배분 변화는 남으며, 이를 일반적인 noise 강건성으로 확대하지 않는다.
4. 거리·가림 민감도: camera_near: C=188, reference p95 G/L=3.984/6.094mm, C→P 정규화 p95 G/L=0.0185/0.0277; camera_far: C=154, reference p95 G/L=3.832/6.094mm, C→P 정규화 p95 G/L=0.0185/0.0309; camera_occluded: C=60, reference p95 G/L=0.586/0.586mm, C→P 정규화 p95 G/L=0.0159/0.0159. 가림은 남은 visible reference만 평가하며, 관측 불가능 geometry 복원을 요구하지 않는다. 정수 투영 충돌·후보 순서·seeded anchor도 조건 간 달라질 수 있어 scale 하나의 인과 효과로 분리하지 않는다.
5. 실제 구조와 Surface: peg Final refs 160→140, Surface mean/p95 5.284/8.609→5.277/8.626mm, negative P 2→1/32; box Final refs 12→3, Surface mean/p95 11.580/19.077→11.676/19.561mm, negative P 14→15/32. 구간별 증가와 감소는 보조 평가 표에 모두 제시했다. Box 한 구간의 개선으로 다른 구조 손실을 상쇄했다고 판단하지 않는다.
6. 추가 검증 판단: 실제 Peg/Box의 3D tail과 GT 보존, 긴 구조·밀도 변화의 reference 보존에서 전반적 열세다. 현재 설정은 공간 배분의 단순 control로 남길 수 있으나 일반 geometry preservation 정책 후보로 적극 채택할 근거는 얻지 못했다. 더 구조적인 grouping이 필요한지 검토할 이유는 생겼지만, 이 실험은 그 대안의 우월성을 검증하지 않았다. 동일 설정의 독립 형상·관측·episode 검증 전에는 채택하지 않는다. GNN/BC 성능 개선은 주장하지 않는다.

## 원거리 depth와 Surface·ROI trade-off

원거리 positive depth를 필터링하지 않았다. 평가 bbox는 전체 observed X에서 정하므로 depth outlier가 3D bbox 대각선을 키우고 XYZ 격자를 넓혀 후보를 일부 구역에 몰아넣을 수 있다. 따라서 정규화 거리·구역 점유만으로 개선을 판단하지 않으며 raw 거리와 UV 결과를 함께 본다. 아래 X depth와 bbox는 두 방식에 동일하다.

| 대상 | X depth median / p95 / max m | XYZ bbox 대각선 mm | P depth median / p95 Global → Local | P z>X p95 Global / Local |
| --- | ---: | ---: | --- | ---: |
| peg | 0.2060 / 0.3271 / 0.3650 | 263.857 | 0.1865/0.3325 → 0.1810/0.3081 | 3 / 1 |
| box | 0.4960 / 0.5350 / 30.7260 | 32889.655 | 0.4955/1.3864 → 0.4885/0.9307 | 12 / 5 |

| 대상 | 방식 | 3D / UV bbox 정규화 p95 | XYZ 후보 점유 구역 / P=0 구역 | 최대 3D 미피복 후보 source / UV / XYZ m |
| --- | --- | ---: | --- | --- |
| peg | global | 0.025249 / 0.029807 | 4 / [6] | 178953 / [265, 349] / [0.008720703423023224, 0.08583007752895355, 0.23499999940395355] |
| peg | local | 0.033747 / 0.032317 | 4 / [6] | 156962 / [290, 306] / [0.041642576456069946, 0.060955073684453964, 0.3089999854564667] |
| box | global | 0.000905 / 0.050114 | 1 / [] | 114427 / [251, 223] / [-0.010265625081956387, -0.07414062321186066, 0.5839999914169312] |
| box | local | 0.001067 / 0.037724 | 1 / [] | 106747 / [251, 208] / [-0.025066405534744263, -0.2645898461341858, 1.4259999990463257] |

Box의 X depth max 30.726m를 유지한 XYZ 격자에서는 C가 단 한 구역에 들어갔다. 이 coarse 격자는 이 관측에서 국소 분산을 구별하지 못한다. 작은 정규화 거리 역시 큰 bbox 분모의 영향이 있으므로 raw 거리와 UV 격자를 병기했다.


Known-negative ROI는 겹칠 수 있어 union으로 중복을 제거했다. 일반 Surface나 관측 경계 자체를 자동으로 오선택이라 판정하지 않는다.

## Feature·검증·재현 범위

- 같은 S3 Task03 baseline metadata가 있는 source에만 23 field를 exact gather했다. Availability mask·compact positions를 저장하고 나머지 feature는 생성하지 않았다. Geometry-level offline 검증이며 완전한 Vision Tensor 출력 검증은 아니다.
- 필수 계약 테스트: Ran 66 tests in 0.722s; OK. PASS/FAIL/SKIP와 원본 hash·NPZ readback·상대 링크·PNG 검증은 verification.json에 기록했다. 성능 개선은 테스트 PASS 조건이 아니다.
- 기존 ExperimentLogger(team=visual, job_type=evaluate), UUID/project/group 및 artifact inventory를 재사용했다. WANDB_MODE=disabled이며 업로드는 하지 않았다. Git 시작 상태가 dirty이므로 source snapshot·변경 전 diff·implementation diff를 함께 보관했다.
- CPU NumPy/SciPy offline 실행만 수행했다. Simulator·renderer·CUDA/GPU/backend, 새 episode, 학습/GNN/BC, production sampler/normal/feature/tracking 변경, dependency 설치·외부 업로드·Git commit은 미실행이다.
- 실험 완료와 알고리즘 채택은 별개다. 한 frame과 7개 synthetic fixture로 일반화를 입증하지 않는다. 일반 sampling 방식 채택에는 독립 형상·관측·episode·센서에서 동일 설정 검증이 필요하다. Task-Relevance Scoring은 향후 독립 비교로 남긴다.

## 산출물

- [box_gt_auxiliary.png](../../visual/results/diagnostic_previews/03f/box_gt_auxiliary.png)
- [browser.log](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/browser.log)
- [browser_dom.html](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/browser_dom.html)
- [browser_verification.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/browser_verification.json)
- [candidate_distance_ecdf.png](../../visual/results/diagnostic_previews/03f/candidate_distance_ecdf.png)
- [candidate_groups.png](../../visual/results/diagnostic_previews/03f/candidate_groups.png)
- [fixture_manifest.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/fixture_manifest.json)
- [git_end.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/git_end.json)
- [git_start.diff](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/git_start.diff)
- [git_start.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/git_start.json)
- [html_preview.png](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/html_preview.png)
- [implementation.patch](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/implementation.patch)
- [independent_regions.png](../../visual/results/diagnostic_previews/03f/independent_regions.png)
- [local_structural_sampling.html](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/local_structural_sampling.html)
- [local_structural_sampling.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/local_structural_sampling.json)
- [local_structural_sampling.npz](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/local_structural_sampling.npz)
- [peg_gt_auxiliary.png](../../visual/results/diagnostic_previews/03f/peg_gt_auxiliary.png)
- [preregistered_config.json](../../visual/results/diagnostic_previews/03f/preregistered_config.json)
- [previous_failed_report.md](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/previous_failed_report.md)
- [report.md](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/report.md)
- [selection_before_gt.npz](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/selection_before_gt.npz)
- [source_hashes.json](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/source_hashes.json)
- [surface_negative_tradeoff.png](../../visual/results/diagnostic_previews/03f/surface_negative_tradeoff.png)
- [synthetic_seven.png](../../visual/results/diagnostic_previews/03f/synthetic_seven.png)
- [tests.log](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/tests.log)
- [unfiltered_depth_xz.png](../../visual/results/diagnostic_previews/03f/unfiltered_depth_xz.png)
- [verification.json](../../visual/results/diagnostic_previews/03f/verification.json)

## 문서 생성 오류와 재실행 구분

최초 수치 실행 `c4709984427d4283815a3971a415b4af`는 계약 테스트·선택·평가·PNG 생성 후 HTML의 원본 PNG 파일명 링크 검사에서 실패했다. 실패 기록과 당시 source/config/선택/수치는 보존했다. 현재 UUID는 그 저장 결과를 읽은 **문서 후처리 실행**이며 실제·synthetic FPS 및 성능 평가를 다시 실행하지 않았다. Group·quota·fixture·threshold·seed는 바꾸지 않았다.

- [실패 기록](../../experiments/c4709984427d4283815a3971a415b4af/attempt_failure.json)
- [최초 수치 JSON](../../experiments/c4709984427d4283815a3971a415b4af/local_structural_sampling.json)
- [최초 선택/source 보존 자료](../../experiments/c4709984427d4283815a3971a415b4af/source_hashes.json)

## 기존 근거

- [승인된 계획](03f_local_structural_sampling_plan.md)
- [Task03-E 보고서](03e_priority_fps_metric_control.md)
- [Task03-E 원본 JSON](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.json), [원본 NPZ](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.npz)
- [Task03 원본 comparison](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [원본 samples](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz)
- [runner](../../visual/runtime/test/run_local_structural_sampling.py), [계약 테스트](../../visual/runtime/test/test_local_structural_sampling.py)

재실행은 새 보호 baseline.json과 새 UUID에서만 수행한다. 기존 보고서가 있으면 overwrite를 거부한다. 결과 제출 후 다음 알고리즘 선택 없이 중단한다.

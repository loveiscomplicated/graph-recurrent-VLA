# Task03-D Box hole rim 병목 분석 Phase 1

전체 HTML·NPZ·실행 소스와 원래 run 기록은 [HF 후속 게시 인덱스](../../visual/results/publications/sampling_followups.md)에 있다. Git에 포함한 핵심 그림·작은 수치는 아래 미리보기 링크로 확인할 수 있다.

**Frame 0의 S3는 후보 단계에서 S2보다 더 많은 rim reference를 덮지만, priority 선택 후에는 더 적게 남는다.**
S2/S3 후보 coverage는 **57/103 대 91/103**, priority coverage는 **13/103 대 3/103**이다.
Surface는 S2에 후보 밖 reference 3개를 보완했고 S3에는 보완하지 않아, 최종 coverage는 **16/103 대 3/103**이다.
관측 가능한 103개가 모두 입력에 있으므로 이 프레임의 최종 차이는 관측 손실로 설명되지 않는다.

Frame 120에서는 GT 41개 중 관측 가능한 reference가 8개다. S2/S3 모두 후보와 priority의 rim coverage가 **0/8**이며,
S3의 최종 2개는 Surface 경로에서 보완됐다. 후보 수, 후보의 rim coverage, 실제 선택 후 retention은 서로 다른 지표다.
이 진단은 손실이 나타나는 단계를 구분한다. 오염이나 FPS 거리 경쟁의 인과 기여율은 확정하지 않는다.

- 진단 run: `d3f14ee494894daabb8ba2fab8e1b00c`. [결과 JSON](../../visual/results/diagnostic_previews/03d_phase1/diagnostic.json), [검증 기록](../../visual/results/diagnostic_previews/03d_phase1/verification.json), [테스트 로그](../../experiments/d3f14ee494894daabb8ba2fab8e1b00c/tests.log).
- 원본 Task03 run: `2551448b0fae4c6bbce3caf48774275f`. [기존 보고서](03_sampling.md), [원본 comparison.json](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json).
- [실행 source snapshot](../../experiments/d3f14ee494894daabb8ba2fab8e1b00c/diagnostic_source/visual/runtime/test/run_rim_bottleneck.py), [구현 diff](../../experiments/d3f14ee494894daabb8ba2fab8e1b00c/implementation.patch), [초기 분석 원문](../../experiments/d3f14ee494894daabb8ba2fab8e1b00c/analysis.txt).

Artifact 링크는 현재 호스트의 로컬 파일을 가리킨다. `experiments/`와 원본 대용량 artifact는 Git 추적 대상이 아니므로 별도 checkout에는 없을 수 있다.

## 1. 범위와 평가 정의

원본 ep000 RGB-D, tracker masks, frame 0/120의 저장 NPZ와 frame metrics JSON, `comparison.json`만 사용했다.
조건은 `S0_n256_r0_t20_s0`, `S2_n256_r0.25_t20_s0`, `S3_n256_r0.25_t20_s0`다.
Box rim을 분석하고 Peg는 object allocation count 확인에만 사용했다.

| 집합 | 정의 |
| --- | --- |
| V | 가시 GT line reference |
| X | Tracker의 전체 유효 Box 관측점 |
| O = R(X) | 관측 가능한 reference |
| C | 저장된 전체 orientation 후보 |
| P | `priority_selected=True`인 실제 선택점 |
| S | 나머지 Surface FPS 선택점 |
| F = P ∪ S | 저장된 전체 선택점 |

`R(A)`는 **동일 물체의 동일 point/reference pair가 2D ≤2px AND 3D ≤3mm를 동시에 만족**할 때 덮이는 unique reference ID 집합이다.
Reference ID는 저장된 `evaluation/box/reference_uv`와 `reference_xyz`의 행 번호다. 한 node가 여러 reference를 덮을 수 있고, 여러 node가 같은 reference를 덮어도 한 번만 센다.

관측률은 `|O|/|V|`, 후보·priority·Surface·최종 retention은 각각 `|R(C)|/|O|`, `|R(P)|/|O|`, `|R(S)|/|O|`, `|R(F)|/|O|`다.
GT 기준 최종 coverage `|R(F)|/|V|`는 별도로 표시한다. 분모가 0이면 N/A이며 JSON에 raw 분자·분모와 `null`을 보존한다.
S0는 General FPS 대조이므로 candidate/priority coverage와 해당 집합 차이는 N/A다.

후보는 NPZ의 `evidence/fd2/1/20/orientation_flat` 및 `evidence/pca5/1/20/orientation_flat`을 원래 raster에 매핑했다.
새 normal estimation, feature extraction, candidate extraction, sampling 또는 FPS를 실제 진단에서 실행하지 않았다. GT는 진단에만 사용했다.

## 2. Frame 0 단계별 coverage

관측률은 **103/103 = 100%**다. 아래 C/P/S/F의 reference 비율은 모두 관측 가능한 103개가 분모다.

| 조건 | 단계 | 실제 node | reference와 매칭된 node | unique matched reference | 관측 대비 coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| S0 | X | 11,665 | 153 | 103 | 103/103 (100.00%) |
| S0 | C | N/A | N/A | N/A | N/A |
| S0 | P | N/A | N/A | N/A | N/A |
| S0 | S | 127 | 2 | 6 | 6/103 (5.83%) |
| S0 | F | 127 | 2 | 6 | 6/103 (5.83%) |
| S2 | X | 11,665 | 153 | 103 | 103/103 (100.00%) |
| S2 | C | 297 | 101 | 57 | 57/103 (55.34%) |
| S2 | P | 32 | 4 | 13 | 13/103 (12.62%) |
| S2 | S | 95 | 1 | 3 | 3/103 (2.91%) |
| S2 | F | 127 | 5 | 16 | 16/103 (15.53%) |
| S3 | X | 11,665 | 153 | 103 | 103/103 (100.00%) |
| S3 | C | 599 | 127 | 91 | 91/103 (88.35%) |
| S3 | P | 32 | 1 | 3 | 3/103 (2.91%) |
| S3 | S | 95 | 0 | 0 | 0/103 (0.00%) |
| S3 | F | 127 | 1 | 3 | 3/103 (2.91%) |

S3는 후보 node 599개 중 127개가 rim reference와 매칭되며 91개 reference를 덮는다. S2는 후보 node 297개 중 101개가 매칭되며 57개를 덮는다.
그러나 같은 32개 priority node를 선택한 뒤 S3는 1 node가 3 reference를, S2는 4 node가 13 reference를 덮는다.
따라서 S3의 더 낮은 최종 coverage를 전체 후보 수나 후보 reference coverage 부족만으로 설명할 수 없다.
S2 Surface의 3개는 `R(C)` 밖의 보완이며, 두 조건 모두 priority에서 남지 않은 후보 reference를 Surface가 회복한 수는 0개다.

## 3. Reference 집합 차이와 S2 S3 직접 비교

각 셀은 reference ID 집합의 원소 수다. **서로 겹치는 행을 원인별 손실률처럼 더하지 않는다.**

| 집합 연산 | Frame 0 S2 | Frame 0 S3 | Frame 120 S2 | Frame 120 S3 |
| --- | ---: | ---: | ---: | ---: |
| V ∖ O | 0 | 0 | 33 | 33 |
| O ∖ R(C) | 46 | 12 | 8 | 8 |
| R(C) ∖ R(P) | 44 | 88 | 0 | 0 |
| R(C) ∖ R(F) | 44 | 88 | 0 | 0 |
| R(P) ∩ R(S) | 0 | 0 | 0 | 0 |
| R(S) ∖ R(P) | 3 | 0 | 0 | 2 |
| R(S) ∖ R(C) | 3 | 0 | 0 | 2 |
| (R(C) ∖ R(P)) ∩ R(S) | 0 | 0 | 0 | 0 |
| O ∖ R(F) | 87 | 100 | 8 | 6 |

S0의 `V ∖ O`는 frame 0/120에서 0/33개, `O ∖ R(F)`는 97/5개다. 후보·priority 관련 연산은 N/A다.

| Frame | 비교 집합 | S2 ∩ S3 | S2 ∖ S3 | S3 ∖ S2 |
| --- | --- | ---: | ---: | ---: |
| 0 | R(C) | 56 | 1 | 35 |
| 0 | R(P) | 0 | 13 | 3 |
| 0 | R(F) | 0 | 16 | 3 |
| 120 | R(C) | 0 | 0 | 0 |
| 120 | R(P) | 0 | 0 | 0 |
| 120 | R(F) | 0 | 0 | 2 |

Frame 0의 최종 S2/S3 교집합은 0개다. 16 대 3의 차이를 동일한 13개 reference의 증감으로 해석하면 안 된다.
후보 교집합은 56개이며 S3에만 있는 후보 reference도 35개지만, 실제 priority와 최종 reference는 양쪽이 겹치지 않는다.

## 4. S0와 Frame 120 대조

| Frame | Box GT / 관측 reference | S0 최종 | S2 최종 | S3 최종 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 103 / 103 | 6 | 16 | 3 |
| 120 | 41 / 8 | 3 | 0 | 2 |

Frame 120 관측률은 **8/41 = 19.51%**이며, `V ∖ O = 33`개는 sampling 전 관측 단계에서 이미 매칭되지 않는다.
S2/S3의 후보 node는 각각 24/65개이나 rim coverage는 모두 0/8이다. S2는 24개 후보 전부를 priority로 선택해도 rim reference를 덮지 않는다.
S3는 Surface node 148개 중 1개가 후보 밖 reference 2개를 덮는다.

| 조건 | 단계 | 실제 node | reference와 매칭된 node | unique matched reference | 비율 |
| --- | --- | ---: | ---: | ---: | ---: |
| S0 | X | 5,618 | 11 | 8 | 8/41 (19.51%) |
| S0 | C | N/A | N/A | N/A | N/A |
| S0 | P | N/A | N/A | N/A | N/A |
| S0 | S | 197 | 1 | 3 | 3/8 (37.50%) |
| S0 | F | 197 | 1 | 3 | 3/8 (37.50%) |
| S2 | X | 5,618 | 11 | 8 | 8/41 (19.51%) |
| S2 | C | 24 | 0 | 0 | 0/8 (0.00%) |
| S2 | P | 24 | 0 | 0 | 0/8 (0.00%) |
| S2 | S | 173 | 0 | 0 | 0/8 (0.00%) |
| S2 | F | 197 | 0 | 0 | 0/8 (0.00%) |
| S3 | X | 5,618 | 11 | 8 | 8/41 (19.51%) |
| S3 | C | 65 | 0 | 0 | 0/8 (0.00%) |
| S3 | P | 49 | 0 | 0 | 0/8 (0.00%) |
| S3 | S | 148 | 1 | 2 | 2/8 (25.00%) |
| S3 | F | 197 | 1 | 2 | 2/8 (25.00%) |

Frame 120의 최종 **retention**은 S0/S2/S3 **37.50% / 0% / 25.00%**다.
GT 41개를 분모로 한 최종 coverage는 **7.32% / 0% / 4.88%**이며 retention과 분모가 다르다.

## 5. Object allocation

| Frame | 조건 | 물체 | observed | budget | candidates | priority | surface_fps | selected |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | S0 | box | 11,665 | 127 | 599 | 0 | 127 | 127 |
| 0 | S0 | peg | 11,780 | 129 | 703 | 0 | 129 | 129 |
| 0 | S2 | box | 11,665 | 127 | 297 | 32 | 95 | 127 |
| 0 | S2 | peg | 11,780 | 129 | 870 | 32 | 97 | 129 |
| 0 | S3 | box | 11,665 | 127 | 599 | 32 | 95 | 127 |
| 0 | S3 | peg | 11,780 | 129 | 703 | 32 | 97 | 129 |
| 120 | S0 | box | 5,618 | 197 | 65 | 0 | 197 | 197 |
| 120 | S0 | peg | 1,353 | 59 | 247 | 0 | 59 | 59 |
| 120 | S2 | box | 5,618 | 197 | 24 | 24 | 173 | 197 |
| 120 | S2 | peg | 1,353 | 59 | 206 | 40 | 19 | 59 |
| 120 | S3 | box | 5,618 | 197 | 65 | 49 | 148 | 197 |
| 120 | S3 | peg | 1,353 | 59 | 247 | 15 | 44 | 59 |

S0의 candidates 열은 원본 metrics에 저장된 PCA5 후보 진단 count다. General FPS가 이를 priority로 사용했다는 뜻은 아니다.
모든 열은 원본 frame metrics와 저장 선택점·후보·관측 support를 대조해 일치했다.

Frame 0은 Peg/Box budget이 129/127이고 S2/S3 priority가 모두 32/32다. 따라서 이 프레임의 Box coverage 차이를 물체별 최종 node budget 차이로 설명할 수 없다.
Frame 120은 Peg/Box budget이 59/197이다. 기존 구현은 object budget 가중치와 후보 capacity로 전역 priority 64개를 배분한다.
S2는 Box 후보가 24개여서 Peg/Box priority가 40/24, S3는 15/49다. 저장 count와 기존 source를 확인한 설명이며 allocation을 재실행하거나 변경하지 않았다.

## 6. Negative ROI 점유

기존 Task03 ROI를 그대로 사용했다. 아래 각 C/P/S/F 셀은 해당 경로의 node count다. 개별 ROI는 겹치며 `known_negative_union`은 논리 OR로 중복을 제거했다.

### Frame 0

| ROI | X | S0 S 및 F | S2 C / P / S / F | S3 C / P / S / F |
| --- | ---: | ---: | --- | --- |
| `silhouette_2px` | 1,164 | 19 | 0 / 0 / 15 / 15 | 2 / 1 / 17 / 18 |
| `occlusion_2px` | 510 | 5 | 0 / 0 / 6 / 6 | 94 / 5 / 3 / 8 |
| `same_plane_tracker_cut_2px` | 96 | 5 | 0 / 0 / 4 / 4 | 5 / 1 / 2 / 3 |
| `invalid_depth_2px` | 296 | 6 | 0 / 0 / 2 / 2 | 0 / 0 / 5 / 5 |
| `robot_occlusion_2px` | 510 | 5 | 0 / 0 / 6 / 6 | 94 / 5 / 3 / 8 |
| `contamination` | 482 | 21 | 59 / 3 / 15 / 18 | 186 / 4 / 16 / 20 |
| `plane_interior` | 8,387 | 38 | 0 / 0 / 25 / 25 | 0 / 0 / 35 / 35 |
| `known_negative_union` | 10,506 | 82 | 59 / 3 / 61 / 64 | 281 / 9 / 70 / 79 |

| 조건 | union / C | union / P | union / S | union / F |
| --- | ---: | ---: | ---: | ---: |
| S0 | N/A | N/A | 82/127 (64.57%) | 82/127 (64.57%) |
| S2 | 59/297 (19.87%) | 3/32 (9.38%) | 61/95 (64.21%) | 64/127 (50.39%) |
| S3 | 281/599 (46.91%) | 9/32 (28.12%) | 70/95 (73.68%) | 79/127 (62.20%) |

### Frame 120

| ROI | X | S0 S 및 F | S2 C / P / S / F | S3 C / P / S / F |
| --- | ---: | ---: | --- | --- |
| `silhouette_2px` | 689 | 28 | 24 / 24 / 25 / 49 | 33 / 29 / 20 / 49 |
| `occlusion_2px` | 445 | 21 | 0 / 0 / 23 / 23 | 8 / 4 / 21 / 25 |
| `same_plane_tracker_cut_2px` | 177 | 12 | 0 / 0 / 11 / 11 | 0 / 0 / 10 / 10 |
| `invalid_depth_2px` | 0 | 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| `robot_occlusion_2px` | 315 | 14 | 0 / 0 / 17 / 17 | 8 / 4 / 14 / 18 |
| `contamination` | 105 | 32 | 0 / 0 / 29 / 29 | 21 / 14 / 23 / 37 |
| `plane_interior` | 4,325 | 113 | 0 / 0 / 94 / 94 | 0 / 0 / 83 / 83 |
| `known_negative_union` | 5,556 | 193 | 24 / 24 / 170 / 194 | 62 / 47 / 146 / 193 |

| 조건 | union / C | union / P | union / S | union / F |
| --- | ---: | ---: | ---: | ---: |
| S0 | N/A | N/A | 193/197 (97.97%) | 193/197 (97.97%) |
| S2 | 24/24 (100.00%) | 24/24 (100.00%) | 170/173 (98.27%) | 194/197 (98.48%) |
| S3 | 62/65 (95.38%) | 47/49 (95.92%) | 146/148 (98.65%) | 193/197 (97.97%) |

Frame 0 contamination priority는 S2 3개, S3 4개이며 union priority는 3/32 대 9/32다.
Frame 120의 S2 후보·priority 24개는 모두 silhouette ROI에 있다. S3는 union priority가 47/49다.
각 ROI의 경로별 비율과 관측 ROI 대비 비율, raw 분자·분모는 결과 JSON에 저장했다.
**일반 Surface node가 평면이나 negative ROI에 있다는 사실을 false selection 또는 예산 낭비로 단정하지 않는다.**

## 7. Positive depth 분포와 경로별 점유

모든 값은 CV-camera z의 metre 단위다. 임의 원거리 cutoff 없이 min/median/p95/max를 기록했다.
마지막 열은 X의 p95를 넘는 점유 count와 해당 경로 대비 비율이다. 이는 관측 분포의 분위수 구간이며 filtering 또는 오류 label이 아니다.
JSON에는 X median 이하, median 초과 p95 이하, p95 초과의 세 구간이 모두 있다.

| Frame | 조건 | 경로 | node | min m | median m | p95 m | max m | z > X p95 / 경로 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | S0 | X | 11,665 | 0.295000 | 0.496000 | 0.535000 | 30.726000 | 562/11665 (4.82%) |
| 0 | S0 | S | 127 | 0.296000 | 0.524000 | 8.431000 | 30.726000 | 57/127 (44.88%) |
| 0 | S0 | F | 127 | 0.296000 | 0.524000 | 8.431000 | 30.726000 | 57/127 (44.88%) |
| 0 | S2 | C | 297 | 0.303000 | 0.500000 | 0.632000 | 0.690000 | 47/297 (15.82%) |
| 0 | S2 | P | 32 | 0.303000 | 0.506000 | 0.665300 | 0.690000 | 12/32 (37.50%) |
| 0 | S2 | S | 95 | 0.296000 | 0.533000 | 8.431000 | 30.726000 | 47/95 (49.47%) |
| 0 | S2 | F | 127 | 0.296000 | 0.529000 | 8.431000 | 30.726000 | 59/127 (46.46%) |
| 0 | S3 | C | 599 | 0.295000 | 0.488000 | 0.666500 | 1.426000 | 80/599 (13.36%) |
| 0 | S3 | P | 32 | 0.306000 | 0.580500 | 1.405650 | 1.426000 | 20/32 (62.50%) |
| 0 | S3 | S | 95 | 0.296000 | 0.522000 | 8.431000 | 30.726000 | 40/95 (42.11%) |
| 0 | S3 | F | 127 | 0.296000 | 0.525000 | 8.431000 | 30.726000 | 60/127 (47.24%) |
| 120 | S0 | X | 5,618 | 0.413000 | 0.511000 | 0.531000 | 1.177000 | 276/5618 (4.91%) |
| 120 | S0 | S | 197 | 0.413000 | 0.512000 | 0.761000 | 1.177000 | 30/197 (15.23%) |
| 120 | S0 | F | 197 | 0.413000 | 0.512000 | 0.761000 | 1.177000 | 30/197 (15.23%) |
| 120 | S2 | C | 24 | 0.530000 | 0.537000 | 0.538000 | 0.538000 | 17/24 (70.83%) |
| 120 | S2 | P | 24 | 0.530000 | 0.537000 | 0.538000 | 0.538000 | 17/24 (70.83%) |
| 120 | S2 | S | 173 | 0.413000 | 0.513000 | 0.765400 | 1.177000 | 24/173 (13.87%) |
| 120 | S2 | F | 197 | 0.413000 | 0.516000 | 0.761000 | 1.177000 | 41/197 (20.81%) |
| 120 | S3 | C | 65 | 0.426000 | 0.533000 | 0.539000 | 0.569000 | 38/65 (58.46%) |
| 120 | S3 | P | 49 | 0.426000 | 0.535000 | 0.539000 | 0.569000 | 34/49 (69.39%) |
| 120 | S3 | S | 148 | 0.413000 | 0.513000 | 0.786950 | 1.177000 | 21/148 (14.19%) |
| 120 | S3 | F | 197 | 0.413000 | 0.517000 | 0.753000 | 1.177000 | 55/197 (27.92%) |

Frame 0 S3 priority의 median/p95/max는 0.5805/1.40565/1.426m이며, S2는 0.506/0.6653/0.690m다.
X의 최대 30.726m positive depth는 두 조건 모두 Surface 경로에 남았다. 두 Surface의 p95/max는 8.431/30.726m로 같다.
C/P와 Surface의 depth 범위가 다르므로 Surface의 원거리 극단치를 S3 priority 선택의 직접 원인으로 돌릴 수 없다.
ROI와 depth 점유의 차이는 확인됐지만, 거리 경쟁·초기점·tie·priority seeding이 rim 손실에 기여한 비율은 미확인이다.

## 8. 구현과 source geometry 검증

| 변경 파일 | 핵심 위치와 내용 | diff |
| --- | --- | --- |
| [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py) | 47행 `reference_hits`, 100행 `count_ratio`, 106행 `staged_reference_coverage`; 기존 joint adjacency로 reference ID 집합 집계 | +74줄 |
| [run_rim_bottleneck.py](../../visual/runtime/test/run_rim_bottleneck.py) | 47행 source 계약, 84행 최소 geometry adapter, 97행 저장 geometry 검증, 176행 프레임 진단, 257행 entrypoint | 신규 342줄 |
| [test_sampling_evaluation.py](../../visual/runtime/test/test_sampling_evaluation.py) | 116행 `StagedCoverageTests`; 신규 테스트 8개 | +135 / −1줄 |

독립 entrypoint를 선택하여 기존 `run_sampling.py`의 전체 실험 기본 동작을 보존했다.
Production `sampling.py`, estimator/FPS, sampling algorithm, allocation, feature extraction 코드는 변경하지 않았다.
기존 `evaluation_context()`, Task02 `rim_reference()`, `CoverageMatcher`를 재사용했다.

- 입력 RGB-D/mask SHA256와 저장 source ID 및 실제 mask row 일치.
- `flat_index = v*W+u`, 저장 `(v,u)`, 물체 소속, invalid depth 및 mask overlap 제외 규칙 일치.
- 기존 RGB-D adapter와 `build_organized_geometry` 사용. CV-camera frame, metre 단위, pixel centre `(u+.5,v+.5)` 유지.
- 저장 selected XYZ와 source raster에서 재구성한 XYZ가 **exact equality**, 최대 오차 0m.
- `C ⊆ X`, `P ⊆ C`, `F ⊆ X`, `R(C) ⊆ O`, `R(F) ⊆ O`, `R(F) = R(P) ∪ R(S)` 성립.
- 저장 GT reference의 좌표와 **ID 순서까지 정확히 일치**. 실제 매칭에는 저장 배열 사용.
- 원본 source snapshot hash 검증. 기존 `CoverageMatcher.__init__/score`, `evaluation_context`, 평가 상수와 Task02 정의는 원본 AST와 동일.

착수 시 source byte 차이가 있던 `edge_evidence.py`는 마지막 빈 줄 제거뿐이며 AST가 같았다.
추가 normal estimation 없이 평가 ROI를 재구성하도록 mask·validity 필드만 전달했고, 기존 negative ROI count와 정확히 일치했다.
GT 이산화, tolerance, ROI 또는 assertion을 완화하지 않았다.

## 9. 테스트와 실제 진단 실행

저장소 루트의 기존 `.venv`에서 실행했다.

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03d-mpl .venv/bin/python -m unittest \
  visual.runtime.test.test_sampling_evaluation \
  visual.runtime.test.test_sampling \
  visual.runtime.test.test_sampling_geometry -v

WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03d-mpl \
  .venv/bin/python -m visual.runtime.test.run_rim_bottleneck

git diff --check
```

| 검증 | 결과 |
| --- | --- |
| 관련 테스트 31개, 신규 8개 포함 | **31 PASS / 0 FAIL / 0 SKIP** |
| 같은 pair의 2px AND 3mm, 서로 다른 node가 각 조건만 만족하는 반례 | PASS |
| 한 node의 다중 reference, 여러 node의 동일 reference 중복 제거 | PASS |
| Priority/Surface overlap·union, Surface의 후보 밖 reference 보완 | PASS |
| 빈 point 집합, 관측/GT 분모 0, S0 N/A | PASS |
| source alignment와 손상된 geometry/source 거부, 평가 정의 변경 시 실패 | PASS |
| Frame 0 target 103/103, S0/S2/S3 6/16/3 | 정확히 일치 |
| Frame 120 target 41/8, S0/S2/S3 3/0/2 | 정확히 일치 |
| 기존 frame-level coverage와 negative ROI count | 정확히 일치 |
| 원본 입력 실행 전후 SHA256, 실행 source snapshot·UUID·sidecar | PASS |
| `git diff --check` | PASS |

기존 frame-level coverage는 저장된 1px/2mm, 2px/3mm, 3px/5mm 항목을 모두 재대조했다. 새 threshold/tolerance sweep은 실행하지 않았다.
기존 테스트의 sampler 실행은 synthetic fixture에 한정한다. 실제 ep000에서는 저장된 frame 0/120과 세 조건만 진단했으며, 54조건×121프레임을 재실행하지 않았다.

기존 `ExperimentLogger`, `experiment_artifacts.digests`, `register_existing_output`으로 UUID·local metrics·manifest·sidecar를 관리했다.
`WANDB_MODE=disabled`이며 원본 입력에는 새 sidecar를 쓰지 않고 `source_run_id`와 hash로 연결했다. 원본 NPZ/JSON을 덮어쓰지 않았다.
실행 시 `git_dirty=true`와 source snapshot을 보존했다. 이번 문서 정리는 해당 결과를 읽어 작성했으며 추가 진단·테스트를 실행하지 않았다.

## 10. 설명 가능한 단계와 후속 판단

Frame 0은 관측 손실 없이 S3가 더 넓은 후보 reference coverage를 확보했지만, 실제 priority에서 더 적게 남고 Surface 보완도 없는 단계 차이가 확인됐다.
Frame 120은 33개 관측 손실과 양쪽 후보의 0/8 rim coverage를 구분할 수 있다. S3의 최종 2개는 priority 보존이 아닌 Surface 보완이다.
이 범위를 넘어 오염이나 원거리 point가 FPS 거리 경쟁을 통해 얼마만큼 손실을 만들었는지는 저장된 집합만으로 확정할 수 없다.

**Phase 1의 단계 구분에는 FPS trace가 필요하지 않았다.** 이후 질문이 S3의 91개 후보 reference가 왜 priority에서 3개만 남는지라면,
별도 승인 후 최소 범위로 frame 0 Box S2/S3 기본 조건의 priority 선택 순서·source flat ID·현재 최소 거리 제곱·argmax/tie를 기록하는 것이 유용하다.
GT rim hit 여부는 선택 후 annotation에만 사용한다. Surface 보완 차이까지 조사할 때만 같은 두 조건의 Surface 초기 상태와 선택 순서를 추가한다.
Trace만으로 filtering이나 allocation 변경의 인과 효과까지 입증되는 것은 아니다.

FPS 내부 trace, sampler 변경, GT quota, filtering ablation, threshold sweep, PCL, Graph/BC 학습, GPU 실험은 실행하지 않았다.
CUDA/MPS와 새 simulator backend도 실행하지 않았다. 설치·외부 업로드·commit은 없었다.
**Phase 1은 완료했으며 다음 변경은 별도 검토·승인 전까지 진행하지 않는다.**

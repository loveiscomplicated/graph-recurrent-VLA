# Task03-E — Depth-Weighted Structural FPS & Local Geometry Preservation

전체 HTML·NPZ·실행 소스와 원래 run 기록은 [HF 후속 게시 인덱스](../../visual/results/publications/sampling_followups.md)에 있다. Git에 포함한 핵심 그림·작은 수치는 아래 미리보기 링크로 확인할 수 있다.

실행 UUID: `7f657cf88baf4f1f89e594252e243e1c`. ep000 Frame 0의 geometry-level offline ablation을 완료했다. 네 조합의 λ=1 exact reproduction 관문은 모두 PASS다.

## 고정 조건과 baseline 관문

- S2(FD2), S3(PCA5), Peg/Box, num_points=256, structural_ratio=.25, threshold=20°, seed=0. λ는 1/.25/0만 실행했다.
- 기존 `fps_indices()`에 `[x,y,sqrt(λ)z]`를 Priority 거리 계산용으로만 전달했다. Camera XY-only는 normalized image UV FPS와 다르다.
- 후보 source index·raster 순서·support·첫 점·quota를 유지했다. Peg 32+97=129, Box 32+95=127, estimator/λ마다 256점이다.
- Task01/03 episode·mask hash, camera/intrinsic/cam2world, dense Z와 원본 depth, validity·mask·overlap 및 선택 XYZ/pixel/object가 exact 일치한다. Backprojection·normal·candidate·allocation은 재계산하지 않았다.
- 네 λ=1 Priority 32점 및 기존 Priority 초기 집합을 포함한 3D Surface/Final 순서·XYZ가 exact 일치한다. Frame0의 모든 저장 structural tolerance/거리/quadrant 지표, 전체 Surface 거리 통계, negative ROI 수치도 exact 재현했다. 기준 tolerance 변경은 없다.
- 원본 orientation 배열은 저장 후보를 그대로 복사하며 count/order hash와 저장 metadata의 candidate membership을 검증했다. 독립적인 normal 재계산에 의한 후보 재검출은 수행하지 않았다.

| 조건 | 후보 | 첫 flat index | Final budget |
| --- | ---: | ---: | ---: |
| S2/peg | 870 | 213708 | 129 |
| S2/box | 297 | 101598 | 127 |
| S3/peg | 703 | 214225 | 129 |
| S3/box | 599 | 96553 | 127 |

## Structural retention과 Surface trade-off

아래 nodes는 실제 joint-matching 선택 node 수이며 refs는 unique reference 수다. P 실제 선택 수는 32, F는 Peg 129/Box 127이다. 분모는 Peg observed 462(GT 464), Box observed/GT 103이다. Candidate와 Surface의 전체 수·matching node·reference ID 및 P/S 교집합은 JSON/NPZ에 별도 보존했다.

| 조건 | λ | P nodes / refs | F nodes / refs | Surface 추가 / overlap refs | Surface mean / p95 mm | P depth median / p95 m | Negative union P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S2/peg | 1 | 29 / 139 | 29 / 139 | 0 / 0 | 5.138 / 8.587 | 0.1930 / 0.3250 | 2/32 |
| S2/peg | 0.25 | 29 / 137 | 29 / 137 | 0 / 0 | 5.168 / 8.607 | 0.1865 / 0.3239 | 3/32 |
| S2/peg | 0 | 29 / 142 | 29 / 142 | 0 / 0 | 5.263 / 8.676 | 0.1835 / 0.3228 | 2/32 |
| S2/box | 1 | 4 / 13 | 5 / 16 | 3 / 0 | 11.992 / 20.179 | 0.5060 / 0.6653 | 3/32 |
| S2/box | 0.25 | 7 / 16 | 8 / 19 | 3 / 0 | 11.903 / 19.857 | 0.5025 / 0.6648 | 5/32 |
| S2/box | 0 | 7 / 17 | 8 / 20 | 3 / 0 | 12.019 / 20.210 | 0.5000 / 0.6625 | 6/32 |
| S3/peg | 1 | 30 / 151 | 30 / 151 | 0 / 0 | 5.170 / 8.550 | 0.2030 / 0.3329 | 2/32 |
| S3/peg | 0.25 | 30 / 143 | 30 / 143 | 0 / 0 | 5.268 / 8.694 | 0.1930 / 0.3325 | 2/32 |
| S3/peg | 0 | 30 / 160 | 30 / 160 | 0 / 0 | 5.284 / 8.609 | 0.1865 / 0.3325 | 2/32 |
| S3/box | 1 | 1 / 3 | 1 / 3 | 0 / 0 | 11.253 / 18.751 | 0.5805 / 1.4057 | 9/32 |
| S3/box | 0.25 | 2 / 6 | 2 / 6 | 0 / 0 | 11.461 / 19.088 | 0.5035 / 1.3683 | 11/32 |
| S3/box | 0 | 3 / 9 | 4 / 12 | 3 / 0 | 11.580 / 19.077 | 0.4955 / 1.3864 | 14/32 |

## 질문별 판단

1. Z 가중치를 낮추면 실제 Priority 위치가 달라진다. λ=.25/0에서 baseline 대비 교체된 Priority node 수는 S2/peg 26/27점, S2/box 24/25점, S3/peg 24/25점, S3/box 17/23점이다. λ=1→.25→0에서 Priority depth median은 Peg S2 .1930→.1865→.1835m, S3 .2030→.1930→.1865m, Box S2 .5060→.5025→.5000m, S3 .5805→.5035→.4955m로 가까운 쪽으로 이동했다. Peg는 face_0_2의 실제 node가 늘고 face_2_5가 줄었다. Box는 일부 rim 선분의 node가 늘었지만 S3의 최대 depth 1.426m는 유지되고 p95도 1.4057→1.3683→1.3864m로 높게 남는다. XY-only도 X/Y 자체의 depth 의존성을 제거하지 않는다.

2. 이 frame의 두 estimator 모두 Box rim coverage가 개선됐다. Priority unique reference coverage(λ=1→.25→0)는 S2 (13 → 16 → 17)/103, S3 (3 → 6 → 9)/103이다. Final은 S2 (16 → 19 → 20)/103, S3 (3 → 6 → 12)/103이다. Priority의 실제 matching node 수는 각각 4→7→7과 1→2→3, Final은 5→8→8과 1→2→4다. Reference 증가와 node 증가를 구분해야 하며 유지된 구간은 rim 전체의 일부다.

3. Peg 모든 구간이 더 촘촘하게 보존되지는 않았다. Final coverage는 S2 (139 → 137 → 142)/462, S3 (151 → 143 → 160)/462다. face_0_2는 S2 refs 24→36→40, nodes 4→7→7 및 S3 refs 29→43→44, nodes 4→7→7로 증가했다. 반면 face_2_5는 S2 refs 84→72→70, nodes 19→16→15 및 S3 refs 90→66→66, nodes 20→17→15로 감소했다. 전체 matching node 수는 모든 λ에서 S2 29, S3 30으로 같다. Peg의 Surface는 추가 reference를 하나도 보완하지 않았다. 예컨대 λ=0 S3는 30개 node가 160개 reference를 덮는다. Reference coverage 상승은 실제 node 수 증가를 뜻하지 않는다. 물리적 중심선 순서·간격은 저장 정보로 검증되지 않아 N/A다.

4. Surface representation의 손익은 물체·estimator에 따라 다르다. λ=.25에서 S2 Box는 rim 개선과 함께 Surface mean/p95가 11.992/20.179→11.903/19.857mm로 줄었지만 λ=0에서는 12.019/20.210mm로 소폭 증가했다. S3 Box는 11.253/18.751→11.461/19.088→11.580/19.077mm로 악화됐다. Peg는 두 control 모두 mean/p95가 baseline보다 증가했다. Box Priority negative ROI union은 S2 3→5→6/32, S3 9→11→14/32로 늘었다. Priority 변경에 따라 뒤따르는 원본 3D Surface FPS도 달라진다. Depth 분위수와 ROI는 설명용이며 filtering이나 일반 평면의 오선택 판정으로 사용하지 않았다.

5. Z 조정 후에도 전역 FPS의 국소 구조 누락이 남는다. S2 Box rim_axis1_+1은 모든 λ에서 0/25로 저장 visible span 46.33mm 전체가 reference 기준으로 누락된다. S3 Box rim_axis1_-1도 모든 λ에서 0/26이며 span은 48.74mm다. λ=0의 Peg도 S2 320/462, S3 302/462 observed references가 누락된다. S2 Box rim_axis1_-1의 실제 node 투영 간격은 일부 줄지만 다른 선분 전체 누락을 보완하지 못한다. 한 frame의 최적 λ나 production sampling 정책을 채택하지 않는다.

## 국소 분포와 물리적 간격

GT와 구간 label은 sampler의 입력이 아니다. Reference 배열 순서는 물리적 순서로 쓰지 않았다. 아래 coverage는 관측 reference 분모이며 corner의 복수 귀속은 ambiguous로 보존한다. Segment node 수는 해당 node가 매칭한 모든 reference가 하나의 label에 속할 때만 집계한다. 여러 label에 걸친 node는 ambiguous로 기록하므로 구간별 node 수를 전체 matching node 수로 단순 합산하지 않는다.

| 조건 | 구간 | observed | P covered λ=1/.25/0 | F covered λ=1/.25/0 | F actual nodes λ=1/.25/0 |
| --- | --- | ---: | --- | --- | --- |
| S2/peg | ambiguous | 2 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| S2/peg | face_0_2 | 91 | 24 / 36 / 40 | 24 / 36 / 40 | 4 / 7 / 7 |
| S2/peg | face_0_5 | 106 | 31 / 29 / 32 | 31 / 29 / 32 | 6 / 6 / 7 |
| S2/peg | face_2_5 | 263 | 84 / 72 / 70 | 84 / 72 / 70 | 19 / 16 / 15 |
| S2/box | rim_axis1_+1 | 25 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| S2/box | rim_axis1_-1 | 26 | 6 / 9 / 10 | 6 / 9 / 10 | 2 / 4 / 4 |
| S2/box | rim_axis2_+1 | 26 | 0 / 0 / 0 | 3 / 3 / 3 | 1 / 1 / 1 |
| S2/box | rim_axis2_-1 | 26 | 7 / 7 / 7 | 7 / 7 / 7 | 2 / 3 / 3 |
| S3/peg | ambiguous | 2 | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 / 1 |
| S3/peg | face_0_2 | 91 | 29 / 43 / 44 | 29 / 43 / 44 | 4 / 7 / 7 |
| S3/peg | face_0_5 | 106 | 30 / 34 / 49 | 30 / 34 / 49 | 5 / 6 / 7 |
| S3/peg | face_2_5 | 263 | 90 / 66 / 66 | 90 / 66 / 66 | 20 / 17 / 15 |
| S3/box | rim_axis1_+1 | 25 | 3 / 3 / 6 | 3 / 3 / 6 | 1 / 1 / 2 |
| S3/box | rim_axis1_-1 | 26 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| S3/box | rim_axis2_+1 | 26 | 0 / 3 / 0 | 0 / 3 / 3 | 0 / 1 / 1 |
| S3/box | rim_axis2_-1 | 26 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 0 / 1 |

- Peg: 기존 visible face map의 4-neighbour face transition과 seam_endpoints가 exact 일치한다. 각 reference의 face-pair membership과 모서리 복수 귀속을 저장했다. Raster seam의 두께·접점을 물리적 선분의 중심선으로 바꾸지 않았으므로 face-pair의 물리적 순서·간격·미보존 길이는 **N/A**다. 새 GT geometry나 큰 helper 구현은 범위에서 제외했다.
- Box: 기존 `rim_reference`가 사용한 pose/centre/radius와 transform을 역변환해 확인 가능한 네 선분 label을 붙였다. 새 reference를 생성하지 않았다. 1e-8m line identity 검사는 부동소수점 roundoff 확인이며 matching tolerance와 무관하다. 물리적 corner·복수 귀속은 ambiguous이며 spacing은 N/A다.
- S2 Box rim_axis1_-1에서 Final 실제 node의 최대 투영 간격은 λ=1/.25/0 순서로 25.54/15.62/19.83mm다. λ=0의 첫 node는 visible reference min보다 바깥에 있지만 joint match에 포함된 실제 선택점이므로 간격 계산에 유지했다. λ=0 S3 rim_axis1_+1은 두 node 사이 17.25mm 간격이 남는다. 다른 선분의 node 부족과 전체 누락은 별도다.
- 확인된 Box 선분에서는 실제 matching node XYZ를 해당 축에 투영해 정렬했다. 실제 인접 node 간격·끝부분 node 미배치 span과 연속 missing-reference run의 양 끝 span(mm)을 JSON에 기록했다. 분석 domain은 저장된 visible reference의 min–max span이다. 전체 물리 선분의 occluded 부분 길이는 평가하지 않았다.
- Missing-reference run span은 연속으로 누락된 저장 reference의 첫/끝 좌표 차이다. 단일 누락 reference의 span은 0이다. 연속적인 구조의 미보존 길이를 보간하지 않았으므로 continuous_unpreserved_length_mm은 null(N/A)이다. 이 값으로 실제 미보존 길이의 부재를 주장하지 않는다.
- Reference별 nearest pixel/3D 거리·nearest source index는 전체 선택점과 실제 structural matching node 집합을 나누어 P/F 각각 저장했다. 두 독립 최근접점으로 joint match를 합성하지 않는다. observed reference의 mean/median(p50)/p95/max, covered/missing ID, image quadrants도 저장했다. Peg 미관측 2개 ID는 별도 mask로 남겼다.
- 한 node가 여러 reference를 덮는 joint pair 목록, node별 reference 수, 여러 node에 덮이는 reference 수를 별도 기록했다. 높은 reference coverage를 촘촘한 실제 node 배치로 해석하지 않는다.

## Feature 대응과 제약

모든 선택 XYZ/pixel/source index/object ID/역할은 원본 관측과 exact 대응한다. Baseline과 겹치는 저장 행에서 23 metadata field를 source index로 gather하고 정렬을 검증했다. 이 구현은 같은 estimator·동일 baseline condition의 행만 exact available로 간주하는 보수적인 범위다. 다른 저장 condition의 feature를 임의 병합하지 않는다. `F/feature_available_positions`가 compact `F/available_features/*`의 대상 행을 가리키며, unavailable 행에는 feature 값을 만들지 않는다. P/S availability mask도 별도 저장한다. 과거 dense PCA5 feature cache 대체·feature 재추출·optimized PCA·Vision Tensor packaging은 미실행이다.

| 조건 | λ | exact feature rows / Final | unavailable |
| --- | ---: | ---: | ---: |
| S2/peg | 1 | 129 / 129 | 0 |
| S2/peg | 0.25 | 27 / 129 | 102 |
| S2/peg | 0 | 28 / 129 | 101 |
| S2/box | 1 | 127 / 127 | 0 |
| S2/box | 0.25 | 53 / 127 | 74 |
| S2/box | 0 | 55 / 127 | 72 |
| S3/peg | 1 | 129 / 129 | 0 |
| S3/peg | 0.25 | 37 / 129 | 92 |
| S3/peg | 0 | 26 / 129 | 103 |
| S3/box | 1 | 127 / 127 | 0 |
| S3/box | 0.25 | 53 / 127 | 74 |
| S3/box | 0 | 45 / 127 | 82 |

## 검증·재현과 완료 범위

- 최초 시도는 실제 FPS 이전의 runner 검증 오류로 중단했다. Surface metadata의 depth_only_candidate까지 false라고 가정한 assertion을 기존 source 계약에 맞게 Priority 행으로 한정했다. 기존 sampler·입력·평가 tolerance는 변경하지 않았다. [실패 시도 기록](../../experiments/9702d2709f0946f895e504554537c492/attempt_note.json)을 보존했다. 실제 λ=1 재현 불일치는 없었다.
- 필수 신규/기존 테스트 PASS: `Ran 56 tests in 0.726s; OK`. [tests.log](../../experiments/7f657cf88baf4f1f89e594252e243e1c/tests.log). λ 거리식, λ=0 Z 불변, source/feature alignment, GT·구간 label 독립성, 관문 실패 시 control 차단, budget/중복, same-pair·dedup 및 N/A 계약을 포함한다.
- 저장 결과 후처리의 물리적 간격 명시와 synthetic 계약은 [tests_final.log](../../experiments/7f657cf88baf4f1f89e594252e243e1c/tests_final.log)로 추가 확인했다. 실제 데이터 FPS 반복은 없다.
- [verification.json](../../visual/results/diagnostic_previews/03e/verification.json)에 실제 데이터의 네 exact baseline, source/input hash 보존, 모든 λ의 동일 후보·초기점·quota, 원본 XYZ/역할/feature 대응 및 상대 링크·PNG decode 결과를 기록했다.
- 실제 Chrome headless에서 HTML의 PNG 7개 모두 complete/naturalWidth/naturalHeight를 확인했다. [browser verification](../../experiments/7f657cf88baf4f1f89e594252e243e1c/browser_verification.json), [HTML 미리보기](../../experiments/7f657cf88baf4f1f89e594252e243e1c/html_preview.png). 최종 후처리 계약 12개도 PASS이며 원본 선택 NPZ는 변경되지 않았다.
- 기존 ExperimentLogger(team=visual, job_type=evaluate)를 WANDB_MODE=disabled로 재사용했다. UUID/project/group 규칙은 바꾸지 않았다. 기존 파일·source·artifact sidecar를 보존하고 새 산출물만 등록했다.
- CPU NumPy/SciPy offline 평가만 실행했다. Simulator·renderer·CUDA/GPU/backend는 이 작업에서 실행하지 않았다. 독립 episode/frame/seed, 추가 λ, production 정책 채택, 학습·GNN/BC·PCL·GPU 최적화·외부 upload·dependency 설치·commit은 미실행이다.
- FPS 실행 당시 source는 source_snapshot과 source_hashes.json의 sources_before/after에 보존했다. 결과 해석·그림의 색상·선분 투영 간격 요약을 저장된 결과에서 보완한 최종 source는 report_source_snapshot 및 report_generation_sources로 구분한다. 실제 데이터의 FPS는 반복 실행하지 않았다.
- Production sampling/normal/allocation/tracking 및 기존 Phase1–3 구현은 변경하지 않았다. Git 시작 상태가 dirty이므로 commit만으로 재현할 수 없으며 source snapshot·input hashes·implementation diff를 함께 보관한다.

재실행: `WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_priority_fps_metric_control --baseline <새 실행 전 baseline.json>`; 기존 보고서를 덮어쓰지 않는다. 각 실행은 새 UUID를 만든다. 결과 보고 후 다음 알고리즘 변경 없이 중단한다.

## 산출물

- [priority_fps_metric_control.html](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.html)
- [priority_fps_metric_control.json](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.json)
- [priority_fps_metric_control.npz](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.npz)
- [selection_overview.png](../../visual/results/diagnostic_previews/03e/selection_overview.png)
- [box_structural_zoom.png](../../visual/results/diagnostic_previews/03e/box_structural_zoom.png)
- [peg_structural_zoom.png](../../visual/results/diagnostic_previews/03e/peg_structural_zoom.png)
- [spatial_xz.png](../../visual/results/diagnostic_previews/03e/spatial_xz.png)
- [coverage_surface_tradeoff.png](../../visual/results/diagnostic_previews/03e/coverage_surface_tradeoff.png)
- [local_reference_distances.png](../../visual/results/diagnostic_previews/03e/local_reference_distances.png)
- [box_line_node_spacing.png](../../visual/results/diagnostic_previews/03e/box_line_node_spacing.png)
- [verification.json](../../visual/results/diagnostic_previews/03e/verification.json)
- [source_hashes.json](../../experiments/7f657cf88baf4f1f89e594252e243e1c/source_hashes.json)
- [implementation.patch](../../experiments/7f657cf88baf4f1f89e594252e243e1c/implementation.patch)
- [git_start.json](../../experiments/7f657cf88baf4f1f89e594252e243e1c/git_start.json)
- [git_end.json](../../experiments/7f657cf88baf4f1f89e594252e243e1c/git_end.json)

## 기존 근거

- [승인 전 계획](03e_priority_fps_metric_control_plan.md)
- [Task03 원본 comparison](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [Frame0 NPZ](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz)
- [Task01 dense geometry](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz)
- [Phase1](../../visual/results/diagnostic_previews/03d_phase1/diagnostic.json), [Phase2](../../visual/results/diagnostic_previews/03d_phase2/priority_fps_trace.json), [Phase3](../../visual/results/diagnostic_previews/03d_phase3/priority_fps_seed_control.json)
- [runner](../../visual/runtime/test/run_priority_fps_metric_control.py), [tests](../../visual/runtime/test/test_priority_fps_metric_control.py)

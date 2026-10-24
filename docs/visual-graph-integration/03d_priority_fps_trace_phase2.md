# Task 03-D Phase 2 — Frame 0 Box Priority FPS 선택 trace

**결과:** 저장된 S2(FD2)·S3(PCA5)의 Box Priority 32점은 현재 `fps_indices()` 호출 한 번씩으로 **원소별 순서까지 정확히 재현**됐다. S2는 32점 중 4점이 rim reference 13/103개를, S3는 1점이 3/103개를 덮는다. S3의 후보 coverage는 91/103으로 S2의 57/103보다 높지만, 실제 32점 선택 뒤에는 반대가 된다. 모든 거리·ROI·GT 비교는 선택 순서 검증 뒤에 계산했다.

- 새 진단 run: `aa5d71078e0841fd886b5026d6030ebc`. [HTML 시각화](../../experiments/aa5d71078e0841fd886b5026d6030ebc/priority_fps_trace.html), [32-step 원본 trace JSON](../../visual/results/diagnostic_previews/03d_phase2/priority_fps_trace.json), [검증 결과](../../visual/results/diagnostic_previews/03d_phase2/verification.json).
- 그림: [후보·선택 순서 overlay](../../visual/results/diagnostic_previews/03d_phase2/priority_overlay.png), [거리 경쟁·누적 coverage](../../visual/results/diagnostic_previews/03d_phase2/priority_distances_coverage.png). 분홍색 GT rim 표시는 **평가 전용**이다.
- 재현 자료: [source/input hash](../../experiments/aa5d71078e0841fd886b5026d6030ebc/source_hashes.json), [구현 diff](../../experiments/aa5d71078e0841fd886b5026d6030ebc/implementation.patch), [테스트 로그](../../experiments/aa5d71078e0841fd886b5026d6030ebc/tests.log), [진단 코드 snapshot](../../experiments/aa5d71078e0841fd886b5026d6030ebc/diagnostic_source/visual/runtime/test/run_priority_fps_trace.py).
- 입력: Task03 `2551448b0fae4c6bbce3caf48774275f`; Phase 1 `d3f14ee494894daabb8ba2fab8e1b00c`. [Phase 1 보고서](03d_box_rim_bottleneck_phase1.md)와 원본 결과는 변경하지 않았다.

`experiments/`의 링크는 현재 호스트 로컬 artifact다. 다른 checkout에는 없을 수 있다.

## 범위와 구현

Frame 0 Box, `S2_n256_r0.25_t20_s0` 및 `S3_n256_r0.25_t20_s0`만 읽었다. `num_points=256`, `structural_ratio=.25`, orientation threshold `20°`, seed `0`, Box Priority quota `32`다. 저장된 `orientation_flat`의 오름차순 raster 인덱스를 FPS 후보 순서로 사용하고, 원본 RGB-D에서 복원한 XYZ를 `float64`로 변환했다. Box object ID `1`을 넣은 `SeedSequence([0,1])`에서 파생 seed `3964924996`을 얻었다. 각 방법마다 기존 `visual/runtime/sampling.py::fps_indices()`를 **한 번** 호출해 원본 저장 Priority `flat_index`와 순서대로 대조했다. 별도의 선택 알고리즘이나 runtime hook은 없다.

순서 확인 후 저장 prefix를 입력으로 사용해 기존 FPS와 같은 좌표별 `float64` squared distance 갱신을 사후 재구성했다. 첫 step은 `default_rng(seed).integers(N)`의 결과이므로 거리 argmax와 분리하고, step 2–32에서는 선택 직전 잔여 전체 후보의 argmax와 정확한 동률 중 첫 candidate-local 행을 검증했다. 저장 reference와 기존 `CoverageMatcher`의 동일 point/reference pair **2px AND 3mm** 기준으로 GT annotation을 붙였다. Trace JSON의 각 step에는 원본 flat index, local index, XYZ, depth, 최소 squared distance, argmax/tie, 새 reference ID, 누적 coverage 및 negative ROI가 있다.

거리 경쟁의 A는 GT rim reference에 매칭되는 모든 잔여 후보, B는 **현재까지 덮이지 않은 reference에 매칭될 수 있는 잔여 후보**다. 각 집합의 선택 직전 개수·최대 최소거리·전체 승자와의 차이를 별도로 저장했다. 빈 집합과 최초 RNG step의 유한 거리 비교는 `null`로 표시한다. `sampling.py`, estimator, 기존 runner, allocation, Phase 1 코드는 변경하지 않았다. 구현 파일은 [runner](../../visual/runtime/test/run_priority_fps_trace.py)와 [tests](../../visual/runtime/test/test_priority_fps_trace.py) 두 개다.

## 실제 선택 결과

| 항목 | S2 FD2 | S3 PCA5 |
| --- | ---: | ---: |
| 후보 node / 후보 rim reference | 297 / 57/103 | 599 / 91/103 |
| Priority node / Priority rim reference | 32 / 13/103 | 32 / 3/103 |
| RNG 첫 candidate-local index | 58 | 117 |
| RNG 첫 원본 flat index (v,u) | 101598 (198,222) | 96553 (188,297) |
| RNG 첫 XYZ, m | (-0.040305, -0.069180, 0.308000) | (0.049605, -0.080684, 0.306000) |
| 첫 점의 rim match | 없음 | 없음 |
| 새 rim reference를 덮은 step | 18, 25, 26, 32 | 22 |
| 해당 step에서 새 ID 수 | 3, 4, 3, 3 | 3 |
| 비초기 31 step에서 전체 승자의 거리가 A/B 최대보다 큼 | 27 / 31 | 30 / 31 |
| 비초기 31 step에서 A/B 최대와 전체 승자가 같음 | 4 / 31 | 1 / 31 |
| 비초기 전체 argmax 정확한 동률 | 0 | 0 |
| 선택점의 알려진 negative ROI union 소속 | 3 / 32 | 9 / 32 |

초기 seed 값은 같지만 후보 수가 달라 첫 RNG 정수가 각각 다른 local row와 실제 위치에 대응한다. 첫 위치가 이후 거리 상태를 결정하며, 동일 seed라는 사실만으로 같은 공간적 출발점을 뜻하지 않는다. Step 2의 전체 승자 최소 squared distance는 S2 `0.1514078358 m²`, S3 `1.2937974106 m²`이고, A/B 최대는 각각 `0.0461672720 m²`, `0.0422444206 m²`다. 이 step에서 rim 후보는 전체 승자보다 가까운 거리 경쟁에 있었다.

S2는 step 18에서 reference ID `96–98`, step 25에서 `64,66,68,69`, step 26에서 `83–85`, step 32에서 `46,48,50`을 새로 덮었다. S3는 step 22의 flat index `107770`에서 ID `3–5`를 새로 덮었다. 나머지 S3 31개 Priority 점은 GT rim reference를 덮지 않았다. S2/S3 모두 A와 B의 **최대 거리 값은 32개 step에서 동일**했다. S2는 이미 덮인 reference에만 매칭되는 잔여 후보 때문에 A/B 후보 개수가 달라지는 step이 14개 있었고, S3에는 없었다. 따라서 B를 분리해 계산했지만 이번 trace의 최대 거리 승부 수치는 A와 같았다.

전체 승자보다 rim 후보 최대 최소거리가 작은 step이 S2 27개, S3 30개다. 이는 관측된 FPS가 GT rim coverage가 있는 후보보다 다른 후보의 공간적 최소거리를 더 크게 평가한 step이 많았다는 뜻이다. S3는 후보를 더 많이 갖고도 Priority에서 한 번만 rim에 닿았다. Negative ROI union도 S3 선택점에서 더 많았지만 ROI 겹침과 FPS 거리 경쟁을 손실의 독립적 원인이나 인과 기여율로 합산할 수 없다. 다른 seed, threshold, 정책 또는 원거리 point 제거의 반사실 결과는 실험하지 않았다.

## 검증

[검증 JSON](../../visual/results/diagnostic_previews/03d_phase2/verification.json)의 항목은 모두 **PASS**다. 원본 evaluated `sampling.py` SHA256과 현재 FPS·후보·allocation 정의의 AST가 일치했고, Phase 1 helper가 나머지 원본 evaluation 정의를 대조했다. 후보 순서와 원본 Box support, RGB-D XYZ, 저장 pixel/object/role/feature 행 길이와 orientation role을 검사했다. FPS 반환 flat index 32개는 저장 순서와 정확히 같았다. 모든 비초기 step에서 argmax와 첫 행 tie 규칙이 성립했다. 합성 테스트에서 실제 FPS 호출당 RNG factory 1회와 `integers()` 1회, 중복 XYZ의 고유 원본 행 선택, annotation 변경 전후 FPS 순서·거리 불변성을 확인했다.

후보 coverage 57/103·91/103 및 누적 Priority coverage 13/103·3/103이 Phase 1의 **실제 reference ID 집합**과 일치했다. 원본 입력의 실행 전후 SHA256이 같고, HTML의 로컬 링크 9개가 모두 존재한다. 신규·기존 sampling·evaluation matcher 테스트 **33개 PASS**. 실행 결과에는 PASS/FAIL/SKIP과 source hash가 저장돼 있다. 실험 결과 저장은 새 `ExperimentLogger` UUID와 기존 artifact 등록 계약을 사용했으며 외부 업로드는 하지 않았다.

## 미확인 사항과 후속 판단

이 결과는 **저장된 한 frame·두 조건의 실제 선택 경로**와 거리 경쟁을 설명한다. 최초 RNG 위치, 후보 분포, negative ROI, GT rim coverage 감소 사이의 관찰 관계를 보였으나 어느 요인의 인과 기여율도 입증하지 않는다. GPU·다른 backend는 이 호스트에서 실행하지 않았고, simulator, Surface FPS, 다른 frame, 전체 54조건 평가도 이번 Phase 2에서 실행하지 않았다. 저장 feature는 재추출하지 않았으므로 각 값의 독립적인 의미 검증은 이번 범위 밖이다. 저장 행 길이와 role 정렬, XYZ·pixel source 정렬을 검증했다. Sampling 정책 변경 여부는 이 보고서와 trace를 별도로 검토한 뒤 결정한다.

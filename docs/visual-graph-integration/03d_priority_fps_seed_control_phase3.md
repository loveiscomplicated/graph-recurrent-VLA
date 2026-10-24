# Task 03-D Phase 3 — FPS Initialization vs Candidate Distribution

전체 HTML·NPZ·실행 소스와 원래 run 기록은 [HF 후속 게시 인덱스](../../visual/results/publications/sampling_followups.md)에 있다. Git에 포함한 핵심 그림·작은 수치는 아래 미리보기 링크로 확인할 수 있다.

**상태: 완료 / 승인 범위 종료.**

저장된 seed 0/1/2의 B Priority coverage는 S2 7–13/103, S3 3–5/103이다. 공통 초기점 114957의 C Priority는 S2 **8/103**, S3 **3/103** (S2−S3 5 refs)이다. 이 초기점과 기존 FPS 조건에서도 후보 집합 차이에 따른 결과 차이가 남는다. 단일 공통 초기점으로 초기점 효과의 일반적 크기나 각 후보 특성의 독립적 인과 기여율을 추정하지 않는다.

## 실행 조건과 재사용 artifact

- 새 run: `f36f1e9e2bb64866b8d6c792db158635`. 기존 ExperimentLogger / artifact manifest·sidecar 계약 재사용, `WANDB_MODE=disabled` 로컬 기록.
- Frame 0, Box, S2(FD2) / S3(PCA5), num_points=256, structural_ratio=.25, threshold=20°, Box budget=127, Priority quota=32.
- 동일 GT rim line reference 103개. 동일 point/reference pair에서 `2px AND 3mm`, reference ID 중복 제거. GT rim/negative ROI는 평가 전용이다.
- [Task03 samples](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz), [frame metrics](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_metrics.json), [config](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json) (`2551448b0fae4c6bbce3caf48774275f`).
- [Phase1 JSON](../../visual/results/diagnostic_previews/03d_phase1/diagnostic.json) (`d3f14ee494894daabb8ba2fab8e1b00c`), [Phase2 JSON](../../visual/results/diagnostic_previews/03d_phase2/priority_fps_trace.json) (`aa5d71078e0841fd886b5026d6030ebc`). Seed 0 Priority ID/coverage 및 matched node는 Phase2를 그대로 사용하고 Phase1과 대조했다.
- 저장 결과 분석은 FPS·candidate·normal·feature를 재실행하지 않았다. Source geometry와 기존 matcher/ROI만 재사용했다. Final coverage는 저장 metrics 값을 그대로 보존했다.

## A. 저장된 Final / B. 저장된 Priority

| 방법 | Seed | 첫 flat | 첫 XYZ (m) | B rim node /32 | B Priority refs /103 | A Final refs /103 | Negative union /32 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| S2 | 0 | 101598 | (-0.0403046869, -0.0691796839, 0.307999998) | 4 | 13 (12.62%) | 16 | 3 |
| S2 | 1 | 119061 | (0.0423281267, -0.0462656245, 0.504000008) | 3 | 7 (6.80%) | 10 | 4 |
| S2 | 2 | 112919 | (0.0456230491, -0.0689199194, 0.497000009) | 2 | 7 (6.80%) | 10 | 4 |
| S3 | 0 | 96553 | (0.0496054664, -0.080683589, 0.305999994) | 1 | 3 (2.91%) | 3 | 9 |
| S3 | 1 | 117010 | (0.0381562486, -0.0567187481, 0.527999997) | 2 | 5 (4.85%) | 5 | 9 |
| S3 | 2 | 102621 | (-0.0419121087, -0.0674238279, 0.31099999) | 1 | 3 (2.91%) | 3 | 10 |

A는 최종 256점 중 Box 127점의 저장 coverage, B는 그중 실제 Priority 32점의 coverage다. B의 새 진단값으로 A를 바꾸지 않았다.

| 방법 / Seed | Priority depth min / median / p95 / max (m) | Unique GT reference IDs (B) |
| --- | --- | --- |
| S2 / 0 | 0.303000003 / 0.506000042 / 0.665299952 / 0.689999998 | [46, 48, 50, 64, 66, 68, 69, 83, 84, 85, 96, 97, 98] |
| S2 / 1 | 0.303000003 / 0.511000037 / 0.664399981 / 0.689999998 | [60, 62, 64, 79, 80, 81, 86] |
| S2 / 2 | 0.303000003 / 0.511000037 / 0.664399981 / 0.689999998 | [64, 66, 68, 69, 81, 82, 83] |
| S3 / 0 | 0.305999994 / 0.580500007 / 1.40565002 / 1.426 | [3, 4, 5] |
| S3 / 1 | 0.294999987 / 0.565500021 / 1.40565002 / 1.426 | [1, 2, 3, 87, 88] |
| S3 / 2 | 0.308999985 / 0.598999977 / 1.40565002 / 1.426 | [49, 51, 53] |

## C. 동일 초기점 offline control

후보 교집합은 252개다. GT와 무관한 `shared = sorted(set(C_S2) & set(C_S3)); initial_flat = shared[len(shared)//2]` 규칙으로 **114957**를 선택했다. XYZ는 `[0.030216798186302185, -0.07050585746765137, 0.5730000138282776]` m이며 S2/S3에서 exact equality다. Candidate-local index는 `{'S2': 152, 'S3': 449}`다.

- 각 후보 입력은 저장된 오름차순 raster 순서(S2 297개 / S3 599개)를 유지했다. 기존 `fps_indices(..., 32, seed=fps_seed(0, BOX), initial=[local])`를 방법당 정확히 1회 호출했다.
- `seed`는 기존 SeedSequence 계약에 맞게 전달했으며 initial 경로에서 RNG 생성/최초 draw가 없음을 실행 중 차단 검증했다. 초기점 포함 32개의 고유 source index다.
- 두 FPS가 끝난 후에만 control을 GT matcher 및 negative ROI로 평가했다. 전체 32-step trace는 생성하지 않았다.

| 방법 | 실제 node | Rim matching node | C Priority refs /103 | Unique GT reference IDs | Negative union /32 | Depth min / median / p95 / max (m) |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| S2 | 32 | 3 | 8 (7.77%) | [38, 40, 42, 78, 79, 80, 91, 92] | 5 | 0.303000003 / 0.505500019 / 0.665299952 / 0.689999998 |
| S3 | 32 | 1 | 3 (2.91%) | [5, 6, 7] | 9 | 0.294999987 / 0.578500032 / 1.40565002 / 1.426 |

공통 초기점 control의 실제 선택 flat index는 선택 순서대로 다음과 같다.

- S2: `[114957, 95454, 109831, 104212, 98091, 120059, 112394, 118550, 116497, 111367, 112908, 111383, 120073, 118013, 117521, 113933, 110854, 115982, 101595, 111880, 119042, 114965, 112907, 107798, 120080, 105748, 115471, 99555, 118035, 118008, 118523, 95015]`
- S3: `[114957, 106747, 87254, 109819, 87767, 84782, 109829, 101170, 108294, 108794, 105738, 107270, 119062, 107771, 108795, 102193, 109318, 105746, 113403, 103135, 116989, 111355, 105741, 84783, 106758, 119032, 112907, 94500, 102117, 107259, 105736, 116497]`

Reference 집합 비교(C):

- intersection: 0개, `[]`
- S2_only: 8개, `[38, 40, 42, 78, 79, 80, 91, 92]`
- S3_only: 3개, `[5, 6, 7]`

선택 source index 교집합은 3개 (`[112907, 114957, 116497]`), S2-only / S3-only는 각각 29 / 29개다. 각 선택 XYZ 및 candidate-local index와 전체 차집합은 아래 JSON에 보존했다.

## 해석과 연구적 판단

저장된 seed 0/1/2의 B Priority coverage는 S2 7–13/103, S3 3–5/103이다. 공통 초기점 114957의 C Priority는 S2 **8/103**, S3 **3/103** (S2−S3 5 refs)이다. 이 초기점과 기존 FPS 조건에서도 후보 집합 차이에 따른 결과 차이가 남는다. 단일 공통 초기점으로 초기점 효과의 일반적 크기나 각 후보 특성의 독립적 인과 기여율을 추정하지 않는다.

- 저장 seed의 S3 Priority 변동 폭은 2 references(1.94%p), S2는 6 references(5.83%p)다. S3의 덮인 ID 집합도 seed에 따라 달라진다. 이는 관측된 초기점 민감성의 범위이며 초기점의 독립적인 인과 효과를 정량화한 값은 아니다.
- 같은 seed끼리의 B S2−S3 차이는 seed 0/1/2에서 10/2/4 references다. C의 차이 5는 seed 0보다 작고 seed 1/2보다 크다. 따라서 seed 0의 큰 차이 전부가 지속되는 것은 아니지만, 이 공통 초기점에서도 S3의 낮은 coverage는 남는다.
- 관측된 seed별 B Priority 범위 및 같은 초기점 C를 함께 읽어야 한다. A와 C는 sampling 단계가 다르므로 동일 조건 성능 비교로 사용하지 않는다.
- 한 공통 초기점 결과를 전체 seed·episode에 일반화하지 않는다. 후보 수·위치·오염·원거리 depth의 독립적인 인과 기여율은 분해하지 않았다. Phase2의 기존 경로 분석을 재사용하며 새 trace를 만들지 않았다.
- 다음 연구 판단은 초기점 민감성과 후보 집합 효과를 함께 검토하는 것이다. 새 sampling 정책, 후보 제거, depth cutoff, GT quota, FPS 변형의 채택 근거로 사용하지 않는다. 추가 초기점/episode 검증은 별도 승인 대상이다.

## 구현·검증·source hash

- 추가 구현: [run_priority_fps_seed_control.py](../../visual/runtime/test/run_priority_fps_seed_control.py), [test_priority_fps_seed_control.py](../../visual/runtime/test/test_priority_fps_seed_control.py). 기존 production·Phase1/2·run_sampling 파일은 보존했다.
- Phase1 source geometry/matcher helper와 Phase2 `verify_fps_source` 검증 경로를 재사용했다. 새 sampler나 logger/publisher 구현은 추가하지 않았다.
- **PASS**: 신규 및 관련 기존 테스트. `Ran 45 tests in 0.644s; OK`. [test log](../../experiments/f36f1e9e2bb64866b8d6c792db158635/tests.log). 테스트의 FPS/전체 sampler 반복은 합성 입력에만 적용했다.
- **PASS**: evaluated source와 현재 FPS·candidate·allocation 정의 일치, seed0 Phase1/2 reference ID 및 저장 순서 일치, seed1/2 source/feature/role alignment, 23 metadata row 및 seed 간 공통 source feature equality.
- **PASS**: 저장 분석 FPS 호출 차단, 후보 교집합 252 / initial 114957, initial XYZ exact equality, 초기 RNG draw 차단, 32 unique indices·초기점 포함·실제 후보 membership·local/global 대응·raster 순서 유지.
- **PASS**: 합성 GT annotation 변경 시 선택 불변, 기존 same-pair joint matching 및 reference 중복 제거, 저장 negative ROI union 점유 일치, 원본 artifact·입력·기존 파일 hash 불변, HTML/Markdown 상대 링크와 PNG 생성 확인.
- **FAIL**: 없음. 불일치 시 assertion·tolerance·GT·seed 규칙을 변경하지 않고 중단하도록 구현했다.
- **SKIP / 미검증**: simulator·renderer·CUDA/GPU backend, 다른 frame/episode, 추가 공통 초기점 sweep, PCL·Graph·BC·정책/학습 실험, 외부 upload·dependency 설치·commit. 현재 호스트에서는 CPU NumPy offline 진단만 실행했다.
- Git 시작/종료 및 보존한 기존 변경: [start](../../experiments/f36f1e9e2bb64866b8d6c792db158635/git_start.json), [end](../../experiments/f36f1e9e2bb64866b8d6c792db158635/git_end.json), [diff](../../experiments/f36f1e9e2bb64866b8d6c792db158635/implementation.patch). 시작 Git dirty 상태이므로 commit만으로 재현할 수 없으며 source snapshot과 hash를 함께 보존했다.

| Source / input | SHA-256 |
| --- | --- |
| `visual/runtime/geometry.py` | `36d20b7873d5ef9e1381e3bbe319da89e16fa5b44455488786dadd250bd814a9` |
| `visual/runtime/rgbd.py` | `be76fef13d5f58a7199f3952c1f59ce3bf9d1a2ddac09a6773dce39e9f891164` |
| `visual/runtime/sampling.py` | `45860d0adfbefb2818b67fe5705cb8c36d7c344c1c52c44d4ca5abc6765e37ed` |
| `visual/runtime/test/run_priority_fps_seed_control.py` | `75be5d3cdc7b4050f5a75ac1e5c8bd4ca665b0b98b0229ef3abda9a80cdb8827` |
| `visual/runtime/test/run_priority_fps_trace.py` | `f41905efd42b375f74da5baaac770ade914df21f192bda3a6182792edae68f3b` |
| `visual/runtime/test/run_rim_bottleneck.py` | `8323bd3048438be9c90b663f1bc03ef6fdfd67d10eb123c1fd4930538428fda3` |
| `visual/runtime/test/sampling_evaluation.py` | `daebf8834d2c0d4ca7f95e927c87ed3d6c7fc3777909a49504e265443585d60b` |
| `visual/runtime/test/test_priority_fps_seed_control.py` | `69c8b3a1c420bec843bd8a3d948db009cd78c9ea6c99cb1376e15b60c3f47645` |

위 표는 실제 진단 실행 시점의 source hash다. 전체 입력/evaluated source hash와 원본 보존 기록은 아래 JSON에 있다. 그림·보고서만 저장 결과로 갱신한 경우 실제 실행 source는 `execution_source/`, 최종 source는 `diagnostic_source/` 및 `source_hashes.json`의 `report_generation_sources`로 구분한다. FPS 결과는 재실행하지 않는다.

## 산출물

- [priority_fps_seed_control.html](../../experiments/f36f1e9e2bb64866b8d6c792db158635/priority_fps_seed_control.html)
- [priority_fps_seed_control.json](../../visual/results/diagnostic_previews/03d_phase3/priority_fps_seed_control.json)
- [seed_initial_points.png](../../visual/results/diagnostic_previews/03d_phase3/seed_initial_points.png)
- [seed_priority_final_coverage.png](../../visual/results/diagnostic_previews/03d_phase3/seed_priority_final_coverage.png)
- [shared_initial_overlay.png](../../visual/results/diagnostic_previews/03d_phase3/shared_initial_overlay.png)
- [shared_initial_coverage.png](../../visual/results/diagnostic_previews/03d_phase3/shared_initial_coverage.png)
- [source_hashes.json](../../experiments/f36f1e9e2bb64866b8d6c792db158635/source_hashes.json)
- [verification.json](../../visual/results/diagnostic_previews/03d_phase3/verification.json)
- [tests.log](../../experiments/f36f1e9e2bb64866b8d6c792db158635/tests.log)
- [implementation.patch](../../experiments/f36f1e9e2bb64866b8d6c792db158635/implementation.patch)

**Phase 3 결과 제출 후 중단한다. 후속 변경은 별도 승인 후 진행한다.**

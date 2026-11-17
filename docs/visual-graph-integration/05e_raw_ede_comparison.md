# Task05-E — Raw Extended Delaunay vs Strict EDE

Run UUID: `0f107550c9b64f1a84677a18e988cb51`. [정적 HTML](../../experiments/0f107550c9b64f1a84677a18e988cb51/raw_ede_comparison.html).

## 결과와 책임 경계

고정 입력 16조건 / 32 object slice / 11,095 Point-row. Candidate U=30,143, Strict U=25,046, Raw U=30,143, 복원 U=5,097. Strict ⊆ Raw와 저장 거부 후보의 exact 차집합을 모든 조건에서 확인했다.
Strict isolated 302 → Raw 270; source identity로 재확인한 singleton 270. 새 연결을 얻은 Point 32; Graph CC 합계 442 → 408. 조건 간 중복 관측을 포함한 합계이며 정확성 향상 지표가 아니다.

Vision은 기존 candidate_edges(extended=True)를 독립 실행한다. Raw는 모든 후보를 그대로 사용하며 Supercover·Mask/Depth/overlap 경로·PairEvidence·depth-jump·reason/witness를 생성하지 않는다. Raw producer의 validator 금지 guard와 회귀 테스트를 통과했다.
Raw Receiver는 v2 Point/Feature/좌표/provenance, row/source metadata, 범위·object/component·canonical pair·origin enum·directed 양방향 exact 순서·deep-copy·실제 robot 3행 공급 시 +3만 검증한다. Qhull·Delaunay·collinear chain을 재실행하지 않는다. 구조적으로 일관된 metadata를 위조한 임의 chord는 Receiver 단독으로 검출하지 못한다. 후보 누락·추가 chord·잘못된 chain/origin은 별도 offline oracle이 거부한다. 기존 Strict 및 다른 정책의 수신 계약과 production default는 유지했다.

## CPU 비용

표의 정책상 존재하지 않는 stage의 0은 실제 수행 시간이 아니라 수행하지 않은 작업을 뜻한다.
현재 host 동일 interpreter/thread에서 조건별 정책당 warmup 10회 + 30표본, AB/BA 교대. 총 측정 960개. perf_counter wall / process_time CPU. IO·통계·assertion·그림은 측정 밖이다.
Aggregate wall 평균 Strict 101.886ms → Raw 8.261ms, 12.33×; CPU 12.33×. 16조건의 30표본씩 동일 가중 pooling이다.
정책이 실제로 제거한 작업량에 따른 비용 차이다. 동일 연산의 구현 최적화 speedup이 아니다. Strict는 Task05-C optimized single-normal SAT, fresh prepare_validation, reason/count/top-k witness까지 포함한다. Raw는 fresh 후보와 packaging만 필요하다. 양쪽 receiver의 검증 책임이 다르므로 receiver 시간은 별도 공개한다.
공통 준비는 packet deepcopy와 저장 component 접근이다. upstream observed component 생성·sampling·PCA5·Feature·EE 변환은 재실행하지 않았으며 비용에 포함하지 않는다. 전체 Vision→Graph→GNN의 20Hz 실측이 아니다. 기존 GPU B1 평균 69.254ms는 다른 측정 범위의 참고값이고 이번 GPU 실행은 없다.

| Policy/stage ms | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| Strict total_wall_ms | 101.886 | 89.955 | 248.912 | 313.167 |
| Strict total_cpu_ms | 101.771 | 89.947 | 248.711 | 309.599 |
| Strict common_point_component_preparation_ms | 0.078 | 0.077 | 0.091 | 0.106 |
| Strict candidate_total_ms | 4.873 | 4.109 | 11.591 | 66.604 |
| Strict validation_lookup_ms | 1.010 | 0.855 | 2.310 | 3.474 |
| Strict supercover_ms | 35.285 | 31.684 | 87.681 | 93.493 |
| Strict support_validation_ms | 12.177 | 10.949 | 30.166 | 31.392 |
| Strict depth_validation_ms | 30.616 | 25.929 | 76.231 | 97.060 |
| Strict edge_packaging_ms | 0.815 | 0.764 | 2.001 | 2.125 |
| Strict metadata_packaging_ms | 0.057 | 0.049 | 0.132 | 0.209 |
| Strict raw_packaging_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Strict packet_packaging_ms | 0.001 | 0.001 | 0.002 | 0.005 |
| Strict receiver_ms | 5.495 | 4.117 | 10.204 | 65.684 |
| Raw total_wall_ms | 8.261 | 7.282 | 19.960 | 75.390 |
| Raw total_cpu_ms | 8.253 | 7.282 | 19.959 | 75.097 |
| Raw common_point_component_preparation_ms | 0.079 | 0.078 | 0.089 | 0.127 |
| Raw candidate_total_ms | 4.785 | 4.108 | 11.589 | 66.784 |
| Raw validation_lookup_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw supercover_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw support_validation_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw depth_validation_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw edge_packaging_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw metadata_packaging_ms | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw raw_packaging_ms | 1.006 | 0.909 | 2.640 | 2.935 |
| Raw packet_packaging_ms | 0.001 | 0.001 | 0.001 | 0.002 |
| Raw receiver_ms | 1.964 | 1.796 | 4.587 | 5.508 |

| Condition | N | Strict mean ms | Raw mean ms | speedup |
|---|---:|---:|---:|---:|
| f0000_r0.5 | 950 | 146.898 | 10.988 | 13.37× |
| f0000_r1 | 1694 | 257.944 | 22.350 | 11.54× |
| f0018_r0.5 | 453 | 72.480 | 5.351 | 13.55× |
| f0018_r1 | 807 | 125.916 | 9.650 | 13.05× |
| f0027_r0.5 | 539 | 81.667 | 6.274 | 13.02× |
| f0027_r1 | 883 | 130.863 | 10.647 | 12.29× |
| f0045_r0.5 | 514 | 73.109 | 5.776 | 12.66× |
| f0045_r1 | 830 | 118.351 | 9.566 | 12.37× |
| f0056_r0.5 | 371 | 55.779 | 4.171 | 13.37× |
| f0056_r1 | 636 | 92.466 | 7.252 | 12.75× |
| f0066_r0.5 | 424 | 57.826 | 6.476 | 8.93× |
| f0066_r1 | 630 | 87.231 | 7.331 | 11.90× |
| f0080_r0.5 | 477 | 65.302 | 5.269 | 12.39× |
| f0080_r1 | 747 | 101.166 | 8.129 | 12.45× |
| f0120_r0.5 | 446 | 63.471 | 5.157 | 12.31× |
| f0120_r1 | 694 | 99.708 | 7.789 | 12.80× |

최대 N 조건: f0000_r1 (N=1694). 상세 p95/max와 최악 표본은 JSON benchmark.conditions / worst_sample에 보존했다.

## 연결성·시각 검토

모든 slice에 Strict/Raw/Difference의 동일 Point·축·camera view PNG, 원본 RGB 전체 context, raw mask, 전체 depth(원본 범위, frame별 최대값), 근거리 depth(0–1.2m clipping), 전체/근거리 Camera XYZ를 제공한다. 파랑=Strict 유지, 주황=Raw 추가. 빈 object도 보존한다.
추가 Edge별 저장 Strict reason/count/witness는 records JSON에 보존했다. 사유 합계는 중복을 포함한다. Top5 UV/3D 길이/endpoint depth 차이 및 reason 조합 첫 사례는 확대 PNG와 review.json에 연결했다. 이는 선택 기준이며 필터가 아니다.
실제 확대 검토: 43 / 전체 추가 5097 Edge. 전체 panel 검토: 32/32. 선택 사례 비율을 전체 오류율로 일반화하지 않는다. 물리적 정답 라벨은 아니다.

| Slice | N | Candidate U | Strict/Raw isolated | Strict/Raw CC | newly connected | Added U |
|---|---:|---:|---:|---:|---:|---:|
| f0000_r0.5 o0 | 499 | 1436 | 5/5 | 6/6 | 0 | 240 |
| f0000_r0.5 o1 | 451 | 1151 | 18/14 | 33/27 | 4 | 266 |
| f0000_r1 o0 | 896 | 2588 | 9/4 | 12/7 | 5 | 493 |
| f0000_r1 o1 | 798 | 2123 | 18/12 | 36/30 | 6 | 503 |
| f0018_r0.5 o0 | 453 | 1299 | 6/6 | 7/7 | 0 | 258 |
| f0018_r0.5 o1 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| f0018_r1 o0 | 807 | 2344 | 12/8 | 13/9 | 4 | 514 |
| f0018_r1 o1 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| f0027_r0.5 o0 | 416 | 1188 | 6/6 | 7/7 | 0 | 208 |
| f0027_r0.5 o1 | 123 | 309 | 1/1 | 7/7 | 0 | 36 |
| f0027_r1 o0 | 733 | 2121 | 9/8 | 10/9 | 1 | 396 |
| f0027_r1 o1 | 150 | 379 | 1/1 | 7/7 | 0 | 42 |
| f0045_r0.5 o0 | 348 | 985 | 4/4 | 5/5 | 0 | 138 |
| f0045_r0.5 o1 | 166 | 380 | 20/20 | 23/23 | 0 | 46 |
| f0045_r1 o0 | 593 | 1687 | 5/5 | 7/7 | 0 | 252 |
| f0045_r1 o1 | 237 | 568 | 20/19 | 24/23 | 1 | 85 |
| f0056_r0.5 o0 | 371 | 995 | 25/25 | 26/26 | 0 | 137 |
| f0056_r0.5 o1 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| f0056_r1 o0 | 636 | 1740 | 39/38 | 40/39 | 1 | 289 |
| f0056_r1 o1 | 0 | 0 | 0/0 | 0/0 | 0 | 0 |
| f0066_r0.5 o0 | 305 | 794 | 14/13 | 21/20 | 1 | 84 |
| f0066_r0.5 o1 | 119 | 315 | 6/6 | 7/7 | 0 | 30 |
| f0066_r1 o0 | 505 | 1328 | 12/12 | 22/22 | 0 | 155 |
| f0066_r1 o1 | 125 | 332 | 6/6 | 7/7 | 0 | 32 |
| f0080_r0.5 o0 | 249 | 648 | 9/8 | 13/12 | 1 | 61 |
| f0080_r0.5 o1 | 228 | 550 | 6/6 | 14/14 | 0 | 86 |
| f0080_r1 o0 | 398 | 1043 | 7/6 | 13/12 | 1 | 110 |
| f0080_r1 o1 | 349 | 879 | 11/9 | 19/17 | 2 | 176 |
| f0120_r0.5 o0 | 207 | 542 | 8/7 | 13/12 | 1 | 51 |
| f0120_r0.5 o1 | 239 | 611 | 8/8 | 16/16 | 0 | 113 |
| f0120_r1 o0 | 316 | 830 | 7/6 | 13/12 | 1 | 78 |
| f0120_r1 o1 | 378 | 978 | 10/7 | 21/18 | 3 | 218 |

저장 Strict 사유별 복원 Edge 수(중복 가능): `{"mask_outside": 4254, "invalid_depth": 70, "overlap": 162, "pair_unsupported": 4365, "pair_missing": 0, "delta_invalid": 0, "depth_jump": 1486}`.
사유 조합별 unique 수: `{"mask_outside|pair_unsupported": 3407, "mask_outside|pair_unsupported|depth_jump": 726, "depth_jump": 732, "mask_outside|invalid_depth|pair_unsupported": 63, "invalid_depth|pair_unsupported": 6, "mask_outside|invalid_depth|pair_unsupported|depth_jump": 1, "mask_outside|overlap|pair_unsupported": 55, "overlap|pair_unsupported": 80, "overlap|pair_unsupported|depth_jump": 25, "mask_outside|overlap|pair_unsupported|depth_jump": 2}`.

Frame0 Box 8.431m plateau는 기존 component 20의 12 Point, 11 chain Edge가 Strict/Raw 모두 유지된다. Raw added incident Edge 및 다른 Strict CC 병합 여부는 records.plateau에 별도 기록한다. 기존 오염 plateau를 Raw 신규 실패나 정상 복원으로 해석하지 않는다.

## 다섯 질문에 대한 관측상 답변

1. Strict가 거부한 5,097개는 한 종류가 아니다. 저장 audit 기준 mask_outside 4,254, pair_unsupported 4,365, depth_jump 1,486, invalid_depth 70, overlap 162개(중복 포함)다. 실제 확대 검토한 43개에서는 관측 공백/가림/구멍을 잇는 사례와 연속된 물체 모서리의 mask 계단을 스치는 사례가 함께 보였다. 353개 확대 후보를 생성했지만 전부 검토한 것은 아니며, 이 비율을 전체 오류율로 해석하지 않는다.
2. 명백히 문제가 있어 보이는 예는 Frame0 Box의 구멍을 가로지르는 연결(f0000_r1:3204, 길이222.0mm/|dz|219mm), gripper 가림을 건너는 Peg(f0018_r0.5:623/624), endpoint 깊이는 비슷해도 Box의 넓은 mask 공백/회색 gripper를 가로지르는 연결(f0120_r1:1045/1051)이다. 자연스러워 보이는 예는 연속된 Peg 측면 경계를 따라가는 f0000_r0.5:2/821/824/827과 짧은 금색 Box 모서리 f0027_r1:2123이다. 경계 소속이나 실제 접촉이 모호한 사례는 판단 불가로 남겼다.
3. Raw는 U를25,046→30,143(+20.35%)으로 늘리고 고립302→270, Graph CC 합계442→408로 줄였다. 새 연결 Point32개, 여러 Strict CC를 묶는 Raw CC23개를 기록했다. 평균 wall101.886→8.261ms(12.33×), process CPU101.771→8.253ms다. 후보 생성은4.873→4.785ms로 유사하고 Receiver는5.495→1.964ms다. 이는 픽셀 검사 작업 제거와 서로 다른 Receiver 책임을 포함한 정책 간 차이이며 전체 Vision/GNN20Hz 검증은 아니다.
4. Edge별 검증을 유지할 관측상 근거는 있다. 같은 저장 component 안에서도 직선 후보가 concavity, 가림, 구멍, 큰 깊이 불연속을 건널 수 있었고 Strict가 이를 제거했다. 다만 정상으로 보이는 모서리도 거부하므로 모든 거부가 타당하다고 말할 수 없다. Frame0의8.431m plateau12 Point/11 chain Edge는 두 정책에서 이미 유지되며 Raw added incident0개다. Strict 자체가 오염된 mask/depth의 물리 정확성을 보장하지도 않는다. 이번 자료만으로 모든 Edge별 검증의 비용이 필수라고 결론내리지 않는다.
5. Raw EDE를 후속 GNN ablation 후보로 유지하는 것은 합리적이다. 입력/후보/수신 구조 계약을 지키며 측정 비용이 크게 줄어 학습 성능과 오류 민감도를 분리해 비교할 가치가 있다. 명백한 관측 공백 연결도 보였으므로 production 기본값 채택의 근거는 부족하다. 최종 정책 채택 및 GNN 통합은 이번에 수행하지 않았다.

## 검증·재현

테스트: 117개 PASS. JSON/NPZ dtype·shape·bytes·order/NaN, packet 재수신, PNG decode, 통계 재계산, 상대 링크와 source 보호 hash를 verification.json에 기록한다. 실제 브라우저 검증: NOT_EXECUTED.
source snapshot / before snapshot / implementation.diff / source_hashes.json / input_gate.json / records / packets / timings_raw.json / review.json / coverage.json / tests.log를 동일 UUID에 보존했다. W&B disabled, ExperimentLogger(team=visual, job_type=evaluate) 사용. 외부 업로드·dependency 설치·commit 없음.
시각화의 빈 Edge 3D 패널 오류를 수정한 report-only recovery는 같은 UUID와 저장 측정값을 유지한다. 원본 계산 source snapshot, report source snapshot/diff, 실패 metadata와 numerical 보호 hash를 보존한다.
GPU·simulator·GNN/GRU/BC 학습·rollout: NOT_EXECUTED. 최종 정책 채택·GNN 통합은 별도 결정한다.

재현 명령: `WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m visual.runtime.test.run_raw_ede_comparison`

## 대표 확대 사례와 측정 범위 보충

확대 검토 43개: 문제가 있어 보임 19 / 자연스러워 보임 6 / 판단 불가 18. 검토자는 Codex이며 사람의 검토 또는 GT 정답 라벨이 아니다. 전체 5,097개를 전수 판정하지 않았다.

| Edge ID | 시각적 판단 | 실제 검토 이미지 |
|---|---|---|
| f0000_r0.5:2 | 자연스러워 보임 | [RGB·Mask·Depth·Camera XYZ](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0000_r0.5_o0_cases_00.png) |
| f0027_r1:2123 | 자연스러워 보임 | [RGB·Mask·Depth·Camera XYZ](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0027_r1_o1_cases_00.png) |
| f0000_r1:3204 | 문제가 있어 보임 | [RGB·Mask·Depth·Camera XYZ](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0000_r1_o1_cases_01.png) |
| f0120_r1:1051 | 문제가 있어 보임 | [RGB·Mask·Depth·Camera XYZ](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0120_r1_o1_cases_02.png) |
| f0080_r0.5:52 | 판단 불가 | [RGB·Mask·Depth·Camera XYZ](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0080_r0.5_o0_cases_00.png) |

Frame0 plateau: [0.5](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0000_r0.5_plateau_detail.png) / [1](../../experiments/0f107550c9b64f1a84677a18e988cb51/plots/f0000_r1_plateau_detail.png). 두 ratio 모두 Raw added incident U=0, 다른 Strict CC와 병합 없음.

Host: macOS-27.0.1-arm64-arm-64bit, CPU arm, Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1. Thread 환경 5개 모두 1. GPU는 실행하지 않았다.
고정 source UUID: `9185355a32b8433abc70f1738f983377`, `9f90293b227e4dbfaaa6290afad9cc3b`, `91ecd0413a87455cabc5e813cddaea8d`, `9f36f7c89dbd4097a8be9358138d1537`, `185bc034567d429daf3f1d09f94dc56c`, `be206b04f7b245c4a9436317b8f87987`.

| 최대 N 조건 f0000_r1 (N=1,694), wall ms | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| Strict | 257.944 | 252.680 | 305.842 | 313.167 |
| Raw | 22.350 | 20.395 | 23.501 | 75.390 |

| 정책별 최악 측정 표본 | Condition | sample (0-based) | total wall ms | candidate ms | receiver ms |
|---|---|---:|---:|---:|---:|
| Strict | f0000_r1 | 10 | 313.167 | 66.604 | 11.315 |
| Raw | f0000_r1 | 24 | 75.390 | 66.784 | 4.766 |

최악 표본에서는 후보 생성 구간이 약 66ms까지 늘었다. 원인은 이 실행만으로 확정하지 않는다. Raw의 전체 pooled p95가 19.960ms여도 최악 75.390ms가 있으므로 지연 보장이나 전체 Vision→Graph→GNN의 20Hz 실측으로 해석하지 않는다.
Stage 합은 total의 완전한 분해가 아니다. 나머지는 Python loop/bookkeeping·내부 assertion·타이머·함수 호출·그 밖의 packaging 비용이다. Strict의 support/depth 사유·count·top4 witness 비용은 정상 실행에 포함된다.
저장 source identity·observed component별 후보만 사용한다. source-index 기준 집합 대조와 모든 row/provenance byte 불변성은 offline oracle 및 packet readback으로 확인했다.

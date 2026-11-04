# Repository cleanup — 엄격한 초기 Peg 가시성의 탈락량과 수집 시간

Task ID: `repo_cleanup_strict_visibility_cost_20261104` · 2026-11-04 KST.

**기존 기록에 엄격한 판정 결과와 수집 시간이 있다.** 과거 128 관측에서는 strict
기준의 추가 탈락이 작았다. Native 512의 비율이나 전체 데이터·퍼셉션 cache 완료
시간을 보장하는 결과는 아니다. 이번에는 기록만 집계했다. Simulator·CUDA·Terra/
SAM2 실행, 데이터 선별·삭제·manifest 변경·commit·publishing은 하지 않았다.

후속 결정: 사용자가 이 기록을 확인한 뒤 초기 Main 관측의 `strict_100_pass`를
현재 계획의 기준으로 선택했다. [전체 결정 기록 D05](repo_cleanup_preflight_20261104.md)를 따른다.
이는 문서상의 기능 결정이며 실제 데이터나 collector를 변경한 것은 아니다.

## 1. 무작위 초기 상태에서 얼마나 추가 탈락하는가

[기존 가시성 CSV](../../outputs/task10i/stage1/verified_pool_v2/candidate_visibility_metrics.csv)의
native random reset 926개를 집계했다. Trajectory 926개가 아닌 초기 상태 시도다.
원래 수집은 기본 기준의 유효 후보 300개에 도달하면 멈췄다.

| 기준 | 통과 | 통과율 | 탈락 |
| --- | ---: | ---: | ---: |
| 기본 requested_pass | 301 / 926 | 32.51% | 625 |
| 엄격 strict_100_pass | 299 / 926 | 32.29% | 627 |

엄격한 기준의 추가 탈락은 **기본 통과 301개 중 2개, 약 0.66%**다. 전체 시도의
약 67.7%가 strict에서 탈락하지만 대부분은 기본 기준에서도 탈락한다.
Physical/replay/unique까지 통과한 기본 후보 300개 중 strict도 통과하는 것은
**298개**다. 같은 분포와 yield를 가정한 reset 시도 수의 단순 배율은
`300/298 = 1.0067`, 약 0.7% 증가다. Strict 전용 새 수집의 실측은 아니다.

추가 탈락한 두 후보의 저장 evidence도 확인했다.

| Candidate | 예상 Peg pixel | Depth 지지 pixel | 불일치 pixel |
| --- | ---: | ---: | --- |
| `10i_s1_0460` | 400 | 399 | Silhouette 경계 1pixel, GT Peg pixel 아님, 실제 Depth가 약 72.88mm 멂 |
| `10i_s1_0678` | 674 | 673 | Silhouette 경계 1pixel, GT Peg pixel 아님, 실제 Depth가 약 72.12mm 멂 |

둘 다 `nearer_pixels=0`, `farther_pixels=1`, `invalid_pixels=0`이었다. 이를 다른
물체가 앞에서 가린 실제 occlusion 두 건으로 단정하지 않는다. Strict가 기하 예상과
실제 raster 경계의 차이에도 민감하다는 점을 고려해야 한다.

## 2. 현재 Stage3 데이터셋에 적용하면 얼마나 줄어드는가

저장 초기 판정과 최종 manifest를 parent ID로 연결했다. 원본 HDF5를 새로 렌더한
결과가 아니며 manifest를 변경하지 않았다.

| 데이터셋 | 기존 logical episode | Strict 적용 시 남음 | 추가 제외 | Parent 변화 |
| --- | ---: | ---: | ---: | ---: |
| R train | 272 | 270 | 2 (0.74%) | 96 → 95 |
| D train | 272 | 270 | 2 (0.74%) | 96 → 95 |
| Shared validation | 48 | 46 | 2 (4.17%) | 16 → 14 |

R/D의 제외 대상은 같은 `existing_30032`의 2개 episode다. R/D는 데이터를 공유하므로
단순 합산하지 않는다. Validation은 `10i_s1_0678`, `existing_30050`의 각 1개다.
세 manifest를 raw path/hash로 중복 제거하면 **356 → 352개**다.
이는 학습 성능 비교나 새 split 제안이 아니다.

## 3. 초기 후보를 찾는 시간

과거 환경은 Linux CPU PhysX + RTX3090 CUDA rendering, 128×128 Main 관측이다.
[Timing summary](../../outputs/task10i/stage1/verified_pool_v2/timing_summary.json)의
시도당 평균은 다음과 같다.

| 계측 구간 | 평균 |
| --- | ---: |
| Reset + rendering | 227.43ms |
| Readback + segmentation | 5.94ms |
| 기본/strict flag를 함께 계산하는 가시성 검사 | 6.10ms |
| 물리 유효성 검사 | 12.18ms |

이 합에는 replay·파일 쓰기·압축·전체 검증 등이 빠져 있다. 생성 run의 logger 시작/
종료 간격은 **834.04초, 약 13분 54초**다. 회귀·후보 생성·replay·진단을 포함했고
끝에서 별도 GL 진단 실패로 failed 상태가 보존됐다. 후보 파일을 유지한 채 별도
verification run에서 검증했다. 따라서 “strict 후보 300개 정상 완료에 14분”이라는
실측으로 표시하지 않는다.

## 4. 실제 expert episode 수집 시간

초기 후보 탐색과 BC 행동 trajectory 수집은 별개다. Stage3는 기본 기준으로 고른
상태를 사용했고, 두 카메라 128×128·3 workers를 채택했다.

| 과거 수집 batch | Parent 시도 | Workers | Wall time |
| --- | ---: | ---: | ---: |
| Primary | 90 | 3 | 1,833.80초 ≈ 30분 34초 |
| Reserve replacement 3회 합계 | 3 | 각 1 | 119.43초 ≈ 1분 59초 |
| 위 batch wall time 합계 | 93 | 혼합 | **1,953.23초 ≈ 32분 33초** |

Full-task와 계획된 nominal/recovery 등 branch를 포함한다. Parent 한 개와 episode
한 개는 같지 않다. Engineering·별도 audit·final reporting·초기 후보 탐색은 이 batch
합계 밖이다. 보고 단계 오류는 별도 finalizer에서 raw 재수집 없이 수정했다.
Strict 전용 새 수집의 시간은 측정하지 않았다.

[Engineering benchmark](../../outputs/task10i/stage3/run/engineering_gates.json)는 같은
3개 Parent의 full-task+nominal에 1 worker 122.39초, 2 workers 81.72초,
3 workers 48.04초를 기록한다. 작은 비교를 임의의 데이터 규모에 그대로 외삽하지 않는다.

## 5. 앞으로의 512 데이터셋 총 시간에 대한 한계

위 시간은 과거 **128 raw 수집**이며 Terra/SAM2/Graph 비용을 포함하지 않는다.
목표 episode/parent 수, branch 구성, native 512 비용과 API/SAM2 조건이 정해지지
않았으므로 새 1차·2차 cache까지 포함한 총 시간을 확정할 수 없다.
512에서는 raster sample과 경계가 달라져 strict 탈락률도 달라질 수 있다.

기존 926개 기록은 엄격한 기준 때문에 후보 수집 시도가 크게 증가한다는 근거를
보이지 않는다. 다만 과거 추가 탈락 0.66%를 미래의 고정 비율로 사용하지 않는다.
새 수집을 자동으로 시작하지 않았으며 기준 결정 후 native 512 사전 측정을 검토할 수 있다.

## 근거와 검증

공통 `ExperimentLogger(team='graph', job_type='diagnostic')`를 W&B disabled로 재사용했다.
집계 run ID: `8d91965220b64163b50b856418a95e82`.

- [집계 JSON 및 입력 해시](../../outputs/repo_cleanup_preflight_20261104/strict_visibility_cost_audit/summary.json)
- [집계 스크립트](../../outputs/repo_cleanup_preflight_20261104/summarize_strict_visibility_cost.py)
- [과거 primary batch](../../outputs/task10i/stage3/collection_run_v3/primary_execution.json)
- [기존 후보 풀 보고서](10i_stage1_main_visible_candidate_pool.md)
- [기존 Stage3 수집 보고서](10i_stage3_expert_dataset_collection.md)
- [전체 정리 사전 조사·결정 기록](repo_cleanup_preflight_20261104.md)

CSV/JSON과 manifest 연결을 확인했다. 새 수집·모델 실행은 없으며 source 변경 없는
기록 집계이므로 기존 단위 테스트는 반복 실행하지 않았다.

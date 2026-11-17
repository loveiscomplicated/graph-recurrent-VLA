# Task10 — Plan A Cached Graph BC 최종 연구 결과

원본: [최종 callback 보고서](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/research_report.md). 원본의 실행 시점 서술과 결과를 보존하고, 문서 모음의 순서 및 상대 링크만 조정했다.

Plan A Current-only Geometry로 E0/E1 모두 Expert Action을 근사했지만, Offline Loss 감소가 일관된 Closed-loop 조작 능력으로 이어지지는 않았다. 최종 E0 Last·Validation Best는 각각 Test12에서 Grasp/Lift 1/12를 기록했고 E1은 0/12였다. 공식 Insertion은 네 조건 모두 0/12다. 단일 학습 seed와 12개 Test seed로 GNN의 일반적 우위를 확정하지 않는다.

Comparison UUID: `3788798a719f4ec4a66c224ca42f5d63`. Final recovery Logger UUID: `1dcfae7b29f04c2e91a6f9da146f34c1`. 이 보고는 원래 전체 요청의 Phase 0–6 및 통합 지시의 최종 분석을 마무리한다. 기존 실패 보고서·pipeline 상태·checkpoint·source는 보존했다. 원래 pipeline은 역사적으로 `FAILED test_E0_last`이며, 아래 48회 평가는 별도 보존된 reset 복원 평가다.

## 실행 범위와 완료 판정

완료: 검증된 C3 + A-SF cap128/object Current-only Geometry, Train96/Validation12의 13,452개 action 정렬 Graph, E0/E1 각각 11,487 updates, 등록된 중간 평가 각 54회와 별도 500-step smoke 각 1회, Last/Validation Best × Test12의 실제 48회, 시간축 비교, Q1–Q10/P1–P6, Markdown/static HTML/PNG, 원자료·영상·checkpoint의 로컬 회수와 hash/readback 검증.

운영상 미완료·제약: 현재 Discord 전달은 옛 알림 호스트에 접근할 수 없어 이용 불가다. 과거 1회 테스트 전달 성공 기록과 현재 이용 불가를 구분한다. 이 callback은 알림을 보내지 않았다. 실제 재부팅 후 완전 무인 복구와 250-step trajectory의 플랫폼 간 exact 재현성은 NOT_TESTED다. 원래 artifact index의 logger metadata 1개 hash 불일치는 보존하고 현재 원격/로컬 일치를 확인했다. 새로운 index는 현재 파일들의 hash를 별도로 제공하며 과거 index를 수정하지 않는다.

## 최종 Test12 실제 결과

| Model | Step | Reach | Grasp | Lift | Alignment | Insertion |
|---|---:|---:|---:|---:|---:|---:|
| E0_last | 11487 | 1/12 | 1/12 | 1/12 | 0/12 | 0/12 |
| E0_validation_best | 10750 | 1/12 | 1/12 | 1/12 | 0/12 | 0/12 |
| E1_last | 11487 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| E1_validation_best | 10750 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |

12개 고정 seed를 각 모델·checkpoint에서 한 번씩 실행했다. 48은 12개 seed를 네 조건에서 반복한 평가 횟수이며 48개 독립 seed가 아니다. 모든 episode는 250 action, 1,250 physics step, time-limit 종료다. 네 조건의 합산 평가 시간은 11426.3초다. 최종 평가 자체의 각 조건별 시간은 E0_last 2844.9초, E0_validation_best 2929.2초, E1_last 2819.2초, E1_validation_best 2833.0초다.

![Final Test 비교](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/figures/final_comparison.png)

[48회 최종 CSV](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/final_rollout_metrics.csv) · [checkpoint–episode–video 연결](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/checkpoint_rollout_links.json) · [12 seed reset 검증](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot/reset_verification.json)

## Q1 모든 Action 시점의 Graph를 생성했는가

예. Train96 11,975개, Validation12 1,477개, 총 108 episode/13,452 graph다. 모델 선택 이후 Test12 1,524개도 생성했다. 120개 cache의 SHA256, NPZ readback, action7/robot76/timestep/episode_id, Node21/Edge4, edge index 범위를 다시 검사했다. terminal T+1 관측에는 action을 붙이지 않았다. GNN 전 입력만 캐싱한다.

모델 입력 tensor는 finite다. 원시 F5 provenance에는 관측 불가능한 normal/variation의 NaN과 `point_valid`가 함께 보존된다. 이를 모델 입력 NaN과 혼동하지 않는다. 유효로 표시된 원시 feature의 finite 여부도 검사했다. 원래 pilot의 streaming/replay·tracker reset·action alignment 검증과 실제 source hash를 보존했다.

[Cache manifest](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/cache_manifest.json) · [Pilot 검증](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/cache_pilot_verification.json) · [이번 local audit](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/local_audit.json)

## Q2 GNN과 GRU가 End-to-end로 학습됐는가

예. Full training CSV가 두 모델 모두 step 1–11,487의 연속 update를 기록한다. encoder/GRU/head gradient와 finite loss/norm을 확인했다. 원래 real-data smoke는 fresh full training과 분리되어 있고 save/resume exact PASS다. Sequence32, hidden144, hidden carry/TBPTT detach/episode reset/padding mask, Method B normalization을 유지했다. E1은 message passing이 없는 point-set control이다.

70개 checkpoint를 CPU에서 deserialize하여 모델 tensor hash, cursor step, config hash, codec/normalizer, optimizer/scheduler/RNG/hidden·cursor 보존을 확인했다. 등록된 13개 시점 × 2모델의 evaluation state guard가 weight/optimizer/scheduler/RNG/hidden·cursor 일치를 모두 기록한다. 이번 callback은 학습이나 rollout을 재실행하지 않았다.

[학습 smoke](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/training_smoke_verification.json) · [70 checkpoint와 26 guard readback](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/readback_verification.json)

## Q3 기존 Full96 대비 Training Fit

| Model | Train96 MSE | Validation12 MSE | Test12 MSE | Train30003 MSE (137 actions) |
|---|---:|---:|---:|---:|
| E0 last | 0.00446280 | 0.00643313 | 0.00567991 | 0.00519468 |
| E0 validation_best | 0.00447478 | 0.00641585 | 0.00566095 | 0.00523914 |
| E1 last | 0.00525246 | 0.00705346 | 0.00629151 | 0.00504203 |
| E1 validation_best | 0.00526106 | 0.00703547 | 0.00628387 | 0.00505752 |

동일 seed30003의 기존 Full B Last native Action7 MSE는 0.003433573, Best는 0.00343804064다. 새 Last의 MSE는 E0 0.00519467717(기존 대비 약 1.51배), E1 0.00504202735(약 1.47배)로 이 seed에서는 더 나쁘다. 원본 baseline 숫자는 전체 Train96 평균이 아니므로 위 Train96 평균과 직접 비교하지 않는다. 같은 action/split/update 예산이지만 geometry, feature14→21, cap, parameter 수, validation 선택 기준·빈도, runtime 차이가 있어 성능 차이를 Geometry나 GNN 하나의 효과로 단정할 수 없다. Native Action MSE는 서로 다른 action 성분을 합친 수치이며 물리 단위 error는 원자료의 translation/rotation/gripper 항목으로 분리한다.

[Full B 비교 전제](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/baseline_comparison.json) · [전체 Offline metrics](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/final_offline_metrics.json)

## Q4 최초 성공은 어느 Optimizer Step인가

500-step smoke를 제외한 등록 Train/Dev 평가에서 E0 최초 Grasp는 step9000, Lift는 step10000이다. E0의 Reach/Alignment/Insertion은 관찰되지 않았다. E1은 step1000에서 Reach/Grasp/Lift가 처음 관찰됐고 Alignment/Insertion은 없었다. 이는 평가된 시점 중 최초이며 두 평가 사이의 실제 최초 학습 시점을 뜻하지 않는다.

| Model | Updates | Scored episodes | Reach | Grasp | Lift | Alignment | Insertion | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 11487 | 54 | 0/54 | 2/54 | 1/54 | 0/54 | 0/54 | 245447 |
| E1 | 11487 | 54 | 1/54 | 3/54 | 1/54 | 0/54 | 0/54 | 194759 |

Reach는 TCP와 peg actor origin의 거리 ≤30mm라는 별도 진단이다. Grasp는 설치된 `is_grasping`, Lift는 grasp와 5mm 초과 상승이 3step 지속되는 조건이다. 따라서 Grasp/Lift를 만족해도 Reach flag가 false일 수 있다. 이 flag들을 엄격한 순차 subtask 성공률로 해석하면 안 된다. Insertion은 공식 환경 success를 사용한다.

## Q5 Validation Loss와 Closed-loop는 함께 개선됐는가

Validation Method B objective는 step250→11487에서 E0 0.262327→0.054594, E1 0.259191→0.055273으로 감소했다. 고정 Dev2의 step1000→Last에서 E0 Reach/Grasp/Lift는 모두 0/2→0/2다. E1은 Reach 1/2→0/2, Grasp 1/2→1/2, Lift 1/2→0/2다. Offline 개선과 일관된 조작 개선이 함께 나타났다는 근거는 없다. 작은 반복 Dev2의 관찰 결과이며 모집단 추세나 인과관계를 검증한 것은 아니다.

![학습 곡선](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/figures/learning_curves.png)

![고정 Dev2 Closed-loop 곡선](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/figures/closed_loop_curves.png)

일반 평가의 Train1+Dev2와 확대 평가의 Train1+Dev5를 구분했다. 확대 Dev5는 Dev2를 포함한 총 5 seed이며 중복 집계하지 않았다. 모든 등록 시점과 실제 checkpoint SHA가 연결된다. [고정 분모 시간축](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/fixed_dev2_timeline.json) · [원래 중간 평가 CSV](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/periodic_rollout_metrics.csv)

## Q6 중간 Checkpoint가 Last보다 나았는가

E1 step1000의 고정 Dev2에서 Reach1/2·Lift1/2가 Last의 각0/2보다 높았다. Dev를 반복 관찰한 기술적 사례이며 Test 우위나 최적 checkpoint 선정 근거로 사용하지 않았다. Last/Validation Best 선택은 2026-10-28T20:10:20.155430+00:00에 동결했고 두 모델 모두 Best step10750이다. 선택 고정 이후 Test cache/offline/final closed-loop를 수행했다.

[선택 동결](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/model_selection_frozen.json) · [중간 관찰 사례](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/intermediate_better_than_last.json)

## Q7 E0와 E1의 실제 차이

E0 Last Validation native MSE 0.00643313은 E1 0.00705346보다 낮다. 반대로 seed30003 Last fit은 E1이 더 낮다. 최종 Test12에서 E0의 Grasp/Lift1/12와 E1의0/12 차이는 관찰되었지만 양쪽 모두 Insertion0/12이며 단일 seed·작은 Test 집합에서 일반 우위를 입증하지 않는다. E0/E1 parameter는 245,447/194,759로 다르다. 같은 Graph/loss/GRU/seed42/optimizer/평가 절차를 썼어도 message passing만의 순수 효과로 해석할 수 없다.

등록된 periodic 평가 포함 training wall time은 E0 14,773.0초, E1 15,116.4초다. 이는 encoder-only inference benchmark가 아니다. Final trace의 perception+policy 시간도 전체 DINO/SAM2/geometry를 포함하며 20Hz real-time 주장을 뒷받침하지 않는다. [episode별 비용/누락/행동 통계](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/failure_trace_summary.json)

## Q8 실패 Episode의 첫 문제

Final에서 E0 Last/Best 각각 Reach 미충족11/12, Grasp 미충족1/12이고 E1은 Reach 미충족12/12로 저장됐다. `first_failed_stage`는 가장 앞에 있는 미충족 predicate 이름이며 시간상 최초 원인이나 순차 stage transition을 판정하지 않는다.

예를 들어 E0 seed30009는 Reach false지만 Last는 t41 Grasp·t51 Lift, 최대28.0mm/최장13step lift streak다. Best는 t42 Grasp·t51 Lift, 최대50.1mm/최장202step streak다. 두 경우 모두 Alignment/Insertion은 없다. 원래 lift threshold를 사후 변경하지 않고 실제 trace와 함께 해석한다. seed30015는 Last t45/Best t43에 proximity Reach 후 Grasp가 없다. 따라서 초기 접근 실패·파지 실패·파지 후 정렬 실패 사례를 관찰할 수 있지만 Geometry 오염/누락, action fit, GRU drift 각각의 인과 기여는 분리하지 못했다.

![실제 Lift trace](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/figures/failure_trajectory.png)

[E0 Best seed30009 영상](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot/E0/rollouts/validation_best/seed30009/rollout.mp4) · [E0 Last seed30015 영상](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot/E0/rollouts/last/seed30015/rollout.mp4) · [E1 Last seed30048 실패 영상](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot/E1/rollouts/last/seed30048/rollout.mp4) · [전체 48회 trace 요약](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/failure_trace_summary.json)

원시 native action이 [-1,1]을 넘는 성분은 trace에 남으며 기존 controller의 clipping 계약을 적용한다. 이는 사후 수정 사항이 아니다. 관측 누락률이나 action box 초과를 측정했다는 사실만으로 실패 원인이라고 확정하지 않는다. GT object state는 평가/사후 분석에만 쓰며 Student input에는 제공하지 않았다.

## 첫 행동과 평가 subset의 추가 분석

| Last model | First Action7 MAE | Mean Action7 MAE over137 expert actions |
|---|---:|---:|
| E0 | 0.20136857 | 0.01815157 |
| E1 | 0.21000955 | 0.01527731 |

첫 Action의 오차는 Episode 전체 평균보다 크다. Expert 관측을 사용하는 offline 첫 예측에서도 이 오차가 존재하므로 모든 fit 차이를 student drift만으로 설명할 수 없다. 실제 live RGB/geometry와 후속 행동에 따른 분포 이탈은 별도로 고려해야 한다. [E0 첫 Action](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/final_offline/E0/last/train_seed30003/first_actions.json) · [E1 첫 Action](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/final_offline/E1/last/train_seed30003/first_actions.json)

| Model | Periodic subset | Actual episodes | Reach | Grasp | Lift | Insertion |
|---|---|---:|---:|---:|---:|---:|
| E0 | train_seed | 12 | 0/12 | 1/12 | 0/12 | 0/12 |
| E0 | dev | 42 | 0/42 | 1/42 | 1/42 | 0/42 |
| E1 | train_seed | 12 | 0/12 | 0/12 | 0/12 | 0/12 |
| E1 | dev | 42 | 1/42 | 3/42 | 1/42 | 0/42 |

Train subset은 같은 seed30003의12회 반복이며 Dev subset은 checkpoint별2/5seed의 합42회다. 독립54seed의 성공률로 읽지 않는다.

| Final model | Missing peg frames /3000 | Missing box frames /3000 | Frames with native component outside box /3000 | Mean perception+policy s/action |
|---|---:|---:|---:|---:|
| E0 last | 363 | 398 | 1936 | 0.9204 |
| E0 validation_best | 65 | 573 | 1858 | 0.9486 |
| E1 last | 275 | 328 | 1472 | 0.9128 |
| E1 validation_best | 275 | 157 | 1540 | 0.9172 |

누락은 live graph의 해당 object node가0개인 frame으로 정의했다. 누락이 없다는 것은 mask/geometry가 깨끗하다는 뜻이 아니다. Native component box 초과는 controller 전 raw command에서 측정했고 rotation-vector norm 제한은 별도 계약이므로 이 숫자를 전체 clipping 횟수로 부르지 않는다. 위 시간은 전체 perception+policy 구간이며 encoder만의 비용이 아니다. 관측 누락·행동 box 초과·초기 fit 오차 각각의 인과 기여는 이번 데이터만으로 분리하지 못한다.

## Q9 Plan A Graph와 GNN의 유용성을 어디까지 지지하는가

End-to-end Action 근사와 일부 파지·상승 행동은 관찰됐다. 안정적인 Alignment/Insertion 제어, 기존 Full96보다 우수한 fit, GNN의 일반적인 정책 우위는 지지되지 않았다. 독립 Test seed는 반복 Dev만 사용한 분석을 보완하지만 12개 seed와 단일 training seed로 trajectory memorization을 배제할 수 없다. Object contamination/near-finger 오제거와 missing geometry의 알려진 한계를 그대로 보존했고 결과에 맞춘 tuning은 하지 않았다.

## Q10 향후 Geometry Memory 비교를 분리할 수 있는가

Geometry 생성, pre-GNN cache, encoder/GRU 학습, current-observation rollout이 모듈로 분리돼 있고 cache identity에 dataset/source/geometry를 포함한다. 향후 동일 BC·평가 계약으로 비교하려면 별도 provider와 새로운 cache/source identity 및 동등성 검증이 필요하다. Plan B adapter나 실제 Memory 비교는 구현·실행하지 않았다. 이번 결과를 이미 보장된 Plan B 호환성으로 표현하지 않는다.

## P1–P6 캐시 최적화 결과

P1: 원본 Detector 요청258회 중 정확 중복230회(89.15%). P2: 동일 입력 memoization 조건 P1은 전체216frame 304.53→275.07초(1.107배), 실제 hit230회이며 탐지·선택·SAM2 순서를 유지했다. P3: 실제8959 Wrist Point의 Mac CPU component median122.35→50.99ms(2.40배); 전체216frame의 gate/mask/graph exact 비교도 PASS다. P4: CPU geometry433frame의 1/2/4worker는103.46/55.98/34.80초, speedup1.848/2.973, efficiency92.40%/74.33%다. P5: 조합P4의 실제 전체216frame은254.27초(1.198배,16.50% 시간 감소)다. CPU 병렬 scaling을 곱해 전체 가속으로 주장하지 않는다. P6: Expert 전체216frame 및 별도16step live smoke에서 exact 동등성을 확인했으나 기존 scheduler의 ownership 이전 계약이 없어 frozen BC cache/training/periodic rollout에는 candidate를 적용하지 않았다.

최적화 candidate는 원본과 source hash가 다르며 원래 실험 source677개/config hash는 유지됐다. 조건당1회, 고정 실행 순서, 별도 호스트의1.92core quota라는 한계가 있다. GPU 병렬 full pipeline·별도 IPC latency는 NOT_TESTED다. 원래 최적화 보고서의 “BC 계속 실행 중” 문구는 당시 상태이며 현재 최종 상태로 읽으면 안 된다.

[독립 최적화 보고서](../../outputs/plan_a_optimization/aa8a96748f1d446a9fe41dc1cb92576b/optimization_report.md) · [최적화 검증](../../outputs/plan_a_optimization/aa8a96748f1d446a9fe41dc1cb92576b/verification.json) · [Graph 동등성](../../outputs/plan_a_optimization/aa8a96748f1d446a9fe41dc1cb92576b/graph_equivalence.json) · [Benchmark CSV](../../outputs/plan_a_optimization/aa8a96748f1d446a9fe41dc1cb92576b/optimization_benchmark.csv)

## Reset 오류와 별도 최종 평가의 정합성

원래 Test 첫 seed30048은 policy action0개에서 strict reset SHA mismatch로 실패했다. 그 폴더와 로그를 보존했다. Mac ARM64 원래 reference를 재현하여 Test12 전부의 원본 hash가 exact였고, Linux는 일부 peg/box quaternion 성분이 최대1.1920929e-7 달랐다. qpos/위치/크기는 같았다. Torch2.8 CPU만으로 바꾼 진단도 해결되지 않아 특정 library/hardware math 한 원인으로 분리하지는 못했다.

복원 평가는 별도 script/output/Logger identity로 초기 actor pose만 byte-verified reference로 복원한 뒤 원래 source의 reset guard를 그대로 통과했다. 모든12seed에서2회 반복 exact PASS다. 초기 reset reference는 policy action 전에만 쓰고 live Graph를 만드는 point source나 policy feedback으로 쓰지 않는다. 관측을 재생성한 뒤 frozen rollout 함수를 사용했다. 이때 recovery script hash는 원본 source hash와 다름을 명시한다. 원본 hash 검사 삭제·완화·seed교체·재학습은 없었다.

[Reset 원인과 검증](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/reset_fix_report.json) · [복원 script identity](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/implementation_identity.json) · [12seed×2 reset 결과](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot/reset_verification.json) · [원래 실패 로그](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/test_E0_last.log)

## 재현성 Identity와 검증 결과

Frozen source identity: `67e6ff39b627291a0118d3a093133141e37d93b0285987fa891d83a8d9dfc12e`. Config SHA256: `587c575e5f8356ec5e8b147791a67697bef8d39733b4241a91b970295d4645ad`. Dataset identity: `eef8be2904980f9effcdafba320bbc5a6bf1a6f55aadd0975d420dbef9c2e7df`. Split identity: `2471629fbc55073acc3b3fbce3cc7b1d36f209c3d318e7c1ac5c4e6f690edae7`. Runtime는 Torch2.14.1+cu130, ManiSkill3.0.1, SAPIEN3.0.3, NumPy2.5.4, Transformers5.19.0, RTX3090이다. 실제 simulator는 CPU PhysX/SAPIEN이고 perception/policy/training은 CUDA였다. 이번 callback의 실행은 로컬 CPU 파일·영상 readback 및 원격 read-only hash audit다. CUDA PhysX/MPS/20Hz real-time은 실행하지 않았다.

| Model | Step | Checkpoint SHA256 |
|---|---:|---|
| E0_last | 11487 | `57e57e524d113ac942f2c57740a3e0e5ec70bd7126d2384c77145042385d54a9` |
| E0_validation_best | 10750 | `4772fc15e323b171ab11c63baa115d505a936cf8900dacd6f8f5cc9cc23f6cc4` |
| E1_last | 11487 | `bd6e5167e3aff41936b1ea945ca8e126dc99c2c4db86a6612a3b6b6e18300a73` |
| E1_validation_best | 10750 | `45cf706ed6bc0b8846332c6a2cd4913a8c0315fd1028b0d4008cfa91723e6435` |

PASS: 원격 frozen source677/677, config, local/remote checkpoint SHA70/70와 모델 tensor/cursor readback, local cache SHA/NPZ120/120, periodic checkpoint/step/video links110/110(smoke2포함), final links48/48, 모든158영상의 첫·마지막 frame decode/20fps/frame수, final 전체250action의 trajectory·trace·saved live graph finite 검사, recovery746파일 원격/로컬 SHA·크기 exact. Registered 중간 평가의26 training-state guard PASS다. 원래 contract/pilot/gradient/TBPTT/normalization 검증 기록은 재사용했고 simulator test를 이번 callback에서 다시 실행하지 않았다.

기존 artifact index2918항목 중2917은 hash 일치한다. 제외되었던 input14개는 로컬 parent inputs에서 hash 일치로 해결했다. 나머지1개는 logger metadata로 원래 index 생성 뒤 finish timestamp가 기록됐다. 원격/로컬 현재 SHA가 같으며 historical mismatch를 숨기지 않았다. [전체 과거 index 대조](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/historical_artifact_reconciliation.json) · [Readback 결과](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/readback_verification.json) · [원격 read-only audit](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/remote_readonly_audit.json)

## 운영 상태와 로컬 회수

Supervisor의 원래 pipeline과 recovery evaluator는 종료됐고 GPU는 유휴다. 원격 event monitor PID6320과 기존 read-only backup을 유지했다. 접속: `ssh -p 13159 root@104.233.209.157 -t 'tmux attach -t plan-a-monitor-3788798a'`. 이 callback은 Codex/agent/monitor 추가 생성, process 종료·재시작, Discord 발송, commit, HF/W&B 게시를 하지 않았다.

과거 실제 callback fixture/중복·재귀·lock/긴 평가 grace 검증은 운영 경로에 남아 있다. 현재 사건은 단 한 handler가 처리하며 원래 event/handoff를 보존한다. 로컬 Mac이 꺼지면 로컬 callback/회수는 동작하지 않으므로 완전 무인 운영을 주장하지 않는다. 현재 notification unavailable은 과학 실험의 Insertion 실패와 분리한다. [기존 감시 검증](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/monitor_verification.json) · [이번 사건 ownership](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/ownership_verified.json)

원래 artifacts는 `/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA/experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot`에, final recovery는 `/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA/outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/recovery_20261029/evaluation_snapshot`에 회수했다. 입력14개는 `/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA/experiments/3788798a719f4ec4a66c224ca42f5d63/inputs`에 보존한다. 이번 보고·감사·연결 index는 현재 callback 폴더에만 추가했다. 원본 dataset의 위치/해시는 audit/manifest를 따르며 credential은 포함하지 않는다.

[현재 Artifact Index](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/artifact_index.json) · [완료·미완료 findings](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/findings.json) · [최종 검증](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/verification.json) · [Static HTML](../../outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/callbacks/ab66955bf3a83c8f9923707c/research_report.html)

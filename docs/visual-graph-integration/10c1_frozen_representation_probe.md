# Task10-C1 — Frozen Representation & Geometry Information Flow Diagnosis

Primary ExperimentLogger UUID `683c518f00ed45ec890dc4a4d1717726`. Parent BC `3788798a719f4ec4a66c224ca42f5d63`; Task10B `35877452d6d94d9691964fc8a8490858`; physical collection `bf2ab695001a487494a732727ece1be8`; analysis `8a7c11f84a8a43c7b6c67783db71c096`.

**완료 범위: 네 BC checkpoint의 원본 Expert13452-frame 동결 추출, 별도 검증된 dense09D train8/validation4 GT 진단, 두 decoder의 R0–R6 위치 및 heldout30003 paired22/secondary36 평가. 원본 Train96 GT 평가와 추가 Validation12 평가는 NOT_TESTED.**

## Source/Label와 실행 계약

원본 HDF108개를 읽었으며 per-timestep actor pose가 없고 cache에도 없다. 기존09D dense capture의 매 frame input/mask/GT hash를 검증하고, **저장된 predicted mask만** 재사용해 같은 C3/SF cap128 Graph를 재구성했다. DINO/SAM2·simulator는 실행하지 않았다. 기존09D의 저장 Graph를 전부 대조한 결과는 dense_graph_reconstruction_contracts.json에 있다. Graph 생성은 원래 PlanA.prepare/represent/graph이며 sampling/threshold/architecture를 변경하지 않았다. 이 자료는 원래 Expert cache와 동일한 observation이라고 주장하지 않는다. dense09D baseline만 fit했고 물리 perturbation은 fit하지 않았다.

Train8=981 frames, validation4=505 frames. Episode split은 기존09D를 사전 고정해 유지했고 seed30003은 normalization/fit/selection에서 배제했다. Validation4는 기존 BC validation과 겹치므로 blind test가 아니다. Test12 파일을 열거나 hyperparameter에 사용하지 않았다. 09D Final73141/73142도 이번 fit에 미사용.

Target은 world actor origin에서 TCP translation을 뺀 뒤 **TCP 회전의 역변환**을 적용한 XYZ(m), 기존09B target과 동일하다. GT는 label/감사만 읽는다. Raw15는 입력 XYZ mean/min/max/bbox-center/span이며 centroid는 부분 관측 표면 중심이다. R3=Peg mean slot index3(192:256), Box=index4(256:320); slot3+4와 robot76 추가 통제도 별도 측정했다. Readout은 TCP/Left/Right/Peg/Box 순서. R5는 실제 policy.encoder 출력128, R6는 현재 frame을 처리하고 action head에 전달되는 hidden144이다. 모든 dense episode t0 reset, 시간순 carry이며 미래 관측/action을 사용하지 않았다. R0-H는 현재 포함 과거32 Robot76의 mean/std/first/last304D, prefix는 첫 frame 반복 padding이다. Dense temporal source는 현재 관측 Graph이며 과거 표면 map은 없다.

원래 Train96+Validation12의 13452 Graph를 네 checkpoint 모두 CPU 순차 forward했다. 저장 action과 tolerance 검증, 원본/새 adapter readout/fusion/hidden/action, 원본 checkpoint/model hash, codec/normalizer 계약은 verification.json과 checkpoint_identity.json에 있다. 숫자 허용오차와 bit exact를 구분한다. Parameter 수 E0=245447, E1=194759. GPU/MPS/simulator backend는 이번 호스트에서 실행하지 않았다.

## Probe 예산과 분리

기존09B standardize/linear/mlp/absolute/delta와09D decode_saved를 재사용했다. Ridge alpha=[0,.01,.1,1,10,100]; 64-hidden ReLU MLP, AdamW lr.001, decay=[.0001,.01], max2000/patience150/min_delta1e-8, seed42. Train-only feature std floor1e-6, target std floor1mm; validation normalized MSE로 선택. 동일 budget을 모든 representation에 적용했다. Probe optimizer는 NumPy feature만 받는 독립 decoder parameter만 포함한다. 정책 parameter는 eval/requires_gradFalse, inference_mode이고 어떤 policy optimizer도 만들지 않았다. 입력 차원·parameter count·alpha/epoch와 후보곡선은 CSV/selection records에 보존한다. 고차원/작은 episode 표본 때문에 Full Readout이 slot보다 나빠도 정보 소실의 증명이 아니다.

## Probe A: 현재 Peg 위치

Validation frame-weighted Euclidean mean(mm), normal/missing 포함. Median/p95/axis MAE/R²/coordinate RMSE와 episode별 mean은 CSV. Episode macro는 representation_comparison.csv에 별도 기록한다.

| Representation | E0 Ridge | E1 Ridge | E0 MLP64 | E1 MLP64 |
|---|---|---|---|---|
| R0 | 36.58 | 36.58 | 30.99 | 30.99 |
| R0_H | 42.05 | 42.05 | 33.37 | 33.37 |
| R1 | 29.61 | 29.61 | 15.83 | 15.83 |
| R2 | 30.14 | 30.14 | 14.06 | 14.06 |
| R3 | 19.58 | 17.30 | 68.72 | 18.34 |
| R4 | 19.43 | 21.82 | 24.71 | 32.71 |
| R4robot | 23.62 | 24.71 | 28.89 | 28.35 |
| R5 | 32.45 | 26.11 | 27.33 | 26.08 |
| R6 | 37.88 | 29.54 | 35.17 | 29.32 |

## Probe B: same-state Peg displacement

Probe A에서 fit/선택된 decoder를 동결해 decode(perturbed)-decode(baseline)을 계산했다. 실제 GT pose 차이가 target이며 world XY5/10mm를 TCP로 회전한 결과다. Primary09A의22개와 secondary CurrentOBB36개를 합치지 않았다. 전체40개 peg 조건의 원래 eligible/classification을 그대로 보존했다. Robot/previous hidden은 pair마다 exact 검증했고 causal vision/calibration/physical validity는 원본10B audited collection 계약을 재사용했다. 본 실행에서 vision history를 새로 재생하지 않았다. Zero displacement 예측의 기준 error는 평균7.5mm이다. Cosine은 zero response에서 undefined로 남겼다.

Primary22 / Ridge:

| Model | Rep | n | Δerror mm | Cosine | Gain | Zero improvement mm |
|---|---|---|---|---|---|---|
| E0_last | R0 | 22 | 7.500 | undefined | 0.000 | 0.000 |
| E0_last | R0_H | 22 | 7.500 | undefined | 0.000 | 0.000 |
| E0_last | R1 | 22 | 5.634 | 0.694 | 0.580 | 1.866 |
| E0_last | R2 | 22 | 5.461 | 0.684 | 0.559 | 2.039 |
| E0_last | R3 | 22 | 17.484 | 0.414 | 1.706 | -9.984 |
| E0_last | R4 | 22 | 11.646 | 0.359 | 0.802 | -4.146 |
| E0_last | R4robot | 22 | 11.024 | 0.364 | 0.846 | -3.524 |
| E0_last | R5 | 22 | 8.160 | -0.099 | 0.010 | -0.660 |
| E0_last | R6 | 22 | 7.932 | -0.215 | -0.011 | -0.432 |
| E1_last | R0 | 22 | 7.500 | undefined | 0.000 | 0.000 |
| E1_last | R0_H | 22 | 7.500 | undefined | 0.000 | 0.000 |
| E1_last | R1 | 22 | 5.634 | 0.694 | 0.580 | 1.866 |
| E1_last | R2 | 22 | 5.461 | 0.684 | 0.559 | 2.039 |
| E1_last | R3 | 22 | 16.911 | 0.450 | 1.650 | -9.411 |
| E1_last | R4 | 22 | 19.102 | 0.321 | 1.729 | -11.602 |
| E1_last | R4robot | 22 | 19.491 | 0.311 | 1.403 | -11.991 |
| E1_last | R5 | 22 | 7.834 | 0.048 | -0.014 | -0.334 |
| E1_last | R6 | 22 | 7.461 | 0.148 | 0.013 | 0.039 |

Full per-condition predictions, magnitude ratio, relative representation L2, axis error, action/expert controller vector는 paired_displacement_metrics.csv 및 decoder_delta_vs_action_delta.csv에 보존했다. **Peg 위치Δ와 controller actionΔ는 서로 다른 물리량**이며 서로의 벡터를 정답으로 대입하지 않는다. Probe cosine/gain은 실제 PegΔ 기준, policy cosine/gain은 expert actionΔ 기준이다.

## Best checkpoint 재현

같은 data/split/budget의 Validation Best10750 확인이며 checkpoint를 재선택하지 않았다.

| Model | Representation | Abs mm | Δerror mm | Cosine | Gain |
|---|---|---|---|---|---|
| E0_validation_best | R2 | 30.14 | 5.461 | 0.684 | 0.559 |
| E0_validation_best | R3 | 19.52 | 17.458 | 0.414 | 1.705 |
| E0_validation_best | R4 | 19.56 | 11.644 | 0.355 | 0.796 |
| E0_validation_best | R5 | 32.45 | 8.161 | -0.101 | 0.010 |
| E0_validation_best | R6 | 37.88 | 7.935 | -0.217 | -0.011 |
| E1_validation_best | R2 | 30.14 | 5.461 | 0.684 | 0.559 |
| E1_validation_best | R3 | 17.30 | 19.289 | 0.438 | 1.783 |
| E1_validation_best | R4 | 21.99 | 20.290 | 0.306 | 1.790 |
| E1_validation_best | R5 | 26.13 | 7.832 | 0.048 | -0.014 |
| E1_validation_best | R6 | 29.55 | 7.461 | 0.146 | 0.013 |

## Q1–Q10 연구 판단

Q1 — SUPPORTED. Raw centroid/summary는 위 absolute와 paired 수치만큼 위치/이동 정보를 제공한다. centroid origin 편향과 관측 누락을 고려해야 하며 입력이 완벽하다는 결론은 아니다.

Q2 — SUPPORTED. Peg slot의 absolute는 robot-only 대비 추가 예측 능력을 보이지만 paired 결과는 raw보다 불안정하다. Full Readout의 decoder 일반화는 더 높은 차원과 정규화 영향을 받는다.

Q3 — SUPPORTED/INCONCLUSIVE. Fusion의 absolute와 same-state 반응을 함께 볼 때 readout의 정보가 그대로 행동에 접근 가능한 형태라고 확인하지 못했다. Readout+Robot 통제로 비교했지만 독립 기여율은 계산하지 않는다.

Q4 — SUPPORTED. GRU 현재 hidden의 absolute 및 paired 복원 성능은 위 수치로 측정했다. Robot-history와 조건이 다른 정보이며 dense09D→heldout30003 transfer의 제한이 있다. 단순 위치 예측 가능성을 이동 추적 정확성과 동일시하지 않는다.

Q5 — CONFIRMED. Robot-only와 history만으로도 자연 expert progress에 따른 target을 일부 예측한다. Same-state에서는 두 통제의 예측Δ가0이며 cosine은 undefined다. 자연 상태 성능 차이는 geometry의 독립적 인과 기여가 아니다.

Q6 — CONFIRMED. 동일 frozen decoder와 paired GT로 모든 단계 Δ를 평가했다. 정상 absolute 성능이 paired 이동 복원 성능을 보장하지 않는다는 구체적 사례가 있다.

Q7 — SUPPORTED. Raw 대비 object/readout에서 paired 일반화가 불안정하고 fusion/GRU의 응답도 제한적이다. 절대 metric만으로 단일 손실 단계를 고르지 않는다. 높은 readout dimension, validation4와 한 개 intervention episode가 해석을 제한한다.

Q8 — INCONCLUSIVE. GRU paired 결과와 policy response를 조건별로 보존했지만, hidden이 정확한 PegΔ를 일반적으로 유지하고 head만 잘못 사용한다고 입증하지 못했다. 성공 사례가 있어도 1episode 조건부 사례다.

Q8 사례 — E1 Last MLP, t10 peg_x−5: hidden probe cosine0.952이지만 gain0.207, Δerror3.98mm이고 policy gain은−0.242다. 일부 방향 성분의 복원과 잘못된 action correction이 함께 관찰되지만, 크기 복원까지 양호한 일반적 hidden 표현이라는 뜻은 아니다. E0 Last Ridge t0 peg_y−5는 hidden cosine0.908/gain0.714/Δerror2.18mm이나 policy gain3.957로 과대 반응한다. 이 두 사례는 사후 조건부 예시이며 별도 독립 검증이 아니다.

Q9 — SUPPORTED. E0/E1 차이는 slot/readout/fusion/hidden에서 서로 다르게 나타나며 Best 비교도 표에 있다. E0/E1 parameter/구조가 달라 message passing의 순수 효과로 분리할 수 없다.

Q10 — 다음 단일 후보는 **Encoder/Readout에 입력 geometry summary의 명시적 경로를 추가하는 통제 실험**이다. Raw의 상대적으로 나은 same-state 복원과 learned 단계의 불안정성, Task10B의 약한 action correction을 함께 근거로 삼는다. 이는 Task10C2에서 검증할 후보이며 이번에 architecture를 바꾸거나 BC를 학습하지 않았다. GRU 또는 BC objective가 주원인이라고 확정하거나 loss weight 변경을 바로 권하지 않는다. 동일 BC budget과 frozen diagnosis 기준을 유지한 비교가 필요하다.

## 미실행·제약과 판정

CONFIRMED: frozen feature extraction, episode split, decoder-only fits, heldout interventionΔ 평가, paired mask 보존, weight hash/readback.
SUPPORTED: 관측 Raw geometry에는 useful displacement signal이 있고 learned downstream의 이용 가능한 geometry 표현에 개선 여지가 있다.
INCONCLUSIVE: 모듈별 정보 소실의 증명/인과 기여율, head-only 문제, message-passing 독립 효과, 한 개 주된 원인의 확정.
NOT_TESTED: 원본 Train96의 actor-GT probe, 원본 Validation12 추가 absolute 평가, Test12 probe, fresh simulation/replay, CUDA/MPS, Z 방향 독립 개입, 새 BC/finetuning/architecture 변경. Clean/contamination은 GT source purity가 없는 frame에 임의 부여하지 않는다. Fine-reach/direction-reversal은 이 별도 capture에서 original action 정합이 확인되지 않아 별도 평가하지 않았다.

Sparse5-step 분석과 t0-only GRU pilot은 별도 디렉터리/CSV로 보존하되, dense 주결과와 혼합하지 않는다. 초기 report-hash assert는 report가 full SHA 대신 manifest를 참조하므로 중단되었다. 원래 파일/threshold는 바꾸지 않고 manifest actual hash 검증으로 정정한 새 Logger에서 실행했다. 외부 publishing·git commit·원본 artifact 수정 없음.

## 산출물

[Absolute CSV](absolute_position_probe_metrics.csv), [Episode macro](episode_macro_metrics.csv), [Paired CSV](paired_displacement_metrics.csv), [Information flow](information_flow_delta.csv), [Action comparison](decoder_delta_vs_action_delta.csv), [Verification](verification.json), [Hash index](artifact_index.json). features/는 original Expert unlabelled caches, dense_features/는 labelled diagnostic sequences, dense_probes/는 decoder weight/predictions/selection, paired_features/는 same-state features다.

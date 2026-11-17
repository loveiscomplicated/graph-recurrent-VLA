# Task 10-D — Geometry-conditioned Reach BC

ExperimentLogger UUID `60ba447bf1b541e793a3d54ce68aa719`. 기준 비교 `3788798a719f4ec4a66c224ca42f5d63`. 작성 시각 2026-10-29T18:49:47.703311+00:00.

**실제 수집·캐시·학습·최종 평가 실행 완료.** A/B 모두 실패: 이 데이터 변경만으로 충분하다는 근거 없음.

## 결과 검토와 해석 범위

최종 미학습 대각선 Reach는 A 0/6, B 0/6이다. 최종 위치 오차 평균은 A 120.935mm, B 91.450mm다. 성공 수와 오차를 각각 판단하며, 표본 내 차이를 일반적인 성공률 우위로 확장하지 않는다.

t0의 유효한 동일 Robot 상태 paired intervention 응답 오차는 A 1.630mm, B 1.167mm다. 응답 오차·방향·zero-response 대비 개선을 함께 만족하는 B의 적절한 보정 개선은 확인하지 못했다.

Q5=INCONCLUSIVE, Q6=INCONCLUSIVE, Q7=INCONCLUSIVE, Q9=INCONCLUSIVE. 이 판정은 개선 가설의 근거 수준이며 실행 완료 여부와 구분한다. Reach-only이므로 Grasp/Insertion 및 동적 Recovery는 NOT_TESTED다.

### 노출량·Geometry 신호의 실제 크기

A는 고유 96 Episode/4429 frame, B는 고유 192 Episode/8757 frame을 사용했다. 둘 다 실제 노출 525090 timestep이다. Frame별 반복 횟수는 A 59–420, B 59–60이며 Parent 노출 분포의 total variation은 0.1276다. 따라서 동일 계산 예산은 확인했지만 Parent sampling 분포까지 동일한 단일 요인 실험이라고 해석할 수 없다.

96개 새 same-state 첫 Expert command Δ는 median 0.346mm, 범위 0.090–0.974mm다. 초기 Peg가 양쪽 Graph에 모두 존재한 Pair는 84/96개다. 관측 표면 centroid의 기대 이동 대비 오차 median 3.129mm이며, 이는 actor origin/학습 latent 복원 오차와 다르다. 이 진단으로 Episode를 제외하거나 조건을 조정하지 않았다.

별도 원본 trace 감사 200개에서 최대 Peg 이동 0.002685mm, 최대 회전 0.001594°였다. Robot76에는 Robot Base pose/velocity도 포함되므로 초기 Robot76 exact 검사는 Base 상태 보존을 포함한다.

### 사후 진단 — 학습 조건에서의 초기 보정 fit

모든 최종 평가가 끝난 뒤 선택된 가중치를 고정하고, 학습 캐시의 동일 Robot 초기 상태 96쌍을 hidden=None으로 추론했다. 새 Live rollout·학습·선택 변경은 없으며 사전 등록된 검증 결과와 구분하는 POST_HOC 진단이다. 유효 reference 기준은 기존 0.1mm 그대로다. 응답 오차는 A 1.115mm, B 0.911mm; zero-response 대비 개선은 A -0.727mm, B -0.523mm다.

학습 조건의 보정 fit과 held-out 보정/Reach를 분리해 Q8을 검토할 수 있지만, 이 결과만으로 데이터·표현·최적화·제어 연결 중 단일 원인을 확정하지 않는다. 학습 Pair를 이미 보았으므로 이 진단은 일반화 증거로 쓰지 않는다. [사후 진단 JSON](../../outputs/task10d/remote/posthoc_training_correction.json) · [전체 Pair CSV](../../outputs/task10d/remote/posthoc_training_correction.csv)

### 실제 실패 양상과 평가 속도

| 조건 | 모델 | Approach 도달 | Target pose 도달(보조) | 조기 close | 최종 collision valid |
|---|---|---:|---:|---:|---:|
| nominal | A | 0/6 | 0/6 | 0/6 | 6/6 |
| nominal | B | 0/6 | 0/6 | 0/6 | 6/6 |
| seen_type | A | 0/6 | 0/6 | 0/6 | 5/6 |
| seen_type | B | 0/6 | 0/6 | 0/6 | 5/6 |
| unseen_diagonal | A | 0/6 | 0/6 | 0/6 | 6/6 |
| unseen_diagonal | B | 0/6 | 0/6 | 0/6 | 6/6 |

Target pose 도달은 사후 기술 지표로, 같은 timestep에 등록된 위치/자세 threshold를 만족했는지만 센다. Primary Reach의 ordered approach·opening·충돌 조건을 대체하지 않는다. Collision valid는 최종 정적 물리 검사의 판정이며 trajectory 전체의 contact-free 보장을 뜻하지 않는다. Missing Peg/Box 빈도·최소 거리·자세 오차·사례별 frames/s는 [실패 양상 CSV](../../outputs/task10d/remote/final_failure_diagnostics.csv)에 있다. 이 관찰만으로 표현/최적화/제어 중 단일 원인을 확정하지 않는다.

![전체 Final target pose trace](../../outputs/task10d/remote/figures/all_final_target_pose_traces.png)

### 재사용한 인터페이스와 Source

원격 실행 기준은 `/workspace/task10d_geometry_reach/source`다. 전체 signature·파일 SHA256은 [API 감사](../../outputs/task10d/remote/api_source_audit.json), 학습 시점의 신규 source hash는 [A/B 동결 manifest](../../outputs/task10d/remote/dataset_ab_manifest.json)에 있다. 최종 평가 이후 원래 baseline source와 복사된 inherited source 각 747개, 원래 Graph cache와 기존 training checkpoint도 원래 manifest/receipt SHA로 다시 확인했다: [보존 검사](../../outputs/task10d/remote/final_source_protection.json). 원래 Raw H5 전체의 사후 재해시는 이 검사 범위에 포함하지 않았고 Task10D는 해당 파일에 쓰지 않았다. Decorator가 있는 `offline`의 API 감사 경로는 Torch wrapper를 가리키며, 실제 본문은 아래 `train.py`의 동결 SHA로 식별한다.

| 파일 | 재사용한 함수/클래스 | SHA256 |
|---|---|---|
| `graph/plan_a_bc/cache.py` | cache_episode, CachedDataset.batch, graph_at | `d1054f2b90fbdaa5f65c3914ad6df65eeaa54a6067eeda7433749a1c8792fb43` |
| `graph/plan_a_bc/geometry.py` | CurrentGeometry.from_h5/build, validate_graph | `13e17a0897fc9a5e7a46a414f68052ea1547af9d02a998800641a4db2d842f6d` |
| `graph/plan_a_bc/model.py` | CachedGraphPolicy(E0), normalizer, gradient_report | `f4d2adca1a6222a9e4bd826a44b1e1fd2cabf7681dfcc50ac59a9f0e028433ce` |
| `graph/plan_a_bc/rollout.py` | sensors (live current RGB-D) | `71714e2e6587014bfe83601457a1e55dad5463d4325c8e936985d73440973afa` |
| `graph/plan_a_bc/train.py` | train_update, offline, optimizer, save, hidden_for | `a64de8e38d1eaf0cb80736334d5278249932604125feb2e72fb7c574285bba4a` |
| `graph/integration_baseline/collect_recurrent_expert.py` | capture, append_h5, install_audit (06-B/C) | `8fbe550e98e23ae1359c850fee47a223417244d3f9b4252dc76a4082c9e846ca` |
| `recurrent/expert.py` | ScriptedExpert.reset/act | `3cc854654d16b2462886c64fa12f146dd8a51718913d354f86cd7d6e1c0bcee9` |
| `graph/integration_baseline/geometry09a.py` | physical_state, collision_audit, expert_reference (09-A/10-B) | `73247db15016205b2f1d45c6bcb01eb41bf8c63a6685f8f10c3f8f9ec3ddf1a9` |
| `graph/integration_baseline/geometry09b.py` | memory, restore | `a9c881d794661f1544590deff5306bce0878c340958ab121450f1a2e897da225` |
| `graph/geometry_reach10d/optimizations.py` | DetectionMemo, components | `ea48aaae01b0cd11a1b1c31e36079cb8b90d98f4067dedf0060bf094e61580b9` |
| `graph/geometry_reach10d/parallel_cache.py` | spawned CPU geometry + ordered CUDA vision sink | `cf64f62f90e442bd84c5c48f2601582ebf69c934f60c0d60ee548f887689aac1` |

학습 전의 metadata key 오류와 legacy Full96 fixture 경로 오류는 보존된 engineering failure 기록으로 분리했다. 원본 테스트를 수정/생략하지 않고 원본 fixture에서 실행했으며 신규 Reach 계약에서 실제 11,487 update/525,090 frame을 검증했다. A/B 초기화 후 연구 소스·모델·데이터·선택 규칙을 변경하지 않았다.

실제 검증 CSV의 condition 열과 보고서 merger의 중복 keyword 오류는 원본 연구 소스를 보존한 report-only 어댑터로 처리했다. Reader의 중복 열을 메모리에서 제외한 뒤 같은 condition 값을 병합하며, 원본 CSV의 모든 값이 보존되는지 양쪽 11,487 train행/12 validation행에서 확인했다. 정책 재실행은 없었다: [보고서 호환 검증](../../outputs/task10d/remote/report_compatibility_verification.json).

### 실제 자원과 처리량

실행 GPU는 NVIDIA GeForce RTX 3090, Torch 2.14.1+cu130/CUDA 13.0다. 시작 시 CUDA free/total은 23.30/23.56GiB였고 CPU quota는 1316571 100000(약 13.17 core), logical/affinity는 96/96다.

| 샘플링 구간 | GPU 평균 / p95 % | 장치 VRAM peak MiB | CPU 부모 평균 % | 부모 RSS peak MiB |
|---|---:|---:|---:|---:|
| evaluation_A_final | 48.3 / 100.0 | 11393 | 100.9 | 5240 |
| evaluation_B_final | 45.1 / 100.0 | 10825 | 100.8 | 4579 |
| full_graph_cache | 70.6 / 100.0 | 10229 | 98.8 | 2687 |
| training_A | 37.4 / 100.0 | 12793 | 34.3 | 2145 |
| training_B | 37.7 / 100.0 | 12991 | 34.0 | 2145 |

별도 runtime inventory에서 cgroup RAM limit은 91.27GiB, 당시 cgroup current는 12.15GiB였다. psutil의 host total 251.53GiB와 container limit을 구분한다. RSS 표는 샘플링 부모 프로세스 기준이며 모든 자식의 합계를 뜻하지 않는다.

본 수집의 실제 겹친 window는 1333.9초, 6.516 frames/s, 512.8 episodes/hour다. 완료 Pilot10개는 재사용했다. Cache의 기록된 window는 5702.2초, 1.600 graphs/s, 126.3 episodes/hour이며 engineering pause와 경합을 포함한다.

실제 optimizer compute 기준 updates/s는 A 22.49, B 22.54; Development/저장/검증까지 포함한 wall 기준은 A 7.33, B 7.28다. UUID 생성부터 자동 보고서 생성까지 16012.7초이며 engineering과 운영 중단을 포함한다. 최종 검토·백업 시간은 이 수치 이후다.

수집 2 worker의 1.697배는 startup 이후 capture window 비교다. 두 Episode의 프로세스 startup/전체 시간을 포함하면 24.35초→23.35초(1.043배)다. 장기 본 수집에서는 startup이 amortize되지만 이 차이를 숨겨 전체 가속으로 해석하지 않는다. Cache/renderer/collector speedup은 서로 다른 제한된 benchmark이며 고정 순서·조건당 1회 측정이다. 전체 직렬 baseline 200개를 재실행하지 않았으므로 전체 수집 시간이 몇 배 단축됐다는 인과적 end-to-end 수치는 INCONCLUSIVE다.

### CPU에 유지한 연산과 측정 한계

CPU world backprojection 대비 CUDA primitive는 전송 포함 약 3.35ms→0.42ms로 더 빨랐다. 다만 CPU Geometry 후속 처리로 되돌아오는 작은 연산이며 전체 causal frame 비용 대비 절대 이득이 작아, 정확성이 확인된 CPU Geometry를 2 worker로 overlap했다. 이는 CPU가 더 빠르다는 주장이나 GPU full Geometry 구현의 throughput 검증은 아니다. 실제 6,097점의 Wrist CC는 최적화 CPU 약 0.101초, CUDA dense 약 0.246초였고 정확한 partition에도 CUDA가 느려 기각했다. GPU 거리 연산은 더 빨라도 수치 동등성이 깨져 기각했다.

DINO/SAM2 순서는 Episode별 causal하게 유지했다. 독립 detector batching과 AMP의 end-to-end 이득/Mask 동등성은 검증하지 않아 적용하지 않았다. 원본 train_update의 작은 CPU hidden/metric 동기화는 동결 구현을 유지했고, 대량 Graph packing/transfer는 사전 검증한 CUDA resident 경로로 줄였다. Resource CSV의 GPU util/VRAM은 장치 전체 측정이며, CPU process 값은 해당 부모 프로세스 기준이라 Geometry 자식의 총 CPU 사용률로 해석하지 않는다.

[검토 수치 JSON](../../outputs/task10d/remote/final_review.json) · [노출량 검토](../../outputs/task10d/remote/dataset_review.json) · [원래 자동 생성 보고서](../../outputs/task10d/remote/generated_report_before_review.md)

## 실행 범위

수집 후보 200개, 실제 시도 200개, 승인 200개. Graph 캐시 200개 / 9122 frames. Final Reach A/B 각 18/18개. Final same-state intervention 32개(A/B 합산).

동일한 초기 가중치·구조·Method B·정규화·Optimizer 및 사전 고정 11,487 update를 실행했다.

E0 Edge-aware GNN L2/d64, Node21/Edge4, Readout320+Presence2+Robot76, Projection128, GRU144, sequence32, Hidden Carry/TBPTT, Action7 및 pd_ee_delta_pose를 그대로 사용했다. E1·새 Feature·새 Controller·Skip Connection은 도입하지 않았다.

## 데이터 생성과 고정

원본 Train96 seed와 사전 선택 Parent24(sorted Train96[::4])를 사용한다. 추가 Offset은 X/Y ±5mm의 네 방향이다. 새 위치에서 ScriptedExpert를 새로 초기화하고 실제 env.step으로 reach/descend를 실행했다. close/grasp/lift/insertion Action은 학습에 없다. 매 Action t에 관측 t를 대응시키고 T+1 관측에는 Action을 붙이지 않았다. 원본 HDF5/Graph/Checkpoint는 보존하고 Train96 Reach도 현재 Runtime에서 재수집하여 A/B 공통 원본을 만들었다.

Reach는 elevated approach target(10mm, 0.05rad) 도달 이후 close 직전 target(4mm, 0.03rad)에 도달하고 gripper opening≥65mm, 조기 close 없음, Peg 이동<2mm 및 물리 유효성을 만족해야 한다. 이는 Peg actor origin 30mm나 Grasp/Insertion 성공과 다르다. Target/Offset/GT는 생성·평가 metadata에만 있고 Student는 RGB-D, 예측 Mask, Graph, Robot76만 받는다.

과거 Train96과 현재 Runtime의 reset byte hash는 52/96만 같았다. Pilot의 기존 HDF5 대비 Robot/TCP 관측에는 약 4e−7 수준 차이와 RGB/Depth 차이가 있어 원본 Graph를 섞지 않았다. 현재 새로 수집한 Parent 원본–Offset 비교 96/96에서는 Robot76, Camera calibration, Box/Hole가 정확히 같다. 따라서 A/B는 공통 현재 Runtime을 비교하며 과거 데이터의 byte identity를 PASS로 바꾸지 않는다.

Engineering Pilot 10/10 승인. 첫 Action 변화뿐 아니라 전체 목표 이동·실행 시퀀스를 검증했다. 같은 Robot 초기 상태의 추가 사례는 최대 24 parent×4 offset이며 독립 Robot 상태 96개라는 뜻은 아니다. 이후 trajectory는 Robot 상태가 달라지므로 모든 t를 exact same-state 사례로 세지 않는다.

Graph 캐시는 원본 Plan A cache_episode/CurrentGeometry/validate_graph와 동일 C3+A-SF 계약을 사용한다. SAM2는 Episode/Camera별 reset 및 관측 prefix만 사용한다. DINO Memo는 같은 RGB/Prompt/가중치의 순수 detector 결과만 재사용한다. Gate 전 임의 cap128, 탐지 조건·Filter Threshold·Sampling 의미 변경은 없다. 완료된 Pilot Graph는 원본 raw/hash/geometry identity를 재검증해 파일 그대로 채택했다.

## 계산 비용

새 Offset 2개/87 frames에서 최적화 전후 모든 저장 Graph·Mask·Gate·F5/Pixel Provenance가 정확히 일치했다. 캐시 Benchmark 속도비 1.077배, GPU Renderer 1.132배, 수집 Worker 1.697배. 독립 CPU Geometry와 CUDA Vision의 overlap은 격리된 동일 조건에서 1.560배였다. 이 별도 수치를 곱하여 end-to-end 향상이라 주장하지 않는다.

최초 동시 실행 Benchmark는 GPU 경합이 있어 후보 1.046배로 기록했다. Task10D 캐시 Worker만 126초간 SIGSTOP/SIGCONT로 보존하여 원본/후보를 순서대로 재측정한 결과 73.62초→47.19초였다. 87개 frame의 모든 비시간 배열·Gate가 정확히 같아 2 CPU Geometry worker와 CUDA Vision의 overlap을 채택했다. 완료된 Raw/Graph를 해시 검사해 보존하고 미완료 Episode부터 새 Worker로 이어갔다. 기존 Plan A 서비스는 건드리지 않았다.

RTX3090 CUDA Vision과 CUDA GNN/GRU를 사용한다. GPU Renderer는 CPU Physics의 전체 RGB-D/Action/Robot Trace가 byte-identical한 경우에만 채택했다. GPU PhysX 라이브러리 준비가 약 303초 동안 진행되지 않아 해당 자식 probe를 중단했다. GPU Physics Expert/벡터화 수집은 NOT_TESTED이며 CPU/GPU Physics 동등성을 주장하지 않는다.

Backprojection/Connected Components/거리 primitive의 전송 포함 CUDA 후보는 geometry_kernel_benchmark.json에 있다. Graph/PCA/F5/Source ID의 최종 조립은 검증된 CPU 경로를 유지한다. GPU의 높은 사용률만을 위해 dense quadratic adjacency나 수치가 다른 경로로 바꾸지 않았다. AMP는 적용하지 않고 FP32 모델/기존 geometry 정밀도를 유지했다. SAM2의 설치된 _C postprocessing 확장이 없는 Runtime 동작도 기존 baseline과 동일하게 보존했다.

## A/B 학습 예산

실제 기록된 유효 timestep: A=525090, B=525090. 사전 계획 노출: {'A': 525090, 'B': 525090}. B 추가 데이터 frame 비율=0.494. 길이 매칭에서 제외된 추가 Episode=0. 제외 목록과 생성 실패는 보존했다.

A의 대체 Episode 길이를 B의 추가 Episode와 같게 선택하여 매 update의 mask/timestep 수를 맞춘다. 같은 길이이면 동일 parent 원본을 우선하고, 길이가 다르면 동일 길이의 다른 원본을 사용한다. 따라서 남는 Parent 재가중 차이는 sampling_frequency.csv에 공개하며 Geometry 포함과 완전히 분리된 인과 효과라 단정하지 않는다. 원본 통계는 다시 fit하지 않는다. action_distribution.json/CSV는 고유 데이터와 동결 Sampler의 Action 분포를 기존 Train96 통계와 비교한 진단이며 새 Normalizer로 사용하지 않는다. 학습률은 기존 Full96의 epoch 완료 update 경계에서 동일하게 진행한다. Reach 데이터 cycle 수는 별도로 기록한다. Checkpoint는 공통 Development8의 Method B loss 첫 최솟값으로 선택하며 Final Seed61000..61005는 모델 선택 후에만 접근한다. Offset별 같은 seed를 A/B에 paired 적용한다.

## 최종 Reach 비교

모델 선택을 동결한 뒤 동일 Final18 조건의 ScriptedExpert ceiling도 실제 실행했다: 18/18. 이 데이터는 학습·선택·Student 입력에 쓰지 않았고 Expert 결과에 따라 Student 사례를 제외하지 않았다.

| 조건 | 모델 | 성공/사례 | 최종 위치 오차 mm | 최종 자세 오차 rad |
|---|---|---:|---:|---:|
| nominal | A | 0/6 | 122.656 | 0.367 |
| nominal | B | 0/6 | 90.104 | 0.670 |
| seen_type | A | 0/6 | 120.170 | 0.393 |
| seen_type | B | 0/6 | 97.898 | 0.599 |
| unseen_diagonal | A | 0/6 | 120.935 | 0.433 |
| unseen_diagonal | B | 0/6 | 91.450 | 0.709 |

## 실제 Peg 이동에 대한 Action 보정

같은 Robot 상태·동일 previous hidden·동일 Causal Vision prefix에서 실제 Peg Actor를 이동하고 새 RGB-D로 Graph를 재생성했다. t=0은 초기 조건 변화이며 t=10은 same-history 단일 시점 진단이다. 동적 Recovery Rollout은 아니다. baseline repeat의 Mask/Graph/readout/hidden/action이 정확히 재현되지 않으면 해당 anchor를 분석에서 제외하고 원자료와 FAIL 상태를 남긴다. 09-A/10-B의 controller command Δ, cosine, signed gain, response error, zero-response 계산을 재사용했다. Expert Δ≤0.1mm는 방향 판정에서 제외했다.

| t | 모델 | 유효 reference 수 | 방향 cosine | signed gain | Δ error mm | zero 대비 개선 mm |
|---|---|---:|---:|---:|---:|---:|
| 0 | A | 8 | 0.132 | 1.462 | 1.630 | -1.316 |
| 0 | B | 8 | 0.223 | 1.473 | 1.167 | -0.852 |
| 10 | A | 8 | 0.166 | 0.315 | 0.525 | -0.064 |
| 10 | B | 8 | 0.194 | 0.182 | 0.468 | -0.006 |

## Q1–Q10 판정

- **Q1 CONFIRMED:** 실제 Peg 이동 뒤 새 목표와 연속 Reach Action을 생성했음을 Pilot에서 확인했다. 전체 시도·실패·목표 및 Trajectory는 raw metadata에 있다.
- **Q2 SUPPORTED:** 최대96개의 정확한 같은 초기 Robot/Box 상태 Offset 비교가 추가된다. 독립 parent는24개이며 이후 Frame은 trajectory 분기를 포함한다. 기존 Train96의 관찰 Pair 부족(10-C2)을 GT 기반 다양성 부재의 증명으로 바꾸지 않는다.
- **Q3 CONFIRMED:** 87개 새 Graph의 exact equivalence와 별도 속도 Benchmark를 확인했다. 전체 벽시계 시간은 throughput_summary.json으로 구분하며 가속 배수를 곱하지 않는다.
- **Q4 CONFIRMED:** 동일 update/실제 frame 노출, 초기화, normalizer, codec, scheduler와 선택 예산을 verification.json에서 대조한다.
- **Q5 INCONCLUSIVE:** t0에서 양쪽 모두 유효하고 Robot 상태가 같은 paired reference의 Δ error는 A=1.630mm / B=1.167mm다. B의 응답 오차 개선을 제한적으로 지지한다. 방향·gain을 함께 보며 반복 Offset/timestep은 독립 표본이 아니다.
- **Q6 INCONCLUSIVE:** A/B 모두 실패: 이 데이터 변경만으로 충분하다는 근거 없음
- **Q7 INCONCLUSIVE:** Nominal/새 seed 축 방향/미학습 대각선 성능을 분리한다. 개선과 nominal 손실의 동시 여부는 위 paired 결과로 판단하며 단일 학습 seed에서 일반적 우위를 주장하지 않는다.
- **Q8 INCONCLUSIVE:** Expert target/action 신호·offline fit·same-state Δ·closed-loop Reach를 구분한다. 실패가 데이터 신호 부족, 표현, 최적화, 제어 연결 중 하나만의 원인임을 이 A/B로 확정할 수 없다.
- **Q9 INCONCLUSIVE:** 미학습 seed/대각선의 실제 Reach와 유효 intervention이 함께 개선되는 경우에만 제한된 초기 위치 조건부 제어의 증거가 된다. ±5mm 범위, 6 Final seeds, 학습 seed1개라 memorization 가능성은 남는다.
- **Q10 SUPPORTED:** B의 nominal 유지·미학습 Reach·보정 Δ가 함께 좋으면 다음 Grasp/Insertion 확장을 검토한다. Offline만 좋거나 둘 다 실패하면 이 데이터 신호/목표/제어 연결의 진단을 우선한다. BC 일반의 가능/불가능을 단정하지 않는다.

## Contract tests 및 제한

| 항목 | 판정 |
|---|---|
| original_reset_identity_for_perturbations | PASS |
| historical_original_reset_byte_identity | FAIL |
| peg_offset_robot_box_hole_preservation | PASS |
| physical_collision_workspace | PASS |
| scripted_expert_target_refresh | PASS |
| expert_reach | PASS |
| observation_action_alignment | PASS |
| parent_variant_provenance | PASS |
| episode_state_reset | PASS |
| C3_wrist_peg_gate_main_only_box_node21_edge4_cap128_source_F5 | PASS |
| causal_vision_memory_no_future | PASS |
| empty_missing_objects | PASS |
| cache_hash_readback | PASS |
| original_optimized_graph_equivalence | PASS |
| CUDA_GNN_GRU_gradients | PASS |
| hidden_carry_detach_episode_reset | PASS |
| method_B_codec_normalizers_action7 | PASS |
| save_resume_weight_optimizer_rng | PASS |
| matched_AB_updates_frames | PASS |
| identical_AB_initialization | PASS |
| current_live_RGBD_graph_regeneration | PASS |
| no_expert_action_or_GT_student_leakage | PASS |
| paired_AB_seed_offset | PASS |
| frozen_success_threshold_and_final_access | PASS |
| same_previous_hidden_causal_history | PASS |
| heldout_expert_reference | PASS |
| intervention_baseline_repeat | PASS |
| CUDA_PhysX | NOT_TESTED |
| MPS | NOT_TESTED |

Supervisor의 Task10D 전용 data/experiment/monitor 프로세스는 SSH 세션과 분리되어 있다. 기존 Plan A 프로세스·감시 경로를 변경하지 않았다. 이벤트 Detector는 기존 구현을 재사용하며 ERROR/STALL/COMPLETION만 기록한다. Codex 자동 호출 연결은 NOT_TESTED이므로 완전 무인 복구 운영이라 주장하지 않는다. Discord 실제 상태는 monitor/notification_status.json에 별도로 기록한다. Git commit 및 외부 Publishing은 수행하지 않았다.

CUDA Physics·GPU vectorized Expert·MPS·동적 Recovery·Grasp/Insertion은 NOT_TESTED다. 사용된 Backend의 실제 실행과 기각한 후보를 구분했다. Final seed/offset 반복은 상관된 측정이며 이 수를 독립 표본 수로 간주한 p-value나 일반적 성공률 우위를 제시하지 않는다.

## 산출물

[수집 프로토콜](../../outputs/task10d/remote/collection_protocol.json) · [A/B 동결](../../outputs/task10d/remote/dataset_ab_manifest.json) · [캐시](../../outputs/task10d/remote/graph_cache_manifest.json) · [성능](../../outputs/task10d/remote/throughput_summary.json) · [검증](../../outputs/task10d/remote/verification.json) · [SHA256 Index](../../outputs/task10d/remote/artifact_index.json)

![ab_learning_curves](../../outputs/task10d/remote/figures/ab_learning_curves.png)

![ab_reach_success](../../outputs/task10d/remote/figures/ab_reach_success.png)

![expert_action_correction](../../outputs/task10d/remote/figures/expert_action_correction.png)

![expert_reach_trajectories](../../outputs/task10d/remote/figures/expert_reach_trajectories.png)

![gpu_cpu_throughput](../../outputs/task10d/remote/figures/gpu_cpu_throughput.png)

![peg_displacement_action_response](../../outputs/task10d/remote/figures/peg_displacement_action_response.png)

![tcp_target_error_vs_step](../../outputs/task10d/remote/figures/tcp_target_error_vs_step.png)

# Task 10-I Stage4A-2 — SAM2 Mask 기반 후보 검증과 초기 정체 유지 평가

완료일: **2026-11-03 KST**. 실험 Identity는 사전 등록한 `20261102_v1`을 유지했다.

**결론: Mask 윤곽과 외부를 어둡게 한 Crop을 Qwen에 제공하는 조건 C는 개선을 보이지 않았다. 초기 Mask를 고정해 SAM2로 추적하는 경로는 후기 재분류보다 유효한 관측을 더 많이 유지했지만, 잘못된 초기화와 Wrist 실패가 남았다. 학습 입력 준비 또는 Production 채택 판정은 하지 않는다.**

- 전체60장의 두 물체 동시 초기 Mask 성공: A/B **27/58=46.6%**, C **24/58=41.4%**. 성공 기준은 각 Object Mask IoU≥0.8이다.
- 같은8 Parent·16개 t80 이미지에서 현재 프레임 검사 A는 두 물체 동시 **2/16=12.5%**, t0 초기화 후 SAM2 추적은 **11/16=68.8%**였다. 입력 이력과 Codec이 다른 운영 경로 비교이며 Memory 하나만의 인과 효과로 해석하지 않는다.
- 전체 추적 구간의 두 물체 동시 성공은 **623/801=77.8%**. 한 Main Fixture의 Robot 오초기화는81프레임 내내 지속됐다.
- 세 조건은 추적 대상16개 Sequence에서 초기 후보 ID와 최종 Mask SHA가 모두 같았다. 세 조건의 동일한 추적 성능을 Mask 입력의 이득이나 독립 표본3배로 세지 않는다.

## 1. 질문·자료·범위

이번 실험은 (1) Box 안에 섞인 Actor를 SAM2 Mask로 구분해 보여주면 Qwen 검증이 개선되는지, (2) 후기 외형이 달라진 물체를 매번 분류하는 대신 초기화 후 추적하면 관측을 유지할 수 있는지 측정했다. 기존 외형 Prompt 평가의15개 Train Parent, native512 RGB60장, 보존 DINO 후보986개를 그대로 사용했다. 이 자료는 이미 연구에 사용됐으며 독립적인 새 Parent 일반화 평가가 아니다.

각 Parent는 F0=t0 및 F1=t80을 사용하고0134만 기존 규칙의t78이다. Main/base_camera와 Wrist/hand_camera는 개별 입력이다. 선택·추론·평가에 결과 기반 Parent 교체가 없다. GT 가시성이16px 이상일 때 주 평가,1–15px은TINY_VISIBLE,0px은ABSENT로 분리했다. Peg60개/Fixture58개가 정적 주 평가 분모다. Null·Empty·미초기화는 Visible 성공률에서 실패다.

Hole 위치 추정기, 재검출·재초기화 정책, Graph 생성, Controller 학습 및 Closed-loop 성능은 평가하지 않았다.

## 2. 모델·Prompt·입력 계약

| 항목 | 실행값 |
|---|---|
| Qwen | Qwen/Qwen3.5-9B |
| Revision | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| 정밀도 | 전체 BF16, 양자화 없음, SDPA, cuda:0, Weight Offload 없음 |
| Generation | Batch1, Greedy, enable_thinking=False, max_new_tokens16 |
| Processor | 기존 Source, min/max pixels262144 |
| SAM2 | 기존 SAM2.1 Hiera Small Weight, FP32 Weight·BF16 autocast |
| SAM2 설정 | configs/sam2.1/sam2.1_hiera_s.yaml, 설치 기본 후처리, Logit>0, multimask_output=False |
| 장비 | 기존 Vast.ai58576054 / RTX3090 24GB |

SAM2 Checkpoint SHA256: `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`. 공통 B/C Prompt SHA256: `e8d0e4180cd1424cfa6947d4cfa6720a74a0ba265470f727177151ae23ac3656`.

| 조건 | Qwen 입력 |
|---|---|
| A | 과거 외형 Prompt의 동결 응답을 재사용 |
| B | 전체 RGB + 기존 Box 표시 + 기존10% Padding Crop |
| C | 전체 RGB + 예측 Mask 윤곽 + 같은 Crop에서 Mask 외부 RGB만25%로 처리 |

B/C의 텍스트는 완전히 같다. 기존 Prompt 첫 두 문장만 아래로 교체했고 PEG/BOX/ROBOT의 색상·형태 정의 및 나머지 문장은 유지했다.

```text
Identify the candidate region marked in image 2.
Image 1 shows the complete scene. Image 3 is a crop of the same candidate region.
```

C의 Mask 내부 RGB는 그대로다.986개 입력의 Crop·Mask 대응 및 외부25% 처리를 로컬에서 독립 확인했다. 전체 RGB는 가공하지 않았다. Mask는 DINO 후보 Box만으로 생성했으며 GT Mask로 대체하거나 GT를 보고 고르지 않았다. 후보별 SAM2 Prompt는 이전 후보의 Mask/Logit/Memory를 사용하지 않고 이미지 Embedding만 재사용했다.986개 모두 비어 있지 않은 Mask를 생성했다.

조건별 선택은 원래 Query와 같은 Label로 승인된 후보 중 원래 DINO Score1위다. 동점은 Rank, ID 순이다. 새 점수 결합·Threshold·GT 기반 순위 보정은 없다. Qwen은 Query/Score·Actor ID·Pose·GT Mask를 입력받지 않았다.

Qwen Load1회, 신규 응답1,972개, 엄격한Label1,972/1,972, 첫Forward Cache 길이0도1,972/1,972이다. 새 DINO Forward0회, Sampling/Retry/Prompt Sweep0회다. B/C와 보존A의 비교에서만 SAM2 입력이 허용되며, 과거 DINO-free 실험의 입력 계약과 혼동하지 않는다.

모든 정적·추적 원시 예측을 동결한 다음에 성능 GT를 열었다. 복원용 Physics·불투명 파일Hash 읽기와 의미적 정확도 평가는 구분한다. 원시 응답 SHA256은 `9f43abdb34cfbc2ae9509843d7881bc0fe3fe9fed74dc2086bd62dd08dd4eaab`이며 채점 이후에도 같다.

## 3. 전체60장 정적 초기 Mask 결과

| 조건 | Peg IoU≥.8 | Fixture IoU≥.8 | 두 물체 동시 | Peg Wrong-object | Fixture Wrong-object |
|---|---:|---:|---:|---:|---:|
| A | 33/60 (55.0%) | 48/58 (82.8%) | 27/58 (46.6%) | 1/44 (2.3%) | 4/57 (7.0%) |
| B | 33/60 (55.0%) | 45/58 (77.6%) | 27/58 (46.6%) | 1/46 (2.2%) | 7/58 (12.1%) |
| C | 31/60 (51.7%) | 43/58 (74.1%) | 24/58 (41.4%) | 1/43 (2.3%) | 7/57 (12.3%) |

Wrong-object는 다른 Actor가 예측 Mask의 50% 이상인 경우다. 분모는 Tiny를 제외한 비어 있지 않은 예측이며 Visible Recall 분모와 다르다. 낮은 IoU, 배경 오염, 다른 Actor 선택을 구분한다.

| 조건 | Peg IoU Mean / Median | Fixture IoU Mean / Median | Parent-macro 동시 성공 |
|---|---:|---:|---:|
| A | 0.633 / 0.977 | 0.851 / 0.971 | 46.1% |
| B | 0.649 / 0.977 | 0.791 / 0.965 | 46.1% |
| C | 0.610 / 0.875 | 0.781 / 0.965 | 40.6% |

Visible Object에 대한IoU Mean은 Null/Empty를0으로 포함한다. 각 분포의p05/p95, Pixel Precision/Recall, 오염 구성, 후보별 정답Mask 유지/잘못된Mask 승인 분모는 원시 지표에 보존했다. Median만 높다는 이유로 누락을 제외한 성능으로 판정하지 않는다.

**C−B의 두 물체 동시 성공 Parent-macro 차이: -5.56 pp, 95% CI [-13.33, +1.67] pp, Parent n=15.** Point Estimate는 감소했고 CI는0을 포함한다. 이를 유의한 개선이나 확정적 전반 퇴행으로 선언하지 않는다. C−A의Fixture 성공 차이는 -8.33 pp, 95% CI [-15.00, -3.33] pp, Parent n=15다. 이는 Mask 표현과 지칭 문구가 함께 다른 비교다.

| 조건 | Main 동시 | Wrist 동시 | F0 동시 | F1 동시 |
|---|---:|---:|---:|---:|
| A | 12/30 (40.0%) | 15/28 (53.6%) | 22/28 (78.6%) | 5/30 (16.7%) |
| B | 12/30 (40.0%) | 15/28 (53.6%) | 22/28 (78.6%) | 5/30 (16.7%) |
| C | 13/30 (43.3%) | 11/28 (39.3%) | 22/28 (78.6%) | 2/30 (6.7%) |

**관측:** Mask 조건은 이번 표현 방식으로 후기의 정답 거부·오승인을 해결하지 못했다. **미확정:** 검증 전에 분할하는 모든 방식이 무효하다는 결론은 아니다. Contour·배경 Dimming·같은Crop 범위를 묶어 적용했으므로 각 요소의 효과와 Qwen 내부 원인을 분리하지 않았다.

## 4. 초기화 후 추적 — 전체 실패 포함

Parent0023/0720/0621/0157/0686/0921/0044/0844를 사전 Hash 규칙으로 고정했다. Main/Wrist 각각t0–t80의81시점, 총1,296장을 저장된 상태에서 native512로 복원했다.8개 환경 생성·8개 명시적Seeded Reset을 기록했고 Physics Step·새Action·Rollout0회다. 상태·TCP/Hole·Calibration, 과거512 t0/t80 RGB 일치를 확인했다. Solver의 동역학 Replay라고 주장하지 않는다.

SAM2 Image Predictor가 생성한 t0 예측 Mask를 Video Predictor의add_new_mask에 입력했다. 실제Video t0 출력부터 평가했다. 조건별 독립State48개를 실행해 Camera-Frame3,888개, Object Mask7,776개를 저장했다. Video Tensor만CPU에 두고 Weight·Tracker State는GPU에 두었다. 이후 재검출·재초기화·미래조건부Prompt·GT 보정은 없다.

**초기 후보 ID가16/16 Sequence에서 세 조건 모두 같고, 최종 Mask 파일 SHA도16/16에서 같았다.** 따라서 아래는 공통 결과다. Parent는8개이며 같은 결과를3배의 독립 표본으로 세지 않는다.

| 범위 | Peg IoU≥.8 | Fixture IoU≥.8 | 두 물체 동시 |
|---|---:|---:|---:|
| 전체 | 1093/1296 (84.3%) | 682/801 (85.1%) | 623/801 (77.8%) |
| Main | 629/648 (97.1%) | 520/627 (82.9%) | 501/627 (79.9%) |
| Wrist | 464/648 (71.6%) | 162/174 (93.1%) | 122/174 (70.1%) |

전체 두 물체 동시 성공의 Parent-macro는78.1%이며,8 Parent Bootstrap10,000회/Seed42의 절대95% CI는57.6–91.4%다. C−B의 차이CI가[0,0]인 것은 이 표본의 결과가 동일해서이지 일반적인 동등성 검정을 통과해서가 아니다. Fixture가보이지 않는시점은 동시 성공 분모에서 빠지므로, 분모801을 전체Camera-Frame1,296과 함께 보고한다.

실제Video t0 IoU≥.8인 Object에 한정한 t1이후 유지 성공은 Peg967/1,040=93.0%, Fixture668/705=94.8%다. 좋은 초기화로 조건화한 값이므로 전체 성공률을 대신하지 않는다. 이 조건 자체도GT로 사후 분류했으며 실행 시GT로 성공한 초기화만 고르지 않았다.

### 같은 t80 이미지 16장의 운영 경로 비교 — 사후 대응 분석

| 경로 | Peg | Fixture | 두 물체 동시 |
|---|---:|---:|---:|
| t80 현재 RGB에서 기존Qwen 검사 A로 선택·SAM2 초기화 | 2/16 (12.5%) | 12/16 (75.0%) | 2/16 (12.5%) |
| t0 예측 초기화 뒤 SAM2만 유지한 t80 | 12/16 (75.0%) | 14/16 (87.5%) | 11/16 (68.8%) |

동일8 Parent·Camera·시점이며 양끝의Actor GT Mask96개도 재확인했다. 과거15 Parent 전체의F1과 이16장을 직접 섞어 비교하지 않는다. Tracker 입력은JPEG100/Subsampling0, 정적 입력은PNG이며 초기화 이력과 Video 경로도 다르다. 따라서 이 비교는 실제 두 운영 방식의 차이를 보여주지만 순수Memory 단독Ablation이나 Closed-loop 성공은 아니다. Codec RGB 절대오차 평균은0–255범위에서0.366, Pixel 최대5이며 원본PNG를 보존했다.

## 5. 남은 실패의 위치와 해석

- **0157 / Main / Fixture:** t0 Mask IoU0.428. 다른Actor가우세한 잘못된Mask를81프레임 내내 유지했다. 전체 추적Fixture Wrong-object81/785=10.3%의81건이 이Sequence다. Main만 보면81/618=13.1%다. 추적이 정체를 유지한다는 사실은 초기정체가맞다는 보장이 아니다.
- **0844 / Wrist / Peg:** t0 IoU0.790으로엄격한기준미달이었고, IoU<.5인 연속실패구간이최대63시점이었다. 초기Mask의부분성·후기외형·추적오류중어느요인의인과효과인지 이번에분리하지않았다.
- **0044 / Wrist / Peg:** t0 IoU0.479, 다른Actor우세2프레임이있었다. 이후부분적으로회복돼 초기기준미달을전구간완전실패로취급하지않는다.
- Peg의IoU<.8 실패203개중130개가초기IoU<.8인3개Camera/Object Sequence에서,73개가초기IoU≥.8인13개Sequence에서발생했다. Fixture 실패119개는각각82개와37개다. 이것은조건별실패집계이며 초기Mask를고치면모두사라진다는인과주장은아니다.

Mask 외부를 어둡게 해도 Robot 초기화를 완전히 막지 못했다. 확인 가능한 다음 병목은 초기Mask의Actor 정체·범위 확인과 Wrist에서의 유지/복구다. 이번에는 Robot 기하 제거·Re-grounding·새Verifier·추가학습을 실행하지 않았다.

## 6. Hole 가시성과 평가 준비

예측을 동결한 뒤 RGB 전체와 진단용 확대 영역을 직접 검토했다. 단일 Assistant의 정성적 판독이며, 독립 주석자 간 합의나 검출 정확도 GT가 아니다. VISIBLE 26개, NOT_VISUALLY_CONFIRMED 33개, OUT_OF_VIEW 1개다. VISIBLE은 부분 입구를 식별한 경우도 포함한다. 미확인은 물리적 Hole 부재를 뜻하지 않는다.

**Hole 기준점 투영이 영상 밖이어도 입구의 일부는 보일 수 있었다.** RGB에서 입구를 확인한 26개 중 10개에서 기준점은 영상 밖이었다. 설치된 Task Source에서 `box_hole_pose`는 `box.pose × box_hole_offsets`이며 로컬 x 오프셋은 0이고 y/z에 랜덤 중심을 사용한다. 이 Task 내부 기준점과 가시 입구면의 중심을 동일하게 취급하면 안 된다. Calibration 불일치라고 단정하지 않으며, 향후 입구면과 가시 경계의 GT 정의가 필요하다.

자동생성원본Audit의OUT_OF_VIEW는기준점투영에근거한임시값이다. 실제영상가시성은 [hole_readiness_reviewed.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/hole_readiness_reviewed.json)의판독값을따른다. 원본은수정하지않고보존했다. Fixture는별도Actor지만Hole은아니므로Fixture Mask를Hole GT Mask로쓰지않았다. Hole 추정기·Hole 위치오차·삽입방향정확도는평가하지않았다.

## 7. 실제 비용과 실행 규모

| 항목 | 측정 |
|---|---|
| Qwen Load | 7.563s, 1회 |
| Qwen B Warm | n986, Mean611.8 / Median602.3 / p95644.7ms |
| Qwen C Warm | n985, Mean608.3 / Median600.9 / p95643.2ms |
| Qwen Peak allocated | 17.913GiB |
| Qwen CPU Process Peak RSS | 19,240,608KiB; CPU Weight Offload는없음 |
| SAM2 정적 | Image Encoding60개 Mean34.7ms, 후보분할·입력제작·I/O986개 Mean114.0ms |
| SAM2 추적 | Frame3,888개 Mean48.6 / Median49.8 / p9553.4ms, 초기화·Load비용별도 |
| SAM2 Peak allocated | 정적0.351 / 추적0.710GiB |
| Load/Forward예산 | Qwen1Load·1,972요청, SAM2 2Load·후보986, VideoState48 |

정적SAM2와Qwen, Video SAM2는순서대로실행했다. 동시상주·두카메라동시20Hz·Controller포함Latency는측정하지않았다. 추적Frame시간은단일Camera의최대두Object 기준이다. Qwen전체후보검사는조건당60장에약10초/이미지규모이며실제후보수가다르다. 일부아티팩트전송이Qwen실행과겹쳤으므로Latency는이운영조건의실측이며격리된성능Benchmark로주장하지않는다.

## 8. 보존·독립 검증·산출물

기존보호파일4758개의SHA, 설치SAM2 Source·Weight·환경Package·동결Source가전후동일했다. 원시응답과예측의채점전후SHA도같다. 기존PAUSED_SCOPE_REVIEW/Stop Stub을유지했다. Production·기존Dataset/Checkpoint·Threshold·Tracking Algorithm·Graph/Controller·Commit·게시변경은없다.

공통ExperimentLogger를재사용했고W&B는disabled다. 로컬·원격의계약테스트각3개가통과했다. CUDA모델과정적상태복원은원격Linux에서실행했다. macOS는CPU파일검증만실행했다.

별도구현검증은원격CPU에서Artifact6,981개SHA, 요청1,972개, 입력이미지Hash5,916개, Mask표현986개, 선택360개, 초기Mask360개와추적Mask7,776개의IoU/Pixel구성/Wrong-object/성공판정을재계산해PASS했다. 기존A 선택120개도과거동결결과와같았다. 로컬에서는수신파일SHA와986개Crop, Cache/형식1,972개및정적Mask360개를별도검증했다. 연속원본PNG/JPEG·GT·추적Mask전체는원격Output에보존하며로컬복사여부는별도검증Receipt에명시한다.

주요자료:
- [protocol.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/protocol.json)
- [prompt_identity.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/prompt_identity.json)
- [model_identity.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/model_identity.json)
- [sam2_identity.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/sam2_identity.json)
- [input_manifest.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/input_manifest.json)
- [tracking_manifest.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/tracking_manifest.json)
- [sam2_initialization_metrics.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/sam2_initialization_metrics.json)
- [sam2_initialization_cases.csv](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/sam2_initialization_cases.csv)
- [candidate_evaluation.csv](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/candidate_evaluation.csv)
- [tracking_metrics.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/tracking_metrics.json)
- [tracking_cases.csv](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/tracking_cases.csv)
- [tracking_sequences.csv](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/tracking_sequences.csv)
- [tracking_identity_audit.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/tracking_identity_audit.json)
- [matched_t80_comparison.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/matched_t80_comparison.json)
- [endpoint_GT_equivalence.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/endpoint_GT_equivalence.json)
- [hole_readiness_reviewed.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/hole_readiness_reviewed.json)
- [latency_vram_summary.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/latency_vram_summary.json)
- [verification.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/verification.json)
- [independent_verification.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/independent_verification.json)
- [artifact_index.json](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/artifact_index.json)
- [comparison.png](../../outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1/comparison.png)

## 9. 다음 판단에 남은 제한

이번 Mask 표현 C는 채택 근거가 없다. 초기 Object Discovery 뒤 정체를 SAM2로 유지하는 방향에는 관측 근거가 생겼지만, 두 물체 동시 77.8%, 동일 t80 68.8%, Fixture Wrong-object 10.3%는 운영 목표를 충족하지 못한다. 특히 Parent0157의 12/108=11.1% 동시 성공처럼 집중 실패가 평균에 가려진다.

Parent15/추적8의알려진분포이며희귀오인식, 새외형·배경, 장기추적, 능동시점변경, 복구정책, Hole정확도, Graph기하품질과실제학습/Closed-loop는미검증이다. 실패를확인했다고이를정답데이터로누적하거나Graph학습을재개하지않았다. 추가실험은자동시작하지않고종료한다.

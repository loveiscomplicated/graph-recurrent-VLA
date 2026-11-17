# Task10-F1 — GT-supervised Graph Encoder Pretraining 최종 연구 보고서

## 실행 완료와 연구 판정

**GNN/State Head 사전학습과 독립 평가를 완료했다. 공간 표현은 부분적으로 개선됐지만, 정확한 작은 물체 이동 복원과 GT-state 정책 입력의 충분한 정확도는 확보하지 못했다.**

Final12 Parent/1,350 frame에서 Peg 평균 위치 오차15.291mm, Hole51.071mm, Peg-tip–Hole57.170mm였다. Robot-only와 Raw Geometry Ridge보다 평균 absolute 오차는 낮았다. 그러나 미학습±5/10mm 실제 물체 이동의 Δ vector 오차는 Peg6.895mm, Box/Hole6.854mm이며, signed gain은 약0.5였다. 5mm 조건에서는 Peg가 zero-response와 거의 같고 Box는 더 나빴다. Absolute loss 감소를 정확한 geometry response나 Closed-loop 성공으로 해석하지 않는다.

하나의 checkpoint에서 A의 estimated State38과 B의 Readout320/Presence2/Robot76를 추출할 수 있고, GT 경로 접근을 차단한 검증도 통과했다. 다만 수치적38D 호환과 실제 GT-state 정확도·관측 가능성은 구별한다. **엄격한 A 입력의 충분성은 BLOCKED/INCONCLUSIVE이며, B의 후속 제어 성능은 NOT_TESTED다.** GRU·Action Head·BC·Closed-loop 학습, Git commit, 외부 publishing은 수행하지 않았다.

## 실행 Identity와 보존

| 역할 | Identity |
|---|---|
| F1 비교/준비 UUID | dcbafd9fd31847a3a83fef6830f87834 |
| 실제 GNN 학습 UUID | a803354bff8f414492456af4c0b43bce |
| Robot-only UUID | 28792bec609f41aaadcdc85820eadb38 |
| 최종 독립 평가 UUID | fba44dec9d214064939147b652c69bd7 |
| 실행 위치 | Vast.ai57987772 / NVIDIA RTX3090 |
| 실행 root | /workspace/task10f1_gt_pretrain_20261030 |
| 원본 제공 인스턴스 | 57640710; 원래 수집/학습 작업은 재시작·변경하지 않음 |
| 선택된 GNN checkpoint SHA256 | 781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6 |
| 학습 Source manifest SHA256 | b40426043ad72aecb89c291f887f24be5c947823f270f80ed99b31bcde1cb740 |

기존 Parent는 Plan A BC `3788798a719f4ec4a66c224ca42f5d63`, Task10-D `60ba447bf1b541e793a3d54ce68aa719`, Task10-E `771c27f66f6d4f27850a41a87478dc16`이다. 기존 F1 구현·캐시를 이어갔으며 새 아키텍처 검색이나 BC weight 이식은 없었다. 학습 Source와 평가 수정 Source는 별도 archive로 보존한다. 기존 완료 F1 artifact455개의 SHA도 유지됐고, Task10-F 원본 데이터/config/audit1,187개와 source620개는 학습 후 재검사에서도 동일했다. [final_original_task10f_preservation.json](../../outputs/task10f1/final_run_evidence/final_original_task10f_preservation.json)

## GT38D 원본 계약과 관측 가능성

원본 Recurrent `recurrent.features.build_features` SHA256은 `80de6fd742be4e7c9a3477cde0419e803b72253c09d1227fc0fec23b965b0761`다. Task10-E에서 복원한 encoder/normalizer 및 Clean checkpoint 계약을 확인했으며, GRU 가중치는 사용하지 않았다.

| Index | 필드 | 추론 source |
|---|---|---|
| 0–6 | TCP world xyz + quaternion | 측정 Robot76 |
| 7–12 | 좌/우 fingertip − TCP | 측정 finger link + URDF pad offset |
| 13–14 | 두 finger joint position | 측정 Robot76 |
| 15–17 | Peg − TCP | GNN estimate3D |
| 18–21 | Peg world quaternion | GNN conditional estimate4D; symmetry 한계 |
| 22–24 | Peg endpoint − Hole | 예측 pose/half-size로 계산; 별도 target 아님 |
| 25–27 | Hole − TCP | GNN estimate3D; Box centroid와 구분 |
| 28–31 | Hole world quaternion | GNN conditional estimate4D |
| 32–34 | Peg half-size | GNN conditional estimate3D |
| 35 | Hole radius | GNN conditional estimate1D |
| 36–37 | freshness / age | Clean [1,0] |

위 위치는 metres이며 TCP-relative vector는 **world 위치의 차**다. Graph XYZ는 TCP axes여서 inverse TCP rotation과 혼동하지 않는다. Quaternion은 wxyz, unit normalize, w≥0 canonical sign, q/−q 동치다. Endpoint는 `peg_p + R(peg_q)*[peg_half_x,0,0] − hole_p`다. Finger pad offset은 측정 link 좌표에서 `[0,+0.00758,0.04525]`, `[0,−0.00758,0.04525]`m다.

측정15D는 예측하지 않는다. Head는 독립 object18D를 예측하고 endpoint3D와 Clean2D를 조립한다. Peg half-size/radius는 episode별 randomized GT supervision이며 실제 추론에 GT나 환경별 상수로 넣지 않는다. Train-only target mean은 학습용 prior/normalization이고 per-frame GT 주입이 아니다. 원래 GT-GRU normalization `((x−obs_mean)/obs_std)*obs_keep`와 학습 loss의 물리 scale은 별개다.

**UNSUPPORTED/불확실:** 부분 surface만으로 arbitrary symmetric pose의 roll/end polarity, 완전히 숨은 물체 상태, 보이지 않는 rim/extent/shape를 항상 유일하게 식별할 수 없다. C 항목의 출력은 분포에 의존한 추정치다. API의 identifiability metadata는 object fields를 무조건 관측 가능하다고 인증하지 않으며38D나 Graph 입력에 추가하지 않는다.

[차원별 계약](../../outputs/task10f1/final_run_evidence/gt38_feature_contract.json) · [관측 가능성 감사](../../outputs/task10f1/final_run_evidence/target_observability_audit.json)

## 데이터 누락의 실제 해결과 runtime

기존 Task10-F212개에는 매 timestep actor GT와 영구 simulator/controller/RNG snapshot이 없었다. 기존 checkpoint는 모델/optimizer/RNG로 simulator snapshot을 대신하지 못했다. 따라서 원래212개의 object GT를 centroid나 seed-only replay로 복구했다고 주장하지 않는다. 원본은 보존하고, **별도212개를 직접 GT 동기 수집**했다. 새 수집에서 RGB-D/Robot76/GT를 같은 pre-action timestep에 기록하고 physical/GNN mask를 분리했다. Action mask OFF3,470 frame도 물리·GT 승인 후 GNN supervision에 사용했다. 실패한 pilot은 승인 데이터에 포함하지 않았다.

새 보완 자료의 portable snapshot readback 및 같은 상태의2×5-step RGB-D/Robot/GT replay 증거를 보존했다. 이것은 새 수집의 재현성 검사이며 원래212개 GT 복원을 뜻하지 않는다. 기존 cache config SHA `095ddca684bfe9819862fa88f31bd8e0b8dc489f8502f30b4ab170ac78c0c45c`와 이후 training config SHA `bb168f4ca1832ed8087923dfb40778da2c256dd06a2cf8406dd005c19f466fe5`의 차이를 lineage로 분리하고, cache 당시 config를 byte-exact 재구성했다. 원래 파일을 고쳐 hash를 맞추지 않았다.

Collector Torch2.8.0+cu128/NumPy2.1.2와 F1 Torch2.14.1+cu130/NumPy2.5.4가 달랐다. 사전 지정6 Episode/328 frame의 whole causal sequence 비교에서2 PASS/4 FAIL이 나왔다. 특정 라이브러리를 단독 원인으로 단정하지 않았고 허용오차를 늘리지 않았다. **새212개 전체를 F1 native runtime에서 재캐시**하고 source576의 graph tensor는 제외했다. Collector runtime과 graph runtime provenance는 각각 유지한다. Python3.12.15, ManiSkill3.0.1/SAPIEN3.0.3/Gymnasium1.4.0, CPU PhysX + CUDA renderer/vision을 사용했다. GPU physics는 실행하지 않았다.

Native 재캐시는08:10–09:48 UTC 약98분 동안 기존 프로세스를 유지했다. 완료 캐시를 재생성하지 않았다. Episode ID가 FPS RNG에 들어가므로 logical capture ID와 graph sampling ID를 분리하고 원래 sampling namespace를 보존했다. C3+A-SF, Main+Wrist Peg, Main-only Box, self-filter, Wrist gate, cap128, Node21/Edge4, current-only/persistent OFF, temporal extras7개0을 유지했다. 새로운 Corrective 입력4 frame에서 original/optimized Graph·Gate·Detection Memo가 정확히 일치했다. Empty object graph도 검증했다.

사용자가 중지했던57640710을 다시 켠 뒤 누락93개 physical audit/trace를 확보했다. 새212개 전부의 초기 및11,375 action timestep 물리 검사, SHA, source/config/runtime, parent/branch, GT38 alignment를 확인했다. 두 독립 과정에서2,650 file/990,405,874 bytes를 검증한 후에만 DATA_READY를 게시했다. Marker SHA는 `e50217dfc28ce59651c188042ad43d8db385ebd359db21bec6bff3f7e6dbab94`다. F1 전송 후 같은 전체 파일 검증을 반복하고 최종 dataset/budget을 동결했다. 기존 자동 Codex callback은0회이며 현재 세션이 직접 이어갔다.

[최종 release](../../outputs/task10f1/final_run_evidence/data_ready_release.json) · [최종 dataset](../../outputs/task10f1/final_run_evidence/final_dataset_manifest.json) · [GT sidecar manifest](../../outputs/task10f1/final_run_evidence/gt_label_sidecar_manifest.json)

## 실제 분포와 Split

| Split | 일반 Episode | 독립 Parent | pre-action frame | valid 실제 물체 Pair |
|---|---|---|---|---|
| Train | 272 | 96 | 21,121 | 192 |
| Validation | 48 | 16 | 3,704 | 80 |
| 신규 Final | 12 | 12 | 1,350 | 476 |

Full-task120개/14,800 frame와 새 Corrective212개/11,375 frame다. Corrective는 Train176/Validation36개이며 nominal reach6,569, disturbed robot1,510, recovery3,296 frame다. Full-task는 Reach/Descend/Close/Lift/Alignment/Insertion을 포함한다. Corrective만으로 Full-task 전체 분포를 확보했다고 주장하지 않는다. 같은 Parent의 prefix/branch는 모두 같은 split이며272개 Train Episode를272개 독립 Parent로 해석하지 않는다. Final91000–91011은 미학습 Parent다. 기존 Test/과거 Final 및 Task10-B 평가 개입22개는 fitting에 사용하지 않았다.

| 실제 phase | Train frame | Validation frame | Final frame |
|---|---|---|---|
| disturbed_robot | 1087 | 423 | 0 |
| expert_recovery | 2232 | 1064 | 0 |
| full_task_align | 3979 | 477 | 410 |
| full_task_close | 768 | 96 | 96 |
| full_task_descend | 1273 | 158 | 160 |
| full_task_insert | 1717 | 221 | 206 |
| full_task_lift | 1080 | 137 | 137 |
| full_task_reach | 3156 | 388 | 341 |
| nominal_reach | 5829 | 740 | 0 |

같은 Robot/Camera/causal prefix에서 실제 Peg/Box를 이동한 별도 Pair는 총864 후보 중748개가 물리 승인됐다. Train은±7.5mm axis, Validation은 별도6mm diagonal, Final은±5/10mm axis다. Final576 후보 중476 승인만 결과 해석에 쓴다. Train의192 pair만 Δ loss에 사용했다. TCP 교란112 branch의 temporal before/after 비교는 같은-robot 물체 이동과 별도로 집계하며, Held-out recovery는2 Parent/32 branch라는 제한을 가진다.

Nominal Full-task는 실제 ScriptedExpert/controller 명령, finite state, GT 정합성 및 insertion 도달을 확인한 자료다. Corrective에 적용한 SafetyV2의 매 physics substep contact audit를 Full-task 전체에도 수행했다고 주장하지 않는다. 이 범위 차이는 verification의 NOT_TESTED 항목에 남긴다. Raw RGB-D/GT/Graph 원자료는 remote root의 raw/cache/pairs 및 sealed import에 보존하고 manifest로 연결했다.

## 모델·학습·Checkpoint 선택

E0 Edge-aware GNN L2/d64와 5×64 slot(TCP/Left/Right/Peg/Box)을 그대로 재사용했다. Head는 `[Readout320, Presence2, measured Robot76]`398D →256→128→18이며, 총193,906 parameter다. GNN/Head는 함께 학습했다. 초기화는 새 seed42이며 BC weight를 사용하지 않았다. GRU/Action module이나 action loss는 없다.

Absolute loss는 positions6D의 MSE/.1m, sign-invariant quaternion loss `mean(1−dot²)`, shape4D의 MSE/.02m 합이다. 독립 endpoint target을 중복 추가하지 않는다. Absolute는64 random eligible frame와16 pair의 양 endpoint32 frame 전체에 적용하고, Δ position6D/.01m MSE를 weight1로 더한다. Pair는 fixed robot의 실제 object movement다. Shape는 positive, quaternion은 unit canonical 출력이다. Raw orientation error는 대칭 quotient를 적용하지 않은 SO(3) error다.

AdamW lr3e−4, wd1e−5, betas(.9,.999), eps1e−8, clip1, update별 cosine→0, validation250 step, early stopping 없음, 총12,000 update를 그대로 실행했다. Robot/target 통계는 Train21,121 frame로만 fitting했다. Validation absolute episode-macro objective의 첫 minimum으로 선택했다. 이 기준은 branch 수에 가중되는48-Episode 기준이며16-Parent 독립 표본의 통계검정과 동일하지 않다. Final은 선택 동결 후에만 평가로 열었다.

| 모델 | optimizer updates | 선택 step | 선택 validation objective | 학습 wall(s) |
|---|---|---|---|---|
| GNN | 12,000 | 5000 | 0.120 | 329.011 |
| Robot-only | 12,000 | 3000 | 0.233 | 144.345 |

GNN CUDA peak allocation 1.175GiB, train/validation graph load42.681s였다. GNN training329.011s와 Robot-only144.345s는 약0.1315 GPU-hours의 해당 학습 stage wall이며, 수집·재캐시·전송·외부 대기 전체 비용과 같지 않다. 청구 금액은 검증하지 않았다. CUDA vision/CPU geometry overlap, Detection Memo,8 geometry worker, native packed graph 및 GPU-resident training pool을 사용했다. CPU가 빨랐던 geometry를 GPU로 억지 이전하지 않았다.

선택 checkpoint에는 encoder/readout/head, normalization, robot schema, geometry/source/dataset identity와 optimizer/scheduler/RNG가 포함된다. `training/best.pt`와 최종12k `training/last.pt`를 보존했다. 실제 승인된 혼합 pool의 engineering2-update에서 gradient와 저장 후 다음 update loss/weights의 정확한 재현을 확인했다. 본12k 학습 자체는 중단 없이 끝났으며 full-run resume 재실행은 하지 않았다.

![고정 예산 전체 곡선. 선택은5,000 step이며 이후에도12,000까지 실행했다. Position/rotation은 서로 다른 물리 단위로 표시한다.](../../outputs/task10f1/final_run_evidence/figures/training_validation_curves.png)

고정 예산 전체 곡선. 선택은5,000 step이며 이후에도12,000까지 실행했다. Position/rotation은 서로 다른 물리 단위로 표시한다.

## Held-out Absolute State

| GNN field | unit | mean | median | p95 | Episode macro |
|---|---|---|---|---|---|
| peg_position | mm | 15.291 | 11.228 | 42.952 | 15.254 |
| hole_position | mm | 51.071 | 44.442 | 127.258 | 51.614 |
| peg_tip_hole | mm | 57.170 | 52.515 | 130.351 | 57.558 |
| peg_orientation | deg | 4.374 | 2.944 | 12.573 | 4.420 |
| hole_orientation | deg | 8.780 | 6.988 | 22.536 | 8.769 |
| peg_half_x | mm | 8.164 | 6.878 | 19.529 | 8.094 |
| peg_half_y | mm | 1.616 | 1.205 | 4.335 | 1.608 |
| peg_half_z | mm | 1.556 | 1.139 | 4.409 | 1.550 |
| hole_radius | mm | 1.600 | 1.171 | 4.498 | 1.595 |

| 동일 Final frame / model | Peg mm | Hole mm | tip–hole mm | Peg deg | Hole deg |
|---|---|---|---|---|---|
| GNN | 15.291 | 51.071 | 57.170 | 4.374 | 8.780 |
| RobotOnly | 21.033 | 63.312 | 70.585 | 5.294 | 9.397 |
| RawGeometryRidge | 81.424 | 109.430 | 130.149 | 22.270 | 11.358 |
| TrainMean | 85.423 | 126.762 | 160.498 | 25.522 | 11.022 |

Raw Geometry Ridge는 같은 실제 Graph의 object별15D summary×2+Presence2, 총32D를 Train 통계로 표준화하고 alpha1 고정 ridge로18D를 fitting한 기준이다. Robot-only는 같은 예산의76→256→128→18이며 TrainMean은Train 평균이다. Raw centroid의 Box center는 Hole center의 proxy다. 직접 proxy가 지원하지 않는 orientation/shape는 NaN으로 남겨 GT나 평균값으로 채우지 않았다. 위 RawGeometry는 ridge 기준이며 모든 Raw 방법에 대한 우월성을 주장하지 않는다.

![Final1350 frame의 TCP-relative 위치. 낮은 absolute 평균 오차만으로 이동량 복원을 판단하지 않는다.](../../outputs/task10f1/final_run_evidence/figures/gt_vs_predicted_object_position.png)

Final1350 frame의 TCP-relative 위치. 낮은 absolute 평균 오차만으로 이동량 복원을 판단하지 않는다.

![Field별 Final 평균 오차. mm와deg를 섞은 숫자 하나로 표시하지 않는다.](../../outputs/task10f1/final_run_evidence/figures/state_field_errors.png)

Field별 Final 평균 오차. mm와deg를 섞은 숫자 하나로 표시하지 않는다.

| GNN field / visibility | frames | mean mm | p95 mm |
|---|---|---|---|
| peg_position / present | 1104 | 13.527 | 33.144 |
| peg_position / missing | 246 | 23.206 | 65.957 |
| hole_position / present | 1079 | 48.375 | 130.835 |
| hole_position / missing | 271 | 61.804 | 106.624 |

Peg missing구간 평균23.206mm는 Robot-only의 같은 구간20.706mm보다 나빴다. Presence가 있더라도 actor/hole center나 shape가 정확히 관측된다는 뜻은 아니다. Phase별, normalized error, mean/median/p95와 episode/parent macro는 CSV에 별도로 저장했다.

![Final object-present/missing 구간. 각 bar의n은 frame 수이며 독립 Parent 수가 아니다.](../../outputs/task10f1/final_run_evidence/figures/missing_visible_geometry_errors.png)

Final object-present/missing 구간. 각 bar의n은 frame 수이며 독립 Parent 수가 아니다.

## Held-out 실제 물체 이동과 TCP 교란

| actual object motion / model | object | valid pairs | Δ mean mm | Δ median mm | Δ p95 mm | direction cosine | signed gain | magnitude ratio | zero 대비 개선 mm |
|---|---|---|---|---|---|---|---|---|---|
| GNN | peg | 192 | 6.895 | 5.526 | 13.153 | 0.519 | 0.545 | 0.877 | 0.605 |
| RobotOnly | peg | 192 | 7.500 | 7.500 | 10.000 | — | 0.000 | 0.000 | 0.000 |
| RawGeometryRidge | peg | 192 | 7.781 | 4.936 | 25.725 | 0.661 | 0.366 | 0.820 | -0.281 |
| RawCentroidHoleProxy | peg | 192 | 8.340 | 4.437 | 23.975 | 0.769 | 0.724 | 1.248 | -0.840 |
| GNN | box | 284 | 6.854 | 5.200 | 13.408 | 0.544 | 0.506 | 0.874 | 0.646 |
| RobotOnly | box | 284 | 7.500 | 7.500 | 10.000 | — | 0.000 | 0.000 | 0.000 |
| RawGeometryRidge | box | 284 | 20.756 | 8.398 | 42.875 | 0.390 | 0.607 | 2.549 | -13.256 |
| RawCentroidHoleProxy | box | 284 | 43.976 | 6.695 | 162.689 | 0.610 | 3.039 | 6.852 | -36.476 |

각 object는12 Parent다. GNN의 parent-macro Δ 오차는Peg6.895mm/Box6.867mm, signed gain은0.545/0.507이다. 위 cosine은direction이 정의되는 pair만 평균해 Parent별 정의 수 차이가 있으며 Parent-macro cosine도 CSV에 따로 남겼다. Robot-only의 fixed-robot response는 정확히0이므로 direction은 undefined이고 gain/ratio는0이다. GT magnitude0 또는 prediction magnitude≤1e−9m에서는 cosine을None으로 보존했다. Missing geometry의 극미소 rounding response에서 cosine 자체를 물리적 방향성으로 해석하지 않는다.

| GNN object | GT shift mm | pairs | Δ error mm | zero error mm | signed gain | cosine |
|---|---|---|---|---|---|---|
| peg | 10.0 | 96 | 8.804 | 10.0 | 0.539 | 0.565 |
| peg | 5.0 | 96 | 4.986 | 5.0 | 0.551 | 0.471 |
| box | 10.0 | 142 | 8.462 | 10.0 | 0.587 | 0.641 |
| box | 5.0 | 142 | 5.246 | 5.0 | 0.426 | 0.448 |

**5mm Peg의Δ error4.986mm는 zero5mm와 거의 같고, Box5.246mm는 더 나쁘다.** 10mm에서는 일부 개선되지만 meanΔ error8.804/8.462mm가 남는다. 평균 gain~0.5는 under-response를 보인다. Raw centroid Peg cosine0.769 및 Raw ridge0.661은 GNN0.519보다 높아서 방향 복원까지 모든 기준보다 우수하다고 주장할 수 없다. Both-missing과 visibility-changing 결과도 별도 CSV로 보존했다.

![±20mm 표시 확대. 범위를 넘는 점 수를 panel에 명시했다. 모든 outlier는 원본 CSV와 아래 full-range 그림에 남아 있다.](../../outputs/task10f1/final_run_evidence/figures/heldout_object_displacement_response_zoom.png)

±20mm 표시 확대. 범위를 넘는 점 수를 panel에 명시했다. 모든 outlier는 원본 CSV와 아래 full-range 그림에 남아 있다.

![제외하지 않은 full-range displacement plot. Raw/visibility-change outlier 때문에 작은 이동 영역이 눌리는 현상도 보존한다.](../../outputs/task10f1/final_run_evidence/figures/heldout_object_displacement_response.png)

제외하지 않은 full-range displacement plot. Raw/visibility-change outlier 때문에 작은 이동 영역이 눌리는 현상도 보존한다.

![Final valid pair Δ vector error boxplot. 표시에서 outlier 점만 생략했으며 mean/p95 계산과 CSV는 모두 포함한다.](../../outputs/task10f1/final_run_evidence/figures/raw_geometry_vs_pretrained_gnn.png)

Final valid pair Δ vector error boxplot. 표시에서 outlier 점만 생략했으며 mean/p95 계산과 CSV는 모두 포함한다.

TCP 교란은 robot/camera/causal history가 달라지는 temporal before/after다. 실제 Peg 이동과 동일 조건이 아니다. Validation2 Parent/32 recovery branch에서 GNN Peg/Hole Δ error는31.834/34.137mm, Robot-only는77.051/39.233mm였다. 이 결과도 정밀한 relative-state tracking을 지지하지 않으며 작은 Parent 수의 제한을 유지한다. [tcp_disturbance_response_metrics.csv](../../outputs/task10f1/final_run_evidence/tcp_disturbance_response_metrics.csv)

## Readout320와 GT38D Frozen Interface

Oracle assembly26,175 frame에서 원본 encoder와 최대 raw 차이5.96046448e-08, 사전 허용오차1e−7 이내다. Legacy normalizer 적용 최대 차이는4.47034836e-06이며 float32 측정 재조립의 차이를 기록했다. Oracle과 predicted 경로는 분리했다. Predicted route는 model object18 + measured robot15 + derived tip3 + Clean2만 사용했다.

별도 sensor-only bundle에는 Node/Edge tensors, pointer,Robot76만 남겼다. Python audit hook으로 raw/import/GT sidecar 경로 접근을 금지하고 denial positive-control이 실제 발동함을 확인했다. 이 상태에서 `encode_graph`, `predict_state`, `extract`, standalone CLI가 동일한4 frame 결과를 냈다. A와B는 같은 checkpoint를 사용했고 repeat forward는 bit-exact, weight SHA는 전후 `668d2ad1be0149e6735f428fd4f3e729adc5d7159f0f1f835478afa68cb0b646`로 동일했다. Checkpoint file SHA도 변하지 않았다.

![같은 robot/camera/prefix의 실제 object movement에 대한 latent slot 변화. Slot 구조 보존과 변화 민감성은 확인했으나 latent norm만으로 정확한 공간 decoding을 증명하지 않는다.](../../outputs/task10f1/final_run_evidence/figures/readout_slot_responses.png)

같은 robot/camera/prefix의 실제 object movement에 대한 latent slot 변화. Slot 구조 보존과 변화 민감성은 확인했으나 latent norm만으로 정확한 공간 decoding을 증명하지 않는다.

| cached graph / batch1 /32 frame | mean ms | median ms | p95 ms |
|---|---|---|---|
| A_state38_ms | 3.650 | 3.639 | 3.680 |
| B_readout320_ms | 2.745 | 2.749 | 2.771 |
| A_B_shared_extract_ms | 5.281 | 5.274 | 5.361 |

A는CPU38D assembly를 포함하고 B는GPU에 남는다. Shared extract는 weight hash 비용도 포함한다. RGB-D capture/DINO/SAM2/geometry/packing/미래GRU는 이 비용에서 제외된다. 전체 vision latency로 해석하지 않는다.

실제 API는 `StateEncoder.encode_graph(graph)`, `predict_state(graph, robot76)`, `extract(graph,robot76,schema)`와 `contract.assemble_gt38(fields, measured_robot)`다. 직접 prediction API는 Torch tensor이며38D assembly는 NumPy를 사용한다. `extract`가 변환을 포함해 두 경로를 함께 제공한다. B slot order는TCP/Left/Right/Peg/Box다. Head를 제거하지 않고 Freeze한다.

```bash
python -B -m graph.gt_pretrain10f1.extract   --checkpoint /workspace/task10f1_gt_pretrain_20261030/gnn_pretrained.pt   --graphs /workspace/task10f1_gt_pretrain_20261030/sensor_only_frozen_handoff/sensor_graphs.npz   --receipt /workspace/task10f1_gt_pretrain_20261030/sensor_only_frozen_handoff/receipt.json   --schema /workspace/task10f1_gt_pretrain_20261030/sensor_only_frozen_handoff/robot_schema.json   --out /tmp/new_frozen_features.npz --frames 0 1 2 3
```

새 출력 경로를 사용한다. 이4-frame feature artifact는 실제 학습된 checkpoint의 데모이며 전체 GRU feature cache나 GRU 학습은 아니다. Local macOS에서는 CUDA를 실행하지 않았으며 이 보고서의 실제 CUDA 결과는57987772에서 얻었다.

## 평가 실패의 보존과 수정

첫 평가의 Robot-only 동일-input bit-exact 검사가1.1920929e−7 차이로 실패했다. 별도 실제 GPU 실험에서 batch31/64의 행렬 형태가 rounding을 바꾸고 batch64 반복 row는 정확히 같음을 재현했다. 학습 가중치나 threshold를 바꾸지 않고 **Robot-only의 마지막 partial batch만64행으로 padding 후 추가 행을 버렸다.** GNN packing과 결과 계산 경로는 그대로다. 원래 실패 로그·CSV·source를 `evaluation_failed_v1` 및 학습 Source archive에 보존하고, 별도 evaluation v2 source와 UUID로 다시 평가했다. 모든864 candidate의 Robot-only exact zero-response 검사는 v2에서 통과했다. GNN이나 baseline을 재학습하지 않았다.

## 연구 질문별 판정

| 질문 | 등급 | 근거/한계 |
|---|---|---|
| Q1 | CONFIRMED | 측정15D를 제외한 독립 object18D를 예측하고 endpoint3D는 파생한다. |
| Q2 | SUPPORTED | 평가한 분포에서 부분 복원: Peg15.291mm/Hole51.071mm. 정밀 상태로는 부족하다. |
| Q3 | SUPPORTED | 평균 absolute 오차는 비교한 Robot-only/Raw ridge/TrainMean보다 낮다. 모든 Δ 방향 지표 우위는 아니다. |
| Q4 | INCONCLUSIVE | 정확한 새 배치 추적은 확보하지 못했다. gain~0.5,5mm에서zero 대비 개선 미미/악화. |
| Q5 | SUPPORTED | 5×64 구조와 실제 object-movement response 및 Head decoding은 보존. 정확한 slot별 공간 의미는 미확정. |
| Q6 | CONFIRMED / INCONCLUSIVE | 순서·단위 호환과 GT 없는 조립은 확인. 실제 GT 동등 정확도/보편 관측 가능성은 미확인. |
| Q7 | CONFIRMED | 동일한 Frozen GNN+Head에서 A/B 추출을 실행했다. |
| Q8 | CONFIRMED | GPU-resident graph에서 A3.650/B2.745/shared5.281ms 평균; scope 제한 유지. |
| Q9 | SUPPORTED | 숨은 geometry, symmetric orientation, rim/extent/shape,partial-surface center bias가 남는다. |
| Q10 | INCONCLUSIVE | 정밀 Predicted-state 제어로 넘어갈 충분한 근거는 없다. B 인터페이스는 준비됐으나 제어 성능은 별도 연구다. |

이번 작업의 실행 완료와 연구 가설 성공을 구분한다. Closed-loop insertion, 새로운 GRU 및 Action 성능은 **NOT_TESTED**다. B가 반드시 실패한다고도 결론내리지 않는다. 후속 A/B GRU는 새로운 사용자 지시가 있을 때만 시작한다.

## 검증과 산출물

| 검증 | 상태 | 근거 |
|---|---|---|
| final_data_budget_frozen_before_primary_training | PASS | data_ready_release.json |
| parent_branch_test_final_split | PASS | final_dataset_manifest.json |
| GT38_per_dimension_order_units_source | PASS | gt38_feature_contract.json |
| all_corrective_source_graph_GT_physics_hash_runtime | PASS | target_import_verification.json plus independent final validation |
| all332_RGBD_Robot76_GT_timestep_alignment | PASS | gt_label_sidecar_manifest.json |
| all_graph_Node21_Edge4_cap128_temporal0_main_only_box | PASS | final_dataset_manifest.json |
| measured_robot_oracle_GT38_without_object_reconstruction_shortcuts | PASS | state38_assembly_verification.json |
| GNN_readout_head_gradients_finite_save_resume | PASS | training_smoke_verification.json |
| GNN_fixed12k_finite_budget_validation_checkpoint_optimizer_RNG | PASS | training/result.json |
| RobotOnly_fixed12k_finite_budget_validation_checkpoint_optimizer_RNG | PASS | robot_baseline_training/result.json |
| fresh_E0_L2_d64_readout320_18D_head_no_policy | PASS | model.py plus selected state dict |
| training_execution_source_unchanged | PASS | training_execution_source.json |
| evaluation_complete.json | PASS | evaluation_complete.json |
| sensor_only_inference_verification.json | PASS | sensor_only_inference_verification.json |
| post_evaluation_analysis.json | PASS | post_evaluation_analysis.json |
| GT_and_dataset_denied_predicted_assembly_and_A_B_determinism | PASS | sensor_only_inference_verification.json |
| oracle_predicted_assembly_separated_and_observability_disclosed | PASS | state38_assembly_verification.json |
| heldout_actual_object_motion_separate_from_TCP_motion | PASS | geometry_response_summary.csv |
| absolute_state_metrics.csv | PASS | absolute_state_metrics.csv |
| paired_displacement_metrics.csv | PASS | paired_displacement_metrics.csv |
| robot_only_baseline.csv | PASS | robot_only_baseline.csv |
| raw_geometry_baseline.csv | PASS | raw_geometry_baseline.csv |
| geometry_response_comparison.csv | PASS | geometry_response_comparison.csv |
| distribution_state_metrics.csv | PASS | distribution_state_metrics.csv |
| tcp_disturbance_response_metrics.csv | PASS | tcp_disturbance_response_metrics.csv |
| paired_displacement_scale_visibility_summary.csv | PASS | paired_displacement_scale_visibility_summary.csv |
| optimization_equivalence.json | PASS | optimization_equivalence.json |
| engineering_verification.json | PASS | engineering_verification.json |
| new_corrective_original_optimized_graph_gate_memo_equivalence | PASS | operations/direct_watch_20261030/corrective_optimization_equivalence.json |
| standalone_scientific_figures | PASS | figures/ |
| GPU_physics | NOT_TESTED | CPU physics + CUDA renderer/vision executed; no GPU physics claim |
| new_GRU_action_BC_closed_loop | NOT_TESTED | explicitly prohibited and not executed |
| universal_symmetric_hidden_geometry_observability | NOT_TESTED | not identifiable in general; conditional estimates only, no GT filling |
| nominal_full_task_substep_SafetyV2_contacts | NOT_TESTED | Full-task uses nominal executed ScriptedExpert/controller/finite-state/insertion validation; corrective SafetyV2 substep evidence applies to all212 corrective branches only |

단일 seed·12 Final Parent,2 held-out TCP recovery Parent, correlated branches, 조건부 관측 가능성 및 full-task substep contact audit 미실행의 제한을 유지한다. 검증 PASS는 낮은 오류나 정책 성공의 PASS가 아니다.

[gnn_pretrained.pt](../../outputs/task10f1/final_run_evidence/gnn_pretrained.pt) · [checkpoint_manifest.json](../../outputs/task10f1/final_run_evidence/checkpoint_manifest.json) · [gt38_feature_contract.json](../../outputs/task10f1/final_run_evidence/gt38_feature_contract.json) · [target_observability_audit.json](../../outputs/task10f1/final_run_evidence/target_observability_audit.json) · [final_dataset_manifest.json](../../outputs/task10f1/final_run_evidence/final_dataset_manifest.json) · [graph_cache_manifest.json](../../outputs/task10f1/final_run_evidence/graph_cache_manifest.json) · [split_manifest.json](../../outputs/task10f1/final_run_evidence/split_manifest.json) · [gt_label_sidecar_manifest.json](../../outputs/task10f1/final_run_evidence/gt_label_sidecar_manifest.json) · [source_identity.json](../../outputs/task10f1/final_run_evidence/source_identity.json) · [training_execution_source.json](../../outputs/task10f1/final_run_evidence/training_execution_source.json) · [final_pretrain_config.json](../../outputs/task10f1/final_run_evidence/final_pretrain_config.json) · [train_metrics.csv](../../outputs/task10f1/final_run_evidence/train_metrics.csv) · [validation_metrics.csv](../../outputs/task10f1/final_run_evidence/validation_metrics.csv) · [absolute_state_metrics.csv](../../outputs/task10f1/final_run_evidence/absolute_state_metrics.csv) · [paired_displacement_metrics.csv](../../outputs/task10f1/final_run_evidence/paired_displacement_metrics.csv) · [paired_displacement_scale_visibility_summary.csv](../../outputs/task10f1/final_run_evidence/paired_displacement_scale_visibility_summary.csv) · [robot_only_baseline.csv](../../outputs/task10f1/final_run_evidence/robot_only_baseline.csv) · [raw_geometry_baseline.csv](../../outputs/task10f1/final_run_evidence/raw_geometry_baseline.csv) · [state38_assembly_verification.json](../../outputs/task10f1/final_run_evidence/state38_assembly_verification.json) · [geometry_response_comparison.csv](../../outputs/task10f1/final_run_evidence/geometry_response_comparison.csv) · [geometry_response_summary.csv](../../outputs/task10f1/final_run_evidence/geometry_response_summary.csv) · [distribution_state_metrics.csv](../../outputs/task10f1/final_run_evidence/distribution_state_metrics.csv) · [tcp_disturbance_response_metrics.csv](../../outputs/task10f1/final_run_evidence/tcp_disturbance_response_metrics.csv) · [sensor_only_inference_verification.json](../../outputs/task10f1/final_run_evidence/sensor_only_inference_verification.json) · [frozen_extraction_cost.json](../../outputs/task10f1/final_run_evidence/frozen_extraction_cost.json) · [verification.json](../../outputs/task10f1/final_run_evidence/verification.json) · [execution_source_archive_manifest.json](../../outputs/task10f1/final_run_evidence/execution_source_archive_manifest.json) · [artifact_index.json](../../outputs/task10f1/final_run_evidence/artifact_index.json) · [completion_audit.json](../../outputs/task10f1/final_run_evidence/completion_audit.json)

선택 model 안에 Head와 optimizer/scheduler/RNG를 함께 보존하며, `training/last.pt`는12k 종료 상태다. 모든 핵심 파일 SHA/readback과 최종 문서·그림·전달 검증은 artifact index/completion audit로 연결한다. 원래 준비 보고서와 remediation/실패 기록은 덮어쓰지 않고 이 최종 보고서를 별도로 저장했다.

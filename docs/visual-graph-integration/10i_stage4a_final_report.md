# Task10-I Stage4A — Perception 실패 조사 및 신뢰성 개선

**상태: COMPLETED — Stage4A 연구 조사·독립 검증 완료. 최종 채택 판정: NO_GO.**

검증일:2026-11-01. Baseline을 유지하며 Production 설정, Dataset, checkpoint와 정책 학습은 변경하지 않았다.

## 연구 질문과 실행 경계

이 실험은 `PegInsertionSide-v1`의 실제 RGB-D에서 물체가 처음 누락되는 단계와,
잘못된 표면이 최종 Graph까지 남는 단계를 구분한다. 초기 Main-visible 조건은
Stage1–3에서 고정한 그대로 사용했다. 초기 물리적 가시성이 인식 성공을 보장한다는
가정은 사용하지 않았다. 개선 후보는 별도 연구 wrapper로 구현했고 Production
perception, Dataset, checkpoint, split, Task10-H에는 적용하지 않았다.

GNN/GRU/정책을 다시 학습하거나 성공률을 평가한 실험은 아니다. 여기서 Graph의
object recall은 충분히 보이는 대상의 올바른 대표점이 하나 이상 존재하는 비율이다.
완전한 표면, 올바른 연결 구조, 정책 입력의 충분성을 뜻하지 않는다. 마스크 Pixel
Recall, source-pixel retention, 공간 Surface Coverage도 각각 다른 지표로 취급한다.
부록의 `graph/missing`은 충분히 보이는 대상에 대해 **대표점 전체가0**인 비율이다.
오점만 있는 nonempty Graph는 missing=0이지만 correct-point recall=0일 수 있다.
따라서 missing과1−recall을 같은 수치로 해석하지 않는다.

기존 보고서 10개와 연결 Source/Artifact의 식별자는
[reference inventory](../../outputs/task10i/stage4a/report_reference_inventory.json)에 있다.
09-D의 손가락 근처 정상 표면 제거, 09-E의 Box persistence 위험, 10-B/F1/G의
관측·표현 실패를 재조사하되, 과거 결론을 이번 데이터의 원인으로 그대로 대입하지 않았다.
Stage3 인계 계약은 `/workspace/task10i_stage3_20261031_v5_finalization/run`의
`stage4_raw_dataset_receipt.json`, `stage4_recache_input_manifest.json`,
`stage4_source_and_schema_contract.json`을 기준으로 검증했다.

## 데이터, 분모와 독립성

R Train과 D Train은 각각 96 Parent / 272 Logical Episode이며, 공통 236 Logical
Episode를 한 번만 계산한다. 각 Train Dataset의86.76%가 공통 logical source다. 따라서 고유 Train source는 308개이고 Shared Validation
48개를 합친 고유 source는 356개다. 같은 Parent의 corrective branch를 독립 Parent로
세지 않는다. Train union은 114 Parent, Shared Validation은 16 Parent다.

Train Parent를 `sha256('task10i-stage4a-v1:'+parent_id)` 순서로 정렬해 Development
24 Parent, Internal Test 24 Parent, Train Reserve 66 Parent로 사전 등록했다. Reserve
66개는 기존 Stage2 Coverage Holdout60이 아니다. 10-G Final/Test, Coverage Holdout60,
정책 Blind Test는 개선안 선택에 사용하지 않았다.

| Partition | Parent | Logical source | 등록 T+1 관측 | 사용 순서 |
|---|---:|---:|---:|---|
| Development | 24 | 64 | 4,162 | 원인 분석 및 3개 후보 비교 |
| Internal Test | 24 | 96 | 5,903 | Development 후보 Source/Config 동결 이후 |
| Shared Validation | 16 | 48 | 3,085 | 최종 동결 이후 전체 paired 평가 |
| Train Reserve | 66 | 148 | 9,911 | 이번 후보 평가에 사용하지 않음 |

관측은 각 source의 **T+1 전체**를 포함한다. terminal 관측에는 Action이 없다.
Frame 평균, Episode-macro, Parent-macro와 Phase/Segment/Camera/가시 support별
stratum을 별도로 제공한다. 기본 비교는 같은 관측의 결과를 Episode 안에서 평균한 뒤,
같은 Parent의 Episode를 평균하고 Parent 차이를 집계한다. R/D 중복은 늘리지 않는다.

GT-visible support는 0, 1–7, 8–31, 32–127, 128+ pixel로 구분한다. 0은
`PHYSICALLY_NOT_VISIBLE`, 1–7은 tiny-support stratum이다. 이 둘을 충분히 보이는
물체의 detector miss에 합산하지 않는다. 정확한 reference가 없는 관측은
`GT_UNSUPPORTED` 행으로 유지하며 성능의 0 또는 성공으로 대체하지 않는다.

원래 selection의 `graph/object_recall` 구현은 GT Actor pixel 중 valid depth가 있는
reference≥8점을 분모로 쓴다. 물리적 가시성과 depth 관측 가능성이 다를 수 있으므로,
raw Actor pixel≥8을 분모로 하는 correct-point recall도 별도 보조 지표로 등록했다.
기존 selection을 사후 변경하지 않고 두 정의를 구별해 보고한다.

공간 coverage는 현재 GT pixel을 depth로 backproject한 표면 sample에서4mm 이내
대표점이 있는 비율이다. Peg는 양 Camera, Box는 Main만 사용한다. Camera별 pixel
density와 중복 영향을 받으므로 균일한 실제 surface-area coverage가 아니다. 원래
all-node coverage는 근처 FP에도 credit을 줄 수 있어, 올바른 Actor source 대표점만
사용하는 coverage를 추가로 평가한다. 2D erosion으로 얻은 projected boundary와
GT collision aperture의 물리적 rim도 별도 지표다.

Precision은 prediction이 없으면 미정의다. 같은 Parent의 B/C를 비교해도 각 조건에서
prediction이 존재한 frame 집합은 다를 수 있으므로 Recall과 pooled TP/FP/FN을 함께
제공한다. Distance p95의 Parent-macro는 frame별 p95를 Episode와 Parent에서 평균한
값이며, 모든 point를 합친 pooled p95가 아니다.
정량 부록의 surface distance는 오점을 포함한 전체 대표점의 거리다. C절의 올바른
source만 고른 표면 audit와 분모가 다르며, 빈 point set의 distance는 미정의다.
`graph/false_presence`의 원래 분모는 valid-depth GT reference=0인 경우이고, mask의
비가시성은 raw Actor pixel=0으로 정의한다. 두 계약의 차이를 숨기지 않는다.
실제 EXACT_GT의22,310개 object-frame 행을 전수 대조했을 때 raw pixel 수와 valid-depth
sample 수 차이, 0 여부 차이, ≥8 여부 차이는 모두0이었다. 이번 평가의 target에서는
두 visibility 분모가 실제로 같았지만 다른 입력까지 같다고 가정하지 않는다. Box는
Main 관측 기준이며 Wrist에서만 보이는 Box까지 채택하는 정책으로 바꾸지 않았다.
[분모 전수 대조](../../outputs/task10i/stage4a/completion_58576054/visibility_depth_denominator_review.json).

## 새 인스턴스와 Baseline 동등성

Instance57987772에서 옮긴 자산을 Instance58576054, RTX3090 24GB에서 재개했다.
기본 `/venv/main`의 cu132를 바꾸는 대신 독립
`/workspace/venvs/task10i-stage4a-cu130`에 기존 package pin을 복원했다.
Torch2.14.1+cu130 / torchvision0.29.1+cu130, CUDA SAPIEN Renderer와 CPU PhysX를
사용했다. 이는 GPU PhysX 실행 결과가 아니다.

이전 F1 cache의 실제 설정은 `torch.set_num_threads(1); torch.manual_seed(42)`,
deterministic algorithms OFF, matmul TF32 OFF, cuDNN TF32 ON이다. training용
`seed_all`로 math flag까지 바꿨던 초기 재현에서는 Mask9 frame, Graph2 frame의
차이가 났다. 이 실패를 보존하고 실제 cache 설정을 복원한 뒤 비교했다.

Episode2000003의 **전체 101 Action frame**에서 기존 F1 및 이전 Stage4A의 Mask와
Graph의 5개 tensor가 byte-equal이었다. terminal frame은 이전 F1 Graph가 없어 이
101-frame proof의 범위에 넣지 않았다. 다른 Episode를 모두 과거 cache와 재실행해
동등성을 확인했다는 주장은 하지 않는다.
[Forward Equivalence 전체 보고서](10i_stage4a_forward_equivalence_58576054.md),
[실제 proof](../../outputs/task10i/stage4a/final_evidence_58576054/forward_equivalence/new_host_forward_equivalence.json).

완료된 기존 60개 Sequence의 Mask/Graph/receipt SHA를 검증해 재사용했고, 미완료
4개는 t0에서 Memory를 초기화해 처리했다. 이로써 Development64개 / 4,162개 관측의
Baseline을 완성했다. partial trace를 완료 Episode로 세거나 SAM2 Memory 복구 없이
중간 timestep에서 이어 붙이지 않았다.

CPU cgroup quota 약6.47 core에서 BLAS oversubscription을 확인했다. 실제8 frame
geometry control에서 기본 thread 평균0.5568s와 BLAS/OMP/MKL 각각1 thread 평균
0.3147s의 출력은 모두 과거 Graph와 byte-equal이었다. 별도 전체101-frame
Mask/Graph 동등성도 통과한 후 남은 실행에 이 세 thread 환경변수를 명시했다.
속도 개선을 perception 정확도 개선으로 세지 않는다.

`perframe_cache.py`는 원래 `prepare` bytecode를 instance 전용 namespace에서 실행하며
동일 현재 관측/config/mask의 결과만 memoize한다. 과거 surface map이나 SAM2 Memory를
대체하지 않는다. 27개 실제 frame/mask 조건에서 반복 호출까지54개 Graph 비교,
270개 tensor와 Gate가 native와 동일했다. Source hash를 proof와 대조한 뒤 동결했다.
[memo 동등성](../../outputs/task10i/stage4a/final_evidence_58576054/cache_equivalence/equivalence.json).

## Actor GT 확보와 reference의 한계

Stage3 Raw HDF5에는 매 프레임 Actor Segmentation이 저장되어 있지 않았다. 원본을
수정하지 않고 native Expert Action replay와 당시 collection의 snapshot/teleport
계약을 재현해 평가용 Actor Segmentation을 생성했다. Peg/Box Actor mapping은
실제 SAPIEN segmentation ID map에서 분리하고 Robot과 겹치지 않음을 검사했다.

각 프레임에서 양쪽 Camera의 RGB, raw depth, intrinsic, extrinsic_cv,
cam2world_gl을 저장 Raw와 **모두 array-equal** 비교한 경우만 `EXACT_GT`로 인정한다.
이 ID/Mask/Pose/Hole/Phase는 reference 생성·지표·Oracle 전용이다. Practical 후보의
입력에는 전달하지 않는다.

Development는 보존된 canonical reference와 SHA를 다시 대조했고 4,162/4,162개
관측에 EXACT_GT가 있다. 새 partition의 일부 replay에서는 RGB 일부 channel만 ±1
다른 경우가 있었다. 별도 저장 상태 render-only 복원 probe3개에서도 이를 완전히
없애지 못했으며, 일부는 카메라 matrix 약1e−7 차이도 남았다. 단순히 depth가 같다는
이유로 이 결과를 완전한 GT로 선언하지 않았다.

정확 지표와 별도로, 두 Camera의 depth/calibration은 정확하고 RGB는 channel당
최대1, Camera당 최대32개 값 차이만 있는 관측을
`APPROXIMATE_REFERENCE_RGB_ULP_ONLY`로 정의했다. 이 보조 정의는 Validation 전에
등록했으며 정확 GT flag, primary 분모나 판정을 바꾸지 않는다. 이 조건조차 만족하지
못하면 `INSUFFICIENT_REFERENCE`다. 세 종류의 reference 결과를 합산하지 않는다.
[restore probe](../../outputs/task10i/stage4a/final_evidence_58576054/restore_probe/probe.json),
[보조 reference 정의](../../outputs/task10i/stage4a/final_evidence_58576054/approximate_reference/definition.json).

최종 reference accounting은 다음과 같다. Approximate 열은 Primary에서 여전히
`GT_UNSUPPORTED`이며, EXACT_GT와 합친 성능을 만들지 않았다.

| Partition | 전체 관측 | EXACT_GT | RGB-ULP-only 근사 | 불충분 reference | Primary 미지원 |
|---|---:|---:|---:|---:|---:|
| Development | 4,162 | 4,162 | 0 | 0 | 0 |
| Internal Test | 5,903 | 4,813 | 558 | 532 | 1,090 |
| Shared Validation | 3,085 | 2,180 | 332 | 573 | 905 |
| 전체 | 13,150 | 11,155 | 890 | 1,105 | 1,995 |

특히 Validation의905개를 제외한 subset 성능을 전체3,085개의 완전한 GT 성능으로
일반화하지 않는다. 입력·추론 자체는3,085개 전체이며, 성능 지표에서 exact가 없는
frame의 정답을 모른다는 한계다. Reference 미지원이 알고리즘 성공이나 실패라는 뜻도
아니다. 별도 근사 pixel/point rows는 reference 민감도 진단용으로 보존했다.
[최종 accounting](../../outputs/task10i/stage4a/final_evidence_58576054/approximate_reference/reference_accounting.json).

근사 reference subset만의 Wrist Peg pooled pixel 결과도 별도로 확인했다. Internal은
precision0.99655→0.99651, recall0.88194→0.96388이고, Validation은
precision0.99871→0.99492, recall0.70124→0.96101이었다. 이 수치는 pixel-weighted
근사 label 결과이며 앞뒤의 EXACT_GT Parent-macro와 합치거나 직접 같은 분모로
비교하지 않는다. 근사 subset에서도 Recall 복구와 Precision 손실이 함께 나타났다.
완전한 GT에 대한 추가 독립 검증으로 승격하지 않았다.

## 실제 Source 순서와 stagewise 저장

실제 실행 순서는 다음과 같다. 요청의 개념도 순서로 Source를 바꾸지 않았다.

```text
RGB-D → DINO proposals/initial assignment → causal SAM2
  → valid depth → confirmed Robot 제거 → uncertain Robot 제거 → mask overlap 제거
  → Wrist Gate → shape-first sampling → cross-view identity
  → 3mm deduplication → cap128/object → Graph21/Edge4
```

각 Frame/Camera/Object의 원래 mask/box, 단계별 pixel index, Robot owner/depth,
threshold, Gate와 reject reasons, representative3D point와 전체 source association,
F5 availability/validity, Node/Edge tensor를 독립 output에 저장했다. 각 행은
Parent/Episode/Timestep/Timestamp/Camera/Object identity와 GT TP/FP/FN,
Robot/Background/Other-object pixel 수를 연결한다. DINO region과 SAM2 mask도
단순 존재 여부가 아닌 실제 support로 보존했다.

Depth/Robot/Gate 뒤의 source-pixel 수와 Graph 대표점 수는 같은 분모가 아니다.
Dedup된 대표점에 붙은 여러 source pixel도 대표점 여러 개로 세지 않는다.
Graph의 object presence와 source-pixel support를 혼동하지 않는다.

Depth-valid 행은 다음3D 단계에 유효한 support를 뜻한다. 원래 `prepare.clean`은
Robot/overlap을 제거하지만 invalid-depth mask pixel 자체는 남길 수 있어, 이를 깊이로
clamp한 support와 구별했다. 새 refinement 감사에서는 이 실제 mask를 재구성해
기록된 결과와 대조한다. 논리적 retention funnel을 원래 구현의 배열 변경 순서로
바꿔 설명하지 않는다.

## A. DINO: raw proposal, 실제 선택과 누락

실제 detector는 `IDEA-Research/grounding-dino-tiny`다. 모델 및 tokenizer 전체 SHA는
[model identity](../../outputs/task10i/stage4a/final_evidence_58576054/completion/baseline/model_identity.json)에 보존했다.
기존 Prompt는 Peg=`red and white block`, Box=`cardboard box`이고 lower-case 뒤
마침표를 붙인다. Token은 각각 `[CLS] red and white block . [SEP]`,
`[CLS] cardboard box . [SEP]`였다. 128×128 RGB는 processor에서800×800으로 변환됐다.
새 정보가 추가된 것으로 해석하지 않는다. RGB PIL 입력, normalization/resize 설정,
normalized cxcywh→원본 pixel box 변환과 target size `[H,W]`를 확인했다.

box/text threshold는 기존0.1로 같지만 역할은 다르다. 설치된 processor의 box keep는
max token score를 사용하고 text threshold는 decoded phrase를 결정한다. 기존
consumer는 decoded phrase를 버리므로 text threshold만 바꿔도 semantic association이
교정된다고 볼 근거가 없다. 실제 GDino 경로에는 NMS가 없다. score 내림차순 이후
area fraction<0.25인 top5, missing alternative score0.3, Peg/Box exclusion IoU0.3,
5-step scan과 두 scan IoU≥0.5 confirmation이 추가로 작동한다.

같은 forward의 모든 query/token logits, normalized box, input token IDs와 실제
postprocessed 결과를 hook으로 저장했다. threshold 미만 proposal도 관측 가능하다.
전체 query의 위치 IoU≥0.5는 정답 Actor를 자신 있게 인식했다는 뜻이 아니며,
uncalibrated score를 확률로 해석하지 않는다.

실제 detector가 호출된 source frame을 `(Episode,Camera,Object,t)`로 중복 제거한
충분한 가시 support≥8pixel subset의 결과는 다음과 같다. uncalled frame을 raw
proposal 부재로 세지 않았고, 미래 prefix rescan 기록을 과거 decision에 소급하지 않았다.

| 호출된 source subset | 관측 | localized raw query 존재 | threshold 후 localized box 존재 |
|---|---:|---:|---:|
| Main Peg | 78 | 78 | 78 |
| Main Box | 64 | 64 | 64 |
| Wrist Peg | 366 | 366 | 337 |
| Wrist Box | 62 | 62 | 52 |

이 표는 전체 frame의 detector recall이 아니다. Wrist에서는 threshold 아래에
위치상 유용한 proposal이 있는 경우가 있지만, 단순 threshold 완화가 정답 선택이나
오염 억제를 보장하지 않는다. 실제237개 initialization을 원래 scan 규칙으로 cache에서
재구성해 일치시켰다. 선택 box IoU<0.5 또는 GT box 부재는 Main Peg2, Main Box6,
Wrist Peg6, Wrist Box64였다. 위치 오차와 semantic 오류는 별도로 해석한다.

반환된 proposal pool의 한 GT-box IoU≥0.5 match만 TP로 두고 나머지 미매칭/중복
proposal을 세면, Main Peg754/832, Main Box360/424, Wrist Peg3,174/3,511,
Wrist Box730/782개였다. 이는 **후보 pool의 localization/duplicate 통계**이며 최종
인식 FP rate가 아니다. threshold를 낮추기 전에 ranking/assignment를 검증해야 하는
이유다. tiny Wrist Box2개 호출의24 proposal은 이 FP 비율에 합치지 않는다.

실제 assigned initialization에서 충분한 가시 support의 box localization 실패율은
Main Peg2/64, Main Box6/64, Wrist Peg6/45, Wrist Box62/62다. 나머지 Wrist Box2개
initialization은 tiny stratum이다. 미초기화된19개 Wrist Peg Episode는 precision
분모에 넣지 않고 detection/initialization 누락으로 따로 센다.
[실제 호출·초기화 집계](../../outputs/task10i/stage4a/final_evidence_58576054/detector_outcomes/scope.json).

초기화된 box만의 Parent-macro IoU / centre error는 Main Peg0.8695 / 1.07pixel,
Main Box0.8587 / 2.70pixel, Wrist Peg0.8022 / 8.84pixel, Wrist Box0.0049 /
100.94pixel이었다. 지원 Parent는 각각24/24/22/23개이며, tiny stratum과 미초기화는
localization error 분모에서 제외하고 누락 지표에 따로 남겼다.
[위치 오차 집계와 입력 SHA](../../outputs/task10i/stage4a/completion_58576054/initialization_localization_summary.json).

전체 Development t=0의256개 Camera/Object 입력은 raw RGB hash/prompt로 원래
DINO 출력에 연결해 재구성했다. 다른 branch에서 같은 RGB가 memo hit한 경우도
포함하며 실제 initialization box와 array-equal이었다. 다음은 IoU≥0.5인 proposal이
처음 모두 사라지는 제한 또는 실제 joint 선택 결과다. 논리 Episode 수이며 독립
Parent 수와 다르다.

| t=0 결과/첫 제한 | Main Peg | Main Box | Wrist Peg | Wrist Box |
|---|---:|---:|---:|---:|
| localized initialization | 60 | 58 | 20 | 0 |
| box threshold 후 localized proposal 없음 | 0 | 0 | 0 | 10 |
| area<0.25 제한 후 없음 | 0 | 0 | 2 | 32 |
| top5 제한 후 없음 | 0 | 0 | 4 | 10 |
| localized score가 missing alternative0.3을 못 넘음 | 4 | 2 | 38 | 10 |
| joint rank/exclusion이 다른 box/누락을 선택 | 0 | 4 | 0 | 0 |
| tiny support 별도 | 0 | 0 | 0 | 2 |

Wrist Peg의38개 score-limited case는11 Parent, area-limited2개는1 Parent,
top5-limited4개는2 Parent에서 나왔다. Main Peg의4개 초기 누락은2 Parent다.
따라서 원시 DINO0.1 threshold와 joint assignment의0.3 missing score는 서로 다른
병목이다. Wrist Peg 전체 초기 누락을 area 제한 하나로 설명하지 않는다. top5-limited
4개 중2개는 잘못된 box가 채택됐고2개는 초기화되지 않았다. GT box localization
분석으로 semantic correctness까지 단정하지 않는다.
[전수 t=0 재구성](../../outputs/task10i/stage4a/final_evidence_58576054/initialization_bottlenecks/scope.json),
[제한별 count](../../outputs/task10i/stage4a/final_evidence_58576054/initialization_bottlenecks/restriction_counts.json).

전수 t0의 실제 채택 box를 IoU≥0.5로 GT box에 matching한 detection 지표는 다음과
같다. FP는 **채택됐지만 이 IoU 조건에 미달한 box**이며, pixel semantic purity의
정의와 다르다. 누락은 box 없음과 잘못된 box 모두를 포함한다. 모든 t0 입력을
재구성했으므로 이 표는 detector가 호출된 frame만 고른 표와도 구별된다.

| Camera/Object | 충분히 visible | 채택 / localized TP | Detection recall | Missed-visible rate | 채택 box FDR | Parent-macro recall |
|---|---:|---:|---:|---:|---:|---:|
| Main Peg | 64 | 60 /60 | 0.93750 | 0.06250 | 0/60=0 | 0.91667(24) |
| Main Box | 64 | 64 /58 | 0.90625 | 0.09375 | 6/64=0.09375 | 0.87500(24) |
| Wrist Peg | 64 | 22 /20 | 0.31250 | 0.68750 | 2/22=0.09091 | 0.41667(24) |
| Wrist Box | 62 | 62 /0 | 0 | 1 | 62/62=1 | 0(23) |

괄호 안은 지원 Parent 수다. Wrist Box tiny2개는 별도이며, t0 단일 관측이므로
각 row의 frame-weighted recall은 Episode-macro와 같다. 이후 실제 DINO가 호출되지
않은 frame까지 raw query recall을 추정하지 않는다. 초기 detector availability와
현재 SAM2/Graph recall은 별도 결과다.

## B. SAM2와 temporal identity

실제 tracker는 SAM2.1 Hiera Small, `configs/sam2.1/sam2.1_hiera_s.yaml`과 보존된
`sam2.1_hiera_small.pt`를 사용했다. checkpoint SHA는6d1aa6f3…c4d38이며 전체 값은
model identity에 있다. 원래 logits>0 binary mask와 JPEG preprocessing을 유지했다.

기존 `OnlineVisionAdapter`는 한 번 found인 semantic slot을 다시 탐지하지 않으며,
두 slot이 모두 found이면 scan을 중단한다. 초기화 후 마스크가 비거나 잘못된 Actor로
이동해도 found 기록은 남는다. 따라서 초기 box 부재와 초기화 후 SAM2 실패는 다른
경로다. Episode/Branch 시작 reset, observed-prefix 길이 검사, Camera별 별도 state,
늦은 Object 초기화의 기존 Memory 보존을 검증했다. 미래 frame은 입력하지 않았다.

Development 전체의 충분한 visible frame에서 Main Peg70/4,162개는 초기화되지 않은
slot에서 누락됐다. Wrist Peg는1,558개가 초기화되지 않은 누락,112개가 초기화 후
빈 segmentation,16개가 올바른 Peg pixel 없는 잘못된 mask의 downstream reject였다.
이들은 단순 occlusion으로 분류하지 않았다. 이 데이터에서는 Peg가 양 Camera에서
계속≥8pixel 보였으므로 자연적인 Peg reappearance/absent negative stratum은 없다.

Main Box는 충분히 보이는3,944개 중46개에서 올바른 SAM2 pixel이 없었다. 그중19개는
초기화 후 빈 mask,26개는 잘못된 source가 최종 채택됐고1개는 downstream에서 거부됐다.
실제로 Box가 안 보이는131개 관측 중23개에서는 raw mask가 남았다. Wrist Box는
가시2,240개 중2,232개에서 올바른 pixel이 없고, 비가시 관측의 ghost/wrong-object
presence는1,221개였다. Wrist Box Graph는 계약상 항상 제외되어 이 수치를 최종 Graph
오염으로 그대로 합산하지 않는다.

Main Box가 보이지 않는131관측 중23관측에는 실제 Box Graph도 남았다. Frame-weighted
false presence는0.17557, 지원12 Parent의 macro는0.30060이었다. 세 Development 후보가
모두 이 false presence를 줄이지 못했다. Internal의 지원10 Parent와 Validation의
9 Parent에서는 해당 값이0이지만, 이를 모든 episode의 부재 안전성으로 일반화하지
않는다. 자연 Peg-absent negative stratum은 세 partition 모두 없어 별도 미지원이다.

내부 tracker slot ID는 고정되지만 mask의 지배적 Actor는 바뀔 수 있다. Development의
non-empty mask 사이 dominant Actor 전환은 Main Peg2, Main Box17, Wrist Peg2회였다.
이는 내부 ID 교환 횟수와 다르다. raw XOR/union flicker도 물체 움직임을 보정하지 않은
지표다. 실제 0-support 직후≥8pixel로 재등장한 Main Box2회 중1회만 복구됐으며,
Wrist Box18회는 복구되지 않았다. tiny/unsupported frame을 넘어 재등장 run을 이어
붙이지 않는다.
[temporal definitions](../../graph/task10i_stage4a/temporal_audit.py),
[실제 Episode 지표](../../outputs/task10i/stage4a/final_evidence_58576054/completion/extended/temporal_episode_metrics.json).

다른 목표 물체가 mask의>50%를 차지하는 엄격한 semantic-confusion proxy는 Main
Peg5/4,092 nonempty mask, Wrist Peg16/2,492개였다. Robot-majority는 Main Peg81,
Main Box489, Wrist Peg8, Wrist Box2,749개였다. 빈 mask·mixed mask와 Robot/다른
물체 identity를 분리했다. Wrist Box의2,757 nonempty mask 중2,749개가 Robot-majority인
결과는 단순히 Confidence만 낮춰 해당 Camera의 Box를 Graph에 추가할 근거가 없음을
보여준다. 이 Frame count를 독립 Parent 수로 해석하지 않는다.
[Actor majority 집계](../../outputs/task10i/stage4a/final_evidence_58576054/detector_outcomes/mask_identity_counts.json).

## C. Depth, 좌표와 실제 오염

Depth는 raw mm를 m로 바꾸며 nonpositive 값을 invalid로 처리한다. Camera의
intrinsic/extrinsic, CV/GL 변환, TCP/world backprojection, Wrist mount, Robot FK와
recorded link/pad geometry의 정합성을 실제 관측에서 검사했다. 4,162개 관측의 모든
representative source backprojection 최대 차이는2.0902e−6m, foreign/history source는
0이었다. 따라서 큰 오차 전체를 Camera Source나 단위 오류로 설명할 증거는 없다.

전체 평가208 source/13,150관측의 추가 Camera 감사도 통과했다. CV/GL inverse
calibration의 최대 matrix 원소 오차는4.194e−7, 관측 TCP에 대한 Wrist mount의
episode 내 최대 drift는3.109e−7이었다. 각 source의 Main transform/intrinsic 고정성과
등록20Hz timestep clock을 검사했다. 이는 실제 hardware 동기화나 처리 latency20Hz의
증거가 아니다.
[전체 Camera 계약](../../outputs/task10i/stage4a/final_evidence_58576054/camera_contract/scope.json).

정해진 t0/midpoint/terminal의64 source distance 감사에서 올바른 source의 sampled
surface distance는 Peg 평균1.108mm/p952.009mm, Box 평균3.296mm/p956.662mm였다.
10,000점 collision surface 근사이므로 Box4mm 검사는 reference sampling floor의
영향을 받는다. 정확한 analytic mesh distance로 부르지 않는다.

반면 잘못된 Box Background source의 평균 distance는4.376m였다. 추가275개 source
진단에서 ground85개, table-workspace190개였고 ground raw depth median은11.055m였다.
실제 ground/table pixel이 mask에 들어간 semantic FP다. 설치된 camera default far100m와
signed-mm depth 계약을 확인했지만 이를 각 실제 Camera의 measured near/far로
주장하지 않는다. 임의의 depth overflow 보정으로 정상화하지 않았다.
[표면 reference 감사](../../outputs/task10i/stage4a/final_evidence_58576054/surface_audit/surface_reference_check.json),
[Actor/depth 진단](../../outputs/task10i/stage4a/completion_58576054/box_background_actor_depth_diagnostic.json).

## D. Robot Filter: 오염 제거와 접촉 근처 정상 표면 손실

실제 Robot Filter는 per-collision-shape CV depth agreement 또는 volume interior를
confirmed로 분류하고, 추가6mm band를 uncertain으로 분류한다. 둘 다 제거한다.
confirmed tolerance는2.5mm다. Robot 전체 convex hull로 contact 물체를 제거하는
별도 경로를 추가하지 않았다.

| GT valid-depth Peg pixel | 전체 | confirmed 정상 오제거 | uncertain 정상 오제거 | 보존 |
|---|---:|---:|---:|---:|
| Main | 2,237,443 | 1,660 | 1,193 | 2,234,590 |
| Wrist | 20,196,885 | 1,182 | 3,653 | 20,192,050 |

전체 비율은 작아도 observed finger-pad centre20mm 이내 stratum에서는 Main
2,853/17,133=16.65%, Wrist4,835/11,636=41.55%의 정상 Peg pixel을 제거했다.
이는 contact-force GT가 아닌 거리 기반 finger-near 정의다. 원래 SAM2가 이 모든
pixel을 포함했다는 뜻도 아니다. 실제 SAM2 support의 정상 pixel 제거와 GT 전체
표면 보존을 별도로 저장했다.

실제 GT Robot pixel의 제거도 측정했다. Main13,307,418개 중 confirmed13,227,707,
uncertain35,202개, Wrist4,555,250개 중 confirmed4,278,562, uncertain159,850개를
제거했다. 따라서 uncertain band를 통째로 없애는 방안은 오염 비용 없이 정당화되지
않는다. 이번 practical 후보는 이 Filter를 완화하지 않았다. contact 근처 정상 표면
손실은 **미해결된 후속 병목**으로 남긴다.

## E. Wrist Gate: 거부 조건별 진단

설치 Source에서 7개 rule를 확인했다. clean≥6점, valid-depth fraction≥0.5,
clean fraction≥0.25, largest12mm component fraction≥0.6, 20mm local3-neighbor
support fraction≥0.65, component diagonal≤0.35m, cross-camera identity conflict≤0.1이다.
Main 물체 존재 자체를 필수로 요구하지 않는다. 실제 겹치는 projective support만
identity contradiction에 사용한다.

Development4,162개 Gate에서 good2D(정답≥6pixel, precision≥0.9)2,392개는 모두
통과했다. upstream mask가 빈1,670개와 bad2D16개는 모두 거부됐고 ambiguous84개는
통과했다. good2D 거부가 주된 누락 원인이라는 가설은 이 정의와 데이터에서 지지되지
않았다. 통과가 마스크 전체 보존 또는 무오염을 뜻하지는 않는다.

개별 rule의 reject flags와 support를 모두 저장했다. 겹친 rule는 독립 거부 횟수로
합산하지 않는다. Gate를 끄는 조건은 O4 진단으로만 실행했고 Practical 후보에서는
7개 rule와 threshold를 유지했다.

## F–G. Fusion, Sampling과 Graph 신뢰성

Peg Main+gated Wrist / Box Main-only / current-only / Persistent Surface Map OFF를
유지했다. 관측 Timestamp, source identity, Main/Wrist coordinate contract를 검사했다.
GT object association label은 inference에 사용하지 않았다. Wrist Box의 대규모
semantic confusion을 고려해 무조건 활성화하지 않았다.

Main Peg의 pooled source-pixel funnel은 다음과 같다. 모두 같은4,162개 관측이고
GT-visible support 총2,237,443pixel이다. 마지막 행의 attached source 수를 전체
3D coverage로 해석하지 않는다.

| 단계 | 올바른 Peg pixel | Robot | Background | Other object |
|---|---:|---:|---:|---:|
| SAM2 | 2,162,310 | 156,998 | 19,860 | 5,079 |
| valid depth | 2,162,310 | 156,998 | 19,814 | 5,079 |
| confirmed 제거 후 | 2,160,680 | 4,482 | 19,814 | 5,079 |
| uncertain 제거 후 | 2,159,527 | 2,483 | 19,814 | 5,079 |
| overlap/Gate 후 | 2,158,912 | 2,439 | 19,799 | 3,164 |
| shape-first sample | 647,075 | 2,292 | 18,524 | 3,164 |
| identity/dedup source | 647,075 | 2,292 | 18,524 | 3,164 |
| 최종 cap128 Graph source | 374,790 | 2,194 | 18,076 | 3,131 |

Filter 후 source precision 약98.84%가 Graph source에서 약94.12%로 낮아졌다.
structural selection과 farthest sampling이 잘못된 sparse boundary/outlier까지
유지하는 반면 dense 정상 표면을 줄이는 현상이다. 이것은 sampling RNG만으로 누락을
설명할 근거가 아니라, **오염된 support의 구조적 보존**을 설명하는 실제 증거다.
Wrist Peg는 Gate 후 TP12,650,607/FP286, 최종 source TP196,157/FP100으로 훨씬
깨끗했다. Main이 존재를 유지하면서 Wrist 복구의 공간적 이득은 별도로 가능하다.

F5 normal/variation/depth-pair availability와 validity, empty slot, Node cap128,
Graph21/Edge4, finite 값, endpoint를 기록했다. Node14 이후 temporal/status feature는
비활성이다. 존재 flag는 정답 표면의 충분성을 보장하지 않는다.

실제 Shape-First adapter는 `keep_ratio=0.5`, thinning `phase=0`, 추가 Surface 요청
Peg97/Box95를 사용한다. 이는 camera packet의 추가 surface 요청이며 최종 fused
cap128과 다른 단계다. Structural anchor가 없으면 관측 source row0에서 FPS를 시작하고,
그 외에는 anchor가 초기 거리를 정한다. 이 경로에는 random pre-reduction이나
무작위 첫 점이 없다. 최종 budget도 source identity 문자열 순서를 기준으로 시작하는
deterministic farthest selection이다. 다른 sampling 구현의 seed 옵션을 이 실제 실행
경로의 RNG로 잘못 설명하지 않는다. 입력 mask 변화에 따른 대표점 재배치는 가능하다.

Development의 surface-edge 진단은 Peg380,023개 undirected edge 중 current RGB-D
free-space crossing6,354개, Box83,251개 중23개였다. GT square aperture crossing은
Box2개였다. surface edge가0인 frame은 Peg1,533개, Box3,267개였다. Box sampled-mesh
unsupported48,318개는10k reference sampling floor가 크므로 정답 edge 오류율로
단정하지 않는다. edge builder의3개 interior witness보다 촘촘한9-point 감사라는
차이도 있다. 완전한 object surface connectivity를 보장했다는 주장은 하지 않는다.
Oracle GT-mask 조건의 Graph 결과는 별도 비교한다.

current-only는 과거 3D point를 유지하지 않는 계약이다. SAM2가 현재 RGB에서 다른
표면을 계속 mask로 선택하는 temporal ghost까지 막는다는 뜻은 아니다. 비가시 Box의
ghost/wrong source presence와 foreign/history point 검사를 별개로 보고한다.

## 독립 평가 중 발견한 Phase metadata 계약 수정

Internal 후보의96개 / 5,903개 관측 추론이 완료된 뒤 paired 집계에서 기존 Corrective
trace의 길이 가정이 실패했다. 예를 들어 Raw51 Action에 대해 legacy trace는 anchor
이후41개 행만 담았고 Expert Phase 라벨은 없었다. HDF5 Segment는51개 전체에 있었다.
전체 Raw를 보존하고 Phase는 `EXPERT_PHASE_UNAVAILABLE`로 유지하도록 **평가 코드만**
수정했다. 라벨이 있는 partial trace는 검증된 clock join 없이 임의 정렬하지 않는다.

3개 계약을 추가해 task 검사50개가 통과했다. 원래 실패 root와96개 완료 Mask/Graph는
보존하고 새 finalization v5에서 집계만 재개했다. 원래 Development 동결본의 모든
inference file SHA가 같음을 확인했으며, Internal 모델 추론은 반복하지 않았다.
최종 freeze도 같은 inference file map을 유지한 뒤 Shared Validation을 시작했다.
[복구 receipt](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/recovery_receipt.json).

## O0–O4의 원인 분리

모든 조건은 `ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE`이다. Practical RGB-D 개선으로
주장하지 않는다. 기존4개 Development causal prefix66관측/4Parent를 비교하고,
global Graph 누락이 실제로 있는2개 사례의 전체 prefix107관측/2Parent를 추가했다.
이들은 실패 선택 진단이며 모집단 성능 추정이 아니다. Shared Validation으로 사례를
고르거나 후보를 재조정하지 않았다. 두 Oracle 묶음 사이에 prefix/Parent 중복이 있을 수
있으므로 합쳐서173개 독립 관측이나6개 독립 Parent로 세지 않는다.

O0는 저장된 원래 causal 결과, O1은 각 물체가 처음 GT-visible인 현재 frame에 GT box로
SAM2를 초기화한 뒤 causal propagation, O2는 **원래 detector가 해당 slot을 초기화한
경우에만** 현재 GT-visible mask, O3는 초기화 여부와 무관한 GT mask, O4는 원래 mask에서
Wrist Gate만 bypass한 조건이다. O4에서도 depth/Robot/overlap/identity/sampling은 남는다.
O1은 완전한 초기화 시점·box도 제공하므로 box 위치만의 단일 scalar ablation으로
해석하지 않는다. O2는 detector availability 누락을 자동으로 메워 주지 않는다.

원래 O1의 per-object/local-clock 실행이 Baseline의 grouped/absolute-clock과 달라질 수
있어 원래 `OnlineVisionAdapter.process` bytecode를 쓰는 native 대조도 실행했다.
Camera별 state를 분리하고 initializer만 GT-box로 바꿨다. 기존66관측과 추가107관측
각각에서 Mask 차이0, Graph의5 tensor 차이0이었다. 따라서 이 두 진단 범위에서 실행
방식 차이에 의한 혼동은 관측되지 않았다. 다른 입력까지 같다고 일반화하지 않는다.

### 기존4개 prefix의 표면과 Graph

Parent-macro 결과다. 이 subset은 O0부터 Peg/Box fused correct-point presence가 모두1이므로
아래 결과로 global missing 복구를 주장할 수 없다. Coverage는 all-node proximity proxy다.

| 조건 | Peg purity | Peg coverage | Peg projected boundary | Box purity | Box coverage | Box observed physical rim |
|---|---:|---:|---:|---:|---:|---:|
| O0 | 0.96283 | 0.32092 | 0.34270 | 0.89236 | 0.42192 | 0.34089 |
| O1 /native 동일 | 0.96230 | 0.31427 | 0.39705 | 0.93725 | 0.42650 | 0.35648 |
| O2 | 1 | 0.31681 | 0.39172 | 1 | 0.44505 | 0.29843 |
| O3 | 1 | 0.30800 | 0.42442 | 1 | 0.44505 | 0.29843 |
| O4 | 0.94056 | 0.31400 | 0.33810 | 0.89236 | 0.42192 | 0.30464 |

Observed rim은 GT square aperture의512개 평가 sample 중 현재 Main GT-depth 표면4mm
이내인 sample만 분모로 삼았다. 지원 frame은42/66, Parent4개다. GT hole 정보는
평가에만 사용한다. Full rim512개 전체의 coverage는 O0/1/2/3/4에서
0.04058/0.04440/0.03953/0.03953/0.03894였다. 보이지 않는 뒷면까지 현재 Graph가
채워야 한다는 기준으로 해석하지 않는다. 이 aperture의 x extent는10k sampled collision
surface에서 얻으므로 analytic CAD rim과 완전히 같은 reference라고 주장하지 않는다.

GT mask는 source purity를1로 만들었지만 cap128의 공간 coverage를 단조롭게 높이지
않았다. 특히 Box projected boundary는 개선되어도 물리적 observed rim은 악화됐다.
Peg O3의 surface edge는66관측 합계49개, edge 없는 관측59개였다(O0는704개/35개).
이는 intra-object surface edge만의 결과이며 Robot/Box interaction을 포함한 전체 Graph
연결성이0이라는 뜻은 아니다. 깨끗한 mask만으로 충분한 표면·rim·연결 구조가
보장되지 않는 downstream 한계가 확인됐다.

O4에서는 Peg purity가0.96283→0.94056으로 악화됐다. Main Box mask와 clean support는
66관측 모두 같았지만 Box 대표점은8관측에서 달라졌다. 예를 들어0362,t0의 Main Box
sample 정답131pixel이 bypass 후 cross-view identity에서126pixel로 줄었다. 잘못된
Wrist Peg mask가 depth-agreeing 다른 identity로 보이면 원래 `consistent_views`가
Main Box point를 Unknown으로 제거한다. 최종 TP/FP 개수는 같아도 대표점 위치는
달라질 수 있다. Gate 제거의 효과를 Wrist Peg만의 독립 변화로 취급하지 않는다.
[coupling 실측](../../outputs/task10i/stage4a/completion_58576054/gate_cross_object_coupling.json).

### 실제 global Graph 누락2종

Development의 object별 첫 lexical global miss를 추가 forward 전에 등록했다. Peg는
0924,t0 누락 이후t5까지6-frame prefix, Box는0121,t95를 포함하는101-frame prefix다.
아래는 실제 누락 시점의 올바른 대표점/전체 대표점이다.

| 실패 시점 | GT valid-depth sample | O0 | O1 /native | O2 | O3 | O4 |
|---|---:|---:|---:|---:|---:|---:|
| Peg0924,t0 | 1,710(양 Camera) | 0/0 | 123/128 | 0/0 | 128/128 | 0/0 |
| Box0121,t95 | 178(Main) | 0/0 | 126/128 | 128/128 | 128/128 | 0/0 |

Peg는 GT box를 주면 복구되지만 원래 detector availability를 유지한 O2는 여전히0이다.
이 시점의 첫 병목은 초기 detection/assignment다. Gate bypass로 복구되지 않았으며,
Practical V2도 양 Camera 초기 누락을 독립적으로 해결하지 못했다. O3의 clean128점도
pixel-derived surface coverage는0.22924로 완전한 표면이 아니다.

Box는 원래 slot이 초기화되어 있어 O2가 복구한다. GT box 초기화부터 전체 causal
sequence를 실행한 O1도 t95에 정답126점을 유지했다. 더 나은 초기 prompt 경로가 이후
실패를 예방할 수 있음을 보이지만, 이 한 사례로 SAM2의 모든 drift를 box 위치만으로
설명하지 않는다. 원래 t95 mask는 Robot을 포함했고 downstream 제거 후 Graph가 비었다.
GT mask를 주면 같은 depth/Robot/Graph에서128개 정답점과 coverage0.71910을 얻었으므로,
이 사례의 global 누락을 단위·좌표 오류나 Gate 과거부로 설명할 근거는 없다.

[기존 Oracle 정의](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/oracles/oracle_summary.json),
[native 대조66관측](../../outputs/task10i/stage4a/final_evidence_58576054/oracle_grouping/oracle_grouping_control.json),
[Graph/rim 감사](../../outputs/task10i/stage4a/final_evidence_58576054/oracle_graph/scope.json),
[global 누락 등록·완료](../../outputs/task10i/stage4a/final_evidence_58576054/global_missing_oracles/scope.json),
[native 대조107관측](../../outputs/task10i/stage4a/final_evidence_58576054/global_missing_oracles/grouped/oracle_grouping_control.json).

## Practical 후보와 Development Pareto 비교

3개 후보를 같은4,162 Raw 관측과 EXACT_GT에서 비교했다. 처음부터 모든 변경을
동시에 적용하지 않았다. Prompt/모델/Threshold 대규모 sweep나 fine-tuning은 실행하지
않았고, 확인된 초기화/Memory 복구 및 먼 semantic outlier 병목을 우선 구현했다.

1. **Wrist reanchor V2**: 원래 Gate가 실패할 때5-frame 주기로 재탐지한다. 현재 Main의
   clean Peg를 Wrist로 투영해4mm depth-agreeing witness≥6개를 요구하고, top8
   box/mask가 witness≥0.6을 지지해야 한다. raw depth/Robot guard, shared-depth
   disagreement 제거 후 원래 Gate로 다시 검증한다. 이전 correction이 실패하면
   폐기하며 미래/GT 오류 시점으로 reset하지 않는다. Main/Box mask는 그대로다.
2. **Main Peg depth component**: clean point의12mm 연결 component에서 최소6점,
   ≥80% 지배 component가 있을 때만 작은 분리 성분을 제거한다. 애매하면 원래 mask를
   유지한다. GT object size/pose를 쓰지 않는다. Wrist/Box는 바꾸지 않는다.
3. **Main Peg+Box depth component**: 같은 rule를 Main Box에도 확장한 scope ablation.
   Wrist Box는 여전히 OFF이며 나머지 Filter/Gate/Graph는 같다.

Refinement 전의 Wrist reanchor V1은0362 prefix에서 TP를 복구했지만 Robot FP를
급증시켜 탈락했다(TP0→33,095, FP2,712→21,050, Robot19,277). V2의 같은20-frame
smoke는 TP33,079/FP1,508, 별도0666은TP73,828/FP1이었고2000003 positive control은
변하지 않았다. 이3-prefix smoke를 전체 독립 성능으로 주장하지 않는다.

아래는 Development Parent-macro다. Coverage는 원래 all-node4mm proximity이며,
올바른 source만의 coverage는 별도 검증한다. 모든 delta는 같은 지원 Parent의 차이다.

| 지표 | Baseline | Wrist V2 | Main Peg component | Main Peg+Box component |
|---|---:|---:|---:|---:|
| Peg Graph correct-point recall | 0.99202 | 0.99202 | 0.99202 | 0.99202 |
| Box Graph correct-point recall | 0.98401 | 0.98401 | 0.98401 | 0.98401 |
| Peg all-node surface coverage | 0.45400 | 0.48376 | 0.46907 | 0.46904 |
| Box all-node surface coverage | 0.47981 | 0.47981 | 0.47981 | 0.47780 |
| Peg Robot fraction | 0.003142 | 0.003637 | 0.000834 | 0.000834 |
| Peg Background fraction | 0.034473 | 0.034610 | 0.007539 | 0.007539 |
| Box Robot fraction | 0.022774 | 0.022774 | 0.022774 | 0.018654 |
| Box Background fraction | 0.031099 | 0.031099 | 0.031099 | 0.018581 |
| Wrist Peg pixel Recall | 0.65286 | 0.95296 | 0.65286 | 0.65286 |
| Main Peg pixel Recall | 0.93836 | 0.93836 | 0.92160 | 0.92160 |
| Main Peg pixel IoU | 0.86032 | 0.86032 | 0.85341 | 0.85341 |
| Main Box pixel Recall | 0.92401 | 0.92401 | 0.92401 | 0.87669 |
| Main Box pixel IoU | 0.77110 | 0.77110 | 0.77110 | 0.73550 |

Wrist Peg pixel precision은 양쪽에서 정의된22 Parent에서0.94261→0.98463(+0.04202)다.
Baseline prediction이 없어 precision 미정의인 Parent를0 또는1로 넣지 않는다.
Main Peg component의 precision은0.89403→0.90154, Main Box 확장에서는 Box
precision0.81687→0.81945였다. Box 오염이 줄어도 coverage는0.00201 감소했다.
Wrist V2는 Peg Robot fraction이0.000495 증가했다. 개선·악화를 함께 유지한다.

Component 정리는 Main Peg pixel Recall을1.68pp, Box 확장에서는 Box Recall을4.73pp
잃었다. 정상적인 분리 표면도 지배 성분 밖이면 제거될 수 있다. 큰 distance outlier가
줄고 cap128 안의 공간 배치가 나아지는 것과, 원래 정상 mask 표면을 보존하는 것은
별개다. 이 손실을 Graph 오염 감소로 상쇄해 숨기지 않는다.

후보 표의 mask는 **Geometry에 전달한 최종 mask**다. V2에서는 SAM2 재초기화 뒤
depth/Robot/shared-surface refinement까지 거쳤으며, component 후보도 원래 SAM2
출력의 일부를 제거한다. 후보 trace의 기존 stage 이름 `sam2`를 원시 network 출력과
혼동하지 않는다. 원시 proposal/propagation mask는 `correction_masks.npz`에 따로
보존했고, refinement의 실제 TP/FP 손실도 별도 읽기 전용 감사로 확인한다.

사전 기준은 Graph recall gain≥0.02, mask precision loss≤0.005, Robot/Other-object
fraction 증가≤0.002, false presence 증가≤0.005, Box recall loss≤0.005, latency
p95 ratio≤2.0을 모두 요구한다. Development Baseline recall이 이미0.99202여서
2pp 개선은 ceiling상 불가능하다. 이를 결과를 본 뒤 coverage gate로 바꾸지 않았다.
3개 후보 모두 primary gain0으로 `NO_GO`다. Peg absent negative stratum 부재도
`UNSUPPORTED`로 남기며 안전성 PASS로 바꾸지 않는다.

Wrist V2는 clean Main Peg witness가 있어야 복구한다. 두 Camera가 함께 초기화에
실패한 구간을 독립적으로 찾아내는 경로는 추가하지 않았다. 기존 fusion이 이미 Main
점으로 presence를 유지한 상태에서 Wrist 표면을 복구하면 공간 coverage는 늘어도
object-presence recall은 그대로일 수 있다. 실제 primary gain0은 이 구성의 범위와도
맞는다. Main component 후보도 누락된 point를 새로 생성하지 않는다.

수치 안전조건 실패 수→coverage gain→이름 순의 고정 규칙으로 Wrist V2를 최선의
연구 challenger로 동결했다. 이는 Production 채택이 아니라 독립 falsification 대상의
선택이다. Internal/Validation 결과를 보고 component 후보로 바꾸거나 threshold를
재조정하지 않는다.

사전 선택 계약은01:16:26 UTC에 등록됐고 Development freeze는02:51:03 UTC,
final freeze는04:30:44 UTC였다. 두 freeze의 inference file665개 SHA가 모두 같다.
[선택 계약](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/selection_contract.json),
[Development freeze](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/development_candidate_freeze.json),
[final freeze](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/final_candidate_freeze.json).


![Development의 coverage·오염·latency trade-off](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/development_tradeoffs.png)

그림은 Development의 기술 통계다. 첫 두 panel의 coverage는 all-node proxy이며, latency
Baseline bar는 Wrist trial의867.5ms다. 각 component trial의 별도 paired Baseline 시간은
아래 timing 표를 따른다. 그림의50ms 선은 실제 처리 가능률이 아닌20Hz 주기의 기준선이다.
[내보내기 PDF](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/development_tradeoffs.pdf).

## 동결 후보의 Internal Test와 Shared Validation

Internal96개 /5,903관측, Shared Validation48개 /3,085관측을 모두 처리했다. 후보는
`RGBD_MAIN_SUPPORTED_WRIST_PEG_REANCHOR_V2`로 고정했고, Internal 이후에도 Source/Config를
바꾸지 않았다. 두 partition 모두 `NO_GO`다. 다음 B/C는 같은 지원 Parent 집합이며
CI는 Parent 단위2,000회 bootstrap95% 구간이다. 여러 보조 지표에 대한 동시 유의성
보장은 하지 않는다. Precision은 양 조건의 prediction-defined frame까지 같다는 뜻이
아니므로 Recall 및 pooled pixel 결과와 함께 해석한다.

### Internal Test

| 지표 | B | C | Δ | Parent95% CI | 지원 Parent |
|---|---:|---:|---:|---|---:|
| Peg Graph correct-point recall | 0.99022 | 0.99143 | +0.00121 | [+0.00000, +0.00363] | 24/24 |
| Peg all-node coverage proxy | 0.46760 | 0.47984 | +0.01224 | [-0.00519, +0.03609] | 24/24 |
| Peg correct representative fraction | 0.94488 | 0.94558 | +0.00071 | [-0.00140, +0.00409] | 24/24 |
| Peg Robot fraction | 0.00264 | 0.00323 | +0.00058 | [-0.00001, +0.00148] | 24/24 |
| Wrist Peg pixel precision | 0.99494 | 0.99427 | -0.00068 | [-0.00149, -0.00011] | 22/24 |
| Wrist Peg pixel recall | 0.78814 | 0.96038 | +0.17224 | [+0.06497, +0.29767] | 24/24 |
| Box Graph correct-point recall | 0.97071 | 0.97071 | +0.00000 | [+0.00000, +0.00000] | 24/24 |

### Shared Validation

| 지표 | B | C | Δ | Parent95% CI | 지원 Parent |
|---|---:|---:|---:|---|---:|
| Peg Graph correct-point recall | 1.00000 | 1.00000 | +0.00000 | [+0.00000, +0.00000] | 16/16 |
| Peg all-node coverage proxy | 0.51197 | 0.50448 | -0.00748 | [-0.02969, +0.01033] | 16/16 |
| Peg correct representative fraction | 0.95459 | 0.95300 | -0.00158 | [-0.00389, -0.00001] | 16/16 |
| Peg Robot fraction | 0.00326 | 0.00484 | +0.00158 | [-0.00002, +0.00398] | 16/16 |
| Wrist Peg pixel precision | 0.99789 | 0.99531 | -0.00258 | [-0.00495, -0.00054] | 16/16 |
| Wrist Peg pixel recall | 0.81093 | 0.95394 | +0.14300 | [+0.02481, +0.28137] | 16/16 |
| Box Graph correct-point recall | 0.98109 | 0.98109 | +0.00000 | [+0.00000, +0.00000] | 16/16 |

Internal primary gain은+0.121pp로 사전2pp에 미달했다. Validation에서는 정확한 GT가
지원되는 관측의 Baseline Peg presence가 이미1.0이고 후보도1.0이다. Wrist pixel Recall은
+14.30pp 증가했지만 fused Graph coverage proxy는−0.748pp, correct representative
fraction은−0.158pp, Robot fraction은+0.158pp였다. Wrist precision도−0.258pp였다.
사전 허용 오염/precision 한계 안에 있다는 사실을 실제 악화가 없다는 뜻으로 바꾸지
않는다. Coverage CI도0을 가로지르므로 안정적인 공간 개선을 확인했다고 주장하지 않는다.

두 partition의 Box 출력은 후보가 변경하지 않아 paired delta0이었다. Peg absent
negative stratum과 일부 exact reference는 미지원이며, 이를 안전성 PASS로 계산하지
않았다. Latency guard는 사전 Development60-frame 측정값을 재사용했다. Internal/Val
전체 end-to-end latency를 새로 측정했다는 의미가 아니다.

[Internal 판정](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/internal_test_decision.json),
[Validation 판정](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/validation_decision.json),
[최종 scientific handoff](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/handoff.json).

## 올바른 source만으로 다시 계산한 Graph coverage

GT Actor가 맞는 실제 representative만 사용한다. FP node에는 coverage credit을 주지 않는다. Raw-visible object recall과 valid-depth surface sample coverage의 분모는 다르다.

| Partition | Object | Metric | B | C | Δ | Parent95% CI | Parent |
|---|---|---|---:|---:|---:|---|---:|
| development | Peg | raw_visible_object_recall | 0.99202 | 0.99202 | 0.00000 | [0.00000, 0.00000] | 24 |
| development | Peg | valid_source_coverage_4mm | 0.45390 | 0.48366 | 0.02975 | [0.00538, 0.05826] | 24 |
| development | Peg | valid_source_boundary_coverage_4mm | 0.48986 | 0.53233 | 0.04247 | [0.01333, 0.07796] | 24 |
| development | Box | raw_visible_object_recall | 0.98401 | 0.98401 | 0.00000 | [0.00000, 0.00000] | 24 |
| development | Box | valid_source_coverage_4mm | 0.47981 | 0.47981 | 0.00000 | [0.00000, 0.00000] | 24 |
| development | Box | valid_source_boundary_coverage_4mm | 0.55712 | 0.55712 | 0.00000 | [0.00000, 0.00000] | 24 |
| internal_test | Peg | raw_visible_object_recall | 0.99022 | 0.99143 | 0.00121 | [0.00000, 0.00363] | 24 |
| internal_test | Peg | valid_source_coverage_4mm | 0.46747 | 0.47971 | 0.01224 | [-0.00490, 0.03522] | 24 |
| internal_test | Peg | valid_source_boundary_coverage_4mm | 0.47722 | 0.50128 | 0.02406 | [0.00299, 0.05210] | 24 |
| internal_test | Box | raw_visible_object_recall | 0.97071 | 0.97071 | 0.00000 | [0.00000, 0.00000] | 24 |
| internal_test | Box | valid_source_coverage_4mm | 0.46153 | 0.46153 | 0.00000 | [0.00000, 0.00000] | 24 |
| internal_test | Box | valid_source_boundary_coverage_4mm | 0.53467 | 0.53467 | 0.00000 | [0.00000, 0.00000] | 24 |
| validation | Peg | raw_visible_object_recall | 1.00000 | 1.00000 | 0.00000 | [0.00000, 0.00000] | 16 |
| validation | Peg | valid_source_coverage_4mm | 0.51190 | 0.50441 | -0.00749 | [-0.02974, 0.01029] | 16 |
| validation | Peg | valid_source_boundary_coverage_4mm | 0.51781 | 0.52179 | 0.00397 | [-0.01775, 0.02340] | 16 |
| validation | Box | raw_visible_object_recall | 0.98109 | 0.98109 | 0.00000 | [0.00000, 0.00000] | 16 |
| validation | Box | valid_source_coverage_4mm | 0.43386 | 0.43386 | 0.00000 | [0.00000, 0.00000] | 16 |
| validation | Box | valid_source_boundary_coverage_4mm | 0.52789 | 0.52789 | 0.00000 | [0.00000, 0.00000] | 16 |

Development의 올바른 source coverage 이득+2.975pp는 Internal에서+1.224pp, Validation에서는
−0.749pp였다. False point의 근접 credit을 제거해도 독립 Validation의 비개선 결론은
같다. Projected boundary는 Validation에서+0.397pp이나 CI가0을 포함한다. 이 boundary를
물리적 aperture rim 전체의 보존으로 해석하지 않는다. Raw-visible 기준 object recall은
이번 지원 데이터에서 원래 valid-depth 기준과 같은 결과였지만, 두 정의를 합치지 않는다.

[정답 source coverage 원본과 CI](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/correct_source_coverage.json),
[전체 matched strata](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/all_matched_strata.csv).

## Raw SAM2에서 refinement까지의 실제 보존

저장된 원시 proposal/active propagation mask5,950건을 원래 순수 함수로 재구성했고,
recorded guard/trace와 refined mask가 일치했다. 실제 채택된 active mask5,564건도
최종 mask와 일치했다. 그중 EXACT_GT가 있는 active propagation만 아래에 모았다.
채택된 새 reanchor 당일 proposal과 거부된 대안까지 섞어 독립 frame처럼 세지 않는다.
미지원370건의 정답을 추정해 아래에 추가하지 않았다.

| Partition / EXACT active mask | raw SAM2 TP / FP | Robot 제거 후 TP / FP | shared-surface 후 TP / FP | shared 단계 정상 pixel 손실 / FP 제거 |
|---|---:|---:|---:|---:|
| Development /2,733 | 12,971,786 /1,377,000 | 12,971,101 /51,088 | 12,969,978 /44,093 | 1,123 /6,995 |
| Internal /955 | 4,333,025 /267,874 | 4,330,547 /11,685 | 4,328,678 /8,588 | 1,869 /3,097 |
| Validation /1,506 | 5,992,815 /955,825 | 5,991,173 /41,765 | 5,984,376 /31,662 | 6,797 /10,103 |

이 subset의 valid-depth 단계에서는 TP/FP 수가 변하지 않았다. Robot 제거와 shared-depth
검증은 서로 다른 단계다. Validation의 shared 단계가 정상 pixel6,797개를 추가 제거한
것은 실제 손실이며, 원래 Robot-filter 손실로 돌리지 않는다. Main clean mask의 누락이
depth-agreeing Wrist 표면 제거로 이어질 수 있다는 trade-off가 남는다. 이 수치만으로
최종 coverage 악화 전체를 refinement 하나에 귀속하지도 않는다. 출력 추가·cap128의
대표점 재배치까지 함께 변했기 때문이다.
[재구성 scope](../../outputs/task10i/stage4a/final_evidence_58576054/refinement_retention/scope.json),
[단계별 원시 count](../../outputs/task10i/stage4a/final_evidence_58576054/refinement_retention/retention_counts.json).

## 후보의 temporal 보조 진단

Development 전체의 temporal 비교는 후보 동결 뒤 읽기 전용 보조 분석으로 수행했다.
따라서 이 분석을 이미 선택 규칙에 사용했다고 소급 주장하지 않는다. Practical 후보와
수치 기준은 바꾸지 않았다. Baseline의 Wrist Peg는64 Sequence 중43개에서만 올바른
mask run이 있었고, Wrist V2는64개 모두에서 올바른 run을 만들었다. 충분히 보이는
frame의 correct-mask 누락은1,686→40개로 감소했다. 양쪽 모두 이 Peg stratum에서는
올바른 run을 얻은 뒤 terminal까지 다시 끊기는 사례가 없었다.
여기서 run은 GT-visible≥8pixel에서 TP>0인 support run이다. 완전한 mask나 높은 IoU를
보장하지 않으며, 일부 정상 pixel을 남긴 채 오염되는 drift는 precision/IoU로 별도 평가한다.

raw mask XOR/union의 Parent-macro는 Wrist Peg0.05712→0.07305로 증가했다. 빈 mask가
계속 유지되면 XOR가0이므로 이것만으로 Baseline이 더 안정적이라고 판단하지 않는다.
fragment count43→64도 기존에 계속 누락되던 Sequence에 처음 올바른 run이 생긴 결과다.
내부 slot ID switch와 dominant Actor 변화도 분리했다.

Main Peg component는 Main Peg XOR0.06409→0.06587, Main Peg+Box component는
Main Box XOR0.14082→0.14499였다. 이 변화는 motion compensation 없는 지표이며, 세
후보의 raw temporal rows와 Episode/Parent 통계를 함께 보존했다.
[temporal 보조 분석 범위](../../outputs/task10i/stage4a/final_evidence_58576054/development_temporal/scope.json).

## Phase별 개선과 악화

Wrist V2의 Development Parent-macro 결과다. Phase는 평가에만 사용했고 inference
trigger에 입력하지 않았다. 표의 coverage는 all-node proximity이며, 이후 올바른
source만의 coverage와 대조한다.

| Phase | 지원 Parent | Wrist Peg Recall B→C | Peg Graph coverage B→C |
|---|---:|---:|---:|
| Reach | 24 | 0.5328→0.9376 | 0.3638→0.3542 |
| Descend | 24 | 0.7258→0.9729 | 0.4825→0.5127 |
| Close | 24 | 0.7477→0.9692 | 0.5126→0.6002 |
| Lift | 24 | 0.7456→0.9584 | 0.5113→0.5888 |
| Align | 24 | 0.7451→0.9550 | 0.5367→0.6164 |
| Insert | 24 | 0.7461→0.9564 | 0.5644→0.6584 |
| TCP disturbance | 1 | 0.0579→0.9924 | 0.2387→0.2337 |
| Expert recovery | 1 | 0.1874→0.9973 | 0.2309→0.2482 |

초기 Reach의 Graph coverage는 악화됐다. 재초기화로 Wrist pixel을 많이 복구해도
두 시점의 cap128 대표점 배치가 달라져 공간 이득이 단조롭지 않다. 이 표만으로 그
변화를 하나의 sampler 내부 원인으로 확정하지 않는다. Disturbance/recovery의16개
corrective branch는 같은 한 Parent라 일반화 근거가 약하다. 기존 source의 Phase
metadata가 없는 구간은 `EXPERT_PHASE_UNAVAILABLE`로 유지했다. terminal과 corrective
prefix도 별도 scope로 기록한다.

## Observer-free latency

RTX3090에서 정해진3개 Development의 causal20-frame prefix, 총60 frame을 측정했다.
Raw-proposal 저장, GT metric, Graph artifact 쓰기는 timed path에서 제외했다.
기존 JPEG adapter IO, RGB-D read/FK, DINO/SAM2, correction, geometry/Graph는 포함한다.
공통 vision forward를 한 번 측정하고 양 branch 시간을 따로 더했다.

| 후보 | Baseline mean / p95 (ms) | Candidate mean / p95 (ms) | p95 ratio |
|---|---:|---:|---:|
| Wrist V2 | 704.8 / 867.5 | 847.6 / 1,187.4 | 1.3688 |
| Main Peg component | 704.9 / 869.4 | 721.4 / 882.1 | 1.0147 |
| Main Peg+Box component | 710.2 / 875.3 | 729.0 / 891.6 | 1.0187 |

Wrist V2 run의 Baseline 평균은 DINO48.1ms, SAM2352.0ms,
Geometry/Graph218.8ms, 공통 read/FK53.1ms였다. Candidate는 추가 DINO3.7ms,
추가 SAM279.1ms를 썼고 correction 전체215.7ms에 Filter/Gate 검증도 포함됐다.
세부 timer는 전체 elapsed와 포함 관계가 있으므로 서로 전부 더하지 않는다.
p95들의 합도 전체 p95가 아니다. 드문 재탐지는 평균 비용과 call 기록으로 확인한다.

별도로 기록한 하위 처리 시간은 다음과 같다. Candidate의 Robot/Mask/Gate는 correction과
build 전체의 해당 subcall을 합친 값이고, Geometry/Graph는 build만의 elapsed다.
따라서 표의 행을 더해 전체 latency를 만들면 중복 계산된다.

| 처리 항목 | Baseline mean /p95(ms) | Candidate mean /p95(ms) |
|---|---:|---:|
| Robot classification | 84.38 /94.68 | 87.72 /95.88 |
| Boolean mask filtering | 0.207 /0.249 | 0.334 /0.470 |
| Wrist Gate | 6.43 /24.83 | 25.47 /68.50 |
| Geometry/Graph build | 218.80 /285.66 | 145.90 /211.84 |

Raw depth 변환/backprojection 등은 RGB-D read 또는 geometry의 해당 실행 시간에
포함된다. Boolean filtering timer만을 전체 Depth 처리 비용이라고 부르지 않는다.
Candidate build가 더 짧은 것은 앞선 correction에서 prepare/classification을 계산한 뒤
같은 현재 frame의 memo를 이용하기 때문이며, 전체 pipeline 속도 개선이 아니다.
DINO/SAM2 Baseline p95는403.38/430.94ms, 추가 SAM2 p95는129.58ms였다. 추가 DINO의
all-frame p95는0이지만 평균3.71ms다. 호출이 드물어 p95에서 빠진 결과이며 비용이
없다는 뜻은 아니다. 실제 correction 전체 p95는306.20ms였다.

20Hz control period50ms를 만족한 실행이 아니다. GNN/GRU, PhysX step 및 실제
closed-loop scheduling은 포함하지 않았고 전체 Validation latency distribution을
측정한 것도 아니다. 따라서 이 결과로 실시간20Hz를 주장하지 않는다.
[timing 원본](../../outputs/task10i/stage4a/final_evidence_58576054/finalization/latency_reanchor/latency.json).

## 실제 실패 사례의 시각 검토

전체10개 요청 범주를 Development에서 탐색했고8개 범주/8개 Parent의 EXACT_GT
사례를 보존했다. `case00`–`case07`의8개 PNG를 직접 열어 RGB, GT, Baseline/Candidate
mask와 Graph source overlay를 대조했다. 단일 frame 그림만으로 temporal 전환을
확정하지 않고 이전 frame 수치와 reappearance event도 확인했다.

| 사례 | 실제 관측과 확인 내용 |
|---|---|
| case00,0924,t0 | Main Peg835pixel/Wrist875pixel이 보이는데 양쪽 Peg mask와 Graph가0. V2도 Main witness가 없어 복구하지 못함 |
| case01,0121,t95 | Main Box178pixel이 보이지만 SAM2가 Robot 부위를 Box로 포함. Box Graph source0; downstream의 적절한 Robot 제거와 상류 오인식을 구별 |
| case02,0124,t28 | 접촉 근처 Peg raw mask의 Robot 오염 사례. Filter 후 source와 원래 mask를 별도 표시; raw FP를 최종 Graph FP라고 그대로 세지 않음 |
| case03,0362,t0 | Wrist Baseline Peg mask가 실제 Box 쪽을 선택. V2는 실제 Peg로 복구하지만 Main이 이미 presence를 유지해 fused recall 이득은 없음 |
| case04,0520,t60→61 | Main Box GT12→13pixel, 올바른 mask pixel11→0. 8–31 support stratum의 실제 segmentation 소실 |
| case05,0134,t21→22 | Main Box slot의 dominant Actor가 correct→other-object. t22 정답9pixel 중8pixel은 남지만 전체 mask24pixel 중 타 물체9pixel이 최다; 완전한 ID 교환과 구별 |
| case06,0427,t14 | Wrist Box GT0pixel인데 mask258pixel이 Robot 부위에 남음. Wrist Box Graph는 계약상 제외되어 Graph ghost로 합산하지 않음 |
| case07,0529,t65–67 | Wrist Box가0-support 뒤9pixel로 재등장했지만 mask가 복구되지 않음. 작은8–31 stratum이며 Peg recovery 증거가 아님 |

정의상 good Wrist mask의 Gate reject(≥6TP,precision≥0.9)와 good2D지만 final support<6
(raw TP≥32) 사례는0개였다. 이 두 범주는 `NOT_OBSERVED_UNDER_REGISTERED_DEFINITION`
으로 남겼다. Sampling이 표면 대부분을 제거한다는 pooled funnel과, 위의 거의 완전한
source 소실 조건이0이라는 결과는 다른 질문이다. 사례 수를 독립 표본의 실패율로
해석하지 않는다.

[전체 실패 gallery](../../outputs/task10i/stage4a/final_evidence_58576054/posthoc/gallery/index.html),
[사례와 분모](../../outputs/task10i/stage4a/final_evidence_58576054/posthoc/gallery/case_manifest.json),
[인접 frame/event 대조](../../outputs/task10i/stage4a/completion_58576054/gallery_transition_review.json).

## 실행 식별자와 재현 경로

| 실행 | ExperimentLogger UUID / Root |
|---|---|
| 새 Host 전체101-frame 동등성 | `813d4b24cd414cbfb327e25145473372` |
| Development Baseline 복구 | `09f016cc9b4f461c9ca213c49532334a` |
| Development Wrist V2 | `4b06e09650c345ee90611f7fc2780200` |
| Development Main Peg component | `e6b6053d5db74c6c885ee6b7a0871fe5` |
| Development Main Peg+Box component | `dd7fcc8fc0ee4d36929a79d82f16b94c` |
| Internal 완료 후보 추론 | `0f5e569897c2407d8400e5c4caa0c027` |
| Phase metadata 복구/최종 coordinator | `c3faafc077f24bcc875c686ff2a961da` |
| Shared Validation partition | `0b61e29810f24ab588593cd8ee140d4f` |
| Shared Validation 후보 추론 | `9cea1971d4e14a6cb7ecf27d49880dd3` |
| 최종 통계·Gallery·보호 감사 | `0283a6db678946cb8a1ddf8f7b418065` |

현재 독립 scientific root는 `/workspace/task10i_stage4a_finalization_20261101_v5/run`,
Source는 같은 top의 `source/`다. Development audit root는
`/workspace/task10i_stage4a_completion_20261101_v1/run`, handoff audit는
`/workspace/task10i_stage4a_posthoc_20261101_v3/run`이다. 각 단계의 Source/Config,
모델/입력 hash, receipt, 중간 Mask/Point/Graph는 해당 root에 보존한다. source/config
freeze와 재개 receipt를 확인한 뒤 기존 root를 덮어쓰지 않는 새 root에서 실행한다.

계약 검사는 `.venv/bin/python -B -m unittest discover -s graph/task10i_stage4a -t . -p 'test_*.py' -v`
이며, 실제 GPU 연구는 task 전용 cu130 activation과 3개 CPU thread 환경변수1,
W&B disabled/HF offline 설정을 사용한다. Source의 `resume_finalization.py`는 완료된
Internal 추론을 재사용하는 metadata 오류 복구 경로이며 새 연구의 후보 선택을 우회하는
일반 resume 명령으로 해석하지 않는다.

## 보호 자산과 접근 범위

기존 F1/B/C의 명시된19개 checkpoint를 먼저 hash 검증했고, 기존 F1/Task10G
artifact index가 가리키는158개 `.pt` 자산으로 검사를 확장했다. 이 두 개수를 더해
177개 독립 checkpoint라고 세지 않는다. 확장 목록에는111개 rollout initial snapshot과
기존 Task10G Final의36개 snapshot이 포함된다. 모두 기존 metadata의 SHA와 같았고
tensor deserialization이나 해당 Final/Test의 영상·성능·평가 결과를 후보 선택에
사용하지 않았다. 따라서 기존 Final snapshot의 byte를 전혀 읽지 않았다는 표현도
사용하지 않는다. 이 접근은 보존 checksum 검증에 한정된다.
[명시 checkpoint 검사](../../outputs/task10i/stage4a/completion_58576054/protected_checkpoint_hashes.json),
[확장 보호 검사](../../outputs/task10i/stage4a/completion_58576054/protected_checkpoint_hashes_expanded.json).

새 정책 Blind Test는 열지 않았고 Task10-H 실험을 실행·변경하지 않았다. Raw/GT712개 파일 및 기존
Dataset manifest의 최종 hash 검사가 모두 PASS했다. 기존 Main-visible 판정 코드와
Train/Validation/Final split은 변경하지 않았다. Development/최종 freeze의 모든 inference
source SHA가 같고 Validation 종료 후의 source도 같았다. 전체13,150관측에서 cap128,
Graph21/Edge4, finite tensor/endpoint, 현재 Episode/timestep/Camera provenance를 확인했다.
검사한 source 관계3,221,412건에서 foreign/future/history source와 Wrist Box source는0건이었다.
[원본/동결 source 보호](../../outputs/task10i/stage4a/final_evidence_58576054/posthoc/handoff_audit/protection_checks.json),
[전체 Graph provenance](../../outputs/task10i/stage4a/final_evidence_58576054/posthoc/handoff_audit/causal_graph_checks.json).

## 계약 테스트와 검증 범위

Task 전용 테스트50개와 기존 경로의 회귀 테스트34개가 통과했다. 같은 테스트를
반복 실행한 횟수는 합산하지 않는다. 아래 단위 테스트는 synthetic/fake model을
포함하므로, 실제 모델·Renderer 실행 증거와 구별한다.
[Task 결과](../../outputs/task10i/stage4a/completion_58576054/final_task_tests.log),
[기존 경로 결과](../../outputs/task10i/stage4a/completion_58576054/final_inherited_tests.log).

| 필수 계약 | 검사와 실제 실행 증거 | 주장 범위 |
|---|---|---|
| Baseline Forward Equivalence | 실제101-frame Mask/5 Graph tensor byte equality, current-frame memo54 Graph 비교 | 등록 fixture 전체; 모든 과거 Episode의 재실행 증명은 아님 |
| Camera/Coordinate | coordinate mapping/world-transform/calibration reject 테스트, 실제 Development source backprojection 검사 | CPU PhysX/CUDA SAPIEN 실행; macOS GPU나 GPU PhysX 검증 아님 |
| GT Actor ID/Alignment | 실제 Actor map의 Peg/Box/Robot ID 분리, 양 Camera RGB/depth/3개 calibration array equality; reference reject 테스트 | 완전 일치 frame만 EXACT_GT; mismatch는 미지원 |
| False-positive/Empty-object | `test_false_positive_control`, `test_empty_object_control`, 실제 오염 pixel/point 집계 | synthetic 음성 대조와 자연 데이터 결과를 분리; 자연 Peg-absent 검증은 별도 지원 여부 표시 |
| Fully-visible/Occluded | positive control, depth disagreement/invalid depth, Main occlusion에서 complementary Wrist surface 보존 테스트 | 단위 테스트의 가림 제어와 실제 Box 재등장 결과를 구별 |
| Causal SAM2/Reset | 기존 prefix/suffix invariance, late initialization memory 보존, Episode reset, branch/clock 불연속 거부 | 모델 교체 없이 전체 causal prefix 실행; 미래 관측에 의한 correction 없음 |
| Main/Wrist isolation | 별도 Camera state, cache Camera/frame/Episode key 음성 대조, source identity 및 Wrist Box block | 실제 representative source까지 현재 Episode/timestep/Camera를 추적 |
| Robot Filter source identity | confirmed/uncertain 별 실제 정답·오점 제거, finger-gap/source conflict 테스트 | 정상 접촉 표면의 오제거를 성공으로 숨기지 않음 |
| Wrist Gate rejection attribution | 7개 원래 rule별 값/reject reason, GT 없는 Gate 테스트, raw clean-fraction 우회 방지 | Gate 제거는 Oracle 진단 전용; Practical 후보에서 원래 Gate 유지 |
| Sampling/Graph | cap128, shape-first/final-FPS 순서, dedup source 고유성, node permutation/schema/empty slot 테스트 | 공간 coverage와 source support/존재 flag를 구별 |

전체 Linux replay와 causal inference는 새 독립 root/tmux에서 실행했다. 로컬 테스트의
통과는 Linux 모델 실행을 대체하지 않는다. 실제20Hz closed-loop, GNN/GRU 정책 성공률,
외부 W&B/HF 게시, macOS simulator/MPS 및 GPU PhysX는 이번 검증 범위에 없다.

## 요구사항별 완료 감사

완료 여부는 실행 root의 state 이름만으로 판정하지 않는다. 아래 증거에서 입력·분모,
실제 실행 범위와 실패/미지원 결과를 함께 확인한다. 연구가 완료되어도 후보의 GO나
모든 frame의 EXACT_GT 확보를 뜻하지 않는다.

| 요청 범위 | 대조한 권위 있는 증거와 검증 범위 |
|---|---|
| §1 기존 자산/Stage3 계약 | `report_reference_inventory.json`, source/data audit, Stage3의3개 handoff 문서; R/D 공통236개 중복 제거 |
| §2 GT와 Student 분리 | 원본 HDF5 schema, replay/repair frame별 equality와 Actor ID map, 후보 입력 API; EXACT/근사/미지원 별도 분모 |
| §3 단계별 최초 손실 | `stage_rows.json`, `support.npz`, `trace.json`, raw DINO query 및 retention/attribution 결과; 실제 Mask/Point/source identity 보존 |
| §4A DINO | raw900 query/token score, 실제 scan/t0 선택 재구성 및 threshold/area/top5/joint 제약; 미호출 frame을 proposal 부재로 세지 않음 |
| §4B SAM2 | 원래 source의 found/scan 정책, causal 전체 sequence와 temporal rows, reanchor event/raw mask; 초기화 부재와 propagation 실패 분리 |
| §4C Depth/좌표 | mm/invalid convention, 실제 source backprojection, camera mount/inverse/timestamp 감사; sample-mesh distance의 근사 오차 공개 |
| §4D Robot Filter | confirmed/uncertain 별 정상 Peg 오제거와 실제 Robot 제거, near-finger stratum; contact 보존 한계 공개 |
| §4E Gate | 실제7개 rule와 threshold, 각 frame reject reason/quality; Gate bypass는 O4 전용 |
| §4F Fusion/Identity | Peg 양 Camera/Box Main-only, dominant Actor와 slot ID 분리, 현재 source/Camera 검사 |
| §4G Sampling/Graph | cap128/shape-first/identity/dedup funnel, F5/schema/edge audit, projected boundary/물리적 aperture rim을 별도 평가 |
| §5 통계 | Episode/Parent/Frame-weighted 및 pooled TP/FP/FN, Camera/Phase/Segment/support/t0·later strata, matched Parent CI; tiny/undefined/unknown 유지 |
| §6 실패 사례/Partition | 사전 등록 Dev24/Internal24/Val16 Parent, 실제10종 실패 범주 탐색과 EXACT_GT gallery; 관측되지 않은 범주를 조작하지 않음 |
| §7 Oracle | O0–O4의 실제 구현 의미와 causal prefix, native grouped GT-box 대조, global Graph-missing 보조 사례; SIM_ONLY_ORACLE 표시 |
| §8–9 후보/선정 | V1 실패·V2 및 component scope 분리, 동일 Raw/reference, 사전 numeric gate와 Pareto 손실, 두 freeze의 inference SHA 동일 |
| §10 인과성/시간 | Episode/branch reset·관측 clock·Camera state, correction 실제 시점, observer-free60-prefix timing;20Hz 미달과 측정 제외 항목 명시 |
| §11 전체 독립 검증 | Internal96/5,903와 Shared Val48/3,085의 complete registered partition receipt, paired 결과와 미지원 frame 보존; 재튜닝/후보 교체 없음 |
| §12 자동 실행/테스트 | 단계별 Python runner/tmux,50개 task+34개 기존 테스트, 실제 GPU forward/replay proof; synthetic 대조와 실제 관측 증거 구별 |
| 보호/인계/보고서 | Raw/GT712개·기존 Dataset·동결 source·checkpoint SHA, 전체 Graph provenance, 이 전체 보고서·실제 image review·최종 completion receipt |

## 최종 판정과 인계

Stage4A의 조사, 후보 구현·비교, freeze, Internal Test, 전체 Shared Validation,
Oracle 원인 분리, 실패 사례 검토와 보호 감사는 완료했다. **실제 RGB-D 후보의 최종
판정은 NO_GO**다. 세 후보 모두 사전 primary를 통과하지 못했고, 독립 검증을 받은
Wrist V2는 Validation의 올바른 source coverage도 개선하지 못했다. 후보를 Production에
적용하거나 Stage3 전체를 새 후보로 recache하지 않는다. Existing Baseline의 Mask/Graph
재현성 PASS와 새로운 Perception 구성의 채택 실패를 구별한다.

원인도 하나의 Confidence threshold로 요약되지 않는다. 실제 첫-frame assignment
누락, found slot의 재탐지 부재와 잘못된 초기화의 지속, 접촉 근처 Robot-filter 오제거,
Wrist semantic conflict의 Main Box 파급, 오염된 sparse boundary의 sampling 보존,
GT mask에서도 남는 cap128/edge/rim의 한계가 각각 관측됐다. 반면 일반적인 depth
단위·Camera 좌표 오류나 정상 Wrist mask의 Gate 오거부가 이번 누락의 주원인이라는
가설은 확보한 증거로 지지되지 않았다.

후속에 남기는 연구 문제는 양 Camera 동시 초기 누락을 복구할 RGB-D 경로,
정상 접촉 표면을 보존하는 Robot 분리, 잘못된 Wrist identity의 타 물체 파급 억제,
semantic outlier에 강한 sampling과 물리적 rim/edge 보존이다. 이 Stage에서 검증된
해결책이라고 주장하지 않는다. 새로운 선택 기준이 필요하면 새 protocol과 새 독립
검증을 등록해야 하며, 이미 확인한 Shared Validation을 계속 tuning set으로 쓰지 않는다.

최종 light evidence snapshot은 아래 local root에456개 파일/각 SHA와 실제 remote
원본·resolved path를 보존했다. 원본 RGB-D/모델 weight/전체 trace는 다운로드하지 않고
remote의 독립 scientific root에 유지했다. Gallery 이미지8개와 trade-off plot을 직접
검토했다. 통계 formatter는 저장 결과만 읽으며 추가 모델 forward나 후보 선택을 하지
않는다. Raw trace의 `sam2` stage 이름, precision 미정의, reference 미지원, Phase 부재와
표면 sample/mesh 근사의 범위는 앞의 정의 그대로 인계한다.

```text
outputs/task10i/stage4a/final_evidence_58576054/
outputs/task10i/stage4a/final_report_digest_58576054_v1/
outputs/task10i/stage4a/final_handoff_58576054_v1/completion_audit.json
```

[최종 evidence manifest](../../outputs/task10i/stage4a/final_evidence_58576054/download_manifest.json),
[formatter 입력 SHA](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/input_hashes.json),
[Oracle·근사 reference·raw refinement 수치](../../outputs/task10i/stage4a/final_report_digest_58576054_v1/additional_diagnostics.json).
완료 receipt는 별도 공통 ExperimentLogger UUID로 이 보고서·실제 review·필수 evidence의
SHA를 연결한다. 완료는 연구 절차의 완료이며 후보의 성공/안전성·정책 성공률·20Hz
실행이나 외부 게시를 뜻하지 않는다.

## 전체 paired 정량 부록

B/C는 동일하게 지원되는 Parent 집합의 평균이다. Precision은 prediction이 없으면 미정의이므로 양 조건의 유효 frame 집합까지 같다는 뜻은 아니다. Δ CI는 Parent를 단위로 2,000회 bootstrap한 95% 구간이며, Development CI는 탐색적이다. Unsupported reference/precision을 성공이나 0으로 바꾸지 않는다. Graph presence와 coverage 정의는 본문을 따른다.

### Development: 탐색적 비교

| Metric | B | C | Δ | Parent bootstrap95% CI | paired Parent |
|---|---:|---:|---:|---|---:|
| mask/base_camera/0/precision | 0.89403 | 0.89403 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/0/recall | 0.93836 | 0.93836 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/1/precision | 0.81687 | 0.81687 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/1/recall | 0.92401 | 0.92401 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/hand_camera/0/precision | 0.94261 | 0.98463 | 0.04202 | [-0.00388, 0.13187] | 22/24 |
| mask/hand_camera/0/recall | 0.65286 | 0.95296 | 0.30010 | [0.16453, 0.45069] | 24/24 |
| mask/hand_camera/0/iou | 0.64219 | 0.93906 | 0.29687 | [0.16615, 0.43490] | 24/24 |
| mask/hand_camera/0/visible_object_recall | 0.70533 | 0.98404 | 0.27871 | [0.15327, 0.41691] | 24/24 |
| graph/0/object_recall | 0.99202 | 0.99202 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/0/missing | 0.00798 | 0.00798 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/0/correct_fraction | 0.95724 | 0.95663 | -0.00061 | [-0.00146, 0.00002] | 24/24 |
| graph/0/robot_fraction | 0.00314 | 0.00364 | 0.00050 | [-0.00002, 0.00130] | 24/24 |
| graph/0/wrong_object_fraction | 0.00514 | 0.00512 | -0.00002 | [-0.00007, 0.00000] | 24/24 |
| graph/0/background_fraction | 0.03447 | 0.03461 | 0.00014 | [-0.00010, 0.00042] | 24/24 |
| graph/0/visible_coverage_4mm | 0.45400 | 0.48376 | 0.02976 | [0.00540, 0.05727] | 24/24 |
| graph/0/projected_boundary_coverage_4mm | 0.49068 | 0.53319 | 0.04251 | [0.01315, 0.07901] | 24/24 |
| graph/0/surface_distance_mean_mm | 18.47487 | 18.48023 | 0.00536 | [-0.00514, 0.01467] | 24/24 |
| graph/0/surface_distance_p95_mm | 138.77669 | 138.68432 | -0.09237 | [-0.30097, 0.02422] | 24/24 |
| graph/0/normal_valid_fraction | 0.93339 | 0.93043 | -0.00296 | [-0.00686, 0.00003] | 24/24 |
| graph/0/false_presence | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | 0/24 |
| graph/1/object_recall | 0.98401 | 0.98401 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/missing | 0.00558 | 0.00558 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/correct_fraction | 0.94290 | 0.94290 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/robot_fraction | 0.02277 | 0.02277 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/wrong_object_fraction | 0.00322 | 0.00322 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/background_fraction | 0.03110 | 0.03110 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/visible_coverage_4mm | 0.47981 | 0.47981 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/projected_boundary_coverage_4mm | 0.55712 | 0.55712 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/surface_distance_mean_mm | 153.05456 | 153.05456 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/surface_distance_p95_mm | 422.69338 | 422.69338 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/normal_valid_fraction | 0.81210 | 0.81210 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/false_presence | 0.30060 | 0.30060 | 0.00000 | [0.00000, 0.00000] | 12/24 |

### Internal Test

| Metric | B | C | Δ | Parent bootstrap95% CI | paired Parent |
|---|---:|---:|---:|---|---:|
| mask/base_camera/0/precision | 0.87682 | 0.87682 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/0/recall | 0.94318 | 0.94318 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/1/precision | 0.84019 | 0.84019 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/base_camera/1/recall | 0.92404 | 0.92404 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| mask/hand_camera/0/precision | 0.99494 | 0.99427 | -0.00068 | [-0.00149, -0.00011] | 22/24 |
| mask/hand_camera/0/recall | 0.78814 | 0.96038 | 0.17224 | [0.06497, 0.29767] | 24/24 |
| mask/hand_camera/0/iou | 0.78365 | 0.95486 | 0.17121 | [0.06741, 0.29221] | 24/24 |
| mask/hand_camera/0/visible_object_recall | 0.81806 | 0.99143 | 0.17337 | [0.06674, 0.30339] | 24/24 |
| graph/0/object_recall | 0.99022 | 0.99143 | 0.00121 | [0.00000, 0.00363] | 24/24 |
| graph/0/missing | 0.00930 | 0.00857 | -0.00073 | [-0.00218, 0.00000] | 24/24 |
| graph/0/correct_fraction | 0.94488 | 0.94558 | 0.00071 | [-0.00140, 0.00409] | 24/24 |
| graph/0/robot_fraction | 0.00264 | 0.00323 | 0.00058 | [-0.00001, 0.00148] | 24/24 |
| graph/0/wrong_object_fraction | 0.00953 | 0.00812 | -0.00141 | [-0.00419, 0.00000] | 24/24 |
| graph/0/background_fraction | 0.04295 | 0.04307 | 0.00012 | [-0.00007, 0.00033] | 24/24 |
| graph/0/visible_coverage_4mm | 0.46760 | 0.47984 | 0.01224 | [-0.00519, 0.03609] | 24/24 |
| graph/0/projected_boundary_coverage_4mm | 0.47818 | 0.50225 | 0.02407 | [0.00361, 0.05203] | 24/24 |
| graph/0/surface_distance_mean_mm | 23.15399 | 23.02778 | -0.12621 | [-0.40321, 0.01836] | 24/24 |
| graph/0/surface_distance_p95_mm | 128.30822 | 128.22274 | -0.08548 | [-0.33803, 0.06683] | 24/24 |
| graph/0/normal_valid_fraction | 0.92799 | 0.92579 | -0.00220 | [-0.00545, 0.00138] | 24/24 |
| graph/0/false_presence | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | 0/24 |
| graph/1/object_recall | 0.97071 | 0.97071 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/missing | 0.01966 | 0.01966 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/correct_fraction | 0.95578 | 0.95578 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/robot_fraction | 0.01310 | 0.01310 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/wrong_object_fraction | 0.00097 | 0.00097 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/background_fraction | 0.03016 | 0.03016 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/visible_coverage_4mm | 0.46153 | 0.46153 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/projected_boundary_coverage_4mm | 0.53467 | 0.53467 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/surface_distance_mean_mm | 117.90066 | 117.90066 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/surface_distance_p95_mm | 388.65517 | 388.65517 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/normal_valid_fraction | 0.84722 | 0.84722 | 0.00000 | [0.00000, 0.00000] | 24/24 |
| graph/1/false_presence | 0.00000 | 0.00000 | 0.00000 | [0.00000, 0.00000] | 10/24 |

### Shared Validation

| Metric | B | C | Δ | Parent bootstrap95% CI | paired Parent |
|---|---:|---:|---:|---|---:|
| mask/base_camera/0/precision | 0.91091 | 0.91091 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| mask/base_camera/0/recall | 0.96143 | 0.96143 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| mask/base_camera/1/precision | 0.85951 | 0.85951 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| mask/base_camera/1/recall | 0.92007 | 0.92007 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| mask/hand_camera/0/precision | 0.99789 | 0.99531 | -0.00258 | [-0.00495, -0.00054] | 16/16 |
| mask/hand_camera/0/recall | 0.81093 | 0.95394 | 0.14300 | [0.02481, 0.28137] | 16/16 |
| mask/hand_camera/0/iou | 0.80895 | 0.94935 | 0.14041 | [0.02767, 0.28102] | 16/16 |
| mask/hand_camera/0/visible_object_recall | 0.81726 | 1.00000 | 0.18274 | [0.08862, 0.30529] | 16/16 |
| graph/0/object_recall | 1.00000 | 1.00000 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/0/missing | 0.00000 | 0.00000 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/0/correct_fraction | 0.95459 | 0.95300 | -0.00158 | [-0.00389, -0.00001] | 16/16 |
| graph/0/robot_fraction | 0.00326 | 0.00484 | 0.00158 | [-0.00002, 0.00398] | 16/16 |
| graph/0/wrong_object_fraction | 0.00653 | 0.00653 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/0/background_fraction | 0.03562 | 0.03562 | 0.00001 | [-0.00023, 0.00022] | 16/16 |
| graph/0/visible_coverage_4mm | 0.51197 | 0.50448 | -0.00748 | [-0.02969, 0.01033] | 16/16 |
| graph/0/projected_boundary_coverage_4mm | 0.51842 | 0.52241 | 0.00399 | [-0.01859, 0.02484] | 16/16 |
| graph/0/surface_distance_mean_mm | 6.59043 | 6.60823 | 0.01781 | [-0.00557, 0.05010] | 16/16 |
| graph/0/surface_distance_p95_mm | 25.28374 | 25.61557 | 0.33183 | [-0.00843, 0.82957] | 16/16 |
| graph/0/normal_valid_fraction | 0.94053 | 0.93802 | -0.00251 | [-0.00590, 0.00071] | 16/16 |
| graph/0/false_presence | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | 0/16 |
| graph/1/object_recall | 0.98109 | 0.98109 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/missing | 0.01891 | 0.01891 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/correct_fraction | 0.97615 | 0.97615 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/robot_fraction | 0.00704 | 0.00704 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/wrong_object_fraction | 0.00158 | 0.00158 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/background_fraction | 0.01523 | 0.01523 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/visible_coverage_4mm | 0.43386 | 0.43386 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/projected_boundary_coverage_4mm | 0.52789 | 0.52789 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/surface_distance_mean_mm | 65.06454 | 65.06454 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/surface_distance_p95_mm | 82.33626 | 82.33626 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/normal_valid_fraction | 0.84802 | 0.84802 | 0.00000 | [0.00000, 0.00000] | 16/16 |
| graph/1/false_presence | 0.00000 | 0.00000 | 0.00000 | [0.00000, 0.00000] | 9/16 |

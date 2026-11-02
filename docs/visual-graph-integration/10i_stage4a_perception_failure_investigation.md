# Task10-I Stage4A — Perception 조사 중간 실행 기록

**최종 후속 상태: COMPLETED / NO_GO (2026-11-01).** Development64, Internal96,
Shared Validation48개 source의 전체13,150관측 평가와 Oracle·실패 사례·보호 감사를
완료했다. Production은 변경하지 않았다.
[전체 최종 보고서](10i_stage4a_final_report.md),
[완료 인계 receipt](../../outputs/task10i/stage4a/final_handoff_58576054_v1/completion_audit.json).

아래는 재개 이전과 중간 실행 당시의 기록을 보존한 것이다. 아래의 IN_PROGRESS와
미실행 문구를 현재 최종 상태로 해석하지 않는다.

## 당시 상태 기록

Stage4A 전체 요청은 **IN_PROGRESS**다. Practical Candidate 선정, Internal Test,
Shared Validation 독립 검증과 최종 개선 판정은 아직 완료되지 않았다.
아래는 실제 실행 증거와 남은 작업을 분리한 중간 연구 기록이다.

## 보존·실행 경계

원본 Stage1–3 RGB-D/Action/GT, F1/B/C checkpoint, Task10-H, 기존 split과
Main-visible 판정은 수정하지 않았다. Production perception 기본값도 바꾸지
않았다. 변경은 새 `graph/task10i_stage4a/`와 독립 실행 root에 한정한다.
공통 `ExperimentLogger`를 재사용하고 W&B는 disabled로 로컬 기록한다.
외부 게시와 정책 학습은 실행하지 않았다.

기존 문서10개의 경로·SHA는 [reference inventory](../../outputs/task10i/stage4a/report_reference_inventory.json)에 기록했다.
Stage3 인계 root는 `/workspace/task10i_stage3_20261031_v5_finalization/run`이다.
Stage4 receipt가 지정하는 raw recache manifest/schema SHA와 각 dataset manifest를
검증하고356개 고유 logical source의 raw HDF5/GT sidecar SHA를 전수 검사했다.
R/D 공통236개 logical reference를 두 독립 데이터로 합산하지 않는다.
Validation은 header/schema/hash 확인만 허용하고 개선 선정 전 영상은 열지 않는다.

## 사전 등록

Train Parent를 `sha256('task10i-stage4a-v1:'+parent_id)` 순서로 정렬하여
Development24, Internal Test24, 나머지 Train Reserve로 고정한다.
Shared Validation16 Parent는 최종 후보 동결 후에만 독립 검증한다.
같은 Parent의 corrective branch는 항상 같은 partition에 속한다.
10-G Final/Test, Coverage Holdout60, 정책 Blind Test는 선택에 사용하지 않는다.

후보 채택 기준은 Validation 영상 접근 전에 `protocol.json`에 수치화했다.
Parent-macro visible graph recall 절대 개선≥0.02를 요구하며, pixel precision
손실≤0.005, robot/wrong-object contamination 증가 각각≤0.002,
false object presence 증가≤0.005, Box graph recall 손실≤0.005,
p95 latency ratio≤2.0을 동시에 요구한다. 이 기준의 분모·metric 구현과
positive/negative controls를 검증한 뒤에만 후보 판정에 사용한다.
하나의 가중 합으로 trade-off를 숨기지 않으며 미충족 시 NO_GO 또는
TRADEOFF_UNRESOLVED를 유지한다. 현재 후보 성능 판정은 없다.

## 소스 감사에서 확인한 사항

- `visual.runtime.grounding.GDino`: DINO Tiny, prompt별 forward,
  box/text threshold 기본0.1을 같은 값으로 postprocessing에 전달하고 score
  내림차순 정렬한다. 새 observer는 같은 forward의 전체 query/token logits와
  normalized cxcywh boxes, input token IDs, preprocessing tensor shape,
  postprocessed boxes/scores, 실제 threshold와 RGB hash를 저장한다.
  Model score를 calibrated probability로 해석하지 않는다.
- `visual.runtime.initialization`: Peg prompt `red and white block`,
  Box prompt `cardboard box`, 최대 box area fraction0.25, top5,
  missing alternative score0.3, Peg/Box exclusion IoU0.3을 사용한다.
  초기 누락은5-step scan과 두 scan IoU≥0.5 확인으로 복구를 시도한다.
- `OnlineVisionAdapter`: 초기화된 object는 found에 남고 재탐지하지 않는다.
  두 object가 found이면 scan을 중단한다. 따라서 잘못된 초기화 이후 drift와
  영구 누락은 검증할 가설이다. 아직 실제 주원인으로 확정하지 않았다.
- 실제 geometry 실행 순서는 Depth/Robot·Mask overlap → Wrist gate →
  Shape-first sampling → cross-view identity →3mm dedup→cap128→Graph다.
  개념도와 다른 순서를 실제 source trace에 명시한다.
- Robot filter는 per-shape collision CV-depth agreement 또는 collision volume
  interior를 confirmed로, 별도6mm band를 uncertain으로 분류한다.
  둘 다 제거하며 overlap도 제거한다. Stage trace에서 세 제거를 분리한다.
- 기존 PlanA의 Peg Main+gated Wrist / Box Main-only / current-only / persistent
  OFF를 유지한다. 기존7개 Wrist reject rule와 threshold를 그대로 기록한다.

## Baseline 재현 진행과 실패 보존

원격 instance57987772의 RTX3090에서 CUDA DINO/SAM2를 실행한다.
기존 Development Episode2000003, Parent existing_30080, 전체101 action frame을
보존된 F1 Mask·Graph 캐시와 비교한다. 이 Episode는 새 partition의 Development에
속한다. 일부 정상 프레임만으로 전체 baseline 동등성을 선언하지 않는다.

v2 UUID `add7ee134bf44597a290549a45e277c7`의 strict 비교는 **FAIL**이다.
101프레임 중9프레임에서 mask bytes가 달랐고, Graph는2프레임(t30,t87)에서
달랐다. 로그에 출력한20-frame 간격의 일치만으로 PASS 처리하지 않았다.
Source/config/model을 바꿔 개선했다고 해석하지 않는다.

원래 F1 cache entrypoint는 `torch.set_num_threads(1); torch.manual_seed(42)`를
사용한다. 새 v2가 호출한 training용 `seed_all`은 deterministic algorithms와
TF32 flags까지 변경한다. v3는 원래 cache의 math setting에 맞춰 재검증한다.
v3에서는 원래 설정으로 전체101프레임의 Mask 및 Graph exact 비교가 PASS하여,
이번 재현 차이가 runtime math 설정과 연결됨을 확인했다. 알고리즘 개선 결과가 아니다.
v1/v2 소스·실패 산출물은 보존한다.

v3 UUID `bd4371c6889d4a018b72b4553bc1d671`, root
`/workspace/task10i_stage4a_20261031_v3/run`, tmux `task10i-stage4a`.
Coordinator는 baseline gate PASS 후 render-only feasibility, 검증된 native
Action replay, 지원되는 역사적 캐시의 stagewise 감사까지 순서대로 실행했다.
현재 프로세스는 정상 종료했고 다음 연구 단계는 PENDING_IMPLEMENTATION이다.
전체 Stage4A가 완료되었다는 의미는 아니다.
다음 단계가 아직 구현되지 않은 경우 `PENDING_IMPLEMENTATION`을 기록하며
COMPLETED로 처리하지 않는다. 실행 중 상태와 PID는 pipeline_state와 실제
process/tmux를 함께 확인한다.

## GT reference 확보

Stage3 raw capture에는 RGB-D와 robot/calibration을 저장하지만 actor segmentation
저장은 없다. 전수 schema 검사 결과는 source_data_audit에 보존한다.
분리된 GT sidecar에는 object pose가 있으며, 신형 source에는 physics state도
추가로 있다. 이 정보는 reference 생성과 Oracle 전용이며 student에는 전달하지 않는다.

v1 render-only probe의 기존 source 두 개에서 t0은 RGB/Depth/calibration이
모두 array-equal이었다. 이후6개 sampled frame에서는 RGB의 일부 값이1만큼
달랐다. Depth/calibration이 일치하더라도 **GT_UNSUPPORTED**로 남겼다.
이를 완전한 GT나 perception failure로 계산하지 않는다. 신규 source의 seed가
다른 위치에 있어 v1은 KeyError로 종료했다. 실패를 보존하고 parent_result의
실제 seed를 읽도록 수정했다.

신형 physics_box_pose의 직접 사용, static Box의 불필요한 pose 역변환 회피와
native Expert Action의 실제 replay를 별도 검사한다. Replay는 expert를 재계획하지
않고 저장된 action을 그대로 실행한다. 각 프레임의 양쪽 RGB, raw depth,
intrinsic, extrinsic_cv, cam2world_gl 전체가 정확히 같을 때만 EXACT_GT를
허용한다. 나머지는 지원 불충분으로 보존하며 denominator에서 조용히 제거하지 않는다.

### 실제 Action replay의 전체 프레임 정합성

v3에서 아래 Development source의 모든 관측을 검사했다. 각 관측마다 양쪽
카메라의 RGB/Depth 및 세 calibration field가 모두 array-equal이었다.

| Logical Episode | Source / Branch | EXACT_GT / 관측 |
|---|---|---:|
| 3000100 | 기존 Corrective | 36 / 36 |
| 2000003 | 기존 Full-task | 102 / 102 |
| 10i_stage3_10i_s1_0121_nominal | 신규 Nominal | 42 / 42 |
| 10i_stage3_10i_s1_0121_full_task | 신규 Full-task | 107 / 107 |

총287개 관측이다. 마지막 두 sequence는 같은 Parent이므로 독립 Parent 두 개로
계산하지 않는다. 이 성공은287개 관측의 reference만 지지하며 전체356개
Logical Episode의 재현성이나 실제 TCP disturbance/recovery를 모두 검증한 것은 아니다.
Render-only pose restore의 mismatch 결과는 그대로 남긴다.

## 단계별 instrumentation과 검증 범위

`stagewise.py`는 실제 mask/source pixel indices, filter owner/depth,
confirmed/uncertain/overlap 제거, gate 채택 mask와 reject reasons,
shape-first/association/dedup/budget source support,3D representative와 전체
source association, F5 validity 및 graph tensors를 저장하도록 구현했다.
GT metric은 geometry 호출 완료 후 계산한다. Final source-pixel support는
visible spatial surface coverage와 구분한다. Episode2000003 전체101 action frame에서 instrumentation을 거친 Graph가
역사적 캐시와 exact 일치했다. 다른 source와 terminal frame의 전수 검증은 남아 있다.

로컬 task 계약9개와 기존 online vision/causality/PlanA geometry/BC 계약31개가
PASS했다. [새 계약 로그](../../outputs/task10i/stage4a/local_contract_tests.log),
[기존 계약 로그](../../outputs/task10i/stage4a/existing_contract_tests.log).
이는 synthetic/계약 검증이며 실제 dataset-wide GT 정합성, oracle, temporal
failure recovery, camera isolation, robot false-removal과 최종 정확도 검증을
대신하지 않는다. 원격 frozen v3에는 이후 추가한 로컬 reference helper의2개
계약 보강이 아직 포함되지 않는다. 실행 소스와 로컬 최신 소스를 구분한다.

Linux CPU PhysX/CUDA SAPIEN reference probe와 CUDA DINO/SAM2를 실행했다.
이번 Stage4A에서 macOS simulator/MPS, GPU PhysX, 실시간20Hz는 검증하지 않았다.
Observer의 raw-output 저장 overhead가 포함된 시간은 deployment latency로
보고하지 않는다. 최종 후보는 observer 비용을 분리해 latency를 다시 측정해야 한다.

## 남은 전체 요구사항

Baseline 완전 재현과 reference 유효성 확정, 모든 필요한 phase/branch를 포함한
failure collection, stagewise quantitative attribution, raw proposal 감사,
Mask identity/ghost/reappearance/flicker/fragmentation,3D purity·거리·coverage·rim,
Episode/Parent macro와 paired 통계, O0–O4 causal oracle, practical 후보의
분리 실험 및 positive/negative controls, 후보 freeze→Internal Test→최종 freeze→
전체 Shared Validation paired 평가가 남아 있다. 최종 보고·artifact handoff와
요구사항별 completion audit 이전에는 목표를 완료 처리하지 않는다.

## 최초 정량 진단 — Development Episode2000003 한 개

[원시 단계별 rows](../../outputs/task10i/stage4a/v3/stagewise_development_rows.json),
[집계](../../outputs/task10i/stage4a/v3/preliminary_stage_metrics.json),
[Baseline exact101-frame 증거](../../outputs/task10i/stage4a/v3/baseline_forward_equivalence.json),
[Replay 정합성](../../outputs/task10i/stage4a/v3/replay_reference_probe.json).
이 표는 GT-visible support≥8pixel인 관측만의 pooled pixel/source 통계다.
Episode/Parent macro 전체 모집단 결과나3D surface coverage가 아니다.

| Camera / Object | 지원 Frame | SAM2 Precision / Recall | Filter+Overlap 후 Precision / Recall | Final graph source Precision |
|---|---:|---:|---:|---:|
| Main Peg | 101 | 0.8520 / 0.9618 | 0.9682 / 0.9567 | 0.8825 |
| Main Box | 86 | 0.9691 / 0.9799 | 0.9932 / 0.9793 | 0.9792 |
| Wrist Peg | 101 | 0.9995 / 0.9906 | 0.9995 / 0.9879 | 0.9993 |
| Wrist Box | 66 | 0 / 0 | 0 / 0 | 미정의: PlanA 제외 |

Main Peg SAM2 mask의 robot pixel은 대상 frame 합계3767개에서 filter/overlap 후
106개로 감소했다. 같은 단계의 background682pixel은 남았고, sampling 이후
graph source에도678pixel이 남았다. 정상 dense 표면이 budget에서 줄어드는 동안
background source가 강하게 유지되어 precision0.9682가 final source에서는
0.8825로 낮아졌다. Sampling 전후 오염 비율을 조사할 근거이며, 전체 데이터의
주원인이나3D 거리 악화를 증명한 결과는 아니다.

Wrist Box는 GT-visible66frame에서 정상 pixel을 하나도 찾지 못했고, SAM2는
robot4402pixel과 background54pixel을 출력했다. Robot filter 뒤에도 background54개가
남았지만 Main-only 계약으로 최종 Graph 혼입은0이었다. Wrist Box를 무조건
활성화하지 않을 근거다. Main Box의 GT-visible support가8pixel 미만인15frame도
원시 rows에 남겼으며, 작은 support/비가시 통계는 후속으로 별도 집계한다.

다음은 같은 exact-reference/raw에서 O3 GT-visible Mask와 O4 Wrist Gate 제거의
**ORACLE_DIAGNOSTIC**으로 mask와 downstream 기여를 분리한다. 이 조건은
실용 후보가 아니며 GT 입력은 진단에만 사용한다.

동일 Episode의 Main Box에서 충분한 GT-visible support≥8pixel인86frame 중
최종 Graph source가 비는 경우가1회 있었다. GT Box support가0인10frame 중
5frame에서는 Box source가 남았다. 단순 Graph 비어 있음만으로 detection failure를
계산하지 않고, 이 두 종류를 분리해 후속 실패 gallery와 temporal trace를 조사한다.
Wrist Box의 미사용은 이 missed-visible-object 집계의 practical 실패로 계산하지 않는다.

Oracle 후속 root는 `/workspace/task10i_stage4a_20261031_v4/run`이다.
v3의 baseline exact 증거와 dependency source byte equality를 먼저 검사하고,
같은 Raw/Depth/EXACT_GT에서 O3와 O4를 자동 실행하는 별도 tmux runner다.
새 Practical Candidate를 선택하거나 Production Gate를 끄는 실행이 아니다.

[구체적인 실패 trace](../../outputs/task10i/stage4a/v3/preliminary_failure_cases.json):
Main Box의 t54는 GT-visible임에도 SAM2 raw mask부터0이었다. 따라서 이 한 번의
Graph 누락은 depth/filter/gate/sampling에서 처음 사라진 사례가 아니다.
반대로 t17–20,t22는 GT-visible Box pixel이0인데 SAM2 mask29–32pixel이 있었고,
robot filter 뒤에도1–3pixel이 남아 최종 Box source가 되었다. 이는 비가시와
잘못된 잔여 source presence를 분리해야 함을 실제로 보여준다. 시각 gallery와
전체 temporal segment 검증 전에는 이 trace만으로 identity switch를 선언하지 않는다.

최근 실제 process 확인: Oracle UUID `7639157877c443798179294c003cf50e`,
PID227425가 O3 Episode2000003 t75까지 진행 중이었다. tmux pane PID227423도
존재했다. SSH 관측 중 일시적인 banner timeout이 있었지만 실행을 중복 시작하지 않았다.
다음 continuation은 이 PID/tmux와 pipeline_state를 먼저 재확인한다.

## O3/O4 최초 원인 분리 결과

Oracle substage UUID `7639157877c443798179294c003cf50e`는 정상 종료했다.
[O3/O4 원시 rows](../../outputs/task10i/stage4a/v4/oracle_rows.json),
[요약](../../outputs/task10i/stage4a/v4/oracle_preliminary_metrics.json).
동일 Episode2000003의 충분한 visible support≥8pixel인 frame을 비교한 결과다.

| 조건 | Main Peg final source precision | Wrist Peg final source precision | Main Box final source precision |
|---|---:|---:|---:|
| O0 Baseline | 0.8825 | 0.9993 | 0.9792 |
| O3 GT-visible Mask | 1.0000 | 1.0000 | 1.0000 |
| O4 Wrist Gate 제거 진단 | 0.8812 | 0.9825 | 0.9792 |

O3의 GT Peg 픽셀은 filter+overlap 후 Main99.495%, Wrist99.735%가 남았다.
Box는 해당 지원 frame에서100%가 남았다. 이 Episode 전체 평균에서는
filter가 정상 표면 대부분을 지우는 현상은 주된 설명이 아니며, 기존 mask의
오염과 sampling 뒤의 오염 비중 상승을 더 조사할 근거다. Finger-near 조건과
다른 Episode는 별도 분석해야 한다. O4는 Wrist source precision이 낮아져
Gate 제거를 실제 개선안으로 채택할 근거를 주지 못했다.
Pixel/source recall과3D spatial coverage는 여기서도 구분한다. 이 Oracle 결과는
새 RGB-D perception 개선 성공이 아니며 SIM_ONLY_ORACLE 진단이다.

## 실제 RGB/GT/Mask/Graph Source의 시각 확인

원본 HDF5와 보존 Graph NPZ의 SHA를 다시 확인하고 exact replay reference로
GT overlay를 만들었다. t0·t17·t54·t87의 양쪽 카메라를 실제 이미지로 열어 확인했다.
[Main gallery](../../outputs/task10i/stage4a/gallery/base_camera.png),
[Wrist gallery](../../outputs/task10i/stage4a/gallery/hand_camera.png),
[Gallery provenance/counts](../../outputs/task10i/stage4a/gallery/gallery_identity.json).

Main t17에서 Box는 로봇에 완전히 가려져 GT0pixel이지만 SAM232pixel과
최종 source3pixel이 남았다. Main t54는 부분 가림 뒤 Box가13pixel 보이는
작은-support 사례인데 SAM2가0pixel이었다. 이를 fully-visible 실패와 같은
stratum으로만 보고하지 않는다. Wrist t0의 Box prompt는 실제 Box 대신
손가락의 일부를 잡았고, 이후 t17에도 같은 부근 mask가 남는 모습을 확인했다.
현재 Main-only Box 계약이 이 경로의 Graph 오염을 차단했다.

## 전체 Development 감사와 Reference 복원 보강 — 실행 중

사전 등록 Development24 Parent는 중복 제거 후64 Sequence·4,162개 관측이다.
기존 Full-task8, 기존 Corrective8, 신규 Full-task16, 신규 Nominal/Corrective32다.
최종 t=T 관측도 제외하지 않는다. 신규 Corrective32 전체를 모두 disturbance로
표현하지 않으며, branch/segment별 분리를 유지한다.

독립 root `/workspace/task10i_stage4a_20261031_v5/run`, UUID
`d61ada17e04e422eb4509fd9caed9c5b`에서 계약 시험과 실제3-frame engineering
reference/perception smoke를 통과한 뒤 전체64개 reference 검사를 마쳤다.
현재 Causal DINO/SAM2 producer와 독립 CPU geometry worker3개가 동일 입력의
Mask, 단계별 trace, 대표3D point 오염·mesh 거리·4mm visible coverage·투영 boundary
coverage를 수집한다. 이 boundary 지표를 물리적 hole rim coverage로 부르지 않는다.
거리 기준은10,000개 deterministic 외부 collision surface sample이므로 exact
analytic mesh distance가 아니다. Geometry 상태와 output은 원래 데이터와 분리한다.

일부 Corrective는 원래 수집 중 anchor snapshot을 복원한 뒤 disturbance를
시작했다. 단순 reset→action replay는 이 discontinuity를 재현하지 못해 prefix 이후
GT_UNSUPPORTED가 남았다. 원래 `anchor_t10.pt`/`anchor_t25.pt`와 collector의
동일 restore를 사용하는 새 진단 root `/workspace/task10i_stage4a_20261031_v6/run`
(UUID `24f9768250034a82870620e77b499dc5`)를 분리했다. 최초 실제 branch
`10i_stage3_10i_s1_0810_t10_-10_0`은 exact 지원이12/59에서59/59로 늘었다.
원래 reference를 수정하거나 RGB tolerance를 완화하지 않았다. 기존 F1 paired
full-task의 비물리적 teleport/readback/restore 순서도 별도 복원 검사한다.
이 개선은 평가 reference 확보이며 student perception 개선으로 세지 않는다.

로컬 Parent-macro 집계 시험2개를 추가해 frame/branch 수에 따른 Parent 과가중과
unsupported frame 누락을 검증했다. [11개 task 계약 결과](../../outputs/task10i/stage4a/aggregation_contract_tests.log).
새2개 시험은 frozen v5 시작 이후 추가했으므로 원격 v5의 시험 수에 합산하지 않는다.
Validation 영상, Internal Test 결과, Practical Candidate 선정은 아직 열지 않았다.

### Development Reference의 최종 지원 확보

전체64 Sequence의 단순 Action replay는3,420/4,162개 관측이 exact였다.
수집 discontinuity가 있는18개 Sequence(같은 신규 Parent의16개 TCP Branch와
기존 Paired Full-task2개)를 원래 snapshot/이동 순서로 복원한 결과,
해당 지원이359/1,101에서1,101/1,101로 늘었다. 따라서 원래 증거와 수정된
별도 reference를 frame/episode별로 결합하면 Development의 **4,162/4,162개
관측 전부**가 양쪽 RGB·Depth·calibration array equality로 지원된다.

[원래 전수 Reference 결과](../../outputs/task10i/stage4a/v5/development_reference_summary.json),
[수집 순서 복원 결과](../../outputs/task10i/stage4a/v6/reference_repair_summary.json),
[분모 결합 검산](../../outputs/task10i/stage4a/v6/combined_reference_support.json).
기존 reference를 덮어쓰지 않았고 임계값도 완화하지 않았다. 이4,162개는
Development만의 수이며 Internal Test/Shared Validation의 지원 수로 확대하지 않는다.

분기 수에 따른 독립성 한계도 명시한다. 이번 고정 Development는 Full-task24개,
Nominal24개, 실제 TCP Branch16개이며, TCP Branch16개는 전부 Parent
`10i_s1_0810`에 속한다. Recovery의16개 결과를16개의 독립 Parent로 계산하지 않는다.

원본 baseline의 mask/graph가 바뀌지 않도록 수정 Reference 재평가는 새 root
`/workspace/task10i_stage4a_20261031_v7/run`, UUID
`8f14ba0ad4dd44688997ca92e0159324`에 예약했다. 이 runner는 v5의 완료 receipt를
자동 대기한 뒤 저장된 stage mask와 representative points에만 새 GT를 대응한다.
DINO/SAM2/Graph forward는0회이고, 이전 EXACT_GT frame의 metric이 달라지면
실패하도록 검사한다. 원래 v5 결과와 수정 Reference의 v7 결과를 구분한다.

최근 process 확인에서 v5 coordinator PID227775와 perception PID229296이
살아 있었고64개 중14개 Sequence의 baseline 감사가 완료됐다. v7 PID230621은
정상적인 prerequisite 대기 중이었다. Shared Validation과 Practical Candidate
선정은 여전히 미실행이며 전체 Stage4A 목표는 계속 IN_PROGRESS다.

## 추가 원인 증거와 Practical 재초기화 후보 — 아직 동결·채택하지 않음

완료된 Development sequence 중 `10i_s1_0362`의 Wrist Peg는 t0에 GT848pixel이
있는데 SAM2가 다른 영역417pixel을 내고, 이후 정상 Peg가 보이는92frame 동안
정상 지원이0이었다. t0의 잘못된417pixel은 원래 Gate가 Main Box와의 identity
conflict로 올바르게 거부했다. 이 사례를 Gate의 false rejection으로 설명하지 않는다.
`10i_s1_0666`과`10i_s1_0810`도 Wrist Peg raw mask 단계에서 각각82/102개의
충분한 GT-visible frame이 누락됐다. Main Peg가 있으므로 이를 곧바로 fused
Graph 전체의 object absence로 계산하지 않는다.

DINO 원시 proposal의 t0 대조에서0362에는 Peg bbox IoU0.504인 score0.184
proposal이 있었다.0666에는 IoU0.957/score0.200,0810에는 IoU0.582/score0.254
proposal이 있었다. API threshold0.1 아래라서 보이지 않았던 것이 아니다.
기존 missing 대안 score0.3과 camera 공통25% box area 제한이 선택에 영향을 준다.
특히 가까운 Wrist 화면의 실제 Peg bbox가25%를 넘는 경우가 있다.
0362의 더 높은0.323 proposal은 GT Peg bbox와 IoU0이었다. 초기 오선택이
found 상태로 남고 재탐지를 중단하는 memory 정책과 결합한 실제 실패 사례다.

설치된`GroundingDinoProcessor.post_process_grounded_object_detection`도 직접
확인했다. Box keep은 max token sigmoid score > box threshold이며,
text threshold는 decoded text label에만 영향을 준다. 현재 GDino wrapper는
그 text label을 버리므로 text threshold만 바꾸는 것은 현재 box 선택 개선이 아니다.
Prompt·Box Threshold·Text Threshold를 같은 변수로 취급하지 않는다.

### V1 Development prefix smoke의 개선과 악화

새`WristPegReanchor`는 현재 RGB-D/proprioception만 받고, 기존 Wrist Gate 실패 시
5-frame cadence에서 현재 DINO top8 proposal을 검토한다. Main clean Peg가 Wrist의
현재 depth와4mm 이내로 일치하는 witness≥6pixel과 box/mask witness coverage≥0.6을
요구한다. 해당 검증을 거쳐 낮은 score/큰 box도 재초기화 후보로 평가한다.
원래 Wrist Gate7개 조건, Peg Main+Wrist / Box Main-only, current-only, cap128은 유지한다.
SAM2 상태는 그 시점에 실제 관측한 한 장에서 시작하며, 이후 연속적인 local clock으로
propagate한다. GT segmentation/pose/phase/future trigger는 받지 않는다.

V1 root`/workspace/task10i_stage4a_20261031_v9/run`, UUID
`bfca28b2998b442aa105397c3251bd87`의 첫20-frame causal prefix3개 smoke가 완료됐다.
[실행 설정](../../outputs/task10i/stage4a/v9/candidate_spec.json),
[결과](../../outputs/task10i/stage4a/v9/smoke_summary.json).

| Sequence | Baseline 정상 Peg pixel 합 | V1 정상 Peg pixel 합 | Baseline FP pixel 합 | V1 FP pixel 합 |
|---|---:|---:|---:|---:|
| 0362 full-task prefix | 0 | 33,095 | 2,712 | 21,050 |
| 0666 full-task prefix | 0 | 74,049 | 0 | 6 |
| 기존2000003 정상 control | 38,454 | 38,454 | 156 | 156 |

0362의 V1 FP는 Robot19,277 + Background1,773pixel이다. 기존 오류는 Wrong-object2,712pixel이었다.
정상 Peg 복구만으로 이 후보를 채택하지 않는다. Raw SAM2 오염과 downstream
filter 후 오염을 구분해야 하며, 현재 V1은 정밀도·오염 trade-off가 해결되지 않은
feasibility 결과다. 이3개 prefix를 전체 Episode/Parent 결과나 독립 Test로 세지 않는다.
Correction-only 평균시간은 정상 control약126ms, 두 복구 사례약561/578ms였다.
다른 Baseline 실행과 GPU를 공유한 개발 시점이고 Geometry 전체 시간이 아니므로
final latency나20Hz 처리 가능성으로 해석하지 않는다.

V2는 현재 robot filter와, Main에서도 실제 depth가 같은 표면 중 Main Peg mask와
일치하지 않는 부분을 정제한다. Main에서 실제 가려지거나 보이지 않는 complementary
Wrist 표면은 이 이유만으로 삭제하지 않는다. Raw SAM2 mask와 정제 mask를 따로 저장하고,
raw valid-depth/clean-fraction guard도 별도로 유지해 정제 전 robot-dominated mask의
원래 reject rule를 우회하지 않도록 했다. Local 계약 시험은 통과했으나, 아래 저장 공간
장애로 V2 실제 GPU smoke는 실행되지 않았다. Production에는 적용하지 않았다.

## 현재 실행 중단 — 원격 저장 공간 및 Host 접근 장애

**전체 Stage4A는 INCOMPLETE이며 과학적 최종 판정은 아직 없다.**
원격 overlay16GB가16GB 사용 상태(여유140KB)가 되면서 새 V2 source tar 해제가
실패했다. 기존 v5 runner도`SAM2 JPEG write failed`와 Logger metadata의
`OSError28 No space left on device`로 실제 종료됐다. 마지막 정상 완료 기록은
**60/64 Sequence**, 마지막 Episode`3000120`이었다.60개 전체 산출물의 최종 복구
검증은 접근 회복 후 다시 해야 한다. 전체4,162개 GT Reference 확보와 Baseline
전체 완료는 다른 상태이며, Baseline 완료로 오기하지 않는다.

이어 Vast API가 instance57987772를`actual_status=offline`, `cur_state=running`으로
보고했고 direct/proxy SSH가 거부됐다. 디스크 문제와 host offline 사이의 인과관계는
검증하지 못했다. 플랫폼의 read-only execute 대안도 running instance에는 허용되지
않는다는 오류를 반환했다. 기존 Task10-H와 다른 작업을 방해하지 않도록 Stop/Reboot/
Recycle/Destroy 또는 원본 삭제를 실행하지 않았다. 원격 정리·백업 성공도 주장하지 않는다.
[별도 장애 증거와 복구 조건](../../outputs/task10i/stage4a/storage_failure_receipt.json).

v7의 수정 Reference 재평가, v8의 Detector/Temporal/Phase 감사(UUID
`86ed952060a149ef85354077e34282a1`), v10의 Robot/Gate/Graph 독립 감사(UUID
`4c97cc6818314672a0b322fd9130d825`)는 완료되지 않았다. 장애 직전에는 prerequisite
대기 PID230621/230885/231223이 살아 있었지만 현재 host 접근이 없어 생존 여부를
확인할 수 없다. 상태 파일이 RUNNING이라는 이유로 실행 중이라고 단정하지 않는다.
접근이 돌아오면 같은 handle을 확인한 뒤에만 재개한다.

복구용`resume_baseline.py`를 준비했다. 새 Logger UUID와 독립 root에서 기존 완료
Receipt·Mask·stage rows의 SHA를 다시 검증하고 재사용하며, 미완료4개만 episode 시작부터
새 causal memory로 실행한다. 중간에 실패한 Sequence의 memory를 추정해 이어 붙이지 않는다.
태스크 출력 JSON의 무손실gzip fallback은 압축 전 logical bytes의 SHA를 유지하도록
시험했다. 실제 원격 압축/정리나 복구 실행은 아직 하지 않았다. 공간 확인 없이 큰 감사를
시작한 운영상의 부족을 복구 단계의 disk headroom gate로 보강했다.

[로컬29개 계약 시험](../../outputs/task10i/stage4a/storage_contract_tests.log)은 압축 SHA,
Parent macro, temporal/coordinate, missing/false-positive control과 causal reanchor reset을
포함한다. 이 시험을 CUDA V2나 전체 연구 완료의 증거로 사용하지 않는다.
Internal Test, Shared Validation, 최종 Candidate freeze와 final handoff는 계속 미완료다.

## 58576054 이전 후 재개 가능성 검사 — 2026-11-01

새 Host에서 완료60개 Sequence·4,026개 Graph frame 및 기존4,162개 EXACT_GT Reference의
보존을 다시 확인했다. Raw/GT712개와 고정 Source672개 SHA도 일치했다. 남은 Baseline은
4개 Sequence·136개 관측이다. 다만 새 기본 환경은 cu132이며 연구 의존 package가 없어,
환경 복원과 새 Host의 Baseline Forward Equivalence가 선행되어야 한다. 이번에는 검사만
진행했고 추론이나 평가 runner를 재개하지 않았다. 전체 과제의 완료 판정은 바꾸지 않는다.

[58576054 재개 검사 전체 보고서](10i_stage4a_resumption_check_58576054.md).

## 58576054 독립 cu130 환경 복원 — 2026-11-01

사용자 요청으로 `/workspace/venvs/task10i-stage4a-cu130`을 만들고 이전 버전 pin을
설치했다. 원본의132개 명시 버전이 모두 일치하고 CUDA 연산·cuDNN·torchvision NMS,
주요 패키지 import와 SAM2 YAML 검증이 통과했다. 기본 `/venv/main`의 패키지는
변경되지 않았다. 실제 simulator/renderer, 모델 forward와 Baseline Forward Equivalence,
연구 runner는 아직 실행하지 않았다. 전체 과제와 남은 Baseline4개의 완료 상태는 그대로다.

[cu130 환경 복원 전체 보고서](10i_stage4a_environment_restore_58576054.md).

## 58576054 기존 Mask/Graph Forward Equivalence — 2026-11-01

사용자 요청으로 기존 재현 fixture2000003의101개 action frame을 새 cu130 환경에서
다시 계산했다. 원래 F1 캐시와 이전 v5 Stage4A 출력 양쪽에 대해 Mask404개와 Graph
Tensor505개가 모두 bytewise 동일했다. Main/Wrist Peg/Box Pixel 차이0, Graph
차이0, 기존 Stage4A Wrist Gate101개도 동일하다. 새 UUID는
`813d4b24cd414cbfb327e25145473372`이며 기존 Source672개 SHA를 실행 전후 확인했다.
계약 시험39개 PASS·1개 SKIP, actual NPZ101개의 독립 readback도 PASS다.

검증 범위는 이 단일 Development fixture의 공통 action frame이며 terminal은 제외다.
전체 데이터 Forward Equivalence나 Perception 개선 결과로 확대 해석하지 않는다.
남은4개 Sequence·136관측 처리는 시작하지 않았고 전체 Stage4A도 계속 미완료다.
[Forward Equivalence 전체 보고서](10i_stage4a_forward_equivalence_58576054.md).

## 전체 Goal 재개 — 58576054, 2026-11-01

사용자의 재개 요청에 따라 남은4개를 t0부터 재실행했다. 새 Baseline root는
`/workspace/task10i_stage4a_completion_20261101_v1/run/baseline`, UUID는
`09f016cc9b4f461c9ca213c49532334a`다. **64개 Sequence·4,162개 관측**을 완료했고,
완료60개의 원본 Receipt/Mask/Graph는 그대로 재사용했다. 수정 GT 기준 재평가도
4,162개 전부 EXACT_GT로 끝났다. Detector/Temporal/Phase와 Robot/Gate/Graph 감사도
동일 root의 `extended`, `filters`, `corrected`에서 완료됐다.

새 Source는 동결된 기존 production 의존 파일을 유지하고 task helper만 추가했다.
원래 RGB-D/Action/GT, Stage1–3 Dataset, Checkpoint, split, main-visible 판정과
Task10-H는 변경하지 않았다. 아래 수치는 Development 진단이며 독립 검증 결과가 아니다.

### 새 정량 증거

- 실제4,162개 Wrist Gate 기록에서 `good_2d`2,392개는 모두 통과했다.
  upstream Mask가 없는1,670개와 `bad_2d`16개는 거부됐고, `ambiguous_2d`84개는
  통과했다. `good_2d`는 정답 Peg≥6pixel, precision≥0.9인 진단 정의다.
  이 결과에서 Gate의 정상 Mask 과거부를 주원인으로 선정하지 않는다.
- Robot Filter는 전체 GT Peg 표면에서는 손실이 작지만, 관측된 finger-pad 중심의
  20mm 이내에서는 Main2,853/17,133pixel(16.65%), Wrist4,835/11,636pixel(41.55%)을
  제거했다. 이는 contact force 판정이 아닌 기하학적 근접 구간이며, SAM2가 실제로
  포함한 표면과 모든 GT 표면의 분모를 구분한다. 파지 부위 손실은 미해결 한계다.
- 실제 Main Peg Mask에 포함된 정답 pixel은 Robot·Overlap 처리 후2,158,912개였다.
  최종 Graph source에는374,790개가 남았다. 같은 과정에서 Background FP는
  19,799개 중18,076개가 남았다. 이 pooled source-pixel 수는3D coverage가 아니며,
  구조적 sampling이 소수의 잘못된 경계점을 강하게 보존할 수 있음을 보여준다.
- 현재 camera source로부터의 backprojection 검사는4,162개 frame에서 최대
  약2.09µm였고 foreign/history source는0이었다. 정답 Actor에서 온 점의 sampled
  surface distance는 Peg median1.06mm, Box median3.03mm였다. Box의 큰 거리 tail은
  주로 Background에서 왔다. 고정 t0/중간/terminal 표본에서 Box Background 평균은
  약4.38m였다. Box3mm에는10,000개 mesh sample의 근사 오차가 포함된다.

### 실행 중인 후보와 독립 검증 절차

현재 세 후보를 같은 Development64개에서 비교한다. Wrist Peg reanchor V2,
Main Peg 깊이 성분 정제, 동일 정제를 Main Box까지 확장한 object-scope ablation이다.
Root-cause 증거를 보고 순차 추가했으며, 모든 실제 추론은 현재 RGB-D/proprioception만
받는다. 기존7개 Wrist Gate, Box Main-only, current-only, cap128은 유지한다.

같은 현재 frame의 Robot Filter/Gate 중복 계산은 입력 전체의 byte hash를 키로
재사용한다. 이전 frame/episode 결과는 지우며 surface map을 만들지 않는다.
빈/이동/원래 Mask를 사용한27개 실제 frame/Mask 조건에서54개 cached Graph 비교,
270개 Tensor 비교와 Gate가 native 출력에 정확히 일치했다. 최초 계산은 기존
`PlanA.prepare` bytecode를 그대로 실행한다.

또한 새 호스트의 CPU quota 약6.47코어에서 BLAS thread 경쟁을 줄이기 위해
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`을 명시했다.
이 설정에서도 기존101개 action frame 전체의 Mask/Graph 재현 검증이 통과했다.
저장된 Mask를 사용한 추가8개 Geometry 비교에서도 Graph가 bytewise 동일했다.
통제한8개 frame의 Geometry 평균은 약0.557초→0.315초였다.
최종 latency는 observer를 제거한 별도 online 실행으로 다시 측정한다. 이 값으로
20Hz 가능성을 주장하지 않는다. 이전 느린/미완료 root는 보존했고, 완료된 Episode만
SHA 검증 후 재사용하며 partial은 t0에서 다시 시작했다.

현재 자동 coordinator는
`/workspace/task10i_stage4a_finalization_20261101_v4/run`이다. 세 Development 비교,
O0–O4 Oracle, observer 없는 latency, 후보 Source/Config freeze, 별도 Train24 Parent
Internal Test, 최종 freeze, Shared Validation16 Parent의 모든48개 logical source를
순서대로 실행한다. Validation 결과로 설정을 수정하지 않는다. 별도 posthoc runner가
matched-parent 통계, FP node에 coverage credit을 주지 않는 표면 평가, 실패 gallery,
원본 보호·causal source 검사를 준비한다. **아직 전체 Goal 완료나 후보 GO 판정이 아니다.**

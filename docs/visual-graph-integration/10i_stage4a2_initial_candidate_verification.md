# Task 10-I Stage 4A-2 — Initial Candidate Verification

2026-11-02. **최종 판정: `NO_SAFE_RULE`. `PAUSED_SCOPE_REVIEW` 유지.**

t0에서 폐기된 원래 DINO 후보는 Temporary SAM2에서 정상 Peg 마스크를 생성했다.
t30에 승인된 원래 후보는 다시 Box를 분할했다. 두 출력은 모두 높은 SAM2 Quality와
연결된 비로봇 표면을 갖는다. 관측 형상·색·과거 Box 표면 일치라는 구분 신호는 확인했지만,
이를 안전한 하나의 ACCEPT/REJECT 규칙으로 확정하지 않았다. Threshold 완화·GT 기반
후보 재검색·Production Patch는 없다. 초기 인식 문제 해결이나 Production GO를 뜻하지 않는다.

## 동결 입력과 실행 계약

선행 Ownership/Surgical/Paused 보고서와 Source/Artifact Manifest를 확인했다.
Vast.ai **58576054 / RTX 3090**에서 저장 자료만 읽었다. Raw는
`/workspace/task10i_stage3_20261031_v3/run/parents/{10i_s1_0355,10i_s1_0645}/full_task/episode.h5`,
B0 Proposal은 `/workspace/task10i_stage4a2_newdev_A_20261101_v1/run/development/dino_raw/`,
동결 BC Source·Mask·State는 `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/`,
평가용 EXACT_GT는 `/workspace/task10i_stage4a2_newdev_reference_20261101_v1/run/references/`다.
상세 경로·전체 SHA·실행 Script는 [verification.json](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/verification.json)에 보존했다.

| 입력 | 고정 Candidate ID: call / rank / query | 원래 Score | 원래 Consumer |
|---|---|---:|---|
| P: 0355/t0 Main | 133 / 1 / 3 | .2916409373283386 | Null .3 미달 → 폐기 |
| N: 0355/t30 Main | 153 / 0 / 0 | .4480925500392914 | t25와 IoU .995154399 → 늦은 초기화 승인 |
| C1: 0645/t0 Main | 333 / 0 / 0 | .4812784194946289 | 정상 Joint 초기화 승인 |

P Box는 `[18.808135986328125,68.30874633789062,65.5214614868164,93.42169189453125]`,
N Box는 `[40.50697326660156,39.060646057128906,74.56087493896484,71.43675994873047]`다.
P는 이전 감사에서 특정된 Score/Rank/Query를 그대로 선택했고 GT IoU로 검색하지 않았다.
C1은 이전 두 보고서에 있던 정상 Parent의 실제 t0 초기화 후보로 추론 전에 동결했다.
C2는 **`CONTROL_UNAVAILABLE`**다. 검토한 제한 자료에는 근접·가림 Ownership 안전성을
확인할 적합한 원래 초기화 Prompt가 특정돼 있지 않았다. 이전 0645/t6·t86 대조도
그 안전성을 보장하지 않는다. 전체 저장 자료에 사례가 없다는 주장은 아니다.

SAM 2.1 Hiera Small, 기존 Checkpoint SHA
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`와
설치 Config/Processor를 유지했다. 기존 OpenCV RGB→BGR JPEG 저장과 설치 SAM2의
PIL Resize1024·ImageNet 정규화 경로를 사용했다. 각 현재 RGB만 독립 디렉터리의
frame0으로 재인덱싱하고, 새 State에 Box를 **한 번** 전달한 즉시 출력만 읽었다.
원래 시점은 Provenance에 남겼다. 과거/미래 RGB나 기존 Tracker State를 전달하지 않았다.
Preflight·Propagation·Memory Encoding·Production Commit 없이 State를 폐기했다.
**세 마스크와 GT-free 증거를 저장한 뒤에만 GT Actor Mask를 열었다.**

## 실제 마스크와 사후 평가

| 입력 | Pixel | Peg TP | Box FP | Robot FP | Background FP | Peg Precision / Recall / IoU | SAM Quality / Presence Logit |
|---|---:|---:|---:|---:|---:|---|---|
| P | 542 | 526 | 0 | 0 | 16 | .970480 / .992453 / **.963370** | .954184 / 24.063843 |
| N | 898 | 2 | 896 | 0 | 0 | .002227 / .004608 / .001504 | **.982003 / 24.339392** |
| C1 | 840 | 840 | 0 | 0 | 0 | 1.000000 / .989399 / .989399 | .972443 / 24.664948 |

P/N/C1의 실제 DINO Rectangle Actor 구성(Peg/Box/Robot/Background)은 각각
**530/9/0/636**, **33/899/24/132**, **849/0/0/971**이다.
Rectangle은 기존 감사와 같은 정수 Pixel 좌표 포함 조건을 썼다.
N의 Temporary Mask는 **99.7773%가 GT Box**, GT Box와 Mask IoU .994451이다.
N과 C1의 Binary Mask는 기존 B0 Peg Slot과 **byte-equal**이다.
P의 기존 Peg Mask는 비어 있었으므로 동일성 비교의 성공으로 세지 않는다.

- **Q1: 그렇다.** P는 GT Peg530 중526 Pixel을 회수했고 Robot/Box FP는0이다.
  누락4·배경16이 있으므로 완벽한 마스크라는 뜻은 아니다.
- **Q2: 그렇다.** N은 처음부터 대부분 Box를 분할한다.
- **Q3: Quality만으로 구분할 수 없다.** 이 국소 비교에서는 잘못된 N의 Quality와
  Presence가 올바른 P보다 높다. 설치 Decoder의 IoU 추정·Prompted Instance Presence를
  Peg의 Semantic Probability로 해석하지 않는다. 모든 가능한 Quality 변환을 반증한
  실험은 아니지만, Quality가 높다는 승인 근거는 이 두 사례에서 성립하지 않는다.

## 관측으로 확인한 구분 신호와 한계

모든 마스크 Pixel의 Depth가 기존 signed-int16 양수 규칙에서 유효했다.
양수라고 모두 정상 Depth라는 뜻은 아니다. 일부 표면 경계·소수 분리 성분은 큰
깊이 차이를 갖고, 저장 Box 표본 일부는26.662m까지 있어 전체 PCA Extent를 물체 크기로
사용하지 않았다. 아래는 기존 감사의 **4-neighbor·인접3D 거리10mm 미만** 연결 규약에서
가장 큰 성분의 관측치다. 새 Semantic Threshold가 아니다.

| GT-free 신호 | P | N | C1 |
|---|---:|---:|---:|
| 최대 연결 표면 Pixel / 전체 | 517/542 | 810/898 | 826/840 |
| 최대 표면 PCA Extent, m | .2111 / .0642 / .0370 | .2764 / .2808 / .0481 | .2038 / .0687 / .0383 |
| 최대 표면 Plane RMS, mm | 8.63 | 4.29 | 9.70 |
| 2D PCA 축 비율 | 3.526 | 1.060 | 2.768 |
| Robot FK 충돌 / 불확실 Pixel | 0 / 0 | 0 / 0 | 0 / 0 |
| B0 Box Mask와 후보 겹침 비율 | 0 | .989978 | 0 |
| BC Box Mask와 후보 겹침 비율 | 0 | 1.000000 | 0 |

P는 지배적인 하나의 표면을 갖고 현재 Box와 분리돼 있다. N도 지배적인 연결 표면이며
더 평면적이다. Robot 분리는 가능하지만 연결성·비로봇·평면성 자체는 의미 판별이 아니다.
길쭉함과 흰색/붉은색이 함께 보이는 P/C1, 황갈색의 N 사이에는 실제 RGB 차이가 있다.
RGB/HSV 통계를 저장했으나 색 Threshold나 별도 Semantic Classifier는 만들지 않았다.
조명·부분 관측·가림 아래 정상 Peg를 보존할 경계가 이번 자료로 검증되지 않았기 때문이다.
GT 크기·Pose·Actor ID는 이 신호 계산에 입력하지 않았다.

N은 t29의 **엄격히 과거** BC Box 관측과도 같은 표면을 주장한다. 기존 t29 Box State는
`TRACKING_CONFIDENT`, valid=true, nonrobot=1.0, supported900이었다. 그 표면까지의 최근접
거리에서 현재 후보898점 중896점(**99.7773%**)이 기존4mm 감사 척도 이내이며,
Median/P90 거리는0이다. 이는 강한 충돌 신호지만 표면의 반복 관측과 Slot의 Confidence가
Semantic Box 정답을 자동 증명하지는 않는다. 사후 GT에서 t29 Box900점은 실제 Box897,
Peg2, 나머지1로 확인됐다. 이 정답은 Rule Input으로 쓰지 않았다.

반면 **같은 프레임 t30의 BC Box 상태**는 supported4, temporal=false,
`TRACKING_UNCERTAIN`이다. Source를 확인하면 이 값은 원래 B0의 잘못된 Peg 초기화 이후
두 마스크의 Overlap을 제거한 결과에 의존한다. 초기화 전의 독립적인 신뢰 근거로 쓰면
순환 논증이므로, 현재 관측 진단과 과거 상태를 구분해서 기록했다. B0의 Box도 t0에서
Robot1848 FP, t30에서 Robot2073 FP를 포함한다. Box Slot을 항상 정답으로 둘 수 없다.
원래 t0 Box Detection은 t30에서 이미30 step 전의 Rectangle이다.

DINO의 t25→t30 반복 IoU .995154399는 실제로 같은 잘못된 Box 후보를 반복 승인했다.
시간적 일치는 표면 지속성의 근거이며 Peg 의미 정확성의 증명이 아님을 확인했다.

## 규칙·국소 판정

**규칙0개, Wrapper 미생성.** 구분 신호가 없다는 결론은 아니다. 관측된 형태·색과
과거 Box 충돌을 하나의 안전한 승인·거부 경계로 바꾸는 근거와 근접/가림 대조가
충분하지 않다. 두 사례의 답을 맞추기 위한 Threshold나 “Box와 겹치면 거부” 규칙은
도입하지 않았다. 따라서 `candidate_validation_rule.py`는 만들지 않았다.

CSV의 P/N/C1 `DEFER`는 **안전 규칙이 없어 새 초기화를 Commit하지 않는 실험 처분**이다.
실행한 all-DEFER Classifier의 결과가 아니며 `rule_executed=false`를 함께 기록했다.
P와 C1은 정답을 초기화할 기회를 보류하므로 해결 성과로 세지 않는다.
N의 Commit 방지도 배포 가능한 올바른 REJECT로 세지 않는다.
정상 마스크 재현은 확인했지만 정상 인식을 보존하는 검증 규칙의 회귀 통과는 주장하지 않는다.

## 계산·보존·산출물

Temporary Prompt **3개**. 실제 SAM2 Image Encoder3 / Mask Decoder3 /
Single-frame Inference3 / Memory Encoder0. DINO0 / DINOv2 0 / Sequence Replay0 /
Simulator Reset·Render0 / Graph 재캐시0 / 신규 Benchmark0 / 규칙0이다.
Prompt 처리시간(JPEG 저장·현재 프레임 로드 포함, Model Load·사후 Geometry 제외)은
P359.82ms / N545.84ms / C1 522.38ms다. 단일 측정이며 Production Latency 검증이 아니다.

Instance의 Python3.12·Torch2.14.1+cu130·CUDA13.0에서 모델을 실행했고,
RGB-D/Robot FK/사후 평가는 CPU다. macOS MPS·Simulator·GPU PhysX·Renderer는 실행하지 않았다.
설치 SAM2 `_C` 확장 부재로 기존 Native Fill-hole 후처리가 건너뛰어졌으며,
패키지 변경 없이 그 경고와 옵션(fill_hole_area=8)을 기록했다. SAPIEN Import의 Vulkan
경고도 있었지만 Simulator를 생성하지 않았다. N/C1의 저장 B0 마스크 동일성은 확인됐다.

중복을 제거한 **1,324개 입력·동결 Source·Model·Raw·Mask·GT·Pause 파일**의 시작/종료
SHA가 일치했다. Launcher Stop Stub5개와 보존 Original, B0/BC 및 동결 후보를 바꾸지 않았다.
원래 Stage4A-2 모듈 Runner0을 확인했다. 공통 `ExperimentLogger`를 W&B disabled로
재사용했고 UUID는 `ec74e42f93ce4217a4e5a64e4f9e3c35`다. 외부 게시·팀별 Logger는 없다.
사용자의 최소 파일 지시에 따라 실행 코드·Logger 정보는 verification 안에 포함했다.

CSV의 NumPy 직렬화 오류1회는 마스크·Actor 평가 저장 이후 발생했다.
저장 결과에서 CSV/PNG/보존 검증만 이어갔으며 Prompt·Forward 재시도는0이다.
마스크 Decode/SHA·Actor 합계·EXACT_GT·N/C1 원본 동일성·예산·원본 보존을 검사했고,
PNG를 직접 열어 배치도 확인했다. Production 변경이 없어 학습/Simulator 테스트는 실행하지 않았다.

독립 Output Root는 `/workspace/task10i_stage4a2_initial_candidate_verification_20261102_v1/`이며
아래 로컬 복사본과 보고서를 함께 보존했다.

- [SAM2 출력·Packed Mask·Actor 사후 평가](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/candidate_probe_results.json)
- [GT-free 증거·Original/Temporary/처분 비교 CSV](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/candidate_evidence_comparison.csv)
- [두 핵심 후보 비교 PNG](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/case_comparison.png)
- [Source/Model/Raw/Mask SHA·실행 코드·보존 검증](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/verification.json)

정식 SAM2 Memory Commit, 장기 Tracking, 다른 초기 상태에서의 일반화는 미검증이다.
Internal/Final 평가·전체 Graph 재캐시·GNN–GRU 학습을 재개하지 않고 이 국소 작업을 종료한다.

![핵심 후보 비교](../../outputs/task10i/stage4a2/initial_candidate_verification_20261102_v1/case_comparison.png)

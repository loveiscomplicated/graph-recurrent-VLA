# Task 10-I Stage 4A-2 — Dual Grounding Feasibility

2026-11-02. **최종 판정: `TRADEOFF_UNRESOLVED`. `PAUSED_SCOPE_REVIEW` 유지.**

YOLO-World Small은 t30의 실제 Box에 Box Label을 제시해 DINO의 잘못된 Peg 승인과
충돌하는 국소 근거를 제공했다. 그러나 t0의 실제 Peg를 복구하지 못했고 정상 C1 Peg도
놓쳤다. 두 Detector를 함께 쓰는 안전한 승인 정책이나 Production 이득은 입증되지 않았다.
정확도 중단 조건에 따라 Detector 지연·병렬 실행 Microbenchmark는 수행하지 않았다.

## 동결 입력과 실행

사용자 지정 Vast.ai **58576054 / RTX3090 24GB**에서 기존 Main RGB **3장**만 사용했다.
P=`0355/t0`, N=`0355/t30`, C1=`0645/t0`이며 모두 원본 **128×128**이다.
선행 Initial Candidate Verification, Box Semantic Ownership, Paused Scope Review를
검토했다. 추가로 사전 특정된 C2는 제한된 선행 자료에서 확보하지 못해
`CONTROL_UNAVAILABLE`로 기록했다. 대신 P/C1의 동일 프레임 Box Proposal도 비교했다.
추가 Episode 검색·Simulator Reset/Render/Rollout은 없다.

- DINO: 기존 `IDEA-Research/grounding-dino-tiny` 저장 출력만 사용. **재실행0**.
  Peg Prompt=`red and white block`, Box Prompt=`cardboard box`.
  원래 900 Query/logit NPZ와 원래 Threshold .1을 통과한 Proposal을 읽었다.
  Top-K·Joint Null .3·Exclusion Rule·Consumer를 변경하지 않았다.
- YOLO: [공식 YOLOWorld API](https://docs.ultralytics.com/models/yolo-world/)와
  설치 **Ultralytics 8.4.198**의 실제 코드를 확인했다.
  첫 이미지 추론 전에 **`model.set_classes(["peg", "box"])`** 실행.
  두 모델의 Prompt가 다르며 Prompt Sweep은 없다.
- YOLO 설정은 추론 전에 **confidence .01 / NMS IoU .7 / max_det300 /
  imgsz640 / FP32 / class-aware NMS / augment=false**로 동결했다.
  공간 일치는 **IoU≥.5**. Detector Score끼리 비교·합산하지 않았다.
- 원본 RGB를 손실 없는 BGR ndarray로 API에 전달하고, 내부 LetterBox640,
  BGR→RGB·BCHW·float32·/255를 거쳤다. 실제 입력은 `[1,3,640,640]`.
  출력 Box는 원본128×128 좌표다. GT·Mask는 추론 입력에 없다.
- 원래 연구 venv에 설치하지 않고 새 Output Root의 `deps/`에 `--no-deps`로
  의존성을 설치했다. CLIP은 YOLO 클래스의 텍스트 인코더이며 추가 Detector가 아니다.
  기존 CUDA/PyTorch와 패키지 목록의 시작/종료 동일성을 확인했다.

YOLO Weight SHA256:
`9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`.
공식 assets v8.4.0의 `yolov8s-worldv2.pt` 하나만 사용했고 재저장·학습은 없다.
DINO Weight SHA는 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`.
SAM2 Checkpoint SHA는 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`로
보존됐으며 SAM2를 실행하지 않았다. Package·Preprocessing Source·CLIP commit/hash는
[model_identity.json](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/model_identity.json)에 있다.

## 핵심 결과와 사후 GT

**두 모델 결과를 모두 저장하고 SHA를 확인한 뒤 EXACT_GT Actor Segmentation을 열었다.**
GT의 가시 Actor Bounding Box와 IoU≥.5이면 위치가 맞는 것으로 평가했다.
Rectangle 내부 Actor Pixel 수는 보조 진단이며 Mask IoU로 해석하지 않는다.

| 사례 | 기존 DINO Peg 후보 / 실제 Consumer | YOLO 결과 | DINO 기준 후보와 IoU | 사후 판단 |
|---|---|---|---:|---|
| P 0355/t0 | call133/rank1/query3, .291640937 / Null 폐기 | 실제 Peg 없음; 실제 Box .039023019 | 실제 Peg 후보와 최대 .035970 | Peg `DINO_ONLY`; 누락 복구 근거 없음 |
| N 0355/t30 | call153/rank0/query0, .448092550 / 늦은 초기화 승인 | 같은 실제 Box를 **Box .028521508**로 제안 | **.959844** | Pair의 `SEMANTIC_CONFLICT`; DINO Peg 오인식과 구별할 국소 근거 |
| C1 0645/t0 | call333/rank0/query0, .481278419 / 정상 Joint 초기화 | 실제 Peg 없음; 실제 Box .013479628 | Peg 기준0 | 정상 Peg도 `DINO_ONLY`; AND 승인 시 Recall 손실 우려 |

핵심 Box 좌표는 다음과 같다. 전체 후보는 CSV/JSON에 보존했다.

| 후보 | 원본 Pixel XYXY |
|---|---|
| P DINO Peg | [18.808136, 68.308746, 65.521461, 93.421692] |
| N DINO Peg | [40.506973, 39.060646, 74.560875, 71.436760] |
| N YOLO Box | [40.393898, 39.781509, 74.395950, 71.775894] |
| C1 DINO Peg | [33.983185, 71.738342, 85.032066, 106.521164] |

P의 원래 DINO Peg 후보는 GT Peg Bounding Box와 IoU **.900172**였으며,
Consumer의 최종 `None` 때문에 비교에서 제외하지 않았다. YOLO는 그 위치에 Peg를
제시하지 않았다. N의 YOLO Box는 GT Box Bounding Box와 IoU **.940366**이고,
Rectangle1088 Pixel 중 **Box899/Peg33/Robot24/Background132**였다.
YOLO가 이 영역을 Peg로 오인식한 Post-NMS 결과는 없다.

같은 N의 YOLO Box에 DINO rank0과 rank4 두 후보가 연결된다(rank4 IoU **.548332**).
따라서 전체 대응의 `agreement`는 **`AMBIGUOUS_MATCH`**, 개별 Pair의
`pair_relation`은 **`SEMANTIC_CONFLICT`**로 함께 저장했다. 임의로 일대일 대응을
선택하지 않았다. C1의 실제 Box에서도 YOLO는 DINO Box와 IoU **.963959**로 동의하지만,
DINO의 다른 Peg Proposal도 같은 영역에 연결된다. 정상 Peg 기준 후보 자체에는
잘못된 Box 충돌이 없었다.

P에서는 YOLO의 가장 높은 Box Confidence **.274736**가 실제 Box 대신 로봇 영역에
위치했다(Rectangle1219 Pixel 중 **Robot1143**, Box16). DINO Box Proposal도 이 영역에
IoU **.939388**로 동의했다. **같은 Class에 동의해도 정답이 아닌 실제 사례**다.
실제 P Box는 YOLO의 세 번째 후보이며 GT Box Bounding Box IoU **.930220**이다.

YOLO Post-NMS 후보 수는 P/N/C1 각각 **4/1/1**, Peg는 **0/0/0**이다.
각 프레임의 **8400개 decoded Pre-NMS 후보와 두 Class Probability 전체**를 저장했다.
Raw Peg Probability 최대값도 각각 **.000191163 / .000007326 / .000013996**으로
동결 .01 미만이었다. 더 낮은 Threshold로 재실행하거나 이를 승인된 검출로 해석하지 않았다.
DINO 비교 Proposal은 **20/9/17개**, 모든 공간 대응·단독 후보 CSV는 **49행**이다.
N에는 같은 프레임의 원래 DINO Box 호출이 없어 t0의 오래된 Box를 t30 검출로 대체하지 않았다.

## 비용·판정·보존

**Q1: N에서만 보완 신호가 관측됐다.** P 복구 실패와 정상 C1 Peg 누락이 있어
최종 판정은 단순 `PARTIAL_COMPLEMENTARITY`보다 **`TRADEOFF_UNRESOLVED`**가 적합하다.
실제 Consumer를 바꾸지 않았으므로 정상 Recall 손실은 새 Controller의 실측 결과가 아니라,
관측된 Peg 누락이 드러낸 정책 위험이다. Consensus AND 조건도 구현하지 않았다.

**Q2: 미측정.** 정확도 결과가 불충분해 D/Y/S/P Microbenchmark를 수행하지 않았고
`latency_comparison.json`도 생성하지 않았다. GPU 동시 연산 중첩·p50/p95·Peak VRAM·
병렬 결과 동일성은 검증하지 않았다. `PARALLEL_NOT_SUPPORTED`를 판정한 것도 아니다.
YOLO CPU Weight Load379.61ms, 클래스 텍스트 준비8835.67ms는 해당 성공 실행의 준비
기록이며 다운로드/첫 CUDA 전송을 포함한 총 Loading Time이나 Detector Latency가 아니다.
저장된 첫 호출 시간도 Cold/Warmup/계측 복사 비용이 포함되어 성능 비교로 사용하지 않았다.
SAM2·Graph·Physics·GNN/GRU 또는20Hz Closed-loop 성능 주장은 없다.

원본 Source·Raw·GT·Checkpoint·Stop Stub/보존 Original·Freeze를 포함한
**1341개 파일의 시작/종료 SHA가 일치**했다. 기존 연구 venv 패키지 목록도 동일하며,
원래 Stage4A-2 Runner는0이다. YOLO 총 Forward6은 성공3 Frame + 첫 P의 계측 중단1 +
자동 Warmup2다. 초기 Hook이 Warmup까지 세어 검증 Assertion이 실패한 뒤 계측만 수정했다.
추가 입력 프레임이나 Threshold 변경은 없다. CSV49행 Readback, 좌표/Score 유효성,
Actor 합계, 사후 GT Barrier, PNG 배치를 확인했다. Production 코드 변경이 없어
Simulator/학습 테스트는 실행하지 않았다. macOS MPS·Simulator·Renderer는 미실행이다.

공통 `ExperimentLogger`를 W&B disabled로 재사용했다.
성공 평가 UUID는 `063f72cd54b142bda09f511650c12d50`이다. 외부 게시는 없다.
기존 Checkpoint 교체·새 Consensus Controller·SAM2 정책·Graph 재캐시·학습·
Internal30/Final32+12 재개는0이다.

독립 원격 Root는 `/workspace/task10i_stage4a2_dual_grounding_20261102_v1/`다.
필수 결과와 실행 Source·Raw YOLO 후보·로그·Logger 기록은 로컬에도 복사했다.
가중치와 의존성 복사본은 원격 Root에 남겼으며, 로컬 Identity에 SHA를 보존했다.

- [후보·대응·GT 전체 CSV](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/dual_detector_results.csv)
- [보완성·실패·지연 중단 분석](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/consensus_analysis.json)
- [Source/Raw/GT/Checkpoint 보존·Readback](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/verification.json)
- [비교 PNG](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/dual_detector_comparison.png)

![동일 RGB에서 DINO/YOLO와 사후 GT 비교](../../outputs/task10i/stage4a2/dual_grounding_20261102_v1/dual_detector_comparison.png)

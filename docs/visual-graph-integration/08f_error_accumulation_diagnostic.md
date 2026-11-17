# Task 08-F — Train Seed 30003 오류 누적 분기 실험

ExperimentLogger UUID `1e33012210174b8aa49a5485f1b9c063`. Vast.ai 57640710의 RTX 3090 추론, CPU PhysX, 20 Hz 제어. 기존 Full Best A35/B48/C38을 사용했다. 새 학습, Checkpoint 변경, 원본 Full 작업 재개, 외부 게시, Git commit은 수행하지 않았다. 원본 Full pipeline은 사용자 중단 후 `ERROR` 상태이며 Final40/Method는 여전히 0/40 완료다. [Task 08-E 중지·보존 감사](08e_full_trainseed_diagnostic.md)를 참조한다.

## 설계와 유효성

Seed 30003의 저장 Expert Action으로 현재 런타임의 E/E 기준 궤적을 한 번 재생했으며, t=136에 삽입 성공했다. 각 방법과 분기 `k=0,10,25,49`에서 새 Reset 후 동일 Expert prefix를 적용했다. EM은 분기 Action을 Expert, 그 뒤 9 Action을 Model로; ME는 분기 Action을 Model, 그 뒤 9 Action을 Expert로; MM은 분기부터 Model로 최대 250 Action을 실행했다. Student는 실제 현재 RGB-D로 매 Step 추론했고 GRU hidden은 Episode 동안 유지했다. 각 Arm은 한 번만 실행했다.

분기 직전 물리 상태, RGB-D, Robot76, Graph, hidden, 예측 Action을 비교했다. **12/12 분기가 Prefix Identity 검사를 통과**했다. Task08-E 최초 Action 예측도 A/B/C 모두 최대 절대차 0이었다. 저장 HDF5와 동일하다고 확인한 것은 t=0 초기 물리 상태이며, t>0의 분기 기준은 이번 실행의 E/E 궤적이다.

## 주요 결과

10 Action 뒤 TCP–Peg 상대 위치 차이를 비교했다. `고정 Expert suffix`는 ME 대 E/E, `모델 feedback suffix`는 MM 대 EM이다. 두 비교는 동일한 첫 Model Action 효과를 서로 다른 suffix 제어에서 측정한다. 단위는 mm다.

| Method | k=0 첫 Action 차이 | k=0 Expert suffix | k=0 Model suffix | k=0 파지 | Expert prefix 후 파지·Lift | 모든 분기 삽입 |
|---|---:|---:|---:|---|---|---|
| A | 2.45 | 2.74 | 17.72 | 실패 | k=25,49에서 성공 | 실패 |
| B | 2.05 | 2.21 | 9.07 | 실패 | k=10,25,49에서 성공 | 실패 |
| C | 1.73 | 2.27 | 10.27 | 실패 | k=10,25,49에서 성공 | 실패 |

A k=0은 사전 기준인 `Model suffix 차이 ≥ Expert suffix 차이 + 10 mm` 및 `≥ 2배`를 모두 충족했다. B/C k=0은 모델 suffix 차이가 각각 Expert suffix의 약 4.1배와 4.5배였지만 10 mm **추가 차이** 조건은 충족하지 않았다. 기준을 사후 변경하지 않고 연속 수치도 함께 보존했다.

## 해석

**A: SUPPORTED.** 동일 초기 Model Action의 10 Action 뒤 영향이 모델 feedback 조건에서 17.72 mm, Expert suffix 조건에서 2.74 mm였다. k=25/49의 Expert prefix는 파지·Lift를 회복했다. 초기 Action 오차가 이후 피드백 궤적에서 확대되어 파지 실패에 기여한다는 해석을 지지한다.

**B/C: INCONCLUSIVE for 사전 증폭 판정.** 두 방법도 k=0의 Model suffix 차이가 Expert suffix보다 컸고, Expert prefix 후 파지·Lift가 회복됐다. 그러나 정해 둔 10 mm 추가 차이 기준에는 못 미쳤다. 오류 누적의 가능성을 배제하지 않지만 이번 기준으로 확정하지 않는다.

**삽입 실패는 별도 문제다.** 모든 MM 분기는 250 Action 내 삽입 실패였다. Expert prefix가 파지와 Lift를 회복해도 Align/Insert를 해결하지 못했다. Expert prefix는 물리 상태와 GRU 이력을 함께 바꿔 그 둘의 기여를 분리할 수 없다. 한 Seed·Arm당 한 번의 결과를 일반 성공률로 해석하지 않는다. Full 모델의 정상 Expert 관측 Action MSE가 Single Overfit보다 1.2만~2.7만 배 컸다는 Task08-E 결과도 누적의 출발점으로 남아 있다.

후속 우선순위는 다음과 같다.

1. 파지 전에는 초기 Action 오차가 피드백 상태·GRU 이력에서 어떻게 확대되는지, 고정된 물리 상태와 이력을 분리하는 실험을 설계한다.
2. 파지 후에는 Align/Insert 구간의 Action/Controller 오차, object presence, 상대 geometry와 복구 능력을 별도로 조사한다.
3. Epoch 연장은 Train Seed fit 개선과 Validation 악화 위험을 함께 평가하는 별도 사전 등록 실험으로 둔다. 이번 결과만으로 단순 연장을 결정하지 않는다.

## Artifact와 검증

- [정적 HTML](../../experiments/1e33012210174b8aa49a5485f1b9c063/error_accumulation_diagnostic.html), [전체 분기 CSV](../../experiments/1e33012210174b8aa49a5485f1b9c063/branch_metrics.csv), [비교 JSON](../../experiments/1e33012210174b8aa49a5485f1b9c063/comparison.json)
- [Feedback 대 Expert suffix](../../experiments/1e33012210174b8aa49a5485f1b9c063/figures/feedback_vs_fixed.png), [10 Action 이탈 곡선](../../experiments/1e33012210174b8aa49a5485f1b9c063/figures/divergence_curves.png), [Prefix 회복](../../experiments/1e33012210174b8aa49a5485f1b9c063/figures/prefix_rescue.png)
- [Protocol](../../experiments/1e33012210174b8aa49a5485f1b9c063/protocol.json), [E/E 기준 재생](../../experiments/1e33012210174b8aa49a5485f1b9c063/reference_expert/summary.json), [Readback](../../experiments/1e33012210174b8aa49a5485f1b9c063/verification.json), [실행 후 원본 보호 감사](../../experiments/1e33012210174b8aa49a5485f1b9c063/post_run_protection_audit.json)
- 각 Arm의 예측·적용 Action, Controller 명령, RGB-D, Graph, hidden, contact, GT Sidecar, 실제 after-action MP4는 `experiments/1e33012210174b8aa49a5485f1b9c063/branches/{A,B,C}/k{000,010,025,049}/{EM,ME,MM}/`에 있다.

Readback은 JSON 156, NPZ 7597, CSV 2, PNG 6, MP4 36, HTML 링크 17개에서 통과했다. PNG/CSV/JSON 수에는 오류로 중단된 첫 보고서 생성 시도를 별도 보존한 `report_attempt_1/`의 파일도 포함된다. 이 시도는 소스 Snapshot 경로 누락으로 중단됐으며 시뮬레이션은 재실행하지 않았다. 원본 Best Checkpoint SHA256은 Task08-E와 일치했다. CUDA PhysX와 실제 제어 장치에서는 실행하지 않았다.

초기 설정 시도 UUID `3985a485946f454a8f22b7fb88d4cef2`는 Protocol JSON 직렬화 검사에서 시뮬레이션 시작 전에 종료됐다. 해당 실패 기록은 서버에 보존했다. 유효한 실험 UUID는 위에 명시한 `1e33012210174b8aa49a5485f1b9c063` 하나다.

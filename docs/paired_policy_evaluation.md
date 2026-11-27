# 최종 통합 모델과 ACT·DP의 폐루프 비교

## 평가 조건

기존 통합 모델 평가에 예약한 reset seed 300개(4060000–4060299)를 고정 Hole과
이동 Hole의 모든 ACT·DP 체크포인트에 같은 순서로 적용한다. 시드 파일을
다시 생성해 기존 이동 평가 롤아웃의 시드 순서 및 원본 파일의 SHA-256
`7fe07e3645117a7cd897751fd4a4917f05bd648175a0db16219b6d95b7013765`와
대조했다. 장면은 모델 성공 여부나 가시성으로 선별하지 않는다.
각 정책은 해당 시드에서 초기화한 뒤 자기 행동의 결과를 관측하고 다음 행동을 정한다.

CPU PhysX, CUDA RGB-D 렌더링, 공통 카메라 512 × 512, `pd_ee_delta_pose`,
250스텝 한도를 맞춘다. 성공은 한도 안에 시뮬레이터의 삽입 성공 판정을
받았는지로 집계한다. ACT·DP는 학습 때 사용한 RGB 크기와 로봇 상태 변환을
유지한다. 통합 모델 두 구성은 동일한 고정 Surface-Context Tracker–GNN을 쓴다.

이동 조건은 [공통 이동 프로토콜](data_collection.md#policy-independent-moving-hole-transfer-protocol)을
적용한다. 같은 reset seed에는 같은 결정적 60 mm XY 궤적을 적용하며
이동 구간은 제어 스텝 10–40이다. 각 롤아웃의 계획 해시, 실제 변위,
이동 완료 여부와 제어 주파수를 기록한다.

## 체크포인트와 Vast.ai 배치

| 인스턴스 태그 | 평가 체크포인트 |
| --- | --- |
| `ACT fixed hole` | 고정 학습 ACT 75k |
| `DP fixed hole` | 고정 학습 DP 100k |
| `ACT moving hole` | 고정 학습 ACT 100k와 고정·이동 혼합 데모 학습 ACT 100k |
| `DP moving hole` | 고정 학습 DP 100k와 고정·이동 혼합 데모 학습 DP 100k |

통합 모델은 보관된 결과를 사용한다. 고정 Hole은 메인 모델 193/300,
성능 개선 모델 241/300을 HF 릴리스에서 확인했다. 이동 Hole은 성능 개선 모델
고정 학습본 222/300, Graph-heavy 이동 추가학습 앙상블 231/300의 기존 로그가 있다.
메인 단일 Mamba는 같은 이동 Hole 시드 300개에서 고정 학습본
165/300 (55.0%), 이동 추가학습본 180/300 (60.0%)을 기록했다.
두 구성의 전체 롤아웃에서 시드와 기존 성능 개선 모델·ACT·DP의 시드별
이동 궤적 해시가 일치했다.
[메인 모델 이동 Hole 평가 기록](../outputs/moving_hole_graph_only_single/evaluation/paired_summary.json) ·
[고정 Hole 릴리스와 이동 평가 기록](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models)

## 실행과 집계

저장소 루트에서 `python main.py --config 경로/설정.json`으로 실행한다.
ACT·DP의 진입점은 `evaluation.visual_baseline:evaluate`다.
Linux CUDA/Vulkan과 해시가 고정된
평가 체크포인트가 필요하다. 각 태그에서 `paired_eval` tmux 세션을 사용한다.

체크포인트별로 `effective_config.json`, `rollouts.jsonl`,
`closed_loop_summary.json`을 저장한다. 성공 수와 Wilson 95% 구간을 보고하고,
비교할 두 정책의 300개 시드가 모두 일대일로 맞는지 확인한 뒤 성공 판정이
엇갈린 시드를 집계한다. 이동 비교에서는 시드별 궤적 해시도 일치해야 한다.
이번 평가는 고정 두 조건과 이동 네 조건의 총 1,800회 롤아웃이다.

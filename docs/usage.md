# 현재 기준 실행

저장소 루트에서 Python3.11 환경을 사용한다. 설치와 simulator 확인은
[README](../README.md), [ManiSkill 안내](../envs/maniskill/README.md)를 따른다.
아래 root는 기존 PlanA 실험의 config·normalizer·codec·cache·runtime assets를 갖춘
실험 디렉터리다. 빈 디렉터리에서 새 실험을 자동 준비하는 명령은 아니다.

```bash
python -m tools.train --help
python -m tools.rollout --help
python -m tools.evaluate --help

python -m tools.train --root /absolute/experiment/root --smoke --device cuda
python -m tools.train --root /absolute/experiment/root --variant E0 --device cuda
python -m tools.evaluate --root /absolute/experiment/root \
  --checkpoint /absolute/run/checkpoints/step001000.pt --split validation --device cuda
python -m tools.rollout --root /absolute/experiment/root --preflight --device cuda
python -m tools.rollout --root /absolute/experiment/root --run /absolute/run \
  --checkpoint /absolute/run/checkpoints/step001000.pt --device cuda
```

학습은 기존 Graph cache를 읽는다. Offline 평가는 simulator·Terra/SAM2를 실행하지
않고 checkpoint/config/cache/normalizer identity를 검사한 뒤 저장 Graph로 평가한다.
Test는 기존 `model_selection_frozen.json` gate를 필요로 한다. Live는 현재 관측에서
Terra/SAM2와Graph 생성을 실행하고 **영상을 반드시 저장**한다. 기존 실험의 smoke·
split·seed·예산·평가 규칙은 변경하지 않았다.

Live는 현행 CPU PhysX＋CUDA renderer와 native512 카메라 경로를 쓴다.
Mac에서는 CUDA live를 실행 확인하지 않았다. Offline/학습의 `--device cpu`는 CPU
모델 계산을 선택하지만, 기존cache identity에는 생성backend도 포함되므로 다른
backend로 만든cache를 무조건 허용하는 의미는 아니다. Terra API를 포함하는live는
비용이 발생할 수 있다. 기본은Terra 전용이며 인증 실패 시 DINO로 자동 전환하지 않는다.

## Cache와 호환 경로

합의한 원자료＋Grounding＋raw SAM2 mask 1차cache, 저장mask로 만드는Graph 2차cache는
앞으로 구현할 항목이다. `tools.build_graph_cache` / `tools.build_perception_cache`는
아직 없다. `vla.data.graph_cache`의 생성 처리는 기존combined 경로의 이관이며,
Terra/SAM2를 다시 실행한다. 저장mask를 재사용하는 처리로 호출하지 않는다.

Graph 생성 실체는 `graph.current.CurrentGeometry`이며 cache/live에서 같은builder를
선택한다. Cache에는 GNN 이전Graph를 저장한다. 현행identity는
`graphs_modular_live_v5`로 grounding policy와Graph builder를 기록한다. 옛identity를
조용히 바꾸거나 재사용하지 않는다. 과거 실험 재현은 backup commit과 당시의
config/assets를 사용하고, 새로운cache를 만들려고 옛cache를 덮어쓰지 않는다.

이전 `graph.plan_a_bc.train` / `rollout` / `cache`의 import·module 실행은 같은
실체에 위임한다. 이들은 이관 중 호환 경로이며 별개의 기본 구현이 아니다.

## Graph 뷰어

```bash
python -m visualization.server --port 8765
```

`http://127.0.0.1:8765/`에서 기존 저장rollout/Task09 Graph를 읽는다.
`/captures`는 관측·mask, `/results`는 실험 결과 목록이다. 데이터 배치와 대응 형식은
[뷰어README](../visualization/README.md)를 참조한다. 이번에는 기존 형식을 유지했으며,
주live가 저장하는 임의NPZ를 새로 지원한다고 선언하지 않는다.

## 기록과 검증

기존 `ExperimentLogger`를 재사용한다. Offline 평가도 새run_id를 만들고 원checkpoint의
sidecar가 있으면 source_run_id를 연결한다. W&B 없이도 로컬 기록을 남기며 학습/평가에서
HF로 자동upload하지 않는다. [실험 기록](experiment_tracking.md)과
[HF 게시](huggingface_artifacts.md)를 따른다.

```bash
WANDB_MODE=disabled python -m unittest tests.test_modular_live graph.plan_a_bc.tests \
  graph.plan_a_bc.test_robot_live_integration graph.plan_a_bc.test_wrist_gate_default \
  graph.plan_a_bc.test_robot_overlap visual.runtime.test.test_live_grounding \
  visual.runtime.test.test_coordinate_frames visualization.test_viewer \
  visualization.test_captures visualization.test_persistent -v
WANDB_MODE=disabled python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
```

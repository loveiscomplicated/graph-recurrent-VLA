# Repository cleanup — 현재 주 live 기준 모듈화 진행 보고서

Task ID: `repo_cleanup_modular_live_20261104` · 2026-11-04 KST.

## 결과와 범위

사용자 D12에 따라 별도 통합 모델 확정안을 기다리지 않고 현재 주 live를 기준으로
Graph 생성·GNN·controller·통합/실행을 분리했다. 기존 기본 모델의 CPU 초기 가중치,
action·hidden 출력과 gradient norm이 분리 전과 정확히 일치한다. 교체 가능한 factory를
학습·offline 평가·live·기존 cache 경로에 연결했다.

**전체 repository cleanup 완료 보고가 아니다.** 현재 기준 실행의 모듈화와 진입점 정리
단계다. 과거 산출물 전체 원격 보존·복원 검증, 그 이후의 과거 소스 제거, 저수준 runtime
helper 추가 이관, 합의한 저장 mask 기반 2단계 cache는 아직 남았다. 데이터 수집·학습
실험이나 유료 API 실행으로 범위를 확대하지 않았다.

## 정리 전 백업

- 정리 전 HEAD: `d42f777bd8a2a23eef4f814f5922783383c17948`.
- 원래 branch: `archive/visual-graph-wip-20261102`.
- 소스·설정·문서 변경115개, 약1.076MB를 파일별 검토·SHA 대조 후 백업했다.
- Backup commit: **`d222fe2bd555b4c6c1bc6e6fb65c5017bbe242d5`**.
  메시지: `backup: preserve live GPU work and cleanup decisions before modularization`.
- 이후 같은 checkout에서 `codex/cleanup-modular-live-20261104` branch로 전환해 작업했다.
  Push나 다른 worktree 정리는 실행하지 않았다.
- [백업 범위·파일 SHA](../../outputs/repo_cleanup_modular_live_20261104/backup_scope.json),
  [백업 commit 확인](../../outputs/repo_cleanup_modular_live_20261104/backup_commit.json).

이 commit은 Git 밖 데이터·checkpoint·영상·ignored 실행 소스의 원격 백업을 의미하지
않는다. [사전 조사5–6절](repo_cleanup_preflight_20261104.md#5-git-밖-보존-대상과-용량)의
보존 목록을 유지한다. Outputs 약53.40GB, experiments 약11.83GB, visual 약2.20GB,
audit_outputs 약0.32GB는 당시 logical size이며 겹침을 제거한 백업 용량이 아니다.
원자료와 calibration/robot/TCP/action/split, mask·Graph cache, 모든 checkpoint와
optimizer/RNG, source archive·외부 weight, 실패/중간/최종 결과·영상, config/metadata/
metrics/sidecar/게시 receipt를 보존해야 한다. Strict visibility 핵심 소스처럼 ignored
outputs 안에 있는 코드도 포함한다. 이번에 이 파일들을 삭제·덮어쓰기·업로드하지 않았다.

## 실제 변경

| 현재 위치 | 역할과 이전 연결 |
| --- | --- |
| `graph/current.py` | 기존 `plan_a_bc.geometry`의 CurrentGeometry·FK·Graph 검사 실체 이관 |
| `gnn/edge_mlp.py` | 기존 edge-aware GNN·structured readout·Graph21 E0/E1 설정 소유 |
| `controller/recurrent.py` | 기존 GRU/LSTM 제어 모델 소유 |
| `vla/contracts/` | 기존 Graph/PackedGraphs/packing과 최소 component protocol |
| `vla/integration/` | feature·presence·robot 상태 조합과 통합 정책. 모델 교체와 상태 전달 |
| `vla/state.py`, `recording.py` | Tensor tree/None 상태의 reset·detach·slice·stack·저장. 기본 GRU 저장 배열 유지 |
| `vla/workflows/components.py` | config 또는Python factory로 Graph/GNN/controller 선택·identity 검증 |
| `vla/workflows/training.py`, `rollout.py` | 기존 학습·주 live 실행 조정 실체 이관 |
| `vla/workflows/evaluation.py` | 기존 offline 계산을 사용하는 독립 평가 진입점 |
| `vla/data/` | 기존 입출력·identity·Graph cache reader 및 legacy combined generator |
| `vla/perception/grounding.py` | Terra 전용 기본 정책, 명시적 기존 auth fallback 선택 |
| `tools/train.py`, `rollout.py`, `evaluate.py` | 새 기본 module 실행 진입점 |
| `visualization/` | 공통 Graph 상수·좌표 변환 직접 참조. 기존 실제 로더·서버·UI 유지 |

기존 `graph.plan_a_bc`·`graph.integration_baseline.model/graph`·`recurrent.models`의
공개 import를 얇은 위임/re-export로 유지해 기존 caller를 끊지 않았다. Train/rollout/
cache module alias는 같은 실제 module을 반환하므로 테스트 monkeypatch도 동일 실체에
작용한다. 과거 구현의 별도 복사본을 새 package에 중복 유지하지 않는다.

Graph facade 아래 PlanA·persistent_geometry·visual/runtime·GPU 실험 helper는 여전히
활성 의존성이다. 실험 폴더 이름만 보고 삭제할 수 없다. 공통 simulator, logger,
publisher 구현은 변경하지 않았다. Root 문서와 agent 지침은 D12와 실제 상태로 갱신했다.

## 교체와 호환성

- `components.graph_builder/graph_encoder/controller`의 `module:attribute` factory를
  cache·train·offline·live에서 사용한다. Encoder 출력 차원과 presence 차원을 조정할 수 있다.
- 통합부와 학습 cursor는 controller의 `.rnn`이나 단일 GRU hidden tensor를 가정하지
  않는다. LSTM tuple과 stateless controller로 실제 forward/backward·chunk update를 검증했다.
- 기존 기본 state_dict의 `graph_encoder.*`·`policy.*` key와 초기화 RNG 순서를 보존했다.
  Graph21/E0/E1 설정은 GNN 모듈이 소유하고 교체 encoder는 수정하지 않는다.
- Factory 선택을 checkpoint/로그에 기록하고 재개·평가에서 대조한다. Identity 없는
  옛 checkpoint는 기본 component만 허용한다. Factory 이름만으로 코드 재현을 보증하지
  않으므로 Git commit·config/hash·cache도 함께 필요하다.
- 현재 BC/action codec·7값 action·기존 split/예산/선택 gate·물리/controller audit는
  유지한다. 임의 action 공간·새 loss까지 자동 교체되는 범용 프레임워크라고 주장하지 않는다.
- `grounding.policy` 생략은Terra 전용이다. Missing key/401/403에서 DINO로 자동 전환하지
  않는다. 보존한 대안은 명시적 `terra_auth_fallback`이며 과거 직접 LiveGrounding 사용은
  기존 기본값을 유지한다. 실제 API 호출은 하지 않았다.
- Cache identity는 `graphs_modular_live_v5`로 grounding policy와Graph builder 선택을
  포함한다. 이전cache를 조용히 재인증하거나 새identity로 덮어쓰지 않는다.
- Legacy combined cache generator는Terra/SAM2를 재실행할 수 있다. 이를 저장mask 기반
  새2차cache 명령으로 노출하지 않았다. `tools.build_graph_cache`는 아직 없다.

자세한 계약과 실제 명령은 [architecture](../architecture.md)와 [usage](../usage.md)를 따른다.

## 보존한 주 live 동작

CurrentGeometry의 robot 준비/SAM2 overlap, GPU robot filter, shared frame/correction,
wrist gate, sampling/Graph 생성과 World/EE 선택 계산을 이관했다. Live의 CPU PhysX＋
CUDA renderer·두native512 카메라, 현재 관측만 사용하는 입력 경로, stop-the-clock과
제어/GT 평가 경계, 필수video와진단export 호출을 유지했다. 선택 controller state가
단일tensor가 아닐 때의 저장만 확장하고 기본GRU trajectory의hidden 배열은 유지했다.

이것은 소스 대조·CPU 계약 테스트 결과다. 이번 Mac에서 CUDA·SAM2·live renderer나
실제 GPU overlap 속도를 실행한 결과는 아니다. GPU 처리의 수치/성능 동등성은 GPU
호스트에서 새로 확인해야 한다. Native512 strict-gated 새expert 수집도 실행하지 않았다.

## 검증 결과

환경: `.venv/bin/python`3.11.13, Torch2.14.1, ManiSkill3.0.1, SAPIEN3.0.3, macOS CPU.
CUDA 없음. 실제 simulator reset/step·renderer·SAM2 inference·Terra API는 실행하지 않았다.

| 검증 | 결과 |
| --- | --- |
| 분리 전 source에서 생성한 E0/E1 golden | state_dict key·가중치 SHA256·variable-length action/hidden·gradient norm 정확 일치 |
| 모듈·기존 live/로봇/gate·grounding·좌표·뷰어 suite | **77개 실행,69개 통과·CUDA8개 skip·실패0** |
| 기존 logger/artifact suite, 별도process | **42개 통과** |
| 독립offline 평가 | 실제 임시 Graph cache/checkpoint를 읽고 validation 예측·logger 기록 생성. Test gate 전 거부, gate 후test cache 로딩 검증 |
| 새tools CLI | train/rollout/evaluate의 `--help` 실행 성공 |
| 실제 저장Graph 뷰어 | 16개목록 중 Task09 seed30002의106프레임 Graph를로드. 259노드, t0→t1·World 선택·TCP node input inspector 확인 |
| 정적 검증 | Python1,191개 AST parse 성공, 새 안내·보고서의 local 링크 정상, git diff 공백 검사 통과 |
| Graph 이관 대조 | import·protocol 상대 경로 변경을 정규화하면 CurrentGeometry·RobotFK 클래스 AST가 backup source와 동일 |

Golden fixture는 [tests/fixtures](../../tests/fixtures/README.md), 테스트는
[test_modular_live.py](../../tests/test_modular_live.py)에 있다. Golden의 완전 일치는
이 호스트의CPU/Torch 조건이며 다른host/kernel이나CUDA의동등성을 대체하지 않는다.

실제 뷰어 확인은 임시localhost8768 서버와 in-app browser로 수행했다. 저장파일만
읽었으며 확인 뒤 탭과임시서버를 종료했다. [노드검사 AX 기록](../../outputs/repo_cleanup_modular_live_20261104/viewer_node_inspector.txt),
[화면](../../outputs/repo_cleanup_modular_live_20261104/viewer_world.png).
기존 Task07/09 지원을 유지한 것이며 주live 임의NPZ의신규viewer adapter를 구현한 것은 아니다.

[최종회귀로그](../../outputs/repo_cleanup_modular_live_20261104/modular_tests_final.log),
[인프라로그](../../outputs/repo_cleanup_modular_live_20261104/infrastructure_tests.log),
[정적 검증](../../outputs/repo_cleanup_modular_live_20261104/static_verification.json).
초기검증에서 이관 전module을patch하던mock 경로, 빈edge array fixture의형상,
새factory/config/identity fixture와wrist backend stub을 보정했다. GPU 오류를CPU로
숨기는 fallback은 추가하지 않았다. Baseline의logger suite를다른suite와한process에
혼합하면 global importlib mock/imageio의test 순서 문제가 재현돼 인프라는 별도process로 검증했다.

## 남은 정리

1. 활성 저수준 GPU/perception/Graph helper 의존성을 더 분리하고 실제GPU에서 영상 포함
   end-to-end·World/EE·수치와성능을 검증한다. 그 전까지 관련과거package는 유지한다.
2. 전체산출물 inventory에 원격고정revision/경로·hash·복원검증을 연결한다. 기존HF 완료
   증거가 있는task와11월4일 신규자료를 구분한다. 비공개자료를 임의공개하지 않는다.
3. 보존/복원이 확인된과거소스만 현재tree에서 제거하고 docs/restore에 codeSHA·의존성과
   실행/복원명령을 연결한다. 모델학습 코드의이관이 과거실험 전체재현 확인은 아니다.
4. 저장raw SAM2 mask를공유Graph builder로 전달하는 합의된2단계cache를 구현한다.
   새expert 수집은별도실행으로 두고 기존데이터를임의로512로 변환하지 않는다.

현재진입점이사용하는실험root 준비는아직과거실험계약을 따른다. Repo cleanup의후속작업에서
기본설정·필수asset 안내를 단순화하되 재현에필요한설정과공통운영규칙을 보존해야 한다.

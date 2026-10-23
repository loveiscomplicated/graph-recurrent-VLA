# 공통 개발 규칙

## 범위와 현재 구조

이 문서는 실험 인프라의 일관성과 재현성을 위한 규칙이다. 모델 구조, feature,
그래프 구성, sequence, dataset, 학습·평가 방식, 팀 내부 config 형식은 팀이 결정한다.
`docs/PROJECT_CONTEXT.md`의 독립 feasibility 원칙은 유지한다. 그 문서의
“팀 디렉터리가 비어 있다”는 초기 설명과 달리 현재 Graph 코드는 구현되어 있다.

| 위치 | 현재 역할 |
| --- | --- |
| `envs/maniskill/`, `examples/` | 공통 simulator 생성, raw GT, 실행 예제 |
| `visual/` | `.gitkeep`만 있음. 학습·평가 구현 없음 |
| `recurrent/` | GT state 입력 controller 7종 비교, 관측 손상 4단계, 학습·평가·리포트 (`recurrent/README.md`) |
| `graph/prototype/`, `v2/`, `v3/`, `v4/` | Graph 모델 비교, TOML config, 학습·평가 및 진단 |
| `graph/subtasks/`, `graph/subtasks/reach_bc/` | subtask 데이터·환경·분할, Reach BC 학습·평가 |
| `utils/experiment_logger.py`, `utils/sync_experiment.py` | 공통 scalar/local/W&B 기록과 사후 동기화 |
| `utils/experiment_artifacts.py`, `tools/publish_experiment.py` | 로컬 manifest와 명시적 HF 게시 |
| `tests/`, 각 Graph 코드의 `tests/`, `docs/` | 공통/팀 테스트와 개발·실험 문서 |

루트 `AGENTS.md`가 공통 지침이며 `CLAUDE.md`는 이를 가져오는 진입점이다.
분석 시 두 파일 외에 하위 `AGENTS.md`/`CLAUDE.md`나 별도 instruction 파일은 없었다.
실제 entrypoint와 연결 범위는 [tracking 문서](experiment_tracking.md)의 표를 따른다.
공통 train/evaluate launcher나 모든 팀에 강제되는 config/checkpoint schema는 없다.

## 에이전트 작업 절차

1. 루트 지침과 관련 팀 README, 변경할 코드, 공통 인프라, 관련 테스트를 읽는다.
   Training/evaluation은 [tracking](experiment_tracking.md), 산출물은
   [HF publishing](huggingface_artifacts.md) 문서를 추가로 읽는다.
2. 기존 API와 실제 호출 사례로 요구사항을 해결할 수 있는지 확인한다.
3. 기존 Logger/Publisher를 재사용한다. 팀별 W&B logger나 HF upload 구현을 추가하지 않는다.
4. API가 부족하면 공통 모듈의 확장과 관련 계약 테스트를 먼저 검토한다.
   미구현 기능을 사용 가능하다고 문서화하거나 팀에 강제하지 않는다.
5. 기존 train/validation/test 분리, checkpoint 선택, optimizer/scheduler,
   RNG, resume 및 출력 파일 동작을 불필요하게 바꾸지 않는다.
6. 변경에 관련된 기존 테스트를 실행한다. 공통 인프라에 영향이 있으면 아래 계약
   테스트를 포함하고, simulator 변경 시 설치 패키지와 runnable example로 확인한다.
7. 공통 인터페이스가 바뀌면 API 설명과 사용 예시를 함께 갱신한다.

외부 접속은 학습의 필수 조건으로 만들지 않는다. W&B 비활성화·인증 누락·SDK/전송
실패 시 로컬 기록을 유지하고, HF 인증/업로드는 종료 후 별도 작업으로 둔다.
로컬 디스크 쓰기 실패까지 자동 복구되는 것은 아니다. 산출물 경로는 writable하고
실험마다 충돌하지 않아야 한다. 비밀은 각 플랫폼의 환경변수/비밀 파일로만 주입한다.
플랫폼별 인증 우선순위와 fallback은 각 인프라 문서를 따른다.

## 공통 Experiment Identity

| 연결 | 현재 코드의 동작과 규칙 |
| --- | --- |
| 실행 → `run_id` | `ExperimentLogger` 생성마다 `uuid4().hex` 생성. ID를 외부에서 받는 생성자 인자는 없음 |
| 로컬 기록 | `experiments/<run_id>/` 기본 경로. `root` 인자 또는 `EXPERIMENTS_DIR`로 루트 변경 |
| 재현성 | 실제 적용 config와 CLI override/hyperparameter, seed를 전달. 시작 시 Git commit/dirty 자동 기록. Seed 설정 자체는 팀 코드 책임 |
| W&B | `wandb.init(id=run_id, resume='never')`. 사후 sync도 기존 ID 사용 |
| 산출물 | 저장 직후 `link_output(path)`로 `<파일명>.tracking.json` 연결. checkpoint 자체에 ID를 주입하지 않음 |
| 평가/재개 | 새 실행 ID 생성. `link_output(path, checkpoint=True)`가 원본 sidecar에서 `source_run_id`를 읽고 sidecar 유지. 없으면 추정하지 않음 |
| HF | 같은 로컬 실행을 `--run-id`로 선택. 원격 `<team>/<run_id>/`에 config·선택 파일·manifest 게시. 성공 revision/URL은 로컬 receipt 보존 |

가능한 entrypoint에서는 실제 dataset hash, 환경 설정, resume 출처도 기존
`record()`로 보존한다. Git dirty 상태는 commit만으로 재현할 수 없음을 보고한다.
`record()`가 metadata를 병합하므로 `run_id`, team, Git commit 등 기존 identity를
덮어쓰지 않는다. 사후 게시를 위해 새 Logger나 별도 HF ID를 만들지 않는다.

현재 HF manifest는 로컬 metadata 전체의 복사본이 아니다. `source_run_id`, W&B URL,
dataset hash 등 일부 metadata는 원격 manifest에 자동 포함되지 않는다. 세부 제한은
[HF 문서](huggingface_artifacts.md#현재-제약과-확장-제안)를 참고한다.

## 검증과 완료 보고

저장소 루트의 실제 Python 환경에서 실행한다. 아래 테스트는 인증 없이 동작하며
외부 online 업로드를 검증하지 않는다. Simulator/device 검증과 구분해 보고한다.

```bash
.venv/bin/python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
```

`check_experiment_tracking`은 실제 online run을 만드는 점검 도구이고,
`--init-repos`/일반 게시/`--sync-wandb`도 원격을 변경한다. 단순 문서 검증에 사용하지 않는다.
HF `--dry-run`은 네트워크를 사용하지 않지만 로컬 manifest는 갱신할 수 있다.

완료 시 재사용한 공통 인프라, 변경 범위, 실행한 테스트와 결과, 미실행 backend/device,
남은 제약을 간단히 보고한다. 문서만 바꾸는 작업에서는 pipeline/logger 코드를 수정하지
않고 필요한 수정은 제안으로 남긴다.

## 추후 CI에서 검증할 후보

현재 이 검사를 수행하는 CI workflow는 없다. 다음은 도입 제안이다.

- `CLAUDE.md`의 `@AGENTS.md`, 지침 간 로컬 링크, API/CLI 예시의 실제 존재 확인.
- 공통 Logger/Artifact 계약 테스트 실행: UUID·sidecar identity, offline/disabled,
  인증 우선순위, 선택 게시, 실패 격리·부분 성공·재시도.
- 새 train/evaluate 코드의 공통 Logger 사용과 팀별 직접 W&B/HF transport 추가 검토.
  정적 탐지는 공통 모듈·점검·sync·테스트의 합법적 SDK 호출을 허용해야 한다.
- secret scan 및 비밀 파일/로컬 실험 산출물의 Git 추적 방지.
- 새 공통 API 변경 시 관련 문서 갱신 여부 확인. 모델 구조나 팀 디렉터리 형식은 강제하지 않음.

# 공통 Experiment Tracking

## 저장소 분석과 적용 범위

2026-10-15 작업 시점의 실제 파일 기준이다. `docs/PROJECT_CONTEXT.md`의
팀 디렉터리가 비어 있다는 설명과 달리 Graph prototype은 구현되어 있다.
이 작업은 공통 관측 도구를 추가하며 팀의 모델·데이터·학습 설계를 정하지 않는다.

| 팀/진입점 (`python -m` 뒤 모듈명) | 설정 | 기존 기록/체크포인트 | 통합 |
| --- | --- | --- | --- |
| Visual (`visual/`) | 구현 없음 | 구현 없음 | 향후 연결 예제 제공 |
| Recurrent (`recurrent/`) | 구현 없음 | 구현 없음 | 향후 연결 예제 제공 |
| `graph.prototype.train` | TOML + CLI override | stdout, history/training JSON, `best.pt` | epoch train/validation |
| `graph.prototype.v2.train` | v2 TOML + CLI | 위와 같음 | epoch + 실제 loss components |
| `graph.prototype.v3.train` | TOML + geometry CLI | 위와 같음 | epoch + variant 기록 |
| `graph.prototype.v4.train` | v4 TOML + history/device/resume CLI | 위 + `last.pt`, resume JSON | epoch + 재시작 설정 |
| `graph.subtasks.reach_bc.train` | v2 TOML, 고정 seed, CLI 경로 | training JSON, `checkpoints/best.pt` | 정식 train + 최종 test summary |
| `graph.prototype.evaluate` | checkpoint config + CLI | offline/latency/선택 rollout JSON | 평가 summary |
| `graph.prototype.v2.analyze.offline` | checkpoint config | offline/phase/latency 결과 | 모델별 평가 run |
| `graph.prototype.v2.diagnostics` | CLI, checkpoint config | rollout stages JSON | stage별 평가 summary |
| `graph.prototype.v3.compare` | candidate/baseline checkpoint + CLI | paired comparison JSON | 비교 평가 run |
| `graph.prototype.v4.evaluate` | checkpoint config + CLI | offline/cache latency/rollout JSON | 평가 summary |
| `graph.subtasks.reach_bc.evaluate` | 고정 paired manifest + CLI | held_out/fresh/latency JSON, 영상 | 평가 summary, episode별 partial summary |

학습은 train split의 gradient update와 validation checkpoint 선택을 유지한다.
평가는 기존 test split 및 선택된 checkpoint를 사용하고 optimizer를 실행하지 않는다.
Reach BC의 test는 기존대로 모든 epoch와 checkpoint 선택 후 한 번 읽는다.
기존 W&B/TensorBoard infrastructure는 없었으며 JSON writer는 변경하지 않았다.

데이터 준비, overfit/smoke 전용 도구, `reach_bc.train --smoke`, bottleneck,
report/figure/video 도구, 기존 JSON을 합치는 v2/v4 compare 등은 자동 run 생성 대상이 아니다.
v2 analyze의 summary/probes는 모델별 offline 평가와 다른 경로이며 추가 연결 대상이다.
`run_matrix` 등 launcher가 통합된 train/evaluate 모듈을 실행하면 자식 실행에서 기록된다.
Visual/Recurrent의 미래 entrypoint는 아래 예제로 연결해야 한다. 없는 학습 시스템을
새로 만들거나 다른 branch의 코드를 통합한 것으로 간주하지 않는다.

## 팀장이 최초 설정

### W&B 프로젝트와 인증

1. 팀장 계정으로 W&B에서 사용할 Entity(팀 또는 계정의 slug)를 확정한다.
2. 그 Entity 아래 `graph-recurrent-vla` 프로젝트를 만들고 프로젝트 visibility를 Public으로 설정한다.
3. 계정/플랜에서 지원하면 해당 팀에 전용 service account를 만들고 실험 기록에 필요한
   최소 role을 준다. 서비스 계정 제공 여부와 실제 권한 범위는 관리자 화면에서 확인한다.
   지원하지 않으면 팀장이 관리하는 전용 키를 사용한다. workspace별 별도 키면 회전/폐기가 쉽다.
4. 키는 Secret 저장소에만 보관한다. 팀원의 W&B 계정·초대·`wandb login`은 필요 없다.
   Public은 조회 범위이며 쓰기 권한이나 인증을 제공하지 않는다.

공식 문서: [프로젝트 API의 Entity와 project](https://docs.wandb.ai/models/ref/python/functions/init),
[API 키와 서비스 계정 관리](https://docs.wandb.ai/ko/platform/app/settings-page/user-settings).
공유 workspace에서 실행되는 코드는 주입된 키 권한을 사용할 수 있다. Public 기록에 올릴
config는 팀장이 검토한다. 로거는 credential 이름의 필드를 재귀적으로 가리지만
일반 문자열에 임의로 삽입한 비밀까지 판별하지 않는다. 환경 전체를 config로 넘기지 않는다.

### `yonsei-ai-gpu`의 Workspace별 설정

세 Workspace를 각각 설정한다. 다른 Workspace의 환경변수/인증을 상속한다고 가정하지 않는다.

| Workspace | `WANDB_RUN_GROUP` | 의미상 team |
| --- | --- | --- |
| `Graph-Recurrent-VLA-controller` | `controller` | `recurrent` |
| `Graph-Recurrent-VLA-vision` | `vision` | `visual` |
| `Graph-Recurrent-VLA-graph` | `graph` | `graph` |

각 Workspace의 관리되는 환경/시작 설정에 다음을 등록한다.

- `WANDB_API_KEY`: Secret 주입 방식을 쓸 때만 필요하다. 아래 Workspace 내부
  `wandb login` 방식을 쓰면 이 환경변수는 없어도 된다. 키를 Git, 이미지, init script,
  shell history, 노트북, README에 넣지 않는다.
- `WANDB_ENTITY`: 위에서 정한 동일 Entity slug.
- `WANDB_RUN_GROUP`: 위 표의 값. 자동 workspace 이름 감지의 불확실성을 없앤다.
- `WANDB_MODE`: 생략(자동 선택) 또는 `online`.
- `EXPERIMENTS_DIR`: 로컬 로그가 남을 영속 볼륨의 절대 경로. 생략하면 repo의 `experiments/`.

프로젝트는 코드에서 `graph-recurrent-vla`로 고정된다. `WANDB_PROJECT`로 분산되지 않는다.
Group 우선순위는 호출 인자 → `WANDB_RUN_GROUP` → `VESSL_WORKSPACE_NAME`의 정확한
Workspace 이름 매핑 → team 기본값(visual→vision, recurrent→controller)이다.
`VESSL_WORKSPACE_NAME`이 실제로 주입되는지는 가정하지 않으므로 관리 설정에 group을 지정한다.
`team`은 실행된 코드의 소유 팀이다. 예를 들어 controller Workspace에서 Graph 코드를 실행하면
team=graph, group=controller가 된다.

팀장이 각 Workspace의 실제 Python 환경에 한 번 설치한다.

```bash
python -m pip install -r requirements-tracking.txt
```

같은 이미지/영속 venv 또는 Workspace init script에 설치 명령을 보존하면 재시작에도 적용된다.
SDK는 선택 의존성이고 기본 simulator requirements는 변경하지 않는다.

[Vessl Secrets 공식 문서](https://docs.vessl.ai/guides/organization/secrets)는 조직 Secret과
Run 환경변수 연결을 설명한다. **현재 Workspace 화면의 Secret 연결 기능과 cluster 정책은
직접 확인해야 한다.** Workspace에서 Secret 참조를 지원하면 각 실행 환경에 연결한다.
지원하지 않으면 아래처럼 Workspace 내부에 `wandb login`으로 인증을 저장할 수 있다.
일반 텍스트 환경변수 입력란을 Secret 저장소라고 간주하지 않는다.

### Workspace 내부에 `wandb login`으로 저장하는 방법

Workspace별 Secret 연결을 지원하지 않을 때 사용할 수 있다. 팀장은 세 Workspace의
`접근 제어`에서 우리 팀 외 사용자의 접근 여부를 확인한다. Workspace 관리자와 같은
OS 사용자/홈을 사용하는 사람은 저장된 키를 읽거나 사용할 수 있으므로 이 범위를
기준으로 접근을 제한한다. 개인 계정 키 사용에 대한 플랜 조건은 별도로 확인한다.

각 Workspace에 접속해 **실제 학습을 실행할 동일 OS 사용자와 Python 환경**에서 한 번 실행한다.

```bash
python -m pip install -r requirements-tracking.txt
wandb login
```

키는 대화형 프롬프트에 입력하고 command 인자로 넣지 않는다.
W&B CLI가 인증을 저장하며 로거는 환경변수 키 또는 저장된 인증의 존재를 확인한다.
기본 경로는 `~/.netrc` (또는 `~/_netrc`)이며 `NETRC`를 지정한 환경도 지원한다.
현재 W&B 호스트의 항목만 확인하고 키를 metadata/config에 복사하지 않는다.
기본 파일을 사용했다면 파일 권한도 확인한다.

```bash
chmod 600 ~/.netrc
```

팀장은 비밀이 아닌 다음 설정을 Workspace 시작 환경 또는 학습 터미널에서 자동으로
읽는 shell profile에 영속 저장한다. 아래 export만 터미널에서 실행하면 현재 세션에만 적용된다.

```bash
export WANDB_ENTITY="기존-WnB-팀-slug"
export WANDB_RUN_GROUP="graph"
```

각 Workspace에 맞게 group을 `controller`, `vision`, `graph`로 바꾼다.
프로젝트는 `graph-recurrent-vla`로 고정되어 있다. 인증 계정이 기존 팀의 프로젝트에
기록할 권한을 갖고 있어야 한다. 로그인 계정의 개인 Entity를 팀 Entity 대신 쓰지 않는다.
팀원은 이후 기존 training command만 실행한다.

`WANDB_API_KEY`가 이미 주입돼 있으면 저장된 인증보다 우선한다. 다른 계정을 쓰려면
관리자가 기존 환경변수 주입을 먼저 해제하고 `wandb login --relogin`으로 교체한다.
다른 OS 사용자로 실행하거나 컨테이너에서 홈을 공유하지 않으면 인증도 공유되지 않는다.
인증 없음/읽기 불가/손상된 netrc는 offline으로 처리한다.

[W&B 로그인 API 안내](https://docs.wandb.ai/models/ref/python/functions/login).
설치된 SDK 0.30.0의 login 문서에서도 환경변수 및 netrc 경로를 확인했다.
이 작업에서 실제 Workspace에 로그인하거나 키를 저장하지는 않았다.

### 재시작·재생성 후 확인 및 복구

이 작업에서는 Vessl 관리 콘솔에 접속하거나 인증을 설정하지 않았다. 세 Workspace의
재시작/재생성 시 인증 유지도 실행 검증하지 않았다.
[Vessl Workspace 생성 문서](https://docs.vessl.ai/guides/workspace/create)에 따르면 `/root`는
PVC로 지속되고 init script는 시작할 때 실행된다. 이것만으로 Secret 연결이나 삭제 후 재생성의
보존을 보장할 수 없다. shell에서 한 번 `export`한 변수는 지속 설정이 아니다.

팀장은 각 Workspace에서 다음 절차를 수행한다.

1. Secret 방식은 관리 설정의 Secret 참조를 보존한다. `wandb login` 방식은 인증 파일이
   해당 사용자 홈의 영속 볼륨에 있는지 확인한다. Entity/group/로그 볼륨/Python 의존성도 보존한다.
2. 기존 학습 command로 작은 실행을 한 후 `metadata.json`의 `wandb_mode=online`,
   `wandb_url`, 대시보드의 동일 ID/metric 도착을 확인한다. 키 값은 출력하지 않는다.
3. stop/start 후 새 터미널과 실제 학습 프로세스에서 동일 확인을 반복한다.
4. 재생성 시 Secret 방식은 기존 Secret 참조를 다시 연결한다. 내부 login 방식은
   팀장이 새 Workspace에서 `wandb login`을 다시 수행한다. 두 방식 모두
   Entity/group/볼륨 및 SDK 설치를 다시 적용한다. 인증 파일을 Git으로 백업하지 않는다.
5. 키가 만료/폐기되면 Secret을 교체하거나 `wandb login --relogin`을 수행한 후
   workload를 재시작한다. 이후 offline bundle을 동기화한다.

## 팀원 실행과 로컬 기록

기존 README의 학습 command를 그대로 실행한다. 예:

```bash
python -m graph.prototype.v4.train --model flat --history current --output outputs/graph/v4/new-run
```

환경변수 키 또는 `wandb login`으로 저장된 인증이 있고 **Entity도 있으면** online을 사용한다. 둘 중 하나가 없으면 explicit
`WANDB_MODE=online`이어도 offline으로 전환하므로 대화형 로그인을 요청하지 않는다.
`WANDB_MODE=offline`은 강제 offline, `disabled`는 SDK 없이 로컬 기록만 한다.
W&B 미설치, import/init/log/finish 오류는 경고를 내고 로컬 기록을 보존한다.
온라인 init timeout은 10초이며 실패 시 SDK를 반복 초기화하지 않는다.
transport 오류의 종류만 기록하고 인증 정보가 들어갈 수 있는 예외 문자열은 기록하지 않는다.

```text
experiments/<UUID run_id>/
  config.yaml       # JSON 형태의 YAML 1.2; 실제 TOML 값과 CLI override 보존
  metadata.json     # team/group/model/seed/Git commit/dirty/시간/status/outputs
  metrics.json      # run_id, epoch(step), scalar metrics를 가진 JSON 배열
  wandb/            # SDK가 동작했을 때만 native offline/online 로그
```

`config.yaml`은 기존 팀 config 형식을 바꾸지 않는 기록용 snapshot이다.
`metadata.json`에는 기존 데이터 hash와 환경 config도 가능한 진입점에서 저장한다.
Git dirty=true이면 commit만 checkout해서 변경 중인 코드까지 재현할 수는 없다.
중요 실험은 코드를 commit하고 config와 기존 dataset hash를 함께 보관한다.

각 process 호출은 UUID4로 새 run을 만들며, v4 checkpoint resume도 새 run이다.
resume 원본 경로/설정은 command config와 기존 resume JSON에 남으며, 원본 sidecar가 있으면 source_run_id도 기록한다.
동일 로컬 결과 경로를 여러 process가 공유하면 기존 체크포인트 충돌은 여전히 가능하므로
팀의 기존 output 경로 규칙을 지킨다. 로거의 UUID 디렉터리는 공유되지 않는다.

기존 checkpoint 및 평가 결과 파일 옆의 `<파일명>.tracking.json`이 run_id와 실험 디렉터리를
연결한다. checkpoint 내용·위치·optimizer·scheduler는 변경하지 않는다. 평가에는 새 run_id를
주고 checkpoint sidecar가 있으면 `source_run_id`로 학습 run과 연결한다.
기존 파일의 sidecar가 없으면 출처를 추측하지 않는다. metric JSON 각 event에도 ID가 있다.

예외/KeyboardInterrupt 때 context manager가 failed 상태와 종료 시각을 저장하고
W&B `finish(exit_code=1)`을 호출한 뒤 원래 예외를 전파한다. SIGKILL/호스트 종료는
정상 종료 처리를 실행할 수 없다. flush된 이전 metric과 running metadata를 복구에 사용한다.
로컬 저장 볼륨은 writable하고 충분한 공간이 있어야 한다.

## 기록하는 metric과 확장

기존 epoch scalar만 기록한다. `train/loss`, `val/loss`는 실제 action MSE의 alias이며
`optimization_loss`, `learning_rate`, translation/rotation/gripper components도 원래 값 그대로다.
평가 dictionary는 `eval/offline/...`, `eval/latency/...`, `eval/settings/...` 등으로 펼친다.
현재 코드가 제공하는 successes/denominator/success_rate, position error 등의 값만 보존한다.
없는 collision rate나 success rate를 만들거나 새롭게 계산하지 않는다.
리스트·tensor·이미지·영상·per-step rollout trace는 업로드하지 않는다.
학습은 epoch 단위, rollout은 기존 stage summary 또는 Reach episode summary 단위다.

새 팀 코드에서는 실제 설정이 확정된 지점에서 다음과 같이 연결한다.

```python
from utils.experiment_logger import ExperimentLogger

with ExperimentLogger(team="visual", config=cfg, seed=seed, model=model_name) as experiment:
    # 기존 학습/평가 코드에서 실제 계산한 scalar만 전달
    experiment.log({"train/loss": loss_value}, step=epoch)
    experiment.log(evaluation_metrics, prefix="eval")
    experiment.link_output(checkpoint_path)  # 기존 저장 직후 sidecar
```

로거는 seed를 **기록만** 하며 RNG를 재설정하지 않는다. model/dataset/training 인터페이스를
강요하지 않는다. config는 JSON으로 보존 가능한 값이어야 한다. 새 metric은 scalar dictionary에
추가하면 된다. 분산 학습은 아직 통합 대상이 아니며 향후 팀에서 주 process만 로깅해야 한다.

## 대시보드 비교

`<entity>/graph-recurrent-vla` Runs 화면에서 Group을 `graph`/`vision`/`controller`로 묶고
`config.experiment.team`, `config.experiment.model`, seed, Git commit, job_type으로 필터링한다.
`train/loss`, `val/loss`, learning_rate와 실제 평가 metric을 panels에 추가한다.
학습/평가 run은 ID와 source_run_id로 연결한다. 다른 metric 정의/단위/분모를 가진 팀의
결과는 같은 이름이라고 바로 비교하지 않는다. Public URL에서 팀원이 조회할 수 있는지
팀장이 로그아웃 상태에서 확인한다.

## Offline 동기화

팀장 또는 관리된 인증 환경에서 SDK와 안전하게 주입된 키/Entity를 사용한다.
SDK native offline bundle이 있으면 아래 명령으로 전송한다.

```bash
wandb sync --entity "$WANDB_ENTITY" --project graph-recurrent-vla experiments/<run_id>/wandb/offline-run-<timestamp>-<run_id>
```

[공식 offline 동기화 안내](https://docs.wandb.ai/ja/support/models/articles/can-i-run-wandb-offline).
로그를 다른 호스트로 옮길 때 native bundle 전체와 config/metadata/metrics를 함께 복사한다.
UUID는 그대로 유지된다. 원래 group과 config도 bundle에 보존된다.

SDK가 없었거나 online init이 실패해 usable native bundle이 없다면:

```bash
python -m utils.sync_experiment experiments/<run_id>
```

JSON 이벤트를 같은 run ID로 업로드한다. 이미 생성된 online run에는 중복 전송하지 않는다.
기존 run이 있으면 `resume=never`로 거부되며 native bundle을 복구해 동기화한다.
실행 중인 실험은 먼저 종료한다. 강제 종료된 경우 결과를 점검한 뒤 metadata의 status를
failed로 복구해야 한다. JSON 재전송과 native sync를 같은 run에 중복 적용하지 않는다.

## Hugging Face와 Notion 연결

이번 변경은 HF 업로드 파이프라인을 만들지 않는다. 향후 checkpoint/dataset/rollout 저장 시
`<run_id>/best.pt`, `<run_id>/rollout.mp4`와 같은 경로를 사용하고 config/metadata 및 sidecar를
함께 보관한다. 평가 영상은 평가 run_id와 source_run_id를 함께 기록한다.
HF card 및 Notion 분석에는 run_id, 학습 source_run_id, W&B URL, HF revision/경로를 넣으면 된다.
원격 URL이 정해지면 `experiment.record(hugging_face_checkpoint=uri)`로 metadata에 추가할 수 있다.

## 검증

```bash
python -m unittest tests.test_experiment_logger -v
```

Mock online 계약(환경변수 키/CLI 저장 인증), 무인증 offline 선택, 잘못된 호스트·손상된 netrc,
SDK 미설치·실패 격리, config redaction,
seed/Git commit, 동시 UUID 생성, group 분리, 예외 시 로컬 보존, checkpoint 연결을 검증한다.
SDK 설치 시 실제 offline binary 생성도 별도 subprocess로 검증한다.
실제 외부 online 업로드는 유효한 키와 Entity가 있어야 별도로 확인할 수 있다.

### 이번 호스트에서 실행한 결과

- macOS, Python 3.11, W&B SDK 0.30.0에서 로거 테스트 10개 통과.
- 실제 기존 `outputs/graph/dataset.pt`로 v1/v2/v3/v4 학습을 각각 수정 전 HEAD와
  수정 후에 실행했다(1 epoch, 최대 1 batch; v1/v2 flat, v3 flat_sampling,
  v4 flat/current). model weights, optimizer state, scheduler state, 시간 제외
  epoch metrics가 모두 일치했다. 정식 학습 성능이나 수렴 결과를 의미하지 않는다.
- 무인증 상태의 기존 v1 training command와 offline test/latency evaluation을 실행해
  실제 `.wandb` offline bundle 2개와 학습 source_run_id 연결을 확인했다.
- 검증 출력: `outputs/tracking_validation/ef0faa08dbdc4b83bb81e5dbacaaaa72/validation.json`
  (로컬 ignored artifact).
- 기존 Graph/Reach 회귀 테스트와 로거 테스트를 합쳐 51개 실행: 48개 통과,
  Linux 전용 Reach CPU 통합 테스트 3개 skip. 기존 v2/v4 live regression에서
  macOS CPU simulator reset/step도 실행했다.
- Reach BC 정식 30 epoch, Linux 전용 Reach 평가, 전체 simulator rollout 평가,
  CUDA/MPS 학습, Vessl 환경, 외부 online 업로드는 이번 변경에서 실행하지 않았다.
  online 동작과 JSON 복구 전송은 mock 기반 계약 테스트다.

후속 인증 변경에서는 저장된 login 인식과 JSON 동기화까지 포함해 로거 테스트 14개가 통과했다.
실제 W&B online 업로드 및 Vessl 재시작 시 인증 유지 여부는 여전히 실행 검증하지 않았다.

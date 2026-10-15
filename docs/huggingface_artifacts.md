# Hugging Face 실험 산출물

개발 절차와 identity 원칙은 [공통 개발 규칙](engineering_rules.md)을 따른다.
기존 `utils/experiment_artifacts.py`와 `tools/publish_experiment.py`를 재사용한다.
팀별 HF uploader를 새로 만들지 않으며, 명시적 요청 또는 정의된 publishing workflow에서만
게시한다. 모든 checkpoint/영상을 기본 자동 업로드하는 학습 코드를 추가하지 않는다.

## 기존 구조와 구현 범위

기존 `ExperimentLogger`의 UUID `run_id`, `experiments/<run_id>/config.yaml`,
`metadata.json`, `metrics.json`, `link_output()` 및 `.tracking.json` 연결을 재사용한다.
팀의 checkpoint 내용, 저장 위치, 학습 설정, trajectory 형식은 변경하지 않는다.
Graph train/evaluate의 기존 등록 파일은 자동으로 manifest에 포함된다.
Publisher는 별도 클래스가 아니라 `utils/experiment_artifacts.py`의 `plan_upload()`,
`publish()`, `initialize_repositories()`, `sync_wandb_links()` 함수와 이를 호출하는 CLI다.
Reach BC의 선택적 영상 경로에는 영상과 episode JSON 등록 두 줄을 추가했다.
Visual/Recurrent는 기존 Logger에 연결하면 같은 도구를 사용할 수 있다.

학습 중에는 로컬 inventory만 갱신한다. HF SDK import, 인증 확인, hashing,
Repository 생성, 업로드는 학습에서 수행하지 않는다. 종료된 실험에 별도 CLI를 실행한다.
미완료(`status=running`) 실험은 게시하지 않는다. 실패로 종료된 실험은 게시할 수 있다.
SIGKILL로 running 상태가 남았다면 산출물과 종료 여부를 확인하고 metadata를 복구해야 한다.

## 설치와 인증

학습/게시를 실행하는 Python 환경에서 설치한다.

```bash
python -m pip install -r requirements-tracking.txt
```

W&B와 같은 방식으로 Vessl/Vast.ai에서 HF 전용 인증 파일을 만들 수 있다.

```bash
mkdir -p /root/.secrets
chmod 700 /root/.secrets
nano /root/.secrets/hf.env
```

`nano`가 없는 Debian/Ubuntu 환경에서는 `apt-get update && apt-get install -y nano`로
설치하거나 기존 편집기를 사용한다. 파일에는 다음 한 줄을 입력한다.

```dotenv
HF_TOKEN=<쓰기 권한을 가진 실제 Hugging Face token>
```

저장 후 파일 권한을 제한한다.

```bash
chmod 600 /root/.secrets/hf.env
```

게시/초기화 CLI가 이 파일을 자동으로 읽으므로 별도 `source`나 `hf auth login`은 필요 없다.
파일은 사용자가 생성하며 도구가 token을 생성하거나 파일에 쓰지 않는다.
인증 우선순위는 **기존 프로세스의 `HF_TOKEN` → HF dotenv 파일 → 공식 로그인 token**이다.
기존 환경변수가 읽기 전용 token이면 파일에 새 token을 적어도 환경변수가 우선하므로
환경변수를 갱신하거나 `unset HF_TOKEN`으로 해제한다.

로컬 macOS/Linux에서는 편집기로 `~/.secrets/hf.env`를 만들고 같은 내용을 입력한 뒤
경로를 지정한다.

```bash
mkdir -p "$HOME/.secrets"
chmod 700 "$HOME/.secrets"
# 편집기로 ~/.secrets/hf.env 생성
chmod 600 "$HOME/.secrets/hf.env"
export HF_ENV_FILE="$HOME/.secrets/hf.env"
```

`HF_ENV_FILE`이 없으면 `/root/.secrets/hf.env`를 사용한다. 파일이나 `python-dotenv`가
없으면 기존 환경변수/공식 인증을 계속 사용한다. 대안으로 플랫폼의 비밀 환경변수 기능에서
`HF_TOKEN`을 주입하거나 `hf auth login`으로 로그인할 수 있다.
공식 HF token 저장소와 `HF_HOME`도 SDK의 규칙을 따른다. W&B 파일은 기존
`/root/.secrets/wandb.env`를 사용하며 HF 파일은 별도로 관리한다.

Token 값을 출력하지 않고 현재 설정 유무를 확인하려면 다음을 실행한다.

```bash
python -m scripts.check_huggingface_auth
```

기존 환경변수 설정 여부, 인증 파일 로드 여부, 사용 가능한 token 유무를 출력한다.
Token이 있으면 exit 0, token이 없거나 SDK/설정 읽기에 실패하면 exit 1이다.
네트워크 연결 없이 설정 유무만 검사한다. 실제 인증·Organization 쓰기 권한은
`python -m tools.publish_experiment --init-repos`의 결과로 확인한다.

기존 `Graph-Recurrent-VLA` Organization의 Repository 쓰기/생성 권한을 가진
계정과 token이 필요하다. 읽기 전용 token으로는 생성과 업로드를 할 수 없다.
Fine-grained token은 대상 Organization과 Repository에 필요한 쓰기/생성 범위를
허용해야 한다. Organization 역할이 있어도 token 범위가 부족하면 API가 거부할 수 있다.

Token을 command 인자로 전달하지 않는다. 코드·manifest·config·로그에 token을 저장하지 않는다.
Logger의 기존 재귀적인 비밀 키 마스킹을 재사용하고, JSON config/metrics/episode metadata도
게시 전 다시 마스킹한다. 전체 환경변수나 비밀이 들어 있는 임의 문자열을 config로 전달하지 않는다.
Checkpoint·영상은 불투명한 원본 파일이므로 사용자가 비밀을 삽입하지 않아야 한다.
SDK 예외 메시지나 응답 본문을 출력하지 않고 예외 유형·HTTP 상태·일반적인 복구 안내만 기록한다.

Vessl과 Vast.ai에서 CLI는 동일하다. 해당 프로세스에서 token과 산출물 경로가 접근 가능해야 한다.
`EXPERIMENTS_DIR` 또는 `--experiments-dir`로 영구 볼륨을 지정할 수 있다.
호스트 재생성 후 비밀 주입과 볼륨 연결은 각 플랫폼 설정에서 유지해야 한다.

## Repository 초기화

```bash
python -m tools.publish_experiment --init-repos --dry-run
python -m tools.publish_experiment --init-repos
```

명시적인 초기화 명령에서만 `create_repo(private=True, exist_ok=True)`를 호출한다.
기존 Repository는 재사용하며 기존 공개 범위를 변경하지 않는다. 새 Organization은 생성하지 않는다.
Token이 없거나 읽기 전용이거나 Organization의 쓰기 역할을 확인하지 못하면 생성을 건너뛴다.
실패한 Repository는 `failed`로 보고하며 일부만 성공한 경우 결과를 각각 표시한다.

| 종류 | 기본 Repository | 코드에서 지원하는 용도 |
| --- | --- | --- |
| Model | `Graph-Recurrent-VLA/graph-recurrent-models` | checkpoint, 재현성 정보, 선택한 metrics |
| Dataset | `Graph-Recurrent-VLA/graph-recurrent-rollouts` | 영상, episode metadata, 재현성 정보, 선택한 metrics |

`graph-recurrent-trajectories`와 `graph-recurrent-evaluations`는 이번 단계에서 생성하거나
업로드하지 않는다. Repository 설정은 `Repository`와 `DEFAULT_REPOSITORIES`에 분리되어 있다.
미래 목적지를 추가할 때에는 명시적인 artifact routing과 검증도 함께 확장한다.

### 원격 생성 상태와 지원 범위 구분

Organization은 기존 [Graph-Recurrent-VLA](https://huggingface.co/Graph-Recurrent-VLA)를
사용하며 새 Organization을 만들지 않는다. 위 두 목적지는 사용자가 지정한 현재 기본
Repository이며 코드의 `DEFAULT_REPOSITORIES`에도 등록되어 있다.
`Graph-Recurrent-VLA/graph-recurrent-trajectories`와
`Graph-Recurrent-VLA/graph-recurrent-evaluations`는 향후 계획이며 현재 routing/게시 지원은 없다.
기본 목적지로 설정되어 있다는 사실만으로 원격 생성·접근 가능성이 검증된 것은 아니다.

기존 작업의 실행 기록: 2026-10-15 이 호스트의 실제 초기화 요청은 실패했다. 인증을 확인한 결과 token은 `read`였다.
후속 권한 검사에서는 생성 API 호출 없이 `skipped`를 반환했다. 두 Repository 조회도 404였으므로
생성 또는 존재를 확인하지 못했다. 404만으로 private Repository가 없다고 단정하지 않는다.
이번 지침 작성에서는 원격 인증·생성·업로드를 재실행하지 않았다. 접근 권한이 있는 계정에서
확인하고 필요한 경우에만 명시적 초기화 workflow를 사용한다.

## 로컬 manifest와 산출물 등록

```text
experiments/<run_id>/
  config.yaml
  metadata.json
  metrics.json
  manifest.json
  huggingface_uploads.json  # 한 Repository 이상 게시 성공 후 생성
```

`manifest.json`은 Logger 초기화/metadata 변경/산출물 등록/종료 시 갱신된다.
기존 실험은 CLI가 기존 metadata로 manifest를 생성한다. 로컬 경로는 실험 디렉터리 기준
상대 경로이며, 기존 팀 출력이 외부에 있으면 `../../...` 형태로 연결된다. 파일을 이동하지 않는다.

```json
{
  "schema_version": 1,
  "run_id": "existing-wandb-run-id",
  "team": "graph",
  "model_name": "flat",
  "task": null,
  "environment": {"env_id": "PegInsertionSide-v1"},
  "git_commit": "recorded-commit",
  "git_dirty": false,
  "random_seed": 7,
  "wandb_run_id": "existing-wandb-run-id",
  "config_path": "config.yaml",
  "metrics_path": "metrics.json",
  "checkpoint_paths": ["../../outputs/example/best.pt"],
  "rollout_video_paths": [],
  "episode_metadata_paths": [],
  "trajectory_paths": [],
  "additional_metrics_paths": []
}
```

제공되지 않은 model/task/environment/seed 등은 `null`로 남긴다. task/environment는
`logger.record(task=..., environment=...)`로 기존 실행의 실제 정보를 기록한다.
Git commit은 Logger가 실행 시작 때 기록한 값을 쓰며 게시 시점의 commit으로 교체하지 않는다.
`git_dirty=true`이면 commit만으로 작업 중인 코드를 복원할 수 없다.

```python
from utils.experiment_logger import ExperimentLogger

with ExperimentLogger(team="graph", config=config, seed=seed, model=model_name) as logger:
    # 기존 학습/평가 및 저장 코드
    logger.link_output(checkpoint_path)  # 기존 호출도 그대로 동작
    logger.link_output(video_path, kind="rollouts")
    logger.link_output(episode_json_path, kind="episode_metadata")
    logger.link_output(trajectory_path, kind="trajectory")  # inventory에만 기록
```

명시적 `kind`가 우선한다. 기존 등록 파일은 `.pt/.pth/.ckpt/.safetensors/.bin`을 checkpoint,
영상 확장자를 rollouts, `.h5/.hdf5/.npz/.parquet`을 trajectory, `.json`을 metrics로 분류한다.
형식만으로 의미를 구분할 수 없는 파일은 `kind`를 명시한다. 디렉터리 재귀 탐색은 하지 않는다.
평가의 `link_output(checkpoint, checkpoint=True)`는 기존처럼 원본 sidecar를 유지하고
`source_run_id`를 연결한다. 다른 run이 sidecar 소유권을 바꾼 파일은 게시를 거부한다.
과거 영상이 등록되지 않았다면 원본 run의 실제 산출물인지 확인한 후 metadata의 `outputs`와
`artifact_kinds`에 명시적으로 연결한다. 폴더의 모든 영상을 해당 run으로 추정하지 않는다.

## 선택적 게시

```bash
python -m tools.publish_experiment --run-id RUN_ID --include checkpoint rollouts metrics --dry-run
python -m tools.publish_experiment --run-id RUN_ID --include checkpoint --private
python -m tools.publish_experiment --run-id RUN_ID --include rollouts metrics --retries 2
python -m tools.publish_experiment --run-id RUN_ID --include metrics
```

`--include`는 필수다. 선택한 종류의 **등록된** 파일만 대상으로 하며 로컬 checkpoint 폴더 전체를
탐색하지 않는다. Config와 재현성 manifest는 각 대상에 항상 동반된다. `metrics`를 선택하면
Logger metrics와 추가 등록 metrics JSON을 포함한다. Metrics만 선택하면 현재 Model Repository에
게시한다. Checkpoint와 rollouts를 모두 선택하면 선택한 metrics/config를 양쪽에 연결한다.
Rollouts는 명시적으로 등록한 episode JSON도 동반한다. Trajectory는 전송하지 않는다.

선택 종류에 등록된 파일이 없거나, 선택 파일 누락·빈 바이너리·잘못된 JSON·심볼릭 링크가 있으면
전체 전송 전에 검증 오류를 반환한다. 선택하지 않은 파일의 누락은 게시를 막지 않는다.
JSON은 UTF-8 JSON으로 읽을 수 있어야 한다. 기존 `config.yaml`은 JSON인 YAML 1.2 snapshot이다.
Checkpoint를 역직렬화하거나 영상을 재인코딩하지 않는다. 게시하는 동안 원본 파일을 변경하지 않는다.

`--dry-run`은 SDK/인증/네트워크 없이 파일 목록과 `total_bytes`를 출력한다.
이 용량은 양쪽 Repository의 공통 파일과 manifest를 포함한 논리적 총량이며 원격 중복 제거 전 값이다.
실제 전송에서는 이미 있는 동일 파일을 제외한다. 파일 hash 계산에는 전체 파일 읽기가 필요하다.

목적지는 종류별로 재정의할 수 있으나 기존 Organization 밖의 경로는 거부한다.

```bash
python -m tools.publish_experiment --run-id RUN_ID --include checkpoint rollouts \
  --repo-id checkpoint=Graph-Recurrent-VLA/graph-recurrent-models \
  --repo-id rollouts=Graph-Recurrent-VLA/graph-recurrent-rollouts
```

`--private`는 업로드 대상이 private인지 확인한다. 초기화는 이 옵션 없이도 항상 private로 생성한다.
일반 게시 명령은 Repository를 생성하거나 공개 범위를 바꾸지 않는다.
Token/SDK 누락은 원인을 출력하고 `skipped`로 종료한다(exit 0).
검증/게시 실패는 `failed`와 exit 1을 반환하므로 별도 게시 job에서 재실행할 수 있다.
기존 학습 entrypoint는 게시 CLI를 호출하지 않으므로 HF 장애가 학습을 실패시키지 않는다.

## 원격 구조·중복 제거·복구

두 Repository 모두 다음 구조를 사용한다. `team`은 `visual`, `graph`, `recurrent`, `integration`이다.

```text
<team>/<run_id>/
  artifacts/
    config/<sha256>/config.yaml
    checkpoint/<sha256>/best.pt          # Model Repository
    rollouts/<sha256>/episode.mp4        # Dataset Repository
    episode_metadata/<sha256>/episode.json
    metrics/<sha256>/metrics.json        # 선택 시
  manifests/<manifest_sha256>/manifest.json
```

원격 manifest는 선택한 산출물의 Repository 기준 경로, SHA-256, 크기와 재현성 정보를 담는다.
게시하지 않은 로컬 파일의 경로나 존재를 원격 산출물로 표시하지 않는다.
다른 종류를 나중에 게시해도 기존 manifest를 덮어쓰지 않는다. 변경된 내용은 새 hash 경로에 저장한다.
공통 config와 동일 파일은 같은 경로를 재사용한다.

현재 revision에서 파일 목록을 읽고 LFS SHA-256 또는 Git blob SHA-1과 크기로 실제 동일성을 확인한다.
로컬 receipt가 없어도 원격 파일을 기준으로 중복을 제외한다. 예상 hash 경로에 다른 내용이 있으면
덮어쓰기를 거부한다. `parent_commit`으로 동시 수정 충돌을 감지한다.
각 Repository 안에서는 하나의 commit으로 게시하며 두 Repository 사이의 원자성은 제공하지 않는다.
성공할 때마다 `huggingface_uploads.json`에 URL, revision, manifest 경로를 저장한다.

일시적인 실패에는 제한된 재시도를 수행한다(`--retries 0..5`, 기본 2).
인증/권한/파일 검증 오류에는 즉시 반환한다. 복구 후 같은 명령을 재실행하면 이미 성공한 대상과
응답 유실 전에 commit된 파일도 원격에서 확인하여 다시 보내지 않는다. 서로 다른 선택의 성공 기록을
receipt에 함께 보존한다. 서버에 쌓인 이전 버전 삭제나 garbage collection은 이번 MVP 범위에 없다.

### VideoFolder 고려

이번 MVP는 영상과 팀의 원본 episode JSON을 보존하며 Dataset Viewer 지원을 보장하지 않는다.
Viewer가 필요하면 공식 [VideoFolder 형식](https://huggingface.co/docs/datasets/en/video_dataset)에 맞는
`metadata.jsonl`과 `file_name` 연결, dataset card의 split/data_files 설정을 별도로 추가한다.
현재 JSON episode metadata는 VideoFolder의 행 metadata와 같은 형식이라고 가정하지 않는다.

HF API 사용은 공식 [Repository 관리](https://huggingface.co/docs/huggingface_hub/guides/repository),
[업로드](https://huggingface.co/docs/huggingface_hub/guides/upload),
[HfApi](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)를 기준으로 했다.

## W&B URL 연동

HF 성공 receipt는 W&B 접속 없이 저장한다. 원래 W&B run의 summary에 링크를 반영하려면 별도 실행한다.

```bash
python -m tools.publish_experiment --sync-wandb RUN_ID
```

기존 W&B 인증 로딩을 재사용하며 기록된 `entity/project/run_id`의 run을 Public API로 읽고
`summary["huggingface"]`에 성공 receipt를 기록한다. 새 run 생성, resume, metric 재전송은 하지 않는다.
W&B 인증 누락 또는 네트워크 실패에도 HF commit/receipt는 유지된다. 나중에 위 명령만 재실행한다.
W&B에 아직 run이 없으면 기존 offline sync 절차를 먼저 수행한다.

## 현재 제약과 확장 제안

로컬 `metadata.json`에는 `source_run_id`, `wandb_url`, dataset hash 등 `record()`한 정보를
보관할 수 있지만 `save_manifest()`는 정해진 필드만 원격 manifest로 전달한다.
현재 원격 manifest에는 이 세 필드가 자동 포함되지 않고, 원본 `metadata.json`이나
`.tracking.json` 전체도 자동 게시하지 않는다. Git commit/dirty, config, seed, team/model,
task/environment와 run ID는 현재 manifest/동반 config로 보존한다.

따라서 평가 run에서 원본 학습 run으로의 연결과 W&B/HF 양방향 링크가 완전히 자동화됐다고
설명하지 않는다. 원본 로컬 metadata와 sidecar를 함께 보관하고, 필요하면 기존
`--sync-wandb`로 W&B에서 HF receipt를 연결한다. 향후 공통 manifest schema에
`source_run_id`, W&B URL, dataset hash를 안전하게 추가하고 계약 테스트를 보완하는 것을
제안한다. 이 지침 작업에서는 코드를 변경하지 않는다.

현재 trajectory는 로컬 inventory만 지원한다. Dataset Viewer/VideoFolder,
두 Repository 사이의 원자적 게시, 원격 버전 삭제/GC도 미구현이다.
`publish()`는 파일 검증/로컬 receipt 쓰기 단계에서 예외가 날 수 있으므로 향후 workflow도
별도 게시 job에서 실패를 처리하고, 성공한 로컬 checkpoint/학습 결과를 삭제하거나
덮어쓰는 복구 동작을 추가하지 않는다.

## 검증 및 실행 한계

```bash
.venv/bin/python -m unittest tests.test_experiment_artifacts tests.test_experiment_logger -v
```

Mock 테스트는 manifest, 기존 sidecar 연결, 선택 업로드, 누락 파일, dry-run, 인증/SDK 누락,
중복 제거, 원격 내용 충돌, 재시도, 응답 유실, 부분 성공, W&B 독립 재시도를 확인한다.
기존 Logger의 scalar/config/run ID 및 실제 W&B offline SDK 테스트도 함께 실행한다.
HF 인증값을 사용하지 않는 테스트이며 simulator나 GPU를 요구하지 않는다.

현재 호스트의 설치된 `huggingface_hub` 2.0.0 API signature를 확인했다.
실제 HF 쓰기/업로드, Vessl/Vast.ai 배포 실행, Linux Reach BC 영상 생성, CUDA 학습은 실행하지 않았다.

이번 Agent 지침 정비의 재검증 결과와 미실행 범위는
[tracking 검증 기록](experiment_tracking.md#agent-지침-정비-시-재검증-2026-10-15)을 참고한다.

# Task 10-I: live Terra 기본 경로와 DINO 인증 fallback

작성일: 2026-11-03 KST.

현재 `graph.plan_a_bc`의 live rollout과 causal cache 생성은 Terra Standard를
기본 grounder로 사용한다. `OPENAI_API_KEY`가 없거나 공백인 경우, 또는 Terra가
HTTP401/403을 반환하면 Grounding DINO로 전환한다. 선택·호출·결과·전환 원인은
콘솔에 출력하고, 실제 backend와 호출 횟수도 결과 JSON에 남긴다.

## 구현 범위

- [live_grounding.py](../../visual/runtime/live_grounding.py): 독립 실험의 외형
  Prompt와 엄격한 좌표 계약을 runtime으로 옮겼다. 실험 launcher·output·GT 코드는
  가져오지 않는다. Prompt SHA256은 기존과 동일한
  `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`다.
- [cache.make_vision](../../graph/plan_a_bc/cache.py)와
  [rollout.evaluate](../../graph/plan_a_bc/rollout.py)는 `GDino` 직접 생성 대신
  `LiveGrounding`을 생성한다. Terra 활성 상태에서는 DINO Weight를 로드하지 않는다.
- [initialization.detect_initial](../../visual/runtime/initialization.py)은
  joint initializer를 지원한다. Terra의 Peg/Box Box를 SAM2에 직접 전달하며
  DINO confidence·넓이 제한·후보 공동 할당을 적용하지 않는다. Score는 `None`이다.
- [OnlineVisionAdapter](../../graph/integration_baseline/online_vision_adapter.py)와
  [causal_masks](../../graph/integration_baseline/vision.py)는 카메라별 initializer
  상태를 사용한다. Backend 인증 상태·RGB 캐시는 grounder에서 공유하고 DINO prefix는
  각 카메라/episode에 분리한다. 따라서 Terra로 일부 물체를 찾은 뒤 인증에 실패해도
  새 DINO 결과가 과거 frame0에 묶여 누락 물체의 초기화를 막지 않는다.

공통 simulator, Graph geometry·학습 설정·checkpoint 선택·optimizer·run identity는
변경하지 않았다. 기존 dirty geometry/filter 작업은 그대로 두었다. 과거 독립 DINO
실험과 frozen 배포 소스의 모델 선택은 바꾸지 않았다.

## 요청과 전환 규칙

Terra 요청은 `gpt-6.1-terra`, Responses API, `service_tier='default'`,
`reasoning.effort='high'`, `detail='original'`, `max_output_tokens=5120`,
`store=False`다. 기존 채택 보고서의 설정을 유지했다.
[공식 Terra 문서](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
[공식 서비스 tier 설명](https://developers.openai.com/api/docs/guides/fast-mode).

전체 현재 RGB 한 장과 고정 외형 Prompt만 전송한다. Peg/Box 각각 Box 하나 또는
null이며, 0–1000 좌표를 실제 Width/Height에 따로 맞춘다. 중복 JSON Key,
Boolean·비유한 값·범위 밖·역순·퇴화 Box와 추가 Key를 거부한다. 좌표 보정,
Clipping, Box 확대, Code-fence 제거, HTTP 자동 Retry는 하지 않는다.

| 조건 | 동작과 로그 |
| --- | --- |
| 환경변수에 유효한 비공백 문자열 존재 | Terra 선택, `key_source=environment` |
| 환경변수 없음, 저장소 루트 `.env`에서 로드 | Terra 선택, `key_source=dotenv` |
| 키 없음/공백 | `FALLBACK Terra -> DINO reason=missing_key` |
| Terra HTTP401/403 | 이유를 출력하고 해당 grounder 수명 동안 DINO 사용 |
| DINO 첫 사용 | `LOAD backend=dino`, 로드 완료 시 `READY backend=dino` |
| 정상 Terra 호출 | `REQUEST` 뒤 `RESULT`: 이미지 크기, 지연, found/null |
| 동일 RGB 재사용 | `CACHE backend=terra`, 외부 호출 없음 |
| Timeout·연결 오류·429/5xx·기타 HTTP 실패 | 자격증명이 포함될 수 있는 예외/응답 본문을 숨긴 오류를 출력하고 중단 |
| 완료되지 않은 응답·모델/tier 불일치·잘못된 JSON/Box·Refusal | 명시적 오류. 인증 fallback으로 취급하지 않음 |

키 로딩은 기존 `utils.experiment_logger.load_env_file`을 재사용한다. 작업 디렉터리와
무관하게 저장소 루트 `.env`를 읽고 기존 환경변수를 덮어쓰지 않는다. 명시적으로
빈 환경변수도 덮어쓰지 않는다. HTTP Redirect를 끄고 키 값·Authorization Header·원시
응답·Provider 예외 문자열을 로그/결과에 넣지 않는다.

초기 t0 및 누락 물체가 있는 동안 5개 처리 frame 간격으로만 Grounding을 수행한다.
Terra는 해당 현재 frame의 유효 Box를 바로 초기화에 사용한다. 두 물체를 모두 찾으면
추가 API 호출 없이 SAM2를 계속 사용한다. 이미 생성된 SAM2 상태는 유지한다.
DINO fallback은 기존 후보/공동 할당 및 누락 물체의 두 일관된 scan 규칙을 유지한다.
중간 인증 실패 시에는 해당 카메라의 DINO prefix를 그 frame부터 시작한다.
Reset은 카메라별 prefix를 초기화한다. Tracking-loss recovery는 새로 구현하지 않았다.
64개 이미지의 Box만 저장하는 제한된 RGB 캐시가 동일 입력의 반복 호출과 causal
readback probe를 지원하며, 이전 frame·응답을 API 입력에 넣지 않는다.

## 로그와 결과 위치

`[live-grounding]` StreamHandler가 기본 INFO 이상을 stderr에 출력한다. 별도 root
logging 설정이 없어도 보이며, 설정된 상위 handler에도 전파된다. `SELECT`에서
backend·credential source·dotenv 경로·device를 확인할 수 있다.

`status()`에는 primary, 실제 backend, fallback_reason, api_requests, dino_calls,
cache_hits만 담는다. `OnlineVisionAdapter.records[*]['grounding']`에 같은 공개
상태를 복사하고 기존 `detector_calls`는 joint scan 횟수로 집계한다. 실제 HTTP 작업과
캐시 사용은 별도 counter로 구별한다.

각 cache/rollout episode의 `vision_trace.json`에는 카메라별 record가 저장된다.
Cache receipt/failure와 live `episode.json`에도 실제 grounding 상태를 남긴다.
기존 ExperimentLogger를 그대로 사용하며 별도 W&B/HF transport를 추가하지 않았다.

캐시 identity는 `graphs_common_coordinate_frame_v4_terra_primary`이며 공개 Grounding
정책·Prompt Hash를 포함한다. 이전 v2/v3 cache는 새 정책으로 자동 재사용하거나
덮어쓰지 않는다. 기존 cache/학습을 이번 작업에서 실행하거나 migration하지 않았다.

루트 requirements와 Visual pyproject에 requests/python-dotenv를 명시했다.
Visual lock은 `python-dotenv==1.2.4`만 새로 추가하고 기존 package version은 유지했다.
uv0.13.4로 갱신했고 `uv lock --check --offline`이 통과했다. macOS에서 Linux CUDA
Wheel 전체를 설치하지는 않았다. SAM2 metadata 준비에는 일시적인 Python3.12/
setuptools 환경을 사용했으며 저장소의 기존 환경을 재설치하지 않았다.

## 검증 결과

루트 `.venv` Python3.11에서 **83개 테스트 중 81개 통과, CUDA 전용 2개 skip**.

```bash
.venv/bin/python -m unittest \
  visual.runtime.test.test_live_grounding \
  visual.runtime.test.test_initialization \
  graph.integration_baseline.tests.test_online_vision \
  graph.plan_a_bc.tests graph.plan_a_bc.test_robot_overlap \
  graph.integration_baseline.tests.test_causality \
  graph.integration_baseline.tests.test_vision_speed -v

.venv/bin/python -m unittest \
  tests.test_experiment_logger tests.test_experiment_artifacts -v

uvx --from uv==0.13.4 uv lock --check --offline --project visual
```

첫 명령은41개 중39개 통과·CUDA2개 skip, 두 번째는42개 모두 통과했다.
테스트는 키 누락/공백·dotenv/환경 우선순위,401/403 전환, 다른 오류의 비전환,
임의 해상도 좌표 변환, API 설정, RGB 캐시, 기본 cache frontend 연결,
Streaming 늦은 물체 초기화·SAM2 메모리 유지·Reset, 중간 인증 실패 시 카메라별
prefix, 기존 DINO 회귀와 v3 캐시 거부를 확인했다.401/403은 Mock으로 검증했으며
실패 키를 실제 서비스에 보내지는 않았다.

전체 모듈을 단일 프로세스에서 합친 최초 테스트는 기존 Logger 테스트의
`importlib.import_module` Mock이 imageio의 동적 warning-registry 조회에 영향을 줘
1개 오류가 발생했다. 그 로그를 보존하고 runtime과 인프라를 별도 프로세스로 실행해
각각 통과했다. Logger 제품 코드는 수정하지 않았다.

### 실제 Terra API smoke

- 입력: 기존 `visual/data/step2/ep000.npz`의 `base_camera/rgb[0]`,512×512.
- API **1회 성공**, 실제 Terra/default tier 확인, Peg/Box 모두 유효 Pixel Box 반환.
- DINO 호출 **0회**, fallback 없음. 전체 호출 Wall 약35.55초. 단일 smoke이며
  모델 Latency benchmark나 정확도/Mask IoU 평가로 해석하지 않는다.
- 공통 ExperimentLogger UUID: `f486e3ffc21745658326b9851b0b34da`, W&B disabled.
- 새 Simulator 실행·SAM2 실모델·CUDA/MPS·전체 live policy rollout은 이 호스트에서
  실행하지 않았다. CPU synthetic streaming은 FakeTracker로 검증했다.

[실제 API 결과](../../outputs/task10i/stage4a2/live_terra_primary_20261103_v1/api_smoke.json) ·
[실제 API 로그](../../outputs/task10i/stage4a2/live_terra_primary_20261103_v1/api_smoke.log) ·
[Runtime 테스트](../../outputs/task10i/stage4a2/live_terra_primary_20261103_v1/runtime_tests.log) ·
[인프라 테스트](../../outputs/task10i/stage4a2/live_terra_primary_20261103_v1/infrastructure_tests.log) ·
[최초 혼합 실행 로그](../../outputs/task10i/stage4a2/live_terra_primary_20261103_v1/tests.log).

DINO fallback의 실모델 실행에는 기존 HF 자산과 지정 device가 필요하다. 이번
작업에서는 인증 전환·호출 연결을 Mock으로 검증했고 DINO/SAM2 Weight를 새로
다운로드하거나 CUDA에서 실행하지 않았다. Training, Git commit, 원격 배포와 외부
Artifact 게시는 수행하지 않았다.

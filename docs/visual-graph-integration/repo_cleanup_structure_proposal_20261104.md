# Repository cleanup — 목표 디렉터리 구조 제안

Task ID: `repo_cleanup_structure_proposal_20261104` · 2026-11-04 KST.

상태: **P04 구조 제안, 미확정·미구현.** 사용자의 레포 구조 질문에 대한 구체안이다.
폴더 생성·소스 이동·삭제·commit·upload는 하지 않았다. 통합 모델의 확정 구조를
공유받은 후 실제 파일 배치와 연결을 조정한다. 모델·feature·Graph schema·sequence·
학습 알고리즘·config/checkpoint schema를 이 문서로 확정하지 않는다.

개정: 사용자가 반복 개발할 `gnn/`, `controller/`, `graph/`를 분리하고 `vla/`는
공통 안정 부분과 통합 인터페이스를 맡기는 안을 제시했다. 아래는 이를 반영한
**개정 제안**이다. 전안처럼 모든 채택 코드를 vla에 넣는 구조를 최종안으로 고정하지 않는다.

기본 원칙은 실행 명령의 위치, 실제 처리 코드, 시뮬레이터·운영 인프라, 데이터 검사
도구와 과거 복원 문서의 역할을 분명히 하는 것이다. 과거 버전 소스를 대량으로 담은
새 `legacy/` 디렉터리를 기본 실행 코드 옆에 만드는 방향은 채택된 D09와 맞지 않는다.

## 1. 제안 트리

반복 개발 component와 공통/통합 package를 나누는 구조를 제안한다. 아래 이름과 세부 분할은
검토용이며 현재 사용 가능한 API나 CLI 목록이 아니다.

```text
graph-recurrent-VLA/
├── README.md                         # 설치·기본 명령·주요 문서
├── AGENTS.md
├── requirements*.txt                 # 필요한 기존 의존성·환경 재현 정보
│
├── envs/
│   └── maniskill/                    # simulator·raw GT·관측 장치 경계
├── examples/                         # simulator 확인·실행 예제
│
├── graph/                            # Graph 생성 방식의 반복 개발
├── gnn/                              # Graph를 처리하는 학습 모델의 반복 개발
├── controller/                       # 제어 모델의 반복 개발
│
├── vla/                              # 공통 안정 부분과 통합 실행
│   ├── perception/                   # 현재 채택 Terra·SAM2·유지할 DINO 대안
│   ├── contracts/                    # 합의할 최소 공통 경계; 세부 정의는 유보
│   ├── data/                         # 공통 episode·cache 입출력·검증
│   ├── integration/                  # component 조립·통합 모델 연결
│   └── workflows/                    # 공통 수집·캐시·학습·평가·live 조정
│
├── tools/                            # 사용자가 실행하는 명령
│   ├── build_graph_cache.py           # 주 실행: 1차 cache → 2차 Graph cache
│   ├── train.py                      # 주 실행: Graph cache 기반 학습
│   ├── rollout.py                    # 주 실행: 현재 관측 기반 live·필수 영상
│   ├── evaluate.py                   # 보조 실행: 독립 오프라인 평가
│   ├── collect.py                    # 보조 실행: 새 expert 원자료 수집
│   ├── build_perception_cache.py      # 보조 실행: Terra/SAM2 포함 1차 cache
│   ├── publish_experiment.py          # 기존 공통 게시 진입점 유지
│   └── discord_notifier.py            # 기존 운영 도구 유지
│
├── visualization/                    # 실행 가능한 Graph 3D 검사 도구 유지
├── utils/                            # 기존 logger·artifact·sync 인프라 유지
├── scripts/                          # 기존 인증·환경·운영 점검 스크립트
├── configs/                          # 채택된 기본 실행 설정; 형식은 추후 결정
├── tests/                            # 공통 및 채택 코드·뷰어 관련 검증
│
├── docs/
│   ├── usage.md                      # 기본 실행 흐름 안내
│   ├── architecture.md               # 모델 확정 후 갱신할 실제 구성 설명
│   ├── experiment_tracking.md        # 기존 운영 규칙
│   ├── huggingface_artifacts.md       # 기존 게시·복원 규칙
│   └── visual-graph-integration/      # 전체 연구·분석 보고서 유지
│       ├── README.md                 # 보고서·결정·복원 안내 색인
│       └── restore/                  # task별 코드/산출물 복원 안내
│
├── experiments/                      # 기존 run_id 기반 실행 기록, Git 제외
└── outputs/                          # 원자료·cache·영상·checkpoint 등, Git 제외
```

실제 의존성이 없으면 불필요하게 작은 하위 package를 더 만들지 않는다. Graph/GNN/
controller의 내부 구조와 실제 통합 모델의 연결은 공유받은 확정안에 맞춘다. 세
component의 구현을 vla/models 등에 다시 복사해 이중으로 유지하는 방식은 제안하지 않는다.
`configs/`는 새 기본 실행의 설정 위치 제안이며 모든 과거 팀 config를 통일하거나
새 schema를 강제하는 뜻이 아니다. Raw/cache/산출물의 실제 루트는 설정 가능하게
검토하며 기존 대용량 outputs/experiments를 지금 일괄 이동·복사하지 않는다.

## 2. 실행 경로를 찾는 규칙

- 기본 명령은 tools에 모으고 README에서 주 실행 세 가지와 보조 명령을 구분한다.
- Tools에는 인자 처리와 공통 workflow 호출을 둔다. Workflow/integration이 실제
  graph/gnn/controller 구현을 호출하며 component마다 중복된 기본 launcher를 만들지 않는다.
- vla/workflows는 기존 logger/publisher를 호출하며 별도 로깅·업로드 구현을 만들지 않는다.
- 내부 helper나 과거 실험 launcher를 별도의 기본 진입점으로 함께 안내하지 않는다.
- 현재 `.venv/bin/python -m visualization.server`, `tools.publish_experiment`,
  `utils.sync_experiment` 등 살아 있는 공통 운영/뷰어 경로는 필요한 호환을 유지한다.
- 명령 이름과 인자 계약은 실제 모델·현재 구현을 반영한 뒤 확정한다. 예시 파일명은
  아직 구현되지 않았으므로 현재 바로 실행 가능한 명령으로 문서화하지 않는다.

## 3. 캐시와 live의 공통 처리

Perception은 현재 관측에서 bbox·SAM2 mask를 얻는 로직과 그 공개 정책을 담당한다.
1차 cache 생성은 이를 시간 순서대로 실행해 원본 관측·기록과 연결한다.
기본은 Terra 전용이고 유지할 DINO fallback은 명시적으로 구분된 대안이다.

Root graph는 저장 mask 또는 현재 mask를 받아 같은 robot filter·wrist gate·기하/Graph
생성을 사용하는 역할이다. World/EE와 GPU 처리 경로를 유지한다. 실행 스케줄은
공통 계산과 구분해 live의 robot 준비/SAM2 overlap을 보존한다. 이 문서는 샘플링,
feature·edge·점 수나 모델 입력 schema를 새로 결정하지 않는다.

Dataset/cache 코드는 simulator 안에 넣지 않는다. Env는 환경 생성·관측·raw GT의
범위를 유지하고, train/eval/Graph 처리와는 분리한다. Graph cache는 학습에 필요한
GNN 이전 입력을 보존하는 요구를 유지하되 구체 저장 형식은 미정이다.

### 반복 개발과 통합 경계

- `graph/`: 관측·mask에서 Graph를 구성하는 알고리즘과 채택 GPU backend의 개발 영역.
  물체 표본·노드/엣지·관련 특징 생성의 구체 설계는 아직 결정하지 않는다.
- `gnn/`: Graph에서 표현을 학습하는 모델의 개발 영역. 출력 차원·readout·보조 head의
  소유 경계 등은 실제 통합 모델의 확정안에 맞춘다.
- `controller/`: 제어 모델의 개발 영역. GRU 등 특정 구조, recurrent state와 action
  표현·학습/sequence 설계를 이 폴더 분리로 강제하지 않는다.
- `vla/integration/`: 선택한 component를 연결하고 통합 모델을 구성하는 영역. 모델 정의를
  component와 중복 구현하지 않는다. 실제 입출력과 조립 방식은 공유받은 모델을 따른다.

공통 contracts는 component들이 참조할 수 있고, integration/workflows는 component를
호출하는 방향을 제안한다. Component가 vla의 최상위 통합 모델이나 실행 workflow를
역으로 import하는 구조는 순환 의존성 위험이 있으므로 피한다. 공통 경계는 최소한으로
잡고 아직 검증되지 않은 feature/embedding/sequence 규격을 미리 고정하지 않는다.

### Freeze의 의미

Vla의 공통 baseline·계약을 안정적으로 관리한다는 의미와 학습 parameter의 동결은
별개다. 어느 weight를 학습/동결할지는 모델·실험 조건의 결정이며 이번 구조안으로
고정하지 않는다. 또한 vla 전체가 변경 불가능한 것은 아니다. Integration/workflows는
합의한 component 연결을 반영해 바뀔 수 있고, 공통 부분의 변경에는 관련 검증이 필요하다.

Graph feature나 모델 입출력까지 현재 시점에 모두 고정하면 iteration이 막힌다.
반대로 과거 기준 실행의 bitwise/동작 재현을 원하면 디렉터리명만으로 freeze하지 않고
소스 commit·적용 config·checkpoint·입력 identity로 식별한다. Frozen 기준을 유지하려고
vla 안에 graph/gnn/controller의 별도 복사본을 쌓는 방식은 cleanup 목적과 맞지 않는다.

## 4. 현재 코드에서 어디를 옮길지의 판단 예

| 현재 위치 | 이동/보존 검토 방향 |
| --- | --- |
| visual/runtime의 grounding·tracking | vla/perception의 채택 코드 후보 |
| visual/runtime의 projection·PCA·sampling 및 Graph용 tensor 처리 | 책임과 의존성을 확인해 root graph 쪽에 모을 후보. Runtime 폴더 전체를 perception에 복사하지 않음 |
| plan_a_bc/geometry, plan_a_geometry/core, persistent_geometry의 채택 helper 및 GPU filter 실험 helper | 공통 Graph 생성 경로의 보존·추출 후보. 전체 과거 패키지를 그대로 새 경로에 복사하지 않음 |
| 현재 train/rollout orchestration | 최종 통합 모델과 맞춘 뒤 workflows로 모을 후보. 과거 E0/E1 모델·학습 예산을 기본으로 고정하지 않음 |
| 현재 graph encoder와 controller 모델 코드 | 채택 여부를 확인한 뒤 gnn/controller로 분리할 후보. 현재 IntegrationPolicy에 함께 들어 있다는 이유로 기존 구조·차원·학습 동결을 새 모델의 기본값으로 고정하지 않음 |
| recurrent/expert 및 수집에 필요한 기존 코드 | 새 expert 수집에 필요한 부분을 보존. Controller 비교용 과거 패키지 전체와 구분 |
| visualization/ | 현재 실행 가능한 검사 도구로 보존하고 채택 Graph/좌표 helper와 연결 |
| utils·envs/maniskill·운영 tools/scripts | 공유 운영 범위를 유지. 불필요한 경로 변경·규칙 재구현을 피함 |
| 과거 prototype·실험별 launcher/config·더 쓰지 않는 비교 구현 | 소스·전량 산출물 원격 보존·복원 검증 후 현재 tree 제거 검토 |

이 표는 파일별 확정 이동표가 아니다. 실제 통합 모델과 live·수집·뷰어·공통 도구의
전체 의존성으로 파일별 유지/이관/제거/보류 목록을 만들어야 한다.

특히 현재 `visualization/server.py`와 `data.py`는
`graph.integration_baseline.graph`의 NODE_TYPES/EDGE_TYPES를 가져오고,
`visualization/data.py`와 `persistent.py`는
`visual.runtime.vision_packet.pose_matrix`를 가져온다.
이 두 과거 경로를 먼저 제거하면 뷰어가 끊긴다. 필요한 정의·변환·로더를 보존·연결한
다음 과거 package를 제거해야 한다. 미래 Graph schema는 공유받기 전 확정하지 않는다.

## 5. 과거 실험과 실행 산출물

- 과거 코드는 Git 이력을 보존하고 docs/visual-graph-integration/restore의 task별 안내에서
  실제 code SHA, 필요 source archive, HF 고정 revision/경로, 기존 Drive 위치와 복원 명령을 연결한다.
- 산출물은 best/last만이 아니라 중간·실패·진단·원자료·metadata/sidecar까지 inventory로
  확인하고 보존한다. 단순 링크 작성이나 upload 접수만으로 삭제 가능 판정하지 않는다.
- Root experiments는 실험 소스 저장소가 아닌 기존 run_id 실행 기록으로 설명한다.
  과거 graph/experiments의 채택 runtime 코드는 보존·이관하고 나머지는 검증 후 제거한다.
- Outputs/experiments의 현재 로컬 자료를 무조건 비우지 않는다. HF 보존 및 복원 검증,
  현재 실행·뷰어·테스트의 참조를 확인한 뒤 개별 정리한다.
- 연구 전체 원문은 기존 docs/visual-graph-integration 위치에 유지하고, 복원 안내와
  기본 사용법을 별도 색인으로 찾을 수 있게 한다. 기존 상대 artifact 링크도 원격 보존
  위치나 유효한 복원 안내로 갱신해야 한다.

## 6. 실제 적용의 선행 조건

통합 모델 확정안 수령 → 현재 변경과 보존 대상 재확인 → 필요한 backup commit →
파일별 의존성/이관표와 산출물 inventory → 원격 보존·복원 검증 → 작은 단위의 이동·
기본 진입점 정리 → live/뷰어/관련 테스트·GPU/World·EE·필수 영상 검증 순서로 진행한다.

이번에는 문서와 기존 import를 확인하고 구조안을 작성했다. 새 package/CLI 생성,
코드 이동·삭제·학습·수집·GPU·viewer 실행은 하지 않았다. 문서의 local 링크만 검증하며
기존 실행/공통 인프라 테스트는 반복 실행하지 않았다.

[전체 사전 조사·확정 결정 기록](repo_cleanup_preflight_20261104.md)

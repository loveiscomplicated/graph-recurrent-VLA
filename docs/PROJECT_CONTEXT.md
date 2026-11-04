# Project context: current live baseline and cleanup

2026-11-04 사용자 결정: 별도 통합 모델 확정안은 없다. 현재 주 live를 기준 실행으로
삼고 Graph 생성·GNN·controller를 각각 교체할 수 있게 정리한다. 현재 구현은 향후
모델 설계를 고정하는 최종안이 아니다.

- `graph/`, `gnn/`, `controller/`: 각 component의 구현과 반복 개발.
- `vla/`: 최소 공통 계약, 상태 관리, component 조립, 기존 학습·평가 실행 연결.
- `tools/`: 기본 명령. 실제 사용 가능 범위는 [실행 안내](usage.md)를 따른다.
- `envs/maniskill/`: simulator 생성·관측·raw GT만 담당한다.
- `utils/`: 기존 experiment identity·local/W&B logging·명시적 artifact publishing.
- `visualization/`: 저장 Graph를 실제로 읽는 실행 가능한 검사 도구.

채택 helper는 `graph/geometry`·`graph/perception`·`graph/backends`와 `vla/perception`으로
분리했다. 이전 경로는 호환을 제공하며 기본 실행은 과거 실험·진단 package를 불러오지 않는다.
[파일별 이동과 검증](visual-graph-integration/repo_cleanup_helpers_20261104.md)을 따른다.

공통 simulator에 feature·Graph·dataset·모델 로직을 추가하지 않는다. 모델 parameter의
학습/동결은 실험 설정의 책임이며 vla 공통 코드의 안정성과 구분한다. 주 live의 robot
filter·wrist gate·GPU 처리·World/EE 선택·필수 영상 저장을 보존한다.

합의한 데이터 정책은 초기 Main strict visibility, native512, episode 시간순 원자료와
Grounding·SAM2 mask 1차 cache, Graph 2차 cache다. 이는 dataset 생성 실행이나 2단계
cache 구현 완료를 뜻하지 않는다. 기존 combined cache를 저장 mask 재생성기로 소개하지 않는다.
과거 소스·산출물은 inventory와 원격 보존·복원 검증 후 정리한다. Git 밖 원자료·ignored
실행 소스를 임의 삭제하지 않는다. [정리 보고서](visual-graph-integration/repo_cleanup_modular_live_20261104.md)를 따른다.

## 초기 독립 prototype 단계의 기록

아래는 초기 배경이다. 빈 디렉터리·통합 계약 없음이라는 설명은 현재 상태가 아니다.

This repository's shared responsibility is a working ManiSkill 3
`PegInsertionSide-v1` environment. It provides environment creation, reset/step,
rendering examples, and direct simulator ground truth access. It does not define
feature representations, model architectures, datasets, training loops,
evaluation schemes, configuration systems, or integration contracts.

The Visual team uses RGB observations to build its own visual prototype and can
compare outputs with simulator GT. The Graph team starts from simulator GT and
chooses geometry, graph construction, and a GNN prototype. The Recurrent team
collects a simulator GT sequence and chooses features and a feed-forward or
recurrent controller prototype. Each team works without waiting for another.

`visual/`, `graph/`, and `recurrent/` are intentionally empty workspaces. Teams
may organize their directories however they wish. Architecture decisions follow
the prototype results, not this shared repository's scaffold.

# 공통 계약과 통합 실행

Graph/GNN/controller의 구현을 중복 보관하지 않고 선택한 component를 연결한다.

- `contracts/`, `state.py`: 공통 Graph 형식과 controller 상태 관리.
- `integration/`: feature·presence·robot 상태를 합쳐 controller에 전달.
- `perception/`: Terra·명시적 DINO 대안·SAM2·online 추적과 관측 프레임.
- `data/`, `action_codec.py`: 기존 입출력·정규화 통계·robot schema·action 표현.
- `workflows/`: component 선택·학습·평가·live 조정·공통 실행 검사.

정규화의 기본 통계는 `data.normalization.StateStatistics`다. 과거 GT feature 전용
`StateNormalizer.fit/apply`는 기존 prototype subclass에 유지해 기본 경로가 과거 feature
구현을 불러오지 않도록 했다. 통계와 JSON/checkpoint의 수치 형식은 유지한다.

사용자 명령은 `python -m tools.train/rollout/evaluate`다. SAM2의 기본 외부 weight는
기존 `visual/checkpoints` 위치를 계속 사용하며 이번 이동에서 데이터를 복사하지 않는다.
기존 combined cache generator는 저장 mask 기반의 새 2단계 cache 구현이 아니다.

[실행](../docs/usage.md) · [교체 계약](../docs/architecture.md) ·
[helper 분리 보고서](../docs/visual-graph-integration/repo_cleanup_helpers_20261104.md).

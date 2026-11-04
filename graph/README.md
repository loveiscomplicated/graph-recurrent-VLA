# 현재 Graph 생성 코드

기본 builder는 `graph.current.CurrentGeometry`다. 저장 관측·mask와 live 관측·mask가
같은 builder를 선택하며, GNN 이전의 Graph를 만든다.

| 위치 | 내용 |
| --- | --- |
| `geometry/` | Camera/Surface 계약, 기하·records·robot filtering·wrist gate·Graph 구성 |
| `perception/` | projection·PCA·sampling·packet·legacy 비교에 필요한 계산 함수 |
| `backends/` | CUDA robot 준비와 filtering, native C++, 선택 resident backend |
| `vision_input.py` | 기존 packet receiver·검증 |

GNN은 root `gnn/`, 제어 모델은 `controller/`, 통합·실행은 `vla/`에 있다.
현재의 feature·선택 규칙·좌표계·GPU 스케줄을 이관했으며 새 정책으로 바꾸지 않았다.
Geometry protocol과 native C++의 원본 바이트도 보존했다.

`plan_a_geometry`, `persistent_geometry`, 일부 `experiments` leaf의 이전 import는
현재 구현으로 위임한다. 이곳의 실험 CLI·설정·보고서는 원격 보존과 복원 검증이
끝날 때까지 유지한다. 폴더 이름만 보고 전체 삭제하지 않는다.

[실행](../docs/usage.md) · [교체 계약](../docs/architecture.md) ·
[파일별 이관·검증 보고서](../docs/visual-graph-integration/repo_cleanup_helpers_20261104.md).

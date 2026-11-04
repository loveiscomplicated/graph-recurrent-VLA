# 현재 live의 모듈 경계

별도의 모델 확정안을 기다리지 않고 현재main live를 기준으로 분리한다. Graph/GNN/
controller는 각각root package에 구현하고 vla가 공통 계약과 통합을 맡는다.
기본E0/E1, Graph21, robot76, 7값action은 **현재baseline**이며 미래 설계의 최종안이 아니다.

```text
관측 + mask → graph.current.CurrentGeometry → vla.contracts.Graph
                                               ↓ pack_graphs
                                      gnn.edge_mlp.GraphEncoder
                                               ↓ feature + presence
robot 상태 → normalizer ────────────→ vla.integration.IntegrationPolicy
                                               ↓ fused sequence
                                     controller.recurrent.RNNPolicy
                                               ↓ action + state
                                      기존 codec / 제어 / 평가
```

`vla/workflows`는 cache 읽기·학습·offline 평가·live를 조정한다. `tools`는 얇은 실행
진입점이다. Component가workflow나 최상위 통합 모델을 역으로import하지 않는다.
공통Graph형과packing은 `vla/contracts`, 상태의reset/detach/slice/stack은 `vla/state.py`에
두며 GNN·controller의 구현을 vla에 다시 복사하지 않는다.

## Component 교체

기존 `config.json`에 선택적으로 `components`를 추가한다. 생략하면 모두 현재baseline이다.
Factory는 신뢰하는project Python 코드의 `module:attribute`로 지정한다.

```json
{
  "components": {
    "graph_builder": "my_project.graph:make_builder",
    "graph_encoder": "my_project.encoder:make_encoder",
    "controller": "my_project.control:make_controller"
  }
}
```

my_project는 구현자가 만들 예시이며 이 repo에 존재하는 module이 아니다.
`vla.workflows.components.LiveComponents`에 Python factory를 직접 전달할 수도 있다.
실행 가능한 교체·학습 예시는 [테스트](../tests/test_modular_live.py)에 있다.

| Component | Factory와 반환값의 계약 |
| --- | --- |
| Graph builder | `factory(assets, schema, coordinate_frame=...)`. `build/from_h5/reset/close`와 기존 진단 속성. `build`는raw mask 사전 또는producer를 받아 Graph·records·gate를 반환 |
| GNN | `factory(model_config)` → `nn.Module`. `forward(packed_graphs, trace=None)` → `[G,D]`feature와 `[G,P]`presence. `output_dim=D`를 공개하거나 `graph_feature_dim`을 설정 |
| Controller | `factory(input_dim, model_config)` → `nn.Module`. `forward([B,T,F], state)` → `[B,T,7]`action과state. state는Tensor/tuple/list/dict/None |

`presence_dim` 기본2. `state_batch_dim`은controller 속성을 우선하고 없으면1이다.
모든state leaf가 같은batch 축을 쓰는 전제다. GRU/LSTM은1, batch 선두state는0을
설정한다. 통합부는controller의 `input_dim`·`action_dim` 속성이 있으면 검사한다.
형상 불일치·sequence의구멍·World/EE 혼합을 거부한다. 현재학습은7값action BC이므로
다른action 공간이나loss로 바꾸려면codec/학습 경계도 변경해야 한다.

Factory 선택identity를checkpoint와로그에 기록하고 재개/평가 시 대조한다. Cache에는
Graph builder 선택을 기록한다. Factory 이름은source revision을 대신하지 않으므로
재현에는Git commit·config/hash·입력cache도 필요하다. Identity 없는옛checkpoint는
기본component만 허용하며 state_dict key를 유지해 엄격히로드한다. E1의message layer
제거는기본GNN에만 적용하고 교체encoder 내부를 수정하지 않는다.

## 현재Graph·perception 보존

Graph facade는 `graph/current.py`이며, 채택 helper의 실제 구현은 다음에 있다.

| 위치 | 역할 |
| --- | --- |
| `graph/geometry/` | CPU 기준 기하, robot filter·wrist gate, records, CPU/CUDA Graph 조립 |
| `graph/perception/` | RGB-D projection·PCA·Shape-First·voxel/FPS·packet 구성 |
| `graph/backends/` | CUDA robot filter·사전 준비, native C++, 선택 resident backend |
| `vla/perception/` | Terra/DINO 초기화·SAM2·online adapter·관측 프레임 |
| `vla/data/`, `vla/action_codec.py`, `vla/workflows/` | 상태 통계·robot schema·입출력·action codec·RNG와 live audit |

기본 tools/cache 경로와 뷰어는 과거 실험/진단 package를 역으로 import하지 않는다.
이전 leaf module은 같은 구현의 alias 또는 추출 함수의 re-export를 제공한다. 과거
CLI와 보고서 코드는 원래 위치에 유지하며, 과거 package 자체를 삭제한 것은 아니다.
전체 source 의존성과 이관표는 [helper 분리 보고서](visual-graph-integration/repo_cleanup_helpers_20261104.md)를 따른다.

GPU filter/준비·wrist gate·sampling/edge·World/EE 계산과 mask producer를 통한 overlap을
유지한다. 고정 geometry protocol과 native C++는 바이트 동일하게 옮겼다. SAM2 외부
weight의 기존 위치도 유지한다. 실제 전체 live 성능 비교는 별도로 실행해야 한다.

Perception policy 진입점은 `vla/perception/grounding.py`다. 기본 `terra_only`이며,
옛auth fallback을 사용할 때만 `grounding.policy=terra_auth_fallback`을 명시한다.
SAM2/adapter는 이관한 동일 구현을 재사용한다. 모델 weight의 일괄 freeze는 추가하지 않았다.

뷰어는Graph 상수·좌표변환 공통부를 직접 참조하고 기존 저장형식과 검사 기능을 유지한다.
과거코드 전체삭제·2단계cache·새expert 수집·원격복원 확인은 남은작업이다.

# Task10-I — 로봇 준비/SAM2 겹치기를 기본 경로에 연결

2026-11-03. 앞선 구현은 선택적 SAM2 연결 함수 안에서 동작했지만 기본 Graph
호출자는 마스크를 먼저 완성하고 있었다. 이번에는 **기본 `CurrentGeometry`와
Plan A의 캐시 생성·실시간 실행 진입점**을 전환했다. 별도 opt-in flag나
`GpuMaskBoundary` context를 붙일 필요가 없다.

## 실제 기본 흐름

```text
현재 센서·FK 입력
    ├─ CPU 준비 제출: world shape / ConvexHull / 투영 bounds
    └─ 기존 OnlineVisionAdapter의 현재 프레임 DINO/SAM2
                   ↓ CUDA raw masks
    준비 결과 합류 → 기존 CUDA 로봇 필터 1회
                   ↓
    원본 raw + clean mask로 기존 wrist gate → 기존 점 선택/Graph
```

`CurrentGeometry(..., perception_backend='triton_fused')`의 로봇 필터 기본값은
이제 `robot_filter_backend='cuda'`다. 첫 프레임에 persistent preparation worker와
filter를 만들고 다음 프레임에서도 재사용한다. 현재 pose/depth/hull을 프레임 간
재사용하는 캐시는 없다. `build`에 넘긴 zero-argument mask producer를 호출하기
**전에** 현재 입력의 준비를 제출한다. 추론·준비 예외에서 pending 작업을 정리한다.

기본 `graph/plan_a_bc/cache.py`와 `rollout.py`는 다음 순서를 직접 사용한다.

```python
geometry = CurrentGeometry(assets, schema)
try:
    def current_masks():
        return {c.camera: adapters[c.camera].process(
            c.rgb, episode, timestep, mask_device='cuda')[0] for c in cameras}
    graph, records, gate = geometry.build(cameras, data, surfaces, current_masks)
finally:
    geometry.close()
```

`OnlineVisionAdapter.process(..., mask_device='cuda')`는 알고리즘·prefix·prompt·
모델을 바꾸지 않고 raw mask를 CUDA tensor로 반환한다. 필터 전에 dense mask를
CPU로 내려 다시 올리지 않는다. 기존 두 개의 pixel-count scalar만 기록용으로
읽는다. 다른 기존 호출자의 반환 형식을 보존하기 위해 adapter 단독 기본값은
여전히 CPU이며, 위 **기본 Graph 진입점들이 CUDA 출력을 요청한다.**

필터 뒤 CPU gate가 필요한 raw/clean mask를 내려 받는다. `PlanA.prepare`의
`clean_masks=` 인자는 이를 받아 두 번째 로봇 분류를 생략한다. Raw mask는
gate 분모와 원본 통계에 그대로 사용한다. Clean mask의 dtype/shape/subset/
overlap을 확인한다. 이 private fast path의 `filters[camera]`는 `None`이며
계산하지 않은 full-screen 진단을 제공하는 척하지 않는다. Standalone
`PlanA.prepare`의 기존 전체 진단 경로는 유지된다.

`geometry.last_raw_masks`는 저장·영상용 CPU **raw mask**이고,
`last_robot_schedule == 'before_vision'`가 기본 두 진입점에서 기록된다.
캐시 receipt와 실시간 trace에도 backend/schedule을 남긴다. 종료 시 worker를 닫는다.

## 호환성과 범위

- 이미 완성된 mask dictionary를 넘기는 기존 `build`도 새 기본 CUDA 필터를
  사용한다. 이 경우 schedule은 `masks_already_available`이며, 끝난 SAM2와
  소급해서 겹쳤다고 주장하지 않는다. 고정 입력·과거 pinned runner를 재작성하거나
  원래 정지 실험을 재개하지 않았다.
- `robot_filter_backend='cpu'`는 직렬 필터 비교/진단용 명시적 선택이다.
  `perception_backend='cpu'`에서는 CPU 필터가 자동 선택된다. CUDA가 없는
  기본 CUDA 경로는 조용히 CPU로 바꾸지 않고 오류를 낸다.
- 작업 트리의 별도 선택적 `cuda_resident` prototype 변경은 이번 커밋에 포함하지
  않는다. 이번 채택은 기본 `triton_fused` 캐시·실시간 진입점에 대한 것이다.
- 기존 `GpuMaskBoundary`를 중복으로 씌워도 이미 clean mask가 제공됐으면
  필터를 다시 실행하지 않도록 호환했다. 기본 실행에는 이 context가 불필요하다.
- 캐시 identity는 `graphs_common_coordinate_frame_v3_robot_overlap`로 갱신했고
  backend/schedule도 포함했다. 기존 v2 cache는 자동 덮어쓰기/재사용 대신 거부한다.
- Robot tolerance2.5mm, uncertainty6mm, interior margin0.5mm, link-order/ties,
  raw overlap 제거와 gate·점 선택·Graph 규칙은 유지했다. 물리 backend나 SAM2
  모델 내부를 이번 변경에서 수정하지 않았다.

## 검증

최종 원격 RTX3090 suite **42개 모두 통과**했다. 로컬 macOS의 관련 CPU suite는
**46개 중44개 통과, CUDA 전용2개 skip**이었다.

- 실제 `cache_episode`/`run_episode` 함수 본체를 synthetic 의존성과 함께
  호출하여 마스크 생산이 build 내부로 지연되고 CUDA mask 출력이 요청되는지
  확인했다. 물리/정책 rollout을 실행했다는 뜻은 아니다.
- 기본 생성자에서 CPU 준비가 vision callback보다 먼저 시작되는지와 vision
  실패 후 pending 정리를 검사했다. 새 backend를 opt-in하지 않은 상태다.
- 실제 CUDA `CurrentGeometry`를 World/EE 모두에서 직렬 CPU robot filter
  reference와 비교했다. Graph 이산 필드, source IDs, gate, raw mask가 일치했고,
  XYZ/edge attr1e-6, F5 2e-5, validity/availability exact 계약을 통과했다.
- `classify_robot`가 다시 호출되면 실패하도록 막아 단일 필터 적용을 확인했고,
  원래 raw gate 분모도 검증했다. Legacy context 호환 검사도 통과했다.
- Online adapter의7개 prefix에서 늦은 물체 초기화와 상태 메모리를 포함해
  CUDA mask 출력이 기존 CPU 출력과 정확히 일치했다. CUDA 경로에서는 필터
  이전 dense mask `.cpu()`가 호출되면 실패하도록 검사했다.
- 기존 filtering/no-readback/owner/depth 경계, PCA/packet/view consistency,
  tracking 준비 순서/오류 정리와 기본 schema 계약을 함께 통과했다.
- v2 cache를 새 identity로 잘못 재사용하지 않는지 검사했다.

실제 SAM2 weights 추론, 실제 시뮬레이터 폐루프, 학습·데이터 전체 재생성,
전체 inference latency 측정은 실행하지 않았다. **기본 경로 연결은 완료했지만
새 end-to-end 절감량은 미측정**이다. 이전3.82ms CPU 준비 시간이나7.16ms 필터
단독 시간에서 새 전체 시간을 빼거나 추정하지 않는다. CUDA 검증은 기존 원격
환경에서 했으며 로컬 macOS CUDA는 미실행이다.

## 재현성과 산출물

기존 `ExperimentLogger`를 재사용했고 W&B disabled, publishing 없음이다.
원격 root: `/workspace/task10i_robot_overlap_default_20261103_v1`.
원래 정지된 7개 Task10-I PID는 마지막 확인에서도 모두 T 상태였다.

- [최종 검증 결과·runtime/source identity](../../outputs/task10i/robot_overlap_default_20261103_v1/remote/final_v2/results.json)
- [최종 CUDA42 테스트 로그](../../outputs/task10i/robot_overlap_default_20261103_v1/remote/final_v2/tests.log)
- [로컬 CPU 테스트 로그](../../outputs/task10i/robot_overlap_default_20261103_v1/tests_cpu_final.log)
- [실행 전 source snapshot identity](../../outputs/task10i/robot_overlap_default_20261103_v1/source_identity.json)
- [최종 변경 파일 identity](../../outputs/task10i/robot_overlap_default_20261103_v1/final_identity.json)
- [기본 API와 사용 예](../../graph/plan_a_bc/README.md)

기존 CUDA 환경에서 최종 명령과 source hash는 `results.json`에 기록했다.
별도 인스턴스·설치·외부 모델 API·학습·게시·이전 실행 재개는 하지 않았다.

# Task 10-I: 손목 카메라 gate 연결성 최적화 시제품 검증

2026-11-03 KST. **로컬 독립 시제품 검증 완료. 기존 서버 실험에 적용하거나 재개하지 않았다.**
이 문서는 별도 진단 보고서이며, 원래 네 해상도 인식·Graph 비교의 최종 결과 보고서가 아니다.

## 문제와 변경 범위

기존 `graph/plan_a_geometry/core.py:components`는 손목 Peg의 유효 3D 점에 대해
`cKDTree.query_pairs(0.012)`로 모든 반경 이웃 쌍을 생성하고, COO 인접 행렬을 만들어
연결 성분을 구한다. 최종 물체당 128점 제한은 이 검사 뒤에 적용된다. 고해상도에서
점 밀도와 이웃 수가 함께 증가해, 최종 출력보다 중간 이웃 쌍 저장량이 훨씬 커진다.

독립 실험 폴더에 `radius_components.py`와 `test_radius_components.py`만 추가하고
실험 README에 사용 경계를 설명했다. 공통 simulator, PlanA, Graph, 기존 인터페이스,
서버 동결 소스와 protocol은 변경하지 않았다. 원래 실패 시도와 수정 시도도 보존했다.

## 구현 원리와 동등성

1. 모든 입력 점을 그대로 유지한다. voxel 대표점 선택이나 downsampling은 하지 않는다.
2. 변 길이가 radius/2인 셀로 분류한다. 같은 셀의 최대 거리는 sqrt(3)/2×radius로
   radius보다 작으므로, 셀 내부는 원래 반경 그래프에서도 반드시 한 연결 성분이다.
3. 연결 가능성이 있는 셀 사이만 검사한다. 원래 좌표와 radius로 KD-tree
   `count_neighbors`를 호출해 연결 쌍이 하나라도 있으면 union-find로 합친다.
   셀 사이 연결 여부는 거리 근사나 셀 중심 거리로 대체하지 않는다.
4. 이미 같은 성분인 셀은 다시 연결할 필요가 없다. 결과 라벨은 최초 입력 점 순서로
   정규화해, 최대 성분 크기가 같을 때도 기존 `counts.argmax()`의 선택을 유지한다.

연결 성분 판정에 필요 없는 전체 엣지 목록과 COO/CSR 변환 배열을 만들지 않는 것이
핵심이다. 공간 셀은 계산용 구획이며 원래 점, mask, 거리 기준, 선택 픽셀을 변경하지 않는다.
작업 메모리는 입력 점과 셀 수에 비례하는 구조다. 모든 점 분포에서의 실행 시간이나
임의의 극단적 좌표 범위까지 보장하는 범용 구현으로 검증한 것은 아니다.

## 검증

실행 환경: Mac CPU, Python 3.11, NumPy 2.4.6, SciPy 1.17.1.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest graph.experiments.resolution_perception_graph_20261103_v1.test_radius_components -v
```

**21개 PASS:** 기존 PlanA 계약 16개와 새 비교 테스트 5개.
새 테스트는 빈 입력·중복점·분리된 같은 크기 성분·입력 순서 변경·40 seed와 4개 반경의
무작위 군집·거리 경계의 nextafter·셀 경계·평면·연쇄 연결을 포함한다.
기존 함수와 라벨 배열을 정확 비교했고, 정상·빈 마스크·다른 객체 충돌·분리된 support에서는
전체 `wrist_gate`의 선택 mask와 trace dictionary도 정확히 일치했다.

실제 수정 실행에서 이미 저장된 성공 사례 3건의 센서·실제 SAM mask·Graph stage 기록
15개 파일만 읽어 로컬에 복사했다. 전송량은 압축 5,579,715 bytes다. 모델 호출이나 서버
Graph 생성은 하지 않았다. 세 사례 모두 기존 gate 기록의 raw pixel 수와 clean point 수가
같고 모든 해당 점의 depth가 유효함을 확인해, 원래 gate가 사용한 입력 점을 재구성했다.
GT 파일은 사용하지 않았다.

| 입력 사례 | 유효 점 수 | 원래 반경 이웃 쌍 수 | 새 연결성 함수 중앙값 ms | 기존 선택 픽셀과 일치 |
|---|---:|---:|---:|---|
| 0924/t80, 128 | 8,295 | 3,378,727 | 8.43 | 정확히 일치, 8,295점 |
| 0844/t80, 512 | 61,056 | 336,108,437 | 55.12 | 정확히 일치, 61,052점 |
| 0531/track t11, 1024 | 119,380 | 318,049,409 | 92.28 | 정확히 일치, 119,378점 |

세 사례에서 새 함수는 각각 `[8295]`, `[61052, 2, 1, 1]`, `[119378, 2]` 크기의 성분을
반환했다. 최대 성분의 원본 source pixel 집합이 저장된 C3 필터 출력과 같았다.
반경 이웃 쌍 수는 scalar KD-tree count로 계산했으며 목록으로 생성하지 않았다.

시간은 연결성 함수만 각 3회 실행한 예비 측정이다. 128 사례에서는 기존 함수도 같은 Mac에서
3회 실행했고 라벨 전체가 정확히 일치했다. 기존 중앙값은 **107.72ms**, 새 함수는
**8.43ms**였다. 512/1024의 기존 함수는 수억 쌍을 재할당하지 않도록 재실행하지 않았으며,
저장된 선택 픽셀과 비교했다. 두 함수의 번갈아 실행 등 정식 지연 프로토콜은 수행하지 않았다.

별도 합성 평면 smoke에서는 120×80mm 평면의 모든 raster 점을 유지했다.

| raster 크기 | 점 수 | 새 연결성 함수 단일 실행 초 |
|---|---:|---:|
| 128 | 16,384 | 0.022 |
| 512 | 262,144 | 0.236 |
| 768 | 589,824 | 0.581 |
| 1024 | 1,048,576 | 1.159 |

모두 하나의 연결 성분을 반환했다. 합성 측정 프로세스의 누적 RSS peak는 404,209,664 bytes다.
이는 Python/SciPy·입력·임시 배열과 앞 해상도 실행의 high-water를 포함하며, 함수만의 순수
추가 할당량이나 Linux 서버 RSS 측정이 아니다. 실제 입력 검증 프로세스 peak
632,078,336 bytes 역시 로드한 입력과 의존성을 포함한다.

## 실행 경계와 남은 확인

- 기존 서버 batch는 정지 상태로 유지했다. API 추가 호출, 학습, rollout, 설치, 게시를 하지 않았다.
- CPU 연결성 진단이며 기존 학습/평가 entrypoint나 ExperimentLogger identity 규칙을 변경하지 않았다.
- CUDA SAM2, CPU PhysX/SAPIEN, 전체 서버 Graph builder는 이번 시제품 검증에서 실행하지 않았다.
- 새 함수의 ms는 전체 gate나 전체 Graph 시간이 아니다. 로봇 필터, local support 검사,
  cross-camera 검사 등 기존 비용은 남는다. 과거 서버의 병렬 상세 계측과 Mac 함수 시간을
  나누어 전체 속도 개선율로 발표하지 않는다.
- 실제 성공 3건은 모든 장면이나 OOM 사례의 동등성을 증명하지 않는다. 서버 적용 전에는
  별도 검증 실행으로 더 넓은 실제 clean support의 성분·gate·Graph tensor 동등성을 확인해야 한다.
  기존 실험을 단순 SIGCONT하거나 소스를 덮어써서 혼합하면 안 된다.
- 추가 framework, JPEG/RSS 계측 확장, 자동 메모리 복구 등은 구현하지 않았다.

## 코드와 원자료

- [시제품 함수](../../graph/experiments/resolution_perception_graph_20261103_v1/radius_components.py)
- [동등성 테스트](../../graph/experiments/resolution_perception_graph_20261103_v1/test_radius_components.py)
- [실제 입력 검증·시간·소스 및 입력 해시](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/gate_optimization_probe/verification.json)
- [합성 평면 smoke](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/gate_optimization_probe/synthetic_smoke.json)
- [선택 입력과 원본 서버 경로](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/gate_optimization_probe/selected_cases.json)

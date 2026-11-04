# Task10-I: 현재 GPU 손목 gate의 추가 최적화 검토

2026-11-04 KST. 기준 commit: `ef3485c` 및 현재 작업 트리의 기본 gate 호출 경로.
**코드와 보존된 측정 자료만 검토했다. 코드 수정, gate/모델 실행, 새 성능 측정,
테스트 재실행, 원격 작업, commit은 하지 않았다.** 이 검토 문서만 추가했다.

## 판단

**추가 최적화 가능성이 있다. 우선순위는 같은 프레임의 중복 준비 제거와
GPU→Python 동기화 감소다.** NumPy 보정 경로에는 실제 중복 업로드도 있다.
그 다음은 셀 정렬·연결성 반복과 희소점의 국소 지지 검사다.

이 순서는 코드상 중복 여부와 판정 변경 위험을 기준으로 한 검토 의견이다.
현재 원자료는 전체 gate 시간만 기록했으므로 어느 단계가 주 병목인지,
각 변경으로 몇 ms가 줄어들지는 알 수 없다. 아래 항목은 제안이며 구현·검증된
새 개선 결과가 아니다.

## 먼저 검토할 항목

| 우선 | 후보 | 코드에서 확인한 비용 | 판단 동등성 위험 |
| --- | --- | --- | --- |
| 1 | 같은 프레임의 카메라/view 준비 재사용 | 반복 metadata 검증, 행렬 계산·업로드, view 재생성 | 낮음–중간: World/EE와 현재 입력 identity 유지 필요 |
| 1 | NumPy 보정 경로의 중복 업로드 제거 | 손목 clean mask를 같은 호출에서 두 번 업로드 | 낮음: 원래 배열 관계 확인 필요 |
| 2 | trace readback·검증 동기화 묶기 | `tolist`, `float`, `int`, camera별 `bool` 반환 | 낮음–중간: 전체 trace와 오류 검증 유지 |
| 3 | 셀 정렬·후보 생성 줄이기 | 같은 key의 unique와 별도 sort, 과다한 이웃 셀 후보 | 중간: 거리 경계·원본 순서 tie 유지 |
| 3 | 연결성 반복의 host 왕복 줄이기 | 매 반복마다 `torch.equal`로 종료 여부를 Python에서 확인 | 중간–높음: 완전 수렴과 성분 tie 유지 |
| 4 | 희소점 국소 지지를 공간 질의로 변경 | 미확정 K점에 대해 전체 N점 거리 계산, O(KN) | 중간–높음: 이웃 3개 기준과 경계 유지 |

### 1. 카메라/view 준비 중복

[tensor_gate.py](../../graph/persistent_geometry/tensor_gate.py)의
`prepare_tensor_gate`는 자체 `synchronized` 검증 후 `prepare_tensor_views`를
호출한다. 그 안에서도 `synchronized`가 실행된다. 이어 `wrist_gate_tensor`는
전달된 `views`에서 depth만 꺼내고 **World view를 다시 준비**한다
(현재 소스 52–53, 98, 111–112행).

[prepare_tensor_views](../../graph/persistent_geometry/tensor_fusion.py)는
카메라 검증, frame 변환 행렬 계산 및 작은 calibration tensor 업로드를 한다.
따라서 depth를 재사용해도 나머지 준비가 반복된다.
`masked_world_points`도 준비된 view가 있는데 `camera.world_from_camera`를 다시
계산·업로드한다. 이 property는 CPU 행렬 역변환을 수행한다.

검토 후보는 프레임 단위로 검증한 calibration/depth를 공유하고, gate용 World
변환과 Graph용 World/EE 변환을 같은 입력 snapshot에서 준비하는 것이다.
**EE view를 World gate에 그대로 대입하면 좌표계가 달라지므로 안 된다.**
Episode/time/pose뿐 아니라 현재 depth·calibration과의 결합을 유지해야 하며,
이전 프레임 결과를 timestamp만으로 재사용하는 cache는 동등성 근거가 없다.

### 2. NumPy 보정 경로의 확실한 중복

[wrist_gate_dispatch.py](../../graph/persistent_geometry/wrist_gate_dispatch.py)의
21행에서 `clean_all`의 모든 카메라를 CUDA로 변환한 다음, 22행에서 손목 항목을
`clean_masks`의 CUDA 변환으로 덮어쓴다. 현재 `PlanA.prepare` 호출에서는
`clean_masks`가 바로 `clean_all[wrist]`다. CPU ndarray 입력이면 같은 손목 clean
mask에 대한 업로드·할당이 반복된다. 512의 bool `[2,H,W]`는 0.5 MiB다.

또한 이 어댑터는 depth view를 전달하지 않아 호출마다 원래 sensor depth의
GPU 준비/변환을 수행하고, 마지막에 accepted mask를 CPU로 내린다. 512의
accepted bool `[H,W]` readback은 0.25 MiB다.

[Task10-I 보정](../../graph/task10i_stage4a2/tri_model_integration.py)은 같은 프레임에서
현재 마스크와 여러 trial 마스크로 `system.prepare`를 호출할 수 있다. 마스크가
다르면 전체 prepared-result memo가 적중하지 않으므로, **불변 sensor/calibration과
변하는 후보 mask를 나누어 재사용할 여지**가 있다. 현재 memo가 이미 있으므로
동일 입력에 대해 gate가 항상 다시 실행된다고 해석하지는 않는다.

이 항목은 보정 후보 경로의 비용이다. 입력이 이미 GPU에 준비된 기존 5.79 ms
측정에는 초기 업로드와 최종 검증 readback이 포함되지 않았으므로, 그 수치에서
직접 절감될 비용과 구분해야 한다.

### 3. host 동기화와 동적 배열 크기

`tensor_gate.py`에는 다음 지점이 있다.

- 셀 크기 `dims.tolist()` (12행).
- 연결성 반복의 `torch.equal(old, labels)` (34행).
- support 및 rare 점의 `torch.nonzero` (55, 61행).
- 통계 `summary.tolist()` (67행), diameter의 `float(...)` (72행),
  accepted count의 `int(accepted.sum())` (88행).
- `prepare_tensor_gate`의 카메라별 subset/overlap `bool(...)` (110행).

설치된 Torch 문서에서도 CUDA `nonzero`의 host-device synchronization을
명시한다. 나머지 Python scalar/목록 반환도 GPU 결과가 host에서 필요해지는
지점이다. Boolean indexing·unique 등 동적 크기 배열 생성도 추가 비용 후보지만,
정확한 동기화 횟수와 각 비용은 profiler 없이 세지 않았다.

낮은 위험의 후보는 gate 통계·diameter·채택 후보 수를 GPU에서 계산한 뒤
묶어서 한 번 읽는 것이다. 현재 accepted mask는 판정이 통과하면
`largest & allowed`, 거부하면 0이다. 따라서 그 후보 수를 앞서 구해 두면
마지막 `accepted.sum()` readback을 피할 수 있다. 전체 trace 값은 유지해야 한다.

Subset/overlap 검사에서 `v[:, r[0] & r[1]]`로 동적 배열을 만든 뒤 `.any()`를
계산하는 부분도 고정 크기 Boolean reduction으로 표현할 여지가 있다.
검증 자체를 빼는 최적화는 제안하지 않는다.

연결 성분 반복은 매번 작은 Torch 연산 여러 개와 host 종료 검사를 수행한다.
몇 차례 device 반복 후 종료를 확인하거나 전용 연결성 kernel로 묶는 방향을
검토할 수 있다. 임의의 고정 반복 횟수로 잘라 버리면 긴 연결 사슬에서 결과가
달라질 수 있다.

### 4. 셀 정렬과 이웃 후보

`components`는 셀 code에 `unique(..., sorted=True)`를 적용한 뒤 같은 code를
다시 stable `argsort`한다. 한 번 정렬한 결과에서 연속 셀 구간·count·inverse를
구성하면 반복 정렬을 줄일 가능성이 있다. 끝의 `unique(first[roots])`와 뒤따르는
`bincount`도 gate에 필요한 최대 성분만 결정하는 목적에 맞춰 검토할 수 있다.
단, 현재 API를 사용하는 비교 테스트와 원본 첫 점 순서의 tie 규칙을 보존해야 한다.

현재 이웃 offset은 `[-3,3]^3`의 절반으로 **171개**이며 매 호출 Python에서
구성해 GPU tensor를 만든다. 이 상수 표를 재사용하는 변경은 작고 위험도 낮다.

더 큰 후보 축소의 기하학적 근거도 있다. 셀 변이 `s=r/2`일 때 offset `d`의
셀 간 거리 하한은 다음과 같다.

`s² × Σ max(|dᵢ|−1, 0)²`

합이 4보다 큰 offset을 먼저 제외하고 **경계인 4는 남기면** 171개 중
89개가 남는다. 이는 저장 입력의 gate를 실행한 결과가 아니라 상수 offset
집합에 대한 산술 검토다. 실제 radius kernel 전에 생성하는 이웃 좌표·검색 후보를
줄일 여지가 있다. 현재도 이후 AABB 검사로 후보를 거르므로, kernel 검사 수가
같은 비율로 줄거나 전체 gate가 48% 빨라진다는 뜻은 아니다. 부동소수점 셀 경계에
대한 보수성 검증도 필요하다.

### 5. 희소점의 국소 지지

같은 셀에 3점 이상 있으면 현재 코드가 이미 지지를 확정한다. 나머지 rare 점만
128개씩 묶어 **전체 N점과 FP64 거리를 계산하고 모두 센다** (62–65행).

판정은 이웃이 3개 이상인지이므로, 공간 셀을 재사용해 관련 이웃만 방문하고
3개를 찾으면 그 점의 검사를 끝낼 수 있다. 점을 줄이는 방법은 필요하지 않다.
셀 내 지지가 이미 확정되는 밀집 입력에서는 이 변경의 효과가 작을 수 있다.
기존 결과는 rare 점 수·실제 검사량을 기록하지 않아 29,775점 사례에서 이 단계가
주 병목인지 알 수 없다. `local_support_fraction`은 최종 지지 판정 비율이므로
rare 점 비율을 대신하지 않는다.

## 바로 선택하기 어려운 변경

- FP64→FP32, FMA 활성화, 거리 식의 계산 순서 변경은 반경/깊이 경계와 trace를
  바꿀 수 있다. 현재 radius kernel의 `enable_fp_fusion=False`도 고려해야 한다.
- `sqrt(d²) <= r`를 단순히 `d² <= r²`로 바꾸는 것도 경계 반올림 검증이 필요하다.
- 점 downsampling, point cap, 반경/threshold 완화는 현재 정확 비교의 전제를 바꾼다.
- 낮은 support 비율에서 조기 반환하면 판정 여부는 같아도 전체 trace와 복수 거부
  이유를 계산하지 못할 수 있다. 현재 trace 계약을 유지한다면 제약이 있다.
- Triton tile 크기, 전용 union-find, kernel fusion은 가능성이 있지만 단계별
  시간·cell 점 수 분포·반복 횟수 근거 없이 먼저 선택할 이유는 부족하다.

## 기존 측정에서 확인한 주의점

원자료 15개를 순서별로 다시 묶으면 아래와 같다. **새 측정이 아니다.**

| 원래 pair 순서 | 표본 수 | GPU 시간 중앙값 |
| --- | ---: | ---: |
| CPU → GPU | 8 | 6.601340 ms |
| GPU → CPU | 7 | 5.071680 ms |

전체 중앙값 5.790390 ms는 유지된다. 실행 순서와 시간의 차이가 관측되지만
공유 호스트 상태, GPU clock, CPU launch 지연 등 어느 원인인지는 이 자료로
특정할 수 없다. 다음에 성능을 검증한다면 같은 입력·출력 조건과 함께 실행 순서,
resident gate와 통합 준비 경계, CPU wall/GPU 실행 구간을 구분해 볼 필요가 있다.
현재 자료만으로 목표 시간을 3 ms나 1 ms로 제시할 근거는 없다.

## 결론 및 범위

첫 검토 대상은 **프레임 준비 중복과 NumPy 경로의 중복 업로드**, 다음은
**trace/검증 readback과 연결성 반복의 동기화**다. 그 뒤 정렬·이웃 셀 후보 축소를
검토하고, 희소 입력의 거리 계산은 실제 처리량에 따라 우선순위를 정할 수 있다.

이번 작업은 정적 검토와 기존 JSON 값의 재집계, 상수 offset 산술에 한정했다.
Simulator, CPU/GPU gate, 모델 또는 전체 live 경로를 새로 실행하지 않았다.
성능 향상 수치와 새로운 판정 동등성 결과는 제시하지 않는다.

근거:

- [기존 GPU gate 측정 보고서](10i_wrist_gate_cuda_first_step_20261104.md)
- [15회 원자료](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/results.json)
- [측정 경계 코드](../../graph/experiments/wrist_gate_cuda_10i_20261104_v1/benchmark.py)
- [기본 경로 연결 보고서](10i_wrist_gate_default_integration_20261104.md)
- [GPU gate](../../graph/persistent_geometry/tensor_gate.py)
- [반경 witness kernel](../../graph/persistent_geometry/tensor_gate_kernels.py)
- [NumPy gate 어댑터](../../graph/persistent_geometry/wrist_gate_dispatch.py)

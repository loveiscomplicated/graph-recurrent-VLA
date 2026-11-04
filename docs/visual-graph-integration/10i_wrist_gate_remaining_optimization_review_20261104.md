# Task10-I: 4.09ms GPU 손목 gate의 남은 최적화 여지

2026-11-04. **추가 최적화 여지가 있다. 다음은 같은 bool mask의 인덱스 압축 재사용,
셀 key 중복 정렬 제거를 먼저 검토하는 것이 좋다.** 이후 작은 GPU 연산의 kernel
융합과 이웃 후보 축소를 검토할 수 있다. 현재 고정 입력에서는 희소점 검사의
우선순위가 상대적으로 낮다.

이번 작업은 현재 source 검토와 **기존 gate 그대로의 GPU 진단 실행**이다.
production 코드·판정 기준을 수정하거나 최적화 후보를 구현하지 않았다.
손목 gate는 계속 **World만 사용**하며 EE/Graph 준비는 진단에서 실행하지 않았다.

## 이번에 직접 확인한 내용

입력은 기존 [gate 단독 4.087ms 측정](10i_wrist_gate_gpu_only_timing_20261104.md)과
같은 512×512 `10i_s1_0531_track_t011`이다. raw/clean mask, depth 및 World view를
사전에 준비하고 `wrist_gate_tensor`만 실행했다. remote RTX 3090 / EPYC 7543에서
실행 전 source SHA를 확인했다.

별도 1회 실행의 Python 함수 반환 시점에서 내부 tensor를 관찰하고, 계산 개수는
gate 완료 후 읽었다. 이 실행은 시간 측정에 사용하지 않았다.

| 항목 | 현재 입력의 관찰값 |
|---|---:|
| 유효 점 N | 29,775 |
| 사용한 spatial cell | 365 |
| 셀 최대 점 수 | 131 |
| 3점 미만 cell | 11 |
| 전체 점과 추가 거리 검사를 하는 rare 점 K | **19 (0.0638%)** |
| rare 검사 거리 쌍 K×N | 565,725, batch 1회 |
| 171 offsets로 생성한 cell 이웃 slot | 62,415 |
| 실제 occupied 이웃 cell 쌍 | 7,288 |
| AABB 검사 후 남은 쌍 | 3,220 |
| 실제 radius 연결이 확인된 cell 쌍 | 3,186 |

별도로 `torch.profiler` CPU/CUDA·shape 기록을 켜서 gate 5회를 실행했다.
호출별 관찰은 다음과 같다.

- `cudaLaunchKernel` **310회**, 실제 trace의 kernel event는 **312개**였다.
  API를 통해 launch한 kernel과 driver 경로의 kernel을 구분한 수치다.
- `aten::nonzero` **8회**. support·rare 각 1회 외에 동일 mask별 2회씩인 세 구간이 있다.
- `cudaStreamSynchronize` **19회**. 이 수치 전체를 명시적인 Python trace readback
  횟수로 해석하지 않는다. 동적 크기·할당 관련 처리를 함께 포함한다.
- 길이 29,775 배열의 `aten::_unique2` 2회와 별도 `argsort` 1회가 관찰됐다.
- 작은 CPU→GPU copy도 3회 남아 있다. 코드상 cell stride·offset 표·dims 준비와
  대응하는 비용 후보이며, 상수 offset 표는 프레임 간 재사용이 가능하다.

Profiler의 CPU range에서 연결성 `components`가 gate 전체 range 시간의 약 54%를
차지했다. 해당 range에는 label propagation이 포함된다. 이 비율은 **계측한 실행
내부의 관찰**이며 4.087ms의 정확한 단계별 분해가 아니다.

프로파일 기록·range wrapper는 CPU launch 간격과 전체 시간을 늘릴 수 있다.
GPU annotation range도 실제 kernel 실행 시간을 합한 값과 다르다. raw trace에서
`cat='kernel'`인 event만 합하면 평균 약 1.736ms/호출이지만, 이것을 4.087ms에서 빼서
전부 제거 가능한 CPU 비용으로 해석할 수 없다. 프로파일러 종료 후 별도 10회 wall
중앙값은 5.074ms였다. 현재 source는 기존과 동일하며, 서로 다른 실행 상태의 값을
최적화 효과나 회귀로 판단하지 않는다.

## 우선순위

| 우선 | 후보 | 확인한 근거 | 지켜야 할 조건 |
|---|---|---|---|
| 1 | **같은 bool mask의 압축 인덱스 공유** | `a[mask]`, `b[mask]`가 각각 내부 `nonzero` 실행. 세 구간에서 6회 | pixel/cell 쌍의 순서·대응·거리 판정 유지 |
| 2 | **셀 key를 한 번 정렬하고 구간·count·inverse 공유** | 동일 cell code에 `unique(sorted=True)` 후 별도 stable `argsort` | 원본 순서, component 동률 처리, 정수 key/overflow 검사 유지 |
| 3 | **작은 연산들을 kernel로 묶기** | 호출당 kernel 약 312개. label 전파·좌표 복원·projection 등에 작은 연산이 많음 | FP64·연산 순서·FMA 정책 및 완전 수렴 유지 |
| 4 | **상수 offset 재사용 및 기하학적 후보 축소** | 현재 171 offset, 실제 생성 62,415 slot | 경계 offset과 모든 실제 연결을 보수적으로 포함 |
| 5 | **rare 이웃 검사를 공간 질의로 변경** | O(KN)이지만 이 입력의 K는 19뿐 | 자기 자신을 포함한 기존 3점 기준·반경 경계 유지 |

### 1. bool 압축은 결과를 공유할 수 있다

[components](../../graph/persistent_geometry/tensor_gate.py)에는 다음 세 구간이 있다.

- `a=... [valid]`, `b=j[valid]`: 같은 365×171 bool mask를 각각 처리.
- `a=a[possible]`, `b=b[possible]`: 같은 7,288개 mask를 각각 처리.
- `a=a[keep]`, `b=b[keep]`: 같은 3,220개 mask를 각각 처리.

각 구간에서 정렬된 선택 인덱스를 한 번 구하고 두 배열에 `index_select`로 적용하면
같은 mask의 `nonzero` **6회를 3회로 줄일 후보**가 된다. 이 경로의 나머지 두
`nonzero`까지 없어진다는 뜻은 아니다. 구현 후 실제 kernel·동기화 수와 전체
gate 단독 시간을 다시 측정해야 한다.

### 2. 반복 정렬은 cell 준비와 최대 성분 선택으로 나누어 볼 수 있다

우선 같은 cell code의 `unique`와 stable `argsort`가 중복 정렬하는 부분을 줄일 수
있다. 한 번 정렬한 code에서 `unique_consecutive`·count·원래 점의 inverse mapping을
구성하는 방향이다. 새 scan/scatter 비용이 추가되므로 최종 속도는 비교해야 한다.

끝의 `unique(first[roots])`는 원래 점 순서대로 canonical component label을 만들기
위한 별도 정렬이다. gate의 최대 성분 판정만을 위해 이 정렬이 필요한지도 후속
검토 대상이다. 단순 제거하면 component label API와 최대 크기 동률 시 첫 성분을
고르는 동작이 달라질 수 있어 cell key 중복 제거보다 신중해야 한다.

### 3. kernel 융합은 launch 수를 줄이는 방향이다

label 전파는 이미 4회마다 host 수렴을 확인하지만 각 반복은 여전히 여러 Torch
연산을 launch한다. 좌표 복원과 카메라 간 projection·depth 비교도 여러 중간 tensor를
생성한다. 전용 kernel로 묶으면 launch와 임시 메모리를 줄일 가능성이 있다.

label을 임의 고정 횟수 후 잘라서는 안 된다. 다른 thread의 갱신 값을 의도치 않게
읽는 전파 변경도 피해야 한다. FP32 전환이나 FMA 활성화는 현재의 정확한 결과
유지 조건과 별도로 검증해야 하므로 낮은 위험의 변경으로 취급하지 않는다.

### 4. 171→89는 후보 표의 크기이며 전체 시간 개선율이 아니다

cell 변 `s=r/2`에서 offset `d`의 cell 간 거리 하한은
`s² × Σ max(|dᵢ|−1,0)²`다. 합이 4를 초과하는 offset만 제외하고 경계 4를 남기면
171개 중 **89개**가 남는다. 이 입력에서 생성 slot의 이론적 후보는
62,415→32,485개다. 상수 표 자체는 device별로 재사용할 수 있다.

현재도 AABB 검사로 radius kernel 전에 후보를 거른다. 따라서 이 변경은 주로
상류의 좌표·검색·압축 작업을 줄이며 radius kernel 수나 gate 전체 시간이 같은
48% 비율로 줄어든다는 뜻은 아니다. 실제 부동소수점 셀 경계에서 빠지는 연결이
없는지 검증해야 한다.

### 5. rare 검사는 입력 분포에 따라 우선순위가 달라진다

현재는 19점이 전체 29,775점과 거리를 비교한다. 모든 점이 전체 점과 비교하는
N² 경로는 아니다. 공간 셀을 재사용하고 3개 지지가 확인되면 끝내는 방향은 가능하나,
이번 입력에 대해 가장 큰 개선이라고 판단할 근거는 없다. 향후 sparse 입력에서
K가 커지면 우선순위가 올라갈 수 있다. 최종 local support fraction으로 K를
추측하지 않고 이번 실행에서 내부 rare 배열 길이를 직접 확인했다.

## 제안과 검증 범위

**다음 구현은 bool 압축 재사용과 cell key 중복 정렬 제거부터 진행하는 것이 합리적이다.**
같은 gate 단독 경계에서 전후 실행 순서를 교대하고 accepted mask·전체 trace를
정확히 비교해야 한다. 현재의 Python trace 즉시 반환 계약에서는 마지막 packed
readback도 무조건 없앨 수 없다. threshold·trace를 빼서 시간을 줄이는 변경은
위 제안에 포함하지 않는다.

이번 진단의 snapshot 관찰·profile 5회·추가 wall 10회 모두 accepted mask 및 전체
trace가 기존 CPU 기준과 정확히 같았다. 전체 simulator/SAM2/live rollout은 실행하지
않았으며 로컬 macOS에서 CUDA가 실행됐다고 주장하지 않는다. 최적화 구현·새로운
개선 배율·목표 시간은 아직 제시할 근거가 없다.

## 산출물

기존 `ExperimentLogger`를 `WANDB_MODE=disabled`로 재사용했다. run ID는
`f2a4f9d7411345daa76bdfc8f9dec5f4`이며 원격 artifact 게시는 하지 않았다.
이전 source identity의 runtime hash가 현재 로컬·원격 코드와 일치함을 확인했다.

- [진단 결과·내부 개수](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/remote/results.json)
- [조건·source identity](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/remote/config.json)
- [재집계: 실제 kernel·nonzero·정렬](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/analysis_summary.json)
- [연산별 profile](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/remote/profile_events.json)
- [Chrome trace](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/remote/profile.json)
- [관찰용 script](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/profile_gate.py)
- [실행 로그](../../outputs/task10i/wrist_gate_remaining_review_20261104_v1/profile.log)

다음과 같이 기존 source snapshot이 있는 원격 환경에서 실행했다.

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 python profile_gate.py \
  --input ORIGINAL_CPU_TIMING_INPUT_DIR \
  --identity outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/source_identity.json \
  --output NEW_DIAGNOSTIC_OUTPUT_DIR
```

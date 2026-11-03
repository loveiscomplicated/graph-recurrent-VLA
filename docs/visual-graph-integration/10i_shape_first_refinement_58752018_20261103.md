# Task 10-I — Shape-First 처리 범위·거리·호출 비용 최적화

2026-11-03. GPU 경로/출력 유지 이후의 내부 비용을 먼저 분리 측정하고,
외곽선 ROI, voxel 대표점, FPS 초기 거리·반복 실행, 선택 출처 조회를 차례로
최적화했다. 느려진 중앙값 선택 실험은 되돌렸다. 로봇 필터는 변경하지 않았다.

## 최종 비교

VastAI **58752018**, AMD EPYC7543, RTX3090. Python3.12.15,
Torch2.14.1+cu130, Triton3.8.0, NumPy2.5.4의 기존 환경을 사용했다.
입력은 앞선 작업과 같은 실제 저장 hand-camera **512×512** RGB-D와 추론 mask다.
Peg mask61,056 pixels, Box0 pixels, 최종 선택점1,105개다.

기준은 직전 작업의 **GPU 연산 + GPU 출력** 경로다. CPU 후처리와 비교한 표가
아니다. compact CUDA 좌표·특징이 준비된 시점부터 전체 sampling 결과가 GPU에
계산 완료된 시점까지 측정했다. 양 끝 CUDA 동기화를 포함한다. 역투영·PCA/특징,
파일 IO, 검증용 다운로드, 로봇 필터/gate, 최종 packet·Graph 조립은 제외한다.

warmup20회 뒤 variant별100회를 번갈아 실행한 최종 결과:

| 경로 | 중앙값 | p95 |
| --- | ---: | ---: |
| 기존 GPU sampling/output | **12.308ms** | **13.255ms** |
| 최적화 후 GPU sampling/output | **8.920ms** | **9.768ms** |

중앙값 **27.53% 감소**, 약 **1.38배**다. 같은 입력의 후보·대표점·최종 선택점,
특징·validity·출처 필드가 모두 exact이며 CUDA에 유지된다. `device='cpu'`를
명시하는 출력 규약도 보존한다. 이 한 입력에서 얻은 수치를 다른 mask 크기·형태,
두 실제 카메라 또는 새 학습 데이터셋의 고정 배율로 일반화하지 않는다.

## 먼저 확인한 병목

수정 전 baseline-only 실행은 중앙값11.594ms, p9512.437ms였다. 이후 같은
process의 paired 비교를 각 변경마다 수행했다. 아래 구간 표는 최종 실행에서
기존/최적화 함수를 따로 호출한 **독립 진단 표본**의 중앙값이다.

| 구간 | 기존 | 최적화 후 |
| --- | ---: | ---: |
| 외곽선 component/walk | 2.432ms | 2.196ms |
| Voxel PCA 후보 대표점 | 1.211ms | 0.766ms |
| Voxel silhouette 후보 대표점 | 1.436ms | 0.859ms |
| FPS 전체 | 4.344ms | 2.958ms |
| 최종 선택 출처/evidence 구성 | 1.095ms | 0.143ms |

CPU가 kernel을 enqueue하는 동안 앞선 GPU 연산이 진행될 수 있다. 구간을
따로 호출하고 기다리면 그 겹침이 사라진다. 따라서 구간 중앙값을 더해 전체
sampling 시간을 만들 수 없다. 예를 들어 기존 FPS 초기화/반복을 따로 측정한
값은3.439ms/2.746ms지만, 실제 FPS 전체는4.344ms였다.

## 적용한 변경

### 외곽선 ROI

원본 mask를 GPU에서 한 번 scan해 최소 사각형의4개 scalar를 구한다. 모든
raw-mask pixel을 포함하는 사각형과 background pad1 안에서 component와
contour walk를 실행하고 결과를 원본 source pixel 번호로 돌려놓는다.

이 입력의 사각형은 `v=[1,512), u=[260,472)`, **108,332 pixels**다.
pad를 포함한 component raster는 기존 `514×514=264,196`에서
`513×214=109,782`로 줄었다. Raw mask 전체를 포함하므로 외부 연결성,
hole/island 제외, image-edge topology와 canonical 순서를 유지한다.
Observed support만으로 사각형을 정하거나 mask를 훼손하지 않는다.

ROI만 적용한 초기 paired 비교는11.773→11.629ms였다. 최종 구간 비교도
2.432→2.196ms로, 처리 pixel 감소율만큼 시간이 줄지는 않았다. Bounding scan,
scalar 반환, component 반복·launch·allocation 비용은 여전히 있다.

### Voxel 대표점

기존 FP64 centroid 계산과 cell별 원본 점 선택 규칙을 유지했다. 거리 계산,
최소 거리/first-source 동점 처리와 대표점 flag 기록을 Triton kernel로 묶었다.
이미 source가 오름차순인 PCA 입력은 중복 source 정렬을 생략한다. Silhouette은
정렬 후 기존 contour 순서로 복원한다. 대표 source 목록에 다시 `isin`하는
대신 row flag를 직접 사용한다. Voxel key의 안정적인 다중 정렬은 유지했다.

### 정확한 FPS 초기 거리 pruning

기존 구현은 **61,056점 ×1,008구조점 =61,544,448쌍**을 비교한다. 새 커널은
구조점을 tile로 읽고 tile AABB까지의 보수적인 FP64 거리 하한을 구한다.
그 tile이 point block의 현재 최소 거리를 개선할 수 없을 때만 거리 행렬
계산을 건너뛴다. 근사 이웃 탐색, 점 제거나 거리 cutoff를 추가하지 않았다.

FP64 덧셈 순서와 FMA 비활성화를 유지한다. Seed 단계에서 필요한 것은 각 점의
최소 거리값이며, 동률인 구조점 자체의 ID는 출력하지 않는다. 하한이 현재 거리와
같아도 그 최소값은 바뀌지 않으므로 해당 tile을 제외할 수 있다.

최종 kernel tuning에서 선택한 `block128/tile32/warps4`의 결과:

| 초기 거리 kernel microbenchmark | 거리쌍 | CUDA-event 중앙값 |
| --- | ---: | ---: |
| 기존 full comparison, block128/tile32/warps8 | 61,544,448 | 3.804ms |
| 최종 pruning, block128/tile32/warps4 | **25,987,072** | **1.932ms** |

점–구조점 거리쌍 계산은 약 **57.8% 감소**했다. 하한 계산 자체의 비용은 이
쌍 개수에 포함하지 않으며, kernel 시간에는 포함된다. Counter는 별도 검증에서만
켜고 실제 실행/시간 측정에서는 끈다. 모든 점의 초기 거리 배열을 기존 커널과
`rtol=0, atol=0`으로 직접 비교했다.

Tile16 후보는23,076,864쌍으로 더 적게 비교했지만2.222ms로 더 느렸다.
거리쌍 개수만으로 우열을 정하지 않고 실제 커널 시간으로 tile32를 선택했다.
Pruning 효율은 구조점 tile의 공간적 응집도에 따라 달라진다.

### CUDA Graph와 workspace 재사용

표면97점 추가의 기존 반복은 argmax97회 + distance update96회,
총 **193회 Python 측 kernel dispatch**였다. 일정 크기 이상에서는 seed와
반복을 CUDA Graph에 담고 cache hit마다 **한 번의 graph replay**로 제출한다.
GPU 내부에는 기존 반복 kernel들이 그대로 남으며, 입력 buffer 갱신과 결과
생성 kernel도 별도로 존재한다. GPU kernel 실행 자체가1회가 된다는 뜻은 아니다.

- 4,096점 이상이고 새 surface row가16개 이상일 때 graph 경로를 사용한다.
- Point/seed 수는 power-of-two capacity bucket으로 관리한다.
- Device, dtype, stream, capacity, 추가점 수로 구분한 plan을 최대8개 보존한다.
- 매 호출 현재 XYZ·구조점 row를 복사하고 used/padding 상태를 재설정한다.
- 초기 거리, argmax와 output 작업 buffer를 재사용한다.
- 결과는 다음 replay에 덮어쓰이지 않게 별도 tensor로 materialize한다.
- 작은 호출은 eager 경로다. Frame 관측이나 선택 결과를 재사용하는 cache가 아니다.

Point/seed 수, ROI 크기 등 반복적으로 변하는 scalar 인자를 runtime 인자로
바꿔, 정확한 사각형 크기/후보 수마다 별도 kernel specialization을 요구하지 않게
했다. Tile/block layout 등 필요한 정적 인자는 유지했다.

### 정렬·집합 조회

최종 candidate/선택 출처를 구성하는 반복 `isin`은 역투영·법선 단계에 이미
있는 raster→compact row map과 bit flag lookup으로 대체했다. 좌표나 전체
mask를 CPU로 가져오지 않는다. `R[::2]`로 만든 source의 stride2도 직접 반영한다.

반면 전체 깊이 정렬을 두 번의 `kthvalue`로 바꾼 실험은 **0.257→1.490ms**로
악화되어 철회했다. 최종 구현은 원래 full sort 기반 중앙값을 유지한다.

## 초기 비용과 남은 부분

최종 비교 process의 첫 함수 호출은 기존182.571ms, 최적화108.117ms였다.
이는 first-call JIT loading/initialization과 graph 생성 등을 함께 포함한
측정이며, 디스크 JIT cache는 앞선 실행으로 이미 형성되어 있었다. 빈 JIT cache의
최초 실행이나 graph 생성 비용만 따로 측정한 값은 아니다. 중간 후보에서 약
950ms의 첫 호출도 관측했으므로 **8.920ms는 warm replay 성능**으로 해석해야 한다.
새 capacity/stream 조합의 plan 생성 때도 초기 비용이 발생한다.

Bucket padding은 추가 작업량을 만들며, 수렴 단계의 CPU scalar 확인은 남아
있다. ROI 밖 원본 mask는 bounds scan에서 한 번 읽고, component 반복마다 CPU가
종료 flag를 확인한다. 이번 변경은 외곽선 수렴 전체를 device-only loop로 바꾸거나
모든 임시 allocation을 제거한 작업이 아니다. Multi-stream 동시 호출 성능은
실측하지 않았다. 최종 Graph 조립의 CPU 소비 경로도 그대로다.

## 정확성·검증

최종 실제512 비교에서 기본 출력34개 CUDA tensor와 `diagnostic=True`의 전체
후보/voxel 대표점 목록을 이전 implementation과 비교해 모두 exact임을 확인했다.
NaN 위치, dtype, source 순서도 같았다. Peg의 선택점은 구조점1,008개 + surface97개,
총1,105개이며 Box는0개다. Source/선택 규칙과 point 수를 바꾸지 않았다.

- 최종 source(`final4_source`)의 GPU sampling/Graph 계약: **15개 모두 pass**.
  새 계약은 ROI 원본 좌표/순서, stride2 출처 조회, 같은 graph bucket에 서로 다른
  n/seed 수와 geometry를 넣는 경우, 이전 결과 tensor의 독립성을 포함한다.
- 그 직전 source의 관련 PCA까지 포함한 묶음: **29개 모두 pass**. 이후에는
  느린 kthvalue를 원래 sort로 되돌리고 seed tile만 조정했으므로, 변경된 sampling/
  Graph15개를 다시 실행했다. PCA/feature 코드는 수정하지 않았다.
- 로컬 macOS `.venv` 초기 묶음:46개 중34 pass, CUDA12 skip.
  최종 CUDA는 Vast에서 실행했으며 macOS CUDA는 실행하지 않았다.
- 기본 출력에서 CPU/NumPy 변환을 막는 계약과 명시적 CPU 반환 계약도 통과했다.
- 실제512의 full two-camera rollout, 새 모델·데이터셋 학습/평가, 정책 성공률은
  실행하지 않았다. Robot filtering/gate는 이 task 범위 밖이다.

중간 lookup 후보는 stride2를 무시해3개 packet/Graph 계약이 실패했다. 이를
수정하고 회귀 테스트를 추가했다. 실패 로그와 candidate source도 보존했다.
허용오차를 늘리거나 CPU fallback으로 검증을 통과시키지 않았다.

## 실행·산출물

공통 `ExperimentLogger` 재사용, W&B disabled, publishing 없음. 다른 job 중지,
패키지 설치나 driver/service 변경은 하지 않았다. 원격 실행은 독립 copy이며
이전 frozen source를 덮어쓰지 않았다. 원격 root:
`/workspace/task10i_shape_first_refinement_20261103_v1`.

| 실행 | UUID |
| --- | --- |
| 수정 전 baseline + 분리 측정 | `4815b29fb47d4b88aef375b83f414bb9` |
| ROI만 비교 | `c6c8b4411c3446a1b51270a10ee9f408` |
| ROI + voxel 비교 | `051206ff7d5142538e1a2b6875e3266e` |
| FPS graph/pruning 추가 비교 | `bd860e30436a40148fb12a172703d343` |
| 최종 paired/구간 비교 | **`368924aa9c0b45e896a4e74f578ac515`** |
| 최종 seed kernel 전체 거리/계산쌍 검증 | `1db222a2ebb34a808f2b7d1792031784` |

중간 비교는 각 process에서 측정했고 host 부하/clock/실행 조건에 따라 값이
변동했다. 단계별 향상률을 곱하거나 절대 ms를 더해 최종 결과를 만들지 않는다.
최종 수치는 같은 process에서 before/after를 번갈아 측정한 위 첫 표다.

- [최종 summary](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_release/summary.json)
- [최종 200개 raw wall 표본](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_release/timings.npz)
- [수정 전 구간 raw 표본](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_release/stage_timings.npz)
- [최적화 후 구간 raw 표본](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_release/optimized_stage_timings.npz)
- [Seed tile 비교·모든 최소 거리 exact 검증](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/release_seed_verification/tuning.json)
- [중앙값 선택이 느려진 후보의 기록](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_comparison/summary.json)
- [최종 CUDA sampling/Graph 계약](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final4_contracts.log)
- [PCA 포함 계약](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final2_contracts.log)
- [중간 실패 로그](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/final_contracts.log)
- [로컬 계약](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/local_contracts.log)
- [Source manifest](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/remote/source_manifest.json)
- [다운로드·통계 재계산 receipt](../../outputs/task10i/shape_first_refinement_58752018_20261103_v1/download_receipt.json)
- [실행 코드와 방법](../../graph/experiments/shape_first_refinement_58752018_20261103_v1/README.md)

Archive SHA256:
`889a94b20dd23429f4adf4af933a185635751252bd38645b239bfddf3b899c2c`.
최종/기준 full source와 중간 candidate snapshot, 입력, runner, Logger metadata,
성공/실패 로그, 원본 표본을 내려받았다. Archive hash, 최종 runtime/kernel/test와
현재 checkout의 일치, runner2개 hash, raw 표본에서 재계산한 중앙값/p95를
확인했다. Manifest는1,435개 파일이다. Git dirty checkout에서 진행했으므로
commit만으로 재현된다고 주장하지 않는다. 동시에 존재한 로봇 필터 작업의
파일/문서는 이 task에서 수정하지 않았다.

# Task10-I — V4 Main 카메라 꼭짓점 검출기의 병목

조사일: 2026-11-05 KST. 상태: **프로파일링·중복 재사용 진단 완료**.

현재 약58ms/frame이 드는 주요 이유는 V2 기본 검출, V3 경계 회복, V4 Box 회복을
순서대로 수행하면서 전처리를 반복하고, 많은 국소 직선 후보를 개별적으로 적합·검증하기
때문이다. 최종 꼭짓점 수는 작지만 그 점을 선택하기 위한 내부 계산은 작지 않다.

같은28개 Main RGB-D 입력에서 평면 추정과 역투영의 완전히 같은 계산을 프레임 안에서
재사용하는 진단 실험을 했다. 평균은 **57.17 → 49.89ms, 12.72% 감소**했고, 모든 출력
꼭짓점이 동결 V4와 같았다. 실제 검출기에는 이 재사용을 적용하지 않았다.

## 범위와 측정 방법

대상은 기존 독립 프로토타입의 전체 `detect(...)` 호출이다. Main `base_camera`의
native512 RGB-D 한 장에서 Peg와 Box를 함께 검출한다. 주 live에 연결된 경로의 시간이
아니며 Box-only 시간도 아니다. SAM2는 사용하지 않는다.

입력은 이전 [프레임당 시간 측정](10i_corner_v4_main_latency_20261105.md)과 같은
`capture_v4_final`의28상태다. RGB/depth/intrinsic/extrinsic만 미리 메모리에 읽었다.
손목 카메라, GT, actor segmentation, simulator 렌더링, 파일 읽기, 영상 저장, policy는
측정에 포함하지 않았다. 출력 비교·로그·통계도 detector timer 밖에서 수행했다.

| 실행 | 방법 | 표본 |
| --- | --- | ---: |
| 비계측 기준 | 전체28입력 warmup 후 외부 wall timer만 사용 | 28×3=84 |
| 함수별 비용 | 큰 함수만 nested inclusive/exclusive wall timer로 계측 | 28×5=140 |
| 작은 함수 호출 수 | 별도 cProfile 실행 | 28 |
| 동일 계산 감사 | 별도 실행에서 입력·출력 byte hash와 RNG/인자 비교 | 28 |
| 재사용 진단 | 네 모드 각각28입력 warmup, 모드·입력 순서를 섞어5회 비교 | 모드별140, 합계560 |

비계측 평균은55.07ms, 큰 함수 계측 평균은55.64ms였다. 두 pass의 차이는 약1.04%이며
계측과 실행 시점 변동이 함께 포함된다. 이전560회 benchmark의57.82ms와 이번55.07ms는
서로 다른 실행이다. **검출기를 고쳐 빨라진 결과가 아니다.**

cProfile에서는 전체 호출이74.41ms까지 늘었다. 작은 함수가 많아 계측 영향이 크므로
그 시간은 절대 성능값으로 사용하지 않고 호출 수와 병목 위치 확인에만 사용했다.

Host는 Apple M4 Pro, macOS27.0.1 arm64, Python3.11.13, NumPy2.4.6/Accelerate,
OpenCV5.0.0이다. OpenCV 기본14 threads와 optimized 설정을 유지했다.
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`로 실행했고
`VECLIB_MAXIMUM_THREADS`는 미지정이다. 이를 단일 thread 실행이라고 해석하지 않는다.
OpenCL flag=True이나 입력은 NumPy 배열이며 명시적 UMat은 쓰지 않았다.
OpenCL 내부 dispatch는 추적하지 않았고 CUDA/MPS는 실행하지 않았다.

## 시간이 쓰이는 위치

아래는 큰 함수 계측 pass의 평균이다. **하위 계측 함수 시간을 뺀 exclusive time을
묶었으므로 행 사이의 중복이 없다.** 비율의 분모는 root 함수55.634ms다.

| 계산 | 평균 ms/frame | 비율 | 관측 내용 |
| --- | ---: | ---: | --- |
| RANSAC 평면 추정 전체 | **11.99** | **21.6%** | V2 4.92 + V3 회복 4.91 + V4 Box helper 2.16 |
| 국소 경계 직선 적합 | **10.30** | **18.5%** | 짧은 두 span에서 양쪽 직선과 교차점 비교 |
| Box 추가 회복의 나머지 계산 | 10.78 | 19.4% | RGB feature·cube 후보·quadrant·가시성·포장 등; 별도 계측 평면/edge 제외 |
| 물리적 depth edge 검증 | 5.41 | 9.7% | 회복 경로 후보마다 양쪽 edge 지지 확인 |
| RGB 색상 mask 생성 | 3.62 | 6.5% | 두 경로의 색상 분리·연결 성분 분석 |
| 전체 depth의 XYZ 역투영 | 1.79 | 3.2% | 평균2.96회/frame |
| 기타 | 11.75 | 21.1% | V2 내부 잔여 계산, polygon, plane label/mask, 가시성·merge·포장 등 |
| 합계 | **55.63** | **100%** | 반올림 차이 있음 |

Box helper `recover(...)`의 **inclusive** 시간은14.34ms였다. 이 안에는 위 표의 평면
추정·edge 검증 일부가 들어 있다. 14.34ms를 표에 다시 더하면 중복 합산이다.
또한 V2/V3에도 Box 계산이 있으므로 이것은 Box 전체 시간이 아니다.

### 1. 정확도 회복 경로들이 전처리를 다시 실행한다

[`detector.py`](../../local_corner_probe_10i_20261104/detector.py)는 처음에 V2 전체
검출을 호출해 기존의 검증된 꼭짓점을 보존한다. 이어 자체 mask/XYZ/plane/contour를
만들어 V3 추가 후보를 계산한다. 마지막에는
[`box_corner_refinement.py`](../../local_corner_probe_10i_20261104/box_corner_refinement.py)의
Box 회복을 호출하고, 그 안에서 XYZ와 Box 평면을 다시 만든다.

감사에서는 동일 프레임 내 호출의 입력 points/depth/intrinsic뿐 아니라 plane tolerance,
최소 점 수, RANSAC trial 수, 최대 plane 수, RNG 입력 상태까지 비교했다. 같은 key의
출력도 byte hash가 같았다.

| 연산 | 28프레임 총 호출 | 프레임당 호출 | 그중 완전히 같은 중복 | 중복이 관측된 프레임 |
| --- | ---: | ---: | ---: | ---: |
| `_planes` | 139 | **4.96** | **75회, 2.68/frame** | 28/28 |
| `unproject` | 83 | **2.96** | **55회, 1.96/frame** | 28/28 |
| `color_masks` | 56 | 2.00 | 같은 전체 인자 key 없음 | 0/28 |

V2와 V3의 Peg 평면은 모든 입력에서 중복이었다. Box는 V3와 V4 helper가 같은
mask/seed로 계산하는 구간이 중복이고, V2까지 같은 경우도 있었다. V2 Box의 연결 성분
최소 크기35와 회복 경로의8은 다르므로 **모든 mask/Box 평면을 무조건 하나로 합칠 수는
없다.** 색 변환·연결 성분 결과를 공유하되 서로 다른 threshold 결과는 보존하는 설계가
필요하다.

RANSAC 자체도 plane 하나를 찾을 때 최대1300개 sample과160개 가설의 잔차를 비교하고,
관측 점 전체의 재선택과 SVD 재적합을 반복한다. 이런 작업을 같은 입력으로 다시 하는
것이 가장 명확하게 제거 가능한 비용이다.

### 2. 국소 모서리 하나에 작은 선형대수 호출이 많이 생긴다

`_local_boundary_corners`는 contour의 rough corner마다 span7과11을 시도하고, 각 span의
이전·이후 점을 각각 직선으로 적합한다. `_line`은 SVD를 수행하며 outlier를 제거하면
SVD를 다시 수행한다. 그 뒤 교차점, 잔차, 두 span 사이의 일치 여부를 검사한다.

cProfile 호출 수는 다음과 같다. 내부 dispatcher를 SVD와 중복으로 세지 않았다.

- `_line`: 전체 **528.46회/frame**, 그중 현재 V3 회복 경로494회/frame.
- `np.linalg.svd`: plane/line 적합을 합쳐 **605.82회/frame**.
- 회복 경로의 `_physical_edges`: **66.71회/frame**.
- 전체 Python/native 함수 호출: 약**11.3만회/frame**.

따라서 최종 노드가 몇 개라는 사실만으로 가벼운 계산이 되지 않는다. 작은 배열의 평균,
norm, SVD, mask indexing 등을 Python loop에서 반복하는 비용이 누적된다.

### 3. Box 추가 후보를 만들고 버리는 비용도 남는다

V4 helper는 mask와 grayscale에서 각각 Shi–Tomasi/subpixel corner를 추출한 뒤,
feature×관측 plane×네 quadrant의 지지를 검사한다. Cube exterior 후보는 관측 면에서
크기를 추정하고, 개별 후보의 depth/가림/edge 지지를 확인한다. 일부 후보에서는 같은
평면의512×512 거리 mask를 다시 만든다. 이 마지막 중복의 단독 시간은 따로 분리하지
않았으며 다음 최적화 후보라는 코드 근거다.

RGB feature 생성 전체는1.74ms/frame으로 계측됐다. 이 중 `goodFeaturesToTrack`만
제거한다고 전체 병목이 해결되지는 않는다. 비용의 상당 부분은 그 뒤 후보 해석과 검증에
있다. 이번 범위에서 SAM2·렌더러·디스크 대기를 원인으로 볼 근거는 없다.

입력별 차이도 이 계산량과 연결된다. 큰 함수 계측에서 Parent0121 t80은 전체22.05ms,
국소 적합3.81ms, 평면4.76ms였고, reset400은 전체79.78ms, 국소 적합18.10ms,
평면16.03ms였다. 장면마다 관측 면·경계 후보 수가 달라서 평균 하나로 모든 프레임의
deadline을 보장할 수 없다.

## 같은 계산 재사용의 진단 결과

| 모드 | 평균 | 중앙값 | p95 | 기준 대비 평균 감소 |
| --- | ---: | ---: | ---: | ---: |
| 원래 계산 | **57.17ms** | 60.78ms | 80.80ms | — |
| 같은 평면 결과만 공유 | **50.00ms** | 52.56ms | 70.45ms | **12.53%** |
| 같은 XYZ만 공유 | 56.06ms | 60.03ms | 79.71ms | 1.93% |
| 평면+XYZ 공유 | **49.89ms** | 51.52ms | 71.41ms | **12.72%** |

모드별140회 측정했다. 공유 시 같은 프레임의 plane 중복375회, XYZ 중복275회를
생략했다. 모든 warmup·본 측정에서 꼭짓점 개수/순서/object_id/uv/xyz가 동결 V4와
절대오차1e-10 이내로 일치했다. GT를 다시 읽어 정확도를 평가한 실험은 아니다.

이 실험은 앞서 감사한 **고정28입력의 호출 key를 이용한 process-local 진단**이다.
Runtime에 큰 배열을 hash하는 cache를 구현한 것이 아니다. 프레임 간 cache는 없으며,
매 프레임 cache를 지운다. 재사용된 plane 계산의 RNG 종료 상태도 복원했다. 의도는
전처리 결과를 명시적으로 전달하는 구조로 바꿀 때 얻을 여지를 확인하는 데 있다.
일반 입력용 최적화 구현이나 배포 가능한 cache 성능으로 해석하지 않는다.

평면 단독과 평면+XYZ 사이 평균 차이는0.11ms였다. OS background 부하가 있는 이
측정만으로 작은 추가 차이를 확정하지 않는다. 확인된 큰 효과는 평면 중복 제거다.
평면+XYZ를 공유해도 p95는71.41ms이므로 20Hz/50ms deadline을 만족했다고 할 수 없다.

## 개선 우선순위

1. **현재 프레임의 전처리 결과를 공유한다.** 깊이 변환·valid·XYZ·HSV/연결 성분,
   같은 mask와 seed에서 얻은 평면/label/mask를 한 번 계산해 각 회복 경로에 전달한다.
   mask threshold 차이와 RNG 의미는 보존한다. 이번 진단에서 약7.3ms 여지를 확인했다.
2. **국소 직선 적합을 줄이거나 묶는다.** contour별 centroid/인접 구간을 반복 생성하는
   계산을 공유하고, 2D 직선 방향은 작은 covariance의 해로 구하는 방법과 batch 적합을
   비교할 수 있다. robust refit·span 일치 판정은 출력 정확도 검증 없이 생략하지 않는다.
3. **후보별 반복 검증을 공유한다.** plane별 거리/가시성 mask와 기하 basis를 재사용하고,
   RGB 후보·plane·quadrant 연산을 묶는다. V4 추가 회복을 언제 실행할지도 비용 대비
   이득으로 검토할 수 있다. 현재 Main28프레임에서 V4가 V3보다 추가한 것은 정답2점과
   3mm 초과1점이다. 이 수치는 [V4 정확도 보고서](10i_visible_corner_512_box_v4_20261105.md)의
   같은 입력 비교이며, helper14.34ms를 제거하면 그대로 빨라진다는 실측 주장은 아니다.

GPU 전환만으로 위 비용이 해결된다는 측정은 없다. 먼저 중복 준비와 작은 호출 수를
줄이는 것이 이번 결과에 근거한 순서다. 실제 최적화 시 다른 입력에서도 출력·정확도와
tail latency를 다시 검증해야 한다.

## 보존·검증·재현

브랜치 `codex/cleanup-modular-live-20261104`를 유지했다. `recurrent`로 전환하지 않았다.
프로파일러와 진단 driver만 기존 ignored 폴더에 추가했다. 검출기 네 파일은 실행 전후
`v4_final_freeze.json`의 source hash와 같았다. Production/live 코드와 detector 코드는
수정하지 않았다. 이번 backend 검증은 저장된 입력에 대한 로컬 실행이다. 새 simulator,
CUDA/MPS, 서버 GPU는 실행하지 않았다.

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했고 `WANDB_MODE=disabled`로
로컬 기록했다. 원격 게시·학습·공통 API 변경은 없다.

- 프로파일·동일 계산 감사 run: `6cde3abb7d1f44f5a48d73bad6ed8fac`.
- 재사용 진단 run: `2313ffd1f3b24b459f0c87b069c3a951`.

새 output 경로를 지정해 재현한다. 기존 디렉터리는 덮어쓰지 않도록 되어 있다.

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.profile_latency --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --output local_corner_probe_10i_20261104/new_profile --rounds 5
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.probe_preparation_reuse --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --audit local_corner_probe_10i_20261104/new_profile --output local_corner_probe_10i_20261104/new_reuse_probe --rounds 5
```

원자료:

- [프로파일 driver](../../local_corner_probe_10i_20261104/profile_latency.py) / [재사용 진단 driver](../../local_corner_probe_10i_20261104/probe_preparation_reuse.py)
- [프로파일 summary](../../local_corner_probe_10i_20261104/main_profile_20261105_v1/summary.json) / [함수별 exclusive·inclusive CSV](../../local_corner_probe_10i_20261104/main_profile_20261105_v1/coarse_functions.csv)
- [호출별 계측 CSV](../../local_corner_probe_10i_20261104/main_profile_20261105_v1/coarse_events.csv) / [cProfile 함수 CSV](../../local_corner_probe_10i_20261104/main_profile_20261105_v1/cprofile_functions.csv)
- [동일 계산 감사 원자료](../../local_corner_probe_10i_20261104/main_profile_20261105_v1/data_audit.json) / [프로파일 실행 로그](../../local_corner_probe_10i_20261104/main_profile_20261105_v1.log)
- [재사용 진단 summary](../../local_corner_probe_10i_20261104/main_reuse_probe_20261105_v1/summary.json) / [모드별 측정 CSV](../../local_corner_probe_10i_20261104/main_reuse_probe_20261105_v1/samples.csv) / [실행 로그](../../local_corner_probe_10i_20261104/main_reuse_probe_20261105_v1.log)

Ignored 자료 링크는 현재 로컬 폴더 보존을 전제로 한다.

# Task 10-I — 샘플링 최종 반환 device 변경과 재측정

2026-11-03. 사용자가 범위를 **샘플링 결과의 GPU 반환과 재측정**으로 명시했다.
최종 Graph 조립 전체를 GPU로 옮기는 작업은 포함하지 않는다.

## 변경 결과

`visual.runtime.cuda_sampling.sample_cuda(...)`를 추가했다. 기존 CUDA
sampling 정보 구성·Shape-First·FPS에 이어 **최종 출력도 CUDA tensor로 유지**한다.
선택 XYZ, source pixel, 구조점/surface 구분, 특징, object validity,
candidate evidence와 선택 출처까지 모두 같은 device에 있다.

```python
from visual.runtime.cuda_sampling import sample_cuda

# device 생략: 입력 sampler의 CUDA device 유지. NumPy/CPU 배열 반환 없음.
gpu = sample_cuda(projected, features, depth, masks, intrinsic)

# 명시적 CPU 반환: tensor만 CPU로 복사. sampling 계산 자체는 여전히 CUDA다.
cpu = sample_cuda(projected, features, depth, masks, intrinsic, device='cpu')
```

`selection_output(raw_result, device=...)`가 같은 반환 규약을 제공한다.
`diagnostic=True`일 때만 전체 후보·대표점 목록도 해당 device로 반환한다.
`numpy_selection`은 **이미 CPU인 tensor만** NumPy로 바꾸며 CUDA tensor를
받으면 오류를 낸다. 기존 `to_cpu_selection`은 `device='cpu'`를 명시하는
호환 wrapper로 유지했다. MPS 등 지원하지 않는 출력 device는 거절한다.

현재 `graph_perception.make_packet`의 NumPy Graph 소비자는
`sample_cuda(..., device='cpu')`를 **명시적으로** 호출한다. 이 호환 소비 경로의
CPU 반환은 남아 있고, 샘플링 API의 기본 CUDA 반환과 구분해야 한다. CPU 입력만
있는 host에서 CUDA sampling을 CPU로 자동 fallback시키는 기능도 추가하지 않았다.

픽셀·점 선택 정책, contour 순서, voxel 대표점, FPS 계산은 바꾸지 않았다.
로봇 필터·gate와 이후 Graph 조립, 모델·데이터셋·학습은 수정하지 않았다.

## 동일 실행 재측정

VastAI **58752018**, AMD EPYC 7543, RTX3090. Python3.12.15,
Torch2.14.1+cu130, Triton3.8.0, NumPy2.5.4의 기존 환경을 사용했다.
설치·service/driver 변경, 다른 작업 중지, publishing은 없었다.

입력은 앞선 작업과 같은 실제 저장 hand-camera **512×512** 관측과 RGB 추론
mask다. Peg61,056 pixels, Box0 pixels이며 최종 선택점은 **1,105개**다.

시작은 compact CUDA 좌표/특징이 준비되고 동기화된 시점, 종료는 **각 반환
device의 샘플링 결과 계산 완료**다. GPU 반환 variant도 종료 시 CUDA를
동기화하므로 enqueue 시간만 측정한 결과가 아니다. 역투영·PCA/특징 계산,
file IO, 출력 검증·NPZ 저장, 최종 packet/Graph 조립은 제외한다.

Warmup20회, variant별100회, 실행 순서를 회전·반전했다. OMP/BLAS/MKL은
1 thread다. CPU affinity와 공유 host 부하는 고정하지 않았다.

| 반환 경로 | 중앙값 | p95 | 최소–최대 |
| --- | ---: | ---: | ---: |
| 보존된 기존 CPU/NumPy 반환 | **11.785ms** | **12.501ms** | 11.479–13.701ms |
| 기본 CUDA tensor 반환 | **11.409ms** | **12.013ms** | 11.184–14.777ms |
| 명시적 `device='cpu'` tensor 반환 | **11.739ms** | **12.324ms** | 11.493–21.072ms |

동일 실행에서 기존 반환 대비 CUDA 유지의 중앙값 차이는 **0.376ms(약3.2%)**다.
명시적 CPU tensor 반환과 비교하면 0.330ms다. 이 차이는 순수 PCIe 복사 kernel
시간만이 아니라 CPU allocation, 동기화, wrapper/NumPy 처리와 실행 순서 차이도
포함한다. CPU 반환을 제거해 수 ms가 반드시 줄어든다고 주장하지 않는다.

앞선 작업의 13.975ms와 이번 11.409ms를 직접 빼서 CPU 복사 비용으로 보면
안 된다. 이번 실행에서는 기존 CPU 반환 자체도 11.785ms였다. 비교 기준은
같은 process·같은 입력·같은 sampling 계산의 위 세 variant다.

기존 경로는 이전 작업의 byte-preserved `cuda_sampling.py`를 별도 이름으로
import하여 실행했다. 수정된 함수로 과거 반환을 재구현하지 않았다. 기존/현재
Triton kernel 파일의 hash가 같음도 확인했다.

## 실행 검증

- 실제512에서 기본 출력 **34개 tensor 필드가 모두 `cuda:0`**임을 확인했다.
- 동일 입력의 `device='cpu'` 출력은 모든 tensor가 CPU다.
- GPU 기본 출력과 명시적 CPU 출력의 모든 값·dtype·shape·source 순서가
  기존 CPU/NumPy 출력과 exact이다. NaN 위치도 일치한다.
- 기본 GPU 출력 검증·저장을 위한 다운로드는 본 시간 측정 **밖에서만** 했다.
- CUDA 계약에서 선택부터 반환까지 `Tensor.cpu()`, `Tensor.numpy()`,
  `Tensor.to(device='cpu')`를 오류로 바꿔도 기본 API가 동작하는지 확인했다.
  NumPy adapter가 CUDA를 거절하는 계약도 통과했다.
- Vast 실제 CUDA: `test_cuda_sampling`, `test_graph_perception`,
  `test_pca_optimized` **25개 모두 pass**, skip/failure0. 기존 CPU packet 및
  합성 Graph 소비 계약도 포함한다.
- 로컬 macOS `.venv`: 관련12개 중 **3 pass, CUDA9 skip**. 로컬 CUDA는
  실행하지 않았다. `git diff --check`를 통과했다.

내부 component 수렴·동적 cardinality의 scalar host 동기화는 남는다.
최종 배열을 GPU에 유지한다는 뜻이며, Python 제어나 모든 host synchronization을
제거했다는 뜻은 아니다. 실제 camera2개 end-to-end rollout, 정책 성공률 또는
최종 Graph 전체의 GPU 실행을 검증한 결과도 아니다.

## Identity·산출물

공통 `ExperimentLogger` 재사용, W&B disabled, publishing 없음.
실행 UUID: **`ebd9f562cedf48098540052ccee6b7ae`**.
원격 root: `/workspace/task10i_shape_first_device_output_20261103_v1`.
새 독립 source copy로 실행했으며 이전 source/측정은 덮어쓰지 않았다.

- [측정 summary](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/remote/comparison/summary.json)
- [300개 raw timing 표본](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/remote/comparison/timings.npz)
- [검증 출력 NPZ](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/remote/comparison/verified_output.npz)
- [CUDA 계약 로그](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/remote/contracts.log)
- [로컬 계약 로그](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/local_contracts.log)
- [실행 source manifest](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/remote/source_manifest.json)
- [다운로드·통계 확인 receipt](../../outputs/task10i/shape_first_device_output_58752018_20261103_v1/download_receipt.json)
- [실행 방법](../../graph/experiments/shape_first_device_output_58752018_20261103_v1/README.md)

Archive SHA256:
`32da3f33a47d93230530abe890311f28dbcc6b7bdd42ec3df20ca21d038cbf3b`.
다운로드 뒤 archive hash, 현재 runtime/kernel/test/runner5개 source hash,
이전 작업 source와 reference의 동일성, 300개 raw sample에서 재계산한
중앙값/p95를 확인했다. Manifest에는705개 파일을 기록했다. Git dirty 상태의
source snapshot이며 commit만으로 재현된다고 주장하지 않는다. 동시에 존재한
tracking/로봇 필터 관련 변경은 건드리지 않았다.

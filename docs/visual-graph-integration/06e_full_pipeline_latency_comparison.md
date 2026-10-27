# 전체 파이프라인 모듈별 지연 비교

Run UUID: `63f1f0e3fb084fd68b0c95eb7ace36e0`. Vast **57987772**, RTX 3090, AMD EPYC 7K62. 실행 소스 `acbee68`이며 공통 `ExperimentLogger`로 원본 프레임별 시간, 소스 snapshot/hash, 환경, 검사 결과를 보존했다.

이전에 16.65ms로 언급한 Visual prototype은 Task03 PCA5+S3의 CPU SampledPoints까지다. 실제 Raw EDE 통합 경로는 Shape-First, F5/EE packet, observed components, Raw EDE, Graph, GNN/GRU, CPU Action7까지다. 두 경로의 실제 끝점을 유지한 채 공통 처리와 통합에서 추가되는 비용을 각각 측정했다.

## 실행 조건

- 동일한 원본 TRAIN episode 30002의 106프레임과 30047의 112프레임, 총 218프레임. 실제 base_camera RGB-D 해상도는 128×128이다. 모든 조건에서 같은 RGB-D, calibration, TCP/robot state와 같은 DINO/SAM2 가중치를 사용했다.
- 네 조건은 Visual S3 CPU/CUDA, Raw EDE CPU/CUDA다. CUDA 조건은 실험용 FP32 PCA5/픽셀 특징/pair angle/FPS를 선택한다. Raw EDE, Graph, 정책 구조는 그대로 호출한다.
- 조건 이름의 CPU/CUDA는 PCA5 evidence와 FPS의 실행 위치다. DINO/SAM2와 정책은 모든 조건에서 GPU로 실행한다. Raw EDE 후보 생성과 Graph 구성은 두 Raw 조건 모두 CPU다.
- 공통 추적기는 Visual BF16 native compiled image encoder, GPU 전처리, 공용 object state, memory/frame pruning 경로다. Torch 2.14.1+cu130, NumPy 2.5.4, Python 3.12.15, NVIDIA driver 580.178.04. CPU/BLAS thread 1개. CUDA matmul TF32는 껐고 cuDNN TF32는 켰다.
- 모델 load와 최초 compile을 제외하고, 네 조건 모두 두 episode 전체를 warmup했다. 이후 조건 순서를 회전하여 각 조건을 3회 실행했다. 조건당 654프레임이며 그중 순수 추적 585, DINO 실행 63, 첫 프레임 6개다.
- 실제 연속 호출의 시작부터 최종 CPU 출력까지 wall time을 측정했다. H2D/D2H와 동기화 비용을 포함한다. 센서 exposure/readout, 파일 preload, simulator step, 파일 저장과 Logger 기록은 프레임 시간 밖이다.
- 기존 IntegrationPolicy(d64/L2/GRU144/encoder128)를 FP32로 실행하고 episode 안에서 hidden을 유지했다. 두 Raw 조건은 동일한 seed42 초기 가중치와 TRAIN96 robot normalizer를 사용했다. 학습이나 optimizer update는 수행하지 않았다. 가중치 fingerprint는 config에 보존했다.
- 각 모듈의 외부 wall interval은 서로 겹치지 않도록 분할했다. 아래 평균의 합은 실제 전체 평균과 일치한다. p50과 p95는 전체 interval에서 별도로 계산했다.

## 재탐색 없는 프레임: 모듈별 평균 ms

| 모듈 | Visual S3 CPU | Visual S3 CUDA | Raw EDE CPU | Raw EDE CUDA |
|---|---:|---:|---:|---:|
|DINO|0.000|0.000|0.000|0.000|
|Prefix 초기화 판단|0.012|0.012|0.014|0.014|
|GPU 전처리 + SAM2 추적 + CPU mask 반환|37.071|37.572|36.067|36.189|
|PCA5 evidence 전체|4.849|1.415|4.922|1.493|
|S3 sampling|3.533|1.673|—|—|
|Shape-First 후보 구성|—|—|0.440|0.459|
|Contour · scale · voxel|—|—|3.847|3.871|
|Shape thinning · Surface FPS|—|—|2.290|0.794|
|Camera 선택 packet 조립 · 검증|—|—|4.092|4.080|
|F5 gather · EE 변환 · v2 packet|—|—|1.448|1.450|
|Observed component 생성|—|—|0.595|0.597|
|Raw EDE 후보 · edge · metadata 생성|—|—|4.929|4.933|
|Producer packet 수신 검증|—|—|2.373|2.366|
|Robot geometry EE 변환|—|—|0.100|0.100|
|Graph 구성 전체|—|—|4.382|4.375|
|Graph tensor packing|—|—|0.352|0.343|
|GPU 입력 전송|—|—|0.230|0.215|
|GNN · readout · GRU · action head|—|—|2.719|2.693|
|Action7 CPU 반환|—|—|0.054|0.053|
|Downstream 함수 연결|0.005|0.005|0.502|0.503|
|프레임 생성 · 나머지 연결|0.076|0.072|0.131|0.127|
| **실측 전체 평균** | **45.546** | **40.748** | **69.486** | **64.653** |

`—`는 해당 경로에서 실행하지 않는 단계다. Visual 경로의 끝점은 CPU SampledPoints, Raw EDE 경로의 끝점은 CPU Action7이다.

## 실제 전체 지연: 평균 / p50 / p95 ms

| 경로 | 재탐색 없음 | DINO 실행 프레임 | 첫 프레임 | 모든 프레임 | 추적 50ms 이내 |
|---|---:|---:|---:|---:|---:|
|Visual S3 CPU|45.546 / 54.270 / 58.229|162.839 / 163.236 / 166.638|288.947 / 288.377 / 296.971|59.078 / 54.471 / 163.408|44.62%|
|Visual S3 CUDA|40.748 / 48.552 / 49.237|160.543 / 160.707 / 162.592|279.159 / 277.093 / 285.638|54.475 / 48.703 / 160.992|100.00%|
|Raw EDE CPU|69.486 / 78.691 / 99.287|178.716 / 179.622 / 186.106|326.732 / 324.035 / 355.158|82.368 / 80.563 / 181.382|25.47%|
|Raw EDE CUDA|64.653 / 73.345 / 88.289|177.020 / 177.098 / 182.407|316.045 / 315.621 / 334.194|77.784 / 74.891 / 178.375|39.49%|

DINO가 실행된 프레임의 DINO 자체 평균과 SAM2 평균:

| 경로 | DINO 자체 ms | Prefix 판단 ms | SAM2 + mask ms |
|---|---:|---:|---:|
|Visual S3 CPU|125.903|1.683|30.222|
|Visual S3 CUDA|125.802|1.666|30.227|
|Raw EDE CPU|126.199|1.655|30.308|
|Raw EDE CUDA|126.379|1.648|30.294|

## 비용의 원인

Raw EDE CPU의 mask 반환 이후 비용은 **33.405ms**, GPU sampling을 선택하면 **28.451ms**다. Visual S3는 각각 **8.463ms / 3.164ms**다. Raw EDE 자체는 **4.929ms / 4.933ms**, Raw EDE+robot EE+Graph+packing+transfer+GNN/GRU+Action7 반환의 합은 **12.765ms / 12.711ms**다. 나머지는 PCA5, Shape-First, packet 조립/변환/검증, observed component와 연결 비용이다.

Raw EDE CPU에서 PCA5 4.922ms, 후보/contour/voxel/thinning/Surface FPS 6.577ms, camera 선택 packet 4.092ms, F5/EE 1.448ms, producer 수신 검증 2.373ms가 각각 측정됐다. Graph 4.382ms 안에는 receiver validation/copy 2.348ms, robot-object 선택 0.219ms, Peg-Box 전역 선택 1.618ms, tensor 준비 0.191ms가 포함된다. 이 내부 항목들은 상위 Graph 시간에 이미 들어 있다.

Raw EDE 후보 생성 내부 평균은 CPU에서 candidate_total 3.790ms, Delaunay 2.632ms, collinearity chain 0.388ms, raw metadata packaging 0.825ms다. Delaunay/chain은 candidate_total의 하위 항목이다.

PCA5 evidence 내부 평균 ms:

| 경로 | Geometry | PCA5 normal | Pixel feature | Pair evidence | GPU normal+feature+전송 |
|---|---:|---:|---:|---:|---:|
|Visual S3 CPU|0.371|3.046|0.761|0.661|—|
|Visual S3 CUDA|0.374|—|—|0.525|0.411|
|Raw EDE CPU|0.392|3.096|0.768|0.655|—|
|Raw EDE CUDA|0.392|—|—|0.528|0.466|

GPU condition에는 CPU feature unpack 약 0.094–0.096ms도 포함된다. 표의 세부 항목은 PCA5 evidence 전체의 내부 진단이다.

## 정책 내부 CUDA event 진단

모듈 hook으로 별도 연속 pass를 실행했다. 아래 GPU event interval은 상위 wall-time 정책 호출에 포함되는 내부 시간이며 전체 wall-time에 다시 더하지 않는다.

| 경로 | GNN + readout | recurrent encoder | GRU | action head |
|---|---:|---:|---:|---:|
|Raw EDE CPU|1.711|0.086|0.203|0.103|
|Raw EDE CUDA|1.714|0.086|0.207|0.104|

## Episode별 처리량과 지연

| 경로 | Episode | 평균 선택 point | 평균 graph edge | 전체 평균 / p50 / p95 ms |
|---|---|---:|---:|---:|
|Visual S3 CPU|30002|99.48|0.00|32.669 / 32.558 / 36.183|
|Visual S3 CPU|30047|256.00|0.00|55.290 / 55.009 / 58.648|
|Visual S3 CUDA|30002|99.48|0.00|30.597 / 30.784 / 31.589|
|Visual S3 CUDA|30047|256.00|0.00|48.430 / 48.918 / 49.317|
|Raw EDE CPU|30002|64.76|351.45|47.154 / 47.461 / 54.696|
|Raw EDE CPU|30047|260.88|1383.64|86.386 / 84.605 / 102.986|
|Raw EDE CUDA|30002|64.76|351.45|45.283 / 45.433 / 50.980|
|Raw EDE CUDA|30047|260.88|1383.64|79.312 / 78.440 / 91.876|

선택 point 수는 Visual S3에서 최대 256점이며, Raw EDE 경로는 기존 Shape-First 규칙을 따른다. 일부 프레임에는 Raw 경로의 point 수가 더 적다. 비용 차이는 point 증가만으로 설명되지 않는다.

## 원래 경로 대조와 검증

- 계측 조건의 첫 반복에서 모든 218프레임의 mask가 네 조건 사이에 일치했다. 대표 6프레임의 CPU Raw EDE packet은 계측 없는 원래 packet_from_frame 출력과 exact 일치했다.
- 모든 2,616개 본 측정 프레임에서 모듈 합계와 전체 wall-time의 차이는 0.0ms였다. 이는 분할한 동일 프레임 interval의 수치 대조다.
- 계측 없는 원래 연속 경로도 별도 실행했다. Visual S3 CPU의 추적 평균/p50/p95는 45.915/54.530/58.998ms, Raw EDE CPU는 70.348/80.428/101.213ms다. 본 계측의 평균과 차이는 각각 0.8%, 1.2%였다.
- GPU 변경에서 선택 point 수 변화는 없었다. source Jaccard 평균은 S3 0.999894, Raw 0.999610이며, Raw 조건의 CPU Action7 최대 절대 차이는 1.42e-5였다. 수치/선택 차이는 이 성능 실험에서 허용하고 실제 값을 기록했다.
- 로컬 관련 sampling/streaming 계약 테스트 30개 통과. GPU smoke와 전체 3회 실행은 exit 0. 모델/정책 구조, 기존 CPU 기본값과 공유 simulator 코드는 수정하지 않았다.

## 재현 및 기록

코드: `graph/integration_baseline/pipeline_latency_comparison.py`. 공통 ExperimentLogger를 재사용했고 외부 게시/업로드는 실행하지 않았다. Artifact archive SHA-256: `b3d680d1b653b551f4bc41c7e77ccc1f50c26612f3d73c6a0e8ac06340a5c983`.

로컬 원본: `outputs/graph/integration_baseline/pipeline_latency_comparison/63f1f0e3fb084fd68b0c95eb7ace36e0/`. `summary.json`, `frames.json`, `checks.json`, `policy_detail_frames.json`, `native_control_frames.json`, `source_snapshot/`, `source_hashes.json`, `environment_packages.txt`를 확인할 수 있다. `analysis.json`과 `analysis_script.py`는 원본 기록에서 파생한 로컬 사후 분석이다.

```bash
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TQDM_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
python -B -m graph.integration_baseline.pipeline_latency_comparison \
  --inputs /workspace/full_pipeline_inputs \
  --dino /workspace/full_pipeline_models/dino \
  --output-root /workspace/experiments --repeats 3
```

`--inputs`는 기존 dataset/split/schema와 두 episode HDF5, TRAIN96 robot 행을 보존한 입력 폴더다. 각 원본 hash는 summary config에 기록돼 있다. 이전 M4 Pro의 16.65ms(512×512)나 다른 서버의 22.3ms를 이번 128×128 수치에 혼합하지 않았다.

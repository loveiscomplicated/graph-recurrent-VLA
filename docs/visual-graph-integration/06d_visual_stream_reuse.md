# Visual streaming 경로 재사용 검증

이 변경은 통합 Vision의 SAM2 실행 경로를 기존 Visual의 streaming 구현에
연결한다. `Shape-First/PCA5/F5`, Raw EDE, graph k4/M16, GNN/GRU는 변경하지
않았다. 전체 지연의 원인을 Raw EDE로 단정하는 변경이 아니다.

## 구현과 적용 범위

- `visual/scripts/stream_v1.py`의 GPU 전처리, propagation, memory 정리를
  직접 사용한다. 별도의 SAM2 추론 구현을 복사하지 않았다.
- 새 `visual_stream` backend는 SAM2 BF16 autocast, native image encoder
  compile, GPU 전처리, object 공용 state, 최근 non-conditioning memory 16개와
  GPU frame 2개 보존을 사용한다. DINO는 SAM2 autocast 밖에서 실행한다.
- 기존 prefix 초기화 규칙, prompt, threshold, 5-frame scan, 확인 규칙을
  유지한다. 과거 RGB의 raw DINO 결과만 episode 안에서 재사용하고, prefix에
  따라 달라지는 객체 선택은 다시 계산한다.
- 관측 시점 `t`에서 `num_frames=t+1`이고 현재 프레임만 전진한다.
  새 객체는 현재 관측의 box로 기존 state에 추가한다. 기존 객체의 memory나
  conditioning을 초기화하지 않는다. 미래·삭제된 RGB 접근은 예외로 거부한다.
- `gpu_run prepare --vision-backend visual_stream`과 offline/online factory에
  연결했다. 기본값은 기존 `legacy_fp32`다. 기존 run, cache, receipt,
  checkpoint를 새 backend로 바꾸거나 자동 재사용하지 않는다.
- CPU에서 새 backend를 선택하면 명시적으로 거부한다. 실패 시 다른 경로로
  자동 전환하지 않는다. reset/close는 state와 frame buffer를 해제한다.

사용 예시와 전체 검증 CLI는
[integration README](../../graph/integration_baseline/README.md#opt-in-reuse-of-visuals-gpu-streaming-path)에
있다.

## 격리와 서버 보존

개발 clone은 `graph-recurrent-VLA-vision-stream`, 브랜치는
`temporary/20261027-back-to-visual-v1`다. 기준 커밋은
`58e645b4eba1bc5a06c827bb4082508122de467a`이고 GPU 실행 코드는
`392411ac25dfceb9ddedfe013f5b031b67f9ede9`다.

Vast `57640710`의 `/workspace/vision-stream-isolated/repo`에 별도로 clone했다.
기존 `/workspace/graph-recurrent-VLA`는 HEAD `b00311e`와 여러 미커밋 변경이
있어 checkout, pull, reset, package 설치를 실행하지 않았다. GPU가 비어 있을
때만 검증을 시작했다. 다른 GPU PID가 생기거나 디스크 여유가 2GiB 미만이면
검증 도구가 자신이 만든 worker process group만 종료한다. CPU thread 1개,
compile worker 2개, nice 10, Torch allocator 한도 35%를 사용했다.

32GB 디스크 제약 때문에 private venv에서 기존 lock과 일치하는 설치 package와
공개 model weight를 읽기 전용으로 참조했다. Python bytecode 쓰기를 끄고
HF/Torch/Inductor/Triton/CUDA cache와 임시 파일은 전부 격리 디렉터리에 뒀다.
다른 tmux, supervisor, Codex process를 종료하거나 서버 설정을 변경하지 않았다.

## 비교 조건

RTX 3090, Torch 2.8.0+cu128, 원본 TRAIN episode `30002`와 `30047`의
기록된 RGB-D 218프레임을 사용했다. SAM2 checkpoint SHA256은
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`,
SAM2 revision은 `2b90b9f5ceec907a1c18123530e92e794ad901a4`다.
설치된 SAM2의 선택적 `_C` extension은 없어서 hole-filling 후처리가
건너뛰어지는 경고가 있었다. 모든 비교가 이 동일 환경을 사용했으며, 기존 설치를
보존하기 위해 extension을 새로 설치하지 않았다.
모든 variant는 두 episode 전체를 먼저 warmup하고, model load·compile·warmup을
정상 추적 시간에서 제외한다. 전처리 전 raw RGB부터 CPU boolean mask 반환까지
CUDA synchronize로 측정한다. 최초 프레임, DINO scan, 순수 추적을 분리한다.

native max-autotune을 쓰는 독립 실행 간에는 수치 차이가 관측된다. adapter
영향을 판정할 때는 **동일하게 컴파일된 predictor를 공유하고 직접 호출과
adapter의 순서를 번갈아 실행**한다. pixel equality, 초기화, 반복성,
packet/graph/Action7의 완전 일치를 요구한다. 허용 pixel 오차를 늘리지 않는다.
별도 `direct_control` 프로세스의 직접 호출 결과와 모든 독립 실행 차이도 보존한다.

RGB-D→CPU Action7은 별도 실제 순차 pass다. 기존 FP32 policy 수치 설정과
검증된 TRAIN96 전체의 normalizer, 동일 seed 42의 random policy를 사용한다.
optimizer update는 0회다. 이 결과는 실제 카메라, simulator closed-loop,
학습 정책의 성공률 또는 전체 20Hz 달성을 검증하지 않는다.

## 검증 기록

- macOS CPU: 통합 suite 166개 실행, 선택적 데이터 검사 3건 skip, 나머지 성공.
  기본 discover는 기존 Task06-B 선택 fixture의 `source_snapshot/recurrent/expert.py`
  누락으로 setup error가 나므로 `TASK06B_RUN`을 존재하지 않는 선택 경로로
  지정해 해당 class를 명시적으로 제외했다. fixture나 검사 코드는 수정하지 않았다.
- 새 streaming 계약 테스트 13개는 macOS와 Linux에서 통과했다.
- 공통 logger/artifact suite 42개 중 1개 실패. 외부 organization 거부를 기대하는
  기존 artifact 테스트이며 변경 전 기준 커밋 `58e645b`를 별도 임시 디렉터리에
  추출해서도 같은 실패를 재현했다. 공통 logger/publisher는 변경하지 않았다.
- 실물 SAM2: 직접 호출과 adapter mask 일치, prefix-only와 suffix 변경의 인과성,
  미래 frame 거부, 빈 초기화, reset, 늦은 객체 추가 시 기존 conditioning/memory
  보존, 24프레임/24 image encode, frame/memory 상한을 검증한다.
- simulator backend, live camera, training, closed-loop는 이번 작업에서 실행하지
  않았다. 기록과 report는 공통 `ExperimentLogger`를 사용하며 외부 artifact나
  W&B를 게시하지 않았다.

실패한 초기 실행도 보존했다. `e22966ef4a3d499695d9a49c8754fcea`는 detector
trace의 tuple/list JSON readback 불일치로 종료됐고 serializer를 수정했다.
`8cd24005eb544b949be8a32297635ac9`는 독립 프로세스 간 179pixel 차이로 gate를
실패했다. 후속 `b943073463494dfb92cc32b7ac58ce51`은 동일 predictor의 완전 일치를
통과했고 독립 direct↔direct에서도 161pixel 차이를 관측했다. 이 초기 1회 측정의
paired adapter 추적 p50/p95는 45.508/55.460ms이므로 50ms deadline 통과로
보고하지 않는다. 반복 측정 결과를 아래에 따로 기록한다.

## 첫 3회 반복 측정: 실행 경로와 FP32 baseline

실행 ID는 `332b64cf1ce04f8097092ca28561017b`다. 이 실행은 13개 조건 중
6개를 완료한 뒤 FP32 native compile에서 디스크 보호로 중단됐다. 전체 run은
`INCOMPLETE`이고, 완료한 직접 호출/adapter의 경로 복원 gate는 `RESTORED`다.
완료한 각 조건의 순수 추적은
585프레임, DINO scan은 63프레임, 최초 프레임은 6프레임이다.
아래 수치는 독립 process의 정상 측정 pass이며 단위는 ms다.

| 경로 | 추적 평균 | 추적 p50 | 추적 p95 | DINO scan 포함 평균 |
| --- | ---: | ---: | ---: | ---: |
| 기존 FP32 offline | 188.240 | 230.047 | 232.496 | 1959.070 |
| 기존 FP32 online | 207.321 | 248.690 | 250.816 | 1975.435 |
| Visual 직접 호출 | 38.649 | 44.360 | 52.456 | 174.247 |
| 직접 호출 독립 대조군 | 38.764 | 44.204 | 52.273 | 180.155 |
| 새 adapter | 39.143 | 44.744 | 52.920 | 181.003 |

기존 측정의 추적 중앙값 229.5ms, scan 평균 1951ms와 비슷한 느린 경로를
같은 서버와 입력에서 재현했다. 새 adapter의 추적 중앙값은 이 offline 기준보다
약 5.14배 빠르고 scan 평균은 약 10.82배 빠르다. 이는 최적화 묶음 전체의 효과다.
CPU JPEG staging을 포함하는 online 경계와 비교하면 추적 중앙값은 약 5.56배,
scan 평균은 약 10.91배 빨랐다. offline의 JPEG 디렉터리 준비 시간은 frame
측정 밖에 있으므로 offline/online 절대값 차이를 SAM2 연산 차이로 해석하지 않는다.

동일 predictor를 공유한 순서 교대 비교는 다음과 같다.

| 경로 | 추적 평균 | 추적 p50 | 추적 p95 | 50ms 이내 비율 |
| --- | ---: | ---: | ---: | ---: |
| Visual 직접 호출 | 39.831 | 44.104 | 55.087 | 83.76% |
| 새 adapter | 39.065 | 44.181 | 55.228 | 87.01% |

adapter 추가 지연은 p50 **0.176%**, p95 **0.256%**로 10% 허용 범위 안이다.
반복 mask, 초기화, 동일 predictor의 mask와 downstream packet/graph/Action7은
전부 정확히 일치했다. 두 episode의 실물 SAM2 계약 검사도 모두 통과했다.
직접 호출도 50ms p95를 넘었고 adapter도 넘었으므로, 이 결과는 **빠른 실행
경로 복원**과 **추적 deadline 미달**을 함께 기록한다.

FP32 컴파일의 cache 증가로 2GiB 디스크 reserve가 감지됐을 때 worker
`107065`의 process group에만 SIGTERM을 보냈다. 다른 process를 종료하거나
다른 파일을 지워 여유를 확보하지 않았다. GPU와 worker 종료를 확인한 다음,
우리 Inductor/Triton cache 1,611,622,210바이트만
`/dev/shm/vision-stream-isolated-cache`로 옮겼다. 디스크 여유는
1,743,675,392바이트에서 3,414,081,536바이트로 회복됐다. RAM cache는
재생성 가능한 임시 파일이며, 원본 artifact는 계속 디스크에 보존했다.
`resource_recovery.json`에 cgroup memory 한도/사용량과 전후 여유를 기록했다.

동일 source commit과 입력으로 새 `ExperimentLogger` UUID를 만들어 직접 호출,
adapter와 8개 추적 조건을 다시 실행했다. 이미 완료한 느린 FP32 offline/online
측정은 위 실행의 결과를 사용하며, 실패한 run을 성공한 run으로 덮어쓰지 않는다.

RAM 영역의 `noexec` 설정 때문에 첫 재개 `58711af558684107aed974058c999a65`는
Triton CPU helper `.so` 로딩에서 종료됐다. 모델 추론이나 package를 수정한 것이
아니며, 실패 artifact를 보존했다. 이후 private 환경에서 Triton의 설치된
`FileCacheManager`를 재사용해 `.so`만 실행 가능한 우리 디스크 cache로 보내고
나머지 큰 중간 파일은 RAM에 보관했다. 실제 helper 로딩과 cache 경로/내용 검사를
통과한 기록은 `cache_placement_check.json`, helper SHA256은
`d4100ff8d8d8948fa05b8ec9f339eb17bba546d0b8ac8b8c6cfe9ee3954c0678`이다.
이 조정은 원본 package, driver, mount 설정을 바꾸지 않았다.

## 전체 순차 pass와 출력 차이

첫 3회 실행의 별도 순차 pass는 각 218프레임을 한 번 측정했다. policy를 위한
기존 deterministic/FP32 설정을 유지했으므로 위의 Visual 실행 조건과 구분한다.
각 행은 실제 mask→packet→graph→CPU Action7의 순차 호출이며 단위는 ms다.

| 경로/대상 프레임 | 전체 평균 | 전체 p50 | 전체 p95 |
| --- | ---: | ---: | ---: |
| 기존 FP32 online, 전체 | 416.701 | 291.618 | 1884.668 |
| 새 adapter, 전체 | 104.159 | 101.582 | 222.223 |
| 기존 FP32 online, scan 없는 추적 프레임 | 246.704 | 289.569 | 311.640 |
| 새 adapter, scan 없는 추적 프레임 | 88.111 | 98.315 | 129.040 |

새 adapter의 scan 없는 pass에서 mask p50은 52.868ms, packet 32.217ms,
graph/pack/transfer 4.583ms, policy/CPU 반환 5.955ms였다. 각 단계의 중앙값을
더한 값은 전체 중앙값과 같지 않다. packet 단계는 sampling과 feature/좌표 변환을
포함하며 Raw EDE만의 비용이 아니다. 현재 결과로 전체 지연을 Raw EDE에 귀속하지
않으며, 전체 20Hz deadline도 통과하지 않았다.
전체 218프레임의 `component_raw_ede_ms` 자체는 평균 8.182ms, 중앙값
8.415ms였다. 이 계측 범위는 별도 결과 JSON의 component 이름 그대로 해석한다.

원래 FP32와 새 BF16/GPU 전처리 경로가 pixel-exact하다고 주장하지 않는다.
위 실행의 독립 비교에서 두 legacy 경로는 직접 Visual 호출과 4,030pixel,
8개 object-presence 판단이 달랐고 random Action7 최대 절대 차이는 0.012075였다.
초기화 객체와 frame은 같았지만 DINO box/score의 작은 수치 차이도 있었다.
직접 호출 독립 대조군 자체도 219pixel 차이가 있었다. 반면 **동일 predictor의
직접 호출과 adapter**는 mask, packet, graph와 Action7이 정확히 일치했다.
따라서 새 backend는 기존 FP32 checkpoint/cache를 동등한 결과라고 간주해
대체하지 않고, 새 실행에서만 명시적으로 선택한다.

## 재검증과 원인별 대조

재검증 `f0b71e9671a840bfab62e26d31e097e7`는 선택한 10개 조건을 모두 완료했고
exit code 0, 실행 경로 gate `RESTORED`다. source commit과 입력은 첫 실행과
같다. 직접 호출/adapter의 실물 SAM2 계약, 반복성, 초기화와 동일 predictor의
mask/packet/graph/Action7 완전 일치도 통과했다.

독립 pass의 adapter 추적 평균/p50/p95는 **38.139/44.301/51.071ms**, DINO
scan 포함 평균은 **180.215ms**다. 동일 predictor paired pass의 직접 호출
p50/p95는 44.398/53.235ms, adapter는 44.260/48.548ms였다. 해당 paired
pass에서는 50ms p95 gate를 통과했지만 앞의 독립 pass와 첫 실행에서는 넘었다.
따라서 안정적인 20Hz 달성으로 일반화하지 않는다. 음의 adapter 추가 지연
(p50 −0.312%, p95 −8.806%)을 adapter 자체의 가속 효과라고 해석하지 않는다.
이 재검증의 별도 RGB-D→Action7 pass 중앙값도 adapter 99.321ms였다.

아래는 raw DINO 출력을 고정한 기준 대비 단일 조건 변경이다. 각 조건은
3회, 추적 585프레임이며 단위는 ms다. DINO 추론 시간은 포함하지 않는다.

| 변경 조건 | 추적 평균 | 추적 p50 | 추적 p95 | 기준 대비 p50 차이 |
| --- | ---: | ---: | ---: | ---: |
| streaming 기준 | 37.706 | 44.028 | 49.516 | 0 |
| SAM2 BF16 autocast 끄기 | 174.143 | 214.180 | 216.169 | +170.152 |
| PIL 전처리 | 55.017 | 61.112 | 67.645 | +17.084 |
| image encoder compile 끄기 | 50.674 | 57.099 | 64.526 | +13.070 |
| preflight 재사용 | 38.246 | 44.207 | 51.456 | +0.179 |
| 초기화 사건별 state 분리 | 37.768 | 44.328 | 49.526 | +0.300 |
| 오래된 memory pruning 끄기 | 37.791 | 44.364 | 48.130 | +0.335 |
| 기존 deterministic/cuDNN 설정 | 43.056 | 51.362 | 56.015 | +7.334 |

이 조건에서 가장 큰 차이는 SAM2 정밀도였고, 전처리와 compile에서도 지연 증가를
보였다. preflight 재사용과 pruning의 작은 시간 차이는 이 실행의 변동 범위와
함께 해석해야 한다. pruning은 218프레임 시험보다 긴 streaming의 memory
상한을 유지하는 의미도 있다. 각 변경은 상호작용하므로 차이를 더해 전체 원인
비율로 만들지 않는다. 독립 process의 작은 kernel/수치 차이와 순차 측정의
시간 변동도 포함된다.

이 두 실제 episode에서는 관측된 객체가 모두 frame 0에서 초기화됐다. 따라서
초기화 사건별 state 분리 조건도 episode 안에서는 한 state를 사용했다.
`+0.300ms`를 늦은 객체 state 중복의 비용 또는 그 비용이 없다는 증거로
해석하지 않는다. 늦은 객체가 실제로 별도 state를 만들게 하는 제어 입력을
추가로 측정했다.

별도 제어 실험 `7e026e52a7854e9489431fbdcef28cf5`는 관측 RGB0를 64회 반복하고,
그 RGB에서 얻은 DINO box로 peg를 frame 0, box를 frame 5에 추가했다. 동일
compiled predictor를 공유하며 순서를 교대한 3회 측정이다. 실제 trajectory의
기여도나 detector 품질을 나타내는 결과는 아니다.

| 늦은 객체 처리 | image encode 횟수 / 64프레임 | 추가 이후 추적 p50 | 추가 이후 추적 p95 |
| --- | ---: | ---: | ---: |
| 공유 state | 64 | 44.552ms | 49.685ms |
| 초기화 사건별 분리 state | 124 | 50.950ms | 53.892ms |

frame 5 이후를 각각 174프레임 측정했다. mask는 pixel-exact했고 반복 결과도
같았다. 분리 state는 이 제어 입력에서 p50 6.397ms, p95 4.207ms를 더 썼고
image encoding을 중복했다. 실제 late-object correctness 검사의 기존 memory와
conditioning 보존, 24프레임/24 image encode 결과와 함께, 새 객체를 기존
state에 추가하는 구현의 근거로 사용한다.

## 보존한 결과

최종 source/model runtime identity는 두 3회 실행의 완료 조건 전체에서 같았다.
원본 서버 저장소의 HEAD, 미커밋 상태, 확인한 source hash도 최종 비교에서 모두
작업 전과 같았다. 기록은 공통 `ExperimentLogger`의 UUID/metadata/sidecar를
유지하며, 실패 기록을 수정하거나 기존 실행 ID에 새 결과를 덮어쓰지 않았다.

서버 artifact는 `/workspace/vision-stream-isolated/experiments/<uuid>/`, 로컬
사본은 격리 clone의
`outputs/graph/integration_baseline/visual_stream_validation/experiments/<uuid>/`에
있다. `plan.json`, `source_hashes.json`, source snapshot, raw frame timing,
mask/Action7 NPZ, real-weight check, HTML/Markdown 보고서를 함께 보존했다.
cache 배치와 복구용 script/proof도 로컬 사본에 있다. 모델 weight와 compile
cache를 결과 묶음에 복사하거나 외부 게시하지 않았다.
검증 종료와 우리 worker/compiler process group의 종료를 확인한 뒤, 우리 RAM
compile cache의 일반 파일 1,720,138,609바이트를 정리했다. 다른 process와
원본 package/weight/artifact는 그대로 두었고 `cache_cleanup.json`에 기록했다.

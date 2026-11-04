# Task10-I GPU 전환 전 CPU·NumPy 구현의 단계별 재측정

2026-11-04 KST. 사용자 요청의 “최적화 전”은 **이번 대화에서 GPU 전환하기 전의 CPU·NumPy 구현**이다.
단계별 재측정을 완료하고 [50ms 현황 문서](10i_live_50ms_non_sam2_optimization_status_20261104.md)의 표에 추가했다.
현재 GPU 열도 같은 관측으로 다시 측정하여 갱신했다. 이전 GPU 실측은
[기존 보고서](10i_live_stage_timing_58752018_20261104.md)에 보존한다.

## 비교하는 코드와 통제 조건

- CPU 구버전은 `b423d00^`인 **`709798983435875e374e9a8cd6df8af9e6c7bb7f`**다.
  mask-only CUDA projection을 Graph 기본 경로로 연결하기 직전 버전이다.
- 전체 XYZ, CPU PCA/Shape-First, camera→EE→World→EE 왕복, Python 기록·3mm dedup,
  CPU final FPS/Graph, 중간 Raw EDE와 반복 depth getter를 **그대로 유지**했다.
  그 이전에 이미 있던 CPU PCA 개선까지 되돌린 더 오래된 naive PCA 구현을 뜻하지 않는다.
- 현재 GPU는 앞선 live 실측에서 보존한 source snapshot의 runtime bytes를 그대로 사용했다.
  새 observer/runner만 추가했다. production 처리 로직·threshold·SAM2 설정은 수정하지 않았다.
- VastAI **58752018**, RTX3090, 같은 venv·NumPy/SciPy/Torch와 CPU thread1.
  CPU affinity·공유 host 전체 부하는 고정하지 않았다. 다른 job 중단·패키지 설치·게시 없음.
- **512×512, base/hand 두 카메라, seed42**, 같은 current raw RGB·signed-mm depth·보정값,
  로봇 상태·collision surfaces·Terra/SAM2 raw mask를 사용했다. 새 Terra 실제 요청은2회다.
- GPU live를 먼저 실행하여39프레임 입력을 고정했다. 같은 trace의 detailed/natural-warm 재실행에서
  raw mask와 Graph 동일성을 확인한 뒤, 별도 프로세스의 CPU 구버전이 정확히 그 NPZ를 읽었다.
  입력 로딩·파일 기록은 CPU/GPU timer 밖이다.
- 각39프레임 중 t0 초기화·t1..8 준비를 제외하고 **t9..38의30프레임**을 집계했다.
  숫자는 **1프레임의 중앙값·p95**이며 누적/평균이 아니다.

이는 **현재512·Terra/native SAM2 관측으로 입력 조건을 맞춘 CPU 기하 경로 비교**다.
역사적 live의128 기본 해상도·DINO·JPEG SAM adapter까지 되돌려 실행한 비교는 아니다.
렌더링·Terra·SAM2는 실제 실행한 **공통 상류**다. 이를 CPU 독립 실측값으로 복사하지 않았다.
CPU `CurrentGeometry.build`는 고정된 실제 live 입력을 재생했으므로 CPU full closed-loop rollout이라는 뜻도 아니다.

## 단계별 비교

단위 ms/현재 관측1프레임, N30. GPU는 단계별 완료 동기화를 포함한 live 진단값이다.

| 순서 | 단계와 실제 측정 범위 | CPU 구버전 중앙값 | CPU 구버전 p95 | 현재 GPU 중앙값 | 현재 GPU p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 시뮬레이터 RGB·Depth·보정값 생성 / CUDA 관측 출력 | 공통 상류 | 공통 상류 | 5.256 | 10.512 |
| 2 | 센서·Robot76·FK 형상 준비 | 6.466 | 7.868 | 6.235 | 6.905 |
| 3-A | Terra API / RGB→box scan | 공통·미호출 | — | 미호출 | — |
| 3-B | 로봇 준비 요청·snapshot·Depth 준비: 호출자 | 6에 포함 | 6에 포함 | 4.415 | 6.677 |
| 3-B 병행 | 로봇 형상·hull 준비: 병행 CPU worker | 별도 worker 없음 | — | 7.009 | 11.718 |
| 4 | SAM2 이미지 resize·정규화 | 공통 상류 | 공통 상류 | 0.866 | 1.143 |
| 5 | SAM2 분할·추적·mask/추적 기록 | 공통 상류 | 공통 상류 | 427.914 | 430.354 |
| 6 | 로봇 필터 적용 / CPU는 준비도 포함 | 1264.391 | 1321.216 | 9.303 | 9.715 |
| 7 | 손목 카메라 gate | 876.375 | 912.875 | 3.980 | 4.706 |
| 8 | 입력 정합성·condition/view·보정 준비 | 1.394 | 1.651 | 1.423 | 1.672 |
| 9 | 역투영 / GPU는 compact row index 포함 | 4.855 | 5.378 | 1.681 | 1.805 |
| 10 | PCA 법선·기하 특징 | 70.186 | 74.669 | 3.215 | 3.291 |
| 11 | 샘플링 evidence·정보 구성 | 23.576 | 24.561 | 0.345 | 0.370 |
| 12 | Shape-First·외곽선·voxel·FPS·출처 출력 | 86.996 | 90.470 | 15.311 | 16.016 |
| 13 | 좌표 변환·packet 구성·검증 | 31.554 | 35.089 | 7.491 | 7.702 |
| 14 | 관측 구성·카메라 간 일치 검사 | 4.683 | 5.382 | 2.706 | 2.899 |
| 15 | 점별 기록·3mm 중복 제거·기존 보조 분기 | 103.079 | 151.968 | 11.999 | 12.221 |
| 16 | 물체별 최종128점 FPS | 13.203 | 15.594 | 1.689 | 1.841 |
| 17 | 표면·로봇–물체·Peg–Box 엣지 및 Edge4 | 15.133 | 16.693 | 5.035 | 5.868 |
| 18 | 노드·메타데이터·Graph 반환·최종 검증 | 4.890 | 5.761 | 1.998 | 2.175 |
| 추가 | 구버전 중간 component·Raw EDE 생성 | 30.857 | 32.896 | 생략 | — |
| 경계 | 공통 CUDA mask→CPU 입력 경계 | 0.176 | 0.241 | CPU 반환 없음 | — |
| 기타 | 함수 사이 연결·변환·호출 비용 | 18에 포함 | 18에 포함 | 1.516 | 1.663 |

## 서로 다른 실행 경계의 대응

- CPU 센서2는 구버전 `sensors` 함수의 원래 AST를 실행했다. 원래 robot/state helper 두 모듈은
  현재 source와 byte-identical이다. 공유 native CUDA 관측의 host 이미지 readback이2에 포함된다.
  따라서 “과거 CPU renderer의1만 따로 측정한 값”이 아니다.
- 1·3-A·4·5의 **공통 상류**는 두 기하 경로가 공유한 실제 frontend다.
  CPU용 DINO/JPEG/SAM2 실행 시간을 새로 측정한 것으로 해석하지 않는다.
- CPU 형상·hull 준비는 직렬6 내부다. GPU3-B와 같은 비동기 worker가 없어서 따로 값을 만들지 않았다.
  GPU worker는 다른 처리와 겹치므로 전체에 더하지 않는다.
- CPU8은 조건 검사와 C0..C3 mask 구성이고 GPU8은 shared tensor views·보정·mask 인계다.
  CPU에 존재하지 않는 GPU views를 가상으로 생성해 계측하지 않았다.
- CPU gate 내부의 `consistent_views`는 실제 nested timestamp에 따라7로 재집계했다.
  14는 gate 이후 관측 일치 검사다. raw spans는 보존하고, 재분류 전후 총합 불변을 확인했다.
- CPU15에는 구버전 `represent(...,('SF',))`가 수행하던 Legacy/Hybrid 보조 구성도 포함한다.
  GPU tensor SF 경로는 해당 보조 분기를 조기 반환으로 건너뛴다.
- 구버전 중간 component·Raw EDE는 삭제하지 않고 별도 **추가 행**으로 측정했다.
  최종 Graph 엣지17과 구분한다. CPU13의 값에 중복 합산하지 않았다.
- 공통 CUDA mask를 CPU로 읽는 경계는 별도 측정했다. CPU geometry timer는 그 host mask가 준비된 뒤 시작한다.
- CPU 상위 builder의 나머지 검사·연결 비용은18에 포함됐고, GPU의 미분류 연결 비용은 기타 행이다.
- 두 경로에 수행·배치 차이가 있으므로 각 행을 동일한 순수 커널의 device 차이로 해석하지 않는다.
  단계별 percentile 합계 역시 전체 percentile과 같지 않다.

## 전체 경계의 직접 측정

| 경계 | CPU 구버전 중앙값 / p95 | 현재 GPU 중앙값 / p95 |
| --- | ---: | ---: |
| raw mask·센서 준비 이후 `CurrentGeometry.build`→Graph/records 반환 | **2539.591 / 2632.705ms** | **69.140 / 73.597ms** |

이 GPU 합계는 **SAM2를 호출하지 않는 별도 mask-ready 재생**이다.
39개 입력을 한 번 모두 준비 실행한 뒤 같은 입력을 측정했고,
로봇 준비·join·filter·gate·기하·샘플링·Graph를 포함했다.
큰 입력의 업로드·렌더링·센서 준비·SAM2와 외부 mask readback은 timer 밖이다.
CPU는 host 입력/CPU 출력, GPU는 resident 입력/CUDA 출력이다.
CPU에는 단계별 진단 hooks가 있고 GPU 합계에는 내부 단계 동기화가 없다.
따라서 소수점까지의 단일 speedup 배율보다 원자료와 경계를 함께 본다.

**69ms는 이전의 SAM2 외 약77/83ms와 다른 경계다.**
이전 근사치에는 센서 준비(렌더 포함 값에는 렌더링도)가 들어 있다.
이번69ms를 코드 최적화로83→69ms 개선한 결과로 표현하지 않는다.

현재 GPU의 실제 live 추적은 내부 단계 동기화 없는 재실행에서
센서→Graph 중앙값 506.379ms,
p95 513.009ms였다.
렌더링 작업까지 프레임별 합산하면 중앙값 511.808ms,
p95 523.304ms다.
CPU의 실제 render→SAM2→Graph 전체 연속 stopwatch는 측정하지 않았으므로 이 자리에 합성값을 넣지 않았다.

## 출력과 소스 검증

- GPU live117 Graph에서 두 재실행 각각39프레임의 raw mask 및 최종 다섯 배열이 일치했다.
- CPU는39개 프레임을 계측했다. t0·t9·t38에서 계측을 끈 원래 CPU 함수를 다시 실행하여
  다섯 Graph 배열, node IDs, gate trace의 정확한 일치를 확인했다.
- 같은 입력의 CPU/GPU 비교에서 `node_type`, `edge_index`, `edge_attr`, `edge_type`은
  **각39/39프레임 byte-exact**였다. 매 프레임259노드·80 directed edges다.
- `node_input`은 byte-exact가 아니어서 전부 exact라고 보고하지 않는다.
  저장된 t0·t25를 추가 재생한 비교에서 XYZ·validity/status는 exact,
  전체 node feature 최대 절대 차이는 **1.1920928955078125e-7**이었다.
  두 저장 프레임에서 사전 사용하던 float feature tolerance2e-5를 통과했다.
- 별도 GPU mask-ready 재생39프레임의 다섯 Graph 배열은 live 결과와 모두 exact였다.
- source 파일 current1803개·legacy1597개의 실행 전후 hash와 입력 NPZ39개 hash를 검증했다.
  구버전 핵심 파일은 Git commit의 bytes와도 대조했다.
- gate 시간 재집계는 같은 frame의 포함 관계만 사용했다. 시간 삭제·표본 제외·보정은 하지 않았다.
  CPU 합계 최대 2892.499ms와 기록/dedup의 큰 지연도 raw에 남겼다.

이번 장면은 최종 surface edge가0이다. 이전 surface edge가 많은 단일 Peg 표본에서 얻은
CPU Graph792ms 등을 이 표의 값으로 가져오지 않았다.
단일 seed·고정 행동에서의 결과로 다양한 visible-Peg 장면의 성능을 보장하지 않는다.
SAM2 hole-filling 확장 누락 등 기존 환경 제약은 원래 live 보고서와 동일하다.
CUDA는 Vast에서 실행했고 macOS CUDA 실행을 주장하지 않는다.

## 실험 식별과 원자료

공통 `ExperimentLogger`를 재사용했고 W&B disabled, publishing 없음.

| 실행 | run ID |
| --- | --- |
| 공유 live 입력 및 GPU 단계별 측정 | `68a9bb8ded654e569bbb958cb966d7ef` |
| 원래 CPU 구현의 단계별 측정 | `34991122deb346b9bec364e68bdb649a` |
| 현재 GPU mask-ready 합계 | `fd32bbc3a1584b728bce3d18c9535b3e` |

- [공유 live 입력 manifest](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/live_v1/inputs/manifest.json)
- [GPU summary](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/live_v1/summary.json), [GPU 단계별 spans](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/live_v1/detailed/measurement.json)
- [CPU raw summary](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/cpu_v1/summary.json), [CPU raw spans](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/cpu_v1/spans.json)
- [경계를 맞춘 CPU 집계](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/cpu_boundary_aligned_v1.json), [GPU mask-ready](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/gpu_ready_v1/summary.json)
- [Node feature 수치 비교](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/remote/node_equivalence_v1.json)
- [표의 원자료 JSON](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/comparison.json), [프레임별 CSV](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/matched_stages.csv)
- [Source 대조](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/source_verification.json), [검증 receipt](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/verification_receipt.json)
- [CPU source](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/legacy_source_v1.tar.gz), [현재 source](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/current_source_v1.tar.gz)
- [결과·입력 archive](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/results_v1.tar.gz), [live log](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/capture_v1.log), [CPU log](../../outputs/task10i/live_preopt_cpu_timing_58752018_20261104_v1/cpu_v1.log)
- [Runner](../../graph/experiments/live_preopt_cpu_timing_10i_20261104_v1/README.md)

Remote root: `/workspace/task10i_live_preopt_cpu_timing_20261104_v1`.
`results_v1.tar.gz` SHA256:
`f44436d2b6ed15486717b0647449f82cea2c3c5a6075748a7f1f511052f9998c`.
Node feature 추가 검증 JSON은 archive 생성 뒤 별도 보존했다.
실행한 source·기존 experiment·다른 작업의 production 파일을 덮어쓰지 않았다.

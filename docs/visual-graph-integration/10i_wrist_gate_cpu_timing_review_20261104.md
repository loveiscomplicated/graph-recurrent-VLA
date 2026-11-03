# Task10-I: CPU 손목 gate 시간 측정 독립 검토

검토일: 2026-11-04 KST. 대상 실행 ID: `46439da34962499a989535b737b71488`.

**보고서의 수치와 측정 경계는 원자료 및 코드와 일치한다. 수정이 필요한 오류는
발견하지 못했다.** 아래 시간은 기존 원격 측정의 재계산이며, 이번 로컬 검토에서
AMD EPYC 시간을 다시 측정한 결과는 아니다.

## 수치 및 입력 검증

| 입력 | 유효 clean Peg 점 | 15회 중앙값 | 최솟값–최댓값 | 판정 / 채택 픽셀 |
| --- | ---: | ---: | ---: | --- |
| `10i_s1_0531_track_t011` | 29,775 | 2,322.198434 ms | 2,286.004303–2,371.904047 ms | 통과 / 29,269 |
| `10i_s1_0921_t080` | 0 | 5.440191 ms | 5.368481–6.682011 ms | 거부 / 0 |

- 각 15개 원시 값에서 중앙값·최솟값·최댓값을 다시 계산했고, 실행 로그의 30개
  측정 행과도 일치했다. 모든 측정 행의 reference 일치 기록은 `true`다.
- 저장된 두 NPZ의 파일 hash, 각 카메라 raw·clean mask의 hash·shape·dtype·픽셀
  수를 확인했다. clean mask와 양의 signed depth로 다시 센 유효 손목 Peg 점 수는
  각각 29,775개와 0개다.
- 기존 검증 파일에 기록된 산출물 19개의 SHA-256, 실제 실행 소스 복사본 5개,
  현재 runner와 protocol의 hash가 모두 일치했다. CPU 모델은 실행 config에
  `AMD EPYC 7543 32-Core Processor`로 기록되어 있다.

## 측정 경계 검토

[runner](../../graph/experiments/wrist_gate_cpu_timing_10i_20261104_v1/run.py)는
untimed reference 1회, warmup 3회 뒤 `wrist_gate()` 호출 15회를 측정한다.
`perf_counter_ns()` 두 호출 사이에는 gate 호출만 있다. mask·trace 비교와 로그
출력은 타이머 종료 후 실행한다. 로봇 필터와 입력 저장도 측정 루프 전에 수행한다.

[기존 gate](../../graph/plan_a_geometry/core.py)는 손목의 전체 depth에서 world XYZ를
생성한 뒤 유효 Peg 점을 고른다. `query_pairs`와 sparse connected components,
local-neighbor 지지, 다른 카메라로의 depth·semantic 모순 검사, 판정 trace와
최종 mask 구성이 함수 호출 안에 포함된다. 두 카메라에 gate를 각각 실행해 합한
시간은 아니다. SAM2·로봇 필터·Graph·시뮬레이터 시간은 포함되지 않는다.

빈 입력도 전체 손목 world-point 생성과 동기화·메타데이터 검증 등을 수행한다.
반면 연결성 함수는 즉시 반환하고 local-neighbor 질의와 타 카메라 투영은
건너뛴다. 따라서 **5.44 ms는 빈 입력 경로의 시간**이다.

## 현재 코드 및 로컬 출력 재현

현재 checkout과 동결 실행 소스의 `wrist_gate`, `components`, depth·좌표 생성,
SurfaceObservation 검증·subset, `consistent_views`, `filtered_masks`,
`synchronized`, `classify_robot`를 AST로 비교해 일치함을 확인했다.
`CameraObservation.validate`는 현재 metadata 검사를 별도 함수로 분리해 전체 AST가
다르다. 분리된 검사 본문은 매개변수 이름을 정규화하면 동결 소스와 정확히 같다.
전체 호출 경로의 모든 파일이 동일하다는 주장은 하지 않는다.

현재 코드의 `wrist_gate`로 저장 입력을 각각 1회 재실행했다. 두 경우 모두 반환된
accepted mask의 SHA-256과 전체 trace dictionary가 원격 결과와 정확히 일치했다.
로컬 환경은 macOS arm64, Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1이다.
이 실행은 기능 재현이며 별도 성능 benchmark가 아니다. 원격의 15회 출력 일치는
runner와 보존된 기록으로 검토했으며 원격 15회를 이번에 다시 실행하지 않았다.

## 해석 및 변경 범위

이 결과는 고정된 서로 다른 두 입력의 gate 단독 시간이다. 전체 장면의 평균,
최악 시간, 전체 파이프라인 지연 또는 GT 정확도를 증명하지 않는다. 구현은 모든
반경 이웃 쌍을 생성하므로 점 밀도에 민감하지만, 단계별 계측이 없어 2.32초 중
연결성 검사의 비중을 특정할 수는 없다. 다른 입력·측정 경계의 SAM2/필터 중앙값과
단순 합산해 전체 지연으로 발표할 근거도 없다.

Production 코드와 기존 보고서·원자료는 변경하지 않았다. 이번 검토는 이 문서와
별도 검증 JSON만 추가했다. 기존 실험의 공통 `ExperimentLogger` 기록은 읽기만
했으며 새 학습·평가 benchmark, 모델/API 호출, 게시를 수행하지 않았다. 로컬 CPU
gate만 기능 실행했고 원격 EPYC, GPU gate, simulator/live rollout는 실행하지 않았다.

## 근거

- [검토 대상 측정 보고서](10i_wrist_gate_cpu_timing_20261104.md)
- [이번 독립 검증 결과](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/review_20261104.json)
- [원시 결과](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/results.json)
- [실행 설정 및 소스 hash](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/config.json)
- [실행 로그](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/measured.log)

# Task10-J — Grounding DINO 프롬프트 탐색 실험

2026-11-02. **평가 가능한 기존 사례에서 `FOLLOWUP_WORTHWHILE`: 후속 검증 가치 있음.**

Peg `red and white stick`, Box `cardboard block` 조합을 탐색 데이터에서 선택한 뒤 확인용 평가에 고정했다.
현재 런타임의 프롬프트는 변경하지 않았다. 모델 가중치·threshold·면적·top-5·joint/null·scan 규칙은 기존 그대로다.

**입력 제한:** 14개 RGB 중 seed98202/t0의 저장된 EXACT actor segmentation을 찾지 못했다. 이 장면은
모든 탐색 프롬프트의 원시 후보만 보존했으며 정량 성공/실패·후보 선택에는 평가 근거로 사용하지 않았다.
13개 이미지의 평가 가능한 25개 물체 관측으로 아래 수치를 계산했다. 실행 상태는 `PARTIAL_INPUT_GT`다.

## 결과

| 구분 | 기준 성공/분모 | 선택 조합 성공/분모 | 선택 IoU 평균: 기준 → 선택 |
|---|---:|---:|---:|
| 탐색 | 8/17 | 13/17 | 0.471 → 0.731 |
| 확인 | 2/8 | 4/8 | 0.222 → 0.388 |

탐색 복구 5건, 확인용 실패 장면 복구 2건이다. 정상 대조의 기존 성공 손상은 0건이다.
분모는 물체 관측 수다. 0355의 인접 7개 프레임은 같은 Parent이므로 독립 표본이 아니다.

| 물체·구간 | 기준 성공/분모 | 선택 성공/분모 |
|---|---:|---:|
| peg / search | 2/9 | 6/9 |
| box / search | 6/8 | 7/8 |
| peg / confirm | 1/4 | 3/4 |
| box / confirm | 1/4 | 1/4 |

## 0355의 실제 초기화 규칙 재생

원래 `detect_initial`에 저장된 t0/5/10/15/20/25/30의 캐시 검출을 공급했다. GT로 박스를 선택하지 않았다.

| 물체 | 기준 승인 시점 / GT IoU | 선택 조합 승인 시점 / GT IoU |
|---|---:|---:|
| peg | t30 / 0.027 | t0 / 0.898 |
| box | t0 / 0.226 | t0 / 0.945 |

기존 Peg는 t30에서 잘못된 영역을 승인했다. 선택 조합은 두 물체 모두 t0에서 올바른 박스를 승인했다.
이 결과는 SAM2 메모리나 이후 추적을 실행한 결과가 아니다. 단일 프레임 결과와 위 scan 재생을 구분한다.

## 프레임별 선택 IoU

✓는 IoU≥0.5다. `GT 없음`과 `<8px`는 주 성공률 분모에서 제외했다.

| 장면 | Peg 기준 → 선택 | Box 기준 → 선택 |
|---|---|---|
| 98202_base_camera_000 (search) | GT 없음 → GT 없음 | GT 없음 → GT 없음 |
| 0355_base_camera_000 (search) | 0.000 → 0.898 ✓ | 0.226 → 0.945 ✓ |
| 0355_base_camera_005 (search) | 0.000 → 0.901 ✓ | 0.940 ✓ → 0.948 ✓ |
| 0355_base_camera_010 (search) | 0.000 → 0.897 ✓ | 0.279 → 0.279 |
| 0355_base_camera_015 (search) | 0.000 → 0.897 ✓ | 0.911 ✓ → 0.913 ✓ |
| 0355_base_camera_020 (search) | 0.000 → 0.074 | 0.955 ✓ → 0.960 ✓ |
| 0355_base_camera_025 (search) | 0.000 → 0.000 | 0.949 ✓ → 0.955 ✓ |
| 0355_base_camera_030 (search) | 0.000 → 0.000 | 0.946 ✓ → 0.953 ✓ |
| 0645_base_camera_000 (search) | 0.929 ✓ → 0.930 ✓ | 0.917 ✓ → 0.925 ✓ |
| 0645_hand_camera_006 (search) | 0.955 ✓ → 0.955 ✓ | <8px → <8px |
| 0362_hand_camera_000 (confirm) | 0.000 → 0.845 ✓ | 0.000 → 0.000 |
| 0666_hand_camera_000 (confirm) | 0.000 → 0.000 | 0.112 → 0.000 |
| 0810_hand_camera_000 (confirm) | 0.000 → 0.584 ✓ | 0.000 → 0.000 |
| 98200_base_camera_000 (confirm) | 0.737 ✓ → 0.738 ✓ | 0.926 ✓ → 0.935 ✓ |

## 프롬프트별 비교

각 물체의 표현만 바꾸고 상대 물체는 기준 표현을 사용한 탐색 결과다. 최종 조합의 상호작용은 별도 144개 조합표에 있다.

| 물체 | 라운드 | 프롬프트 | 올바른 후보 존재/분모 | 최종 선택 성공/분모 |
|---|---:|---|---:|---:|
| peg | 1 | `red and white block` | 9/9 | 2/9 |
| peg | 1 | `peg` | 6/9 | 0/9 |
| peg | 1 | `white block` | 9/9 | 1/9 |
| peg | 1 | `red and white peg` | 8/9 | 1/9 |
| peg | 1 | `white rectangular block` | 7/9 | 1/9 |
| peg | 1 | `white stick` | 6/9 | 2/9 |
| peg | 1 | `white peg with red tip` | 9/9 | 1/9 |
| peg | 1 | `red and white rectangular block` | 8/9 | 2/9 |
| peg | 2 | `white rectangular peg` | 2/9 | 1/9 |
| peg | 2 | `red tipped white peg` | 9/9 | 2/9 |
| peg | 2 | `red and white stick` | 9/9 | 6/9 |
| peg | 2 | `white rectangular block with a red end` | 9/9 | 3/9 |
| box | 1 | `cardboard box` | 8/8 | 6/8 |
| box | 1 | `box` | 8/8 | 4/8 |
| box | 1 | `brown box` | 8/8 | 8/8 |
| box | 1 | `brown cardboard box` | 8/8 | 8/8 |
| box | 1 | `rectangular box` | 8/8 | 5/8 |
| box | 1 | `brown block` | 8/8 | 7/8 |
| box | 1 | `box with hole` | 8/8 | 2/8 |
| box | 1 | `brown block with a hole` | 8/8 | 6/8 |
| box | 2 | `brown rectangular box` | 8/8 | 7/8 |
| box | 2 | `cardboard block` | 8/8 | 7/8 |
| box | 2 | `wooden box` | 8/8 | 8/8 |
| box | 2 | `brown box with a hole` | 8/8 | 7/8 |

## 해석과 제한

- 기준선에서도 평가 가능한 모든 물체 관측에 IoU≥0.5 후보가 있었다(탐색17/17, 확인8/8). 이번 이득은 주로 후보 순위·점수 및 기존 선택 규칙을 통과하는 정도의 변화에서 나타났다.
- 확인용 Wrist Peg 0362는 IoU0→0.845, 0810은0→0.584로 복구됐다. 0666 Peg는 좋은 후보가 있어도 면적 제한에서 계속 탈락했다. Wrist Box 3건은 모두 실패가 남았다.
- 0355의 t20/25/30에서 현재 프레임만으로 새 초기화를 시도하면 Peg 실패가 남는다. t0 승인 개선을 이후 프레임의 독립 검출 개선이나 추적 성공으로 확대하지 않는다. 0666 Box의 선택 IoU는0.112→0으로 낮아졌으며 두 조건 모두 검출 성공 기준에는 미달한다.
- 1차 최선은 Peg `white stick` / Box `brown box`의 10/17이었다. 사전 고정한 보완 4개씩을 추가한 2차에서 최종 조합이 선택됐다. 2차 이후 추가 탐색은 하지 않았다.
- 확인용은 역사적으로 알려진 실패/정상 사례다. 프롬프트 선택에는 사용하지 않았지만 독립 Final이나 모집단 일반화 검증으로 해석하지 않는다.
- bbox IoU 기반 다른 물체 혼동 지표는 두 조건 모두 0이었다. 이는 실제 의미 혼동이 없다는 뜻이 아니다. 0355의 잘못된 Peg 선택처럼 다른 물체와 bbox IoU도 0.5 미만인 혼동은 이 지표에서 빠질 수 있다.
- 가시 GT bbox는 SAM2에 유용한 프롬프트인지 완전히 설명하지 않는다. 배경·로봇을 포함하는 박스, 부분 가림과 이후 추적 안정성은 후속 SAM2 평가가 필요하다.
- 실패 프레임별 다른 표현을 고르거나 여러 프롬프트의 결과를 합치는 앙상블은 평가하지 않았다. 실제 운영 시 호출 수가 늘어나는 방법은 아니다.
- seed98202의 GT 부재로 해당 사례의 프롬프트 복구 여부는 미확정이다. 정상 대조도 소수이므로 현재 결과만으로 기본 프롬프트를 교체하지 않는다.

## 실행·검증

- 기존 Vast.ai 58752018의 RTX3090 CUDA에서 259회 forward를 실행했다. 모델 로드는 5.66초였다.
- 호출 구분: `{"baseline": 20, "confirm": 16, "repeat": 1, "round1": 140, "round2": 80, "warmup": 2}`. 최대300회 예산을 지켰다.
- 전처리+forward+후처리 시간 중앙값 215.3ms, p95 308.8ms. 다른 GPU 작업과 공유한 환경이므로 전용 latency benchmark가 아니다.
- Python3.12 / Torch2.14.1+cu130 / Transformers5.19.0. 기존 설치·모델을 재사용했고 다운로드·패키지·드라이버 변경은 없었다.
- 기준선 score fingerprint 3개가 오차0으로 일치했다. 동일 입력 반복 안정성과 0355의 잘못된 t30 승인도 재현했다.
- CPU 테스트5개: null 탈락/물체 혼동, 면적/작은 가시성, GT 변경에도 consumer 선택 불변, 정상 대조 보존 우선순위, 연속 scan 승인. macOS와 원격 Linux에서 모두 PASS.
- 결과 CSV에서 분모·성공 수를 독립 재계산했고, 확인 forward 이전 선택 동결, 호출 예산, RGB hash/GT 픽셀 수를 readback 검증했다.
- 원본 입력과 모델 파일은 실행 전후 SHA가 동일하다. 실행 소스와 모델 전체 파일별 SHA는 protocol에 기록했다.
- 공용 `ExperimentLogger` 사용. Run ID `b76bc2359d1f4702a2bd75269eb63c7a`. 기존 프로젝트·그룹 규칙을 유지하며 W&B disabled/local 기록으로 실행했고 외부 게시하지 않았다.
- SAM2·DINOv2·Simulator·GPU PhysX·학습·macOS MPS 추론은 이번 실험에서 실행하지 않았다. 중지된 기존 연구 runner도 재개하지 않았다.

## 재현 및 산출물

- [실행 방법 및 코드](../../visual/experiments/dino_prompt/README.md)
- [전체 요약](../../outputs/task10j/dino_prompt_20261102_v1/run/summary.json) · [입력 manifest](../../outputs/task10j/dino_prompt_20261102_v1/inputs/manifest.json) · [모델·소스·프롬프트 동결](../../outputs/task10j/dino_prompt_20261102_v1/run/protocol.json)
- [259개 원시 추론 출력](../../outputs/task10j/dino_prompt_20261102_v1/run/calls.json) · [프롬프트별 지표](../../outputs/task10j/dino_prompt_20261102_v1/run/prompt_details.csv)
- [1차64개 조합](../../outputs/task10j/dino_prompt_20261102_v1/run/round1_combinations.csv) · [2차144개 조합](../../outputs/task10j/dino_prompt_20261102_v1/run/round2_combinations.csv) · [보완 라운드 근거](../../outputs/task10j/dino_prompt_20261102_v1/run/round2_rationale.json)
- [프레임별 결과](../../outputs/task10j/dino_prompt_20261102_v1/run/baseline_selected.csv) · [개선·악화 목록](../../outputs/task10j/dino_prompt_20261102_v1/run/transitions.csv) · [확인 전 선택 동결](../../outputs/task10j/dino_prompt_20261102_v1/run/selection_frozen.json)
- [기준선 재현](../../outputs/task10j/dino_prompt_20261102_v1/run/baseline_verification.json) · [원본 보존 검증](../../outputs/task10j/dino_prompt_20261102_v1/run/verification.json) · [독립 readback/actor 구성](../../outputs/task10j/dino_prompt_20261102_v1/run/readback_verification.json)
- [실행 로그](../../outputs/task10j/dino_prompt_20261102_v1/execution.log) · [macOS CPU 테스트](../../outputs/task10j/dino_prompt_20261102_v1/cpu_tests_local.log) · [Linux CPU 테스트](../../outputs/task10j/dino_prompt_20261102_v1/cpu_tests_remote.log)
- [원격 산출물50개 전송 SHA 검증](../../outputs/task10j/dino_prompt_20261102_v1/transfer_verification.json) · [로컬 Git·소스 식별](../../outputs/task10j/dino_prompt_20261102_v1/local_source_identity.json)

![GT / 기준선 / 선택 조합](../../outputs/task10j/dino_prompt_20261102_v1/run/comparison.png)

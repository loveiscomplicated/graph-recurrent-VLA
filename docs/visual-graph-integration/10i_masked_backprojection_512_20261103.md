# Task 10-I: 512 전체 이미지 역투영과 마스크 역투영의 실제 비교

2026-11-03 KST. **측정·수정·재측정 완료. 이번 마스크 전용 수정본은 더 느렸으므로 기본 경로에 채택하지 않았다.**

## 결과

같은 native512 Depth와 실제 Mask를 사용했다. Mac **Apple M4 Pro CPU**, Python3.11/NumPy 환경에서 카메라1대의 연산을 측정했다. 초기 입력 로드와 depth mm→m 변환은 timer 밖이다. 수정 전 원본을 먼저 측정·보존한 뒤, 수정 후 두 구현을 번갈아 실행해 비교했다.

| 측정 구간 | 기존 전체 이미지 방식 중앙값 | 마스크 영역 방식 중앙값 | 수정본 / 기존 |
| --- | ---: | ---: | ---: |
| 기하 배열 준비: 유효성 검사·mask 복사·XYZ 할당과 역투영 | **0.735ms** | **2.401ms** | **3.27배** |
| 위 준비 + 마스크의 유효 물체 점 추출 | **1.203ms** | **2.873ms** | **2.39배** |

사용자가 지목한 ‘전체 Depth를 XYZ로 만든 뒤, 유효한 마스크 영역의 점을 사용’까지를 포함한 두 번째 행을 주 비교로 본다. 이 경계에서 수정본은 약 **1.670ms, 138.9% 느렸다**. 순수 곱셈·나눗셈 커널만의 시간이나 전체 Graph 시간이 아니다.

각 구간·구현별 워밍업50회 후 **1,000회** 측정한 중앙값이다. 같은 반복 안에서 full→mask / mask→full 순서를 교대했다. p95는 기하 준비가 **0.804 / 2.561ms**, 물체 점 추출까지가 **1.302 / 3.054ms**였다. 입력1개를 반복한 local microbenchmark이므로 새 학습·평가 분포의 평균, GPU 서버 수치 또는 전체 센서→action 지연으로 일반화하지 않는다.

별도로 **코드를 수정하기 전** 실행한 기존 방식300회 중앙값은 **1.184ms**, p95는1.291ms였다. 이 사전 기록과 사후 paired 비교는 서로 다른 실행이므로, 주 속도비에는 사후 실행 내의 같은 조건 비교값을 사용했다.

## 사용한 입력

- 기존에 로컬에 보존된 `10i_s1_0844_t080`, `hand_camera`, native512×512 Depth와 actual SAM2 mask repeat0.
- 원래512 Depth pixel은262,144개, Peg mask는**61,056개(23.29%)**, Box mask는0개다. 이 입력의 mask pixel은 모두 유효 깊이였다.
- 센서 파일의 원래 signed-mm depth를 기존 규칙대로 양수만 유효로 보아 float32 metres로 변환했다. Mask·Depth·intrinsic은 두 구현에서 동일하다.
- GT·로봇 필터를 실행하거나 읽어 출력 점을 보정하지 않았다. 이 local 입력은 앞으로 선별할 Peg 완전 가시 데이터셋으로 인증한 표본이 아니다. 이번 비교는 고정 입력의 연산 변경 검증이다.
- 측정 대상은 camera-frame 연산뿐이다. Benchmark frame의 identity transform/TCP placeholder는 이 연산에서 읽히지 않으며 실제 world/EE 추정의 증거로 사용하지 않는다.

원본 입력 SHA와 변환한 입력은 [수정 전 기록](../../outputs/task10i/masked_backprojection_512_20261103_v1/baseline_before_edit.json), [동일 입력 NPZ](../../outputs/task10i/masked_backprojection_512_20261103_v1/input.npz)에 보존했다.

## 무엇을 수정했나

[`visual.runtime.pca.build_geometry`](../../visual/runtime/pca.py)에 **`mask_only=True` 선택형 경로**를 추가했다.

1. 기존 입력 검증, Depth 유효성, Peg/Box mask와 overlap 정의를 유지한다.
2. 두 물체 raw mask의 합집합 중 유효 깊이 pixel만 골라 원래 pixel-centre 식으로 X/Y를 계산한다. 해당 pixel의 XYZ는 원래 float32 값과 일치한다.
3. 후속 PCA·pixel 특징 코드가 같은 raster 배치를 사용하므로 **512×512×3 XYZ 배열 크기는 유지**한다. Mask 밖의 X/Y는NaN이다.
4. 물체와 배경 사이의 depth jump 계산을 보존하기 위해 **배경 Z는 원래 Depth로 유지**한다. Z를 읽어 저장하는 작업과 배경 X/Y를 역투영하는 작업은 다르다.
5. 기존 Graph 실험의 [`dense.extract`](../../graph/experiments/graph_latency_10i_20261103_v2/dense.py)에도 같은 옵션을 전달할 수 있게 했다. **기본값은 두 API 모두False**다.

기존 full 경로와 마스크 경로의 object point 수·source pixel·좌표·의미는 같고, 사용하지 않는 배경 X/Y의 계산 여부가 다르다. 배경의 전체XYZ를 소비하는 코드에는 mask-only 결과를 그대로 쓰면 안 된다. 반환 구조의 차이와 사용 예시는 [sampling 문서](../../visual/runtime/sampling.md)에 기록했다.

로봇 필터링, 손목 gate, PCA 알고리즘, sampling, point budget, 중복 제거, 최종 Graph 엣지 또는 학습은 수정하지 않았다. 이번 옵션을 기존 중단 실험에 배포·적용하지 않았다.

## 왜 점 계산이 줄어도 빨라지지 않았나

역투영 대상은262,144개에서61,056개로 줄었지만, 현재 반환 형태에서는 전체 XYZ 배열의 할당·초기화가 남는다. 수정본에는 mask index 추출, 선택 Depth gathering, 선택점 임시 배열 생성과 raster 위치로의 scatter도 들어간다. 기존 방식은 전체 영상에 NumPy broadcasting을 적용한다.

이번 결과는 **이 구현과 입력에서 계산 대상 감소가 실행 시간 감소로 이어지지 않았음**을 보여준다. 위 항목별 비용을 독립적으로 분해해 어느 하나가 몇ms를 차지하는지 입증한 것은 아니다. ‘마스크만 역투영’이라는 모든 구현이 반드시 느리거나, compact point 출력/GPU 구현도 느리다는 결론은 아니다.

성능이 악화된 결과를 보존하고 기본 경로는 기존 방식으로 유지했다. 다른 출력 구조나 GPU 구현으로 범위를 넓혀 추가 최적화하지 않았다.

## 정확도·회귀 검증

실제512 입력에서 보존한 기존 구현과 수정본을 비교했다.

- 물체 XYZ, source index, 유효 깊이 및 배경 Z가 **정확히 일치**했다.
- CPU PCA5가 만든 모든 기하 Feature·validity가 **정확히 일치**했다.
- 인접 pair evidence의 모든 field와 S3 후보의 좌표·source·metadata가 **정확히 일치**했다.
- 입력 배열의 실행 전후 hash가 일치했다. 정답 보정·좌표 평균·downsampling은 하지 않았다.

새 계약4개는 두 물체·overlap, 무효 깊이, 빈/전체/tiny mask, 비연속 배열, pixel-centre와 float32/64 입력, 배경 깊이 경계 및 Feature/후보 동일성을 검사했다. 기존 PCA 회귀와 합쳐 **17개 중14 PASS, CUDA3 SKIP, 실패0**이었다.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/task10i-masked-backprojection-mpl \
  .venv/bin/python -m unittest \
  visual.runtime.test.test_masked_backprojection \
  visual.runtime.test.test_pca_optimized -v
```

**CUDA/MPS, 수정된 GPU Feature 경로, 전체 Graph tensor 비교, simulator/renderer와 정책 rollout은 실행하지 않았다.** 기존 CUDA tests의 skip을 새 GPU 경로의 검증으로 세지 않는다. 이번 결과는 NumPy CPU 단계의 수치·시간 비교다.

## 재현·산출물·실험 식별자

- [측정 코드·재현 안내](../../graph/experiments/masked_backprojection_512_20261103_v1/README.md)
- [1,000회씩의 원시 측정 및 정확성 결과](../../outputs/task10i/masked_backprojection_512_20261103_v1/paired_v1/results.json)
- [호스트·입력·source identity와 측정 경계](../../outputs/task10i/masked_backprojection_512_20261103_v1/paired_v1/config.json)
- [수정 전 구현 snapshot](../../outputs/task10i/masked_backprojection_512_20261103_v1/source_before/visual/runtime/pca.py)
- [전용 계약 테스트](../../visual/runtime/test/test_masked_backprojection.py)

공통 `ExperimentLogger`를 재사용했다. 수정 전 기록 UUID는 **b2c98887008340a1a3216ae260621f5d**, paired 비교 UUID는 **53586cf65f564764a890dd09fafab787**이다. 같은 원본 구현을 별도 snapshot에서 실제 import해 비교했으며 이전 시간 수치를 끌어와 새 수정본과 나눈 것이 아니다.

W&B는disabled, artifact publisher 호출0회다. 공통 Logger/Publisher identity 규칙은 변경하지 않았다. 코드·문서 수정과 별도 ignored 출력만 만들었으며 기존 다른 dirty 작업은 유지했다. Git commit, 외부 게시, 패키지 설치나 원격 접속은 수행하지 않았다.

# Task10-I Stage4A — 58576054 Mask/Graph Forward Equivalence

**최종 후속 상태:** 이후 Stage4A 전체 연구를 재개해 독립 검증·인계를 완료했다(판정
`NO_GO`, Production 변경 없음). [전체 최종 보고서](10i_stage4a_final_report.md).
아래의 “재개하지 않았다/미실행”은 이 개별 검사 시점의 기록이다.

2026-11-01 KST. 판정: **PASS — 기존 재현 검증 Episode의 전체101 action frame에서
Mask와 Graph가 정확히 재현됐다.** 원래 F1 캐시와 이전 서버의 Stage4A 출력 양쪽에
대조했으며, 수치 tolerance를 완화하지 않았다. 이번 실행은 새 호스트의 재현 검증이다.
남은4개 Sequence 처리나 이후 연구 단계는 시작하지 않았다.

## 대상과 실행 identity

| 항목 | 실제 실행 |
|---|---|
| Host / GPU | Vast58576054 / RTX3090 |
| Python | `/workspace/venvs/task10i-stage4a-cu130/bin/python` |
| Torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| 새 Source root | `/workspace/task10i_stage4a_forward_equivalence_20261101_v1/source` |
| 새 Output root | `/workspace/task10i_stage4a_forward_equivalence_20261101_v1/run` |
| 새 ExperimentLogger UUID | `813d4b24cd414cbfb327e25145473372` |
| 대상 Episode / Parent | 2000003 / existing_30080 |
| Partition | 사전 등록된 Development의 Train Parent |
| 비교 범위 | t=0–100, 두 Camera × Peg/Box, Graph5개 Tensor |
| 실행 방식 | `task10i-stage4a-equivalence` tmux에서 계약 시험 후 Python verifier 자동 실행 |
| 최종 상태 | `FORWARD_EQUIVALENCE_PASS`, Logger `finished` |

대상은 기존 `baseline.py`가 선택하던 **동일한 원래 재현 검증 fixture**다.
새 결과를 본 뒤 잘 맞는 Episode를 선택하지 않았다. Shared Validation, Internal Test,
정책 Final/Test와 Holdout 영상은 사용하지 않았다.

원래 F1 캐시는101개의 action frame Graph를 갖고 있다. Stage4A 캐시에는 terminal을
포함한102개 관측이 있지만, 이번 비교는 두 캐시가 공통으로 지원하는101개 전체다.
**terminal t=101은 이번에 검증하지 않았다.** 이 단일 fixture의 통과를 전체64개
Sequence나4,162개 관측 전체의 Forward Equivalence로 확대 해석하지 않는다.

## 입력·Source·설정 보존

기존 v5 Source를 새 독립 root로 복사하고 검증 helper만 추가했다. 동결된672개 파일은
모두 기존 SHA와 일치했고, 실행 후에도 동일함을 다시 확인했다. Production 코드를
수정하지 않았다. 기존 raw RGB-D, calibration, robot 관측, geometry 자산을 읽고
기존 Source의 FK·Mask·Geometry 경로를 그대로 실행했다.

- Raw Episode SHA256:
  `146cbf7d7a72933f9bfdf1f3c396ae3c19c8b350fa84a9e7379e1580856c602a`
- 원래 F1 `graphs.npz` SHA256:
  `a655de1f3916e2b3cb2af4a33bae7b8090eac102d0d4bfc90eb8578570eef2d1`
- DINO 가중치·Tokenizer·설정과 SAM2 Checkpoint SHA도 기존 model identity와 일치했다.
- Torch 설정: seed42, CPU thread1, `deterministic=False`, `matmul_tf32=False`,
  `cudnn_tf32=True`. 학습용 seed helper를 사용하지 않았다.

기존 `baseline.run` 자체는 수정하지 않았다. `CurrentGeometry.build`가 결과를 반환한
**이후** 관찰용 wrapper가 actual Mask/Graph를 저장하고 이전 Stage4A 캐시와 대조한다.
Observer는 RNG를 호출하거나 모델·Geometry의 반환값을 변경하지 않는다. 정답은
결과 비교에만 사용하며 Detector·Tracker·Geometry 입력에 넣지 않았다. Episode 시작의
별도 Camera state reset과 인과적 frame 순서를 기존 adapter가 그대로 처리했다.

[입력 identity](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/input_identity.json) ·
[고정 Source SHA](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/source_freeze.json).

## 정량 결과

| 비교 | 원래 F1 캐시 | 이전 Stage4A 캐시 |
|---|---:|---:|
| Mask가 다른 frame | 0 / 101 | 0 / 101 |
| Main Peg Pixel 차이 | 0 | 0 |
| Main Box Pixel 차이 | 0 | 0 |
| Wrist Peg Pixel 차이 | 0 | 0 |
| Wrist Box Pixel 차이 | 0 | 0 |
| Graph가 다른 frame | 0 / 101 | 0 / 101 |
| Graph Tensor 비교 | 505 / 505 동일 | 505 / 505 동일 |
| Wrist Gate record | 기존 baseline 경로 사용 | 101 / 101 동일 |

Camera/Object/Frame Mask404개, 즉 **6,619,136개 Pixel을 각 참조와 대조**했다.
Graph는 `node_input`, `node_type`, `edge_index`, `edge_attr`, `edge_type`의 shape,
dtype와 실제 배열 bytes를 비교했다. 부동소수 Tensor의 최대 절대 차이도0이었다.
단순 approximate equality나 Mask 존재 여부만 비교한 결과가 아니다.

Graph element 비교 수는 참조별로 `node_input457,800`, `node_type21,800`,
`edge_index71,308`, `edge_attr142,616`, `edge_type35,654`다.

Forward가 끝난 뒤 별도 NumPy readback 검사에서 저장된101개 actual NPZ를 모두
다시 열었다. F1의 pointer 배열을 직접 사용해 Graph를 자르고, 공통 `graph_at` helper를
사용하지 않고도 양쪽 참조와 bytewise equality를 확인했다. Mask1bit와 각 Graph
Tensor의 한 element를 인위적으로 바꾸는 negative control도 모두 불일치로 검출됐다.
실제 저장 파일 SHA와 로컬 복사본 SHA도 대조했다.

[새 호스트 비교 전체 결과](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/new_host_forward_equivalence.json) ·
[기존 baseline 함수의 F1 비교 결과](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/baseline_forward_equivalence.json) ·
[독립 readback 검사](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/independent_readback.json).

## 계약 시험과 실행 범위

새 cu130 venv에서 Stage4A, online vision, causality, PlanA geometry, PlanA BC의
기존 unittest40개를 실행했다. **39 PASS, 1 SKIP, 실패0**이다. Skip은 source-only
배포본에 원래 preregistration 파일이 없어 수행하지 못한11,487-step 학습 schedule
시험이며, 이번 Forward 비교를 건너뛴 것이 아니다.
[시험 로그](../../outputs/task10i/stage4a/forward_equivalence_58576054/contracts.log).

DINO와 SAM2는 이번에 실제 CUDA forward를 실행했고 Geometry/Graph도 재계산했다.
SAM2 `_C` extension 부재 경고는 기존과 같은 fallback 구성에서 발생했다. 이 구성에서도
검증 범위의 Mask/Graph 출력이 일치함을 실제로 확인했다. CPU PhysX simulator step,
SAPIEN Renderer와 재렌더링 GT 재현성은 이번 실행 대상이 아니었다.

관찰용 저장과 비교를 포함한 검증 작업 wall time은 약129초였다. I/O·Observer 비용과
모델 초기화가 포함되므로 Perception latency나20Hz 실행 가능성 수치로 해석하지 않는다.

기존 `utils.experiment_logger.ExperimentLogger`를 재사용하고 W&B를 disabled로 실행했다.
새 Logger ID를 이전 실행 ID로 덮어쓰지 않았다. HF/모델 조회는 offline으로 제한했고
외부 게시를 수행하지 않았다. 기존 Checkpoint, Stage1–3 데이터, split, main-visible
판정과 기존 Stage4A 완료 Receipt를 수정하지 않았다.

## 후속 작업의 경계

새 호스트에서 이 fixture의 Mask/Graph 재현성을 막던 게이트는 통과했다. 이는
Perception 정확도 개선을 뜻하지 않는다. 기존 누락이나 오염도 그대로 재현될 수 있다.
Stage4A의 Baseline 완료 상태는 여전히60/64이며, 미완료4개·136관측과 이후 감사·개선
후보 평가·독립 검증은 남아 있다. 이번 verifier는 그 작업을 자동으로 시작하지 않는다.

[실제 실행 command](../../outputs/task10i/stage4a/forward_equivalence_58576054/launch.sh) ·
[forward 로그](../../outputs/task10i/stage4a/forward_equivalence_58576054/forward.log) ·
[실행 identity](../../outputs/task10i/stage4a/forward_equivalence_58576054/run/run_identity.json) ·
[산출물 SHA256](../../outputs/task10i/stage4a/forward_equivalence_58576054/artifact_sha256.json).

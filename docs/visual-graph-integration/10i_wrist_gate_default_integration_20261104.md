# Task10-I: live·frozen_inference의 GPU 손목 gate 기본 연결

2026-11-04 KST. **현재 checkout의 live와 Task10-I `frozen_inference`가 검증된
CUDA 손목 gate를 기본 사용하도록 연결했다. CUDA가 없을 때만 기존 CPU gate를
선택하며, GPU 의존성·실행 오류는 CPU로 재시도하지 않고 전파한다.**

기존 29,775점 입력은 새 기본 선택 경로와 동일 프레임 memo에서도 통과했고,
accepted mask와 전체 trace가 원래 CPU reference와 정확히 일치했다.

## 기본 선택 및 호출 경로

| 호출 위치 | CUDA가 있을 때 | CUDA가 없을 때 |
| --- | --- | --- |
| live `CurrentGeometry.build` | 기존 `prepare_tensor_gate` → `wrist_gate_tensor`; 마스크·점은 CUDA | 기본 geometry가 CPU perception/filter/gate 선택 |
| Task10-I `frozen_inference`의 최종 Graph | 같은 `CurrentGeometry` CUDA gate | 같은 geometry CPU 선택 정책 |
| Task10-I 보정 후보·`system.prepare`·동일 프레임 memo | 같은 `wrist_gate_tensor`를 호출하고 accepted mask만 명시적으로 NumPy로 반환 | 기존 `core.wrist_gate` |

판단 기준은 `torch.cuda.is_available()`이다. CUDA가 사용 가능한 경우 Triton
import 오류, OOM 등 RuntimeError를 잡아 CPU로 바꾸는 fallback은 없다.
`CurrentGeometry(..., perception_backend='cpu')`로 Graph/로봇 필터를 진단하는
경우에도 손목 gate는 CUDA가 있으면 GPU를 사용한다.

Standalone `PlanA`와 직접 호출하는 `core.wrist_gate`는 독립 CPU reference를
보존한다. `CurrentGeometry`는 자신의 `PlanA`에 `wrist_gate_backend='auto'`를
설정해, 두 요청 진입점과 보정 후보 경로에만 자동 정책을 연결한다. 기존 gate의
반경·threshold·마스크·점 수·trace 필드는 바꾸지 않았다.

## 구현 및 호환 경계

- [자동 gate 어댑터](../../graph/persistent_geometry/wrist_gate_dispatch.py)는
  기존 GPU gate를 재사용한다. NumPy 준비/보정 경로에서는 원래 host 인터페이스를
  유지하고, native tensor 경로에서는 accepted mask를 CUDA에 유지한다.
- [CurrentGeometry](../../graph/plan_a_bc/geometry.py)는 CUDA가 없으면 기본
  `triton_fused` geometry를 CPU로 선택한다. GPU가 있을 때 발생한 오류에는 이
  분기가 실행되지 않는다. `last_wrist_gate_backend` 및
  `graph.packet['wrist_gate_backend']`에 실제 실행 backend를 기록한다.
- [PlanA.prepare](../../graph/plan_a_geometry/core.py)는 prepared 결과에 실제
  `wrist_gate_backend`를 trace와 별도로 기록한다. 기존 raw-pixel 분모와 필터
  진단·C0/C1/C2/C3 마스크 구성은 유지한다.
- [frozen_inference](../../graph/task10i_stage4a2/frozen_inference.py)는
  `from_h5(..., tensor_images=False)`로 기존 DINO/SAM2/보정 코드가 요구하는
  host RGB-D를 유지한다. 최종 geometry는 필요한 입력을 CUDA에 올린다.
  이전의 GPU Graph `.numpy()` 직접 호출 및 TensorRecords 반복 대신,
  inference 뒤 `geometry_artifacts`에서 명시적으로 보관용 변환을 수행한다.
  에피소드 종료 시 geometry worker도 닫는다.
- [동일 프레임 memo](../../graph/task10i_stage4a/perframe_cache.py)는 복제한
  `prepare`의 keyword defaults를 보존하고 `clean_masks`를 전달한다. Raw 준비와
  precleaned 준비가 섞이지 않도록 clean mask와 gate 정책을 key에 포함한다.
- Live frame trace, cache identity/receipt, frozen source config·timing·Graph
  fingerprint·receipt가 gate backend를 기록한다. 기존 cache identity와 비교 시
  gate 정책/장치 필드가 없는 cache는 재사용하지 않는다.

## 검증 결과

| 실행 | 결과 |
| --- | --- |
| 로컬 Mac CPU 회귀 검사 | 69개 중 **66 PASS, CUDA 관련 3 skip**, 실패 없음 |
| 원격 RTX 3090 회귀 검사 | 71개 중 **70 PASS, 기존 사전등록 파일 부재 1 skip**, 실패 없음 |
| live와 같은 결정적 연산 설정의 실제 CUDA 경로 | **1 PASS**, skip 없음 |

원격은 Vast 58752018의 기존 Python 3.12.15 / Torch 2.14.1+cu130 환경을 사용했다.
로컬은 CUDA가 없는 실제 Mac CPU에서 기본 geometry build와 CPU gate fallback을
실행했다. 원격에서도 CUDA 가용 여부를 False로 바꾼 검사를 포함해 fallback과
GPU 모듈을 import하지 않는 경로를 확인했다.

주요 검증 내용:

1. **GPU가 있을 때 CPU gate 호출 금지:** CPU gate를 호출하면 실패시키고,
   live용 CUDA RGB-D/마스크 및 frozen용 host RGB-D/마스크로 기본 geometry를
   실제 실행했다. 양쪽 모두 `torch_cuda`였고 Graph tensor와 gate 결과가 같았다.
2. **보정 및 memo:** NumPy `system.prepare`와 memo를 같은 정책으로 실행했다.
   Raw 분모, 전체 gate trace 및 precleaned 진단 인터페이스가 유지됐다.
3. **실제 512 고정 입력:** 29,775점 NPZ를 이전 파일 hash로 확인하고, 새 기본
   경로와 memo 반복에서 accepted mask hash 및 전체 trace를 CPU reference와
   정확 비교했다. 채택 픽셀은 29,269개다.
4. **오류 처리:** GPU 실행 RuntimeError와 의존성 ImportError를 발생시킨 뒤
   CPU gate가 호출되지 않고 원래 오류가 전파됨을 확인했다.
5. **live/cache 호출부:** 기존 mask producer가 build 내부에서 호출되고 CUDA
   마스크를 요구하는 동작, 실제 frame trace/receipt 저장을 mock 모델·환경으로
   검사했다. 별도로 실제 GPU geometry/Graph와 frozen 보관 변환을 실행했다.
6. **live 결정성 설정:** `torch.use_deterministic_algorithms(True)`와
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`에서도 실제 GPU 경로 검사가 통과했다.
7. 기존 PlanA, 로봇 준비, Task10-I 보정 격리, Graph/cache 및 policy 계약 검사를
   함께 실행했고 `git diff --check`도 통과했다.

처음 원격 검사에는 source bundle에서 `tools`/`recurrent` 지원 모듈이 빠져
import 오류 4건이 있었다. 구현 변경 없이 해당 소스 22개를 추가한 뒤 최종 검사가
통과했다. 초기 실패 로그도 보존했다. 최종 소스 1,090개 파일의 hash를 검증했다.

## 적용 범위 및 제한

변경은 현재 저장소 진입점 코드와 문서에 반영했다. 원격 실행은 별도의 새 소스
snapshot에서 검증했으며, 기존 동결 실험 source/output이나 실행 중인 작업을
덮어쓰거나 재개하지 않았다. 기존 source/config 동결 검사가 유지되므로 실제
새 inference 실행에는 새 출력 root와 변경된 소스를 사용해야 한다.

**CPU fallback은 손목 gate와 기본 geometry의 정책이다.** 원래 live renderer,
SAM2/DINO/DINOv2 및 모델의 장치 요구사항을 모두 CPU 지원으로 바꾼 작업은 아니다.
이번 검증에서 실제 simulator/renderer, 실제 SAM2/DINO/API, 전체 live rollout나
`frozen_inference` 전체 데이터셋 실행은 하지 않았다. 입력·호출부 fixture와 실제
CUDA geometry/Graph, 저장된 비어 있지 않은 gate 입력으로 통합을 검증했다.

이번 작업은 성능 재측정이 아니다. 앞선 [GPU gate 5.79 ms 비교](10i_wrist_gate_cuda_first_step_20261104.md)를
새 live 전체 지연으로 해석하지 않는다. 새 NumPy 보정 경로에는 gate 입력 업로드와
accepted mask readback이 포함될 수 있다.

공통 `ExperimentLogger`를 사용하는 두 기존 진입점은 그대로 재사용하고 source
config에 gate 정책을 추가했다. Logger ID/project/group이나 publisher는 변경하지
않았다. 새 학습·게시·패키지 설치·모델 호출은 없었다. 기존 무관한 작업 중 변경도
보존했다. SAPIEN import의 Vulkan ICD 경고는 있었지만 simulator는 만들지 않았다.

## 코드와 검증 자료

- [기본 선택·fallback·실제 GPU 통합 테스트](../../graph/plan_a_bc/test_wrist_gate_default.py)
- [live/cache 호출부 회귀 테스트](../../graph/plan_a_bc/test_robot_overlap.py)
- [로컬 테스트 로그](../../outputs/task10i/wrist_gate_default_20261104_v1/tests_local.log)
- [최종 GPU 테스트 로그](../../outputs/task10i/wrist_gate_default_20261104_v1/tests_gpu_final.log)
- [결정적 GPU 경로 검사](../../outputs/task10i/wrist_gate_default_20261104_v1/deterministic_result.json)
- [결정적 GPU 경로 로그](../../outputs/task10i/wrist_gate_default_20261104_v1/tests_deterministic.log)
- [초기 원격 import 실패 기록](../../outputs/task10i/wrist_gate_default_20261104_v1/tests_gpu_attempt1.log)
- [실행 소스 identity](../../outputs/task10i/wrist_gate_default_20261104_v1/source_identity.json)
- [추가 지원 소스 identity](../../outputs/task10i/wrist_gate_default_20261104_v1/support_identity.json)
- [소스·결과 검증](../../outputs/task10i/wrist_gate_default_20261104_v1/verification.json)
- [종료 상태 점검](../../outputs/task10i/wrist_gate_default_20261104_v1/postcheck.txt)

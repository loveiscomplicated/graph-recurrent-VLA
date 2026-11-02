# Task10-I Stage4A — 58576054 독립 cu130 환경 복원

**최종 후속 상태:** 이후 Stage4A 전체 연구를 재개해 독립 검증·인계를 완료했다(판정
`NO_GO`, Production 변경 없음). [전체 최종 보고서](10i_stage4a_final_report.md).
아래의 “재개하지 않았다/미실행”은 이 개별 검사 시점의 기록이다.

후속 업데이트: 아래는 환경 복원 당시의 기록이다. 이후 사용자 요청으로 기존
Development fixture101개 action frame의 실제 Mask/Graph Forward Equivalence를
확인했고 PASS했다. [Forward Equivalence 전체 보고서](10i_stage4a_forward_equivalence_58576054.md).
남은4개 Sequence나 이후 연구 runner는 재개하지 않았다.

2026-11-01 KST. **의존성 복원 및 환경 smoke PASS. 연구 runner는 재개하지 않았다.**
새 서버의 기본 cu132 환경을 유지하고, 이전 Baseline과 같은 cu130 패키지를 별도
가상환경에 설치했다. 이 결과는 Baseline Forward Equivalence나 Stage4A 연구 완료를
의미하지 않는다. [이전 데이터 재개 검사](10i_stage4a_resumption_check_58576054.md)에서
확인한 완료60/64 Sequence와 남은4개·136관측의 상태는 그대로다.

## 설치 위치와 실행

- 인스턴스: **58576054**, RTX3090.
- 독립 Python: `/workspace/venvs/task10i-stage4a-cu130/bin/python`
- 운영 Source/로그: `/workspace/task10i_stage4a_environment_20261101`
- 설치·검증은 `task10i-stage4a-env` tmux의 Python runner에서 수행했고 종료됐다.
- 기본 `/venv/main`의 설치 전후 distribution/version snapshot은 동일하다.
  시스템 드라이버·시스템 CUDA 설정을 변경하지 않았다.

원격 Bash/Zsh에서 사용할 때:

```sh
source /workspace/task10i_stage4a_environment_20261101/activate.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

이 task 전용 activation 파일은 venv 선택과 함께
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`을 지정하고 bytecode 쓰기를 끈다.
설치된 SAPIEN3.0.3의 `_ensure_vulkan_icd()`는 기본적으로 `/usr/share/vulkan/icd.d`를
확인하지만, 새 호스트의 실제 NVIDIA ICD는 `/etc/vulkan/icd.d`에 있었다. 실제 설치
Source를 읽어 지원되는 환경변수를 확인했다. 호스트 파일이나 SAPIEN Source를 수정하지
않았다. 이 설정에서 import 경고가 사라졌으며, Renderer 실행 성공을 뜻하지는 않는다.

## 복원 기준과 실제 버전

원본 `/workspace/full_pipeline_environment.txt`와 보존된 SAM2 Source를 사용했다.
PyTorch cu130 wheel index와 PyPI에서 명시된 버전을 설치했다.

| 항목 | 새 독립 환경의 실제 값 |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| torchcodec | 0.16.0+cu130 |
| Torch CUDA runtime | 13.0 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| h5py | 3.16.0 |
| OpenCV / headless | 둘 다5.0.0.93, 이전 pin 유지 |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| Gymnasium | 1.4.0 |
| Transformers / Hydra | 5.19.0 / 1.3.7 |
| SAM-2 | 1.0, 기존 Source의 독립 copy에서 설치 |
| cuDNN | `torch.backends.cudnn.version() == 92400` |

총135개 distribution이 설치됐다. 원본에서 `==`로 버전이 지정된 **132개가 모두 일치**했다.
정규화한 직접 요구사항133개도 불일치0이다. `uv pip check`는135개 전체에 대해 호환성을
통과했다. [원본 pin 비교](../../outputs/task10i/stage4a/environment_restore_58576054/original_pin_comparison.json) ·
[실제 설치 목록](../../outputs/task10i/stage4a/environment_restore_58576054/installed.json).

다음 세부 차이와 복원 방식을 별도로 남긴다.

- 원본 `pip`와 `packaging`은 사라진 conda build 경로의 file URL이며 버전이 없었다.
  새 기본 환경에서 확인한 `pip26.2.2`, `packaging26.3`을 명시적으로 설치했다.
  이 두 도구까지 이전 환경과 동일하다고 주장하지 않는다.
- SAM2 원본은 빌드 출력으로 오염시키지 않도록 운영 root 아래에 복사한 후 설치했다.
  보존된 Source에는 compiled `_C`가 없었으며 `SAM2_BUILD_CUDA=0`으로 설치했다.
  새 환경에도 `_C`가 없는 것을 확인했다. 이후 Forward Equivalence로 실제 결과를
  대조하기 전에는 후처리 출력까지 같다고 선언하지 않는다.
- `cuda-toolkit` 메타 패키지는 직접 설치 요구에서 제외했지만, 실제 Torch wheel의
  의존성이 `cuda-toolkit[...]==13.0.3`을 요구해 **13.0.3.0이 전이 설치**됐다.
  이것도 원본 pin과 일치한다. pip 패키지 설치이며 시스템 드라이버 교체가 아니다.

venv 표시 용량은 약6.4GiB이고 설치 후 `/workspace`의 여유는 약180GiB였다.
이 호스트의 overlay를 영구 백업으로 간주하지 않는다.

## 실제 실행한 검증

아래 검사는 새 호스트의 **새 venv**에서 실행했다.

| 검사 | 결과 |
|---|---|
| Torch CUDA / torchvision 버전 및 CUDA device | PASS, RTX3090 |
| CUDA matrix multiplication | PASS, finite |
| cuDNN convolution | PASS, finite |
| torchvision CUDA NMS | PASS |
| Transformers DINO 클래스 import | PASS |
| SAM2 video predictor import / 실제 YAML compose | PASS |
| SciPy / OpenCV connected component | PASS |
| 임시 HDF5 write/read roundtrip | PASS |
| ManiSkill / SAPIEN import, PegInsertionSide 등록 | PASS |
| 의존성 호환성 / 원본 버전 pin 대조 | PASS |
| 기본 `/venv/main` 전후 package snapshot | 동일 |

새 Python의 기본 수치 설정도 이전 model identity와 같았다:
`deterministic=False`, `matmul_tf32=False`, `cudnn_tf32=True`.
학습용 seed helper를 호출하거나 이 설정을 조용히 바꾸지 않았다.
[최종 smoke 결과](../../outputs/task10i/stage4a/environment_restore_58576054/smoke.json).

첫 smoke는 검증 스크립트가 SAM2 import로 이미 초기화된 Hydra를 다시 초기화해
중단됐다. 설치된 `sam2/__init__.py`를 확인하고 기존 초기화를 재사용하도록 **검증
스크립트만** 수정했다. 재설치 없이 전체 smoke가 통과했다. 최초 script/log와 최종
script/log를 모두 보존했다. 초기 Vulkan 경고도 위 activation 설정으로 해결했다.

## 미실행 범위와 재개 경계

이번 작업에서는 simulator step, CPU PhysX 실행, CUDA SAPIEN 렌더링,
DINO/SAM2 모델 forward, 기존 Mask/Graph와의 exact equivalence, 연구 runner를
실행하지 않았다. 운영 환경 점검이며 새 scientific evaluation 결과가 아니므로
ExperimentLogger run을 만들지 않았다. 기존 Logger·Publisher를 수정하거나 외부 게시를
수행하지 않았다. 원본 데이터, 기존 실험 Source/산출물, Checkpoint와 split을 변경하지 않았다.

후속 연구 재개 시에는 새 isolated Source와 output root, 새 ExperimentLogger UUID를
사용하고, **이 새 venv에서 Forward Equivalence를 먼저 확인**해야 한다. 검증된60개를
재사용하고 미완료4개는 각각 t0부터 새 causal memory로 처리한다. 설치 완료만으로
이후 감사·후보 평가·Shared Validation이 완료됐다고 판단하지 않는다.

운영 상태: **ENVIRONMENT_READY / FORWARD_EQUIVALENCE_NOT_EXECUTED / RESEARCH_NOT_RESUMED**.

근거 파일: [상태](../../outputs/task10i/stage4a/environment_restore_58576054/status.json),
[복원 계약](../../outputs/task10i/stage4a/environment_restore_58576054/restore_contract.json),
[최종 검증 로그](../../outputs/task10i/stage4a/environment_restore_58576054/verification.log),
[SHA256 목록](../../outputs/task10i/stage4a/environment_restore_58576054/artifact_sha256.json).

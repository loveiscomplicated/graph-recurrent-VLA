# Task 10-I — Development 렌더러 오류 확인

2026-11-02 사용자 요청에 따라 중단 상태를 확인했다. **학습은 완료됐고 Development 평가에서 중단됐다.**
학습 재실행, 평가 재시작, Source/Checkpoint/환경 수정은 하지 않았다.

## 완료 범위와 중단 지점

- Instance58752018, root `/workspace/task10i_predicted_state_gru_20261102`.
- 학습11,487 updates 완료. Frozen F1 Hash는 초기/최종 모두
  `668d2ad1be0149e6735f428fd4f3e729adc5d7159f0f1f835478afa68cb0b646`다.
- Step3000/6000/9000/11487 Checkpoint4개의 SHA, 내부 Step, Frozen Hash를 Readback 검증했다.
- Development24 Episode 중23개 완료. 저장된 Trace/MP4 SHA가 모두 일치했다.
- 미완료 조건은 **step11487 / seed30042 / S0**, 즉 마지막 Development Episode다.
- 2026-11-02 14:46:29 KST에 Pipeline이FAILED로 종료됐다. 연구 tmux도 종료됐고 별도 `ssh_tmux`는 건드리지 않았다.
- Checkpoint Selection과 독립Final/Handoff는 아직 시작하지 않았다. PIPELINE_COMPLETE를 기록하지 않았다.

## 직접 원인과 진단의 한계

마지막 Episode의 `environment()` → `gym.make()` → `sapien.render.RenderSystem`에서 다음 오류가 났다.

```text
Extension VK_KHR_external_memory_capabilities is not available.
Extension VK_KHR_external_semaphore_capabilities is not available.
Your GPU driver does not support Vulkan.
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```

CUDA OOM이나 BC Loss/Gradient 오류로 종료된 것이 아니다. 원인은 Vulkan Instance 생성 실패까지 확인했다.
왜23개 평가 후 그 시점에 생성에 실패했는지의 근본 원인은 아직 확정하지 않았다.

확인 당시 GPU Driver는595.84, NVIDIA ICD 경로와 라이브러리는 존재했고 GPU 메모리는1MiB 사용 중이었다.
같은 cu130 환경의 새 프로세스에서 CPU PhysX + CUDA Renderer로 reset과3 step, 양쪽128×128 RGB-D를 실행해PASS했다.
또한 Vision/Policy 없이 같은 환경을26회 create/reset/close하는 별도 진단도 모두PASS했다.
마지막 FD 수는82였고FD 한도는1024였다. 단순한 환경 생성 반복만으로 FD 고갈이나 오류를 재현하지 못했다.

따라서 오류 문구만으로 호스트 드라이버가 지속적으로 고장 났다고 단정하지 않는다.
장시간 Vision+Simulator 프로세스의 자원/상태 문제 또는 일시적인 Vulkan 초기화 실패 등이 후보지만,
현재 증거로 구분하지 못했다. 단순 반복 진단은 실제 장시간 평가 전체의 재현이 아니다.

## 보존과 재개 경계

가중치와 완료23개 결과가 유효하므로 학습을 다시 돌릴 필요는 없다.
재개 시에는 미완료 Development1개부터 완료하고, 사전 등록한 선택 규칙을 적용한 뒤 Final/Handoff를 진행해야 한다.
완료 Episode는 SHA 검증 후 재사용하고 실패한 디렉터리와 로그는 보존해야 한다.
평가 프로세스의 자원 수명 문제를 추가 검증할 수 있으며, 조건·Seed·Budget·가중치를 바꿀 이유는 확인되지 않았다.
이번 요청에서는 재시작하거나 Driver/CUDA Wheel/Source를 수정하지 않았다.

[Pipeline 상태](../../outputs/task10i/predicted_state38_bc_20261102/failure_20261102/pipeline_status.json) ·
[Checkpoint/완료 Episode 검증](../../outputs/task10i/predicted_state38_bc_20261102/failure_20261102/preservation_verification.json) ·
[실패 Traceback](../../outputs/task10i/predicted_state38_bc_20261102/failure_20261102/development_failure_tail.log) ·
[새 프로세스 Renderer Smoke](../../outputs/task10i/predicted_state38_bc_20261102/failure_20261102/simulator_smoke.json) ·
[26회 반복 진단](../../outputs/task10i/predicted_state38_bc_20261102/failure_20261102/renderer_cycles.json).

이 진단은 연구 성능 평가나 새 학습이 아니므로 새 ExperimentLogger UUID를 발급하지 않았다.
기존 공통 인프라와 자산을 수정하지 않았고 Git Commit/Publishing도 수행하지 않았다.

후속: 사용자 요청으로 기존 가중치와23개 결과를 보존한 평가 재개 Runner를 시작했다. [재개 인계](10i_predicted_state38_bc_resume_20261102.md).

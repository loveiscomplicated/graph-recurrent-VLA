# Task 10-I Stage 4A-2 — Qwen VLA 문헌·구현 조사와 기본 인식 진단

작성일: 2026-11-02. 상태: `COMPLETE_VERIFIED` — 알려진 사례의 원인 진단 완료, 독립 성능 검증 아님.

## 1. 확인된 결론

**이번에 조사한 네 가지 오분류는 같은 모델과 같은 RGB 입력에서 프롬프트를 바꾸자 모두 교정됐다.** 기존 기능적 역할 프롬프트는 명확한 GT 역할이 있는 후보 9개에서 5개를 맞혔고, 외형 중심 프롬프트는 같은 9개를 모두 맞혔다. Robot→BOX 두 건과 후기 PEG→BOX 두 건이 교정됐으며, 기존 정답인 명확한 대조 사례 다섯 건에서는 퇴행이 없었다.

따라서 이 사례들을 근거로 “9B 모델이라 로봇이나 막대를 인식하지 못한다”, “프롬프트로 해결되지 않는다”고 판단할 수 없다. 기본 시각 묘사·간단한 외형 분류보다 기능적 역할 추론을 먼저 평가하고, 그 실패를 모델 자체의 한계로 해석한 이전 판단은 성급했다. 처음 세 사례에서 성공한 역할 프롬프트를 다른 카메라·후기 장면에 확대 적용하기 전에 이 진단이 필요했다.

다만 **9/9는 기존 실패와 정답을 알고 선정한 개발 사례의 결과**다. 전체 후보 풀의 Recall, 미사용 Parent 정확도, Full-image Bounding Box 생성 능력, SAM2 초기화 성능이 100%라는 뜻이 아니다. 자유 묘사의 물체 이름 오류와 단순 이진 질문의 실패도 남았다.

## 2. 실행 범위와 입력 경로

사용자 요청에 따라 기본 인식 실패와 VLA 활용 방식의 차이를 조사했다. 이전의 DINO-free Full-image 실험과 별개인 **DINO 후보 검증기 진단**이다. 기존 DINO가 보존한 후보의 원본 장면·표시 장면·Crop을 재사용했으며 새로운 DINO 추론은 하지 않았다. GT는 사례 선정과 사후 평가에 사용했고, VLM 입력에는 제공하지 않았다. 알려진 외형 설명은 프롬프트에 명시했다.

| 항목 | 실제 실행 |
| --- | --- |
| GPU | Vast.ai `58576054`, RTX 3090, `cuda:0` |
| 모델 | `Qwen/Qwen3.5-9B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 정밀도 | 전체 BF16, 양자화 없음, CPU/Disk Weight Offload 없음 |
| Attention / Generation | SDPA, Batch 1, Greedy, `enable_thinking=False`, `use_cache=True` |
| 출력 예산 | 분류·이진 질문 16 Token, 짧은 묘사 96 Token |
| RGB | 기존 시뮬레이터에서 보존한 Native 512 RGB; 신규 Render/Reset 없음 |
| Processor | 기존 Processor, `min_pixels=max_pixels=262144`; Crop 종횡비 유지 및 Patch 배수 반올림 |
| 표본 | Parent 3개(`0197`, `0275`, `0305`), Camera-Frame 7개, 후보 10개 |
| 실행 수 | 모델 Load 1회, 요청 50회, 자동 Retry/추가 Warmup 0회 |
| 미실행 | 신규 DINO, SAM2, Tracking, Simulator, Training, Graph/Controller 평가 |

각 요청은 새 User Message와 빈 초기 Cache로 시작했다. 실제 첫 Forward Hook에서 **50/50 요청의 Cache Length=0**을 확인했다. 모델 상주는 재사용하지만 대화 내용은 누적하지 않았다. 요청 순서는 후보·조건 Hash로 섞었으며, 다섯 조건은 실행 전에 동결하고 출력에 따라 수정하지 않았다.

Torch `2.14.1+cu130`, Transformers `5.19.0`, BitsAndBytes `0.50.2`, Python `3.12.15` 환경을 재사용했다. 이번 BF16 실험에서 BitsAndBytes Quantization은 적용하지 않았다. 기존 PyTorch 연산 경로를 사용했으며 별도 최적화 Kernel을 설치하지 않았다. CUDA 실행은 원격 서버에서 했고 현재 macOS 호스트에서 GPU/Simulator를 실행하지 않았다.

## 3. 사전 고정한 다섯 진단 조건

| 조건 | 이미지 | 질문 | 비교에서 확인할 수 있는 것 |
| --- | --- | --- | --- |
| A `original_three` | Full + 표시 장면 + Crop | 기존 Functional-role 분류 | 기존 오류 재현 |
| B `appearance_three` | A와 완전히 같은 세 이미지 | 색·형태·부품 중심 분류 | A→B는 텍스트 입력 변경의 효과 |
| C `appearance_crop` | B의 Crop 한 장 | B와 같은 프롬프트 | B→C는 두 Context 이미지 제거 효과 |
| D `caption_crop` | Crop 한 장 | 색·형태·부품을 짧게 묘사 | 강제 역할 Label 밖의 시각 묘사 |
| E `robot_binary_crop` | Crop 한 장 | 로봇 부품인지 YES/NO/UNCERTAIN | 이진 Robot Gate의 실패 여부 |

A와 B는 입력 이미지 경로와 SHA가 10/10 후보에서 동일하다. B와 C는 프롬프트 본문·Preamble이 동일하다. A→B에서는 역할 우선 지시, 외형 사전정보, 후보 지칭 방식과 문장 길이가 함께 바뀌었다. **어느 한 문장이나 색상만이 개선 원인인지까지 분리한 실험은 아니다.**

기존 의미 프롬프트 SHA256은 `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06`이다. 모든 조건 원문은 [Prompt Identity](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/prompt_identity.json)에 저장했다. A의 예전 답변은 10/10 동일하게 재현됐다.

## 4. 결과와 분모 교정

| 조건 | 명확한 역할 정답 | PEG | BOX | ROBOT | 형식 준수 |
| --- | --- | --- | --- | --- | --- |
| A 기존 역할 프롬프트 | **5/9** | 1/3 | 2/2 | 2/4 | 10/10 |
| B 외형 프롬프트·세 이미지 | **9/9** | 3/3 | 2/2 | 4/4 | 10/10 |
| C 외형 프롬프트·Crop | **9/9** | 3/3 | 2/2 | 4/4 | 10/10 |
| D 자유 묘사 | 정답률 산출 안 함 | — | — | — | 자유 문장 |
| E Robot 이진 질문 | **7/9** | 1/3 | 2/2 | 4/4 | 10/10 |

역할별 E의 정답은 ROBOT이면 YES, PEG/BOX이면 NO다. 40개 분류·이진 요청은 모두 엄격한 출력 형식을 지켰다. 따라서 이번 A의 네 오분류는 JSON/Label Parsing 실패로 설명되지 않는다.

후보 `0275/F0/hand_camera/peg_rank000`는 Peg와 Gripper가 함께 들어 있는 `AMBIGUOUS` 영역이다. 기존 규칙은 Peg/Box/Robot 전경 중 한 Actor 비율이 70% 이상이고 해당 가시 픽셀이 16 이상일 때만 유일한 역할을 부여한다. 이 후보는 A/B/C 모두 PEG를 출력했지만, 유일한 GT 역할을 지정할 수 없어 모든 조건의 정확도 분모에서 동일하게 제외했다. 후보 자체와 원시 응답은 보존했다.

초기 원격 집계 코드 `finalize.py`는 출력 Label 집합에 없는 AMBIGUOUS를 오답으로 포함해 A=5/10, B=C=9/10으로 기록했다. E에서도 이를 부적절하게 NO로 취급했다. **이 집계 오류를 숨기거나 기존 산출물을 덮어쓰지 않았다.** 원본 [diagnostic_metrics.json](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/diagnostic_metrics.json)은 보존하고, 별도 [독립 재계산](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/perception_independent_analysis.json)에서 명확한 9개 분모로 교정했다. 이 보고서는 교정된 수치를 사용한다.

### 4.1 실제로 교정된 네 사례

| Parent / 시점 / Camera / 후보 | GT 역할 | 기존 A | 외형 B | 외형 Crop C |
| --- | --- | --- | --- | --- |
| 0275 / F1 / Wrist / box_rank000 | ROBOT | BOX | ROBOT | ROBOT |
| 0197 / F1 / Main / box_rank006 | ROBOT | BOX | ROBOT | ROBOT |
| 0275 / F1 / Wrist / peg_rank000 | PEG | BOX | PEG | PEG |
| 0305 / F1 / Wrist / peg_rank000 | PEG | BOX | PEG | PEG |

후기 Peg 두 후보의 GT Peg Bounding Box IoU는 각각 **0.9830, 0.9719**다. 올바른 후보가 주어졌음에도 기존 Prompt가 거부했고, 외형 Prompt는 같은 후보를 PEG로 분류했다. 후보 ID의 `peg`/`box`는 보존된 DINO Query 구분이며 VLM에 정답 Label로 제공하지 않았다.

![기존 오분류 네 사례: 같은 Crop에서 역할 프롬프트와 외형 프롬프트 비교](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/perception_prompt_failure_comparison.png)

### 4.2 자유 묘사가 알려주는 것과 남은 실패

로봇 오분류 후보 `0197/F1/Main/box_rank006`에서 D는 흰 로봇 팔, 검은 부분, 원형 관절을 묘사하고 Robotic Manipulator로 이름 붙였다. `0275/F1/Wrist/box_rank000`에서는 회색 각진 구조와 검은 직사각형 부분을 읽었지만 구체적인 물체명은 불확실하다고 했다.

후기 Peg 두 개에서는 Salmon-pink/Light-gray 두 색과 직사각형 형태를 묘사하면서 문·패널 또는 사람 모양으로 추측했다. 초기 Peg에서는 칼이나 담배로 추측한 응답도 있었다. 따라서 **시각 속성을 읽는 증거는 있으나 자유로운 물체 이름 추론이 안정적이라는 증거는 아니다.** 관측된 색·형태 묘사와 잘못 붙인 일상적 이름을 분리해야 한다.

E는 Robot 네 개를 모두 YES로 판단했지만, 후기 Peg 두 개도 YES로 판단했다. “먼저 로봇인지 물으면 해결된다”는 일반적인 Gate 설계는 이번 결과로 지지되지 않는다. 성공한 답변만 골라 전체 분류기를 대체하지 않는다. 전체 문장은 [사례 CSV](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/perception_cases.csv)와 [50개 원시 응답](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/verifier_raw_responses.json)에 있다.

## 5. 왜 Functional-role을 우선한 것이 문제가 됐는가

관측: 기존 Prompt는 외형보다 작업의 기능적 역할을 우선하라고 명시했다. 고정 Asset 외형 정보는 주지 않았다. 반면 기존 DINO에는 `red and white block`, `cardboard box` 같은 외형 표현을 제공했다. 따라서 앞선 DINO/Qwen 비교에는 모델 차이 외에 **제공한 사전정보 차이**도 있었다.

관측: 동일 BF16 모델에서 같은 이미지의 텍스트만 바꿔 네 오류를 교정했다. 동일 외형 텍스트에서 Crop만 사용해도 이번 명확한 아홉 후보의 Label은 유지됐다. 초기 RGB를 추가한 [직전 비교](10i_stage4a2_qwen35_temporal_reference.md)는 후기 Peg 실패를 고치지 못했다.

가설: 정지 이미지에서 행동 의도·수용 역할을 추론하게 한 지시가 외형과 후보 자체를 확인하는 지시보다 이 장면에 부적합했을 수 있다. Crop이 장면의 큰 부분을 차지하는 후기 Wrist 입력에서 후보와 작업 역할의 연결이 흔들렸을 수 있다. **내부 Attention이나 인과적 특징을 직접 측정하지 않았으므로 이는 기전 가설이다.** 이번 결과가 직접 입증한 것은 텍스트 변경의 효과까지다.

새 Prompt의 색·형태는 이 Task의 알려진 Asset 특성을 사용한다. 사용자 요청에 맞는 Task-specific 사전정보이며 GT 위치·개별 정답 예시를 넣지 않았다. 그러나 기존의 색·형태 불변 Zero-shot 조건과는 다르다. 다른 색 Asset이나 새로운 물체에 대한 일반화 성능으로 보고하지 않는다.

## 6. 다른 Qwen 기반 VLA는 무엇을 실제로 하는가

원 논문과 공개 Source/Config를 조사했다. 코드 5개 Repository의 25개 파일을 Commit에 고정해 저장했고, 모델 Weight 다운로드·설치·학습·재현 실행은 하지 않았다. 다음은 문헌·코드 확인이며 해당 논문의 실험 결과를 우리 환경에서 재현했다는 뜻은 아니다.

| 자료 | 실제로 확인한 경로와 Freeze 범위 | 우리 실험에 대한 의미 |
| --- | --- | --- |
| VLM4VLA | Qwen2.5-VL의 Hidden State에서 Action Query를 꺼내 학습된 Action Head로 전달. 공개 Bridge Config는 `train_vision=true`, `freeze_backbone=false`. | 순수 Zero-shot 단어 분류와 학습된 제어 정책은 평가 대상이 다르다. [구현](https://github.com/CladernyJorn/VLM4VLA/blob/9ce02a4b1babffd8185ee1b01cae6c59a56d9764/vlm4vla/model/backbone/roboqwen25vl.py), [Config](https://github.com/CladernyJorn/VLM4VLA/blob/9ce02a4b1babffd8185ee1b01cae6c59a56d9764/configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge.json). |
| starVLA | QwenOFT는 마지막 Hidden State의 Action Query Embedding으로 연속 행동을 예측하고 L1 Loss로 학습. Qwen2.5 OXE Config의 `freeze_modules`는 비어 있고, 다른 LIBERO Config는 `qwen_vl_interface`를 동결한다. 후자의 모델은 **Qwen3-VL-4B**다. | Frozen VLM + 학습된 Head 구성이 실제로 존재한다. 이를 Frozen Qwen2.5의 무학습 성능으로 바꾸어 인용하면 안 된다. [QwenOFT](https://github.com/starVLA/starVLA/blob/13d8ad43eb4128554c57227507055ecf9c3a62a2/starVLA/model/framework/VLM4A/QwenOFT.py), [OXE](https://github.com/starVLA/starVLA/blob/13d8ad43eb4128554c57227507055ecf9c3a62a2/examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml), [LIBERO](https://github.com/starVLA/starVLA/blob/13d8ad43eb4128554c57227507055ecf9c3a62a2/examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml). |
| LingBot-VLA | `freeze_vision_encoder`와 전체 Qwen VLM을 동결하는 `train_expert_only`가 별도 분기다. 확인한 RobotWin Config는 `freeze_vision_encoder=false`. | Vision Encoder 동결과 전체 VLM 동결을 같은 말로 취급하면 안 된다. 한 Config가 모든 Checkpoint의 학습 조건을 입증하지는 않는다. [구현](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/models/vla/pi0/modeling_lingbot_vla.py), [Config](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/configs/vla/robotwin_load20000h.yaml). |
| Qwen-VLA | Qwen3.5-4B Backbone과 별도 Action Expert 사용. 처음 Text-to-action 단계에서는 VLM을 동결하지만 이후 Continued Pretraining에서 양쪽을 학습하고 SFT/RL을 진행한다. | 완전히 동결한 기본 모델에 Prompt만 주는 방법의 성공 사례로 인용할 수 없다. [원 논문](https://arxiv.org/html/2605.30280v1). 공개 Repository는 README만 코드 조사 범위에 포함했다. |
| GUI-Actor | 동결 Backbone에 학습된 Attention 기반 Grounding Head/Token을 붙이는 LiteTrain을 평가한다. 해당 Freeze Ablation은 **Qwen2-VL** 기반이다. | Frozen Feature에 맞는 학습된 출력 경로의 예시다. 물리 로봇 VLA가 아니며, 해당 표를 Qwen2.5의 무학습 로봇 인식 성능으로 인용하지 않는다. [원 논문](https://arxiv.org/html/2506.03143v1), [학습 코드](https://github.com/microsoft/GUI-Actor/blob/d98d1bbd01862f9112114b83b032f492c365a173/train.py). |
| SAVLA | 논문은 GR00T N1.5의 SigLIP2 + Qwen3-1.7B Backbone을 동결하고 Canonicalizer와 Equivariant Action Head를 학습한다고 설명한다. | 완전 동결 가능성을 배제하면 안 된다는 문헌 근거다. 해당 논문이 연결한 공개 Repository는 조회 시 404여서 Source 재검증은 못 했다. [원 논문](https://arxiv.org/html/2609.16641v1). |

starVLA의 Freeze는 이름뿐인 설정인지도 확인했다. Trainer가 `freeze_modules`를 전달하며, Utility는 지정 Submodule의 모든 Parameter에 `requires_grad=False`를 적용하고 Optimizer 구성에서도 제외한다. [Freeze 구현](https://github.com/starVLA/starVLA/blob/13d8ad43eb4128554c57227507055ecf9c3a62a2/starVLA/training/trainer_utils/trainer_tools.py).

VLM4VLA Table 3의 Qwen2.5-VL-3B는 SimplerBridge에서 전체 Fine-tuning 48.00%, Vision Encoder만 동결하면 23.95%였다. 이는 그 학습법에서 시각 적응이 중요했다는 근거이며 **전체 Frozen VLM이 모든 설계에서 불가능하다는 증명은 아니다.** Table 7의 해상도 증가도 그 조건에서는 일관된 개선을 주지 않았다. 해당 수치는 행동 성공률이므로 우리 후보 Label 정확도나 Box IoU와 직접 비교하지 않는다. [논문 Table 3·7](https://arxiv.org/html/2601.03309v2).

종합하면 사용자의 **Frozen VLM을 구성요소로 쓰겠다는 방향 자체는 배제할 근거가 없다.** 생성된 한 단어 Label의 실패는 Hidden Feature의 유용성까지 검증한 결과가 아니다. 동시에 VLA의 학습 경로 차이를 이유로 우리 기본 오분류를 당연한 것으로 설명해서도 안 된다. 이번 네 사례는 실제 Prompt 비교로 교정됐다.

Qwen2.5를 이 환경에서 직접 실행하지 않았으므로 Qwen2.5와 Qwen3.5-9B의 상대 순위는 판단하지 않는다. 모델 크기만으로도 이번 실패를 설명할 수 없다. Thinking 활성화·다른 Decode 설정에서 최고 성능을 검증한 실험도 아니다.

## 7. 비용과 보존 검증

모델 Load 3.454초, Load 후 GPU Allocated 17.527 GiB. Process CPU RSS Sample Peak 18.163 GiB였다. 최초 요청 A의 End-to-end는 1,395.42 ms였으며 Warm 집계에서 제외했다.

| 조건 | Warm 요청 수 | End-to-end Mean / Median / p95 (ms) | Warm Peak Allocated (GiB) |
| --- | --- | --- | --- |
| A 기존 세 이미지 | 9 | 648.83 / 640.50 / 683.09 | 17.956 |
| B 외형 세 이미지 | 10 | 610.69 / 597.05 / 642.56 | 17.895 |
| C 외형 Crop | 10 | 375.52 / 358.76 / 416.81 | 17.735 |
| D 자유 묘사 | 10 | 2,239.09 / 2,212.67 / 2,711.01 | 17.706 |
| E Robot 이진 질문 | 10 | 326.30 / 328.17 / 330.77 | 17.712 |

조건별 최초 실행을 모두 Cold로 제외한 것이 아니라 **한 Process의 첫 요청 하나만** 제외했다. CPU Preprocessing과 GPU Generation을 포함하는 실제 요청 측정이며, 작은 표본의 p95다. 분류 Prompt마다 입력·출력 Token 수가 다르므로 속도 차이를 순수 시각 연산 개선으로 해석하지 않는다. [요청별 Latency/VRAM 원본](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/verifier_latency_vram.json).

공통 `utils.experiment_logger.ExperimentLogger`를 W&B 비활성화로 재사용했다. Run ID는 `f13e502b3bda48d2bac21e54add83519`다. 새 Logger/Publisher는 만들지 않았다.

원격 실행의 보호 대상 2,381개 SHA, 기존 환경, Model/Processor가 전후 동일했다. 50개 원시 응답은 사후 채점 전후 SHA `f8740f546811b1731bd00f339bc7c7252c97b429775e22d2161757ec5910283d`로 동일했다. 로컬 독립 검증은 원격 Index의 33개 Artifact, 공개 Source 25개, 로컬/원격 실행 Source 6개, 요청 이미지 SHA 90개, GT Actor 구성 10개, Box IoU 20개를 확인했다. A/B 동일 입력과 모든 요청의 초기 빈 Cache도 재확인했다. [원격 보존 검증](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/verification.json), [독립 검증](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/verification.json).

Production Source와 기존 Checkpoint/Dataset은 변경하지 않았다. 기존 `PAUSED_SCOPE_REVIEW`와 Stop Stub을 유지했고, Graph/Controller 후속 실험을 시작하지 않았다. Commit·외부 게시·추가 GPU 실험도 하지 않았다.

## 8. 다음 검증에서 분리해야 할 것

현재 판단은 **Task-specific 외형 Prompt를 쓴 후보 검증기의 가능성이 다시 열렸다**는 것이다. DINO 제거, 최고 성능 Prompt 확정, 독립 Recall 개선은 아직 입증되지 않았다.

다음 단계는 이미 개선한 Prompt를 고정한 뒤 미사용 Parent를 사전 선정해, 동일 후보 풀에서 원시 점수 1위·기존 Prompt·외형 Prompt를 비교하는 것이다. Peg 정답 거부와 Robot/Box 오승인을 함께 측정하고, 최종 선택과 실제 SAM2 초기 Mask까지 확인해야 한다. 현재 진단의 아홉 후보나 그 주변 프레임을 독립 성능 분모로 다시 사용하면 안 된다. Crop-only의 속도 이점도 혼합 후보·가림·다른 색 Asset에서 검증이 필요하다. 이 후속 실험은 이번 조사에서 실행하지 않았다.

## 9. 재현 자료

원격 실행 Root: `/workspace/task10i_stage4a2_qwen35_perception_diagnostic_20261102_v1/`.

- [사전 프로토콜](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/protocol.json), [사례 Manifest](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/input_manifest.json), [요청 Manifest](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/request_manifest.json), [모델 Identity](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/model_identity.json).
- [원시 응답](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/verifier_raw_responses.json), [원본 Artifact Index](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/artifact_index.json), [교정 분석](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/perception_independent_analysis.json).
- [공개 Source Commit/SHA Index](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/reference_source_index.json), [조사 Artifact Index](../../outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1/artifact_index.json).
- [실행 코드](../../graph/experiments/qwen35_perception_diagnostic_20261102_v1/infer.py), [사후 독립 재계산 코드](../../graph/experiments/qwen35_vla_research_audit_20261102_v1/analyze.py).

## 부록: 실제 실행한 외형 프롬프트 원문

아래 본문은 B/C에서 동일하게 사용했고 별도 Preamble은 없었다. 실행 전 저장한 [원문 파일](../../outputs/task10i/stage4a2/qwen35_perception_diagnostic_20261102_v1/appearance_three_prompt.txt)과 Prompt Identity를 기준으로 재현한다.

```text
Identify the object shown in the last image, which is a crop of the candidate.
If other images are provided, the first shows the full scene and the second shows the candidate outlined.
Classify the candidate itself using its visible appearance:
ROBOT: Gray or white robot arm links, or dark or black gripper fingers. A separate object held by the gripper is not a robot part.
PEG: A red-and-white elongated rectangular piece. It can be held by the gripper or partly outside the image.
BOX: A tan or brown block-shaped fixture. An opening may be visible, depending on the view.
Use color, shape, and visible parts together. Do not classify a different object elsewhere in the scene.
Choose OTHER for a clearly different object, or UNCERTAIN if the candidate cannot be identified.
Return exactly one label: PEG, BOX, ROBOT, OTHER, UNCERTAIN.
Do not add explanations or punctuation.
```

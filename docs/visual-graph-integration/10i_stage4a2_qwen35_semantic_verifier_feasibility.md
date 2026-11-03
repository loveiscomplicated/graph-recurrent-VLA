# Task 10-I Stage 4A-2 — Qwen3.5-9B Semantic Verifier Feasibility

2026-11-02. **최종 판정: `LOCAL_SEMANTIC_FAILURE`. `PAUSED_SCOPE_REVIEW` 유지.**

고정한 세 후보에서 Qwen은 **P=`OTHER`, N=`BOX`, C1=`OTHER`**라는 내용을 출력했다.
실제 Peg인 P/C1을 Peg로 구별하지 못했다. 세 응답은 모두 Markdown JSON Fence를
포함해 엄격한 출력 계약에서도 `INVALID_OUTPUT`이었다. 따라서 이번 조건의 Semantic
Verifier는 채택하지 않으며, 사전 Gate에 따라 기존 Perception 모델과의 동시 상주 검사는
실행하지 않았다. 이 보고서는 앞선 추가 로드 승인 대기 중간 보고서를 대체한다.

## 실행 Identity와 실제 양자화

사용자 지정 Vast.ai **58576054 / RTX3090 24GB**에서 실행했다. 시작 시 GPU 사용1MiB,
다른 GPU 프로세스0, `/workspace`는 overlay200GiB 중 약162GiB 여유였다. CPU 가용 메모리도
확인했다. 공식 Repository 파일은 약19.3GB이며 다운로드/임시 Cache 예산60GiB를 확보했다.
기존 Qwen Cache가 없어 새 Output Root에 다운로드했다. 기존 파일 삭제·연구 프로세스 종료·
새 인스턴스 생성은 없다.

모델은 공식 Post-trained Multimodal **`Qwen/Qwen3.5-9B`**, Revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`다.
4개 Safetensors Shard의 SHA256을 공식 Hub LFS 값과 대조했다. 설치된
**`Qwen3_5ForConditionalGeneration` + `AutoProcessor`**의 실제 이미지 API를 사용했다.
Text-only CausalLM은 사용하지 않았다.
[공식 모델](https://huggingface.co/Qwen/Qwen3.5-9B),
[멀티모달 API](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[BitsAndBytes 문서](https://huggingface.co/docs/transformers/quantization/bitsandbytes)를 참조했다.

독립 `env/`에 BitsAndBytes **0.50.2**, Accelerate **1.15.0**을 설치하고 기존 연구
site-packages는 변경 없이 참조했다. Python **3.12.15**, Torch **2.14.1+cu130**,
CUDA runtime **13.0**, driver **595.84**, Transformers **5.19.0**이다.
Batch1·Inference mode·SDPA·Greedy로 실행했고 Kernel Patch/NVFP4는 사용하지 않았다.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_enable_fp32_cpu_offload=False,
)
# from_pretrained: device_map={"": 0}, dtype=torch.bfloat16,
# attn_implementation="sdpa", local_files_only=True
```

실제 양자화 범위는 다음과 같다.

| 실제 측정 항목 | 값 |
|---|---:|
| `Linear4bit` Layer | **358개**: Language248 / Vision110 |
| NF4 Packed Weight Storage | **3,684,841,472 bytes** |
| Quantization State Storage | **116,974,552 bytes** |
| BF16 비양자화 Parameter Storage | **4,080,261,600 bytes** |
| `get_memory_footprint()` | **7,765,103,472 bytes** |
| 실제 Parameter / Buffer Device | Parameter760개 / Buffer4개 모두 `cuda:0` |

Embedding과 LM Head는 각각 `[248320,4096]` BF16이며 각2,034,237,440 bytes다.
Vision Patch Conv·Position Embedding·Hybrid Linear Attention Conv1d·Norm 등도 BF16으로
남았다. Linear4bit의 실제 Compute Dtype은 BF16이다. **모든 파라미터가4-bit인 모델이 아니다.**
`get_memory_footprint()`와 별도 Quantization State, CUDA Allocator 메모리는 구분한다.
이 설치 버전의 단일 GPU 객체에는 `hf_device_map` 속성이 없었으므로 실제 Parameter/Buffer
Device를 검사했다. CPU/Disk Weight Offloading은 없다.

전체 Checkpoint·Processor·Template·설치 Source Hash와 Layer별 목록은
[model_identity.json](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/model_identity.json)에 있다.

## 동결 입력과 Prompt

기존 원본 Main RGB **3장, 후보3개**만 사용했다. 추가 C2·Episode 검색·새 Simulator 영상은
없다. 후보는 기존 Manifest의 P call133/rank1/query3, N call153/rank0/query0,
C1 call333/rank0/query0이다. P의 최종 Consumer 결과가 Null이어도 원래 Raw Peg 후보를
그대로 사용했다.

모든 요청은 **①원본128×128 RGB ②청록색1Pixel 후보 Box 표시 이미지 ③각 변10% 여백 Crop**을
동일 순서로 받았다. 원본과 Crop의 색·형상은 수정하지 않았다. Marker는 위치만 표시하고
Semantic Class를 표현하지 않는다. DINO Label·Score·Consumer·GT·Actor ID·Pose·SAM 평가값은
모델 메시지에 넣지 않았다. 같은 영문 Prompt를 그대로 사용했다.

```text
You are examining a robot manipulation scene.

Identify the main physical object inside the marked candidate region.

Choose exactly one:
PEG, BOX, ROBOT, OTHER, UNCERTAIN.

Use the entire scene and the marked region.
Do not assume the candidate is correct.
If the object identity is ambiguous, answer UNCERTAIN.

Return only JSON:
{
  "label": "...",
  "evidence": "one short sentence"
}
```

이미지 순서 설명 외 추가 규칙은 없다. `Qwen3VLProcessor`에 min/max pixels65536을
고정하고 종횡비를 유지했다. 원본/표시 이미지의 Grid는 `[1,16,16]`으로 각각256×256이며,
Crop은 Patch 정수 반올림 때문에 P192×352 / N256×288 / C1 224×320이다.
서로 다른 해상도를 Sweep한 것이 아니라 동일한 Resize 정책의 결과다.
`enable_thinking=False`가 실제 Template에서 빈 Thinking 종료 구문을 만드는 것을 확인했다.
Generation은 `do_sample=False`, `max_new_tokens=64`, `use_cache=True`, Batch1이다.

[vlm_input_manifest.json](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_input_manifest.json)에
Prompt SHA, 원래 Box/Crop 좌표와 실제 입력 PNG9개의 SHA를 보존했다.

## 실제 응답과 사후 EXACT_GT

**세 원시 응답을 모두 저장하고 파일 SHA를 고정한 뒤 EXACT_GT Actor Segmentation을 열었다.**
GT는 입력/후보 재검색에 사용하지 않았다. 기대 Actor Label과 후보 Rectangle의 Peg/Box
Pixel 수를 대조했고 기존 Temporary SAM2 결과는 저장 자료에서만 읽었다.

| 사례 | 기존 DINO Score / Consumer | GT Actor: Peg/Box Pixel | 실제 응답 안의 Label | 엄격한 결과 / 내용 진단 |
|---|---|---|---|---|
| P 0355/t0 | .291640937 / Null 폐기 | **PEG**:530/9 | **OTHER** | `INVALID_OUTPUT` / `WRONG_CLASS` |
| N 0355/t30 | .448092550 / 늦은 Peg 승인 | **BOX**:33/899 | **BOX** | `INVALID_OUTPUT` / 내용상 `CORRECT` |
| C1 0645/t0 | .481278419 / 정상 Peg 승인 | **PEG**:849/0 | **OTHER** | `INVALID_OUTPUT` / `WRONG_CLASS` |

실제 Evidence 문장은 다음과 같다.

- P: “The object inside the marked region is a magnet, which is neither a peg nor a box.”
- N: “The candidate crop shows a brown cardboard box with a square cutout on its front face.”
- C1: “The object inside the marked region is a two-colored block, which is neither a peg nor a box.”

모든 응답에는 시작의 Markdown JSON Fence와 끝 Fence가 있다. 따라서 `json.loads(raw_text)`는
실패하며 **승인 가능한 출력0개**다. Fence 내부의 내용을 읽은 결과는 별도의 **사후 진단 열**에
저장했고, Format 오류를 자동 교정해 Accepted 검출이나 새 Consumer Rule로 바꾸지 않았다.
이번 실패는 Format만의 문제가 아니다. Fence를 무시하고 내용을 읽어도 실제 Peg 두 사례가
OTHER여서 필수 의미 조건을 충족하지 못한다.

기존 Temporary SAM2의 Peg Mask IoU는 P **.963370**, N **.001504**, C1 **.989399**다.
N의 기존 Mask898 Pixel 중896이 Box였다는 선행 결과도 유지된다. 이번에는 SAM2를 새로
실행하지 않았고 이 결과를 VLM에 알려주지 않았다.

[CSV](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_semantic_probe.csv)는
엄격한 `INVALID_OUTPUT`과 별도 `semantic_content_comparison`을 구분한다.
[원시 응답](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_raw_responses.json)은
생성 Token ID까지 그대로 보존했다. Warm 반복에서도 각 후보의 응답 Text는 첫 출력과 같았다.
표본3개로 모집단 정확도를 계산하지 않는다.

## VLM 단독 Load·상주 요청 비용

첫 실행은 NF4 로드 뒤 계측 코드의 `model.hf_device_map` 접근에서 `AttributeError`로
종료됐다. 이미지 추론 전이었으며 지원되지 않는 모델/Kernel 문제는 아니었다.
사용자가 **추가 로드1회**를 승인한 뒤 선택적 속성 조회로 수정해 재실행했다.
총 모델 로드는 **2회**, 성공 Process 내부는 **로드1회·재로드0회·요청7회**다.
첫 실패 Source/로그/메모리는 `attempt1/`에 보존했다. 승인 없는 추가 로드는 없다.

| 성공 Process 측정 | 값 |
|---|---:|
| `from_pretrained` 시작→CUDA 완료 | **3547.79ms** |
| 로드 후 CUDA Allocation / Reservation | **7.349 / 7.406GiB** |
| 로드 후 Device Free | **15.854GiB** |
| 로드 후 CPU RSS / 50ms 표본 Peak RSS | **.978 / 18.410GiB** |
| 반복 요청 중 Peak CUDA Allocation / Reservation | **7.555 / 7.629GiB** |
| 종료 직전 CUDA Allocation / Reservation | **7.357 / 7.629GiB** |
| 종료 직전 CPU RSS | **2.091GiB** |

로드 시간은 다운로드와 Processor 준비를 제외한 모델 File→CPU→GPU 범위다.
직전에 Checkpoint Hash를 읽었으므로 **OS File Cache가 Warm한 Load**이며 Cold Disk 성능이
아니다. 첫 실패 Process의 Load3612.80ms는 별도 기록으로 남겼다.

의미 평가3회 후 동일 P로 Warmup1회, P/N/C1 Warm Timing 각1회를 실행했다.
각 요청마다 이미지와 Prompt를 다시 Processor에 넣고 새 Generation State/Cache를 사용했다.
모델 객체 ID와 GPU Parameter Device가 요청7회 내내 같았다. Weight 상주는 매번의 Vision
Encoding/Prefill을 없애지 않는다.

| Warm 요청 | Image/Prompt Preprocess | Vision CUDA | 첫 Forward: Vision+Prefill CUDA | 후속 Decode Forward CUDA 합 | End-to-end | 출력 Token |
|---|---:|---:|---:|---:|---:|---:|
| P | 5.33ms | 36.33ms | 221.70ms | 2242.85ms | **2493.71ms** |45|
| N | 6.08ms | 36.58ms | 219.16ms | 2317.50ms | **2561.27ms** |44|
| C1 | 6.54ms | 36.02ms | 220.86ms | 2436.81ms | **2683.83ms** |47|

Warm End-to-end p50/p95/max는 **2561.27 / 2671.57 / 2683.83ms**다.
3회 표본의 기술 통계이며 장기 Tail Latency 추정이 아니다. End-to-end는 실제 PNG 열기·
Processor·H2D·Generation·Text Decode까지의 Wall Time이고, 저장/GC는 제외했다.
CUDA Synchronization으로 요청 완료를 확인했다. CUDA Event 값은 Module Forward 구간의
경과시간이며 Generation Loop·Sampling/Host Overhead 전체를 포함한 Decode Wall Time과는
다르다. Vision은 첫 Forward 안에 포함되므로 두 열을 합산하지 않는다.

첫 의미 요청 P는3280.85ms로 Cold Kernel 비용이 포함됐다. 첫 요청 이후 Output 해제 후
Allocation은 **7,899,388,416 bytes로 모든 요청에서 동일**했다. 로드 직후 대비 약8.13MiB의
초기 증가가 있었고, 이 짧은 반복에서 누적 증가는 없었다. 장기 메모리 안정성의 증명은 아니다.

## 동시 상주 Gate·판정·보존

**기존 모델과의 동시 상주는 `NOT_EXECUTED_SEMANTIC_GATE_FAILED`다.**
실제 B0는 Grounding DINO Tiny+SAM2 Small이고, 현재 중지된 BC 재검출 경로는 DINOv2 Small도
사용한다. Gate 통과 시의 테스트 집합은 Qwen과 이 세 모델의 합집합으로 정했으나,
P/C1 실패 때문에 기존 세 모델은 로드/Forward하지 않았다. 따라서 합산 VRAM·Warm Peak·
여유 VRAM·공존 OOM·공존 Offloading·동시 상주 안정성은 **미검증**이다.
VLM 단독의 여유 메모리로 공존 성공을 추정하지 않는다.

최종 판정은 **`LOCAL_SEMANTIC_FAILURE`**다. t0 Peg를 PEG로 판단하고 t30 Box를 BOX로
구별하는 핵심 조건을 충족하지 못했으며 정상 C1도 보존하지 못했다.
**다음 실제 Pipeline 통합 실험은 현재 결과로 진행하지 않는다.** 다른 Prompt나 고해상도,
비양자화 조건을 실험하지 않았으므로 모델 전체의 불가능성이나 원인을 확정하지 않는다.
Format만 완화하면 해결된다고도 주장하지 않는다. Production Controller/승인 규칙은 없다.

최종 Source·Raw·GT·기존 Checkpoint·Stop Stub/보존 Original·Freeze **1342개 파일**의
시작/종료 SHA가 일치했고, 기존 연구 환경 Package 목록도 동일했다. 새 Qwen Checkpoint도
최종 전수 Hash가 같다. 공통 `ExperimentLogger`를 W&B disabled로 재사용했으며 성공
실험 실행 UUID는 **`6baf310f605043cdba6a381aa164eca6`**다. Logger의 실행 성공은 Semantic
정답 성공을 뜻하지 않는다. 실패한 첫 Process의 Logger도 그대로 보존했다.

DINO/SAM2/DINOv2 새 Forward0, Consumer 변경0, Simulator Reset/Render/Rollout0,
Graph 재캐시0, GNN/GRU 학습0, Internal/Final 재개0, Git Commit/외부 게시0이다.
원래 Stage4A-2 Runner0을 확인했다. macOS MPS·Simulator·Renderer는 실행하지 않았다.
새 Production 코드가 없어 Simulator/Training 테스트는 실행하지 않았다. 대신 실제
멀티모달 요청·응답/CSV Readback·GT Barrier·원본/환경 Hash·입력 RGB 동일성·PNG를 검증했다.

독립 원격 Root는 `/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/`다.
필수8개 산출물과 실행 Source/로그/Logger는 로컬에도 복사했다. 모델/독립 환경은 원격에
남겼으며 서비스는 배포하지 않았다. 실험 Process 종료 뒤 GPU 메모리가 해제됐다.

- [Load·요청별 Latency/VRAM](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_latency_vram.json)
- [동시 상주 Gate Audit](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/model_residency_audit.json)
- [보존·Source·최종 Readback](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/verification.json)
- [입력·후보·실제 응답 비교 PNG](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_candidate_comparison.png)

![동일 입력 규칙과 VLM 결과](../../outputs/task10i/stage4a2/qwen35_verifier_20261102_v1/vlm_candidate_comparison.png)

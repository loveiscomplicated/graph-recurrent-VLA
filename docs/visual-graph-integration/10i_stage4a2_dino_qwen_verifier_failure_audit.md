# Task 10-I Stage4A-2 — Qwen 검사기 역할 혼동·정답 거부 진단

**이전 프레임·답변을 누적하는 구현은 확인되지 않았다.** 실제 실행 Source, 설치된 Transformers의 캐시 생성/M-RoPE 경로, 요청 251개의 실행 기록을 확인했다. 매 후보마다 현재 프레임의 이미지 3장과 같은 Prompt를 새 user 메시지 하나로 입력한다. 모델 가중치만 재사용하며, 앞선 답변이나 요청의 KV/recurrent cache를 다음 요청에 전달하지 않는다.

**Recall 하락의 직접 원인은 후기 Wrist 정답 Peg 두 개를 `BOX`로 분류해 제거한 것이다.** 역할 분류 오류는 확인됐지만, 어떤 시각 단서나 Prompt 문장이 원인인지는 아직 확정하지 않았다. 초기와 후기의 영상 차이, 단일 프레임에서 기능적 역할을 판단하는 한계, 세 이미지 사이의 후보 대응 해석을 원인 후보로 구분한다.

## 1. 범위와 재현

기존 [DINO → Qwen 검사기 파일럿](10i_stage4a2_dino_qwen_candidate_verifier.md)의 동결 산출물만 사후 분석했다. Parent 4개, 512 native Camera-Frame 16개, 후보 251개이며 새 일반화 평가가 아니다. 기존 모델은 Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 전체 BF16, SDPA, greedy, thinking=False였다.

- 분석 Source: [audit.py](../../graph/experiments/dino_qwen_verifier_failure_audit_20261102_v1/audit.py)
- 재현: 저장소 루트에서 `.venv/bin/python graph/experiments/dino_qwen_verifier_failure_audit_20261102_v1/audit.py`
- 결과: [audit_metrics.json](../../outputs/task10i/stage4a2/dino_qwen_verifier_failure_audit_20261102_v1/audit_metrics.json)
- 설치 패키지 증거: [installed_cache_audit.json](../../outputs/task10i/stage4a2/dino_qwen_verifier_failure_audit_20261102_v1/installed_cache_audit.json)

새 Qwen Load/Generation, DINO/SAM2 Forward, Simulator 호출은 모두 0회다. Vast.ai의 기존 Python 환경에서는 설치 코드 조회와 가중치 없는 CPU 캐시 준비 Probe만 실행했다. CUDA는 초기화하지 않았다. macOS에서는 파일 분석·이미지 검사를 수행했다. Production, 기존 Prompt, 선택 정책, PAUSED_SCOPE_REVIEW 및 Stop Stub은 변경하지 않았다.

## 2. 컨텍스트가 누적되는가?

실행 [infer.py](../../graph/experiments/dino_qwen_candidate_verifier_20261102_v1/infer.py)의 루프는 각 후보마다 다음과 같이 새 입력을 만든다.

```python
content = [preamble] + [current_full_rgb, current_marked_rgb, current_crop] + [prompt]
x = processor.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
)
out = model.generate(**x, max_new_tokens=16, do_sample=False, use_cache=True)
```

위 코드는 구조를 보여 주는 축약이다. 실제 Source는 이미지·텍스트를 정식 content dictionary로 구성한다. 이전 `assistant` 응답을 메시지에 추가하거나 이전 `out`을 다음 `generate`에 전달하는 코드가 없다. 기록용 `raw.append(...)` 목록은 결과 파일 저장에만 사용한다. GPU allocator의 reserved memory 재사용도 대화 이력 재사용을 뜻하지 않는다.

`use_cache=True`는 이 요청의 Prefill/Decode 안에서 attention KV와 linear-attention recurrent state를 재사용하는 설정이다. 설치된 Transformers 5.19.0의 `generation/utils.py`에서 사용자 제공 캐시가 없고 기본 dynamic 설정이면 `_prepare_cache_for_generation`이 새 `DynamicCache`를 만든다. 실제 Qwen config에서 유도한 기본 설정도 `cache_implementation=None`, `use_cache=True`, `return_dict_in_generate=False`였다. 별도 generation_config.json은 없었다.

Qwen에는 모델 속성 `rope_deltas`가 있으나 이것은 이전 대화 자체가 아니다. 설치된 `modeling_qwen3_5.py`는 이미지 입력이 있고 새 캐시 길이가 0이면 `get_rope_index(...)`를 실행해 현재 입력의 위치 인코딩을 다시 계산한다. 이전 요청의 delta를 무조건 이어 쓰는 경로가 아니다. 해당 줄과 SHA를 증거 JSON에 저장했다.

추가로 실제 Qwen config와 설치된 `GenerationMixin._prepare_cache_for_generation`을 이용해 CPU에서 캐시 준비를 두 번 호출했다. 서로 다른 `DynamicCache` 인스턴스이며 두 캐시의 시작 sequence length는 모두 0이었다. 이 Probe는 모델을 생성하거나 가중치를 읽지 않았다.

실행 기록도 누적 가설을 지지하지 않는다.

| 검사 | 관측 |
|---|---|
| 요청별 input tokens | 1,183–1,228, 이전 응답 누적에 따른 증가 없음 |
| 요청별 이미지 | 현재 scene/marked/crop 3장 |
| Vision module 호출 | 각 요청 1회, 현재 이미지 3장을 함께 처리 |
| 요청 순서 | 고정 Hash 순서로 초기/후기·Camera·후보를 교차 실행 |
| 회귀 후보 실행 index, 0-based | 43, 109; 마지막에만 발생한 현상 아님 |
| 동일 이미지 SHA 3개 조합이 반복된 경우 | 10개 중복 그룹, Label 불일치 0개 |

원래 실행에서는 최초 Forward의 cache identity/length를 직접 Hook으로 저장하지 않았다. 따라서 위 결론은 Source 경로·기본 설정·CPU Probe·기록을 종합한 것이며, 원 실행의 모든 내부 Tensor를 계측한 결과로 표현하지 않는다. 현재 증거에서는 **대화 누적이나 요청 사이 캐시 전달을 실패 원인으로 볼 근거가 없다.**

## 3. Recall 10/16 → 8/16의 직접 원인

비교는 동일한 PEG Query의 원시 후보 풀이다. raw_top1은 최고 DINO score를 선택하고, Primary는 Qwen이 `PEG`라고 답한 후보만 남긴 뒤 최고 DINO score를 선택한다. 정답 여부는 Box IoU≥0.5다.

| Parent / 시점 / Camera | DINO Top-1 Peg IoU | Qwen | 후보 내부 foreground의 Peg 비율 | 검사 후 Peg |
|---|---:|---|---:|---|
| 0275 / t80 / Wrist | 0.9830 | BOX | 99.62% | null, IoU 0 |
| 0305 / t80 / Wrist | 0.9719 | BOX | 98.79% | null, IoU 0 |

Foreground 분모는 해당 후보 직사각형 안의 Peg+Box+Robot Actor pixels다. Background까지 포함한 전체 직사각형의 Peg 비율도 각각 98.22%, 96.75%였다. 0275에는 실제 Box pixel이 0개였고, 0305에는 Peg 97,204px 대 Box 386px였다. 따라서 두 사례를 단순한 mixed-object 후보나 IoU 경계값 문제로 설명할 수 없다. Qwen이 실제 Peg를 BOX 역할로 잘못 분류했다.

0275의 PEG Query 후보 8개는 전부 BOX였다. 0305의 9개는 BOX 8개·ROBOT 1개였다. 검사 후 정답 후보를 모두 제거했으므로 후속 순위 선택으로 회복할 수 없었다. 기존 정책에서 막혔던 두 정답이 raw_top1에서는 살아났지만 Qwen gate가 다시 제거한 것이다. 새 좌표 생성·JSON 파싱·좌표 보정은 이 회귀와 관계없다.

## 4. 실패가 시점에 집중되는가?

| 구분 | IoU≥0.5인 PEG Query 후보 수 | Qwen PEG | Qwen BOX |
|---|---:|---:|---:|
| 초기 / Main | 10 | 10 | 0 |
| 초기 / Wrist | 6 | 6 | 0 |
| 후기 / Main | 5 | 0 | 5 |
| 후기 / Wrist | 9 | 0 | 9 |

초기 정답 후보 16/16은 PEG, 후기 정답 후보 14/14는 BOX였다. 이는 중복 후보를 포함한 후보 단위 분석이며 독립 Parent 30개가 아니다. 단순히 원시 Top-1 두 개만 우연히 거부한 현상보다 넓은 후기 역할 혼동이 있다.

전체 후기 후보 149개에서는 BOX 138개·ROBOT 11개·PEG 0개였다. Actor 구성 기반 PEG 우세 후보도 초기 32개에서는 PEG 28개·BOX 4개였으나 후기 49개에서는 전부 BOX였다. ROBOT 우세 후보는 초기 28개 중 BOX 9개, 후기 51개 중 BOX 40개였다. 현재 검사기는 후기 장면에서 BOX 쪽으로 강하게 편향된다. 다만 Actor pixel 우세는 intended-object Label의 완벽한 정답을 보장하는 지표가 아니므로, 순수한 정답 Peg 두 사례와 함께 해석해야 한다.

## 5. 이미지 관찰과 원인 가설

![두 회귀 Parent의 초기·후기 입력 비교](../../outputs/task10i/stage4a2/dino_qwen_verifier_failure_audit_20261102_v1/rejected_peg_inputs.png)

위에서부터 0275 초기/후기, 0305 초기/후기다. 열은 원본 전체 RGB, 실제 표시 입력, 실제 Crop 순서다. 그림의 표시 크기만 축소했으며 모델에 제공했던 원본 파일을 사용했다. 결과를 기준으로 고른 실패 진단 이미지이며 대표 표본이나 새 성능 평가로 사용하지 않는다.

**직접 관찰:** 초기에는 Peg가 작고 비스듬하며 물체의 외형과 주변 공간이 비교적 함께 보인다. 후기에는 Peg가 Wrist 화면의 큰 부분을 차지하고 세로로 길게 보이며 화면 위·아래로 잘린다. Crop도 화면 밖의 물체 끝이나 추가 작업 문맥을 복원하지 못한다. 같은 Parent의 초기와 후기 사이에서 크기·방향·가시 윤곽이 크게 바뀐다. 초기에도 Gripper가 가까이 보이므로 단순히 “Gripper가 보이면 실패한다”는 설명은 충분하지 않다.

**유력하지만 미확정인 가설:**

1. **현재 한 프레임의 외형에 의존한 역할 추론.** 매 요청은 처음 보는 장면이다. 초기에서 확인한 Peg의 정체·이동 이력을 후기 요청에 전달하지 않는다. 기능적 정의를 문장으로 줘도 큰 부분 영상에서 삽입 부품과 받는 Fixture를 분리하지 못했을 수 있다. 대화가 과도하게 쌓이는 문제와 반대로, 물체 정체의 시간적 연결은 제공하지 않았다. 과거 RGB를 추가하면 개선되는지는 미측정이다.
2. **가려지거나 잘린 상태의 역할 지속성에 대한 Prompt 부족.** 기존 Prompt는 역할 정의와 색·모양에만 의존하지 말라는 지시를 포함하지만, 이미 파지한 부품이나 화면 밖으로 이어지는 부품도 같은 PEG라는 설명은 명시적이지 않다. `separate physical component`라는 표현이 특정 상태에서 오해될 가능성도 있다. 그러나 초기에도 Gripper가 가까이 있으므로 이 문구만을 원인으로 단정하지 않는다. 특정 문장 변경의 인과 효과는 시험하지 않았다.
3. **다중 이미지에서 표시 후보와 문맥을 대응하는 어려움.** 원본/표시본/Crop 세 장의 순서와 SHA는 맞지만, 모델이 실제로 표시된 부품을 우선했는지, 전체 장면의 받는 Fixture에 끌렸는지는 한 단어 응답만으로 알 수 없다. 후기 ROBOT→BOX 오류도 있어 후보 대응 실패 가능성을 배제할 수 없다. Attention 해석이나 입력 구성의 분리 실험 없이 확정할 수 없다.
4. **모델의 실제 의미 분류 한계.** 형식 준수 251/251이 의미 이해를 보장하지 않는다. 같은 역할 Prompt로 후기의 거의 순수한 Peg 후보를 반복해서 BOX로 분류한 것은 현재 모델·입력 구성의 실제 실패다. BF16에서도 발생했으므로 이번 실패를 4-bit 양자화 탓으로 돌릴 수 없다. 그렇다고 Qwen 모델 전체의 성능 한계라고 일반화할 수도 없다.

좌표·Crop 전달 오류도 점검했다. 원본 산출물 537개의 SHA를 전후 확인했고 요청 이미지 SHA 753건을 검사했다. Crop 251개 모두 기록된 원본 RGB 영역과 pixel-equal이었다. 로컬 검사 Source SHA는 원격 실제 실행 Source와 같았다. 파일 연결·Crop 누락·기존 응답 재사용을 뒷받침하는 증거는 없다. 이 검사는 Processor 내부의 모든 시각적 정보 손실이나 모델 해석 오류까지 부정하는 검사는 아니다.

## 6. 결론과 다음 검증의 우선순위

확정된 실패 사슬은 **현재 후기 RGB → Qwen의 PEG→BOX 오분류 → 역할 gate가 정답 후보 제거 → null**이다. 요청 사이 컨텍스트 누적을 고치는 것으로 해결될 문제는 현재 확인되지 않았다. 반면 초기 물체 정체를 전달하지 않는 단일 프레임 방식과 후기 영상의 외형 변화는 명확한 설계 조건이다.

다음 원인 분리 실험은 이 두 회귀 사례와 대응 초기 사례를 진단용으로 고정하고, 모델·정밀도·해상도를 유지한 채 입력 구성을 하나씩 바꾸는 방식이 적절하다. 우선 세 이미지 방식 대비 명확히 표시한 단일 전체 이미지의 후보 대응을 확인하고, 별도로 파지·부분 가시성에서도 역할이 유지됨을 명시한 Prompt의 효과를 확인할 수 있다. 시간 정보 효과는 과거 RGB만 사용하는 별도 조건으로 분리해야 한다. 이 사례에 맞춘 개선은 개발 결과이며, 성능 주장은 이후 미사용 Parent에서 평가해야 한다.

이번 진단에서는 위 추가 추론이나 Prompt 변경을 실행하지 않았다. SAM2 초기화·Tracking·Graph·Controller·Closed-loop 성능에 관한 새 결론도 없다.

# Task10-G seed98202: 첫 프레임 peg 검출 실패 원인

대상: Final / C / step011487 / seed98202. 분석일: 2026-10-31.

## 결론

첫 프레임의 peg는 실제 RGB에 보이며, **Grounding DINO도 peg를 포함하는 후보 박스를 생성했다. 그러나 그 후보 점수가 초기화 규칙의 `MISSING_SCORE=0.3`보다 낮아서 버려졌다.** 첫 프레임의 실패 지점은 depth/self filtering이나 wrist gate 이전인 **DINO 후보 선택 → SAM2 초기화** 단계다.

첫 프레임 base_camera의 peg 포함 후보는 score **0.2312659** 및 더 좁은 박스의 **0.2277020**이었다. 둘 다 SAM2에 전달되지 않았고, base_camera peg raw mask는 재현한 t=0–30 전체에서 0픽셀이었다. “모델이 후보 자체를 전혀 못 찾았다”는 해석과는 구분된다.

## 원본과 동일한 환경에서의 재현

원래 Vast 서버의 frozen source와 원래 DINO/SAM2 checkpoint를 사용했다. 학습 및 정책 추론은 실행하지 않고, seed98202를 reset한 뒤 원본 trace에 저장된 action을 t=0–29에 그대로 적용하여 t=0–30 관측/검출을 재생했다.

| 검증 | 결과 |
| --- | --- |
| 초기 simulator/controller/RNG snapshot SHA | 원본과 exact 일치 |
| 초기 Graph21/Edge4/Robot76 | 저장된 모든 배열 exact 일치 |
| Wrist gate 전체 필드 | 31/31 exact 일치 |
| 최종 graph의 peg/box 존재 여부 | 31/31 exact 일치 |
| Wrist raw peg 픽셀 수 | 31/31 exact 일치 |
| 저장된 post-action peg/TCP pose와의 최대 차이 | 모두 0 |
| 실행 backend | CPU PhysX + CUDA SAPIEN renderer / CUDA vision |
| 장치 | NVIDIA RTX 3090, Torch 2.14.1+cu130 |

이는 MP4 압축 이미지에 다시 검출한 근사 실험이 아니다. 원래 simulator RGB를 동일한 reset/action 조건으로 재생했다. 원래 실행은 후보 score를 저장하지 않았으므로 아래 점수는 이 exact 재현에서 새로 기록한 결과다.

## 첫 프레임에서 탈락한 경로

[검출 박스 그림](../../outputs/task10g/diagnostics/seed98202_initialization_v1/initialization_evidence.png) · [원본 크기 RGB](../../outputs/task10g/diagnostics/seed98202_initialization_v1/base_camera_rgb_t000.png)

현재 visual 경로는 peg에 `"red and white block"`, box에 `"cardboard box"`를 사용한다. DINO postprocessing은 score 0.1 이상 후보를 반환하지만, 초기화에서는 다음 규칙을 추가 적용한다.

1. 박스 면적이 전체 영상의 25% 이상이면 제거한다.
2. 면적 조건을 통과한 점수순 상위 5개 후보를 검토한다.
3. Peg와 Box를 함께 선택하되, 서로 IoU가 0.3 이상인 두 박스는 함께 선택하지 않는다.
4. 각 물체의 미검출 선택에도 점수 0.3을 부여하고 합계 점수가 가장 큰 조합을 택한다. 다른 조건이 같으면 물체 후보는 0.3을 **초과**해야 미검출을 이긴다.

| base_camera t=0 후보 | Score | 영상 대비 면적 | 처리 |
| --- | ---: | ---: | --- |
| 화면 아래의 넓은 영역 | 0.419002 | 60.779% | 면적 제한으로 제거 |
| 실제 peg를 포함하는 박스 `[0.04,84.50,58.98,128.01]` | 0.231266 | 15.652% | 면적 통과, 미검출 점수 0.3보다 낮음 |
| 실제 peg를 더 좁게 감싼 박스 `[11.45,84.49,58.76,127.99]` | 0.227702 | 12.560% | 면적 통과, 미검출 점수 0.3보다 낮음 |
| Peg의 빨간 끝부분 부근 | 0.222371 | 5.642% | 면적 통과, 미검출 점수 0.3보다 낮음 |
| Box 프롬프트의 실제 box 후보 | 0.616189 | 5.816% | box로 선택 |

실제 선택 점수는 `peg 미검출 + box = 0.300000 + 0.616189 = 0.916189`이다. `peg 후보 + box = 0.231266 + 0.616189 = 0.847455`보다 높다. 두 박스의 겹침 때문에 실제 peg 후보가 탈락한 것이 아니라, 이 경우에는 점수 규칙이 직접적인 원인이다. 저장 후보를 기존 `assign()`에 다시 넣어 `(peg=None, box=선택)` 결과도 확인했다.

영상은 camera별 384×384로 확대하지만 **모델의 실제 입력은 128×128 RGB**다. 첫 프레임 base_camera에서 peg의 흰색 끝은 영상 하단 밖으로 일부 잘려 있다. 낮은 해상도, 잘린 모습, 프롬프트와의 일치도는 낮은 DINO 점수의 가능한 배경이지만 각각의 영향은 이번 진단에서 분리 실험하지 않았다. 확정된 직접 원인은 위 점수/면적 규칙이다.

첫 프레임 hand_camera도 동일하게 peg 미검출이었다. 면적을 통과한 최고 후보 score는 0.189182이고, peg가 있는 왼쪽 아래 영역을 포함하는 후보는 0.187556이었다. 높은 score 0.370424/0.315284 후보는 각각 화면의 99.962%/90.764%를 덮어 면적 제한으로 제거됐다.

## 이후 왜 그리퍼가 peg가 되었는가

미검출 물체는 5스텝 간격으로 스캔한다. 후속 초기화는 score≥0.3 후보가 두 연속 스캔에서 IoU≥0.5로 이어져야 한다.

- **base_camera:** 실제 peg 후보는 첫 31프레임의 모든 스캔에서 0.3 미만이다. t=25/30에는 실제 box 영역을 peg 프롬프트로 검출한 score 0.357/0.367 후보가 있으나, 이미 선택된 box와 겹쳐 제외된다. 따라서 peg 초기화가 이루어지지 않는다.
- **hand_camera:** t=25의 그리퍼 포함 박스 score는 **0.413580**, t=30은 **0.419639**이다. 두 박스의 IoU는 **0.997082**로, 같은 오검출이 반복되어 일관성 조건을 통과한다. 실제 adapter의 `found['peg']`에 `frame=30`, `decision_observation=30`, `prefix_length=31`로 기록되는 것을 확인했다.
- 그 박스를 SAM2에 peg ID로 넘기면서 t=30에 1,577픽셀의 그리퍼 마스크가 생긴다. 이는 처음 올바른 peg를 잡았다가 추적이 그리퍼로 이동한 drift가 아니라, **그리퍼 박스로 peg 추적을 처음 시작한 초기 오검출**이다.
- 현재 adapter는 `found`에 등록된 물체를 detector로 다시 초기화하지 않는다. Geometry gate가 해당 마스크를 거절해도 adapter의 `found`/추적 상태를 해제하지 않는다. 따라서 gate가 차단한 잘못된 identity를 vision 단계가 자동 교정하는 경로는 없다.

이 재현으로 [앞선 영상/trace 진단](10g_seed98202_peg_mask_diagnosis.md)에서 미확정이었던 “초기 detector 오검출인가, SAM2 drift인가”를 **초기 detector 오검출**로 좁혔다. 전체 에피소드의 정책 실패 원인이 이것 하나뿐이라는 인과 주장은 하지 않는다.

## 재현 자료와 변경 범위

- [요약](../../outputs/task10g/diagnostics/seed98202_initialization_v1/summary.json), [전체 detector 호출/후보 점수](../../outputs/task10g/diagnostics/seed98202_initialization_v1/detector_calls.json), [31스텝 재현 기록과 found 상태](../../outputs/task10g/diagnostics/seed98202_initialization_v1/replay.json)
- [샘플 raw RGB와 bool masks](../../outputs/task10g/diagnostics/seed98202_initialization_v1/sampled_rgb_masks.npz), [실행한 진단 코드](../../outputs/task10g/diagnostics/seed98202_initialization_v1/executed_reproduce_initialization.py), [실행 로그](../../outputs/task10g/diagnostics/seed98202_initialization_v1/execution.log)
- [초기화 규칙](../../visual/runtime/initialization.py), [DINO](../../visual/runtime/grounding.py), [online adapter](../../graph/integration_baseline/online_vision_adapter.py). 이 로컬 파일들은 저장된 frozen-source hash와 일치함을 확인했다.

공유/모델/정책/threshold 코드는 변경하지 않았다. 새 파일은 진단 출력과 이 보고서뿐이다. 공통 `ExperimentLogger`를 W&B disabled로 재사용했고, 외부 artifact 게시를 하지 않았다. 실제 재생은 원래 Linux/CUDA 서버에서 실행했다. macOS CPU/MPS에서는 inference나 simulator를 실행하지 않았다.

재현 계산과 검증 자료 저장은 완료됐지만, 마지막 artifact 등록에서 진단 스크립트가 미지원 kind 문자열을 전달해 logger가 실패 상태로 종료됐다. 이 오류는 계산 이후이며 위 exact 비교 결과에는 영향을 주지 않는다. 원래 실행/로그를 보존하고, 로컬 집계 단계에서 지원되는 kind와 별도 `ExperimentLogger` 실행으로 산출물 등록을 마쳤다. 같은 31프레임을 불필요하게 재실행하지 않았다.

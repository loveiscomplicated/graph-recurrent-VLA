# Task 10-I Stage 4A-2 — Final Point Budget Sensitivity

**판정: `BUDGET_LIMIT_CONFIRMED`. Stage 4A-2 상태는 `PAUSED_SCOPE_REVIEW` 유지.**

t86 BC의 Main End-band 82개는 B128에서 35개만 남았지만 B256에서 **82개 모두**
보존됐다. Ends Coverage는 **0.426829 → 1.000000**, 정상 표면 Coverage는
**0.755289 → 0.986150**으로 동시에 개선됐다. t6도 Budget 증가로 개선되지만
B∞에서도 Ends Coverage는 **0.620930**이다. 최종 cap의 기여는 확인됐으며,
관측 GT 전체 표면을 복구하는 데에는 기존 Dedup Pool의 한계도 남는다.

## 범위와 재현

- Episode: `10i_stage3_10i_s1_0645_full_task`, **t6/t86만** 사용. 정상 대조 0개.
- 입력: 저장된 `B0`/`BC_WB_HANDOFF` Mask, 동일 RGB-D·Calibration·Robot·GT.
- 실행 Source: 기존 동결 후보의 `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source`.
- 네 조합 모두 기존 B128의 XYZ, Object ID, 병합 Source Group, Gate 및
  `node_input/node_type/edge_index/edge_attr/edge_type`가 저장 결과와 **동일**했다.
  이 검사를 네 조합 모두 통과한 뒤 Sweep을 시작했다. 기존 저장 Scalar도 일치했다.
- 기존 `CurrentGeometry.build`의 `budget(pools['SF'],128)` 직전 Pool을 관찰·복사했다.
  B128/B256은 원래 `budget` → `select(anchors=False)` → `farthest_rows`를 호출했다.
  B∞는 Pool을 순서·Feature 변경 없이 전부 전달했다. Pool≤Budget이면 FPS를 호출하지 않는다.
- XYZ, Source Pixel Identity, Camera, Point Feature, Shape-first Metadata,
  Dedup Source Group, 전체 순서 포함 Records의 Hash가 세 조건에서 모두 일치했다.
  선택 결과는 **B128 ⊆ B256 ⊆ B∞**이며 선택/탈락 ID가 입력 Pool을 정확히 분할한다.

원래 `PlanA.graph → build_graph(point_cap=None)`에는 추가 cap이 없다.
단, `CurrentGeometry.validate_graph`의 마지막 검사는 물체별 128개 초과를 거부한다.
Production 파일은 수정하지 않고 독립 Wrapper에서 **그 검사만 제외한 동일 Validator**로
Node21/Edge4, 유한값, Edge Endpoint, Robot Node, Current-only Feature를 검사했다.
128개 초과 Graph는 원래 Validator의 `point cap` 거부도 그대로 기록했다.
전체 Model/Loader를 실행·변경하지 않았으므로 기존 GNN에서 사용 가능하다는 뜻은 아니다.

## 동일 Dedup 입력 Pool

아래는 Peg 대표점 수다. `M/W`는 대표 측정 Source의 Main/Wrist이며,
괄호는 정상 Peg 대표점이다. End-band는 정상 대표점의 L/R 수다.

| Frame | 입력 | Pool 전체 | M/W (정상 M/W) | Main End L/R | Wrist End L/R |
|---|---|---:|---:|---:|---:|
| t6 | B0 | 201 | 201/0 (197/0) | 34/19 | 0/0 |
| t6 | BC | 354 | 199/155 (195/154) | 33/19 | 8/18 |
| t86 | B0 | 114 | 114/0 (106/0) | 82/0 | 0/0 |
| t86 | BC | 261 | 112/149 (104/149) | 82/0 | 0/0 |

병합된 고유 Pixel Source와 대표점은 구분한다. BC t6 입력의 Main/Wrist 고유 Source는
201/189개이고 정상 End Source L/R은 34/19 및 10/22개다. BC t86은
114/218개이며 정상 End Source는 82/0 및 0/0개다. 모든 Group과 선택·탈락 ID는 Trace에 있다.

## 실제 결과

Coverage는 원래 평가 코드의 **정상 대표점으로부터 거리 <4mm인 현재 관측 GT 표본 비율**이다.
GT Mesh 최장 축의 양 끝 10% 정의와 Reference를 그대로 재사용했다.
t6 분모는 전체 2,235개, End L238/R192개이며 t86은 전체 9,170개, End L82/R0개다.
t86 Right Coverage는 모든 조건에서 **N/A**다. GT는 선택 완료 후 평가·그림에만 사용했다.

| Frame | 입력 | Budget | 선택 M/W | 정상/전체 | 오점 Robot/BG/Other | Ends Coverage | Surface Coverage | 오염률 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| t6 | B0 | 128 | 128/0 | 124/128 | 0/4/0 | .313953 | .234004 | .031250 |
| t6 | B0 | 256 | 201/0 | 197/201 | 0/4/0 | .453488 | .374497 | .019900 |
| t6 | B0 | ∞ | 201/0 | 197/201 | 0/4/0 | .453488 | .374497 | .019900 |
| t6 | BC | 128 | 79/49 | 123/128 | 0/5/0 | .239535 | .198210 | .039062 |
| t6 | BC | 256 | 145/111 | 251/256 | 0/5/0 | .434884 | .391946 | .019531 |
| t6 | BC | ∞ | 199/155 | 349/354 | 0/5/0 | .620930 | .551230 | .014124 |
| t86 | B0 | 128 | 114/0 | 106/114 | 1/6/1 | 1.000000 | .133479 | .070175 |
| t86 | B0 | 256 | 114/0 | 106/114 | 1/6/1 | 1.000000 | .133479 | .070175 |
| t86 | B0 | ∞ | 114/0 | 106/114 | 1/6/1 | 1.000000 | .133479 | .070175 |
| t86 | BC | 128 | 59/69 | 120/128 | 1/6/1 | .426829 | .755289 | .062500 |
| t86 | BC | 256 | 111/145 | 248/256 | 1/6/1 | 1.000000 | .986150 | .031250 |
| t86 | BC | ∞ | 112/149 | 253/261 | 1/6/1 | 1.000000 | .988986 | .030651 |

정상 대표점 비율은 `정상/전체`, 오염률은 그 보수이며 각 Robot/BG/Other 비율과
병합 Source의 정상 비율은 CSV의 별도 열에 저장했다.

| Frame/입력 | Budget | 정상 Main End L/R | 정상 Wrist End L/R | Left Coverage | Right Coverage |
|---|---|---:|---:|---:|---:|
| t6 BC | 128 | 13/11 | 3/5 | .214286 | .270833 |
| t6 BC | 256 | 20/16 | 6/9 | .441176 | .427083 |
| t6 BC | ∞ | 33/19 | 8/18 | .668067 | .562500 |
| t86 BC | 128 | 35/0 | 0/0 | .426829 | N/A |
| t86 BC | 256 | 82/0 | 0/0 | 1.000000 | N/A |
| t86 BC | ∞ | 82/0 | 0/0 | 1.000000 | N/A |

B0 t6의 정상 End L/R은 36점(20/16) → 53점(34/19) → 53점이며,
Left/Right Coverage는 .298319/.333333 → .512605/.380208 → 동일이다.
B0 t86은 모든 Budget에서 82/0점과 1.0/N/A를 유지한다.

## Q1–Q9 해석

1. **t86 Main End 82개 보존:** B256은 82/82개다. 정상 Main 전체는 51→103개,
   정상 Wrist는 69→145개다. Main 정상 비끝점 1개와 Wrist 정상점 4개는 B256에서 탈락한다.
2. **t86 B256 Ends 회복:** .426829→1.000000, **+57.3171%p**다.
3. **t86 B∞:** Dedup 단계의 Ends 1.0을 그대로 회복한다. Surface도 .988986이지만
   1.0은 아니다. 최종 cap 제거가 미관측/희소 표면을 새로 생성하지 않는다.
4. **t6 경향:** Ends .239535→.434884→.620930으로 개선된다. B∞는
   원래 Dedup Ends .620930을 전부 회복하나 관측 End 표본의 37.9070%는 여전히 덮지 못한다.
   이는 이 Pool의 Upstream 표현 한계이며 어떤 이전 단계가 원인인지는 이번에 분리하지 않았다.
5. **끝과 전체 표면:** 두 BC Frame 모두 함께 개선된다. 특히 기존 t86에서
   **B0→BC B128의 Surface .133479→.755289 개선과 Ends 1→.426829 손실이 동시 발생**했다는
   사실은 유지된다. B256은 이 끝 손실을 회복하면서 Surface를 더 높였다.
6. **추가점의 정상 여부:** B128→B256은 t6 B0 +73개, t6 BC +128개,
   t86 BC +128개가 모두 정상 Peg다. B256→B∞도 BC t6 +98개, t86 +5개 모두 정상이다.
   오점 절대 개수는 그대로이며 오염률 하락은 분모 증가에 따른 것이다. 오점을 정화한 결과가 아니다.
7. **B256과 B∞ 차이:** t86은 +5점으로 Ends 차이 0, Surface **+.2835%p**다.
   t6은 +98점으로 Ends **+18.6047%p**, Surface **+15.9284%p**여서 차이가 크다.
8. **Wrist 없는 B0:** t6 Pool201은 cap 해제로 Ends +13.9535%p,
   Surface +14.0492%p 개선된다. t86 Pool114는 이미 128 이하여서 효과가 없다.
   t6 BC B256 Ends .434884는 같은 Budget의 B0 .453488보다 아직 .018605 낮다.
9. **정상 대조의 예상 밖 손실:** 추가 정상 Frame은 실행하지 않아 **미평가**다.
   대상 두 Frame에서는 Budget 증가에 따른 Coverage 감소가 없었다. 이를 정상 Frame 안전성으로 일반화하지 않는다.

## Graph·실행·보존

동일 Node21/Edge4 Graph의 Node/Edge 수는 다음과 같다. Budget은 기존 호출처럼 **물체별**로
적용되므로 Box도 같은 최종 예산을 사용한다. 위 지표는 Peg만 집계했다.

| Frame/입력 | B128 Node/Edge | B256 Node/Edge | B∞ Node/Edge |
|---|---:|---:|---:|
| t6 B0 | 259/80 | 351/222 | 351/222 |
| t6 BC | 259/80 | 406/172 | 504/788 |
| t86 B0 | 236/780 | 236/780 | 236/780 |
| t86 BC | 250/480 | 378/1900 | 383/1942 |

기존 Instance58576054의 Python3.12.15/Torch2.14.1+cu130 환경에서 **CPU Geometry/Graph만**
실행했다. 원본 B128 Builder 4회, 동일 Pool의 Budget별 Graph 12회를 실행했다.
DINO/DINOv2/SAM2, Simulator Reset/Rollout, 새 GT Render, GNN/GRU Forward,
전체 Dataset Evaluation은 **각각 0회**다. CUDA 계산, GPU PhysX, SAPIEN Render,
macOS Geometry 실행, Closed-loop 20Hz·메모리·추론 속도는 평가하지 않았다.
Import 시 Vulkan ICD 경고는 있었으나 Simulator/Renderer를 생성하지 않았다.

공통 `ExperimentLogger`를 W&B disabled로 재사용했고 HF/W&B 게시는 하지 않았다.
동결 Source/Config, 저장 Raw/Mask/GT, 후보 Freeze, Stop Stub/보존 Original 등
**716개 보호 파일의 Hash 보존**과 실행 종료 시 Task 모델/Runner/Waiter 0개를 확인했다.
Production Source·Schema·Dataset·Checkpoint는 변경하지 않았고 모델을 로드하지 않았다.
Wrapper의 선택/탈락 분할, Camera·End 수, FPS 포함 관계, CSV12행과 PNG Readback을 검증했다.

이번 결과는 **최종 Budget이 이 두 Frame의 국소 손실에 기여했다는 증거**다.
t6의 잔여 표현 부족, 전체 데이터셋 일반화, 정상 대조 안전성 및 실행 비용은 남아 있다.
Production Budget 결정, 새 Sampling, Stage4B 재캐시 및 중지된 평가·학습은 시작하지 않는다.

## 산출물

독립 Root: `outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/`.

- [12조건 CSV](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/point_budget_comparison.csv)
- [Budget 입력 Hash/Source 동일성](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/budget_input_identity.json)
- [실제 Point·Source Group·선택/탈락 Trace](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/point_budget_trace.json)
- [B128 재현·보존·Readback 검증](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/verification.json)
- [독립 실행 Wrapper](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/run_budget.py)

![두 프레임 B0/BC의 실제 Point 배치와 Coverage. t86 확대도에는 전체 범위 Inset을 함께 표시했다.](../../outputs/task10i/stage4a2/point_budget_sensitivity_20261102_v1/point_budget_comparison.png)

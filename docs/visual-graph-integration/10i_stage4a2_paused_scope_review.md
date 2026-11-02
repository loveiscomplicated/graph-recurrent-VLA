# Task 10-I Stage 4A-2 — PAUSED_SCOPE_REVIEW

사용자 요청에 따라 연구 범위 검토를 위해 일시 중지했다. 현재 `BC_WB_HANDOFF`는
**부분 개선 후보이며 Development 필수 안전 기준 미통과**다. 독립 평가 일부 실행은
채택 근거가 아니다. 아래는 기존 코드·로그·저장 결과를 읽은 정리이며 새 추론,
Threshold 변경, 후보 개발, Benchmark 생성, Graph 재캐시나 학습은 하지 않았다.

## 1. 최초 DINO 후보 선택

현재 후보의 최초 초기화는 B0 그대로다. Tiny Checkpoint, Peg `red and white block` /
Box `cardboard box`, grounded postprocess threshold/text threshold .1, area<.25,
Top-5, Joint Null 대안 .3, Peg/Box exclusion IoU .3을 바꾸지 않았다.
t0는 단일 Joint Assignment이며, 이후 미발견 Slot의 원래 경로는 5-step Scan,
score≥.3와 연속 후보 IoU≥.5 확인을 유지했다.

변경한 것은 **초기 선택 이후의 별도 보정 경로**다. Wrist Peg/Main Box만 현재
Support가 불확실하면 재검출하고, score≥.2의 최대5개 SAM 후보를 별도 Memory에서
검증한 뒤 Temporal/Depth/Nonrobot/Re-ID 조건으로 승인한다. 이 경로에는 초기
Joint Null 점수 대신 명시적인 승인·거부가 있다. Main Peg 보정은 꺼져 있다.
별도 A 실험의 Wrist 초기 Phrase `red and white peg` 변경은 현재 후보에 포함하지 않았다.

유지 이유는 B0 동등성 보존과 보정 효과 분리였다. 같은 Raw Output에서 Null을
무조건 낮추면 Wrist의 Actor-supported 선택은16→24로 늘지만 unsupported non-null도
6→40으로 늘어난 기존 통제가 있었다. **이 사실은 Top-5/Area/Null 규칙 전체가
올바르다는 검증이 아니다.** 실제 Wrist 초기 SAM 실패46건 중 위치 기준 첫 Drop은
Joint Null38, Top-5 4, Area2, 위치상 좋은 Prompt 뒤 SAM 실패2였다. 초기 Consumer가
정상 후보를 버리는 문제의 직접적인 교정은 아직 미완료다.

## 2. DINOv2·SAM2 Memory·Handoff의 실제 변경과 효과

- **DINOv2:** B0 호출0을 확인했고, 보정 후보에 Small을 실제 연결했다. 현재 RGB448
  Patch Feature를 예측 Clean Mask로 Pooling/L2 정규화한다. Camera/Episode/시간/RGB
  Hash를 Cache Key에 넣고, 연속3회 Support 및 Nonrobot≥.9의 최초 Reference를 고정한다.
  과거 Reference가 있을 때 cosine≥.6, Negative Margin≥.05를 사용하며 없으면 부재를
  기록하고 Re-ID를 우회한다. 이전 전수 C_KEEP/BC_KEEP의 Mask는 모두 동일했다.
  실제 Similarity 사용1건도 Temporal 확인 실패와 겹쳐 **독립적인 오류 감소 효과는
  확인하지 못했다.** 기존 BC Trace의 Reference247개 EXACT_GT 중 오염0, 미지원16개였고,
  오염 Reference의 장기 인과효과는 미확정이다. 이를 B0의 원인으로 소급하지 않는다.
- **SAM2:** `ever_found/currently_valid` 분리, 불확실성 Hysteresis, 별도 Fresh Per-object
  Memory 후보의 검증 후 Commit, 거부 시 폐기·기존 Memory 보존을 구현했다.
  불확실하다는 이유만으로 현재 Mask를 강제 삭제하지 않는다. 역사적0121에서 Practical
  t6 보정 후 t95 Box가 TP175/FP19, Graph의 올바른 점0→126으로 복구됐다.
  부작용도 있다. 이전 Full C는0250의 부분 Peg Prompt를 오래 유지해 정상 B0 Frame100개를
  손상시켰고, 별도 실패 조건 C_SUPPRESS는 정상 Box를 오히려 악화시켰다.
- **Handoff:** 이른 실험 Anchor 뒤 원래 Adapter가 새로 초기화한 사건을 감지하고,
  현재 Nonrobot≥.9·기존 Support overlap≥.8·면적 비감소와 가능한 Re-ID를 확인해
  이미 실행 중인 원래 SAM Memory로 돌아간다. 새 강제 Reset은 아니다.
  0250 t15에서 승인됐고 이후 Wrist Mask는 B0와 같아졌다. 정상 Wrist 손상 Gate는
  통과했지만, 아래3개 안전 실패는 남았다. 전체 안전성 확보로 해석하지 않는다.

## 3. Development 필수 안전 기준 미통과 3개

다음 수치는 동결된 **등록 평가**의 Parent-macro와95% Parent Bootstrap CI다.
Summary의 별도2,000회 CI와 섞지 않았다. 등록 Bootstrap은10,000회다.

| 정확한 Gate 이름 / 대상 | B0 → 후보 | Δ CI에서 실제 판정값 | 허용 한계 |
|---|---:|---:|---:|
| `graph_background_point_fraction_ci95_upper_delta_maximum` / Box | .0252423485 → .0262911425 | 상한 +.0053014799 | ≤+.005 |
| `graph_wrong_object_point_fraction_ci95_upper_delta_maximum` / Box | .0025526767 → .0044523003 | 상한 +.0057459295 | ≤+.005 |
| `peg_end_coverage_ci95_lower_delta_minimum` / Peg | .4740223509 → .4621323826 | 하한 −.0247103190 | ≥−.01 |

분모: Box 두 오염률은 각 Frame의 **해당 Box Graph 전체 Geometry Point 수**이며
빈 Geometry는 미정의다. 두 조건 모두 정의 가능한 EXACT_GT Paired Frame은 각각
4,463개다. Peg Ends는 현재 관측 GT의 양 끝10%씩에 속한 Valid-depth 표면 표본 중
올바른 Source Graph Point가4mm 이내에 있는 비율이며5,234개 Paired Frame이다.
모두88 Episode를 Episode→같은 Physical Parent 평균으로 묶어 **36 Parent**로 비교했다.
5,845 전체 Frame 중 GT_UNSUPPORTED611개는 제외했다. Frame/Point를 독립 표본으로 세지 않는다.

대표 실패는 Box의 `10i_s1_0355`(Parent 배경점 .132070→.196134,
다른 물체점 .028533→.095267), Peg Ends의 `10i_s1_0645`(.507535→.370778)다.
35개 Gate 중29 PASS/3 FAIL/3 INCONCLUSIVE이며, 미지원3개도 통과로 바꾸지 않았다.
Primary와 전체 Coverage의 평균 이득이 이 안전 실패를 상쇄하지 않는다.

## 4. 손실이 처음 확인되는 단계

**Peg 끝:** 현재 후보0645 full_task t6에서 Wrist SAM은 TP1369/1386, FP1로 복구됐지만,
Graph Ends는 .313953→.239535로 낮아졌다. t86은 Wrist IoU .999228인데 Graph Ends는
1→.426829였다. 따라서 좋은 Mask 뒤 **Geometry/Graph 점 선택에서 손실이 드러난다.**
기존 A의 Source 진단에서는 Wrist 점 증가가 cap128 안에서 Main 끝점을 대체하는 경로를
확인했다(예:2000058 t95 Main127/Wrist0→65/63, End-band Point83→38).
다만 이 A 진단을 현재 후보 전수의 유일 원인으로 일반화하지 않는다. 현재 후보의
모든 손실에 대한 Filter/Fusion/Sampling별 정확한 첫 Drop은 **미확정**이다.

**Box 오염:** 0355의 원래 t0 Mask부터 Robot FP1848개가 있었다. 후보는 t6 새 Prompt를
승인했지만 그 Frame은 GT_UNSUPPORTED이므로 최초 오염 여부를 판정할 수 없다.
다음 EXACT_GT t7에는 후보 SAM Mask와 Graph에 이미 다른 물체 Pixel/Point가 각2개 있었다.
Detector Box와 SAM 중 어느 쪽이 이를 처음 만들었는지는 **미확정**이다.
t66은 Mask TP620/FP3 중 배경2 Pixel인데 Graph에는 배경2/2점만 남았고,
t85는 Mask TP597/FP8 중 다른 물체1 Pixel인데 Graph에는 다른 물체1/1점만 남았다.
소수 Mask 오염이 Downstream에서 집중되는 현상은 확인됐지만, 올바른 Point를
제거한 세부 Filter의 첫 위치는 기존 증거만으로 확정하지 않았다.

## 5. 완료 범위·보존·실제 재개 지점

- 완료: Baseline101 Action Frame 동등성, 역사적 DINO/Feature/SAM 통제·Gallery,
  Development36/88 Episode/5,845 Frame의 B0/A/P/C_KEEP/BC_KEEP/Main-Box/WB-Handoff
  비교와 Graph·등록 평가. 별도26-Frame 실제 통합 Engineering과 CPU QA도 보존했다.
- 부분 완료: Internal30의 B0는60/60 Episode/4,066 Frame 완료. 후보는
  **38/60 Episode/2,565 Frame 완료**, `10i_stage3_10i_s1_0858_full_task`의21-Frame
  원자적 Prefix만 PARTIAL이다. 완료 Receipt가 없고 SAM2 Memory를 직렬화하지 못했다.
- 미시작: Internal GT/Paired Graph·등록 성능 평가, 신규 Final Normal32/Stress12 수집과
  평가, 대기 중 추가 전수 SAM Oracle·Detector Raw 재실행, 독립 Final Reporting/Gallery,
  Supplemental 전수 및 최종 Stratification. Seed/Protocol 등록은 실제 수집과 구분한다.
- 재사용 가능: 원본 RGB-D/GT Sidecar와 EXACT/UNSUPPORTED Flag, 이미 저장된 DINO
  Raw Query/Token Cache, DINOv2 Embedding/Pair/Reference, 완료 Mask/Graph/Provenance,
  기존 통제 CSV·Gallery, 동결 Source/Config/기준. 이미 계산한 값은 추가 추론 없이 읽을 수
  있다. PARTIAL Prefix는 증거로만 재사용하며 정상 SAM Memory의 대체물이 아니다.

메인·대기 Runner와 추론 PID만 개별 중지했고, 관련 실행 프로세스0을 확인했다.
완료 후보38개와 B0 60개의 파일 SHA, 동결 Source2,576개 파일과 Source/Config/후보 Hash를
확인했다. 새 Sequence/Final/추가 Oracle/Raw 재실행의 자동 시작을 막았다.
Launcher5개는 원본을 `.PAUSED_SCOPE_REVIEW.original`로 보존하고 종료 Stub으로 교체했다.
원래 tmux의 작업 Shell이 자연 종료하며 해당 Session도 자동 종료돼, 같은 이름의
Idle `PAUSED_SCOPE_REVIEW` Pane를 남겼다. tmux Kill이나 다른 연구 프로세스 종료는 하지 않았다.

**재개는 아직 승인되지 않았다.** 범위 검토 뒤 승인될 경우에만 완료38개 Receipt의
SHA를 검증해 건너뛰고, 현재0858 full_task를 **t0부터** 재실행하는 지점이다.
21-Frame 뒤에 Suffix를 이어 붙이면 안 된다. 나머지 미시작21개와 모든 후속 단계도
현재는 차단돼 있다. 독립 성능 판정·연구 전체 완료·Stage4B Handoff는 미완료다.

[Pause·보존 검증](../../outputs/task10i/stage4a2/paused_scope_review/pause_verification.json),
[재개 지점](../../outputs/task10i/stage4a2/paused_scope_review/resume_point.json),
[프로세스 최종 확인](../../outputs/task10i/stage4a2/paused_scope_review/final_process_readback.json),
[기존 등록 수치](../../outputs/task10i/stage4a2/increment_v11/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/run/registered/registered_parent_metrics.json),
[정확한 Gate 결과](../../outputs/task10i/stage4a2/increment_v11/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/run/registered/registered_gate_results.csv).

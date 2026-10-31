# Task10-G seed98202: peg 마스크의 그리퍼 오배정 진단

대상: Final / C / step011487 / seed98202. 기존 저장 결과를 2026-10-31 로컬에서 읽기 전용으로 확인했다.

## 결론

**hand_camera의 raw peg 마스크가 TCP 주변 그리퍼 손가락/패드 영역에 오배정된 것으로 확인된다.** 영상의 빨간 overlay는 object_id=0(peg), 초록은 object_id=1(box)이다. TCP는 좌표계 원점이며, 여기서 오배정 대상은 그 주변의 실제 그리퍼 표면을 뜻한다.

이 오배정 마스크가 최종 정책 그래프에 peg로 들어간 것은 아니다. Wrist gate는 250/250스텝에서 거절했고, trace의 `missing_objects[0]`는 250/250스텝에서 true이다. 즉 정책은 이 에피소드 내내 peg 관측 노드가 없는 그래프를 받았다. Box는 250/250스텝에서 존재했다.

## 저장 증거

- [원본 rollout.mp4](../../outputs/task10g/final_results/rollouts/final/C/step011487/seed98202/rollout.mp4)
- [원본 trace.json](../../outputs/task10g/final_results/rollouts/final/C/step011487/seed98202/trace.json)
- [집계 및 샘플 gate 원문](../../outputs/task10g/diagnostics/seed98202_peg_mask/summary.json)
- [t=0](../../outputs/task10g/diagnostics/seed98202_peg_mask/frame_000.png), [t=30](../../outputs/task10g/diagnostics/seed98202_peg_mask/frame_030.png), [t=100](../../outputs/task10g/diagnostics/seed98202_peg_mask/frame_100.png), [t=200](../../outputs/task10g/diagnostics/seed98202_peg_mask/frame_200.png), [t=249](../../outputs/task10g/diagnostics/seed98202_peg_mask/frame_249.png): MP4에서 직접 디코딩한 프레임. 영상 오른쪽 hand_camera에서 그리퍼 손가락의 분홍/빨강 overlay를 확인할 수 있다. 왼쪽 테이블 위 peg의 원래 빨간 재질만으로 마스크 유무를 판단하지 않았다.

| 항목 | 결과 |
| --- | --- |
| 저장된 영상 / trace | 250프레임 / 250스텝 |
| Wrist raw peg mask 최초 nonempty | t=30 (영상 재생 시 약 1.5초) |
| Wrist raw peg mask nonempty | 220스텝 (t=30–249) |
| Wrist gate 통과 | 0/250 |
| 최종 그래프 peg 누락 | 250/250 |
| 최종 그래프 box 누락 | 0/250 |

`t=30`에서 raw peg 1,577픽셀 모두 유효 depth가 있었으나 self/overlap filtering 후 35점(2.22%)만 남았다. Gate 거절 사유는 `robot_or_overlap_dominated`, `fragmented_3d_support`이다. `t=100`은 510픽셀 중 18점(3.53%)만 남고, `t=249`는 1,565픽셀 중 37점(2.36%)만 남았다. 전체 nonempty 스텝의 clean fraction은 0–4.55%였다. 이 수치와 영상이 로봇 표면 오배정을 뒷받침한다. 다만 제거 수치는 로봇 확정/불확실 영역과 mask overlap을 합친 결과이므로 이를 순수 robot 픽셀 수로 해석하지 않는다.

## 코드 확인 및 해석 범위

- [video_frame](../../graph/plan_a_bc/rollout.py): raw mask의 peg를 RGB (255,70,70), box를 (70,255,90)으로 40% 혼합한다. 영상 overlay는 gate 이후의 정책 입력 마스크가 아니다.
- [eval_runner](../../graph/graph_recurrent10g/eval_runner.py): adapter raw mask로 geometry를 만든 후, 최종 graph의 node_type 3/4 존재 여부를 `missing_objects`에 기록한다. 영상은 같은 시점의 raw mask를 표시한다.
- [CurrentGeometry](../../graph/plan_a_bc/geometry.py), [PlanA/wrist_gate](../../graph/plan_a_geometry/core.py): C3 조건에서는 hand_camera peg를 gate가 승인한 점으로 제한한다.
- [filtered_masks](../../graph/persistent_geometry/fusion.py): mask overlap, confirmed_robot, uncertain 영역을 제거한다.

저장 폴더에 독립 raw mask 배열 및 detector 초기 box/초기화 기록은 없다. 따라서 정확한 픽셀별 GT IoU, 최초 detector box 오검출과 SAM2 추적 drift의 구분은 이번 자료만으로 확정하지 않는다. 이 오배정이 정책 실패의 유일한 원인이라는 인과 주장도 하지 않는다.

## 검증 / 변경 범위

원본 MP4의 프레임 수와 5개 샘플 프레임을 디코딩하고, 전체 250개 trace record를 집계했다. 실행 코드 변경, 학습, 재평가, 게시를 수행하지 않았다. 공통 logger/publisher의 새 실행은 필요하지 않은 기존 artifact 분석이다. Simulator 및 CUDA/MPS backend는 이번 확인에서 실행하지 않았다.

## 후속 재현으로 확인한 초기화 원인

원래 서버에서 초기 snapshot/graph 및 첫 31스텝 gate를 exact 재현했다. 첫 프레임의 실제 peg 후보 score 0.231이 미검출 점수 0.3에 밀려 SAM2 초기화에서 제외됐고, t=25/30에 반복된 그리퍼 오검출로 t=30에 peg 추적이 처음 시작됐다. 따라서 위 분석에서 미확정으로 남긴 detector 오검출 대 SAM2 drift 구분은 초기 detector 오검출로 확인됐다. 전체 후보 점수와 검증 자료는 [후속 전체 보고서](10g_seed98202_initialization_diagnosis.md)에 있다.

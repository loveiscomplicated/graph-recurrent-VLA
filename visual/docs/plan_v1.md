# Vision v1 실험 계획 (2026-10-19 ~ 10-25, RTX 3090 24GB × 1)

결과는 [`../REPORT.md`](../REPORT.md)에 있다.

## 0. 출발점

- 전체 구조: **Vision → Graph(point cloud 기반, EE 상대 좌표) → Recurrent controller(GRU/CfC/RKN 후보, 7D ΔEE action, BC) → Action**.
  무거운 VLM/grounding은 처음에 1회만 돌리고, 이후는 가벼운 tracking이 매 step 갱신한다. drift가 생기면 re-anchor로 보정한다.
- 벤치마크: ManiSkill3 `PegInsertionSide-v1` (20 Hz, 100 step 제한). insertion 도중 box를 옮겨 대응을 보는 실험이 후보로 있다.
- 일정: 10/28 회의에서 prototype 결과를 공유하고 Architecture v1을 정한다. 이번 주 결과는 그 회의의 Vision 쪽 근거다.
- Graph 팀 현황(10/9): MLP / Center GNN / Surface GNN 모두 task 실패. expert가 목표 근처까지 데려다주면 일부 성공. 즉 downstream은 GT state를 넣어도 아직 불안정하다.
- 회의에서 Vision에 맡긴 조사: (1) point별 **task relevance score**, (2) tracking drift를 보정하는 **re-anchor**, (3) RGB-D로 먼저 만들고 **RGB-only depth**로 확장.

### v0에서 확인한 것 (3 ep)
- GDINO tiny: init 프레임 IoU ≥ 0.96. 이름보다 외형 묘사 프롬프트가 낫다. 매 프레임 재검출은 불안정하고 top-1은 팔을 물체로 고르기도 한다.
- SAM2.1-small: peg IoU 0.726, `conf` re-anchor + re-ID로 0.901(oracle 0.914). box 0.946.
- 속도: tracking 46 ms + point cloud 27 ms(CPU) → 13.8 FPS, online 추정 9.3 FPS. **20 Hz에 못 미친다.**
- 미해결: (a) 가림 중 외형이 바뀐 peg를 re-ID가 못 알아봄, (b) mask가 gripper로 번지는데 logit이 높아 conf 트리거에 안 걸림, (c) 3 ep라 설정 간 0.02–0.05 차이는 믿을 수 없음(경계선 사례 하나가 평균을 0.73↔0.92로 뒤집음).
- centroid 오차는 보이는 표면 기준 peg 0.78 / box 0.97 cm지만, GT pose 기준 peg 7.6 / box 12 cm다(가시 표면 bias). hole pose는 추정하지 않는다.

### 같은 팀 prototype (10/9 보고서, `feature/visual-prototype`)
- HSV baseline, SAM2.1 Tiny(causal frame source), 고정 간격 re-anchor. dev/eval seed 분리, 좌표 검증(K의 픽셀 중심 규약 u+0.5, 역투영 오차 0.56 mm).
- re-anchor 때 memory 전체를 재초기화하면 오히려 해로웠다(s103). patch 가림에서 SAM이 occluder 위에 FP mask를 냈다.
- flat shading 렌더라 HSV가 비정상적으로 강하다.

---

## 1. 목표 (10/28 회의에 가져갈 것)

1. **믿을 수 있는 평가 기반**: dev/eval을 분리한 30–50 ep 벤치마크와 교란 세트. v0 결론을 그 위에서 다시 확인한다.
2. **실시간 Vision**: causal streaming으로 **20 Hz(tracking + point cloud ≤ 50 ms/frame)**가 되는지, 된다면 어떤 설정인지.
3. **Re-anchor v1**: memory를 버리지 않는 보정, gripper 번짐과 occluder FP를 잡는 트리거, 더 강한 re-ID, box 이동 시 복구 속도.
4. **Downstream이 쓸 수 있는 출력**: (a) 물체 pose와 hole pose 정확도, (b) **vision 추정값으로 스크립트 정책을 돌렸을 때 삽입 성공률**. perception 오차가 control에 주는 영향을 처음으로 정량화한다.
5. **조사 항목 feasibility**: task relevance score, RGB-only depth.

## 2. 평가 원칙

- 모든 비교는 **dev seed에서만 조정하고 eval seed에서는 1회만 평가**한다. eval로 다시 튜닝하지 않는다.
- 모든 수치에 episode 수와 CI를 함께 적는다. 3–5 ep 결과로 우열을 주장하지 않는다.
- GT는 평가와 명시적인 oracle 실험에만 쓴다. 모델 경로가 GT를 읽지 않게 한다(같은 팀 prototype의 input boundary 테스트 방식 참고).
- 결과는 이미지로 직접 확인하고, 실패한 실험도 결과로 기록한다.
- 체크포인트, 영상, 대용량 npz는 git에 넣지 않는다.

## 3. 실험 목록

### P0 — 다른 실험의 전제

**E0. 환경 + 데이터셋 확대**
- 환경을 다시 구성하고(EGL Vulkan ICD 우회, `constraints.txt`로 torch 고정) smoke test를 통과시킨다.
- 수집: dev seed 100–109 (10 ep), eval seed 200–239 (40 ep). base_camera와 hand_camera 512², GT seg/pose/TCP 저장.
- 정책 다양화: 성공 궤적, waypoint 노이즈(±1–2 cm, ±5°), 재시도 포함, 일부 실패 궤적. ep마다 정책을 기록한다.
- 교란 세트: `patch`(물체 영역 1 s 회색 가림), `dropout`(전체 프레임 1 s 끊김), **`box_move`**(insertion 직전 box 2–5 cm 순간이동), `lighting`(조명 세기·색 무작위).

**E1. v0 결론 재검증 (40 ep eval)**
- SAM2.1-small(compile off/on), SAM2.1-base+, SAMURAI-small, EfficientTAM-ti × re-anchor {none, conf, conf_e5}.
- 지표: peg/box IoU와 episode bootstrap 95% CI, hit@0.5, 재획득 성공률과 걸린 프레임 수, 가림 중 FP, ms/frame p50/p95, GPU peak.
- 질문: compile의 이득이 진짜인가, SAMURAI 우위가 유지되는가, conf와 conf_e5의 차이가 CI 밖인가. → v1 기본 tracker를 여기서 정한다.

### P1 — 이번 주 핵심

**E2. Causal streaming + 20 Hz**
- `init_state(video_path)` 대신 causal frame source(미래 프레임 접근 시 예외, 최근 몇 프레임만 유지). JPEG 쓰기와 전 프레임 로드를 없앤다.
- 단계별 ablation: streaming → image encoder compile → point cloud 후처리(erosion/SOR/FPS) GPU 이전 → 입력 해상도 → bf16 → 오래된 memory 정리.
- 지표: ms/frame p50/p95/max, IoU 변화, GPU 메모리, offline 결과와의 동치(비트 단위 또는 IoU ≥ 0.99).
- 목표 p95 ≤ 50 ms. 안 되면 N step마다 갱신하고 중간에는 마지막 값과 age를 내는 방식으로 몇 Hz까지 되는지 본다.

**E3. Re-anchor v1**
- (a) 보정 prompt: memory를 재초기화하지 않고 추적 중인 프레임에 conditioning을 추가한다. frame 0 memory 유지.
- (b) 수락 게이트: 후보 면적이 최근 안정 면적의 r배 이상, 3D extent가 물체 크기 prior와 일치(크기는 첫 프레임 cloud에서 추정, GT 미사용).
- (c) 새 트리거: 면적 급증(1.5배 초과), mask 내부 depth 두 덩어리(gripper 번짐), 3D extent 부족, occluder FP 의심.
- (d) re-ID 강화: DINOv2 patch feature + 마지막 3D 위치 연속성 + 3D 크기 일치.
- (e) TCP prior(별도 ablation, 기본 OFF): grasp 후 TCP pose와 peg 크기로 투영한 bbox를 후보 검증에만 쓴다. 순수 vision만 쓸지는 팀이 정할 문제라 수치만 낸다.
- 평가: 기본 eval + patch / dropout / box_move. 복구 시간, 오선택, 미획득, occluder FP율.

**E4. Pose / hole pose 추정**
- Graph/Recurrent 팀은 지금 GT `peg_pose`와 `hole_pose`를 쓴다. Vision이 이 값을 어느 정밀도로 줄 수 있는지가 전체 프로젝트의 feasibility 질문이다. hole clearance는 mm 단위다.
- 방법: 관측 표면 cloud에 형상 prior를 맞춘다. peg는 직육면체(크기는 관측 추정 또는 task 파라미터, 어느 쪽인지 명시), box는 hole이 있는 box 모델. PCA 초기화 후 ICP/RANSAC, 프레임 간 smoothing 버전도 비교한다.
- 지표: peg 위치/자세 오차, hole 위치/축 방향 오차(mm, deg), 가림 구간별 오차, 처리 시간.
- 산출물: "Vision이 줄 수 있는 state" 표(표면 cloud, centroid, 6D pose, hole pose 각각의 오차와 실패율).

**E5. Vision-in-the-loop 스크립트 정책**
- waypoint 정책이 GT pose로 계획하던 부분을 E4의 vision 추정 pose로 바꿔 closed-loop로 돌린다(E2 streaming, 매 step 또는 N step마다 갱신).
- 비교: GT state / vision state / GT + 인공 노이즈(σ = 1, 2, 5, 10 mm / 1, 3, 5°) / vision + re-anchor on/off / box_move.
- 지표: 40 ep 삽입 성공률, 실패 단계(grasp / align / insert), 실행 시간.
- "perception 오차가 몇 mm를 넘으면 task가 깨지는가" 곡선을 만든다. perception error와 controller error를 분리하는 근거가 되고, Graph/Recurrent 팀도 바로 쓸 수 있다.

### P2 — 시간이 남으면

**E6. Task relevance score**
- point마다 상호작용할 부분인지 score를 준다(peg: 잡는 부분과 삽입되는 머리, box: hole 입구 주변). GT 라벨은 GT pose와 형상으로 만든다(평가 전용).
- 후보: (1) 기하 휴리스틱, (2) CLIP/SigLIP patch similarity, (3) VLM pointing(Qwen2-VL-2B 1회 + SAM mask 안으로 전파).
- 지표: AUROC/AP, 처리 시간. 결론은 "VLM이 필요한지, 기하만으로 충분한지".

**E7. RGB-only depth**
- Depth Anything V2 Small(metric), Video Depth Anything Small, 참고용 Metric3D v2. sim depth로 scale을 맞춘 경우와 아닌 경우.
- 지표: mask 내부 AbsRel/δ1, cloud centroid 오차, E4 pose 오차, ms/frame. 가능하면 E5 성공률까지.

**E8. 카메라 / 도메인 강인성**
- hand_camera 단독, base + hand 융합(world 좌표에서 cloud 합치기)으로 가림 구간이 얼마나 나아지는지.
- lighting 교란과 다른 shader 렌더에서 GDINO/SAM2 vs HSV baseline. HSV가 flat shading 덕을 본 것인지 확인한다.

**E9. 다른 팀용 데이터 export**
- E0 데이터에 대해 Vision 출력(cloud, pose, confidence, stale age)을 GT와 같은 시간축으로 저장한다. Graph 팀이 GT 대신 vision 입력으로 학습·평가해 볼 수 있게 한다.
- 포맷: 프레임당 npz 1개, 물체별 고정 N=1024 + `valid` mask, `points_world`와 EE 기준 상대 좌표, `relevance`(E6 결과, 없으면 1.0), `age_steps`, `confidence`.

## 4. 일정

| 순서 | 할 일 |
|---|---|
| 1 | E0 환경, 수집 스크립트 다양화와 교란, dev / eval 수집 |
| 2 | E1 → 기본 tracker 확정 |
| 3 | E2 streaming과 ablation |
| 4 | E3 re-anchor v1 (dev에서 개발, eval 1회) |
| 5 | E4 pose / hole |
| 6 | E5 vision-in-the-loop, 노이즈 곡선 |
| 7 | E6–E9, 실패 분석, 데모 영상 |
| 8 | 보고서 정리, 10/28 회의용 요약과 안건 |

밀리면 P2를 버리고 P0 → P1 순서를 지킨다. timing을 재는 실험은 GPU를 단독으로 쓸 때만 실행한다.

## 5. 설계 결정 (회의 전까지의 기본값)

| 열린 질문 | 기본값 | 이유 |
|---|---|---|
| re-anchor에 TCP pose 사용 | 기본 OFF, ablation으로만 | 원칙은 팀 회의에서 정한다 |
| 평가 카메라 | base_camera로 주 평가, hand_camera는 E8 | |
| 해상도 | 512² (E2에서 낮춘 해상도도 측정) | 128²는 GDINO/SAM2에 너무 작다 |
| 프롬프트 | 외형 묘사 프롬프트 고정 | VLM 프롬프트 생성은 E6에서만 |
| 가려진 프레임 출력 | 빈 cloud + `valid=False` + `age_steps` | stale 값을 쓸지는 downstream이 정한다 |
| 목표 주기 | 20 Hz | 안 되면 달성 가능한 주기를 보고한다 |
| GDINO | tiny | E1에서 여유가 있으면 base를 1회 비교 |
| 공용 저장소 코드 | 수정하지 않음 | Vision 폴더 안에서만 작업, 다른 팀 모듈은 구현하지 않음 |

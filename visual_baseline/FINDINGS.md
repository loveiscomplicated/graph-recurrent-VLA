# Vision baseline (ACT / Diffusion Policy) — 결과와 원인 분석

작성: 2026-11-10 (UTC), 코드 `visual/baseline/` (로컬 브랜치 `visual-baseline`).
수치 원본은 같은 폴더의 `*/eval_*.json`, `*/train_*.json`, `probe_lateral*.json`, `SUMMARY.md`.

## 설정

- 데이터: 팀 HF `graph-recurrent-rollouts` run c555bed7…, scripted expert 900 episode (512px RGB-D main+wrist).
  정책 입력은 main·wrist RGB 256px + robot15(TCP pose, 손끝, gripper).
- 평가: 학습과 겹치지 않는 seed 100개(500000~, 같은 `ends_and_grasp` 시야 조건), horizon 250,
  성공 = `info["success"]`. 같은 seed에서 expert 98/100.
- 정책: LeRobot 0.7 ACT(chunk 50) / Diffusion(n_obs 2, horizon 16, DDIM 16).

## 결과 (새 seed 100개 성공 수)

| 모델 | 성공 | 비고 |
|---|---|---|
| expert (GT 상태 사용) | 98 | 상한 |
| ACT 100k (ensemble / chunk10 / 매 스텝) | 4 / 7 / 1 | 추론 방식 차이 없음 |
| ACT 25k / 50k / 75k | 2 / 5 / 4 | 과적합 아님 |
| Diffusion 25k / 50k / 75k / 100k | 6 / 1 / 11 / 7 | |
| gripper 이진화 (ACT / DP) | 3 / 9 | 원인 아님 |
| ACT 영상만(robot15 제거) | 1 | proprioception 의존 아님 |
| ACT + DART 600 (20k/40k/60k) | 8 / 6 / 5 | 회복 데이터로도 개선 없음 |
| 학습 seed 50개: ACT / DP / ACT-영상만 / ACT-DART | 30 / 12 / 18 / 9 | 학습 seed에서도 낮음 |

스텝당 추론 시간(워커 공유 상태): ACT p50 14–33 ms, DP p50 15 ms / p95 ~250 ms(8스텝마다 denoising).

## 실패 양상

- 실패의 85–90%가 같은 단계: peg를 잡아 들고(100%) 삽입 깊이까지 가지만 **옆으로 ~2 cm 어긋난 채 멈춤**
  (expert는 lateral < 1.5 mm일 때만 삽입 시작). 잡기 실패(no_lift)는 0.
- 어긋남 크기는 구멍이 학습 평균 위치에서 얼마나 떨어졌는지와 무관(r≈0): "평균 위치로 가기"가 아님.
- 손 안의 peg 위치 편차는 작음(삽입축 5.6 mm, 횡 0.1 mm): 쥐는 방식 차이도 원인 아님.

## 핵심 원인: 입력 영상에서 mm 수준 횡 오차를 읽을 수 없음

정책과 같은 입력으로 "구멍 기준 peg 머리 위치"만 맞히도록 학습한 probe(ResNet18 회귀) 검증 오차:

| 입력 | 횡 오차 중앙값 | p90 |
|---|---|---|
| main+wrist 256px | 15.2 mm | 50 mm |
| main 256px | 14.1 mm | 56 mm |
| wrist 256px | 19.3 mm | 77 mm |
| main+wrist 128px | 14.6 mm | 55 mm |
| main RGB + depth | 15.4 mm | 54 mm |
| main+wrist, 3배 학습 | 13.8 mm | 54 mm |
| **peg를 쥔 프레임만**: main 256px | **6.5 mm** | 39 mm |
| **peg를 쥔 프레임만**: main+wrist, 2배 학습 | 8.1 mm | 40 mm |

위 6줄은 구멍 근처 깊이(-25~-5 cm)의 모든 프레임이라 peg가 아직 바닥에 있는 프레임도 섞여 있다. 정책이
실제로 정렬하는 상황(peg를 쥔 상태)만 보면 중앙값 6.5 mm, p90 ~4 cm다. 어느 쪽이든 삽입에 필요한
1.5 mm보다 4배 이상 크고, 꼬리(p90)는 정책의 실제 횡 오차(~2 cm)보다도 크다. 해상도·depth·학습량을
바꿔도 줄지 않는다. 즉 **end-to-end 영상 모방학습(ACT/DP)은 이 과제·카메라 구성에서 mm 정렬에 필요한
상태를 복원하지 못하고**, 회복 데이터(DART)나 추론 방식 조정으로는 해결되지 않는다.

이전 SAM2+point cloud pose 추정(`visual/REPORT.md`)이 82–85%에 도달한 것은 기하 기반 pose fitting으로
mm 단위 상태를 직접 추정했기 때문으로 보인다. 이는 통합 모델처럼 물체 상태를 명시적으로 추정하는 구조의
필요성을 뒷받침하는 baseline 근거다.

## 한계와 다음 후보

- probe는 global-pool ResNet18 회귀 하나다. 공간 구조를 유지하는 모델(keypoint/spatial softmax), 구멍
  주변 고해상도 crop, 512px 원본은 시험하지 않았다.
- DART는 burst 노이즈 1개 설정(σ 8 mm/step)만 시험했다.
- 다음 후보: (1) 구멍 주변 crop을 512px로 추가 입력, (2) 추정 pose(SAM2 파이프라인 출력)를 정책 입력으로
  쓰는 hybrid baseline, (3) 더 근접한 카메라 시점.

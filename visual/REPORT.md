# Vision 실험 보고 (v1)

v0(10/8 시작)부터 이어온 Vision 실험 중 v1 단계의 결과를 정리했다. 실험 설계는 [`docs/plan_v1.md`](docs/plan_v1.md), 이전 prototype은 [`docs/v0.md`](docs/v0.md), 환경 문제와 우회법은 [`docs/known_issues.md`](docs/known_issues.md)에 있다.

평가 원칙:
- 설정은 **dev(seed 100–109)에서만 조정**하고 **eval(seed 200–239, 40 ep)에서 1회 평가**한다.
- eval을 실패 분석에 쓴 뒤의 변경은 **새 seed 세트** eval2(300–339), eval3(400–439), eval4(500–539), eval5(600–639)에서 확인한다.
- 비교는 모두 같은 episode끼리 짝지은 차이와 **episode bootstrap 95% CI**, 승/패 수로 쓴다. 3–10 ep 결과로는 우열을 말하지 않는다.
- GT(mask, pose)는 평가와 명시한 oracle 실험에만 쓴다. 모델 경로는 GT를 읽지 않는다.
- 로봇 손 위치(TCP)는 사용 여부가 팀 결정 사항이라 **vision만 / vision + TCP**를 항상 나란히 보고한다.

---

## 1. 요약 (10/28 회의용)

실시간(20 Hz) 추적은 된다. vision 추정만으로 스크립트 정책을 돌리면 삽입 성공률이 GT 100% 대비 **52.5–62.5%**다. 로봇 손 위치(TCP), "다가가서 보고 잡기", "삽입 직전 다시 보기"를 더하면 처음 보는 seed 세트 3개에서 **82.5–85%**다. 남은 병목은 mm 단위 정밀도와 grasp 이후 peg 추적이다.

| 질문 | 답 | 근거 |
|---|---|---|
| 기본 tracker | **SAM2.1-small** (compile은 속도용). base+, EfficientTAM-ti는 peg에서 유의하게 나쁘다. SAMURAI, DAM4SAM은 차이가 없고, SAM2Long은 새 seed 세트에서 재현되지 않았다 | E1, B3 |
| v0 결론(compile 효과, re-anchor 효과, SAMURAI 우위) | 40 ep에서 **셋 다 재현되지 않았다**. v0의 차이는 3 ep 중 경계선 사례 하나에서 나왔다 | E1 |
| 20 Hz가 되는가 | **된다.** streaming + compile + GPU point cloud + GPU 전처리 = **p95 45.1 ms (22.5 FPS)**, GPU 0.61 GB, 정확도 손해 없음 | E2 |
| mask IoU | peg 0.76–0.78 (바닥 0.92 / grasp 후 0.72 / 삽입 중 0.62), box 0.97 | E1, E3 |
| re-anchor v1 | peg의 가림·끊김 뒤 재획득 +0.020–0.028 (두 seed 세트에서 유의). box에는 해롭다 → **peg에만 켠다** | E3 |
| vision이 줄 수 있는 state | GT mask 입력 시 peg 6D 중앙 0.5 mm / p90 3 mm, hole 0.8 mm / 0.01°. tracker mask 입력 시 peg p90 24 mm. **표면 centroid는 pose 대용으로 못 쓴다(79 mm bias)** | E4 |
| 삽입에 필요한 정밀도 | GT + 노이즈: 위치 1 mm 82.5%, 2 mm 52.5%, 5 mm 7.5%, 회전 1° 37.5% → **위치 ≲ 1 mm, 축 ≲ 0.3°** | E5 |
| vision-in-the-loop | vision만 **52.5–62.5%**. 최종 설정 **82.5–85%** (eval2/4/5). box_move 12.5% → **60%** (eval3) | E5 |
| RGB-only depth | oracle scale로도 peg 3–7 cm 오차 → **mm 작업에 못 쓴다** | E7 |
| relevance score | 추정 pose + 기하 규칙으로 AUROC 0.94 (peg) / 1.00 (box). VLM 불필요 | E6 |
| HSV 기준선 | 조명만 바꿔도 box IoU 0.93 → 0.003. SAM2는 0.96 유지 → HSV가 강했던 것은 flat shading 때문 | E8 |

### 실험 상태
실험은 서버 한도 및 비용 문제로 일단 중지했다. 아래는 그 시점 기준이다.

| 실험 | 상태 | 결과 |
|---|---|---|
| E0 환경 + 데이터셋 | 완료 | `results/v1/E0/`, `results/v1/init/` |
| E1 v0 결론 재검증 (40 ep) | 완료 | `results/v1/E1/` |
| E2 streaming / 20 Hz | 완료 | `results/v1/E2/` |
| E3 re-anchor v1 | 완료 | `results/v1/E3/` |
| E4 pose / hole | 완료 | `results/v1/E4/` |
| E5 vision-in-the-loop | 완료 | `results/v1/E5/` |
| E6 relevance | 완료 | `results/v1/E6/` |
| E7 RGB-only depth | 완료 | `results/v1/E7/` |
| E8 도메인 강인성 | 부분 (hand camera, 카메라 융합 미실행) | `results/v1/E8/` |
| E9 다른 팀용 export | 완료 | `data/v1/export/` (git 제외, 다시 생성 가능) |

---

## 2. 설계 결정과 근거

| 결정 | 근거 |
|---|---|
| tracker는 SAM2.1-small | base+ Δpeg −0.046 [−0.090, −0.008], EfficientTAM-ti −0.043 [−0.068, −0.020]. SAMURAI +0.013 [−0.008, 0.050]으로 차이 없이 비용 2배 (E1) |
| 실시간 경로는 causal streaming | 미래 프레임 접근 시 예외를 내는 버퍼로 인과성을 강제했다. offline과 mask가 비트 단위로 같다 (E2) |
| re-anchor 트리거는 `lost` 하나만 | 면적 급증·depth 분리·3D 길이·depth 급변 트리거는 dev shadow 실행에서 거의 전부 오경보였다(예: jump 231회 중 실제 오류 1회) (E3) |
| re-anchor는 peg에만 | box는 re-anchor 없이도 재획득 0.98이고, 켜면 가림 중 FP가 0.04 → 0.17로 는다 (E1, E3) |
| re-anchor 보정은 memory 재초기화 없이 conditioning 추가 | 재초기화하면 frame 0 memory까지 사라져 부분 검출 하나로 추적이 무너진다(같은 팀 prototype의 s103 사례) |
| 물체 사양(L, r)은 알고 있다고 둔다 | 관측으로 크기를 추정하면 peg 위치 오차가 0.5 → 10 mm (E4) |
| pose 추정은 형상 prior 맞춤 | 표면 centroid는 보이는 면 bias로 79 mm 틀린다 (E4) |
| 가려진 프레임 출력은 빈 cloud + `valid=False` + `age_steps` | 마지막 값을 쓸지는 downstream이 정하게 한다 (E9) |
| TCP는 기본 OFF, 수치는 ablation으로 | 순수 vision 원칙은 팀 결정 사항이다. 대신 pose 단계에서 쓰면 +0.175 [0.050, 0.325]라는 근거를 같이 낸다 (E5) |
| 결론은 두 seed 세트에서 확인 | SAM2Long의 eval +0.050이 eval2에서 +0.018 [−0.008, 0.058]로 사라졌다 (B3) |

---

## 3. 병목 분석

### B1. 삽입 정밀도 요구가 vision 오차보다 빡빡하다 (가장 큰 병목)
- **관측(E5)**: GT 100% → 위치 노이즈 σ 1 mm 82.5% → 2 mm 52.5%, 회전 1° 37.5%. vision만은 52.5%로, 실효 오차가 σ≈2 mm 수준이다.
- **원인**: hole 여유 간격이 3 mm다. 정책은 force 피드백이 없는 위치 제어라 끝단 오차가 여유를 넘으면 그대로 실패한다.
- **문헌**:
  - 위치 제어 삽입은 peg 끝 오차 < clearance가 성공 조건이다([arXiv 1910.10852](https://arxiv.org/pdf/1910.10852)).
  - 빡빡한 공차에서는 registration 오차를 줄일수록 실패가 준다. 오차원은 perception, kinematics, registration이다([NISTIR 8198](https://nvlpubs.nist.gov/nistpubs/ir/2017/NIST.IR.8198.pdf)).
  - 고정 탐색 방식은 초기 오차 약 1 mm / 1°까지, 학습 기반 compliant 삽입은 3 mm 초기 오차에서도 성공한다([arXiv 1708.04033](https://arxiv.org/pdf/1708.04033)).
  - 접촉 기반 삽입은 0.1–1 mm 공차에서도 성공률이 높다([arXiv 2506.22766](https://arxiv.org/pdf/2506.22766), [arXiv 2507.03925](https://arxiv.org/abs/2507.03925v1)).
- **방향**: (a) vision 정밀도를 1 mm / 0.3° 아래로 낮춘다(B2–B4). (b) controller 쪽 compliance / 탐색 삽입. 문헌은 vision만으로 mm 정밀도를 보장하기보다 접촉 기반 보정과 나누는 쪽이다. (b)는 Controller 팀 영역이라 회의 안건으로 둔다.

### B2. grasp 이후 손에 쥔 peg pose (vision 실패의 주원인)
- **관측**: grasp 후 peg 오차 2–9 mm / 약 1°, 가끔 mask가 손으로 번져 수십 cm. GT mask를 넣어도 grasp 구간 p90이 2.7 mm다(E4).
- **원인**: 손가락이 peg 중간을 가려 한쪽 끝만 보인다. 축 방향 위치가 관측으로 정해지지 않는다.
- **문헌**: 손에 쥔 물체는 kinematics(손 자세) 예측과 관측을 결합하는 것이 표준이다([DLR elib 86975](https://elib.dlr.de/86975), [Wen et al.](https://cris.tau.ac.il/en/publications/robust-occlusion-aware-pose-estimation-for-objects-grasped-by-ada/)). 가림이 클수록 시각·촉각 융합의 이득이 크다([arXiv 2506.10787](https://arxiv.org/html/2506.10787v1)).
- **실험** (`--inhand`): vision만 시간 평균(`temporal`) 52.5% → 52.5% (변화 없음). vision + TCP(`tcp2`) **70%** (+0.175 [0.05, 0.325]).

### B3. tracking 소실과 번짐 (SAM2 memory 구조)
- **관측**: peg IoU clean 0.92 → grasp 후 0.72 → 삽입 0.62. 가림 뒤 재획득률 0.57–0.74. 손으로 번지는 사례(ep226).
- **원인**: SAM2는 최근 6 프레임 + 조건 프레임의 FIFO memory로 추적한다(`num_maskmem: 7`). 긴 가림 동안 memory가 "물체 없음" 프레임으로 채워지면 재등장한 물체를 잇지 못한다. 번짐은 peg와 붙어 있는 손이 사실상 distractor이기 때문이다.
- **문헌**: SAM2Long은 memory tree로 오류 누적을 줄인다([arXiv 2410.16268](https://arxiv.org/pdf/2410.16268)). DAM4SAM은 distractor-aware memory다([arXiv 2509.13864](https://papers.cool/arxiv/2509.13864), [arXiv 2512.09633](https://arxiv.org/pdf/2512.09633)). SAMURAI는 Kalman motion으로 memory를 고른다([arXiv 2411.11922](https://arxiv.org/html/2411.11922v2)).
- **실험** (eval 40 ep, SAM2.1-small 대비 peg IoU Δ [95% CI] (승/패)):

  | tracker | 기본 | patch | dropout | 재획득률 patch / dropout |
  |---|---|---|---|---|
  | re-anchor v1 | 0.000 | **+0.025** [0.008, 0.047] (9/0) | **+0.024** [0.005, 0.046] (6/0) | 0.72 / 0.78 |
  | DAM4SAM (`third_party/dam4sam` @aad389b) | −0.025 [−0.059, 0.004] (10/11) | −0.020 [−0.070, 0.027] (11/11) | +0.025 [−0.029, 0.072] (21/9) | 0.60 / 0.74 |
  | SAM2Long causal (@d70b50a) | +0.050 [0.007, 0.108] | +0.023 [−0.007, 0.054] (11/7) | **+0.055** [0.015, 0.094] (19/3) | 0.66 / **0.84** |

  - DAM4SAM은 유의한 차이가 없고 grasp 후(0.69)와 삽입 구간(0.56)에서는 오히려 낮다. 혼동 대상인 손이 peg와 붙어 있어 대안 mask가 distractor를 대표하지 못하는 것으로 본다(가설).
  - SAM2Long 공개 구현은 영상이 끝난 뒤 최고 pathway를 고르는 offline 방식이다. 각 프레임에서 그 시점 1등 pathway를 쓰는 causal 버전을 만들었고, offline과 같은 이득(+0.050)이 났다. streaming 래퍼 `scripts/stream_sam2long.py`는 offline causal과 비트 단위로 같다.
  - **SAM2Long은 새 seed 세트(eval2)에서 재현되지 않았다**: +0.018 [−0.008, 0.058] (5/6). eval의 이득은 장시간 가림 사례 몇 개(ep207, ep217)에 의존했다.
  - **IoU 개선이 삽입 성공으로 이어지지 않았다**: E5에 SAM2Long을 넣으면 vision만 0.500(SAM2 0.525), vision + TCP 0.675(0.75). 실패 대부분은 정렬 시점의 mm 오차와 시작 시점 추정에서 나온다.
- **결정**: SAM2.1-small 유지 + peg에만 re-anchor v1. SAM2Long은 프레임 손실이 잦은 환경에서 쓸 후보로 남긴다(비용: 물체당 pathway 3배).

### B4. hole 추정 (카메라 해상도 + box 이동)
- **관측**: 정지 box에서 hole 위치 0.6–0.8 mm, 축 0.01°. box 이동 뒤 vision 성공률 12.5%(GT 재계획 95%).
- **원인**: 512² base camera는 0.5 m에서 1 px ≈ 2 mm라 sub-mm는 해상도 한계에 가깝다. box 이동 직후에는 peg가 hole 앞을 가려 다시 볼 수 없고, 평평한 앞면만으로는 면 방향 이동을 ICP로 구분할 수 없다.
- **방향**: 관측이 안 되면 시점을 바꾼다(능동 지각, Bajcsy, Proc. IEEE 1988).
  - `--reobserve`: ICP로 이동을 감지하면 peg를 18 cm 비켜 hole을 다시 본다 → box_move 15% → 37.5% (+0.225 [0.05, 0.40]).
  - 남은 실패(eval2, 28건): 이동 감지 실패 11건(앞면 방향 이동), 이동 뒤 손 안 peg가 틀어짐 15건(순간이동한 box가 바로 앞의 peg를 밀어 TCP 기준 상대 위치가 무효가 됨).
  - `--reobs-always`: 감지와 상관없이 삽입 직전 한 번 비켜서 다시 보고, 그동안 손 기준 peg 위치를 새로 학습한다 → **eval3 box_move 35% → 60%** (+0.25 [0.125, 0.40], 10/0), 일반 조건 손해 없음.

### B5. 실시간성
- tracking + point cloud: p95 45 ms(E2). CPU point cloud(39 ms)와 PIL 전처리(23 ms)를 GPU로 옮긴 것이 핵심이다.
- pose 추정: 다중 가설 탐색이 CPU 242 ms였다. 직전 자세에서 refine 1회만 하고 잔차 < 1.5 mm, free-space 위반 < 10%이면 채택하는 빠른 경로로 **12.6 ms**. FoundationPose의 tracking 모드와 같은 구조다([arXiv 2312.08344](https://arxiv.org/pdf/2312.08344)).
  - 같은 10 ep에서 평균 오차 10.7 → 5.7 mm(축 뒤집힘 감소, 축 평균 9.3° → 1.9°), p90은 5.2 → 6.2 mm.
  - 빠른 경로 판정이 표면 잔차만 봐서 앞뒤 뒤집힌 자세를 통과시키던 버그를 고쳤다(색·테이블 잔차 포함). `scripts/test_pose_v1.py`가 이 경우를 검사한다.

### B6. RGB-only depth
- oracle scale로 맞춰도 peg 중심 3–7 cm 오차다(E7). 물체 경계의 depth 번짐(flying pixel, [arXiv 2110.05885](https://arxiv.org/pdf/2110.05885))과 면 평탄도 붕괴가 원인으로 보인다.
- mm 단계에서는 depth 센서가 필수다. 경계가 날카로운 metric 모델(Depth Pro, [arXiv 2410.02073](https://arxiv.org/html/2410.02073v1))도 mm 수준은 기대하기 어렵다.

### B7. 초기 검출
- **관측(E0)**: "cardboard box" top-1이 peg를 잡거나, 손을 peg로 잡는 경우가 있었다. peg/box 공동 배정으로 eval hit 0.95 / 1.00.
- **문헌**: Grounding DINO는 category token이 region-text 정렬을 지배해 세밀한 속성 구분에 약하다([arXiv 2605.18023](https://arxiv.org/pdf/2605.18023), [arXiv 2305.13921](https://arxiv.org/pdf/2305.13921), [arXiv 2601.23281](https://www.alphaxiv.org/abs/2601.23281)).
- **시작 시점 실패 분석**: tcp2 실패 13건 중 5건이 grasp 전 peg 추정 오류(18–140 mm)였다. GT mask로 원인을 나누면:
  1. 두 카메라 모두에서 peg가 거의 안 보이거나 끝면만 보임(214, 217, 233) → 관측 한계
  2. base는 틀리고 hand는 맞음(225)
  3. mask는 정확한데 점 표본에 따라 추정이 흔들림(215, 1–17 mm) → 첫 추정에 점 1,500개를 써서 최대 17.4 → 2.2 mm
- (1)은 로봇이 움직여 보는 수밖에 없어 **"다가가서 보고 잡기"(`--look`)**를 넣었다. 정책이 원래 지나는 peg 위 5 cm 접근 지점에서 손목 카메라로 peg를 다시 추정한다(GT mask 기준 중앙 0.6 mm).
  - look v2(base 판정이 불확실하거나 두 추정이 1.5 cm 안에서 일치할 때 채택): eval 순효과 0. 노린 사례는 성공으로 바뀌었지만, 원래 성공하던 3건이 불필요한 채택으로 실패했다.
  - look v3(시작 프레임 base 추정이 불확실할 때만 채택): eval2 **+0.125 [+0.025, +0.225] (5/0)**.
- **시작 탐색(`--search`)**: 시작 프레임에 peg 추정이 없을 때 손을 peg 근처 위 25 cm로 옮겨 찾는다.
  - v1(추정이 불확실할 때도 탐색)은 eval4에서 0.725 vs 0.825로 해로웠다. 높이 25 cm의 hand 추정(17–50 mm)을 채택했고, 탐색 뒤 상태 플래그를 잘못 바꿔 접근 지점 look이 막혔다.
  - v2(추정이 아예 없을 때만)는 eval5에서 발동 조건이 생기지 않아 효과를 검증하지 못했다. eval3 해당 3건(402, 417, 437)은 직접 실행에서 모두 성공했다.

---

## 4. 실험별 결과

### E0. 환경과 데이터셋
- 수집 `scripts/collect_v1.py`, `scripts/run_e0_collect.sh`. base_camera / hand_camera 512², GT seg / pose / TCP 저장.
- **정책 4종** (seed % 10): expert(0–3) / noisy(4–6, 중간 waypoint에 ±1–2 cm·±5°) / retry(7–8, 첫 삽입을 1.2–2 cm 어긋나게 → 후퇴 → 재삽입) / fail(9, 2.5–3.5 cm 어긋남, 재시도 없음). fail을 1–2 cm로 두면 peg가 미끄러져 들어가 성공해 버려서(seed 109) 키웠다.
- **교란 세트**:
  - `box_move`: 마지막 refine 직후 box를 xy로 2–5 cm 순간이동. expert는 GT로 다시 refine한다.
  - `lighting`: 조명 세기·색 무작위(`PegInsertionSideLit-v1`). minimal shader라 효과가 약해서 E8에서 default shader를 같이 본다.
  - `patch` / `dropout`: 로드 시 `scripts/v1data.py`가 seed로 적용한다. patch = 물체를 1 s 동안 회색 판으로 가림, dropout = 1 s 동안 RGB·depth 0.

| set | ep | 길이 평균 (min–max) | 성공률 | peg 가림 구간 (수, 평균/최대, ep) | box 가림 구간 |
|---|---|---|---|---|---|
| dev | 10 | 139.8 (113–175) | 0.90 | 8, 11.8/21, 4 | 16, 14.7/57, 7 |
| eval | 40 | 143.3 (104–198) | 0.90 | 21, 13.3/59, 12 | 57, 16.5/91, 31 |
| dev_box_move | 10 | 161.3 (136–195) | 1.00 | 8, 11.8/21, 4 | 16, 14.8/57, 7 |

- 스크립트 정책은 1 cm/step 보간이라 episode가 104–198 step으로 env 기본 제한 100 step을 넘는다(100 step 안 성공은 expert의 6%). 수집은 `max_episode_steps=1000`으로 했다.
- **초기 검출 규칙** (`scripts/init_v1.py`, dev에서 정함): v0의 물체별 GDINO top-1은 dev 10 ep 중 3 ep에서 틀렸다. v1은 peg/box 후보를 **공동 배정**한다(두 bbox IoU < 0.3, "미검출" 점수 0.3 선택 가능, 점수 합 최대). frame 0에 없는 물체는 5 프레임마다 재검출하고 2번 연속 같은 자리면 늦게 초기화한다. 결과 dev 0.90 / 1.00, **eval 0.95 / 1.00**. GT 대체는 하지 않는다.

### E1. v0 결론 재검증 (eval 40 ep)
tracker 5종 × re-anchor {none, conf, conf_e5}. 초기 bbox는 `init_v1.py`(GT 대체 없음). 3개 병렬 실행이라 시간은 비교용이 아니다(단독 측정은 E2). 전체 표 `results/v1/E1/table_eval_main.md`.

| config | peg IoU [CI] | Δpeg vs sam2s_none [CI] (승/패) | box IoU | Δbox | peg 재획득 | 가림 중 FP peg/box | GDINO 호출 / 수락 |
|---|---|---|---|---|---|---|---|
| sam2s_none | 0.760 [0.673, 0.835] | – | 0.973 | – | 0.58 | 0.057 / 0.002 | 0 / 0 |
| sam2s_conf | 0.774 [0.690, 0.847] | +0.011 [0.000, 0.034] (1/0) | 0.972 | −0.000 | 0.63 | 0.057 / 0.005 | 1410 / 2 |
| sam2s_conf_e5 | 0.774 [0.690, 0.847] | +0.011 [0.000, 0.034] (1/0) | 0.973 | 0.000 | 0.63 | 0.057 / 0.002 | 320 / 1 |
| sam2s_c_none (compile) | 0.773 [0.690, 0.846] | +0.015 [−0.005, 0.050] (3/3) | 0.973 | 0.000 | 0.68 | 0.004 / 0.002 | 0 / 0 |
| sam2bp_none (base+) | 0.714 [0.632, 0.786] | **−0.046** [−0.090, −0.008] (7/22) | 0.973 | 0.000 | 0.63 | 0.018 / 0.010 | 0 / 0 |
| etam_ti_none | 0.716 [0.636, 0.786] | **−0.043** [−0.068, −0.020] (4/23) | 0.947 | **−0.032** | 0.68 | 0.000 / 0.012 | 0 / 0 |
| samurai_s_none | 0.773 [0.691, 0.846] | +0.013 [−0.008, 0.050] (6/5) | – | – | – | – | 0 / 0 |

| config | peg clean | peg grasp 후 | peg 삽입 중 |
|---|---|---|---|
| sam2s_none | 0.924 | 0.722 | 0.616 |
| sam2bp_none | 0.925 | 0.665 | 0.573 |
| etam_ti_none | 0.950 | 0.663 | 0.542 |

- **compile**: +0.015 [−0.005, 0.050], 3승 3패 → 효과 없음. v0의 0.73↔0.92는 ep001 경계선 사례 하나에서 나왔다. 속도용으로만 켠다.
- **v0 re-anchor(conf)**: +0.011 [0.000, 0.034], 40 ep 중 1 ep만 개선, GDINO 1,410회 호출 중 수락 2회. re-ID 임계값(SAM2 embedding cos ≥ 0.75)이 보수적이라 대부분 재획득하지 못한다.
- **conf vs conf_e5**: 정확도 같고 conf_e5의 GDINO 호출이 4.4배 적다.
- **SAMURAI**: 차이 없음, 물체당 따로 돌아야 해서 비용 약 2배. 중간 프레임 box 프롬프트와 비호환이라 re-anchor 조합은 비교하지 않았다(`docs/known_issues.md`).
- 실패 유형(`gallery_sam2s_none_peg.jpg`): (1) frame 0에 peg가 화면 가장자리에 걸쳐 초기화가 틀림(ep214, ep208), (2) grasp 뒤 mask가 손 전체로 번짐(ep226), (3) 팔 가림 뒤 영구 소실(ep217, ep207).

### E2. Causal streaming + 20 Hz
구현 `scripts/stream_v1.py`(`CausalFrames`, `StreamTracker`), `scripts/pc_gpu.py`(GPU point cloud + triton FPS kernel), `scripts/e2_stream.py`.
- `CausalFrames`는 SAM2 `inference_state["images"]` 자리에 들어가는 버퍼로, 아직 push하지 않은 프레임이나 이미 버린 프레임을 읽으면 `FutureFrameAccess` 예외를 낸다. SAM2 `init_state`가 frame 0을 미리 읽는 것을 이걸로 잡아 순서를 고쳤다.
- 측정: eval ep200–209(1,550 프레임), GPU 단독, 물체 2개, 워밍업 1 ep 제외, CUDA sync 후 프레임별 시간.

| variant | track p50 | pc p50 | **합계 p50 / p95 / max (ms)** | FPS | GPU GB | peg / box IoU | offline 대비 mask IoU peg / box |
|---|---|---|---|---|---|---|---|
| stream | 70.0 | 39.0 | 107.1 / 115.0 / 210 | 9.3 | 1.03 | 0.750 / 0.976 | 1.0000 / 1.0000 |
| + compile | 66.5 | 40.7 | 105.5 / 117.4 / 130 | 9.5 | 0.89 | 0.752 / 0.976 | 0.9866 / 0.9990 |
| + GPU pc | 64.0 | 4.2 | 68.0 / 70.4 / 79 | 14.7 | 0.89 | 0.752 / 0.976 | 0.9906 / 0.9990 |
| + GPU 전처리 | 40.6 | 4.3 | 44.8 / 45.5 / 457 | 22.3 | 0.89 | 0.753 / 0.973 | 0.9852 / 0.9945 |
| **+ memory 정리 (기본)** | 40.5 | 4.1 | **44.5 / 45.1 / 47** | **22.5** | **0.61** | 0.756 / 0.973 | 0.9913 / 0.9944 |
| + cam 384 | 40.4 | 3.5 | 43.8 / 44.4 / 50 | 22.8 | 0.62 | 0.732 / 0.941 | 0.9503 / 0.9576 |
| compile 없이, image_size 768 | 33.4 | 4.0 | 37.4 / 42.7 / 59 | 26.8 | 0.66 | 0.739 / 0.967 | 0.9584 / 0.9764 |
| compile 없이, image_size 512 | 32.9 | 4.1 | 37.0 / 41.2 / 54 | 27.0 | 0.50 | 0.703 / 0.927 | 0.9211 / 0.9388 |

- streaming은 offline과 비트 단위로 같다(PIL 전처리, 10 ep 전 프레임). compile이나 GPU 전처리를 넣으면 mask가 조금 달라지지만(0.985–0.999) GT IoU는 같다.
- 효과가 큰 순서: GPU point cloud 39 → 4 ms, GPU 전처리 −23 ms(PIL bicubic 1024 resize), compile −3–6 ms, memory 정리로 GPU 0.89 → 0.61 GB와 시간 꼬리 제거(max 457 → 47 ms).
- 해상도를 낮추면 이득은 작고 정확도 손해가 커서 기본 해상도를 유지한다. image_size 변경과 compile을 같이 쓰면 cudagraph와 position encoding 캐시가 충돌한다. weight 전체 bf16은 conv dtype 오류로 중단했다(이미 bf16 autocast 사용 중).
- **re-anchor 비용** (단독, eval patch 10 ep): none p95 50.8 ms, v1(소실 동안 매 프레임 GDINO + DINOv2 + 후보 mask)은 p95 316 / max 435 ms. 실시간에서는 재검출을 tracking과 분리해 5 프레임 간격이나 비동기로 돌리고, 그동안 마지막 값과 `age_steps`를 낸다.

### E3. Re-anchor v1
구현 `scripts/reanchor_v1.py`, 실행 `scripts/e3_run.py`, 게이트 학습 `e3_calib.py` / `e3_fit_gate.py` / `e3_sim_gate.py`.
- **보정 prompt**: memory를 재초기화하지 않고 이미 추적 중인 프레임에 `add_new_points_or_box` / `add_new_mask`로 conditioning을 추가한다. frame 0 memory는 유지한다.
- **트리거**: 면적 급증, mask 내부 depth 두 덩어리, 3D 길이 부족, 중앙 depth 급변을 구현했지만 dev shadow 실행(후보를 받지 않고 기록만)에서 거의 전부 오경보였다. 가림이 시작·해제될 때도 같은 변화가 정상적으로 생기기 때문이다. 기본은 `lost` 트리거만 쓴다.
- **후보 검증**: GDINO 후보마다 SAM2 decoder로 mask를 만들고 5개 특징(DINOv2-S cosine, 면적비, 3D 길이비, 마지막 3D 위치와의 거리, GDINO score)으로 물체별 로지스틱 회귀 게이트를 학습했다(dev shadow 후보 2,658개, 정답 359개, `results/v1/E3/gate.json`). 임계값은 "2 프레임 확인 후 소실 구간마다 첫 수락"을 shadow 기록 위에서 시뮬레이션해 정했다(peg p ≥ 0.8, box p ≥ 0.5).
  - DINOv2 cosine 하나로는 peg 정답과 오답이 잘 갈리지 않는다(정답 중앙 0.54 / 오답 0.35). grasp 후 외형 변화가 크다.
- **TCP prior** (ablation `v1_tcp`): grasp 후 peg 후보는 "TCP × 마지막으로 본 TCP→peg 변환" 예측 위치에서 3 cm 이내여야 받는다.

dev 결과 (10 ep, none 대비 Δpeg):

| 설정 | dev | patch | dropout | box_move | 가림 중 FP peg |
|---|---|---|---|---|---|
| none | 0.777 | 0.731 | 0.668 | 0.762 | 0.006 |
| v0conf | −0.004 | **−0.075** [−0.219, 0.034] | +0.058 [0.010, 0.107] | −0.007 | 0.006 |
| v1 | 0.000 | +0.016 [0.000, 0.049] | +0.047 [0.008, 0.099] | 0.000 | **0.108** |
| v1_tcp | 0.000 | 0.000 | **+0.053** [0.014, 0.102] | 0.000 | 0.006 |

eval 결과 (40 ep, 1회):

| 세트 | none peg IoU | v1 Δpeg | v1_tcp Δpeg | v1 Δbox | 가림 중 FP box (none → v1) | peg 재획득률 (none → v1) |
|---|---|---|---|---|---|---|
| eval | 0.778 [0.694, 0.853] | 0.000 | 0.000 | −0.014 [−0.035, 0.000] | 0.002 → 0.023 | 0.74 → 0.74 |
| patch | 0.737 [0.662, 0.805] | **+0.025** [0.008, 0.047] (9/0) | +0.016 [0.003, 0.034] | −0.018 [−0.036, −0.003] | 0.031 → 0.199 | 0.57 → 0.72 |
| dropout | 0.714 [0.633, 0.788] | **+0.024** [0.005, 0.046] (6/0) | +0.023 [0.005, 0.046] | −0.013 [−0.033, 0.000] | 0.006 → 0.016 | 0.67 → 0.78 |
| box_move | 0.761 [0.675, 0.838] | 0.000 | 0.000 | −0.015 [−0.038, 0.000] | 0.003 → 0.026 | 0.74 → 0.74 |

- peg: 가림·끊김 뒤 재획득이 유의하게 좋아진다. eval2에서도 재현됐다(patch +0.020 [0.004, 0.040], dropout +0.028 [0.005, 0.061]).
- box: 가려진 동안 다른 것을 box로 받아들여 해롭다. box는 re-anchor 없이도 재획득 0.98이다.
- TCP prior는 mask 수락 게이트로는 추가 이득이 없다(후보가 없으면 소용없음). 반면 pose 단계에서 쓰면 성공률이 +0.10–0.175 오른다(E5). **로봇 상태는 mask 재획득보다 pose 안정화에 쓰는 편이 효과적이다.**

### E4. Pose / hole pose 추정
추정기 `scripts/pose_v1.py`, 평가 `scripts/e4_pose.py`. 입력은 rgb, depth, K, cam2world와 task 파라미터(peg L, r)뿐이다.
- **peg**: 직육면체 SDF 최소화(scipy least_squares, soft-L1) + 색 prior(head 주황 / tail 흰색) + 테이블 prior(꼭짓점 z ≥ 0). PCA 3축 × 2방향과 이전 프레임 자세에서 다중 시작하고 **free-space 검사**(모델 표면이 보여야 할 곳의 관측 depth가 더 멀면 위반)로 가설을 고른다. 끝면만 보이는 경우의 모호성을 이것으로 해결했다.
- **hole**: box cloud에서 RANSAC으로 수직 평면을 찾고 인라이어로 법선을 정밀화한다. "면보다 2.5 mm 이상 뒤가 보이는 픽셀"의 연결 영역 중 테두리에 닿지 않고 크기가 맞는 것을 hole로 고른다(팔 가림은 면보다 앞이라 구분된다). dev에서 법선 정밀화로 축 오차 0.74° → 0.01°, 임계값 6 → 2.5 mm로 측면 오차 1.13 → 0.63 mm.

"Vision이 줄 수 있는 state" (eval 40 ep, base_camera, 중앙값 / p90):

| state | GT mask 입력 | tracker mask 입력 | 비고 |
|---|---|---|---|
| peg 표면 centroid vs peg 중심 (mm) | 79.38 / 100.20 | 78.16 / 103.86 | 보이는 면 bias. pose로 쓰면 안 된다 |
| **peg 6D 중심 위치 (mm)** | 0.51 / 3.15 | 1.16 / 24.01 | grasp 후 0.50 / 2.70 (GT mask), 0.93 / 53.29 (tracker) |
| peg 축 방향 (deg) | 0.08 / 0.28 | 0.16 / 6.22 | |
| peg head 끝 (mm) | 0.52 / 3.08 | 1.07 / 28.87 | |
| **hole 중심 (mm)**, 직접 보일 때 | 0.81 / 4.64 | 0.81 / 4.64 | 직접 보이는 프레임 30% |
| hole 중심 (mm), 마지막 값 유지 포함 | 1.03 / 4.02 | 1.03 / 4.02 | box 정지 가정 |
| hole 축 (deg) | 0.01 / 0.02 | 0.01 / 0.02 | |
| peg 추정 실패율 | 0.000 | 0.102 | tracker 소실이 그대로 들어온다 |

- fitting 자체는 중앙값 기준으로 E5가 요구하는 정밀도를 만족한다. 꼬리(p90 3 mm)는 grasp 구간에서 나온다. tracker mask를 넣으면 p90이 24 mm로 커져서, **pose 꼬리 오차의 주원인은 tracking 실패**다.
- 계획의 프레임 간 smoothing(KF)은 따로 만들지 않았다. 직전 자세에서 refine만 하는 빠른 경로(B5)와 손에 쥔 peg의 시간 게이트(B2)가 같은 역할을 한다.
- 크기를 관측에서 추정(`--size est`)하면 peg L 오차 중앙 7.7 mm(최대 89 mm), peg 위치 오차 중앙 10 mm / p90 76 mm로 커진다. 첫 프레임 peg가 잘리거나 가려진 경우가 많기 때문이다. 물체 사양(L, r)이 사실상 필요하다.

### E5. Vision-in-the-loop 스크립트 정책
`scripts/e5_loop.py`: expert 정책의 waypoint 계획에서 GT peg pose(grasp 계획, refine)와 hole pose를 vision 추정으로 바꿨다. 실패 단계는 GT로 판정한다: align = 삽입 직전 peg head가 hole 축에서 여유 간격(3 mm)보다 벗어남, insert = 정렬은 됐는데 못 들어감. 성공률은 시간 제한 없이 쟀다.

**GT 노이즈 곡선** (eval 40 ep, 질의할 때마다 노이즈를 새로 뽑음):

| 조건 | 성공률 | 정렬 오차 중앙 (mm) |
|---|---|---|
| GT | 1.000 (40/40) | 0.02 |
| 위치 σ 0.5 mm | 0.925 | 0.8 |
| 위치 σ 1 mm | 0.825 | 1.59 |
| 위치 σ 2 mm | 0.525 | 3.21 |
| 위치 σ 5 mm | 0.075 | 8.04 |
| 위치 σ 10 mm | 0.075 | 14.25 |
| 회전 σ 1° | 0.375 | 3.62 |
| 회전 σ 3° | 0.050 | 10.38 |
| 회전 σ 5° | 0.000 | 15.65 |
| 위치 1 mm + 회전 1° | 0.300 | 4.04 |
| GT + box_move (재계획) | 0.950 | 0.09 |

- 성공률은 1–2 mm에서 급격히 떨어진다(hole 여유 3 mm). 회전이 더 치명적이다. 1°가 peg 반길이(약 10 cm)에 걸리면 끝에서 약 1.7 mm가 된다.
- **요구 정밀도: 위치 ≲ 1 mm, 축 ≲ 0.3°.** Graph/Controller 팀에도 같은 기준을 쓸 수 있다.

**개선 과정**:

| 단계 | 변경 | 원인 관찰 | 결과 |
|---|---|---|---|
| 1 | vision 추정으로 정책 실행 | 첫 실행 정렬 오차 10–18 cm. 추정은 정확했고(peg 0.5 mm) 원인은 정사각 단면 peg의 90° roll 모호성 | roll 정규화 후 47.5% |
| 1' | hole 추정 정밀화, peg 축 가설 보강 | 정렬 오차 2.2 / 3.9 mm로 여유 간격 근처 | 52.5% [0.375, 0.675] |
| 2 | 손에 쥔 peg를 TCP로 보정(`tcp2`) | 그리퍼가 peg 중간을 가려 축 방향 위치가 안 정해짐(B2) | **70%** (+0.175 [0.05, 0.325], 8/1). vision만 시간 평균은 효과 0 |
| 2' | box 이동 시 비켜서 다시 보기(`--reobserve`) | 이동 뒤 peg가 hole을 가림(B4) | box_move 15% → 37.5% |
| 3 | pose 빠른 경로 + 시작 추정 안정화 | 시작 추정이 점 표본에 따라 1–17 mm 흔들림(B7) | 75%, 루프 575 → 270 ms |
| 4 | 다가가서 보고 잡기(`--look --look-rule v3`) | 시작 시 peg 끝면만 보이면 잡기 실패 | eval2 72.5% → **85%** (+0.125 [0.025, 0.225], 5/0) |
| 5 | 삽입 직전 항상 다시 보기(`--reobs-always`) | 이동 감지 실패 + 충돌로 손 안 peg 밀림 | eval3 box_move 35% → **60%** (+0.25 [0.125, 0.40], 10/0), 일반 손해 없음 |
| 6 | 최종 확인 | – | eval4 82.5%, eval5 85% |

2단계 상세 (eval 40 ep):

| 조건 | 성공률 [95% CI] | 정렬 오차 중앙 (mm) |
|---|---|---|
| GT | 1.000 | 0.02 |
| vision만 | 0.525 [0.375, 0.675] | 1.99 |
| vision만 + temporal | 0.525 [0.375, 0.675] | 1.52 |
| **vision + TCP (tcp2)** | **0.700** [0.550, 0.850] | 0.77 |
| vision + TCP + fast-pose + reobserve | 0.675 [0.525, 0.825] | 0.96 |
| box_move: GT 재계획 | 0.950 | 0.09 |
| box_move: vision + TCP | 0.150 [0.050, 0.275] | 11.8 |
| **box_move: vision + TCP + reobserve** | **0.375** [0.225, 0.525] | 9.1 |

4단계 (eval2, 처음 보는 seed 300–339):

| 조건 | 성공률 [95% CI] | 실패 단계 |
|---|---|---|
| GT | 1.000 | – |
| vision만 | 0.550 [0.400, 0.700] | align 14, 시작 추정 없음 2, insert 2 |
| vision만 + look v3 | 0.550 [0.400, 0.700] | align 13, 시작 추정 없음 2, insert 3 |
| vision + TCP | 0.725 [0.575, 0.850] | align 8, 시작 추정 없음 2, insert 1 |
| **vision + TCP + look v3** | **0.850** [0.725, 0.950] | insert 3, 시작 추정 없음 2, align 1 |

- look v3는 vision + TCP에서만 효과가 있다(+0.125). vision만에서는 잡는 위치가 좋아져도 손에 쥔 뒤 추정(B2)에서 막힌다.
- TCP 효과는 새 세트에서도 재현된다(+0.175 [0.050, 0.325], 8/1).

5단계 (eval3, seed 400–439):

| 조건 | 성공률 [95% CI] | step 중앙 |
|---|---|---|
| GT | 1.000 | 134 |
| 기준 (vision + TCP + look v3 + reobserve) | 0.750 [0.625, 0.875] | 135 |
| + 항상 다시 보기 | 0.775 [0.650, 0.900] | 208 |
| box_move: GT 재계획 | 0.975 | 160 |
| box_move: 기준 | 0.350 [0.200, 0.500] | 211 |
| **box_move: + 항상 다시 보기** | **0.600** [0.450, 0.750] | 288 |

- 항상 다시 보기는 일반 조건에서 손해가 없지만 시간이 약 70 step(3.5 s) 는다. eval3에서는 시작 시 peg가 안 보이는 경우(4/40)가 남은 실패의 큰 몫이다.

6단계 (최종 설정 = vision + TCP + look v3 + 항상 다시 보기 + fast-pose + reobserve):

| 조건 | 성공률 [95% CI] | 실패 단계 |
|---|---|---|
| eval5 GT | 1.000 | – |
| eval5 vision만 (temporal) | 0.625 [0.475, 0.775] | align 14, insert 1 |
| **eval5 최종 설정** | **0.850** [0.725, 0.950] | insert 3, align 3 |
| eval4 최종 설정 | 0.825 [0.700, 0.925] | align 4, insert 2, 시작 추정 없음 1 |
| eval4 최종 + 시작 탐색 v1 | 0.725 [0.575, 0.850] | 탐색 발동 9회 |

- 최종 설정은 처음 보는 세 세트(eval2/4/5)에서 85% / 82.5% / 85%다. eval5에서 vision만 대비 +0.225 [0.100, 0.350] (9/0), GT와의 격차 +0.150 [0.050, 0.275] (6/0).
- eval4 box_move는 서버 한도 및 비용 문제로 22/40 ep에서 중단했다(그 시점 12/22). 같은 설정의 box_move 확정치는 eval3의 60%다.
- **eval 재실행에 대해**: 1'단계(vision 2차)는 eval 1차 실패를 보고 고친 뒤 dev로 확인하고 eval을 다시 돌린 것이라 "eval 1회" 원칙에서 벗어난다. 1차(47.5%)와 2차(52.5%)를 모두 남겼고, 4단계부터는 처음 보는 seed 세트에서만 판정했다.

### E6. Task relevance score
`scripts/e6_relevance.py` (E9 export 입력, 3 프레임 간격, eval 40 ep).
- GT 라벨(평가 전용): peg는 head 끝 3 cm ∪ grasp 후 TCP에서 축 방향 2 cm 이내, box는 hole 입구 평면 ±1 cm에서 hole 반폭 + 1.5 cm 이내.
- 후보: const(전부 1.0), geom(같은 규칙을 **추정** pose로 계산), dist(TCP 또는 추정 hole 중심과의 거리).

| 물체 | const AUROC / AP | geom AUROC / AP | dist AUROC / AP |
|---|---|---|---|
| peg | 0.500 / 0.294 | **0.941 / 0.839** | 0.729 / 0.719 |
| box | 0.500 / 0.035 | **1.000 / 0.999** | 0.877 / 0.152 |

- pose가 약 1 mm로 맞으면 relevance는 기하 규칙만으로 거의 완벽하다. peg 0.94는 tracking이 실패한 프레임에서 깎인 값이다.
- v1 범위에서 VLM이나 언어 조건 feature는 필요 없다. CLIP/SigLIP patch similarity, Qwen2-VL pointing은 물체·작업이 바뀌어 기하 규칙을 손으로 쓸 수 없을 때의 후보로 남긴다.

### E7. RGB-only depth
`scripts/e7_depth.py`. eval 앞 10 ep, 5 프레임 간격, GT mask 안에서 sim depth와 비교. scale은 none(그대로), median(프레임마다 물체 밖 sim depth 중앙값으로 맞춤, oracle 상한), table(첫 프레임 테이블로 1회 보정 후 고정, 배포 가능)으로 맞췄다.

| 모델_scale | peg AbsRel / δ1 | peg cloud 중심 오차 (mm) | box 중심 오차 (mm) | peg pose 오차 (mm / deg) | ms/frame |
|---|---|---|---|---|---|
| DA V2 metric_none | 2 / 0 | 537 | 573 | 487 / 49.5 | 38.4 |
| DA V2 metric_median | 0.19 / 0.804 | 51.6 | 94 | 52.7 / 14 | 34.0 |
| DA V2 metric_table | 0.572 / 0.0073 | 164 | 154 | 137 / 22.2 | 36.6 |
| DA V2 rel_median | 0.257 / 0.21 | 68.9 | 65.2 | 65.5 / 14.1 | 35.4 |
| YOLO26x-depth_median | 0.134 / 0.874 | 27.3 | 212 | 58.1 / 38.7 | 24.0 |
| YOLO26x-depth_table | 0.506 / 0.0216 | 142 | 199 | 149 / 37.9 | 24.1 |

(YOLO26 n/s/m/l/x 전체는 `results/v1/E7/yolo26.json`. 값은 프레임 중앙값.)

계획에 있던 Video Depth Anything과 Metric3D v2는 DA V2 결과로 보아 결론을 바꿀 만한 개선을 기대하기 어려워 돌리지 않았다. 대신 학습 파이프라인까지 제공되는 최신 모델인 Ultralytics YOLO26 depth 5종(n/s/m/l/x, 별도 venv, `imgsz=768`)을 같은 조건으로 측정했다.

- metric 모델은 sim 렌더에서 scale이 2–3배 틀린다. 실내 실사로 학습한 모델이 flat shading 이미지에 맞지 않는 것으로 보인다.
- **oracle scale로 맞춰도 peg 중심 오차 3–7 cm, pose 오차 5–7 cm**다. 요구 정밀도(약 1 mm / 0.3°)와 두 자릿수 차이가 난다. 배포 가능한 1회 보정은 15–27 cm다.
- depth 센서(또는 stereo)를 전제로 하고, RGB-only는 cm 단위 coarse 위치(접근 단계)에만 쓰는 것을 권한다. 다음 후보는 sim RGB-depth 쌍으로 fine-tuning이다(실제 카메라 일반화는 따로 검증해야 한다). Ultralytics는 AGPL-3.0이라 배포 방식에 따라 라이선스 확인이 필요하다.

### E8. 도메인 강인성 (부분)
`scripts/e8_domain.py`. 기본(minimal shader), `lighting`(eval 40 ep), `default` shader(그림자·음영, eval seed 200–219 재수집)를 비교했다. HSV 기준선은 dev 기본 렌더의 GT box 픽셀 HSV 2–98 percentile 범위의 최대 연결 성분이다(직접 만든 단순 버전). peg는 흰 꼬리가 흰 로봇 팔과 색으로 구분되지 않아 HSV 기준선을 만들지 않았다.

| 렌더 | HSV box IoU [CI] | SAM2.1-small box IoU [CI] | SAM2.1-small peg IoU [CI] |
|---|---|---|---|
| 기본 (eval 40) | 0.932 [0.914, 0.950] | 0.973 [0.969, 0.976] | 0.760 [0.673, 0.835] |
| lighting (eval 40) | **0.003** [0.000, 0.009] | 0.960 [0.938, 0.972] | 0.733 [0.653, 0.806] |
| default shader (eval 20) | 0.842 [0.732, 0.929] | 0.972 [0.967, 0.978] | 0.765 [0.624, 0.885] |

- HSV는 flat shading 덕을 봤다. 조명 색·세기만 바꿔도 거의 0이 된다. GDINO + SAM2는 세 조건에서 거의 그대로다(lighting 세트 GDINO hit peg 0.97 / box 0.97).
- 미실행: hand_camera 단독 tracking, base + hand cloud 융합. hand camera는 아래를 내려다봐서 삽입 직전에도 hole 면이 보이지 않아 hole 정밀화에 바로 쓸 수 없다고 보고 우선순위를 낮췄다.

### E9. 다른 팀용 export
`scripts/e9_export.py`, `scripts/run_e9.sh`. 입력은 E1에서 정한 tracker(SAM2.1-small + compile + conf_e5)다. E3 결론(peg에만 re-anchor v1)보다 먼저 만든 export라서, 최종 설정으로 다시 만들려면 tracker 설정만 바꿔 다시 돌리면 된다. 출력 `data/v1/export/eval/<ep>/f####.npz`(283 MB, 5,732 프레임, git 제외)와 episode별 `meta.json`.

| key | 내용 |
|---|---|
| `{peg,box}_points_world` | (1024, 3) world 좌표. 1024개 미만은 0으로 채움 |
| `{peg,box}_points_ee` | TCP 기준 상대 좌표 |
| `{peg,box}_valid_pts` | padding이 아닌 점 |
| `{peg,box}_relevance` | 점별 relevance (현재 1.0, E6 참고) |
| `{peg,box}_valid`, `_age_steps`, `_confidence` | 관측 유효 여부, 마지막 유효 관측 이후 step 수, sigmoid(SAM2 logit) |
| `peg_pose`, `hole_pose` | [x y z qw qx qy qz] (E4 추정기), `*_pose_valid`, `*_pose_age_steps`. 추정이 없으면 마지막 값 유지 + age 증가 |
| `tcp_pose`, `frame_idx`, `timestamp` | |
| `gt_peg_pose`, `gt_box_hole_pose`, `gt_tcp_pose` | 비교·학습용 GT (모델 입력 아님) |

- peg pose는 보이는 축 길이 ≥ 0.6 L일 때만 유효로 낸다(화면 가장자리에 조금 보일 때의 수 cm 오차 제거).
- 유효 비율: 점 cloud peg 0.86 / box 0.86, pose peg 0.50 / hole 0.13(hole은 box 정지 가정으로 첫 추정값 유지). ep200 유효 프레임 peg pose 오차 중앙 1.1 mm, p90 2.4 mm.

---

## 5. 효과가 없었거나 해로웠던 시도

- re-anchor 기하 트리거 4종: 오경보 다수 (E3)
- DAM4SAM: 유의한 차이 없음, grasp 후 오히려 낮음 (B3)
- SAM2Long: eval에서 +0.050이었지만 새 세트에서 재현 안 됨, 삽입 성공률 개선 없음 (B3)
- vision만 시간 평균(`--inhand temporal`): 효과 0 (B2)
- 시작 시 hand camera 선택(`--hand-init`): dev 5 ep에서 오히려 나빠져 중단 (B7)
- look v2: 순효과 0 (B7)
- 시작 탐색 v1: eval4에서 해로움 (B7)
- RGB-only depth 전부 (E7)
- weight 전체 bf16: dtype 오류 (E2)

---

## 6. 열린 질문 (10/28 회의 안건)

1. **로봇 상태(TCP)를 vision에 써도 되는가?** mask 재획득 게이트로는 이득이 없었고(E3), pose 단계에서 쓰면 삽입 성공률이 +0.175다(E5). 최종 85%도 TCP를 전제로 한다.
2. **물체 사양(peg L, r)을 vision에 줘도 되는가?** 관측으로 크기를 추정하면 pose 오차가 0.5 → 10 mm로 커진다(E4).
3. **Graph / Controller 팀에 넘길 출력**: E9 포맷으로 맞출지. 표면 cloud만으로는 pose를 얻을 수 없으니(centroid bias 79 mm) pose 채널을 같이 쓰는 것을 권한다. Graph 팀이 지금 GT 형상으로 만드는 표면 graph와 vision이 주는 "보이는 면" cloud는 분포가 다르다는 점도 맞춰야 한다.
4. **평가 시간 제한**: 스크립트 정책은 100 step(5 s) 안에 거의 끝나지 않는다. 성공률은 시간 제한 없이 쟀다. Controller 팀 평가 조건과 맞출지.
5. **Controller 쪽 compliance**: 남은 실패의 대부분이 mm 단위 정렬·삽입이다(B1). 접촉 기반 보정이나 탐색 삽입을 controller 설계에 넣을지.
6. **hole 정밀도**: 512² base camera는 0.5 m에서 1 px ≈ 2 mm라 지금(0.6–0.8 mm)이 해상도 한계에 가깝다. 카메라 해상도나 배치를 바꿀지.
7. **Vision → controller 결합**: "다가가서 보기", "다시 보기"처럼 vision이 로봇 동작을 요청하는 구조가 효과가 컸다. 모듈 분리 원칙과 어떻게 맞출지.

## 7. 다음 할 일

1. grasp 이후 peg 추적: mask 번짐을 depth / 3D 크기로 거르기(dev 트리거는 오경보가 많아 다시 설계)
2. box_move 대응: box 윗면·옆면 모서리 기반 정합, 또는 hole을 볼 수 있는 접근 경로(정책 쪽 협의)
3. pose 추정 GPU화: 가설 수 축소 + torch least squares, 목표 ≤ 20 ms
4. 시작 시 peg가 안 보이는 경우: 늦은 초기화를 streaming 경로에 넣고 시작 탐색 v2를 발동 조건이 있는 세트에서 검증
5. E8 나머지: hand camera tracking, base + hand cloud 융합
6. 같은 Vision 팀 prototype(`feature/visual-prototype`)과 통합. causal streaming, dev/eval 분리, 좌표 규약(픽셀 중심 u+0.5)은 이미 같은 방식이다

---

## 재현

```bash
cd visual && source env.sh
bash scripts/run_e0_collect.sh && python scripts/init_v1.py --sets dev eval dev_box_move eval_box_move eval_lighting && python scripts/e0_stats.py   # E0
bash scripts/run_e1.sh && python scripts/eval_v1.py --set eval --tags sam2s_none sam2s_conf ... --name eval_main                                                               # E1 (3 병렬, 약 4 h)
bash scripts/run_e2.sh && python scripts/e2_summary.py                                                                                              # E2 (GPU 단독)
SPLIT=dev bash scripts/run_e3.sh; SPLIT=eval CFGS="none v1 v1_tcp" PAR=3 bash scripts/run_e3.sh                                                    # E3
bash scripts/run_e4.sh                                                                                                                              # E4
python scripts/e5_loop.py --cond vision --vcfg none --fast-pose --reobserve --inhand tcp2 --look --look-rule v3 --reobs-always \
       --seeds $(seq 600 639) --name eval5_full                                                                                                     # E5 최종 설정
python scripts/e5_loop.py --cond gt --seeds $(seq 600 639) --name eval5_gt
python scripts/e6_relevance.py; python scripts/e7_depth.py; python scripts/e8_domain.py; bash scripts/run_e9.sh                                     # E6–E9
```

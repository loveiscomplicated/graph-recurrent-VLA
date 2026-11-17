# Task10-G seed98207: 초기 peg 가시성과 평가 해석

대상: Final / B / step009000 / seed98207. 분석일: 2026-10-31.

## 판정

**Simulator의 초기 상태로는 허용되지만, peg가 충분히 보이는 상태에서 집기 능력을 비교하는 조건으로는 관측이 매우 제한적이다.** 첫 프레임 base_camera에는 peg의 일부만 116픽셀 보이고, hand_camera에는 peg가 0픽셀이다. 현재 B의 최종 정책 graph에는 250/250스텝 내내 peg 관측 노드가 없다. 이 에피소드의 실패를 제어 정책 자체의 성능으로만 해석하기 어렵다.

다만 “사람이나 어떤 정책도 절대 성공할 수 없다”는 불가능성 주장은 하지 않는다. 카메라가 달린 팔을 움직여 관측을 확보하거나 사전 분포를 이용하는 방법까지 포함하면 다른 문제다. 현재 결과는 그 같은 탐색/재관측 능력과 분할·가시성 제한이 제어 성능과 함께 섞인 결과다.

## 첫 실제 policy 관측 확인

원래 CUDA 서버에서 같은 factory/seed로 reset한 뒤, 원본 simulator/controller/RNG snapshot SHA가 exact 일치함을 확인했다. 추가 action이나 DINO/SAM2/정책 inference 없이 현재 sensor texture에서 RGB와 평가용 GT segmentation을 읽었다.

[실제 두 카메라 입력과 GT peg 위치](../../outputs/task10g/diagnostics/seed98207_initial_visibility_v1/initial_visibility.png). 청록색은 진단용 GT이며 정책 입력 마스크가 아니다.

| 항목 | Base camera | Hand camera |
| --- | ---: | ---: |
| 실제 입력 해상도 | 128×128 | 128×128 |
| GT peg 가시 픽셀 | **116** | **0** |
| 화면 대비 peg 픽셀 | **0.708%** | **0%** |
| 가시 peg bbox | x=[35,55), y=[117,128) | 없음 |

Base 영상의 맨 아래 11개 행에 peg 일부만 걸린다. Peg 전체가 화면 안에 들어온 상태가 아니며, 손목 영상에는 peg가 전혀 없다. 원본 video는 각 센서를 384×384로 확대하므로 화면상의 크기와 실제 입력 정보량을 구분해야 한다.

초기 peg 중심은 world **[0.010889, -0.297769, 0.022347]m**, half-size는 **[0.101024, 0.022347, 0.022347]m**다. 즉 길이 약 20.2cm의 peg가 샘플 범위의 y 하단(-0.3m) 근처에 놓였다. 설치된 task의 기본 base camera는 [0,-0.3,0.2]에서 [0,0,0.1]을 바라본다. 이 배치와 peg 방향 때문에 대부분이 카메라 시야 밖으로 잘린다.

GT 8개 corner를 base CV camera에 변환하면 일부 corner의 camera-z가 0 이하이다. 그러므로 raw projected corner min/max를 일반적인 2D 물체 bbox나 가시 비율로 해석하지 않는다. 위 픽셀 수는 실제 렌더 GT segmentation에서 직접 계산했다.

## 왜 이런 초기 상태가 통과했는가

설치된 ManiSkill 3.0.1의 `_initialize_episode()`는 peg 중심을 x∈[-0.1,0.1], y∈[-0.3,0]에서 샘플링하고, z는 peg 반높이로 둔다. XY 위치와 z축 회전을 무작위화한다. 이 경로에는 카메라 frustum, 물체 가시 픽셀 수, mask 품질 또는 초기 검출 성공에 따른 rejection sampling이 없다.

Task10-G는 Final seed 98200–98211을 사전에 고정하고 각 seed에 `env.reset(seed=seed)`를 호출한다. 초기 snapshot을 기록하지만, 가시성에 따른 재샘플링이나 관측 확보 동작을 삽입하지 않는다. Seed98207은 이 설정의 허용 범위에 있으므로 그대로 평가된다. 저장된 초기 collision audit에서도 table containment와 collision valid는 true이다. 물리적으로 유효한 배치와 충분히 관측 가능한 배치는 별개의 조건이다.

## 실제 입력과 결과

- B 초기 graph: TCP/Left/Right 각 1노드, Box 128노드, **Peg 0노드**.
- B 전체 250스텝: **Peg 누락 250회**, Box 누락 0회.
- Base RGB에 조금 보인다는 사실이 실제 policy에 peg 정보가 들어갔다는 뜻은 아니다. 이 정책은 raw RGB를 직접 받지 않고 vision에서 만든 graph와 Robot76을 받는다.
- 같은 seed의 A/B/C 초기 snapshot SHA는 모두 동일하다. A도 peg 250/250 누락, C는 247/250 누락이며 세 조건 모두 reach=false다. C의 존재한 3스텝이 올바른 peg였는지는 이번 진단에서 GT 대조하지 않았다.
- GT를 직접 읽는 `ScriptedExpert`가 경로를 계산할 수 있다는 사실은 이미지 기반 학생에게 같은 정보가 있다는 뜻이 아니다. 현재 expert는 peg actor OBB/pose와 goal pose를 직접 사용한다.

## 평가에 대한 권고

현재 결과는 원래 random reset 분포에서의 end-to-end 성능으로 보존하되, 실패 진단에서는 **초기 가시성 부족 / 검출 실패 / 제어 실패**를 분리해서 표시하는 것이 타당하다. Peg가 충분히 보이는 초기 조건에서의 별도 비교가 있어야, 관측 확보 문제와 제어 정책 성능을 구분할 수 있다.

후속 실험에서 task-local 카메라 배치 또는 물체 샘플 영역을 검토할 수 있지만, 최소 가시 픽셀 기준이나 새로운 reset 분포는 근거를 확인한 뒤 사전에 정해야 한다. 기존 Final 결과를 보고 특정 실패 seed만 제외하면 기존 평가와 의미가 달라진다. 이번 작업에서는 카메라/초기화/모델/평가 설정을 변경하지 않았다.

## 근거와 검증 범위

- [가시성 요약](../../outputs/task10g/diagnostics/seed98207_initial_visibility_v1/visibility_summary.json), [검증 집계](../../outputs/task10g/diagnostics/seed98207_initial_visibility_v1/analysis_verification.json), [원본 RGB·GT segmentation·보정 행렬](../../outputs/task10g/diagnostics/seed98207_initial_visibility_v1/initial_rgb_segmentation.npz)
- [실제로 실행된 설치 패키지 초기화 함수](../../outputs/task10g/diagnostics/seed98207_initial_visibility_v1/installed_initialize_episode.py)
- [Task10-G seed/평가 설정](../../graph/graph_recurrent10g/config.py), [reset 및 policy 입력 경로](../../graph/graph_recurrent10g/eval_runner.py), [GT expert](../../recurrent/expert.py)
- [원본 rollout](../../outputs/task10g/final_results/rollouts/final/B/step009000/seed98207/rollout.mp4), [원본 trace](../../outputs/task10g/final_results/rollouts/final/B/step009000/seed98207/trace.json)

초기 reset capture만 원래 Linux 서버의 CPU PhysX + CUDA SAPIEN renderer로 실행했다. 전체 에피소드 GT 재생이나 새 정책 평가/학습은 수행하지 않았다. macOS simulator/vision inference는 실행하지 않았다. 공통 ExperimentLogger를 W&B disabled로 재사용했고 외부 artifact 게시를 하지 않았다. 공유 source는 변경하지 않았으며 새 파일은 진단 artifact와 이 보고서뿐이다.

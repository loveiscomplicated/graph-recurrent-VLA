# Task10-I — 512 RGB-D 가시 Peg/Box 꼭짓점 독립 검출 파일럿

실행일: 2026-11-04 KST. 상태: **구현·파일럿 평가 완료. 다수 꼭짓점의 3mm 이내 검출은 가능하지만, 거의 완벽한 검출 및 모든 점의 3mm 이내 보장은 미달이다.**

## 요청 범위와 변경

현재 512 RGB-D에서 실제 보이는 peg와 box의 물리적 꼭짓점 위치만 검출한다.
Box의 관측 가능한 구멍 입구/출구 rim 꼭짓점도 같은 평면 경계 검출로 처리한다.
꼭짓점의 의미적 ID, pose/GT38 복원, GNN/MLP 학습, recurrent 정책 변경은 수행하지 않았다.

새 코드와 실행 자료는 저장소 루트의
[`local_corner_probe_10i_20261104/`](../../local_corner_probe_10i_20261104/)에 모았다.
루트 `.gitignore`에 `/local_corner_probe_10i_20261104/`를 추가하여 전체 폴더를
Git 추적 대상에서 제외했다. 기존 live/simulator/model 코드는 수정하지 않았다.
`recurrent` 브랜치로 전환하지 않았고 현재 브랜치
`archive/visual-graph-wip-20261102`를 유지했다. 기존 working tree는 dirty였으며
commit만으로 이번 실행을 재현할 수 없다. 평가 config에 실행 소스 SHA256을 기록했다.

## 검출 로직

[`detector.py`](../../local_corner_probe_10i_20261104/detector.py)는
RGB, signed-int16 mm depth, 카메라 intrinsic/extrinsic 및 선택적 peg/box 마스크만 받는다.
Simulator/GT 모듈을 import하지 않고 물체 pose/크기/숨은 hole offset을 입력받지 않는다.

1. 이 파일럿에서는 태스크의 고정 orange/blue-white/yellow 재질을 HSV로 분리한다.
2. 유효 depth를 역투영하고 물체별 RANSAC으로 관측 평면들을 맞춘다.
3. 각 관측 평면 영역의 외곽과 내부 경계에서 직선 구간을 맞춘다.
4. Pixel center의 윤곽선을 경계로 보정하고 직선 교차점으로 subpixel 후보를 만든다.
5. 후보 camera ray와 관측 평면의 교차점으로 3D 위치를 구한다.
6. 평면상 직각 조건, 짧은 edge/큰 fit residual, 주변 관측 support, 앞쪽 occluder를 검사한다.
7. 가까운 중복 후보를 억제하고 world XYZ와 image UV를 반환한다.

숨은 꼭짓점을 cuboid 전체로 채우는 단계와 과거 값 재사용은 없다. 다만 검출기는
유한한 주변 support 검사를 쓰므로 출력이 항상 실제 물리 꼭짓점이라는 보장은 없고,
이를 precision과 recall로 평가했다. 이번 실행에는 두 카메라 통합도 없다.

**이 결과는 RGB 고정 재질 마스크 baseline이다. 실제 주 live Terra/SAM2 마스크의
검출 정확도나 native CUDA renderer 입력에 대한 검증으로 해석하지 않는다.**
동일 함수가 외부 마스크 `[2,512,512]`를 받을 수 있는 것은 구현된 기능이며,
실제 live SAM2 mask로 평가한 결과는 아니다.

## 데이터와 평가 분리

기존 원격512 자료 서버의 과거 SSH 주소 두 곳은 connection reset으로 접속하지
못했다. 서버 시작/신규 인스턴스 생성 없이 로컬 공통 ManiSkill factory로 native512를
다시 렌더링했다. 실제 backend는 macOS `physx_cpu` / `sapien_cpu`이며, SAPIEN은
Apple M4 Pro의 MoltenVK 장치로 렌더링했다. CUDA physics/renderer 및 MPS 모델은 미실행이다.

총30개 정적 상태, 두 카메라60장을 캡처했다.

- Development: 새 reset seed1/2 및 기존 Parent0197의 t0/20/40/60/80, 총7상태14장.
- Held-out: 새 reset seed100–107 및 Parent0275/0305/0531의 같은5시점, 총23상태46장.
- 별도의 최초 pilot seed0 두 카메라는 개발 확인에만 썼고 위60장에 합산하지 않았다.

Recorded Parent는 기존 `outputs/task10i/stage3/collection_run_v3/parents/`의
GT sidecar에서 robot qpos/qvel과 peg/box pose를 복원해 정적으로 렌더링했다.
관측 시 peg 크기와 원본 sidecar의 일치를 검증했다. 새 physics step, expert rollout,
연속 tracker 재생은 수행하지 않았다. 캡처 과정은 GT를 사용하지만 검출 과정과 분리한다.

각 평가의 전체 RGB/depth 기반 예측을 먼저 저장하고 SHA256 receipt와
`inference_complete.json`을 동결한 뒤, GT/actor segmentation을 읽어 채점했다.
Held-out 평가 후 detector를 수정하지 않았다. 로컬 공통 `ExperimentLogger`를 재사용했고
`WANDB_MODE=disabled`로 실행했다. 원격 게시나 메시지 전송은 없다.

## GT 꼭짓점·가시성·오차 정의

설치된 ManiSkill의 `peg_insertion_side.py` 형상 생성 코드를 확인했다. 평가 기준은
peg cuboid8개, box 외곽8개, hole 양쪽 rim8개 꼭짓점이다. GT 위치와 크기는 채점에만 쓴다.

가시성은 검출 결과와 독립적으로 정했다. 투영점이 영상 안에 있고 그 주변7×7 pixel에서
올바른 actor의 실제 관측3D 표면점이 해당 GT 꼭짓점6mm 이내에 있으면 가시 기준에 넣는다.
이는 해상도와 rasterization을 고려한 **관측 support 기반 가시성 근사**다.
정밀한 연속 ray-tracing 가시성의 정답이라고 주장하지 않으며, 6mm는 평가 위치 허용치
3mm와 별개의 고정 가시 support 기준이다. 이미지 밖/관측 support가 없는 GT 점은 제외한다.

GT와 예측의 world XYZ Euclidean 거리가3mm 이하인 대응을 일대일로 최대로 찾는다.
같은 GT 근처 중복 예측은 여러 정답으로 세지 않는다. 의미적 corner ID는 예측하지 않으므로
ID 정확도가 아니라 순서를 모르는 물리 꼭짓점 위치의 대응이다.

- Precision3mm = 일대일3mm 통과 수 / 출력한 모든 후보 수.
- Recall3mm = 일대일3mm 통과 수 / 가시 기준을 만족한 모든 GT 꼭짓점 수.
- 위치 오차 = 각 출력에서 가장 가까운 같은 물체의 가시 GT 꼭짓점까지의 거리.
  이 최근접 오차는 중복을 허용하는 보조 지표이므로 precision과 구분한다.
- 두 카메라 합계는 camera-corner 관측의 합이다. 같은 물리 꼭짓점이 양쪽에서 보이면
  두 관측이며, 서로 다른 고유 꼭짓점으로 합친 지표가 아니다.

## Held-out 결과

실행 UUID: `c1fa316309884e94848cb7f5769d541c`.
23상태46 camera image이며 detector source hash는
[`summary.json`](../../local_corner_probe_10i_20261104/heldout_v1/summary.json)에 있다.

| 물체·카메라 | 출력 | 가시 GT | 3mm 통과 | Precision3mm | Recall3mm | 최근접 오차 중앙값 | p95 | 최대 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg Main | 93 | 111 | 89 | 95.70% | 80.18% | 1.11mm | 2.55mm | 3.79mm |
| Peg Wrist | 49 | 56 | 48 | 97.96% | 85.71% | 0.96mm | 2.01mm | 4.62mm |
| Peg 합계 | 142 | 167 | 137 | **96.48%** | **82.04%** | **1.03mm** | 2.23mm | 4.62mm |
| Box Main | 93 | 103 | 81 | 87.10% | 78.64% | 1.63mm | 3.18mm | 4.23mm |
| Box Wrist | 34 | 42 | 34 | 100.00% | 80.95% | 0.71mm | 1.18mm | 1.33mm |
| Box 합계 | 127 | 145 | 115 | **90.55%** | **79.31%** | **1.43mm** | 2.87mm | 4.23mm |

Box 합계에는 외곽과 rim이 포함된다. 주 live의 기존 Box source는 Main-only이므로
Wrist Box 수치를 기존 live Graph가 이미 활용한다고 해석하지 않는다.

| Box 구조 | Main3mm 통과/가시 | Wrist3mm 통과/가시 |
| --- | ---: | ---: |
| 외곽 | 39/53 (73.58%) | 21/25 (84.00%) |
| Hole rim | 42/50 (84.00%) | 13/17 (76.47%) |

개발7상태14장에서는 Peg precision94.74%/recall70.59%,
Box precision97.56%/recall76.92%였다. 개발과 held-out 상태/가림 분포가 다르므로
두 수치를 직접 개선율로 해석하지 않는다.

검출 시간은 로컬 CPU의 카메라 한 장 기준 중앙값12.04ms, p95 24.48ms였다.
RANSAC·RGB mask·검증을 포함하며 renderer, Terra/SAM2, Graph/정책/파일 저장은 제외한다.
작은 파일럿의 단일 실행 wall time이며 전체 live 처리시간이나 CUDA 성능이 아니다.

## 관찰과 판정

1. 관측 가능한 여러 실제 꼭짓점을1–2mm 정도의 오차로 찾는 구현은 가능했다.
2. 출력269개 중252개가 일대일3mm 기준을 통과했다. Peg5개/Box12개는 중복이나
   3mm 초과를 포함해 통과하지 못했고, 가시 기준312개 중60개는3mm 이내로 찾지 못했다.
3. 따라서 **"모든 보이는 꼭짓점을 거의 완벽하게, 항상3mm 이내"는 현재 달성하지 않았다.**
4. 일부 면이 얇게 보이는 경우, 가림 주변, depth 평면 분리, 고정 재질 mask가 일부만
   보이는 경우에 누락이 남는다. 구체 원인별 기여율은 이 파일럿에서 분리 측정하지 않았다.
5. 이 검출기가 GT-state 복원이나 제어 성공률을 확보한다는 실험은 하지 않았다.

## 검증·재현·원자료

`test_detector.py`의4개 검사가 모두 통과했다: 해석 가능한 평면 사각형의 subpixel
꼭짓점과 world 변환, invalid signed-depth 배제, 중복 예측의 recall 부풀림 방지,
native512 입력 강제. 공통 simulator/live 인터페이스는 변경하지 않았다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector -v
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v1 --output local_corner_probe_10i_20261104/new_evaluation --split heldout
```

- [검출 코드](../../local_corner_probe_10i_20261104/detector.py)
- [입력 capture manifest](../../local_corner_probe_10i_20261104/capture_v1/manifest.json)
- [동결 예측 receipt](../../local_corner_probe_10i_20261104/heldout_v1/inference_complete.json)
- [Held-out 집계](../../local_corner_probe_10i_20261104/heldout_v1/summary.json)
- [프레임별 CSV](../../local_corner_probe_10i_20261104/heldout_v1/metrics.csv)
- [전체46장 gallery](../../local_corner_probe_10i_20261104/gallery.html)
- [대표 상태 이미지](../../local_corner_probe_10i_20261104/visible_corners_examples.png)

이미지의 녹색 십자는 검출, 자홍색 원은 평가용 가시 GT다. 폴더는 사용자 요청에 따라
Git ignore되어 있으므로 이 문서의 로컬 artifact 링크는 현재 workspace 보존을 전제로 한다.

# Task10-I — 512 가시 Peg/Box 꼭짓점 검출 정확도 v2

실행일: 2026-11-04 KST. 상태: **개선 구현 및 새로운 입력의 동일 조건 비교 완료.**

새28상태·56영상에서3mm 기준 출력 precision이 Peg95.80→99.43%,
Box88.54→99.29%로 올랐다. Peg recall도67.49→86.21%로 올랐다.
Box recall은77.65%로 동일하다. 출력316개 중314개가 통과했지만 여전히2개 오검출과
가시 GT382개 중68개 누락이 남아 전체 점의3mm 보장/거의 완벽한 검출을 주장하지 않는다.

## 범위·저장 위치

현재 프레임에서 실제 보이는 Peg/Box 물리 꼭짓점 좌표만 검출한다. Box의 보이는
구멍 rim도 관측 면의 경계로 포함한다. 물체 pose/GT38 복원, GNN/MLP 학습,
recurrent 변경, 가려진 꼭짓점 보완, 카메라 간 융합과 과거 값 유지는 수행하지 않았다.

사용자가 요청한 기존 ignored 폴더
[`local_corner_probe_10i_20261104/`](../../local_corner_probe_10i_20261104/) 안에서 개발했다.
현재 검출기는 [`detector.py`](../../local_corner_probe_10i_20261104/detector.py),
이전 버전의 정확한 사본은 [`detector_v1.py`](../../local_corner_probe_10i_20261104/detector_v1.py)다.
V1 SHA256 `5b7d040c63cc5962c8734126951cb77222625e4a01dc7f4ba415ccca28387c87`는
기존 v1 평가 source hash와 일치한다. 기존 v1 결과를 덮어쓰지 않았다.

시작 시 현재 체크아웃은 `codex/cleanup-modular-live-20261104`였다. 이 작업이 브랜치를
전환한 것은 아니며 `recurrent`로 전환하지 않았다. 주 live/shared simulator/model/
tracking 인프라를 수정하지 않았다. 소스는 ignore되어 있으므로 Git commit만으로
실험 코드가 재현되지는 않는다. source hash와 선택 설정은 별도 freeze/평가 config에 있다.

## 개선한 원인과 처리

1. **흰색 부분만 보이는 Peg 누락:** v1은 연결 성분에 orange pixel이 있어야 Peg로
   인정했다. Orange가 가려지면 실제 보이는 흰색 tail까지 버렸다. V2는 고정 재질의
   blue-white 성분 자체로 관측을 유지한다. 로봇 회색/배경의 약한 blue tint를 함께
   선택하지 않도록 재질 HSV 범위를 개발 자료에서 분리했다.
2. **면 경계의 위치 오차·중복:** 근처에 실제 관측된 직교 면이 있으면 그 평면 제약으로
   후보3D 위치를 보정한다. 가까운 같은 물체 후보를4.5mm 내에서 묶고 독립 면 support가
   많은 후보를 우선한다. 이는 GT 최근접점으로 snapping하는 처리가 아니다.
3. **외곽선 내부의 세 면 접점:** 실제3개 관측 면의 교차점 후보를 추가했다. 각 면의
   두께 있는 pixel core와 후보 부근 현재 support를 요구하고 앞쪽 가림을 검사한다.
   전체 cuboid를 완성해 숨은 꼭짓점을 출력하지 않는다.
4. **그리퍼 접촉부의 가짜 모서리:** 현재 RGB의 중성색과 depth로, 물체 mask 밖의
   회색 접촉 표면이 후보 바로 근처에 있는 경우를 억제한다. Simulator robot label/FK를
   읽지 않는다. 검출 결과가 가림 경계를 물체 꼭짓점으로 오인하는 일부 경우를 줄였다.

개발 중 넓은 white mask와 검증이 부족한3평면 교차는 큰 오검출을 만들었다.
이를 최종 결과로 채택하지 않았으며 `v2_development_a..f`의 결과는 로컬에 보존했다.
더 작은 polygon epsilon/짧은 edge 허용은 개발 precision/recall이 나빠 채택하지 않았다.
최종은 `v2_development_f`의 detector다.

## 데이터 분리와 공정 비교

지난 보고서의 `capture_v1` 30상태·60영상은 이번 개선의 development로 사용했다.
이전에 held-out이라고 불렀던23상태도 재사용했으므로, 이번에 독립 성능이라고 다시
인용하지 않는다. 이 개발 자료에서 V1→V2 Peg precision96.11→100.00%,
recall79.36→90.37%; Box precision92.26→98.73%, recall78.68→79.19%였다.

새 Final은 다음 **28상태·두 카메라56영상**이다.

- 새 reset seed200–207: 8상태.
- 기존 recorded Parent0023/0193/0327/0449의 t0/20/40/60/80: 20상태.
- Development와 episode identity 및 recorded Parent가 겹치지 않음을 검증했다.
- [`v2_final_freeze.json`](../../local_corner_probe_10i_20261104/v2_final_freeze.json)을
  먼저 저장하고 V1/V2를 같은 입력에서 실행했다. Final 채점 후 detector는 수정하지 않았다.

관측은 공유 ManiSkill factory의 native512 카메라로 로컬에서 새로 렌더링했다.
Recorded 상태는 저장 qpos/qvel·Peg/Box pose를 복원하고 Peg 크기 일치를 검사했다.
새 physics step/행동 rollout 없이 정적으로 렌더링했으며, 연속 tracking 평가가 아니다.

두 버전 모두 **같은 RGB 고정 재질 마스크 정책 범주**와 원본 depth를 사용한다.
HSV mask 개선도 detector 개선에 포함된다. 실제 Terra/SAM2를 호출하거나 그 mask를
사용하지 않았다. 비교는 baseline의 실제 색상 분리 코드까지 포함한 detector 전체다.

각 실행에서 모든 예측을 먼저 저장·hash·동결한 뒤 GT와 actor segmentation을 열어
평가했다. 프레임별 `(identity,camera,object_id,visible_reference)`가 V1/V2에서 정확히
같음을 검증했다. 독립 카메라 관측들의 합이며 같은 꼭짓점이 양쪽에 보이면 두 관측이다.

## 평가 기준 유지

[v1 보고서](10i_visible_corner_512_feasibility_20261104.md)의 기준을 그대로 사용한다.
물체별 GT 꼭짓점은 peg8개, box 외곽8개, box hole 양쪽 rim8개다. 위치 오차는 world
XYZ Euclidean 거리다. 3mm 이내 대응을 일대일로 찾으므로 중복 출력은 정답을 늘리지 않는다.

가시성은 투영점 주변7×7 pixel에서 올바른 actor의 실제 관측3D 점이 GT 꼭짓점6mm
이내에 있는지를 보는 고정 근사다. 검출 결과로 가시성을 정하지 않는다. 이는 엄밀한
연속 ray-tracing 가시성 GT가 아니며 얇은 가림/경계에 대한 제한이 있다. 이 평가 규칙을
완화하거나 누락을 분모에서 빼서 정확도를 올리지 않았다.

- Precision3mm: 출력 중 일대일3mm 기준을 통과한 비율.
- Recall3mm: 가시 기준 GT 중 일대일3mm 기준으로 찾은 비율.
- 보조 위치 오차: 출력에서 가장 가까운 같은 물체의 가시 GT까지 거리.
  이 지표는 중복을 허용하므로 one-to-one precision과 구분한다.

## Final 결과

Baseline 실행 UUID `dd00c12307f1469f9901e306bfa0b896`.
V2 실행 UUID `0e78b1b0d9a5419cb7bbbd91db9ed562`.

| 물체 | 가시 GT | V1 출력 / 통과 | V2 출력 / 통과 | Precision V1→V2 | Recall V1→V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg | 203 | 143 / 137 | 176 / 175 | **95.80→99.43%** | **67.49→86.21%** |
| Box | 179 | 157 / 139 | 140 / 139 | **88.54→99.29%** | **77.65→77.65%** |
| 합계 | 382 | 300 / 276 | 316 / 314 | 92.00→99.37% | 72.25→82.20% |

Peg precision은3.63pp, recall은18.72pp 증가했다. Box precision은10.75pp 증가했고
recall은 동일하다. Box에서는 출력17개를 줄이면서 같은139개 정답을 유지했다.
이는 출력의 신뢰도 개선이며 Box 누락 문제의 해결을 뜻하지 않는다.

| 카메라·물체 | V1 Precision | V2 Precision | V1 Recall | V2 Recall |
| --- | ---: | ---: | ---: | ---: |
| Main Peg | 95.83% | 99.24% | 62.59% | 88.44% |
| Wrist Peg | 95.74% | 100.00% | 80.36% | 80.36% |
| Main Box | 85.96% | 98.99% | 75.97% | 75.97% |
| Wrist Box | 95.35% | 100.00% | 82.00% | 82.00% |

Wrist Box는 이 독립 검출 실험에서 평가했다. 기존 주 live Graph의 Box Main-only
정책을 변경하거나 이미 Wrist Box를 사용한다고 주장하지 않는다.

| 위치 오차 보조 지표 | V1 중앙값 | V2 중앙값 | V1 p95 | V2 p95 | V2 최대 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg | 1.00mm | 0.91mm | 2.12mm | 1.41mm | **64.41mm** |
| Box | 1.32mm | 1.22mm | 2.80mm | 2.04mm | **5.77mm** |

Peg의 드문 큰 오검출1개와 Box의3mm 초과1개가 남는다. p95가3mm 아래여도 모든 점이
3mm 이내인 것은 아니다. 작은 파일럿의 관측 비율이며 모집단 정확도 보장은 아니다.

## 실제 실행·테스트·제약

- macOS CPU NumPy/OpenCV/SciPy 검출과 `physx_cpu` / `sapien_cpu` 캡처를 실행했다.
  SAPIEN 렌더링 장치는 Apple M4 Pro/MoltenVK였다. CUDA physics/renderer와 MPS는 미실행이다.
- `utils.experiment_logger.ExperimentLogger`를 두 평가에서 재사용하고
  `WANDB_MODE=disabled`로 로컬 기록했다. W&B 프로젝트/group/run-ID 규칙을 바꾸지 않았다.
  HF 게시, 새 인스턴스 생성, 외부 메시지 전송은 없다.
- 단위 테스트6개 통과: subpixel 사각형·world 변환, invalid signed depth 배제,
  native512 강제, 중복 정답 부풀림 방지, orange 없는 white tail 유지,
  회색 로봇/배경의 Peg 오인 방지. 사각형 fixture는 주변 depth를 더 멀리 두어
  같은 평면의 색상 무늬가 아닌 실제 전경 사각형으로 구성했다.
- 전체 주 live, Terra/SAM2 mask, 연속 시퀀스 안정성, GT38 복원·제어 성능은 미검증이다.
- 두 detector의 정확도 비교를 병행 실행했다. 해당 wall-time은 CPU 경쟁 영향을 받을
  수 있어 속도 개선/악화의 근거로 사용하지 않는다.

## 재현과 원자료

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector -v
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v2_final --output local_corner_probe_10i_20261104/new_v2_eval --split heldout
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v2_final --output local_corner_probe_10i_20261104/new_v1_eval --split heldout --detector-module detector_v1
```

- [동일 입력 비교 JSON](../../local_corner_probe_10i_20261104/v2_comparison.json)
- [Final capture manifest](../../local_corner_probe_10i_20261104/capture_v2_final/manifest.json)
- [V1 summary](../../local_corner_probe_10i_20261104/v1_final_v2/summary.json)
- [V2 summary](../../local_corner_probe_10i_20261104/v2_final/summary.json)
- [V2 프레임별 CSV](../../local_corner_probe_10i_20261104/v2_final/metrics.csv)
- [V2 예측 동결 receipt](../../local_corner_probe_10i_20261104/v2_final/inference_complete.json)
- [모든56장의 V1/V2 비교 gallery](../../local_corner_probe_10i_20261104/gallery_v2.html)

Gallery의 녹색 십자는 예측, 자홍색 원은 평가용 가시 GT다. 모든 코드는 계속 Git
ignore 상태이므로 이 문서의 링크는 현재 로컬 폴더 보존을 전제로 한다.

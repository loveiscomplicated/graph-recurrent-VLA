# Task10-I — 512 가시 Peg/Box 꼭짓점 포착률 개선 v3

실행일: 2026-11-04 KST. 상태: **독립 구현 개선 및 새 Final 입력 비교 완료.**

동일한 새28상태·56영상에서 V2→V3의3mm 포착률은 Peg91.62→93.30%,
Box73.81→78.57%로 올랐다. 추가 출력12개 중11개가 정답이며1개는4.05mm 오차였다.
기존 누락59개 중11개를 회복해 전체 누락을48개로 줄였다. Box 출력 precision은
98.41→97.78%로 소폭 낮아졌으며, 포착률 증가가 비용 없이 얻어진 결과는 아니다.

## 범위와 저장 위치

현재512 RGB-D에서 실제 보이는 Peg/Box 꼭짓점 좌표만 검출한다. Box의 보이는 hole
rim도 같은 관측 경계 검출에 포함한다. 숨은 꼭짓점 완성, 물체 pose/GT38 추정,
GNN/MLP 학습, recurrent 변경, temporal hold와 카메라 융합은 수행하지 않았다.

소스/입력/결과는 사용자 요청대로 기존 ignored 폴더
[`local_corner_probe_10i_20261104/`](../../local_corner_probe_10i_20261104/)에 유지했다.
현재 [`detector.py`](../../local_corner_probe_10i_20261104/detector.py)는 V3이며
[`detector_v2.py`](../../local_corner_probe_10i_20261104/detector_v2.py)와
[`detector_v3.py`](../../local_corner_probe_10i_20261104/detector_v3.py)에 비교 버전을 보존했다.
주 live/shared simulator/model/logger 코드를 수정하지 않았다. 현재
`codex/cleanup-modular-live-20261104` 체크아웃을 유지했고 브랜치를 전환하지 않았다.
Source가 ignore되어 있으므로 commit만으로 재현되지 않으며 별도 source hash를 기록했다.

## 누락 진단과 변경

V2의 이전 Final을 이번 개발 자료로 전환해 조사했다. 고정 가시 기준에서 놓친68개는
가장 가까운2D 후보의 최초 탈락 기준으로 다음과 같이 나뉘었다.

| 최초 원인 근사 | Peg | Box |
| --- | ---: | ---: |
| 주변4pixel 이내 경계 후보 없음 | 18 | 17 |
| 주변의 가까운 표면 때문에 occluder 판정 | 7 | 13 |
| 평면상 각도 기준 탈락 | 3 | 8 |
| 주변 support 부족 | 0 | 2 |

이는 [`diagnose_misses.py`](../../local_corner_probe_10i_20261104/diagnose_misses.py)의
개발용 GT 분석이며 인과 기여율이나 모든 관측의 실패 원인 정답은 아니다.
GT는 이 진단/채점에만 사용하고 검출기로 전달하지 않았다.

최종 변경은 다음과 같다.

1. **기존 검출 보존:** V2 결과를 먼저 유지하고, 추가 후보가 기존 점을 덮어쓰지 않도록
   했다. Final56장 전체에서 모든 V2 출력 좌표가 V3에 정확히 보존됨을 검증했다.
2. **국소 경계 적합:** 전역 다각형에서 사라진 짧은 edge를 국소 직선으로 맞춘다.
   서로 다른 두 길이로 맞춘 교차점이0.8pixel 이내로 안정적일 때 회복 후보로 쓴다.
3. **짧은 Box 경계:** Box의 짧은 경계 후보와 작은 관측 성분을 추가로 허용하되,
   두 edge leg가 실제 depth로 지지되는지를 검사한다. Peg에서는 개발 오검출이 증가해
   동일한 short-boundary 확장을 채택하지 않았다.
4. **관측 면의 직사각 경계:** Peg의 관측 면에서 다른 관측 법선이 제공하는 방향과
   현재 contour extent를 이용해 끝점 후보를 만든다. 크기/pose GT는 쓰지 않는다.
   원본 contour가 이미지에 잘렸으면 이 회복을 중단한다. Box에서의 같은 확장은 개발
   오검출을 늘려 최종 버전에서 제외했다.
5. **깊이로 edge 검증:** 다른 물체/같은 물체의 앞쪽 표면에 잘린 경계와 물리적 crease를
   구분한다. Ray-depth 차이뿐 아니라 평면에서의 실제3D 이격도 확인해 grazing angle의
   mm depth 양자화로 생기는 가짜 edge를 줄인다. 관측된 테이블 접촉 경계도 확인한다.
6. **현재 가시 support:** 꼭짓점 자체의 작은 pixel footprint에 target depth support가
   있는지 확인한다. 추가 후보가 앞쪽 aperture/robot 뒤에 놓이면 배제하고, 평면 보정이
   image 위치를 과도하게 움직이는 경우와 기존 점 근처 중복도 억제한다.

단순히 각도/가림 기준만 느슨하게 한 개발 후보는 오검출이 많아 제외했다. 개발 시도
`v3_development_a..k`와 source hash/결과를 ignored 폴더에 보존했다. Final은 K를 고정했다.

## 데이터 분리와 실행

개발에는 `capture_v1`의30상태와 `capture_v2_final`의28상태, 총58상태를 썼다.
이전 Final28상태는 이제 개발 자료이므로 독립 성능 근거로 재사용하지 않는다.

새 Final은 다음28상태, 두 카메라56영상이다.

- 새 reset seed300–307: 8상태.
- Recorded Parent0621/0686/0720/0844의 t0/20/40/60/80: 20상태.
- 모든 개발 identity/Parent와 새 Final이 분리되었음을 검증했다.
- [`v3_final_freeze.json`](../../local_corner_probe_10i_20261104/v3_final_freeze.json)에
  source/parameter/input manifest hash를 먼저 기록한 뒤 평가했다. Final 후 소스는 수정하지 않았다.

공유 ManiSkill factory로 CPU native512 관측을 새로 렌더링했다. Recorded 상태는 저장
qpos/qvel·Peg/Box pose를 복원하고 Peg 크기의 일치를 확인했다. 새 physics step이나
행동 rollout 없이 정적으로 렌더링했다. 연속 tracker 평가가 아니다.

두 버전은 동일한 RGB-D 입력을 사용한다. 각 버전의 전체 예측을 저장·hash·동결한 뒤
GT/actor segmentation을 열어 평가했다. Frame별 identity/camera/object와 가시 GT 분모가
정확히 일치함을 검증했다. 카메라별 관측을 합산하며 중복 물리 꼭짓점의 cross-view fusion은 없다.

## 평가 기준 유지

[V2 보고서](10i_visible_corner_512_accuracy_v2_20261104.md)의 가시성 근사와3mm 일대일
대응을 그대로 사용했다. Actor depth의7×7pixel 현재 support가 GT 꼭짓점6mm 이내이면
가시 기준에 넣는다. 이 기준은 검출 결과와 독립적이며 엄밀한 연속 ray-tracing 가시성
정답은 아니다. 가시성 조건 완화나 누락 분모 제외로 포착률을 올리지 않았다.

Precision은 출력 중 일대일3mm 통과 비율, recall은 가시 GT 중3mm 이내로 찾은 비율이다.
중복은 정답을 늘리지 않는다. 보조 위치 오차는 출력에서 가장 가까운 동일 물체의 가시
GT까지 거리이며, 의미적 corner ID를 맞힌다는 지표는 아니다.

## Final 결과

V2 UUID `19c43e51d0e846089ffdc1ad4b559049`.
V3 UUID `d85b6121541f4411810d9dceafe49080`.

| 물체 | 가시 GT | V2 출력/정답 | V3 출력/정답 | Recall V2→V3 | Precision V2→V3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg | 179 | 164/164 | 167/167 | **91.62→93.30%** | **100.00→100.00%** |
| Box | 168 | 126/124 | 135/132 | **73.81→78.57%** | **98.41→97.78%** |
| 합계 | 347 | 290/288 | 302/299 | 83.00→86.17% | 99.31→99.01% |

추가 Peg3개와 Box8개를3mm 이내로 찾았다. 추가 Box 후보1개는4.046mm여서 실패했다.
V3의 최종 실패 출력은 Box3개이고, 가시 기준에서 못 찾은 것은 Peg12개/Box36개다.
이전 턴의 포착률86.2%/77.7%와 위 V2 기준91.6%/73.8%는 **서로 다른 Final 입력**의
수치다. 이번 개선율은 위 표의 같은 새 입력에서만 계산한다.

| 카메라·물체 | Recall V2 | Recall V3 | Precision V2 | Precision V3 |
| --- | ---: | ---: | ---: | ---: |
| Main Peg | 95.83% | 95.83% | 100.00% | 100.00% |
| Wrist Peg | 83.05% | 88.14% | 100.00% | 100.00% |
| Main Box | 70.68% | 75.94% | 97.92% | 97.12% |
| Wrist Box | 85.71% | 88.57% | 100.00% | 100.00% |

Box에서 외곽 recall은 Main41/62, Wrist23/25이며 hole rim은 Main60/71,
Wrist8/10이다. Wrist Box는 독립 검출 실험이며 기존 주 live Main-only 정책을 바꾸지 않았다.

최근접 오차 중앙값/p95는 V3 Peg0.93/1.31mm, Box1.44/2.31mm다. 최대는 Peg1.63mm,
Box8.09mm이며, Box의 이 최대 오차는 V2부터 남은 출력이다. 모든 점의3mm 보장은 아니다.

## 검증과 한계

- 단위 테스트12개 PASS. 기존 좌표/invalid-depth/중복/재질 검증 외에 가까운 이웃 면과
  실제 가림 구분, 같은 물체의 self-occlusion, aperture 뒤 가짜점, crop 경계의 가짜
  꼭짓점 방지를 확인했다. Final source hash와 core point 보존도 검증했다.
- `ExperimentLogger`를 재사용해 `WANDB_MODE=disabled` 로컬 평가 기록을 남겼다.
  W&B/HF 게시나 외부 메시지 전송은 수행하지 않았다.
- 실행은 macOS CPU NumPy/OpenCV/SciPy와 `physx_cpu`/`sapien_cpu`였다.
  렌더링 장치는 Apple M4 Pro/MoltenVK다. CUDA physics/renderer와 MPS는 미실행이다.
- **고정 재질 RGB mask 기반 결과**다. 실제 Terra/SAM2 mask나 GPU live에 대한
  검증은 아니다. GT38 복원과 closed-loop 성공률도 평가하지 않았다.
- 회복 검증과 여러 직선 적합 때문에 처리 비용이 늘었다. 병행 비교 실행의 카메라당
  CPU wall-time 중앙값은 V2 약13.35ms, V3 약40.98ms였다. 정식 단독 latency benchmark는
  아니며 renderer/SAM2/정책은 제외한다. 현재는 정확도 확인용 구현이고 GPU 최적화하지 않았다.
- 작은 정적 표본의 결과이며 장기 tracking, 미관측 모양/재질, 노이즈 depth 일반화를
  보장하지 않는다. Box 누락과 드문 오검출은 계속 남는다.

## 재현과 원자료

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector -v
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v3_final --output local_corner_probe_10i_20261104/new_v3_eval --split heldout
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v3_final --output local_corner_probe_10i_20261104/new_v2_eval --split heldout --detector-module detector_v2
```

- [동일 입력 비교 JSON](../../local_corner_probe_10i_20261104/v3_comparison.json)
- [Final 입력 manifest](../../local_corner_probe_10i_20261104/capture_v3_final/manifest.json)
- [V2 summary](../../local_corner_probe_10i_20261104/v2_final_v3/summary.json)
- [V3 summary](../../local_corner_probe_10i_20261104/v3_final/summary.json)
- [V3 프레임별 CSV](../../local_corner_probe_10i_20261104/v3_final/metrics.csv)
- [모든56장의 V2/V3 비교 gallery](../../local_corner_probe_10i_20261104/gallery_v3.html)
- [단위 테스트 로그](../../local_corner_probe_10i_20261104/v3_tests.log)

Gallery의 녹색 십자는 출력, 자홍색 원은 평가용 가시 GT다. 코드는 계속 Git ignore
상태이므로 로컬 artifact 링크는 현재 폴더 보존을 전제로 한다.

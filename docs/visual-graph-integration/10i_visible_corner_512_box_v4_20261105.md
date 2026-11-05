# Task10-I — Box 가시 꼭짓점 검출 개선 v4

실행일: 2026-11-05 KST. 상태: **독립 후보 구현·새 입력 비교 완료. 개선 폭은 제한적이다.**

새28상태·56영상의 동일 입력에서 Box3mm 포착률은 V3 **82.91%→84.42%**, 출력
precision은 **95.38%→94.92%**였다. 추가4개 중3개가3mm 이내였고1개는5.15mm였다.
이전 턴의78.57%는 다른 Final 자료이므로84.42%와 직접 빼서 개선율을 계산하지 않는다.
이번 독립 자료의 관측 개선은 **+1.51pp**이며 큰 폭의 일반화 개선을 입증하지 못했다.

## 범위와 보존

기존 ignored 폴더
[`local_corner_probe_10i_20261104/`](../../local_corner_probe_10i_20261104/)에서
Box 추가 검출만 구현했다. 주 live/shared simulator/model/logger는 수정하지 않았다.
`codex/cleanup-modular-live-20261104`를 유지했고 브랜치를 전환하지 않았다.
코드는 계속 Git ignore 상태이므로 commit만으로 재현되지 않으며 파일별 hash를 기록했다.

- 현재 진입: [`detector.py`](../../local_corner_probe_10i_20261104/detector.py).
- 새 Box 로직: [`box_corner_refinement.py`](../../local_corner_probe_10i_20261104/box_corner_refinement.py).
- 비교 원본: `detector_v3.py`; V4 snapshot: `detector_v4.py`, `box_corner_refinement_v4.py`.
- 모든 기존 V3 출력 좌표가 V4에 그대로 남아 있음을 Final 전체에서 검사했다.
  Peg 결과는 동일하다. 추가 Box 점만 기존 점10mm 근처 중복을 제거해 합친다.

SAM2, 다른 카메라의 후보 전달, temporal propagation, recurrent/GT38 복원은 포함하지
않았다. **고정 재질 RGB mask + 현재 depth의 독립 검출**이다. 내부에서 형상 후보를
생성하더라도 현재 관측 지지가 없는 숨은 꼭짓점을 출력하는 방식은 채택하지 않았다.

## 구현과 가정

1. **RGB 후보:** Box mask와 원본 grayscale에서 Shi–Tomasi 후보를 찾고 subpixel로
   보정한다. 관측 평면 위 후보 주변을4개 quadrant로 검사해 물리적 외곽/오목한 rim
   모서리 형태인지 확인한다. 단순 직선 표면이나 가림 경계를 줄이기 위한 검사다.
2. **Box 외곽 cube prior:** 현재 설치된 ManiSkill의 생성 코드에서 외곽의 세 반변 길이가
   같음을 확인했다. 관측 평면 법선과 두 충분한 face span으로 방향·변 길이를 추정한다.
   시뮬레이터의 실제 instance pose/크기/hole offset은 검출기에 전달하지 않는다.
3. **크기 추정 제한:** 관측 크기0.16–0.27m와 두 span이 추정 변 길이의80% 이상이라는
   class 조건을 쓴다. 설치 태스크의 실제 randomization은 외곽 한 변0.17–0.25m다.
   여기서 넓힌 범위는 class prior이며 각 episode의 GT 크기를 읽는 것이 아니다.
   이미지에 잘린 Box mask나 관측이 부족하면 cube 후보 생성을 보류한다.
4. **현재 영상 정밀화:** 형상 후보가 RGB corner2.5pixel 이내면, 그 image corner의 ray와
   실제 관측 평면을 사용해 위치를 정밀화한다. 평면 보정은5mm 이내로 제한한다.
5. **가시성 필터:** 현재 target depth의 작은 footprint support, 앞쪽 물체에 가려짐,
   물리적 edge 또는 RGB corner 지지를 검사한다. RGB corner가 확실한 형상 후보는
   최대6mm 주변 depth support를 허용하고, 일반 후보는 기존4mm support를 사용한다.
   이는 inference 기준이며 아래 GT 가시성 평가 조건은 바꾸지 않았다.

직선 segment 교차 방식도 개발에서 실행했으나 회복 이득 없이 큰 오검출을 추가해
최종 코드에서 제거했다. 초기 OpenCV5 반환 shape 처리 오류로 중단한 개발 실행도
failed 기록으로 보존했으며 Final 수치에 포함하지 않았다.

이 로직은 해당 태스크의 Box 형상·재질을 이용하는 prototype이다. 일반 모양의 물체에
그대로 적용 가능한 꼭짓점 검출기라고 주장하지 않는다.

## 개발/Final 분리

주 개발은 이전 `capture_v3_final` 28상태를 사용했다. 이 자료는 이제 development다.
별도로 이전 `capture_v2_final`에서도 회귀를 확인했다. 이전 Final을 이번 독립 성능으로
재인용하지 않는다.

- 주 개발 Box: V3 recall132/168=78.57% → V4 140/168=83.33%;
  precision132/135=97.78% → 140/144=97.22%.
- 이전 개발 Box: V3 recall150/179=83.80% → V4 156/179=87.15%;
  precision150/151=99.34% → 156/159=98.11%.

새 Final은 reset seed400–407 및 Parent0121/0349/0589/0770의 t0/20/40/60/80이다.
총28상태·두 카메라56영상이며 모든 기존 capture와 identity/Parent가 분리되어 있다.
실행 전 [`v4_final_freeze.json`](../../local_corner_probe_10i_20261104/v4_final_freeze.json)에
detector/helper/원본 source와 입력 manifest hash 및 class 조건을 저장했다.
Final 이후 검출 소스는 수정하지 않았다.

관측은 공유 factory에서 native512로 로컬 재렌더링했다. Recorded 상태는 저장
qpos/qvel·Peg/Box pose를 복원하고 Peg 크기의 일치를 확인했다. 새 physics step,
action rollout, 연속 SAM2 memory 실행은 없다. 캡처에는 GT를 사용하지만 inference에는
RGB-D·calibration만 전달한다.

## 평가 유지와 Final 결과

[V3 보고서](10i_visible_corner_512_recall_v3_20261104.md)의 가시성 근사 및3mm 일대일
matching을 그대로 사용했다. GT 가시성은 투영점 주변7×7pixel에서 올바른 actor의
관측3D support가 GT 꼭짓점6mm 이내에 있는지를 보는 근사다. 엄밀한 연속 ray-tracing
정답이 아니며 얇은 가림/경계에는 한계가 있다. 검출 결과로 가시 분모를 정하지 않는다.

예측을 먼저 저장·hash·동결한 뒤 GT/actor segmentation을 읽어 채점했다.
V3/V4의 frame별 identity/camera/object/visible GT 분모가 일치함을 검사했다.
두 카메라 수치는 독립 camera-corner 관측의 합이고 고유 꼭짓점으로 fusion한 수치가 아니다.

Baseline UUID `a913622c5c544051ae93d5ec757c632f`.
V4 UUID `444452f7b765462cb8182a57954fab91`.

| Box | 가시 GT | V3 출력/3mm 정답 | V4 출력/3mm 정답 | Recall V3→V4 | Precision V3→V4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main | 148 | 126/119 | 129/121 | 80.41→81.76% | 94.44→93.80% |
| Wrist | 51 | 47/46 | 48/47 | 90.20→92.16% | 97.87→97.92% |
| 합계 | 199 | 173/165 | 177/168 | **82.91→84.42%** | **95.38→94.92%** |

외곽은 Main57/69→59/69, Wrist25/26→25/26이다. Hole rim은 Main62/79→62/79,
Wrist21/25→22/25다. 즉 새 정답은 외곽2개와 rim1개다.

| 추가 후보 | 카메라 | 방법 | 위치 오차 |
| --- | --- | --- | ---: |
| reset401 | Main | 관측 cube + RGB 정밀화 | 1.109mm |
| reset406 | Main | RGB quadrant | **5.147mm, 실패** |
| Parent0121 t0 | Main | 관측 cube + RGB 정밀화 | 2.813mm |
| Parent0589 t0 | Wrist | RGB quadrant | 0.664mm |

Box 누락은34→31개로 줄었고,3mm 실패 출력은8→9개로 늘었다.
Box 최근접 위치 오차 중앙값/p95는 V3 1.34/2.86mm, V4 1.34/2.95mm다.
최대17.98mm는 이전 버전에서 남은 출력이다. 모든 점의3mm 보장은 아니다.
Peg는 V3/V4 모두186/205 정답,187개 출력으로 완전히 동일하다.

## 해석·실행 한계

같은 새 입력에서 추가 정답3개와 추가 실패1개를 관측했다. 작은 정적 표본이므로
모집단에서 개선이 확정되었다고 주장하지 않는다. 특히 개발에서+4.76pp였던 이득이
새 Final에서는+1.51pp였다. 현재 후보의 일반화 이득은 작다.

- 검출은 macOS CPU NumPy/OpenCV/SciPy이며 캡처는 `physx_cpu` / `sapien_cpu`였다.
  Apple M4 Pro/MoltenVK 렌더 장치를 사용했다. CUDA physics/renderer, MPS는 미실행이다.
- `ExperimentLogger`를 재사용했고 `WANDB_MODE=disabled`로 로컬 기록했다.
  공통 identity 규칙을 변경하거나 HF 게시/외부 메시지 전송을 하지 않았다.
- 테스트14개 PASS. 기존12개 외에 명시적 빈 mask 존중과 독립 analytic rotated cube의
  관측 기반 크기/꼭짓점 계산을 검사했다. 실제 Final에서도 source freeze와 모든 V3
  point 보존을 확인했다.
- RGB/plane 검사를 추가해 CPU 비용이 늘었다. 병행 진단 실행의 카메라당 wall-time
  중앙값은 V3 약49.46ms, V4 약57.96ms였다. 단독 latency benchmark는 아니며
  renderer/SAM2/정책은 제외한다.
- 실제 SAM2 mask, 주 live 연결, 장기 tracking, GT38 복원이나 제어 성공률은 미검증이다.
  Wrist Box 사용은 독립 검출 실험이며 주 live의 Box Main-only 정책을 바꾸지 않았다.

## 재현·원자료

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector -v
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v4_final --output local_corner_probe_10i_20261104/new_v4_eval --split heldout
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.evaluate --capture local_corner_probe_10i_20261104/capture_v4_final --output local_corner_probe_10i_20261104/new_v3_eval --split heldout --detector-module detector_v3
```

- [동일 입력 비교 JSON](../../local_corner_probe_10i_20261104/v4_comparison.json)
- [추가4개 점의 개별 오차](../../local_corner_probe_10i_20261104/v4_added_corners.json)
- [Final 입력 manifest](../../local_corner_probe_10i_20261104/capture_v4_final/manifest.json)
- [V3 summary](../../local_corner_probe_10i_20261104/v3_final_v4/summary.json)
- [V4 summary](../../local_corner_probe_10i_20261104/v4_final/summary.json)
- [프레임별 CSV](../../local_corner_probe_10i_20261104/v4_final/metrics.csv)
- [모든56장 비교 gallery](../../local_corner_probe_10i_20261104/gallery_v4.html)
- [테스트 로그](../../local_corner_probe_10i_20261104/v4_tests.log)

Gallery의 녹색은 출력, 자홍색은 평가용 가시 GT다. ignored 자료의 링크는 현재 로컬
폴더 보존을 전제로 한다.

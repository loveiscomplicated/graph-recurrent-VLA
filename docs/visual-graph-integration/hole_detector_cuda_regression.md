# CUDA 꼭짓점 검출 업데이트 이후 Hole 입구 누락 검토

## 비교 방법

[직전 21D 실험](hole_front_object21_latest_detector.md)에서 동일 GT와 평가 프레임을 사용했지만, 검출기 버전과 실행 설정이 함께 바뀌었다. 이를 분리하기 위해 초기 시드 200–1399에서 10씩 간격을 둔 120프레임을 재생했다. 별도로 이전 경로에 입구 근접점이 2개 이상 있고 최신 기본 경로에서 근접점이 없으며 최신 Hole 중심 오차가 50 mm를 넘은 프레임 12개를 확인했다. 이 12개는 실패 재현에 사용했고 대표 표본의 집계에는 포함하지 않았다.

모든 조건은 같은 512×512 Main RGB-D와 카메라 보정을 사용했다. 팀 holdout의 `visible_reference`, `score`를 재사용해 가시 GT와 3 mm 이내 일대일 매칭했다. Box 외곽과 Hole 림은 같은 Box 점 집합에서 함께 매칭한 뒤, GT 기준으로 Hole 앞쪽과 뒤쪽을 구분했다. 입구 누락의 분모는 앞쪽 가시 GT가 하나 이상 있는 프레임이다.

| 실행 이름 | 검출기 | 최적화 옵션 | 평면 피팅 | 검증 | 입력 |
| --- | --- | --- | --- | --- | --- |
| 이전 | bb01149 | 꺼짐 | RANSAC | eager | NumPy |
| 최신·이전 설정 | 2b6c40e | 꺼짐 | RANSAC | eager | NumPy |
| RANSAC eager | 2b6c40e | 켜짐 | RANSAC | eager | NumPy |
| RANSAC compiled | 2b6c40e | 켜짐 | RANSAC | compiled | NumPy |
| Grid eager | 2b6c40e | 켜짐 | grid | eager | NumPy |
| 최신 기본 NumPy | 2b6c40e | 켜짐 | grid | compiled | NumPy |
| 최신 기본 CUDA Tensor | 2b6c40e | 켜짐 | grid | compiled | GPU 상주 |

세 급증 사례(350, 800, 1175)와 낮은 오차 대조 사례(950)에서는 최신 CUDA 기본 설정 중 평면 피팅만 RANSAC으로 바꾸고 같은 scripted expert 궤적을 재평가했다. GNN·Transformer, 카메라, 입력 자료형, 컴파일 설정은 최신 실험과 동일하다. 이 표본은 선정된 사례 검증이다. 최신 검출기의 RANSAC을 전체 1,839프레임에서 평가한 집계는 없다.

## Graph 21D 재실행

선정된 네 시드에서 기존 평가와 시드별 스텝, 파지·정렬 구간 및 GT 21D가 모두 일치했다. 아래 수치는 삽입 직전 다섯 프레임에서 각 시드의 **Peg 끝점 횡방향 오차 중앙값**이다.

| 시드 | 이전 bb01149 (mm) | 최신 grid (mm) | 최신 RANSAC (mm) |
| --- | ---: | ---: | ---: |
| 350 | 13.14 | 112.29 | 13.14 |
| 800 | 4.77 | 87.53 | 4.77 |
| 950 | 4.46 | 3.09 | 4.46 |
| 1175 | 17.17 | 97.94 | 17.17 |

세 급증 사례에서 평면 피팅만 RANSAC으로 바꾸면 기존 수준이 재현된다. 대조 사례 950에서는 grid의 횡방향 오차가 작았다. 네 시드의 삽입 직전 Hole 중심·축 각도 오차 중앙값도 최신 RANSAC에서 이전 값과 수치 허용오차 내에서 일치한다. [짝비교 JSON](../../outputs/hole_detector_cuda_regression/graph_comparison.json)

## 검출기 비교 결과

| 조건 | 120프레임의 Peg 3 mm 매칭 | Box 외곽 3 mm 매칭 | 앞쪽 Hole 3 mm 재현율 | 앞쪽 Hole 0개 매칭 |
| --- | ---: | ---: | ---: | ---: |
| 이전 bb01149 | 729/732 | 514/545 | 393/461 (85.25%) | 7/119 |
| 최신·이전 설정 | 729/732 | 514/545 | 393/461 (85.25%) | 7/119 |
| RANSAC eager | 729/732 | 514/545 | 393/461 (85.25%) | 7/119 |
| RANSAC compiled | 729/732 | 514/545 | 393/461 (85.25%) | 7/119 |
| Grid eager | 713/732 | 514/545 | 359/461 (77.87%) | 19/119 |
| 최신 기본 NumPy | 713/732 | 514/545 | 359/461 (77.87%) | 19/119 |
| 최신 기본 CUDA Tensor | 713/732 | 514/545 | 359/461 (77.87%) | 19/119 |

- 120프레임 중 119프레임에 평가 가능한 앞쪽 입구 GT가 있었다. 뒤쪽 Hole 림에는 가시 GT 판정을 통과한 점이 없었다. 12프레임에서 RANSAC에 있던 앞쪽 입구 매칭이 grid에서 0개가 됐다. 그중 8프레임에서 검출기가 사용한 Box 평면 개수가 줄었고, 나머지 4프레임은 평면 수가 같았다. 평면 수만으로 개별 실패를 설명할 수 없다.
- 따로 선정한 실패 12프레임의 입구 매칭은 전체 RANSAC 조건에서 30/45, 전체 grid 조건에서 0/45였다. grid로 바꿨을 때 10프레임에서 Box 평면이 세 개에서 두 개로 줄었다.
- 전체 132프레임에서 기존 검출기와 최신 RANSAC 경로의 프레임별 세부 매칭 수가 일치했다. 피팅 방식을 고정했을 때 eager/compiled 전환 및 grid에서 NumPy/CUDA Tensor 전환의 프레임별 매칭 수도 일치했다. 이 비교는 동일 매칭 수를 확인했으며 원시 검출 좌표의 일치까지 주장하지 않는다.
- 최신 CUDA 기본값의 grid는 물체 마스크 안에서 주변 깊이가 지지하는 법선을 픽셀 격자로 샘플링하고, 방향·오프셋이 비슷한 표본을 묶어 평면을 찾는다. Hole 후보는 선택된 Box 평면에 의존한다. 이번 실험에서 grid 설정의 영향은 분리됐고, 개별 프레임에서 어떤 평면이나 검증 조건이 입구 점을 탈락시켰는지는 분리하지 않았다.
- 팀의 별도 삽입 420프레임 평가에서도 같은 compiled 경로에서 RANSAC → grid 전환 시 Hole 림 매칭은 124 → 95/1,146이었다. 팀의 2,073프레임에서 집계한 74.3%는 별도 코호트의 기본 CUDA 결과다. [팀 검출기 문서](https://github.com/loveiscomplicated/graph-recurrent-VLA/blob/2b6c40e/docs/corner_detection.md)

## 재현 설정 및 산출물

검증에서 Graph 입력 후보로 확인한 호출은 최신 브랜치의 `TorchCornerDetector.main(device="cuda:0", plane_fit_method="ransac")`이다. 최신 검증 컴파일 기본값을 사용한다. 공유 코드와 기본값은 편집하지 않았다. 최적화별 지연이나 Moving Hole, 다른 모델 가중치는 평가하지 않았다.

Vast.ai RTX 3090에서 CPU PhysX 및 CUDA 렌더링·추론을 사용했다. 검출 감사 132프레임의 고유 시드와 두 작업의 종료 상태를 검증했다. Graph 재실행에서 네 시드 183프레임과 세 체크포인트/옵션 조건 간 평가 프레임 및 GT를 대조했다.

- [검출 집계](../../outputs/hole_detector_cuda_regression/run/summary.json) · [프레임별 검출 매칭](../../outputs/hole_detector_cuda_regression/run/frames.jsonl)
- [RANSAC Graph 실행 설정](../../outputs/hole_detector_cuda_regression/ransac_graph/summary.json) · [프레임별 21D](../../outputs/hole_detector_cuda_regression/ransac_graph/frames.jsonl)
- [검출 비교 코드](../../outputs/hole_detector_cuda_regression/review.py) · [RANSAC 재실행 코드](../../outputs/hole_detector_cuda_regression/replay_ransac.py) · [21D 비교 코드](../../outputs/hole_detector_cuda_regression/compare_graph.py)

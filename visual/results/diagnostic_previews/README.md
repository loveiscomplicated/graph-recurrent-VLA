# Visual 진단 대표 그림

전체 HTML·수치·dense NPZ·영상과 재현 입력은 [HF 게시 결과 인덱스](../publications/README.md)에 있다.
이 디렉터리는 Git에서 바로 검토할 대표 그림만 보관한다.

| 그림 | 확인할 내용 |
| --- | --- |
| [Task 01 frame 0](geometry_f0000.png) | Organized geometry·normal·경계·variation baseline |
| [Task 02-A 합성 비교](normal_synthetic.png) | Float / 1mm 양자화의 paired 반응 |
| [전체 121프레임 분포](normal_distributions.png) | 평면·구조 경계·가림·오염의 분포와 지원 비율 |
| [Frame 80 profile](normal_f0080_profile.png) | 평면의 계단형 depth와 normal 흔들림 |
| [Frame 0 경계 확대](normal_f0000_boundaries.png) | 실제 면 접합선·hole rim·가림·mask 경계 |
| [Task 02-B trade-off](comparison_tradeoffs.png) | Baseline·FD2·FD4·PCA5·PCA7의 평면 오반응과 경계 보존 비교 |
| [Task 02-B 합성 비교](comparison_synthetic.png) | 동일 합성 geometry에서 estimator별 반응 |
| [Task 02-B frame 0](comparison_f0000.png) | 동일 관측의 variation·면 접합선·hole rim |
| [Task 02-B normal validity](comparison_f0000_validity.png) | 유효 normal과 관측/계산 불가 영역 구분 |
| [Task 02-B profile](comparison_f0000_profile.png) | 같은 단면에서 경계 반응의 위치와 폭 |
| [PCA CPU 성능](pca_performance.png) | 원래 PCA5와 최적화 후보의 시간·정확도 보존 |
| [PCA frame 0](pca_f0000.png) | 같은 관측의 normal·variation·구조 보존 비교 |
| [Task 02-C trade-off](structural_tradeoffs.png) | Threshold별 precision·recall·일대일 점수와 경계 보존 |
| [Task 02-C 공간 분포](structural_density.png) | 원본 raster에서 후보가 집중되는 영역 |
| [Task 02-C frame 0](structural_f0000.png) | 방향 변화·depth jump·mask·가림 증거와 GT 비교 |
| [Task 02-C frame 120](structural_f0120.png) | 가림·tracker 지원 손실로 rim coverage가 낮은 사례 |
| [Task 03 trade-off](sampling_tradeoffs.png) | Point budget·priority ratio·오선택·CPU 지연 비교 |
| [Task 03 민감도](sampling_sensitivity.png) | Threshold와 seed에 따른 구조 보존 변화 |
| [Task 03 frame 0](sampling_f0000.png) | 같은 관측의 후보·선택점·GT와 hole rim 손실 |
| [Task 03 frame 0 3D](sampling_f0000_clouds.png) | 원본 관측과 S0–S4 sampled cloud 비교 |
| [Task 03 frame 120](sampling_f0120.png) | 후반 가림에서 관측 손실과 sampling 손실 구분 |

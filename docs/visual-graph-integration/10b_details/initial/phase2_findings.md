# Phase 2

네 frozen checkpoint adapter 동등성 exact PASS. Train96/Val12 저장 예측432개 파일 재사용, 원본 cache SHA·target·timestep 대조 통과. seed30003 모두 t0에서 controller translation error>1mm. Val Last E0 1.927mm/E1 1.837mm, 반전 구간8.287mm/7.510mm. 낮은 Action7 MSE가 translation 우위로 이어지지 않음.

| 모델/Checkpoint | Train MSE | Val MSE | Val translation mm | Val t0–40 mm | Val 반전 mm |
|---|---:|---:|---:|---:|---:|
| E0 validation_best | 0.004475 | 0.006416 | 1.924 | 1.595 | 8.335 |
| E0 last | 0.004463 | 0.006433 | 1.927 | 1.602 | 8.287 |
| E1 validation_best | 0.005261 | 0.007035 | 1.838 | 1.570 | 7.560 |
| E1 last | 0.005252 | 0.007053 | 1.837 | 1.562 | 7.510 |

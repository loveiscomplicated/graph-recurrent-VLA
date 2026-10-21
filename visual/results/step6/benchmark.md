# Step 6 벤치마크 — GDINO tiny + SAM2.1-small + point cloud (base_camera, 512², 2 objects)

- NVIDIA GeForce RTX 3090, torch 2.8.0+cu128, CPU 256 논리코어 (cgroup quota ≈31), torch 8 / OpenCV 8 스레드. SAM2 bf16 autocast, GDINO fp32.
- 워밍업 (측정 제외): GDINO 5회 × 프롬프트 2개 (실제 frame 0) + 전체 파이프라인 1 에피소드 (`ep000`).
- 모델 로드 17.3 s (측정 제외), 모델 weight 상주 1.02 GB.
- GPU 평균 열 = 3 ep 중 최댓값. 'GPU peak'는 `torch.cuda.max_memory_*` (모델 weight 포함, 단계 직전에 reset).

| 항목 | ep000 | ep001 | ep002 | **평균** |
|---|---|---|---|---|
| frames | 122 | 136 | 146 | 404 |
| grounding / query (ms) | 118.4 | 110.2 | 117.3 | 115.3 |
|   ├ 전처리 (HF processor, CPU) | 10.4 | 7.7 | 8.8 | 8.97 |
|   ├ forward (GPU) | 106.3 | 100.9 | 106.2 | 104.47 |
|   └ 후처리 | 1.5 | 1.4 | 2.1 | 1.67 |
| grounding frame 0 합계 (2 query) | 236.8 | 220.5 | 234.6 | 230.63 |
| SAM2 init: JPEG 쓰기 (측정 외 참고) | 5470.0 | 856.9 | 1973.4 | 2766.77 |
| SAM2 init: `init_state` (전 프레임 로드) | 4647.7 | 4573.4 | 4647.9 | 4623.0 |
| SAM2 init: box 2개 추가 | 15.5 | 15.3 | 15.3 | 15.37 |
| SAM2 첫 propagate step (frame 0 memory 인코딩) | 7.2 | 6.9 | 6.9 | 7.0 |
| tracking / frame, 2 obj (ms) | 46.3 | 46.3 | 46.5 | 46.37 |
| tracking / frame p95 | 47.0 | 47.0 | 47.2 | 47.07 |
| point cloud / frame, 2 obj, CPU (ms) | 35.8 | 15.9 | 30.5 | 27.4 |
| point cloud / frame p95 | 44.8 | 42.5 | 47.8 | 45.03 |
| **FPS steady** (track + pc) | 12.18 | 16.08 | 12.98 | 13.75 |
| **FPS online 추정** (+ 프레임 로드/T) | 8.32 | 10.43 | 9.18 | 9.31 |
| **FPS e2e** (grounding·init 포함 / T) | 8.2 | 10.28 | 9.06 | 9.18 |
| FPS wall (JPEG·npz 저장·평가 포함) | 4.84 | 5.79 | 5.13 | 5.25 |
| GPU peak alloc: grounding (GB) | 1.89 | 1.89 | 1.89 | 1.89 |
| GPU peak alloc: tracking (GB) | 3.12 | 3.32 | 3.46 | 3.46 |
| GPU peak reserved: tracking (GB) | 3.95 | 5.68 | 7.54 | 7.54 |
| peg IoU mean | 0.9437 | 0.2472 | 0.8637 | 0.685 |
| box IoU mean | 0.9456 | 0.9161 | 0.9638 | 0.942 |

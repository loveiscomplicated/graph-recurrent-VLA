# Step 6 벤치마크 — GDINO tiny + SAM2.1-small + point cloud (base_camera, 512², 2 objects)

- NVIDIA GeForce RTX 3090, torch 2.8.0+cu128, CPU 256 threads. SAM2 bf16 autocast, GDINO fp32.
- 워밍업 (측정 제외): GDINO 5회 × 프롬프트 2개 (실제 frame 0) + 전체 파이프라인 1 에피소드 (`ep000`).
- 모델 로드 28.9 s (측정 제외), 모델 weight 상주 1.02 GB.
- GPU 평균 열 = 3 ep 중 최댓값. 'GPU peak'는 `torch.cuda.max_memory_*` (모델 weight 포함, 단계 직전에 reset).

| 항목 | ep000 | ep001 | ep002 | **평균** |
|---|---|---|---|---|
| frames | 122 | 136 | 146 | 404 |
| grounding / query (ms) | 293.3 | 162.0 | 270.2 | 241.83 |
|   ├ 전처리 (HF processor, CPU) | 151.6 | 58.5 | 123.5 | 111.2 |
|   ├ forward (GPU) | 140.2 | 102.0 | 144.5 | 128.9 |
|   └ 후처리 | 1.3 | 1.4 | 2.0 | 1.57 |
| grounding frame 0 합계 (2 query) | 586.6 | 324.0 | 540.5 | 483.7 |
| SAM2 init: JPEG 쓰기 (측정 외 참고) | 5920.9 | 845.3 | 5945.4 | 4237.2 |
| SAM2 init: `init_state` (전 프레임 로드) | 7679.6 | 8558.2 | 9708.6 | 8648.8 |
| SAM2 init: box 2개 추가 | 15.6 | 15.7 | 16.5 | 15.93 |
| SAM2 첫 propagate step (frame 0 memory 인코딩) | 7.3 | 6.1 | 6.9 | 6.77 |
| tracking / frame, 2 obj (ms) | 46.3 | 46.6 | 46.3 | 46.4 |
| tracking / frame p95 | 47.0 | 47.3 | 46.9 | 47.07 |
| point cloud / frame, 2 obj, CPU (ms) | 35.3 | 17.2 | 29.3 | 27.27 |
| point cloud / frame p95 | 45.5 | 46.5 | 42.0 | 44.67 |
| **FPS steady** (track + pc) | 12.26 | 15.68 | 13.23 | 13.72 |
| **FPS online 추정** (+ 프레임 로드/T) | 6.92 | 7.89 | 7.04 | 7.28 |
| **FPS e2e** (grounding·init 포함 / T) | 6.71 | 7.76 | 6.87 | 7.11 |
| FPS wall (JPEG·npz 저장·평가 포함) | 4.29 | 5.62 | 4.63 | 4.85 |
| GPU peak alloc: grounding (GB) | 1.89 | 1.89 | 1.89 | 1.89 |
| GPU peak alloc: tracking (GB) | 3.12 | 3.32 | 3.46 | 3.46 |
| GPU peak reserved: tracking (GB) | 3.95 | 5.68 | 7.54 | 7.54 |
| peg IoU mean | 0.9437 | 0.2472 | 0.8637 | 0.685 |
| box IoU mean | 0.9456 | 0.9161 | 0.9638 | 0.942 |

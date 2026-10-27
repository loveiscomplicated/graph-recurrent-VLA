# Experimental GPU PCA5 and sampling

Run `b27c52d1087743589177a15a1c3acb42` on Vast instance 57987772, RTX 3090 (driver 580.178.04), AMD EPYC 7K62. Input SHA-256: `9053b5dcc3f37ae0a124411514e74d52be7711c1bf346afd1dae22fa902d0bae`. The run records its environment, input provenance, source snapshots, individual timings, and comparison metrics in its ExperimentLogger directory. Sources correspond to commit `3673098` (source snapshot hashes are authoritative).

The recorded current-observation depth and masks comprise 121 prototype 512×512 frames and 12 frames each from episodes 30002 and 30047 at 128×128. Each scope uses the same frames on the same host, three timed repetitions with alternating mode order, after full-input warmup. Times include synchronization and host/device transfers, and exclude startup and compilation. No SAM2, Raw EDE, model training or policy inference is measured.

| Scope | Input | CPU p50 / p95 ms | GPU p50 / p95 ms | p50 speedup |
|---|---|---:|---:|---:|
| PCA5 evidence + S3 | prototype 512 | 36.399 / 71.344 | 13.874 / 21.534 | 2.62× |
| PCA5 evidence + S3 | ep30002 128 | 4.058 / 9.246 | 2.077 / 2.893 | 1.95× |
| PCA5 evidence + S3 | ep30047 128 | 9.399 / 14.338 | 3.029 / 3.660 | 3.10× |
| Shape-First + F5/EE packet, including evidence | prototype 512 | 59.747 / 136.672 | 35.292 / 73.103 | 1.69× |
| Shape-First + F5/EE packet, including evidence | ep30002 128 | 9.710 / 15.031 | 7.760 / 10.019 | 1.25× |
| Shape-First + F5/EE packet, including evidence | ep30047 128 | 18.794 / 29.640 | 12.834 / 18.178 | 1.46× |

GPU means the opt-in FP32 CUDA PCA5, pixel features, pair angles and FPS path; geometry construction, pair assembly, candidate construction, contour/voxel selection and packet processing still run on CPU. The CPU default remains unchanged. The earlier 16.65 ms prototype figure used a different host and benchmark; it should not be compared directly to these numbers.

On these frames, GPU-vs-CPU normal validity, 20-degree orientation candidate membership, selected point counts and F5 validity had no observed differences. The worst per-frame/object normal-angle p95 was 0.0000296 degrees. Minimum selected-source Jaccard was 0.921 for S3 and 0.761 for Shape-First; worst 95th-percentile spatial coverage ratio (GPU / CPU) was 1.0014 and 1.0082 respectively. Matched-source F5 column maximum absolute differences were approximately 1.05e-5, 7.45e-6, 6.96e-6, 1.35e-5 and 0. The run's provisional geometry screen passes, but these values do not establish policy quality or task success. FP32 and point-selection differences are permitted in this experiment; no bitwise equivalence is claimed.

Local tests: 38 relevant sampling and Shape-First tests passed. This is a partial GPU implementation and an offline replay benchmark, not a live end-to-end 20 Hz result. Further validation should measure trained policy/task success, live workloads, DINO/SAM2 contention and end-to-end latency before adopting the path.

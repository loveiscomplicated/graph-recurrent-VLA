# Task 10-I — existing packet-to-World observation timing

Measurement only. The primary endpoint calls the unchanged
`observations_from_packet(packet, camera)` on the same real512 hand-camera input.
The CPU EE packet is ready before timing. Object-ID filtering, copies, EE→World
point/normal transforms, both Peg/Box observation construction and validation
are included. Projection/PCA/sampling, GPU→CPU transfer, packet construction,
multi-view consistency, per-point records, deduplication, Graph and robot
filtering/gate are excluded. Box is empty in this recorded input.

Independent diagnostic samples repeat the same expressions with phase clocks;
every returned dataclass field is compared exactly against the public helper.
The primary timing has no internal clocks. All timing is CPU wall time on
VastAI58752018; CUDA is used only to prepare the existing packet outside timing.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmark.py --inputs /path/to/inputs --output /path/to/new/output \
  --warmups 20 --repeats 100
```

Run with `PYTHONPATH` pointing to the preceding packet-timing source snapshot.
This preserves the existing path while unrelated CUDA-ingress work continues in
the main checkout. The measured fusion/contracts sources must match the checkout;
source hashes, input hashes and the shared ExperimentLogger UUID are retained.
No publishing is performed.

Full [report](../../../docs/visual-graph-integration/10i_observation_cpu_timing_58752018_20261103.md).

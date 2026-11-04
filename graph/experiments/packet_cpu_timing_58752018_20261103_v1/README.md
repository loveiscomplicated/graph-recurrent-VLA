# Task 10-I — ready GPU sampling output to existing CPU coordinates/packet

Measurement only; runtime conversion/validation algorithms are unchanged.
`benchmark.py` prepares the real512 hand-camera sampling result on CUDA, waits
for completion, then times its existing CPU tail. All34 tensor fields are
downloaded with the existing `_map_tensors(..., t.to(device='cpu'))` operation.
The CPU NumPy/source packet helpers and `complete_selected_vision_packet` run
unchanged with the recorded camera calibration and TCP pose.

Two endpoints are measured separately: returned EE packet, and the actual
`make_packet` return including `receive_vision_packet` validation/copy. The
entire packet is compared exactly against a fresh public `make_packet` call
outside timing. Independent phase samples separate transfer, NumPy/source
packet, checked EE packet construction, and receiver validation/copy.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m graph.experiments.packet_cpu_timing_58752018_20261103_v1.benchmark \
  --inputs /path/to/inputs --output /path/to/new/output --warmups 20 --repeats 100
```

Inputs are `hand_camera.npz`, `robot.npz` and `masks.npz` from the preceding
native512 case. CPU sensor arrays already exist as in the current runtime.
Only selected sampling output is transferred. Projection/PCA/Shape-First,
world observations, camera fusion/dedup, Graph edges and robot filtering are
excluded. This is not pure matrix-multiplication or pure PCIe kernel timing.
The runner enforces instance58752018, checks for other active CUDA processes,
uses shared `ExperimentLogger`, disables W&B and does not publish anything.

Full [report](../../../docs/visual-graph-integration/10i_packet_cpu_timing_58752018_20261103.md).

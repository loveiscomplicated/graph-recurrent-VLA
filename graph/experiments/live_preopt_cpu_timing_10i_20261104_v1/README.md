# Task10-I original CPU/NumPy stage timing

Measure the CPU implementation immediately before masked CUDA projection became
the default: commit `709798983435875e374e9a8cd6df8af9e6c7bb7f`
(`b423d00^`). Keep its full-image XYZ, CPU PCA/Shape-First, World round trip,
Python records/dedup, CPU final FPS, intermediate Raw EDE and repeated-depth
final surface tests. No shortcut or CPU algorithm fix is applied.

`capture.py` extends the previous source-pinned main-live timing runner. It
records current512 two-camera native RGB/depth/calibration, robot-only surfaces
and actual Terra/native SAM2 VOS raw masks, then verifies identical masks on
the two GPU trace replays. Sensor-stage CPU readback is timed using the original
`sensors` function AST; capture, export and file writes are outside GPU timers.
Three actual live passes preserve the original initialization/warmup boundary.

`legacy.py` runs in a separate source tree/process at the old commit. It consumes
the exact recorded input arrays and executes the original `CurrentGeometry.build`
with all CPU filtering/gate/geometry and downstream processing. File loading is
outside timers. Dynamic spans separate original stages; intermediate Raw EDE is
an additional row. Uninstrumented CPU replays at three frames must match all
five Graph arrays, node IDs and gate trace. Legacy results are CPU Graphs.

This is a controlled CPU-geometry comparison under the current512/Terra/SAM2
input conditions. The old128 renderer, DINO grounding and JPEG SAM adapter are
not substituted into the common frontend. Shared upstream times and standalone
CPU geometry replay times are labelled as such, not presented as a single
stopwatch of the historical complete live runner.

Use the existing Vast58752018 venv, one Torch/BLAS thread, a fixed source manifest,
the existing runtime assets, shared `ExperimentLogger`, disabled W&B and no
publishing. Preserve source, input hashes, spans, raw per-frame times and output
verification under a new experiment directory. Full report belongs in
`docs/visual-graph-integration/`; add the comparison to the50ms status table.

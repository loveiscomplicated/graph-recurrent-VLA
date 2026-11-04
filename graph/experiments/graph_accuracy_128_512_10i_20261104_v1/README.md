# Task10-I current graph accuracy: native 128/512

This independent pilot reuses the completed native capture from
`/workspace/task10i_resolution_perception_graph_20261103_v1` and the current
`rollout.py` imports for LiveGrounding, SAM2Tracker, OnlineVisionAdapter and
CurrentGeometry. Four existing Train parents, both cameras and t0..20 are fixed
before inference. It is an approximately one-second prefix, not a full episode.

`infer.py` uses separate causal camera states for each parent/resolution, the
current CUDA RGB preprocessing, Terra high/default appearance prompt, and the
default CUDA graph path. It does not read GT arrays. Sensor frames are uploaded
from the preserved native CPU-rendered capture, so this is recorded-input
perception/graph accuracy, not renderer equivalence or closed-loop evaluation.
The two size orders alternate by parent. GT scoring begins only after inference
is complete. Existing experiment outputs are never resumed or overwritten.

`evaluate.py` reuses the existing pixel metrics, correct-source coverage and
hierarchical paired bootstrap. Both resolutions use the SAME currently visible
native1024 reference, with Peg in both views and Box in Main only. Correct
representative source labels gate coverage credit. Native masks are scored
against their native segmentation, never resized. A common >=1024-pixel support
criterion defines visibility at native1024 (16-pixel equivalent at128). Tiny and
absent frames are retained separately. Empty precision is undefined; missing
correct points remain failures. No edge GT or physical end-pose accuracy is
claimed. All parent-level results and denominators are retained.

Run from an isolated source snapshot on the existing CUDA host, with its existing
model/runtime assets. The inference command requires an environment credential;
it must never be embedded in argv, config, files or logs. Authentication fallback
aborts this primary-Terra comparison. Both entrypoints create fresh common
ExperimentLogger IDs and local outputs; no external publishing is performed.

`localize_information_loss.py` is the finer frozen-input audit. It separates
cross-view rejection from dedup, records lost/recovered common-reference samples,
and follows the surface samples gained only by native512. End bands, projected
boundaries and supported approach-rim samples are evaluation regions only.
Priority C/R/P branch controls hold the actual Surface sources fixed. FPS256,
native-variable-N and paired matched-N controls are point sets; they do not
change or validate a new production Graph interface. All observed graphs must
match the saved five tensors, and valid F5/XYZ/validity packing must be exact.

`probe_structural_and_edges.py` follows priority/surface source retention through
the final FPS and conserves each object's edge rejection funnel.
`edge_reference_control.py` reuses the existing surface-edge function AST,
replacing exactly its witness XYZ/object reference reads. Final-node reference
must reproduce every original edge before using the same-time pre-FPS dedup pool
as a GT-free diagnostic reference. This control measures connectivity sensitivity;
it does not establish edge GT accuracy or change the production Graph.
Both scripts accept the same `--capture`, `--inference`, `--runtime`, and fresh
`--output` arguments as the localization script and reuse ExperimentLogger.
`plot_information_loss.py <local-output-root>` renders their completed summaries.

Pure metric controls:

```sh
.venv/bin/python -m unittest \
  graph.experiments.graph_accuracy_128_512_10i_20261104_v1.test_loss_metrics \
  graph.experiments.graph_accuracy_128_512_10i_20261104_v1.test_metrics -v
```

# Legacy FPS feasibility extension

This opt-in, team-owned diagnostic preserves `legacy_fps.*` imports. Importing
the package enables nothing. `integration_guard.load_candidate(..., enabled=False)`
returns `None` without reading a manifest. No production sampler, shared simulator,
G0–G4 code, training identity, or policy interface is changed.

The original 100-update pilot has 36 diagnostic encoders and 244 frozen probes.
There was no independent Final Test, policy training, or closed-loop evaluation.
Historical global protection is **PARTIAL**; later compatibility and protection
window checks are **PASS**. See the [research report](../docs/visual-graph-integration/09c_legacy_fps_extension.md)
and [publication/download instructions](../docs/legacy_fps/publication.md).

## Use the frozen dependency tree

Current HEAD differs from the verified common code. Do not replace the current
`graph/`, `visual/`, or `utils/` with another experiment's files. Extract
`pinned_sources.tar.gz` from the selected Dataset ZIP, then use its `pinned_source/`
directory. The tar also preserves the unchanged historical extension sources.
The release manifest pins the 817 common files separately from the ported Python
files. `historical_integration_manifest.json` preserves the original hashes.

From the checkout root, using the existing Python environment:

```bash
python -m legacy_fps.run_pinned --source-root /path/to/pinned_source \
  unittest legacy_fps.tests graph.persistent_geometry.tests.test_contracts legacy_fps.test_release -v
```

The launcher creates a fresh interpreter, validates both trees, and selects the
ported extension plus frozen common modules. `load_candidate` requires explicit
`enabled=True`, matching hashes, and matching imported module locations. A mutable
checkout is rejected even if a different frozen directory passes its hashes.

Machine-local paths use `LEGACY_FPS_ROOT` (default `outputs/legacy_fps`) and
`LEGACY_FPS_CAPTURE` (default `outputs/task09c_capture`). They replace the original
Vast.ai absolute defaults. Historical configs and recorded paths are provenance
and remain unchanged. Full RGB-D/masks are excluded from this minimal release;
re-running geometry or training requires those separately retained inputs.

Existing runners remain available: `legacy_fps.benchmark`, `branches`, `temporal`,
`learning`, `latency`, `render`, `report`, and `verify`. Run their `--help` through
`run_pinned`. Benchmark's `--capture` and `--output` also override input/output
locations. The old protection auditor intentionally requires its recorded
protection files; it is not a new-machine protection baseline. No runner was used
to train or capture new data during this release.

## Prepare and verify a publication

Only the file paths in `docs/legacy_fps/release_selection.json` are selected.
Preparation writes a **new** output directory and preserves original run UUID,
team, Git commit/dirty state, ownership, and checkpoint bytes. It uses the existing
`register_existing_output` API; no new logger or uploader is added.

```bash
python -m legacy_fps.prepare_release --artifact-root /path/to/task09c_legacy_fps \
  --output outputs/legacy_fps_release
python -m legacy_fps.run_pinned --source-root outputs/legacy_fps_release/pinned_source \
  legacy_fps.verify_release --package-root outputs/legacy_fps_release \
  --artifact-root /path/to/task09c_legacy_fps
```

The full local readback additionally uses original saved graph/features to check
2,232 forward outputs, 108 permutations and all probe predictions without fitting.
The selected download includes only representative frame evidence; it supports
checkpoint loading and contract replay, not full experiment recomputation.

Publishing is a separate, explicitly authorized step using
`python -m tools.publish_experiment`. Keep `--private`, inspect `--dry-run`, and
use `--report-files` for the explicit Dataset ZIP selection. Both default target
repositories were checked private for this release. If either becomes public,
use its `-legacy-fps-private` alternative without changing visibility. Download
and repository revisions are recorded in the publication document.

The port only changes path defaults and release/isolation tooling. Numerical
adapter, filter, map, graph, encoder-training and probe code retain the verified
snapshot's bytes. The release uses `ExperimentLogger` and the existing artifact
publisher; GPU experiments and external dependency installation are outside scope.

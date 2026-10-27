# Legacy FPS private release

The release is prepared on `codex/task09c-legacy-fps-release` in a separate
worktree. Execution metadata retains its original Git commit and dirty flag;
the implementation commit is recorded separately as `publication_commit`.

## Download

Publication receipts and immutable Hugging Face revision URLs will be recorded
here after private upload and remote hash verification. See
[usage and local preparation](../../legacy_fps/README.md).

The Model selection contains 36 diagnostic encoder checkpoints, 244 probe
decoders, configuration, split, and hash manifests. The Dataset selection contains
Markdown/HTML reports, aggregate metrics, PNG/MP4 evidence, validation logs,
integration manifests, 817 pinned common source files, historical extension
sources, and representative t0 evidence. Full RGB-D/masks, per-frame tensors,
map histories, frozen feature arrays, and duplicate transfer archives are excluded.

Unzip the Dataset report bundle into a new directory. Open
`task09c_legacy_fps/experiments/8c41299e9d3e4989bbf32ab1b894dc49/legacy_fps_extension.html`
or `repository/docs/visual-graph-integration/09c_legacy_fps_extension.md`.
Extract `pinned_sources.tar.gz` for contract replay with the implementation checkout.
Omitted full-frame evidence links lead to `omitted.html`, which states the exact
release limitation; no missing tensor is presented as downloadable.

Historical global protection remains **PARTIAL**. The later pinned-source
compatibility/protection window is **PASS**. Independent Final Test,
CUDA/MPS/simulator execution on this Mac, retraining and policy evaluation are
**NOT_EXECUTED** during release. This is the 100-update diagnostic pilot.

No Git remote push or HF Space deployment is included.

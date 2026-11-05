# Project context: reusable simulation and data collection

2026-11-05: `develop` branches from `main` for a project pivot. Carry forward
native CUDA RGB-D/output-device support and first-frame GT/depth peg visibility
selection in `data_collection/`. Keep simulator responsibilities in
`envs/maniskill/`; data selection and storage belong in `data_collection/`.
The default filter preserves both end strips plus the historical grasp band;
full camera-facing surface confirmation is an explicit alternative.

Use human-readable role names. No Graph, GNN, recurrent model, old expert or
training workflow is selected by this extraction. Existing legacy branches and
artifacts remain preserved. GPU execution requires a supported Linux GPU host;
the extraction host is macOS without CUDA.

## Original shared prototype context

This repository's shared responsibility is a working ManiSkill 3
`PegInsertionSide-v1` environment. It provides environment creation, reset/step,
rendering examples, and direct simulator ground truth access. It does not define
feature representations, model architectures, datasets, training loops,
evaluation schemes, configuration systems, or integration contracts.

The Visual team uses RGB observations to build its own visual prototype and can
compare outputs with simulator GT. The Graph team starts from simulator GT and
chooses geometry, graph construction, and a GNN prototype. The Recurrent team
collects a simulator GT sequence and chooses features and a feed-forward or
recurrent controller prototype. Each team works without waiting for another.

`visual/`, `graph/`, and `recurrent/` are intentionally empty workspaces. Teams
may organize their directories however they wish. Architecture decisions follow
the prototype results, not this shared repository's scaffold.

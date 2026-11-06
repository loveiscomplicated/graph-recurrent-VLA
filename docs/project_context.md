# Project context: reusable simulation and data collection

## Reusable capabilities on graph

The `graph` branch carries the standalone [corner detector](corner_detection.md)
from `visual`, plus [initial-visibility collection](data_collection.md) and
[GPU observation output](gpu_observations.md) from `develop`. These are reusable
tools, separate from existing model/training experiments. Dataset selection and
storage live in `data_collection/`; the shared simulator remains responsible for
environment creation, raw observations and GT. The default visibility gate
requires both peg ends plus the grasp band; full-surface confirmation is optional.

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

Teams organize `visual/`, `graph/`, and `recurrent/` independently. Architecture
decisions follow prototype results rather than a shared prescribed design.

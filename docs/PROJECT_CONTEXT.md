# Project context: week 3 feasibility prototypes

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

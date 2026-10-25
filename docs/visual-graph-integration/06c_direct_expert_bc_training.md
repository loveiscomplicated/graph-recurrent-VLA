# Task06-C — Direct Expert Data and Graph–GNN–GRU/BC

Pipeline run: `d1fd6173cb3648bfa4c03507f84b7e71`. **Raw collection completed;
the user stopped the run during Vision. Training did not start.**

## Published data

- [Complete raw RGB-D collection on Hugging Face](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/4f63b8ae0cdd14aba83417ff2c5180f7264c6935/integration/b8d6db904ba34f6f8dfefe14c5a90cab): 120 validated successful episodes from 127 attempts. The archive includes both cameras' RGB-D, calibration, robot state, action records, dataset/split manifests, and schemas.
- [Partial causal Vision/Graph packets on Hugging Face](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/8352c757fac743fe61695838c2393d2cc32ec96f/integration/ae2de828d628421fa1560c4393befe73): five fully processed episodes (30000–30004), 616 observations, masks, selected points, EE-frame features and object/edge packet data, together with their five raw episodes. The archive manifest marks this as **5/120 processed**.

The raw collection and processed packet archives are separate, immutable artifacts.
The processed Vision run stopped at `vision` after a normal `user CLI stop` on
2026-10-25 09:11 UTC. It had started episode 30006 but preserved only the five
complete episode receipts above. Those five packets use `base_camera`, as frozen
in `config.yaml`; `hand_camera` is present in the raw collection but has no
processed packet output in this run. No GNN-GRU validation, training, checkpoint,
offline test, or closed-loop policy evaluation was produced.

## Collection and planned training settings

Collection used `PegInsertionSide-v1`, CPU PhysX, `pd_ee_delta_pose`, 20 Hz
control, a 250-step horizon, zero action noise, and no hole shift. The 120
successful episodes are split 96/12/12 by episode/reset group with seed 42.
Each stored episode has T actions and T+1 observations, 76D robot state, both
cameras' RGB/depth/calibration, and native Action7/controller alignment checks.
The collector's readback and integrity gate passed for all 120 manifest entries.

Vision used the existing causal RGB-only initialization and forward SAM2 path,
followed by observed RGB-D point construction, EE-frame coordinates, object
membership, and raw EDE packets. Missing object observations are retained; they
are not filled from GT. Five episodes passed full-frame packet generation before
the stop.

The frozen, **unexecuted** training plan uses CPU, AdamW (`lr=3e-4`,
`weight_decay=1e-5`), CosineAnnealingLR (`T_max=50`), up to 50 epochs with
patience 10, gradient clip 1, TBPTT 32, batch size 2, train-only robot
normalization, validation-only checkpoint selection, and one offline test.
These settings are not training results.

Run status, stop/resume commands, and data contracts are documented in the
[Graph integration baseline README](../../graph/integration_baseline/README.md).
The run's ignored local `experiments/` directory contains the full receipts,
logs, and stop marker. The published packet archive's `publication_scope.json`
and `vision_partial_manifest.json` specify the included subset and completion
state.

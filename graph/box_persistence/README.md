# Task09-E Box-only persistence diagnostic

Independent opt-in research package. Production graph defaults, Plan A and09-C
implementations, datasets, checkpoints and previous reports remain unchanged.
There is no Peg memory, policy, GRU or BC training entrypoint here.

`core.BoxMemory` reuses09-C surfels, duplicate fusion, connected support,
free-space contradiction, aging and estimated registration. B1 and B2 share all
admission/fusion/selection rules; only B2 applies an accepted registration pose.
B1 uses registration as a consistency/motion diagnostic, while its coordinate
frame stays exactly world identity. This is a conservative candidate, not a
claim about every possible static world map. Initial component selection and
registration-based admission can discard valid newly observed surfaces.

The input is the unmodified09-D C3_SF current Box builder, Main only. B0 is
reconstructed through the original Plan A adapter and compared exactly with its
stored cap128 tensors. Peg XYZ/F5/provenance, Robot76,21 features, cap128 and
surface/k4/M16 policies stay fixed. B1/B2 introduce memory admission as part of
the persistence implementation; B0 has no extra preprocessing. Consequently
their difference is not an isolated causal estimate of storing points alone.

The map accepts no GT actor pose, mesh, segmentation or future frame. GT mask
and relative motion upper bounds live in `oracle.py`/`motion_oracle.py` and
evaluation code only. Full and historical *observed* reference surfaces have
different meanings. Historical depth jump/variation are unavailable and zero
in tensors. Temporal/status extras are zero to match Plan A; full lifecycle
status/confidence/uncertainty/provenance remain sidecars.

`protocol.json` fixes geometry thresholds and reserves Final84261/84262 before
measurement. The completed frozen candidates both fail the development geometry
gate, so E0/E1/probe training and independent Final are not executed. This does
not establish A/B GNN ranking or closed-loop manipulation success.

## Modules

- `preflight`: fresh shared ExperimentLogger UUID, identities and protection.
- `evaluate`: all-frame causal maps on12 existing diagnostic episodes; every5th
  and last frame native/cap128/exact primary matched-N geometry and graphs.
- `capture_controls`: live CPU PhysX/SAPIEN and CUDA DINO/SAM2, same-robot
  ±10mm Box interventions with actual visual-only full/partial occluders.
- `motion`: independent prefix copies; separate optional baseline-initialized
  t0 diagnostic when original t0 has no history and t25/65 are not observable.
  Physical timestep and diagnostic tick are explicitly different. Reappearance
  re-presents an actual visible snapshot; there is no fresh physics/contact claim.
- `motion_oracle`: separate GT-mask/motion interventions on the same t0 data.
- `summarize`: preregistered episode-macro gate; no threshold tuning.
- `diagnostics`, `media`, `assemble`, `report`, `verify`: read-only diagnostics,
  PNG/MP4 decode, portable report, all-file readback and paired/protection checks.

Use the existing environment and offline cached models. No dependency installer,
git commit, publishing or old-process control is invoked. Machine-specific
preflight/assembly paths target `/workspace/task09e_plan_b` on Vast57640710; the
retained source snapshot and shell logs record the actual commands. This is not
a general-purpose simulator or final model interface.

Synthetic contracts:

```bash
.venv/bin/python -m unittest graph.box_persistence.tests -v
```

Completed evidence and limitations are in
[the09-E report](../../docs/visual-graph-integration/09e_plan_b_box_persistence.md).

# Task10-F corrective Reach prerequisite pilot

`pilot.py` reuses Task10-D's runtime, capture schema, ScriptedExpert and raw
physical accessors. It runs only on Vast instance 57640710 and writes into a
new root. Inherited sources, checkpoints and caches are never edited.

Run in the installed simulator environment:

```bash
python -B -m graph.corrective_reach10f.pilot --root NEW_TASK_ROOT
```

The root needs read-only copies of Task10-D config.json, normalizer.json,
action_codec.json, collection_candidates.json, inputs/ and the renderer gate.
The existing renderer gate identifies a requested backend, not proof of
cross-host RGB-D equivalence. That additional gate is required before training.

The protocol is written before simulation. Parent reset, full state/controller
and clock restoration, and independent five-step feedback replay are tested
before disturbances. BatchedRNG states are serialized per underlying NumPy RNG;
the installed class cannot be deep-copied directly. Controller interpolation
fields and joint drive targets are explicitly restored because the simulator's
set_state_dict does not restore them. PhysX contact solver memory is not claimed
to be serialized; trajectory replay tests the observed consequence.

Disturbances use eight real pd_ee_delta_pose control steps, a fixed world target
and bounded 2mm translation commands, preserving every RGB-D/Robot76 observation.
There is no TCP/joint teleport disturbance. Fresh ScriptedExpert OBB targets must
agree with the parent target. Recovery explicitly restarts the elevated approach
phase and then uses the original feedback servo and phase thresholds.

Each branch H5 contains the full causal prefix, disturbance and recovery with
T+1 observations/T actions. Prefix and disturbance supervision is false;
recovery supervision is true. No learned GRU hidden is stored or injected.
Metadata stays outside the student feature schema. This module does not yet
implement masked cache loading, causal vision equivalence, fine-tuning or policy
evaluation. Its presence is not evidence those stages ran. See verification.json
and the full report under docs/visual-graph-integration/ for executed scope.

The safety probe checks joint limits, finite states, broad TCP workspace, object
translation and robot/external contact impulses at control boundaries. It does
not prove substep collision freedom or audit all possible robot self collisions.
A complete data gate requires those limitations to be addressed before expansion.

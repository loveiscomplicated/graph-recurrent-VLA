# Task09-D Plan A geometry diagnostic

Independent opt-in package. It uses synchronized current `base_camera` and
`hand_camera` RGB-D, causal predicted masks, current robot observations and
per-link collision geometry. Peg may use both cameras; Box uses Main only.
There is no persistent geometry state, pose transport, policy, GRU or action head.
Production defaults, original captures, splits, weights and reports are read-only.

`core.PlanA.prepare` rejects non-sensor fields. The Wrist Peg gate checks positive
depth, post-filter support, connected 3D support, local neighbors, extent and the
09-C same-time depth-supported identity conflict rule. Missing Main is permitted.
Binary tracker masks provide no calibrated confidence; none is invented.
Confirmed and uncertain robot masks remain separate artifacts; both are excluded
by the reused09-C filter policy. Ground truth is opened after this computation.

C0/C1/C2/C3 share SF selection and3D edge construction. C1 retains both objects
from both views with the existing09-C pointwise semantic association. C2 removes
Wrist Box, and C3 adds the observation-level Wrist Peg gate. These controls do
not claim to reproduce historical G3's node count after the new common dedup rule.
Historical G0/G1/G3 are separately regression-tested before this comparison.

SF reuses the original PCA5, silhouette, keep_ratio0.5 and surface selection.
Legacy reuses erosion2/fallback50, pre2048, SOR k16/std2 and FPS1024, with exact
source row joins. Its original native feature is XYZ; the diagnostic comparison
explicitly augments it with same-source F5. Hybrid protects SF structural anchors,
then fills from post-SOR surface candidates BEFORE Legacy final FPS, using anchored FPS. Erosion/SOR never
acts on structural anchors. Budget overflow deterministically subsamples anchors.
Dedup retains a real representative measurement and attaches all source IDs;
structural representatives have priority, then stable source identity. It never
averages XYZ while borrowing one camera's depth-jump feature.

Legacy fused native points are capped at1024/object; Hybrid native points also at1024.
Native points, cap128/object graphs and exact matched-N point/graph subsets are
separate artifacts. Short support remains short; absent objects stay empty.
All21 tensor features match across E0/E1. The seven temporal/status extras are
zero. TCP/left/right nodes and the original k4/M16 interaction helpers remain.
The reused surface edge gate uses the selected current support as its witness
reference. This conservative choice can fragment sparse graphs and is audited.
Raw EDE remains a separately saved single-camera historical reference.

`protocol.json` reserves new Final73141/73142 and fixes thresholds before Final.
Development uses the existing12 diagnostic episodes, with episode/reset groups
preserved. Geometry/GNN rows use every5th timestep, the last frame and existing
same-robot branches; gate/filter diagnostics inspect every frame. No FrozenTest12
or old Final62091/62092 enters development or new final fitting.

The training runner calls the existing09-C `Dataset` and `train_one` directly:
E0/E1, seed42, AdamW lr0.001, batch16,2,000 updates, validation every50 updates,
first lowest-loss checkpoint. The probe runner preserves the09-C Ridge/MLP loop;
only task identity and reserved Final seeds differ. The frozen final runner uses
the original no-fit application and split/weight/time guards.

Run in an isolated source snapshot with existing dependencies and offline model
cache. `pipeline` is the machine-specific Vast57640710 sequential orchestrator.
It writes stage state/logs under `/workspace/task09d_plan_a/v2` and new large outputs
under `/dev/shm/task09d_plan_a/v2`; these must be retrieved before instance shutdown.
It never installs, commits or publishes. `runtime`, `tests`, `verify` and `media`
provide simulator regression, synthetic contracts, provenance readback and PNG/MP4
inspection. A completed experiment report must distinguish measured failures,
unsupported scenarios and backend/device execution from source-only checks.

The isolated deployment points `code/visual/checkpoints` to the existing read-only
checkpoint directory. `model_identity.json` pins the installed SAM2 checkpoint,
Grounding DINO cache files/configs and environment script before/after capture.
No pretrained model is downloaded or installed. The first implementation attempt
used final-FPS surface candidates for Hybrid; it is excluded, retained separately,
and superseded by the pre-final-FPS SOR candidate implementation before Final.

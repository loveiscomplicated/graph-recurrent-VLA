# PegInsertionSide subtask segmentation v0.1

## Scope and evidence

Implemented against `docs/subtask.md` (Subtask Specification v0), on the existing
91 successful motion-planning episodes. No v1–v4 model/result, Visual/Recurrent
code, shared simulator, or original demonstration was edited. All new executable
code/configuration is Graph-team owned under `graph/subtasks/`; convenience CLIs
are under `scripts/subtasks/`. No new BC experiment was trained. Existing small
model-overfit regression tests were run as part of their unchanged test suites.

ManiSkill 3.0.1 / SAPIEN 3.0.3 installed source and actual CPU execution were
checked on this macOS Apple M4 Pro host. Simulation: `physx_cpu`; renderer:
`sapien_cpu` via MoltenVK. CUDA/vectorized execution was **not run**. The live
contact adapter supports one CPU environment. Repeat commands are in
[graph/subtasks/README.md](../graph/subtasks/README.md).

Installed source inspected:

- `mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py`
- `mani_skill/examples/motionplanning/base_motionplanner/utils.py`
- `mani_skill/agents/robots/panda/panda.py`
- `mani_skill/envs/tasks/tabletop/peg_insertion_side.py`

The official solver reaches a waypoint 5 cm above its grasp pose, descends,
closes the gripper, moves to pre-insertion, refines three times, then inserts.
It has no separate Transport phase. Existing `data['phase']` labels are geometry
heuristics, not oracle metadata, and are not used here. The converted H5 files
contain no original planner phase labels.

## Reference frames and numeric conditions

All poses are world `[x,y,z,qw,qx,qy,qz]`. Distances are metres and metric rotation
errors are radians. Config rotation thresholds are degrees. Relative translation
is `R_parent.T @ (p_child-p_parent)` and relative rotation is
`R_parent.T @ R_child`; angle uses the trace of SO(3), with quaternion normalization
and trace clipping. Quaternion sign reversal gives the same rotation.

Peg head is peg-local `[L,0,0]`, transformed with the full peg rotation.
Hole +X points inward; hole origin is inside the box, **not its front surface**.
This task's box half-depth equals peg half-length L. Entrance is hole-local
`x=-L`. Research insertion depth is `head_x + L`, negative before entrance.
Goal head is `[-L-0.01,0,0]`; goal peg center is `[-2L-0.01,0,0]` in the hole
frame. These formulas use per-episode randomized dimensions and hole offset.

Reach/grasp goals are computed with the installed official OBB helper and initial
TCP closing direction (including its sign), then stored in the peg-local frame.
At each state the reach target is recomposed with the **current peg pose**.
This captures the expert's tail offset and parallel-jaw orientation choice without
using future states. Its OBB center formula uses `min(0.025, half-height)`;
all current peg half-heights are <=0.025, so the grasp TCP is at the peg's center
height. Existing v2 `half-height-0.025` does not implement this clamp and would
introduce a height error. That existing code was left unchanged.

| Subtask | Start | Success | Implemented failure |
|---|---|---|---|
| Reach | Peg on table within 5 mm; fingers >=35 mm each; no grasp | Dynamic reach position <=20 mm, rotation <=15°, open, contact-valid/no grasp, never previously grasped, peg translation <=30 mm from initial | Subtask timeout; peg displaced before acquisition; workspace exit |
| Grasp | Valid Reach success with open gripper | Live `Panda.is_grasping` for 3 consecutive states; peg-in-TCP position drift <=3 mm and rotation drift <=3° across the window | Timeout; displaced before stable acquisition; subsequently lost grasp; workspace exit |
| Transport | Stable grasp, outside coarse target | Live grasp, center within 50 mm of pre-insertion target, rotation <=20°, head before entrance | Timeout; lost grasp for 3 consecutive states after stable acquisition; workspace exit |
| Align | Transport success with grasp | Live grasp; head AND center transverse norms <=3 mm; rotation <=3°; head depth within 10 mm of -10 mm waiting depth and <=0; nonnegative projected surface clearance; 2 consecutive qualifying states | Timeout; lost grasp; workspace exit |
| Insert | Stable Align success | Unmodified live `evaluate()['success']` | Timeout; lost grasp; workspace exit |

`SubtaskSpec.check_*` are boolean predicates; `compute_metrics` records evidence;
`failure_reasons` records reason strings. Each episode needs its own
`ConditionContext`, updated once at consecutive control-state indices. Independent
subtask timeout uses `context.start_t`; default budget is 250 control steps.
Workspace defaults are radial XY <=1 m, Z in [-0.02,0.6] m. These broad failure
bounds and contact-loss defaults are provisional: successful expert trajectories
cannot calibrate failure detection. Grasp success without context is false rather
than a proximity approximation. Align can be configured with `align_steps=1`.

Actual grasp GT uses `min_force=0.5 N, max_angle=85°`, the installed Panda default.
This is distinct from the reward's stricter 20° contact test. Left/right force
vectors and aggregate peg–box contact force are recorded after actual steps.
Contact caches after `set_state_dict` cannot be trusted. At s0 contact is explicitly
unknown (`contact_valid=False`); Reach success requires valid contact evidence.
The initialized table/open state can still satisfy Reach start. Grasp acquisition
and all subsequent checks use uninterrupted action replay; no contact proxy is
presented as measured GT.

Align surface clearance is `hole_radius - max(abs(all 8 peg corner YZ))` in the
hole frame. This is a conservative whole-peg projected footprint bound, stronger
than checking only the head, and suitable for straight translation through the
square aperture at the current orientation. It is a research auxiliary condition,
not the official success or a guarantee about future controlled motion.

Official `has_peg_inserted` checks head-local X >= -0.015 and |Y|,|Z| <= hole
radius. It does **not** require rotation, surface clearance, or grasp. Its docstring
alone is insufficient to reconstruct this test. Insert logs official success
separately from lateral errors, orientation, depth, grasp and forces.

**Undefined v0 failure rules:** IK-unreachable state, severe collision and stalled
insertion are not given operational thresholds. The code records forces and
workspace state, but does not invent thresholds or claim those failure detectors
are implemented. Defining these requires failed rollouts/force-duration evidence.
They are listed in the saved index's `undefined_failure_rules`.

## Threshold evidence and limitations

All thresholds live in `graph/subtasks/config.toml`. Train-only inspection is run
before held-out segmentation; no held-out threshold adjustment was made. The
73/9/9 original split IDs are preserved exactly. Position 20 mm / rotation 15°
Reach candidates were retained. Transport's v0 30–50 mm range uses the 50 mm
candidate, retaining 20°. Align defaults (3 mm / 3° / 10 mm waiting tolerance /
nonnegative corner clearance) are provisional research criteria motivated by
actual 3 mm aperture clearance; v0 has no exact Align threshold. Temporal defaults
are 3 Grasp states and 2 Align states, not a shared hold across all tasks.

Train-only evidence (`threshold_train.json`; endpoint statistics unless stated):

| Metric | Median | P95 | Maximum |
|---|---:|---:|---:|
| Reach position | 7.73 mm | 15.24 mm | 18.53 mm |
| Reach rotation | 0.349° | 0.911° | 1.004° |
| Grasp adjacent position drift (7012 measured transitions) | 0.0077 mm | 0.411 mm | 3.469 mm |
| Grasp adjacent rotation drift | 0.0066° | 0.304° | 2.923° |
| Transport position | 45.90 mm | 49.28 mm | 49.76 mm |
| Transport rotation | 5.023° | 11.142° | 18.599° |
| Align head lateral | 1.472 mm | 2.526 mm | 2.695 mm |
| Align center lateral | 1.339 mm | 2.319 mm | 2.841 mm |
| Align rotation | 0.173° | 0.418° | 0.603° |

Train Align endpoint surface clearance minimum is 0.347 mm, median 1.565 mm.
Grasp stability checks the entire 3-state window against its first state, not just
adjacent transitions. Adjacent statistics are therefore diagnostic context only.
These results show the retained candidates are achievable by the selected experts.
Endpoint distributions are conditioned on detection and do not independently prove
optimal thresholds or false-positive rates. Reach 15° is loose relative to experts;
it was not tightened to their near-perfect orientations. Failure bounds, temporal
counts, Align margin and timeout require further negative/independent-rollout
validation. No claim of universally finalized thresholds is made.

## Ordered segmentation

Action interval `[i,j)` contains states `s_i...s_j` and actions `a_i...a_(j-1)`.
Five records are emitted even when a record is unresolved; unresolved intervals
have null endpoints or invalid flags and cannot be read as training slices.

1. Detect the first gripper close action. Within the first open Reach-success run
   preceding that command, choose the state closest to dynamic Reach target.
   This avoids labeling acquired-peg states as Reach. The boundary estimates a
   waypoint; it can include a few initial descent/settling actions, because no
   original planner breakpoint is present.
2. Grasp begins there and ends at the first stable real grasp after close command.
   Some residual gripper settling may fall at the beginning of Transport.
3. Transport ends at first coarse pre-insertion condition after stable grasp.
   This is a **geometric milestone within a continuous planner move**, not a
   recovered independent expert Transport phase.
4. Align ends at the beginning of the **last stable success run before the first
   entrance crossing**. Earlier successful refinements that lose alignment are
   retained in Align. Missing strict alignment does not permit a valid Insert
   segment merely because official insertion later succeeds.
5. Insert ends at first official success after that valid Align boundary.
   Post-success tail actions remain in the source and appear as uncovered;
   they are not assigned artificially to another task.

No fixed timestep splits or confidence probabilities are used. `candidate_runs`,
`multiple_candidates`, `boundary_status=geometry_estimate`, `start_valid`,
`success_reached`, `failure_encountered`, and `status` expose the available evidence.
Multiple condition runs are distinct from unresolved ordering: a deterministic
last-stable-run rule can resolve an operational boundary while the true planner
phase remains unavailable. Transport/Align oracle phase boundaries remain
unidentified in **all** episodes. Accordingly only valid Reach is marked
`bc_eligible`; other segments are usable source-index candidates, not certified
independent BC datasets.

## Preservation, restoration and stored data

The converted replay has the known first-state bug. Initial state always comes
from the original motion-planning H5 via the existing verified source-ID mapping.
Each audit resets/reconfigures with source reset metadata, restores original s0,
resets controller, then executes the converted actions unchanged to sT. Later
snapshots are never used for live contact collection or boundary rendering.
The registered 100-step wrapper is replaced by a replay horizon T+1; step
termination is recorded through success, and replay continues through the source
rather than stopping early.

`index.json` keeps source paths/SHA256, original source paths/SHA256, audit
SHA256, config SHA256/version, split IDs and episode records. Each segment includes
source converted episode ID, `[start_t,end_t)`, success/status/reason,
`initial_state_ref` (original episode ID, seed/reset kwargs, controller reset,
prefix action range), and geometry goal metadata. Converted and original episode
IDs are different and are never interchanged. Reach goals store peg-local pose
and current starting target; pre-insertion center/head goals store hole-local
positions, hole pose and dimensions.

`segment_arrays` returns original simulator states, original observation/action
slices, and live T+1 geometry/contact rows. It repairs only source s0 using the
original verified reference, enforces episode bounds and T+1/T alignment, verifies
unchanged actions and checks saved peg/box poses and robot qpos against replay GT.
Full source trajectories remain intact; index-only storage avoids per-segment
trajectory duplication. Audit NPZ adds live contact evidence and terminal GT that
existing geometry-only BC tensors do not retain. Dataset load verifies provenance
by default. Simulator snapshot arrays retain velocities for future reconstruction,
but contacts/controller targets still require the stored prefix method.

## Measured 91-episode results

| Subtask | Detected | Condition reached | Start valid | Mean actions | Min / median / max | Unresolved | Multiple success runs |
|---|---:|---:|---:|---:|---|---:|---:|
| Reach | 91 | 91 | 91 | 43.20 | 25 / 42 / 72 | 0 | 0 |
| Grasp | 91 | 91 | 91 | 16.81 | 13 / 16 / 27 | 0 | 24 |
| Transport | 91 | 91 | 91 | 33.84 | 14 / 31 / 59 | 0 | 0 |
| Align | 91 | 91 | 91 | 29.71 | 10 / 29 / 49 | 0 | 33 |
| Insert | 91 | 91 | 91 | 19.96 | 14 / 20 / 28 | 0 | 0 |

Each subtask has 73 train / 9 validation / 9 test detections with valid starts and
success endpoints. No missing/overlapping/reversed operational boundaries occurred.
This reflects measured selected successful experts, not a requirement imposed on
the algorithm: synthetic missing-contact, missing-alignment, missing-close and
invalid-replay tests produce unresolved records rather than force completion.

Full episodes: 105–219 actions, median 149, mean 153.38. Total source actions:
13,958. Covered before first official success: 13,060 (93.57%). The remaining
898 are post-success tails, explicitly reported per episode. There are 455
validated segment slices; source action equality and state/geometry matching pass.
Independent segmentation recomputation is deterministic for every episode.
Condition-run exits across all episodes are Reach 91, Grasp 28, Transport 91,
Align 125, Insert 0. Reach exits on descent/grasp and Transport exits on insertion
are expected; Align re-entries and Grasp stability-window interruptions are
diagnostic events, not automatically new phases or proof of dropped objects.
All 91 uninterrupted CPU replays reproduce official success; initial and maximum
observation error are 0.0. These observations cover all T+1 states.

Eight representative episodes (0,1,2,13,29,38,54,90) have timeline, hole-local
geometry and real boundary-render contact sheets. All eight rendered prefix
replays have zero observation error and matching grasp GT at boundary states.
Rendered contact sheets for all eight and timelines for 0,1,13,29,54 were visually
inspected. They show open approach, grasp acquisition, coarse hole approach,
aligned waiting and actual insertion. This is representative physical review,
not human inspection of every state in all 91 episodes.

For episode 0 the boundaries are 0→47→62→106→147→166 (source T=178).
Train refinement often re-enters alignment; the late stable run avoids declaring
the first fleeting alignment completion final. The original success condition
can become true while research orientation/clearance later worsen; metrics keep
that distinction visible.

Artifacts are under ignored `outputs/subtasks/`: `reports/validation.json`,
`reports/coverage.md`, `reports/threshold_train.json`, `segments/index.json`,
`figures/all_durations.png`, `figures/timeline_ID.png`, `figures/geometry_ID.png`,
`figures/replay_boundaries_ID.png`, `figures/boundary_replay.json`.

Annotated replay videos are generated with:

```bash
.venv/bin/python -m graph.subtasks.visualize --video \
  --output outputs/subtasks/videos --episodes 0 1 2 13 29 38 54 90 --fps 10
```

Each MP4 contains one rendered simulator frame per state and overlays the
geometric phase, original state index, gripper command, live grasp contact,
official success, head lateral error, and insertion depth. The actions are still
replayed at every timestep; `--stride N` only drops frames from the encoded video.
Episode 0 was smoke-tested at 179 frames with maximum replay observation error
0.0. The current host rendered these videos with `sapien_cpu`/MoltenVK; CUDA
rendering was not executed.

## Tests and next-stage judgment

Executed 24 subtask tests with `SUBTASK_INTEGRATION=1`, including a fresh temporary
live replay of episode 0, all 91 source-alignment/condition checks and boundary
render checks. Executed existing regression suites: v1 12 tests; v2–v4 29 tests;
shared simulator 5 tests. Test logs are in `outputs/subtasks/reports/`. CPU only;
CUDA/vectorized variants were not executed. Tests use synthetic counterexamples
for frame invariance, quaternion sign, contact-vs-proximity, temporal stability,
dropped grasp, missing prerequisites, episode bounds, deterministic segmentation,
and save/load. No negative real demonstrations were available for failure-rule
calibration.

1. **Can geometry distinguish five subtasks?** Five ordered physical milestones
   can be detected here. It does not establish five distinct official planner
   phases, especially Transport/Align.
2. **Are these BC datasets suitable?** Reach is ready as a first baseline source.
   Remaining segments are physically consistent candidates, with valid GT starts
   and goals, but need independent prefix-start rollout validation and phase/reset
   protocol before treating them as standalone BC datasets. Refinement/settling
   can cross the estimated semantic boundaries.
3. **Most clear/stable?** Reach has one condition run in every episode and no grasp
   contact dependence at its approach goal after stepped GT verification. Official
   Insert success is also unambiguous as an endpoint, but not a full quality
   guarantee for insertion pose. Align has the greatest boundary re-entry issue.
4. **Use Reach now?** Yes: 91 episodes / 3,931 original action samples (train 3,173,
   validation 398, test 360), all open/no-acquisition with valid endpoints; unchanged
   split, target geometry and T+1 states are available. This supports an initial
   BC baseline, not a claim of trained rollout success or randomized coverage.
   All 3,931 Reach actions command gripper +1, no Reach state reports real grasp,
   and maximum peg translation from its initialized pose is 3.56 mm.
5. **Next independent generation work:** prefix-based subtask reset wrapper with
   per-task horizon/termination, independent start/goal checks after replay,
   controller/contact readiness, randomized initial-state generation and feasibility
   checks, subtask-specific expert planner breakpoints/labels, negative/failed
   rollout diagnostics for collision/reachability/stall thresholds, then fresh
   demonstrations and Reach rollout evaluation. These are outside this change.

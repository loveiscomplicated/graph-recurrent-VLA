# Task 10-D: Geometry-conditioned Reach BC

Opt-in data experiment. The inherited Plan A model, geometry, controller,
ActionCodec and Train96 normalizers stay unchanged. All Task10D code and output
are separate from the original comparison, cache and checkpoints.

`collect.py` applies a real Peg actor translation after reset, checks unchanged
Robot/Box/Hole/camera state, initializes a new ScriptedExpert and executes its
open-gripper reach/descend sequence. RGB-D/Robot76 have T+1 observations and
Action7 has T labels. Object GT and offsets stay in evaluation metadata.

`pilot_cache.py` checks full original/optimized cache equivalence. `cache.py`
uses the original pre-GNN writer or the gated `parallel_cache.py` sink with
exactly matching tensor/pointer/provenance arrays, and adopts pilot NPZ files only
after raw/source/geometry identity checks. Vision state is reset per camera and
episode. The cache stores no trainable representation or expert target.

`renderer_probe.py` compares complete CPU-physics captures with CPU versus CUDA
sensor readback. `worker_benchmark.py` selects collector concurrency from measured
throughput. A single CUDA vision worker consumes completed, independently locked
episodes while CPU collectors execute separate subprocesses. The optional
two-process geometry path overlaps independent current-frame CPU computations
with causal CUDA vision; output is consumed in original timestep order. Its
isolated full-episode gate includes startup/IPC and exact all-array equality.

`freeze.py` creates 11,487 paired A/B updates. A's replacement original episode
has exactly the same length as B's perturbed episode. Thus each update has the
same valid temporal mask. Every frame's sampling count is saved. Perturbed lengths
with no original match are retained as raw/cache artifacts and excluded explicitly
before training. Existing Full96 scheduler update boundaries and normalizers are
reused. `resident.py` packs the same graph ordering on CUDA; exact batch/update and
save/resume smoke checks precede fresh seed42 A/B initialization.

`evaluate.py` creates graphs from current live RGB-D. Reach requires ordered
approach and pre-close targets, pose thresholds from ScriptedExpert, open fingers,
no early close and physical validity. It is not Grasp/Insertion. A/B evaluation
uses a shared episode RNG namespace for paired graph sampling. Counterfactual
branches reuse the same causal prefix and previous hidden, physically move Peg,
refresh expert targets, and require exact baseline repetition for analysis.

The runtime output is `/workspace/task10d_geometry_reach` on the existing RTX3090
server. Dedicated supervisor programs are `task10d_data`, `task10d_experiment`
and `task10d_monitor`. Existing services are never stopped. The monitor reuses the
existing Plan A event Detector and emits only error/stall/completion events.
Automatic Codex callback is not verified or claimed. Publishing and Git commits
are not part of this experiment.

Run the contracts in their matching target fixture environments:

```bash
python -B -m graph.geometry_reach10d.contracts --root /workspace/task10d_geometry_reach
```

The wrapper runs the unchanged legacy contracts from the original Full96 source
directory, then the Reach contracts from the new source directory. The legacy
50-epoch fixture is not the Reach sampler's update clock. The new Reach contract
independently checks all 11,487 paired updates and 525,090 timestep exposures.
Do not combine both fixture domains under the new Reach `config.json`.

`dataset_ab_manifest.json` freezes the research Python source hashes before
formal initialization. Report-only review helpers live outside this package;
they cannot change the frozen model, data, training or evaluation selection.
The complete experiment report belongs in
`docs/visual-graph-integration/10d_geometry_conditioned_reach_bc.md`.

The publication checkout fixes the report merger's duplicate `condition`
keyword for already-tagged validation CSV rows. The executed frozen source and
the report-only compatibility adapter are preserved in the experiment artifacts;
this later reporting fix does not change the recorded model/data/training code.

Read `pipeline_state.json`, `verification.json`, frozen manifests and the report
for the actually executed scope. Module presence alone is not evidence of full
data collection, completed training or Reach success.

The completed run is `60ba447bf1b541e793a3d54ce68aa719`: 200 accepted
training/development captures, 9,122 cached graphs, and 11,487 updates / 525,090
timestep exposures for each condition. Both selected checkpoints are step6,000.
Final Reach is A0/18 and B0/18. B has smaller position/translation-response
errors but worse orientation error and remains worse than zero-response on the
primary intervention metric. This run does not establish adequate conditional
Reach control or the impossibility of BC. See the full report for scope,
sampling-weight confounds, backends and verification.

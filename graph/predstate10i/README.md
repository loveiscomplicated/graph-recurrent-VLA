# Task10-I: explicit F1 prediction to a fresh GRU

`Policy` uses the unchanged F1 state head and `assemble_gt38`; its only controller
input is the resulting38D sensor estimate. All F1 parameters are frozen/eval.
The new controller reuses `recurrent.models.RNNPolicy` with38→128, GRU144 and
144→128→7; no previous recurrent weights are imported.

Use the prepared cu130 environment on Vast.ai58752018. Work is isolated under
`/workspace/task10i_predicted_state_gru_20261102`. Original F1/G/H source,
checkpoints and data remain read-only. `prepare` reconstructs missing host mmap
arrays with the actual Task10G preparation API, then checks exact Schedule,
ActionCodec and RobotNormalizer identity. Existing Graph caches are reused.
Train/Validation feature files contain predicted State38, expert Action7, mask,
episode/parent/branch/timestep and boundary; manifest pins F1, Graph, schema,
normalizer and source/runtime. The BC dataset API exposes no Oracle state.

Normalization is fitted on all Train predicted frames, including unlabelled
disturbance observations. Population moments use float64, stored float32;
native position/shape std floor1mm; quaternion std floor0.01; Clean constants
mean[1,0], std[1,1]. No log, clipping, GT-based fit or auxiliary representation.

Before full launch, execute actual data/CUDA `preflight --stage contracts`,
`evaluate --stage smoke`, pipeline failure fixture and regression tests. The
gate refuses launch unless required evidence is PASS. Smoke performs five BC
updates, validation, save/readback, exact next-update resume, fresh re-init,
real RGB-D inference and valid original H1/H2 anchors on a Development parent.
Empty current geometry means the original mandatory three robot nodes remain;
it does not invent a new zero-node GNN contract.

The copied Task10H engine retains exact anchors/hidden behavior. Full Expert
also observes R/J with independent hidden states, allowing each policy's later
handoff prefix to be checked against the same causal reference without adding
evaluation episodes. GT state is accessed only by Expert and scoring after the
student action/hidden forward, and is never passed into the policy.

`pipeline` runs cache verification,11487 updates,24 Development rollouts,
selection freeze,48 independent Expert/autonomous episodes,72 handoff episodes,
paired analysis/report, and artifact verification in sequential subprocesses.
One rollout owns vision/simulator memory at a time. Child errors stop the
pipeline and create FAILED; no objective/budget changes or automatic retry.
Each saved checkpoint has optimizer/scheduler/RNG/cursor/detached episode hidden
and input identities. Explicit `restore` is tested, but resume is not automatic.

Every stage has an atomic status and separate stdout/stderr log, while a runner
heartbeat records liveness without invoking Codex. A lock prevents duplicate
pipelines. PIPELINE_COMPLETE is written only after actual completion audits.
The full Markdown report is copied to the remote output root's
`docs/visual-graph-integration/` with adjusted links. No upload or agent callback
is created. tmux does not guarantee survival of instance stop/deletion/recycle.

After the original Development renderer failure, `resume_evaluation.py` is an
operational continuation entrypoint copied **outside** the frozen source tree,
under `operations/resume_eval_20261102_v1/runner.py`. It preserves the original
checkpoint/source identities and completed23 episodes, runs the missing
Development episode and each Final episode in a separate fresh process, and
calls the unchanged selection/report/verification functions. It never reruns
training or automatically retries failed jobs. Its own manifest, failure/lock
fixtures and two-process actual RGB-D equivalence evidence are recorded beside
the operation. The renderer's root cause remains unconfirmed; process isolation
is an operational mitigation, not a scientific condition change.

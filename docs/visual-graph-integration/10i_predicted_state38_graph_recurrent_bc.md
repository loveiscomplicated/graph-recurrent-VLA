# Task 10-I — Predicted State38 → New GRU → Action7

All preregistered 120 independent Final episodes completed. Primary comparison is S vs R; J is a practical joint-BC baseline.

S freezes the original F1 GNN, structured readout and 398→256→128→18 state head. Only a fresh 38→128 Projection, GRU144 and 144→128→7 head were trained. No GT object state or GT sidecar is a policy input.

Training used seed42, 11,487 optimizer updates, batch2/sequence32, original Task10G schedule and Method B loss. S checkpoint selected at step 9000 using the original six Development seeds and rank rule, before Final access.

| Condition | Grasp | Lift | Alignment | Official insertion |
|---|---:|---:|---:|---:|
| E | 12/12 | 12/12 | 12/12 | 12/12 |
| S0 | 1/12 | 0/12 | 0/12 | 0/12 |
| S1 | 12/12 | 12/12 | 7/12 | 0/12 |
| S2 | 12/12 | 12/12 | 12/12 | 6/12 |
| R0 | 1/12 | 1/12 | 0/12 | 0/12 |
| R1 | 12/12 | 12/12 | 2/12 | 0/12 |
| R2 | 12/12 | 12/12 | 12/12 | 2/12 |
| J0 | 3/12 | 3/12 | 1/12 | 0/12 |
| J1 | 12/12 | 12/12 | 3/12 | 0/12 |
| J2 | 12/12 | 12/12 | 12/12 | 2/12 |

H1/H2 grasp and lift include Expert prefix achievements. H2 is the original align→insert boundary; it is not a near-contact anchor. Anchor-unreachable parents remain in the denominator.

## Paired interpretation

Mode 0: S gained 0 / lost 0 insertion successes against R; nominal exact paired p=1.
Mode 1: S gained 0 / lost 0 insertion successes against R; nominal exact paired p=1.
Mode 2: S gained 5 / lost 1 insertion successes against R; nominal exact paired p=0.21875.

No autonomous insertion advantage over R was observed on these parents.
S succeeded on some original H2 handoffs but no H1 handoffs. This supports some axial approach/insertion ability after Expert alignment, while alignment from the post-lift state remains unresolved.

First10 H2 lateral drift, in mm (parent mean):

- S: 6.2791 mm across 12 valid anchors.
- R: 13.3113 mm across 12 valid anchors.
- J: 14.7317 mm across 12 valid anchors.

F1 prediction errors at the actual H1/H2 handoff observation (parent means, mm):

| Anchor | Peg relative position | Hole relative position | Tip–hole vector |
|---|---:|---:|---:|
| S1 | 9.9540 | 63.1714 | 60.2950 |
| S2 | 7.5280 | 20.1880 | 19.1525 |

A difference on these 12 parents and one selected checkpoint does not establish general superiority. Similar low performance does not separate F1 prediction error, information loss through the bottleneck, BC fit, and history distribution effects. J also changes GNN training, so S–J is not a pure representation comparison. Prediction and control-error co-occurrence is descriptive, not a causal attribution.

## Reproducibility and scope

Normalizer was fitted solely on Train predicted State38, before Projection; quaternion std floor0.01, position/shape std floor1mm, Clean constants use mean[1,0]/std[1,1]. No additional normalizer experiments, head fine-tuning, GNN training, new expert collection, probe fitting or GT correction were performed.

Controller parameter counts and input dimensions: `{'S': {'total': 336975, 'controller': 142823, 'total_includes_buffers': True, 'input_dimension': 38}, 'R': {'total': 245447, 'controller': 188903, 'total_includes_buffers': True, 'input_dimension': 398}, 'J': {'total': 245447, 'controller': 188903, 'total_includes_buffers': True, 'input_dimension': 398}}`. Frozen F1 parameters/buffers remained identical before/after BC. Original source/checkpoints/data identities are rechecked in completion verification.

Executed on Vast.ai58752018 RTX3090: CPU PhysX + CUDA renderer/vision/policy, control20Hz, physics100Hz, horizon250 including prefix. GPU PhysX and macOS CUDA were not executed. Reported wall inference latency is distinct from simulator dt; real-time20Hz is not claimed.

One rollout process runs at a time. tmux survives SSH disconnect but does not guarantee survival of instance stop/deletion/recycle. No Codex watcher/callback, Git commit or external publishing was used.

## Evidence and media

[Preflight](../../outputs/task10i/predicted_state38_bc_20261102/final/preflight_verification.json) · [Selection](../../outputs/task10i/predicted_state38_bc_20261102/final/development_selection.json) · [Paired results](../../outputs/task10i/predicted_state38_bc_20261102/final/paired_policy_comparison.csv) · [State prediction error](../../outputs/task10i/predicted_state38_bc_20261102/final/state38_prediction_error.csv) · [Handoff drift](../../outputs/task10i/predicted_state38_bc_20261102/final/handoff_lateral_drift.csv) · [Anchors](../../outputs/task10i/predicted_state38_bc_20261102/final/handoff_anchor_diagnostics.csv) · [Latency](../../outputs/task10i/predicted_state38_bc_20261102/final/latency_metrics.csv)

![H1_lateral_trajectory](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/H1_lateral_trajectory.png)

![H2_lateral_trajectory](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/H2_lateral_trajectory.png)

![first10_actions](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/first10_actions.png)

![first10_lateral_drift](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/first10_lateral_drift.png)

![functional_success](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/functional_success.png)

![prediction_vs_control](../../outputs/task10i/predicted_state38_bc_20261102/final/figures/prediction_vs_control.png)

E representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/E/rollout.mp4`, remote)
S0 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/S0/rollout.mp4`, remote)
S1 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/S1/rollout.mp4`, remote)
S2 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/S2/rollout.mp4`, remote)
R0 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/R0/rollout.mp4`, remote)
R1 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/R1/rollout.mp4`, remote)
R2 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/R2/rollout.mp4`, remote)
J0 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/J0/rollout.mp4`, remote)
J1 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/J1/rollout.mp4`, remote)
J2 representative video (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/J2/rollout.mp4`, remote)
S first H2 failure (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/S2/rollout.mp4`, remote)
R first H2 failure (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/R2/rollout.mp4`, remote)
J first H2 failure (`/workspace/task10i_predicted_state_gru_20261102/rollouts/final/seed118200/J2/rollout.mp4`, remote)


Local readback: pipeline completed 2026-11-02 23:23:08 KST; 120/120 Final episodes, verification and completion audit PASS. Media readback checked video endpoints, not full video decoding. Original report and downloaded evidence preserved under outputs/task10i/predicted_state38_bc_20261102/final/. MP4s remain on Vast.ai.

# Task 08-D — Method B single-episode overfit and A/B/C comparison

[정적 HTML 보고서](../../experiments/090a4bd46e75455db893543e35c2da2b/method_b_single_episode_overfit.html) · [검증 결과](../../experiments/090a4bd46e75455db893543e35c2da2b/verification.json)

B training UUID `4fb8f19b0b3f4624a26bbca4df59a529`; B rollout UUID `f4142ed975e54d6c9b58a78e7711055b`; comparison UUID `090a4bd46e75455db893543e35c2da2b`. Vast.ai 57640710, RTX 3090 CUDA training and CPU PhysX rollout.

## 해석 요약

B는 Train96 Motion6 z-score MSE와 Native Gripper 연속 MSE만으로 단일 Seed 30003을 Offline Overfit했고, 같은 초기 상태의 실제 자율 Rollout에서 137 Step에 삽입했다. A는 파지와 4cm Lift 후 t=79 개방 명령과 함께 파지를 잃었고, B/C는 파지를 유지하며 삽입했다. 이 한 사례는 Motion6 정규화 기반 학습의 기여 가능성을 지지하며, BCE와 이진 Gripper 디코딩이 이 Seed의 성공에 필수라는 주장을 약하게 한다. 일반적인 성공률이나 각 변경의 독립적인 인과 효과는 아직 검증하지 않았다.

## Contract and fixed inputs

Seed 30003: 137 actions, 138 observations, success true. HDF5 SHA256 `ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122`. Initial model `386585e094c894e9f3cecc8d57c52a1e3c31f895ffeab8b88e524b0b5c4a45c5`; Robot76 normalizer `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`; Train96 Motion6 codec `9a475b4a51779a913db563b84495ff6f2189f09013c3e2193863702b352b782c`.
B loss: (6 × z-score Motion6 MSE + continuous native Gripper MSE) / 7. Decoder inverses Motion6 z-score and leaves Gripper continuous. Task08 float32 population std + 1e-6; no floor. A/C checkpoints and traces were read only.
AdamW, lr 3e-4, weight decay 1e-5, B1×T32, grad clip 1, seed 42, CosineAnnealingLR T_max=20000. Hidden carries and detaches across five contiguous chunks. All methods used the same 20,000 updates.
Preflight reset: `PASS_PHYSICAL_WITH_RENDER_QUANTIZATION`. Actual B rollout independently checked reset identity in its trace. Full A/B/C training was concurrently active; no other process was stopped.
B selected checkpoint update 20000, SHA256 `72473e581c6c0dfdfa566a26fcf7d73619f310470a6e983a05a69db06cfd8899`. Selection preceded rollout. Reloaded Expert sequence predictions matched exactly: `True`. Batch order and cached tensor hashes match A/C.

## Offline and actual closed-loop

| Method | Action7 MSE | Motion6 MSE | Controller translation mean mm | Rotation mean rad | Gripper target mean mm | Gripper sign | Offline first close | Autonomous first close | First release | Grasp | Lift 4cm | Insert | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| A | 1.13101e-07 | 1.08095e-07 | 0.04473 | 3.67767e-05 | 0.00341 | 100.00% | 49 | 51 | 79 | True | True | False | 250 |
| B | 2.77461e-07 | 5.69002e-08 | 0.01695 | 3.40219e-05 | 0.01003 | 100.00% | 49 | 49 | None | True | True | True | 137 |
| C | 9.13197e-08 | 1.0654e-07 | 0.02174 | 5.02405e-05 | 0.00000 | 100.00% | 49 | 49 | None | True | True | True | 137 |

Optimization losses differ in scale and objective and are not ranked across methods.
B overfit: `OFFLINE_FIT_SUFFICIENT`; initial Action7 error reduction 100.000%; 15k→20k improvement 69.11%; trend `SUFFICIENT`.
Expert offline first close t=49; B predicted t=49; expert transitions [49]; B transitions [49].
B rollout: max Peg lift 0.0812 m, final lift 0.0603 m, first grasp loss None, first success step 137.

B actual t=79: native gripper -1.002770, Controller target -0.01000 m, grasp True, Peg lift 0.0702 m, finger contact force norms [28.85, 28.78] N. B Native Gripper exceeded ±1 on 62/137 autonomous frames and 68/137 offline frames; the installed Controller clipped/scaled it. There was no binary policy threshold.

## B milestones

| Update | Optimization loss | Action7 MSE | Motion6 MSE | Controller translation mm | Rotation rad | Gripper sign |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.00241192 | 9.35954e-05 | 4.31895e-05 | 0.47067 | 0.000877652 | 100.00% |
| 5000 | 0.000256013 | 2.42109e-05 | 7.13492e-06 | 0.37140 | 0.00039164 | 100.00% |
| 10000 | 7.63922e-06 | 9.32142e-07 | 2.50543e-07 | 0.04039 | 8.38518e-05 | 100.00% |
| 15000 | 1.72993e-05 | 8.98175e-07 | 4.22878e-07 | 0.03866 | 0.000117714 | 100.00% |
| 20000 | 1.58642e-06 | 2.77461e-07 | 5.69002e-08 | 0.01695 | 3.40219e-05 | 100.00% |

## Error by Expert phase

| Phase | A Action7 MSE | B Action7 MSE | C Action7 MSE |
|---|---:|---:|---:|
| reach | 3.69451e-08 | 2.92688e-07 | 1.16359e-07 |
| descend | 2.58564e-08 | 6.84461e-08 | 1.25295e-08 |
| close | 1.32965e-07 | 5.33898e-08 | 1.08029e-07 |
| lift | 2.57664e-08 | 3.68213e-08 | 2.40011e-08 |
| align | 1.96648e-07 | 4.09787e-07 | 1.16448e-07 |
| insert | 1.35565e-07 | 2.66528e-07 | 5.56659e-08 |

## Interpretation

B also inserted. Motion6 z-score training is a plausible contributor on this seed; BCE/binary gripper is not required for this particular success.
All three closed-loop policies began from the same verified reset, then visited different states. Equal timestep actions are not same-state causal effects. A single episode does not establish a success rate or generalization.

## Q1–Q11

- Q1 **CONFIRMED:** B offline fit True; Action7 MSE 2.77461e-07; Motion6 MSE 5.69002e-08.
- Q2 **CONFIRMED:** Native and controller errors are in the table; all methods ran 20,000 updates.
- Q3 **CONFIRMED:** Expert/B offline first close 49/49; transitions [49]/[49].
- Q4 **CONFIRMED:** B grasp True, lift 4cm True, insert True; actual CPU PhysX rollout.
- Q5 **CONFIRMED:** B first release None, first grasp loss None; A first release 79.
- Q6 **SUPPORTED:** A↔B differs in Motion6 training normalization under continuous Gripper; one seed and divergent autonomous histories limit causal scope.
- Q7 **SUPPORTED:** B↔C differs in BCE training plus binary Gripper decode; this is a bundled change and affects shared model gradients.
- Q8 **SUPPORTED:** Offline errors alone do not fully predict closed-loop outcomes on this episode.
- Q9 **CONFIRMED:** Stored action, GRU hidden, contact and geometry timelines show each autonomous history; cross-policy equal-timestep comparisons are descriptive.
- Q10 **SUPPORTED:** Prioritize multi-episode A↔B validation of Motion6 normalization. Then test the extra B↔C BCE/binary combination for robustness; this seed does not require it for success.
- Q11 **NOT_TESTED:** Held-out generalization, distribution shift robustness, graph ablation, and statistical success rates.

## Artifacts and limits

B training `/workspace/task08d/experiments/4fb8f19b0b3f4624a26bbca4df59a529`; B rollout `/workspace/task08d/experiments/f4142ed975e54d6c9b58a78e7711055b`; after-action MP4 `/workspace/task08d/experiments/f4142ed975e54d6c9b58a78e7711055b/episodes/train_seed_30003/after_action_rgb.mp4`; t70–100 timeline `B_t70_100_timeline.csv` and JSON. A/C source artifacts remain in Task08-B.
Shadow inference on A observations: PASS; A t79 gripper 0.1511092483997345, B shadow t79 gripper -0.5276437997817993. This uses A observations with B own hidden and applies no action. CUDA PhysX and real-time deadlines: NOT_TESTED. No dependency install, Git commit, or external upload.

## Figures and video

[B after-action RGB MP4](../../experiments/f4142ed975e54d6c9b58a78e7711055b/episodes/train_seed_30003/after_action_rgb.mp4)
![B_hidden_action](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/B_hidden_action.png)
![B_optimization_loss](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/B_optimization_loss.png)
![B_relative_geometry](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/B_relative_geometry.png)
![action7_mse](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/action7_mse.png)
![controller_gripper_70_100](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/controller_gripper_70_100.png)
![frames_049](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/frames_049.png)
![frames_079](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/frames_079.png)
![frames_100](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/frames_100.png)
![frames_136](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/frames_136.png)
![grasp_lift](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/grasp_lift.png)
![gripper_70_100](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/gripper_70_100.png)
![gripper_accuracy](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/gripper_accuracy.png)
![gripper_offline](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/gripper_offline.png)
![motion6_mse](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/motion6_mse.png)
![motion_rotation](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/motion_rotation.png)
![motion_translation](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/motion_translation.png)
![rotation_rad](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/rotation_rad.png)
![stage_error](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/stage_error.png)
![tcp_3d](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/tcp_3d.png)
![translation_mm](../../experiments/090a4bd46e75455db893543e35c2da2b/figures/translation_mm.png)

# Frozen appearance verifier confirmation

Independent experiment source/output root; production stays paused. Reuses the
existing frozen Qwen3.5-9B checkpoint, BF16 SDPA configuration, Processor, DINO
wrapper/selection routines, and shared ExperimentLogger with W&B disabled.

- Prespecified 20 parents outside the four-parent native512 development set.
- Native512 restoration fields exist for 15 parents; five are recorded missing
  before inference with no replacement. Existing F0/F1 Main/Wrist yield60 images.
- This is held out from appearance-prompt development, not globally unseen data:
  earlier other-prompt studies used these train parents.
- Statically restore saved physics state and camera calibration, render without
  stepping. Fifteen constructors and15 explicit seeded resets initialize geometry.
  No new action, expert, rollout, training or dataset modification.
- Existing DINO Tiny weights: one load,120 fixed-prompt forwards. Preserve every
  threshold0.1 proposal. No result-driven candidate/parent sampling.
- Same full/marked/crop images in both frozen original and appearance prompts.
  One BF16 Qwen load, one greedy generation per candidate per prompt. Hard cap8000
  requests; exceeding it stops preparation instead of sampling after outputs.
- Empty cache verified at the first forward of each request. All labels frozen
  before Actor GT evaluation. Compare original selection, raw Top1 and both
  same-query verifier gates; report rejection/approval and final localization.
- Parent-cluster bootstrap, camera/time strata, real denominators and misses.
- No SAM2, tracking, production integration, commit or publishing.

Run `run.sh` only in a fresh remote output directory. Never rerun a failed model
load/inference. Two preparation corrections (parent-count assertion; renaming
`requests.py` to avoid library shadowing) occurred before capture or weight loads;
the prior sources/logs are preserved under `.execution/preflight_revisions/`.

After receiving the immutable indexed artifacts locally, `verify_local.py`
independently checks hashes, pixel-equal crops, Actor roles, candidate/selected
IoUs and GT-free selection replay, then copies the full report into docs.

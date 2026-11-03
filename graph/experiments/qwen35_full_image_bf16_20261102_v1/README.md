# Task 10-I Stage 4A-2: BF16 precision control

Independent, one-shot experiment using the existing Qwen3.5-9B revision/cache
on Vast.ai 58576054 / RTX 3090. This directory is copied to the new remote
output root `.execution/`; no production runner is invoked.

The fixed prompt, 24 train parents, 99 RGB requests, parser, GT evaluation and
16 SAM2 selections reproduce the saved NF4 protocol. The sole inference change
is loading original weights in BF16 without BitsAndBytes quantization. This is
a matched precision comparison on a previously evaluated set, not a new blind
generalization test. `actual_input_snapshot.json` records the current one-image
template rather than inheriting the older verifier contract.

Sequence: prepare → infer (one load, no retry) → evaluate (GT barrier) →
compare_dino (saved results only) → finish → supplement. All predictions are
frozen before GT or DINO contents are read. Strict JSON remains the accepted
SAM2 contract; extracted JSON coordinates are diagnostic only.

All logging uses the shared ExperimentLogger with W&B disabled. No publishing,
training, simulator, long tracking, production changes, commits, or new models.
The original Stage4A-2 PAUSED_SCOPE_REVIEW and stop stub remain in effect.

# Task 10-I Stage 4A-2: matched Qwen3.5-9B precision comparison

User-authorized comparison on Vast.ai 58576054 / RTX 3090, with the existing
checkpoint revision and environment. Three fresh processes each load original
weights once: BF16 without quantization, BF16 vision/merger plus NF4 language
linears, and BF16 vision/merger plus LLM.int8 language linears (threshold 6).
Custom exclusion lists preserve the loader's default lm_head exclusion.

The latest detailed prompt, parser, 99 one-image requests, processor resolution,
generation settings, EXACT_GT support and 16 SAM2 camera-frame selections remain
fixed. The previous all-module NF4 detailed-prompt run is a saved reference;
it is not rerun. This is a retest on known parents, not an independent test set.

Source is copied into the new output root `.execution/`. Run prepare.py on CPU,
then run.py in a dedicated tmux session. All three inference processes terminate
and predictions are SHA-frozen before GT evaluation starts. Model load/inference
errors stop that variant without retry; remaining authorized variants still run.
SAM2 initializes only valid predicted boxes and never commits or propagates.
After run.py finishes, finalize.py verifies hashes/readback, computes paired
parent-cluster comparisons, checks the preserved DINO matches, and writes the
full report and artifact index. Copy the full report to
`docs/visual-graph-integration/10i_stage4a2_qwen35_precision_comparison.md`,
rebasing its artifact links to the local output mirror.

Preparation uses meta tensors only, without loading model weights or running a
GPU forward. Its first CPU preflight log is retained after correcting the
installed model class's `_from_config` construction API before any weight load.
Checkpoint/header accounting also uses only metadata, including unused `mtp.*`
keys in the original checkpoint; the existing conditional-generation path is
unchanged.

Use the shared ExperimentLogger, W&B disabled. Keep production and Stage4A-2
PAUSED_SCOPE_REVIEW/stop stubs unchanged. No simulator, training, new weights,
DINO inference, publishing, commits, offloading, or process termination.

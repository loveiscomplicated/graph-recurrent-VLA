# Task 10-I Stage4A-2: native capture resolution × prompt pilot

Independent BF16 Qwen3.5-9B pilot on Vast.ai 58576054. Production and the
PAUSED_SCOPE_REVIEW runner are unchanged. No DINO, policy rollout, training,
weight change, download, commit, or publishing.

- Deterministic four-parent sample from the previously evaluated 24 train parents.
- Same t0/t80 Main/Wrist, 16 paired camera-frames, native capture 128 and 512.
- All four conditions use processor 512 (256 vision tokens), same greedy generation
  and wrapper-aware parser. P0 is unchanged; P1 is one frozen 245-word alternative.
- One BF16 model load; exactly 64 Qwen requests, no extra warmup/retry requests.
- Temporary SAM2 on all 16 target camera-frames per condition (64 targets), using
  only valid predicted boxes. Invalid/null predictions remain failures/abstentions.
- Capture reconstructs seeded geometry and recorded visible robot/object state,
  adds 512 cameras by shallow dataclass replacement, retaining pose/FOV/mount,
  and renders all four sensors without a physics step. This is static rendering,
  not a dynamic replay of hidden PhysX solver state.
- Reconstruction necessarily reads saved privileged state; the inference program
  reads only RGB and prompts. Fresh segmentation sidecars are opened for metrics
  only after all predictions and SAM2 masks have been frozen.
- Compare within fresh paired capture. Historical 128/256-processor scores are
  not the baseline. Existing GT/results have already been seen: this is an
  exploratory paired pilot, not a blinded or fresh generalization evaluation.
- Hold out remaining 20 parents from this pilot. No automatic expansion or prompt
  optimization after results. A subsequent confirmation requires its own frozen
  setting and execution budget.

Remote execution order: `prepare.py`, `capture.py`, `preflight.py`, `infer.py`,
`evaluate.py`, `finalize.py`, each once. CPU capture/preflight are separate from
GPU inference. Failed GPU execution is not restarted. Technical source fixes made
before any weight load are preserved in the source identities.

The shared `utils.experiment_logger.ExperimentLogger` records local disabled-W&B
runs. No separate logger or publishing client is added.

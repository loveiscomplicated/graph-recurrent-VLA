# Graph prototype v4: fixed history stacking

Graph-team-only code. Flat and Surface use the unchanged v2 encoders and full goal state. No recurrent layers, learned hidden state, new graph, RL or DAgger. Prior checkpoints/results remain intact.

Read [technical report](../../../docs/graph_prototype_v4.md) and [measured comparison](../../../outputs/graph/v4/comparison/results.md).

`--length K` counts past timesteps, excluding current state. The TOML `[history]` section supplies defaults; `--history current|action|state|both` overrides its flags. Past order is oldest to newest. State history includes current state last. Missing states repeat the episode's first state; missing actions are zero. A K-dimensional validity mask accompanies any history configuration. Current Only has no added mask and exactly reproduces v2.

The runtime cache contains detached 159-dimensional state representations only for a fixed evaluation encoder. Training encodes every selected historical state with one shared encoder and retains all gradients. Window deduplication is confined to each batch, with autograd indexing back into the original windows.

```bash
.venv/bin/python -m unittest graph.prototype.v4.tests.test_history -v
.venv/bin/python -m graph.prototype.v4.overfit --output outputs/graph/v4/overfit_rerun.json
.venv/bin/python -m graph.prototype.v4.import_baselines
.venv/bin/python -m graph.prototype.v4.run_matrix --phase train
.venv/bin/python -m graph.prototype.v4.run_matrix --phase evaluate
.venv/bin/python -m graph.prototype.v4.run_matrix --phase ablate
.venv/bin/python -m graph.prototype.v4.latency
.venv/bin/python -m graph.prototype.v4.distribution
.venv/bin/python -m graph.prototype.v4.compare
.venv/bin/python -m graph.prototype.v4.figures
.venv/bin/python -m graph.prototype.v4.report
```

The commands refuse to overwrite checkpoints/evaluation files. `import_baselines` is a one-time operation using the original completed v2 runs; it verifies exactly equal validation/test predictions. For fresh reproduction, train all eight configurations into a new root rather than importing into the existing root:

```bash
RUN=outputs/graph/v4_reproduce
for encoder in flat surface; do
  for history in current action state both; do
    .venv/bin/python -m graph.prototype.v4.train --model "$encoder" --history "$history" --length 4 --epochs 30 --seed 0 --output "$RUN/training/${encoder}_${history}"
  done
done
.venv/bin/python -m graph.prototype.v4.run_matrix --phase evaluate --root "$RUN"
.venv/bin/python -m graph.prototype.v4.run_matrix --phase ablate --root "$RUN"
.venv/bin/python -m graph.prototype.v4.latency --root "$RUN"
.venv/bin/python -m graph.prototype.v4.distribution --root "$RUN"
.venv/bin/python -m graph.prototype.v4.compare --root "$RUN"
```

Single condition or smoke (not a main result):

```bash
.venv/bin/python -m graph.prototype.v4.train --model flat --history both --epochs 1 --max-batches 2 --output outputs/graph/v4/smoke_new
.venv/bin/python -m graph.prototype.v4.evaluate --checkpoint outputs/graph/v4/training/flat_both/best.pt --output outputs/graph/v4/evaluation/flat_both_single_new.json --rollouts --episode-ids 29 --stages alignment
```

Evaluation uses all held-out test samples offline, nine held-out episodes in each full/acquired/transport/alignment start, and seeds 2000–2029 for 30 fresh full starts. Prefix indices, stage thresholds, original reset snapshots and 250 learner-step time limit come from v2. Prefix simulation fills the cache from live states and applied expert commands. Every learner step commits the exact clipped command passed to `env.step`, after stepping. Trace JSON records raw/applied command, input action history/mask, full pre/post states, contact forces and actual TCP displacement separately.

CPU is the measured path. CUDA is unavailable on this host. Surface State/Both were resumed on MPS by user request after successful real batch32/K4 parallel forward/backward and AdamW restoration. MPS index_add requires deterministic warn_only; CPU/MPS numerical results need not be bit identical. Simulation and final evaluation/inference remain CPU. CPU cache latency excludes GT extraction and physics. Main experiment optimizer/schedule/loss/selection are unchanged: AdamW .001 / 1e-5, cosine 30 epochs, batch 32, seed 0, validation-minimum checkpoint, train-only normalizers.

## Active MPS tmux jobs

```bash
tmux attach -t graph-v4-mps
# Windows: state, both, evaluate. Ctrl-b then n changes windows; Ctrl-b then d detaches.
```

CPU archives are `training/surface_{state,both}_cpu_interrupted`. Resume starts at saved best epoch11/14, preserving AdamW and cosine schedule; partially executed later CPU work is not migrated. New `last.pt` checkpoints also preserve shuffle/RNG/optimizer steps. Trainer supports `--device mps --resume <checkpoint> --mps-memory-fraction .4` into a fresh output directory. `tmux_worker.sh state|both` runs the actual jobs. All backend transitions are recorded in resume/training metadata.

## Reproduce the measured CPU → MPS transition in a fresh root

The measured matrix uses completed CPU baselines/Flat/Surface-Action and CPU-prefix → MPS continuation for Surface-State/Both. Keep the 30-epoch scheduler budget even when stopping the CPU prefix. Strict CPU determinism and MPS warn-only are different execution conditions; bitwise reproduction is not promised.

```bash
RUN=outputs/graph/v4_reproduce_mixed
for history in current action state both; do
  .venv/bin/python -m graph.prototype.v4.train --model flat --history "$history" --epochs 30 --seed 0 --device cpu --output "$RUN/training/flat_${history}"
done
for history in current action; do
  .venv/bin/python -m graph.prototype.v4.train --model surface --history "$history" --epochs 30 --seed 0 --device cpu --output "$RUN/training/surface_${history}"
done
.venv/bin/python -m graph.prototype.v4.train --model surface --history state --epochs 30 --seed 0 --device cpu --stop-after-epoch 11 --output "$RUN/training/surface_state_cpu_prefix"
.venv/bin/python -m graph.prototype.v4.train --model surface --history both --epochs 30 --seed 0 --device cpu --stop-after-epoch 14 --output "$RUN/training/surface_both_cpu_prefix"
# Run each continuation in its own tmux window if parallel execution is desired.
.venv/bin/python -m graph.prototype.v4.train --model surface --history state --epochs 30 --seed 0 --device mps --resume "$RUN/training/surface_state_cpu_prefix/best.pt" --output "$RUN/training/surface_state"
.venv/bin/python -m graph.prototype.v4.train --model surface --history both --epochs 30 --seed 0 --device mps --resume "$RUN/training/surface_both_cpu_prefix/best.pt" --output "$RUN/training/surface_both"
.venv/bin/python -m graph.prototype.v4.run_matrix --phase evaluate --root "$RUN"
.venv/bin/python -m graph.prototype.v4.run_matrix --phase ablate --root "$RUN"
.venv/bin/python -m graph.prototype.v4.latency --root "$RUN"
.venv/bin/python -m graph.prototype.v4.distribution --root "$RUN"
.venv/bin/python -m graph.prototype.v4.compare --root "$RUN"
.venv/bin/python -m graph.prototype.v4.report --root "$RUN" --output docs/graph_prototype_v4_reproduce_mixed.md
```

The original measured Current checkpoints were imported from v2 after exact prediction/rollout checks; the fresh-root commands above retrain Current as permitted by the experiment protocol. The original interrupted CPU State/Both each reached 14 epochs; State's selected best was epoch11, so its later unselected CPU work was discarded when resuming. All original CPU artifacts remain in the archives.

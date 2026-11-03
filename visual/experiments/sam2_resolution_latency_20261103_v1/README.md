# Task 10-I: SAM2 native-resolution latency

Controlled timing at native 128, 512 and 768, retaining the existing four train
parents, both cameras, camera pose/FOV and SAM2.1 Small checkpoint. No production
runner, training, detector, publishing, or model configuration changes.

`benchmark.py capture ROOT` statically restores recorded states and renders all
three sizes together. It verifies seed-matched geometry, TCP, unchanged physical
state and scaled camera intrinsics. Static t0/t80: 16 scenes. Tracking t0..20:
eight 21-frame sequences. Rendering, PNG/JPEG writing and model loading are not
included in inference timings.

Two boxes are generated once in normalized coordinates from the native768 actor
segmentation and scaled identically for every resolution. An absent actor gets a
full-frame box; both prompts always execute. This is an oracle-prompt latency
benchmark, not a detector, segmentation quality or generalization evaluation.

`benchmark.py static ROOT` measures preloaded RGB -> image encoding -> two
sequential prompts -> original-resolution NumPy masks, six balanced order
permutations per scene. Three warmup rounds per size are excluded.

`benchmark.py tracking ROOT` separately measures JPEG loading/initialization and
steady propagation -> two original-resolution boolean CPU masks, three balanced
cyclic orders per sequence. One complete warmup sequence per size and frame0 of
each measured sequence are excluded from steady statistics. Video is preloaded
on CPU; state stays on GPU. This is the repository's video predictor API, not a
measurement of simulator capture plus online control.

Both use FP32 weights, BF16 autocast, TF32 allowed, batch1, internal1024, four CPU
threads and CUDA synchronization. Shared ExperimentLogger records local runs with
W&B disabled. Raw times, order positions, memory, input hashes, exact repeat-mask
checks and runtime identities are preserved. Independent statistical units remain
four parents/eight sequences; repeated frames are not independent scenes.

After both timed processes exit, `postflight.py REMOTE_ROOT` rehashes every
input, original raw/GT source and checkpoint, checks completion and empty CUDA
process inventory, and saves installed SAM2 source hashes. `analyze.py LOCAL_ROOT
REPORT_PATH` independently rebuilds summaries from JSON/CSV, checks balanced
order and repeat hashes, computes exploratory parent-cluster intervals, renders
a static chart and writes the full report to `docs/visual-graph-integration/`.

The first static setup stopped before model loading because installed metadata
uses distribution name `SAM-2`. Its source, protocol and log are retained; the
only pre-inference correction was the metadata lookup name. No timed inference
was rerun or dropped.

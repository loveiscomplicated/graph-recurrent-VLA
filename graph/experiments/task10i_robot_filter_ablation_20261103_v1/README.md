# Task 10-I: robot-filter ablation on saved high-resolution masks

This is a **separate diagnostic experiment**, not a change to the shared simulator,
Graph implementation, original suspended Task 10-I process, or published policy.
It uses saved actual SAM2 masks, recorded RGB-D/calibration and robot geometry.
No API, SAM2 inference, training, rollout, package installation or publishing occurs.

## Design

The original `cases.json` was an initial eight-scene availability probe and is
not the final sample. The user requested a larger comparison stratified by
previously successful, failed, and ordinary filter cases. `screen_protocol.json`
fixes explicit thresholds and sampling before the 512 screen. `screen.py` replays
the existing filter only on already-recorded 512 inputs and scores relevant mask
pixels with saved simulator actor labels. `select_cases.py` additionally joins the
prior Stage4A per-frame 128 filter audit by exact parent/timestep and camera.
Labels are evaluation-only: they never enter the filter or Graph builder.

The predeclared screen selects up to 15 cases per stratum, cycling across parents
before taking a second case from a parent. A previously documented exact-frame
near-finger failure has priority in the failure stratum. The selected-case file
contains per-camera counts, the prior audit rows, source hashes, and the selection
rule. These are **case-finding strata**, not overall frequency estimates or a
statistical definition of production success. One input may both benefit from
robot removal and lose valid object pixels; failure takes precedence. `success`
means useful robot-pixel removal under the fixed rule, not successful perception
or policy execution. The limited prior audit coverage and 512 truth-based screen
make this an exploratory, selected sample. Parent/time relationships remain.

Compare the same selected scenarios at native 768 and 1024, using their own
stored actual masks (not upscaled 512 masks). The 512 labels are used only for
scenario selection. At evaluation sizes, freeze all modes' selected Graphs,
records, clean and post-gate masks, gate results, latency rows, and SHA-256
receipts before opening actor GT to score them. The exact input and mask hashes
are saved per scenario. Failed/time-limited/OOM modes remain visible and are not
silently dropped or converted into successful results.

| Variant | Change in this experiment |
|---|---|
| `full` | Existing confirmed robot + uncertain band removal, plus overlap removal. |
| `confirmed_only` | Remove confirmed robot and overlap; retain uncertain band. Current implementation still calculates the full robot filter, so its measured latency is not an optimized implementation of this policy. |
| `off` | Skip robot filter entirely; retain overlap removal, wrist gate, Shape-First, 3mm dedup, final 128-point budget, edges, and validation. |

The methods intentionally may select different source pixels, gate decisions,
and Graph edges. This is a policy ablation, not an exact-output optimization.
All use the same experimental Stage 9 `Candidate` implementation for remaining
work. Full-filter smoke was checked against a previously saved Stage 9 Graph,
selected sources and gate, and the two alternative policies were checked on
synthetic masks to confirm that overlap removal remains enabled.

## What the measurements mean

- One process at a time, torch intra-op 4, OMP/OpenBLAS/MKL 1. Each mode receives
  one untimed warmup and three timed iterations per case, with mode order rotated.
  Scene order is deterministically interleaved and the two size orders alternate.
- Time starts at CPU sensor arrays, mask arrays and **precomputed** robot surfaces
  entering `CurrentGeometry.build`; it ends at the CPU Graph, selected records and
  gate. It includes internal transfers and synchronization. Loading, decompression,
  separate FK, mask inference, action inference, GT scoring and hashing are out.
- Pixel metrics use actor IDs and valid depth. Report raw mask and clean-mask
  correct/robot/other/background counts separately from final **representative
  node** identity, attached-source identity, and wrist gate decisions.
- Finger-near means <20mm from either recorded finger-pad centre, not contact-force
  GT. `correct_near_finger` is restricted to correctly labelled valid pixels in the
  supplied mask, not every GT-visible pixel. Zero support is `N/A`.
- Visible coverage uses correct final representative points and same-resolution
  observed GT depth points within 4mm, Peg using both cameras and Box using Main.
  It is camera-pixel weighted and does not establish full physical surface coverage.
- All cases are selected and a few share parents across strata. The per-group
  counts and paired changes do not estimate real-world or data-population rates.
  Three timed iterations cannot justify p95 or a real-time guarantee.

The complete experiment report is written to `docs/visual-graph-integration/`.

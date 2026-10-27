# Legacy FPS private release

Implementation commit: `afa5392b5f8fe01877894c366f0e3452c8c9cd49` on `codex/task09c-legacy-fps-release`.
The original run UUIDs, team, execution Git commit/dirty flag and checkpoint bytes
are preserved. `publication_commit` links the later implementation separately.
Git commits are local; there was no Git remote push or HF Space deployment.

## Standalone original Markdown

The original local `docs/visual-graph-integration/09c_legacy_fps_extension.md` was
published byte for byte as a separate private Dataset file. [View the original
Markdown at the fixed revision](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/blob/6558a391ed9cf84c51f0b74186a6b4fb372b3e2f/integration/8c41299e9d3e4989bbf32ab1b894dc49/artifacts/reports/b569444dfc474e433911a5856d44c6c64fc7156f73e1c39fbebdc084037d5ca2/09c_legacy_fps_extension.md) or [download it](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/6558a391ed9cf84c51f0b74186a6b4fb372b3e2f/integration/8c41299e9d3e4989bbf32ab1b894dc49/artifacts/reports/b569444dfc474e433911a5856d44c6c64fc7156f73e1c39fbebdc084037d5ca2/09c_legacy_fps_extension.md).
It is 42,463 bytes with SHA-256 `b569444dfc474e433911a5856d44c6c64fc7156f73e1c39fbebdc084037d5ca2`.
Its original links point to the historical sibling artifact tree and may not
resolve on Hugging Face; use the selected ZIP below to browse linked evidence.
[Publication and remote readback receipt](standalone_publication_receipt.json).

## Download

Both repositories were verified **private** before and after publishing. These
links require access to the Graph-Recurrent-VLA organization:

- [Model checkpoints at revision `98f9118b3ea1`](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/98f9118b3ea114c5b555c1d7d2151fa3ac434f30/integration/b220c3fdced649adab3edee0c7d5378b)
- [Dataset files at revision `fbaac6691194`](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/fbaac6691194048043932bffb86cc96678f80bcf/integration/8c41299e9d3e4989bbf32ab1b894dc49)
- [Download the selected report and source ZIP](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/fbaac6691194048043932bffb86cc96678f80bcf/integration/8c41299e9d3e4989bbf32ab1b894dc49/artifacts/reports/3d0e12d6e7f02632ee61589a9e814e43fdcf6df3742549fa6685f00150adf4aa/report-3d0e12d6e7f02632ee61589a9e814e43fdcf6df3742549fa6685f00150adf4aa.zip)

ZIP: **57,146,393 bytes**, SHA-256
`3d0e12d6e7f02632ee61589a9e814e43fdcf6df3742549fa6685f00150adf4aa`.

| Destination | Contents | Unique uploaded files | Logical bytes |
| --- | --- | ---: | ---: |
| Model | 36 encoders + 244 probes, config/split/hash/provenance metadata | 259 | 17,049,705 |
| Dataset | Selected report ZIP + config/metrics/remote manifest | 4 | 57,151,007 |
| Total | Explicit selection only | 263 | 74,200,712 |

There are 280 logical checkpoint paths. Identical bytes share immutable remote
objects; this explains the smaller unique-file count. The ZIP has
562 members (559 selected files plus bundle/config/metrics metadata).
[File-by-file paths, sizes, hashes and ownership](publication_inventory.json)
and [publication receipts and remote verification](publication_receipts.json)
are committed separately from the binaries.

Unzip into a new directory. Open
`task09c_legacy_fps/experiments/8c41299e9d3e4989bbf32ab1b894dc49/legacy_fps_extension.html`
or `repository/docs/visual-graph-integration/09c_legacy_fps_extension.md`.
The bundle README includes fixed-revision Model download code;
`release/model_downloads.json` restores all 280 original checkpoint paths.

Extract `pinned_sources.tar.gz`. Its `pinned_source/` contains the 817 frozen
common files, `historical_extension/` preserves the original 18 extension files,
and `implementation/` contains the port and hash manifests. Use an existing
Python environment with the repository dependencies; no installation was performed
for this release. From `implementation/`, run:

```bash
python -m legacy_fps.run_pinned --source-root ../pinned_source \
  unittest legacy_fps.tests graph.persistent_geometry.tests.test_contracts legacy_fps.test_release -v
```

See [implementation usage](../../legacy_fps/README.md) for local preparation and
machine-local path settings. Recorded `/workspace/...` paths in historical configs
are provenance. `release/path_relocations.json` explains the download layout.
Full RGB-D/masks, per-frame tensors, map histories, frozen feature arrays and
redundant transfer archives are excluded. Only representative t0 tensor evidence
is included. Excluded report links lead to `omitted.html`. The minimal download
supports report inspection and checkpoint/contract replay; full experiment
recomputation needs the separately retained original captures and frame artifacts.

## Verification and limits

- Existing 54 contracts plus 4 release isolation contracts: **58/58 PASS**,
  including replay from extracted sources.
- Source guard: **840 files PASS**, including stale-encoder rejection and
  rejection of imported modules outside the validated tree.
- All **36 encoder checkpoints** strict-load; **2,232** Graph forward readouts
  and **108** permutation cases agree within stated CPU/CUDA tolerances.
- All **244 probe decoders** hash-match, load and reproduce saved predictions.
- **861** real-frame camera XYZ/source-pixel rows are exact. The original World
  transform helper differs from saved Linux output by at most
  `2.9802322387695312e-08 m` on this Mac (atol `1e-7`, rtol `1e-6`);
  cross-host bitwise World-coordinate identity is not claimed.
- **155 PNG / 10 MP4 / 184 video frames** decode; **173** extracted document
  links resolve. ZIP CRC and every selected member's SHA-256 pass.
- Remote revisions and all **263** unique file hashes/sizes pass. The entire
  remote ZIP was downloaded and its SHA-256 matched the local verified archive.
- **1,657** original selected/source files remained unchanged at readback.
  Known available credentials were checked without logging their values.

Shared infrastructure tests: **41/42 PASS**. The existing
`test_reject_running_run_symlink_and_outside_organization` expects another namespace
to be rejected, while current HEAD accepts syntactically valid repository IDs.
The same assertion failure was reproduced in the original worktree. No shared
publisher/logger code was changed. The actual publication used the requested
Graph-Recurrent-VLA repositories with `--private` and explicit file selections.

Historical global protection remains **PARTIAL**. Later pinned-source
compatibility and the new protection window remain **PASS**; the later check does
not replace the historical failure. This is the **100-update pilot** with no
independent Final Test. No retraining, new simulator capture, CUDA/MPS experiment
on this Mac, policy training or closed-loop evaluation was executed for release.

[Detailed local verification](release_verification.json) ·
[Portable source audit](source_port_audit.json) ·
[Original integration manifest](historical_integration_manifest.json).

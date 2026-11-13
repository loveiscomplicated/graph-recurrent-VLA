# Data collection and GPU observation verification

2026-11-05 KST. User-authorized pivot; branch `develop` starts at local `main`
`0386934eec665e866954d7d3f29bbd1e2bc02525`. The original legacy checkout and its
uncommitted reports were retained. Work is in the attached develop worktree.
The implementation was committed as `4afbaee`. No remote push was included.

## Graph branch integration

The reusable collection and GPU observation features were imported into `graph`
from `develop` commits `4afbaee` and `3d3a88b`. All 13 imported implementation,
dependency and test files match `develop` byte for byte. Existing graph/corner
code and unrelated local artifacts were preserved; feature guides were adapted
to the graph branch's existing documentation.

On 2026-11-05 KST, the combined collection, output-device, shared simulator and
corner-detector suite ran **66 tests: 56 passed, 10 CUDA tests skipped**. A real
CPU PhysX/CPU renderer collection accepted seed 0 at 512×512, saved two zero-action
steps and three raw observations, and completed frame readback. The saved initial
Main RGB-D then passed directly into the MPS corner detector, producing 14 finite
world-coordinate candidates on `mps:0`. This checks data/device compatibility,
not corner accuracy or policy success. CUDA rendering/output was not executed
on this Mac.

Current graph integration artifacts:
[test log](../outputs/data_collection_integration/tests.log),
[collection manifest](../outputs/data_collection_integration/episode_smoke/manifest.json),
[collection-to-corner smoke result](../outputs/data_collection_integration/corner_smoke.json).
The remaining sections record the original develop extraction. Linked metadata
copies retain their source-run values; full raw source artifacts remain in the
preserved develop worktree.

Usage: [data collection](data_collection.md) and [GPU observations](gpu_observations.md).

## Native 512 expert dataset on Linux CUDA

2026-11-06 KST, user-requested collection on Vast.ai **59288396**, RTX 3090.
The effective config is [visible_peg.yaml](../data_collection/configs/visible_peg.yaml).
The new run ID is `c555bed7462d4fc09e55ae3bb04421a1`. Original checkout commit:
`64eaad4a3a365bfe5cbbb002a29059604ca1ccbc`, dirty; the selected executed source
snapshot and file hashes are retained, rather than claiming commit-only reproduction.

| Selection or verification | Result |
| --- | --- |
| Consecutive resets within 30 disjoint seed ranges | 2,916 |
| Initial Main ends-and-grasp visibility rejected, no expert action | 2,003 |
| Visibility passed; expert attempted | 913 |
| Official expert success / unsuccessful | 900 / 13 (98.58% success among expert attempts) |
| Dataset selection | Only the 900 successful episodes; no split assignment |
| Raw observations | 92,825 frames; T+1 observations and T actions per episode |
| Episode archives | 30 TAR files, 58,857,973,760 bytes before metadata |
| Raw camera arrays, every frame | Main/Wrist RGB `[1,512,512,3]` uint8; depth `[1,512,512,1]` int16 mm |
| Initial visibility | Independently recomputed from all 900 saved initial frames |
| Per-frame schema/action alignment | All frames passed; every source array also read back at save time |

The policy is the standalone extraction of `recurrent/expert.py`'s GT-guided
servo control law, with native action shape and normalized bounds handling.
Physics ran on CPU (`physx_cpu`), cameras rendered on CUDA (`sapien_cuda`), and
the raw observation output was CUDA. Native camera CUDA residency before the
output wrapper was checked on reset and step. The actual 512 camera override
was then verified on every saved dataset frame. CUDA PhysX was not executed.
The original recurrent implementation and historical datasets were not modified.

Runtime: Python 3.12.15, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0,
Torch 2.14.1+cu132, NumPy 2.5.4, Pinocchio package `pin` 4.1.0.
Three-episode pilot: 17 resets, 3 visibility passes, all 3 expert successes.
Four-worker pilot: 8 successful episodes, complete schema/visibility verification
and archive construction. The 900-episode collection then ran as a supervisor job.

Local collection/dataset/publishing contracts passed **31 tests**; Linux ran the
same 31 tests successfully. Linux visibility/native-CUDA coverage also passed
in the **38-test** run. Logger's **13 tests** passed in an isolated process;
Mac simulator/visibility/output-device checks ran **17 tests**, 15 passed and
2 CUDA skips. An initial combined logger run encountered an existing global
import-mock interaction with imageio's lazy plugin module; isolation passed.
The final checks did not require changing the logger or the unrelated team code.

Evidence copied locally: [dataset manifest](../outputs/visible_peg_dataset/dataset_manifest.json),
[source identity](../outputs/visible_peg_dataset/collection_source_identity.json),
[local contracts](../outputs/visible_peg_dataset/local_contract_tests.log),
[Linux contracts](../outputs/visible_peg_dataset/linux_contract_tests.log),
[Linux CUDA coverage](../outputs/visible_peg_dataset/linux_gpu_tests.log).
Publication completed using the shared `tools.publish_experiment --include
trajectory metrics --private` workflow. The uploaded revision is
`94cb85e8009f20b5b47063e8295f6c69040549c1` in
[`Graph-Recurrent-VLA/graph-recurrent-rollouts`](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/94cb85e8009f20b5b47063e8295f6c69040549c1/integration/c555bed7462d4fc09e55ae3bb04421a1).
All **35** remote files matched their expected sizes and LFS SHA-256 or downloaded
JSON SHA-256. This passed both the Linux workflow verification and a separate
local HF readback check. Full archive payloads were not downloaded again.
The existing repository remained private. Its README/catalog now index the
dataset; that documentation commit is `b0d357c2b73be4d4d3d8f1cb29a979b3cfdf3920`.
Publishing used huggingface_hub 2.2.0 and hf_xet 1.7.0, with automatic transfer
concurrency; no transfer retry or recollection was needed.

Receipts: [HF publication](../outputs/visible_peg_dataset/huggingface_uploads.json),
[Linux remote verification](../outputs/visible_peg_dataset/publication_verification.json),
[local remote verification](../outputs/visible_peg_dataset/local_publication_verification.json),
[catalog entry](../outputs/visible_peg_dataset/catalog_publication.json),
[final workflow status](../outputs/visible_peg_dataset/progress.json).
Vast.ai instance shutdown is checked separately in the
[shutdown receipt](../outputs/visible_peg_dataset/shutdown_verification.json).
At 2026-11-06 02:55 KST the API reported `actual_status=exited`,
`intended_status=stopped`, and `cur_state=next_state=stopped` for **59288396**.

## Scope and implementation

- `envs/maniskill/peg_insertion.py`: bring forward explicit render backend,
  observation output device and camera overrides. CPU physics continues to work;
  native CUDA RGB-D requires an explicit CUDA renderer on a supported host.
- `envs/maniskill/observation_device.py`: reuse the existing raw observation
  transfer wrapper. Already-resident tensors are reused, dtype/shape preserved;
  rewards, flags, info and action handling are unchanged.
- `data_collection/visibility.py`: extract the historical audit gate plus only
  its required pose/calibration and ray/convex-depth math. No repository Graph,
  tracker, perception, GNN, controller or legacy experiment imports remain.
- `data_collection/collect_episodes.py`: bounded consecutive-seed search, initial
  Main-camera selection, optional caller-supplied policy and raw trajectory capture.
  Accepted T0 is used directly; no second reset is performed before stepping.
- `data_collection/storage.py`: raw NPZ frames with readback, atomic file/episode
  exposure, attempt history and committed-only manifests. Object pickle is disabled.
- `examples/check_gpu_observations.py`: verify RGB-D is already CUDA before the
  wrapper on reset and step. Existing `examples/check_env.py` exposes new options.
- Documentation explains the new paths and examples. Direct geometry dependencies
  are declared in requirements. Historical models and training workflows were not
  imported; collection introduces no logger/publisher or remote upload workflow.

The default filter is the historical `requested_pass`: both end strips and the
expert grasp band must have depth confirmation within 1.5 mm; the peg's full box
projection must remain inside the Main image. `full_surface` explicitly selects
`strict_100_pass`. This does not guarantee later visibility, physical validity or
expert success. Historical physical/replay/diversity/split filters remain separate.

## Verification executed on this host

1. Historical saved input comparison: all result fields on all **320** observations
   exactly match the archived audit. Default pass count **105**, strict **102**.
   The extracted private depth helper was also checked after removing unused robot
   classification outputs. This is saved-data validation, not a new simulator run
   of those historical episodes.
2. **20 tests run: 17 passed, 3 CUDA tests skipped.** Includes six historical
   geometry fixtures, collection selection/termination/storage/failure contracts,
   raw output wrapper tests, and the main branch's five simulator checks.
3. Real CPU collection: seed **0**, Main/Wrist **512×512**, **2 zero-action steps**,
   **3 observations** saved/read back and one episode committed. Status COMPLETE.
   Backend: CPU PhysX (`physx_cpu`), CPU image readback renderer (`sapien_cpu`,
   macOS Vulkan/MoltenVK). Zero actions are a smoke check, not expert data.
4. Ruff unused-import checks and `git diff --check` passed. Installed Camera API
   and runnable reset/step examples were inspected/executed in the existing Python
   3.11 environment. ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0, Torch 2.14.1.

Commands, from this worktree (the existing original `.venv/bin/python` was used):

```bash
python -m unittest tests.test_visibility tests.test_data_collection \
  tests.test_observation_device tests.test_maniskill_env \
  tests.test_observation_device_native_gpu -v
python -m data_collection.collect_episodes \
  --output outputs/develop_collection_final_smoke \
  --num-episodes 1 --max-attempts 40 --steps 2 --zero-actions
```

[Verification record](../outputs/develop_extraction_verification/verification.json),
[local historical comparison script](../outputs/develop_extraction_verification/verify_historical_visibility.py),
[smoke manifest](../outputs/develop_collection_final_smoke/manifest.json),
[smoke configuration and package versions](../outputs/develop_collection_final_smoke/config.json).
The comparison script references the preserved original checkout and its ignored
historical input archive; these are verification inputs, not runtime dependencies.

## Remaining execution limits

CUDA rendering, CUDA output transfer and CUDA physics were **not executed on this
Mac**. Run the native GPU example and native GPU test module on a supported Linux
CUDA/Vulkan host in a separate process from CPU render tests. No training or policy
quality evaluation was performed. Raw disk storage and initial GT/depth visibility
selection explicitly materialize arrays on CPU; this does not claim zero-copy
collection end to end. Full expert trajectories require an explicitly supplied
policy; initial-only collection is the default. Automatic resume and exact simulator
state replay are not implemented. Existing outputs are refused, failed partial
folders are retained and excluded from approved dataset manifests.

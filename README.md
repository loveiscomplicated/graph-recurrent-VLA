# Graph Recurrent VLA: shared ManiSkill prototype environment

## Get started

Use Python 3.11 from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python examples/check_env.py
.venv/bin/python examples/inspect_gt_state.py
.venv/bin/python examples/render_episode.py
.venv/bin/python examples/random_rollout.py
```

If an existing `.venv` was created by `uv` without `pip`, install with
`uv pip install --python .venv/bin/python -r requirements.txt` instead.

`check_env.py` prints installed versions, actual simulation and render backends,
observation/action spaces, and reset/step results. `render_episode.py` saves RGB
frames to the ignored `outputs/render_test/` directory. Run the short CPU test
suite with `.venv/bin/python -m unittest tests.test_maniskill_env -v`.

On macOS, install Homebrew `molten-vk` if the simulator cannot create a Vulkan
device. The environment factory detects the usual Homebrew ICD location. For a
different installation, set `VK_ICD_FILENAMES` to its `MoltenVK_icd.json` path.
On this machine, CPU simulation worked with ManiSkill 3.0.1, SAPIEN 3.0.3,
Gymnasium 1.3.0, and PyTorch 2.14.1. CUDA was unavailable here; its path is
available in the factory but has not been run on this host.

## Start a team prototype

The repository shares only `PegInsertionSide-v1` simulator creation, observations,
rendering examples, and model-independent ground truth. The root `visual/`,
`graph/`, and `recurrent/` directories are team-owned. Each team chooses
its own files, features, data, models, training, evaluation, and configuration.

```python
from envs.maniskill import get_gt_state, make_peg_insertion_env

env = make_peg_insertion_env(seed=0, obs_mode="state")
try:
    observation, info = env.reset(seed=0)
    state = get_gt_state(env)
finally:
    env.close()
```

- Visual: use `obs_mode="rgb"` and compare the team's visual output with GT.
- Graph: read GT, then choose geometry, sampling, edges, and architecture inside `graph/`.
- Recurrent: collect GT over time, then choose features, normalization, and model inside `recurrent/`.

The teams can work independently. No cross-team interface or final architecture
is fixed during this feasibility stage. See [ManiSkill details](envs/maniskill/README.md)
for control modes, coordinate frames, cameras, geometry access, and known limits;
see [project context](docs/PROJECT_CONTEXT.md) for scope.

The Graph team's independent surface-geometry comparison prototype is documented
in [graph/prototype/README.md](graph/prototype/README.md), with its measured
results in [docs/graph_prototype.md](docs/graph_prototype.md).

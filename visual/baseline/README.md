# Visual end-to-end baselines (ACT, Diffusion Policy)

Image-to-action policies (LeRobot 0.7) trained on scripted-expert episodes, as
the end-to-end comparison for the integrated model. Inputs are main and wrist
RGB at 256 px plus the 17D robot state; output is the 7D `pd_ee_delta_pose`
action.

## Commands

Prerequisites: a Linux CUDA host with Vulkan, at least 64 GB RAM (training holds
all 256 px episodes in memory, about 36 GB for 900 episodes), and
`pip install -r visual/baseline/requirements.txt` in a fresh environment.
Run everything from the repository root.

```bash
bash visual/baseline/smoke_test.sh                     # 10 episodes, 200 steps: checks every stage
python -m data_collection.collect_dataset --config data_collection/configs/visible_peg_shared_camera.yaml
python -m visual.baseline.prepare_data --archives outputs/visible_peg_shared_camera --output data/visual_baseline/episodes_shared_camera
python -m visual.baseline.eval_seeds --output outputs/visual_baseline_shared/eval_seeds.json
python -m visual.baseline.eval_seeds --output outputs/visual_baseline_shared/train_seeds.json --num-seeds 50 --from-prepared-index data/visual_baseline/episodes_shared_camera/index.json
bash visual/baseline/run_experiment.sh visual/baseline/configs/act.yaml
bash visual/baseline/run_experiment.sh visual/baseline/configs/diffusion.yaml
```

Set `PYTHON=/path/to/env/bin/python` for the shell scripts if `python` is not
the baseline environment.

The team's Vessl workspaces have a max running time of 24 hours and stop
themselves after it. Note the start time, check that collection, training and
evaluation fit in the remaining window (100k steps took about 5 h for ACT and
9 h for Diffusion on the first run's GPU), and keep `outputs/` and `data/` on
persistent storage or copy them off before the limit. The collection config keeps `publish: false`; publish
through `tools.publish_experiment` only after the team agrees.

## What changed from the first baseline run

The first run (results on branch `feature/visual-baseline-results`,
`visual_baseline/FINDINGS.md`) scored 1-11/100 on new scenes. This version
differs in three ways, so its numbers are not directly comparable:

- Camera: the shared base camera from `envs.maniskill` (eye [0.65, -0.30, 0.65])
  instead of the ManiSkill default. Data must be recollected for it.
- Robot state: 17D with a 6D rotation (`state.py`) instead of robot15's
  quaternion, whose w >= 0 canonicalization flipped sign in top-down grasps.
  Prepared episodes from the first run are refused by `train.py`.
- Configs: the first run's YAML was not committed; `configs/act.yaml` and
  `configs/diffusion.yaml` rebuild it from FINDINGS.md and LeRobot defaults.

Evaluation reports success on new scenes (`eval_*.json`) and on training scenes
(`train_*.json`). The first run showed a large gap between them (ACT 30/50 on
training scenes vs 4/100 on new ones), so both are tracked. `probe_lateral.py`
and `collect_dart.py` are kept from the first run but have not been rerun.

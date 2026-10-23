"""Run in a fresh process, separately from CPU renderer tests."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from mani_skill.utils.structs.pose import Pose

from envs.maniskill import get_gt_state, make_peg_insertion_env
from recurrent.initial_states import restore


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class GPUEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        path = Path(cls.tmp.name) / "initial.pt"
        # CPU renderer selection must stay outside the GPU test process.
        subprocess.run(
            [sys.executable, "-m", "recurrent.initial_states", "--out", str(path)],
            check=True,
        )
        cls.fixtures = torch.load(path, weights_only=False)

    def test_restore_and_slot_pose(self):
        fixtures = self.fixtures
        for b in (1, 14, 16, 30):
            env = make_peg_insertion_env(
                device="cuda",
                num_envs=b,
                control_mode="pd_ee_delta_pose",
                render_backend="none",
            )
            try:
                restore(env, fixtures, list(range(2000, 2000 + b)))
                before = get_gt_state(env).hole_pose.clone()
                pose = env.unwrapped.box.pose.raw_pose.clone()
                pose[0, 0] += 0.02
                env.unwrapped.box.set_pose(Pose.create(pose))
                env.unwrapped.scene._gpu_apply_all()
                env.unwrapped.scene._gpu_fetch_all()
                after = get_gt_state(env).hole_pose
                torch.testing.assert_close(
                    after[0, 0] - before[0, 0],
                    torch.tensor(0.02, device=after.device),
                    atol=1e-5,
                    rtol=0,
                )
                torch.testing.assert_close(after[1:], before[1:], atol=1e-5, rtol=0)
                env.step(torch.zeros(b, 7, device=after.device))
                torch.testing.assert_close(
                    get_gt_state(env).hole_pose, after, atol=1e-5, rtol=0
                )
            finally:
                env.close()

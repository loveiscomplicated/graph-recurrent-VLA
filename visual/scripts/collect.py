"""Step 2: ManiSkill3 PegInsertionSide-v1 RGB-D 수집.

정책: ManiSkill 내장 motion-planning 해법의 waypoint(reach → grasp → align → insert)를
pd_ee_pose 컨트롤러 + 보간으로 따라가는 스크립트 (mplib 0.1.1은 이 환경에서 segfault, docs/known_issues.md).
(gripper 가림 / hole 삽입 가림이 실제로 발생하는 궤적을 얻기 위해 랜덤 대신 사용)

에피소드마다 `<out>/ep{seed:03d}.npz` 하나 (카메라별 key prefix `<cam>/`):
  <cam>/rgb          uint8   (T, H, W, 3)
  <cam>/depth        float32 (T, H, W)   meters, 카메라 z (CV 좌표계). 무효/원거리 = 0
  <cam>/seg          int16   (T, H, W)   per-actor/link id (segmentation_id_map.json 참고)
  <cam>/intrinsic    float32 (T, 3, 3)   intrinsic_cv
  <cam>/extrinsic    float32 (T, 4, 4)   world -> camera (CV: x right, y down, z forward)
  <cam>/cam2world    float32 (T, 4, 4)   camera(CV) -> world
  peg_pose, box_pose, goal_pose, tcp_pose  float32 (T, 7)  world, [x y z qw qx qy qz]
  peg_head_pose, box_hole_pose             float32 (T, 7)
  qpos (T, 9), action (T, 7) [xyz, euler XYZ (root frame), gripper] (frame 0 = 0), frame_idx (T,), timestamp (T,) seconds
  peg_id, box_id (seg id), success (T,) bool, seed, peg_half_sizes (3,) [length, radius, radius], box_hole_radii ()
Frame 0 = reset 직후 관측, 이후 env.step 마다 1 frame.

Usage:
  source env.sh
  python scripts/collect.py --seeds 0 1 2 --out data/step2 --res 512
  # CPU simulation/rendering, e.g. on macOS with MoltenVK configured:
  python scripts/collect.py --seeds 0 --out data/step2 --res 512 --cams base_camera \
      --sim-backend physx_cpu --render-backend cpu
"""
import argparse
import json
import os
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import qinverse, qmult

ENV_ID = "PegInsertionSide-v1"
CONTROL_MODE = "pd_ee_pose"  # 절대 TCP pose 목표 (root frame). mplib planner는 segfault → docs/known_issues.md


def sp(pose):
    """mani_skill batched Pose -> sapien.Pose (env 0)."""
    return sapien.Pose(np0(pose.p), np0(pose.q))


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


class ScriptedPolicy:
    """ManiSkill motion-planning 해법(peg_insertion_side.py)의 waypoint를 그대로 쓰되,
    mplib 대신 pd_ee_pose 컨트롤러 + 직선/slerp 보간으로 따라간다."""

    def __init__(self, env, max_dpos=0.01, max_drot=0.05, settle=5):
        self.env, self.u = env, env.unwrapped
        self.base = sp(self.u.agent.robot.pose)
        self.max_dpos, self.max_drot, self.settle = max_dpos, max_drot, settle
        self.grip = 1.0  # 1 open, -1 closed

    def _act(self, pose):
        rel = self.base.inv() * pose  # root frame
        # pd_ee_pose 절대 모드: 회전은 pytorch3d euler "XYZ" (intrinsic, R = Rx·Ry·Rz)
        # (transforms3d.euler 는 numpy 2 에서 copy=False 에러 → scipy 사용)
        w, x, y, z = rel.q
        eul = Rotation.from_quat([x, y, z, w]).as_euler("XYZ")
        a = np.concatenate([rel.p, eul, [self.grip]]).astype(np.float32)
        self.env.step(a[None])

    def move(self, target):
        cur = sp(self.u.agent.tcp.pose)
        dq = qmult(target.q, qinverse(cur.q))
        drot = 2 * np.arccos(np.clip(abs(dq[0]), 0, 1))
        n = max(1, int(np.ceil(max(np.linalg.norm(target.p - cur.p) / self.max_dpos, drot / self.max_drot))))
        for i in range(1, n + 1):
            t = i / n
            self._act(sapien.Pose(cur.p + t * (target.p - cur.p), slerp(cur.q, target.q, t)))
        for _ in range(self.settle):
            self._act(target)

    def gripper(self, value, steps=15):
        self.grip = value
        tgt = sp(self.u.agent.tcp.pose)
        for _ in range(steps):
            self._act(tgt)

    def run(self):
        u = self.u
        FINGER_LENGTH = 0.025
        obb = get_actor_obb(u.peg)
        approaching = np.array([0, 0, -1])
        target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        peg_init_pose = sp(u.peg.pose)
        info = compute_grasp_info_by_obb(obb, approaching=approaching,
                                         target_closing=target_closing, depth=FINGER_LENGTH)
        grasp_pose = u.agent.build_grasp_pose(approaching, info["closing"], info["center"])
        grasp_pose = grasp_pose * sapien.Pose([-max(0.05, u.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])
        # reach / grasp
        self.move(grasp_pose * sapien.Pose([0, 0, -0.05]))
        self.move(grasp_pose)
        self.gripper(-1.0)
        # lift a bit (pd_ee_pose 직선 이동 시 table 마찰 방지)
        self.move(sapien.Pose([0, 0, 0.05]) * sp(u.agent.tcp.pose))
        # align
        goal = sp(u.goal_pose)
        insert_pose = goal * peg_init_pose.inv() * grasp_pose
        offset = sapien.Pose([-0.01 - u.peg_half_sizes[0, 0].item(), 0, 0])
        pre_insert_pose = insert_pose * offset
        self.move(pre_insert_pose)
        for _ in range(3):
            delta = goal * offset * sp(u.peg.pose).inv()
            pre_insert_pose = delta * pre_insert_pose
            self.move(pre_insert_pose)
        # insert (pre_insert 기준 +x 로 밀어넣기; 해법과 동일하게 insert_pose*[0.05,0,0])
        self.move(pre_insert_pose * sapien.Pose([0.01 + u.peg_half_sizes[0, 0].item() + 0.05, 0, 0]))
        return 0


def solve(env, seed):
    env.reset(seed=seed)
    return ScriptedPolicy(env).run()


def np0(x):
    """batched torch tensor (num_envs=1) -> numpy of env 0."""
    return x[0].detach().cpu().numpy()


class Recorder(gym.Wrapper):
    """env.step / reset 마다 관측 + GT를 버퍼에 쌓는다."""

    def __init__(self, env, cams):
        super().__init__(env)
        self.cams = cams
        self.frames = []

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.frames = []
        self._record(obs, np.zeros(self.env.unwrapped.single_action_space.shape, np.float32), info)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._record(obs, np.asarray(action, np.float32).reshape(-1), info)
        return obs, rew, term, trunc, info

    def _record(self, obs, action, info):
        u = self.env.unwrapped
        f = {}
        for c in self.cams:
            sd, sp = obs["sensor_data"][c], obs["sensor_param"][c]
            # depth: int16 mm. 원거리(>32.767m)는 int16 overflow로 음수 → 무효(0) 처리
            d = np0(sd["depth"])[..., 0].astype(np.float32)
            d[d <= 0] = 0.0
            ext = np.eye(4, dtype=np.float32)
            ext[:3] = np0(sp["extrinsic_cv"])
            f[f"{c}/rgb"] = np0(sd["rgb"])
            f[f"{c}/depth"] = d / 1000.0
            f[f"{c}/seg"] = np0(sd["segmentation"])[..., 0].astype(np.int16)
            f[f"{c}/intrinsic"] = np0(sp["intrinsic_cv"])
            f[f"{c}/extrinsic"] = ext
            f[f"{c}/cam2world"] = np.linalg.inv(ext).astype(np.float32)
        f["peg_pose"] = np0(u.peg.pose.raw_pose)
        f["box_pose"] = np0(u.box.pose.raw_pose)
        f["goal_pose"] = np0(u.goal_pose.raw_pose)
        f["peg_head_pose"] = np0(u.peg_head_pose.raw_pose)
        f["box_hole_pose"] = np0(u.box_hole_pose.raw_pose)
        f["tcp_pose"] = np0(u.agent.tcp.pose.raw_pose)
        f["qpos"] = np0(u.agent.robot.get_qpos())
        f["action"] = action.reshape(-1)
        f["frame_idx"] = int(u.elapsed_steps[0]) if hasattr(u.elapsed_steps, "__len__") else int(u.elapsed_steps)
        f["timestamp"] = f["frame_idx"] * u.control_timestep
        succ = info.get("success", False) if isinstance(info, dict) else False
        f["success"] = bool(np.asarray(succ.cpu() if hasattr(succ, "cpu") else succ).reshape(-1)[0])
        self.frames.append(f)

    def stacked(self):
        return {k: np.stack([f[k] for f in self.frames]) for k in self.frames[0]}


def actor_ids(u):
    id_map = {int(k): v.name for k, v in u.segmentation_id_map.items()}
    peg = [k for k, n in id_map.items() if n.startswith("peg")]
    box = [k for k, n in id_map.items() if n.startswith("box_with_hole")]
    return id_map, peg[0], box[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="data/step2")
    ap.add_argument("--res", type=int, default=512, help="sensor camera width=height")
    ap.add_argument("--cams", nargs="+", default=["base_camera", "hand_camera"])
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--sim-backend", default="auto")
    ap.add_argument("--render-backend", default="gpu")
    args = ap.parse_args()
    # SAPIEN may change cwd on macOS while creating the Vulkan renderer.
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    env = gym.make(
        ENV_ID,
        obs_mode="rgb+depth+segmentation",
        control_mode=CONTROL_MODE,
        sensor_configs=dict(width=args.res, height=args.res),
        max_episode_steps=args.max_steps,
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    env = Recorder(env, args.cams)
    print(f"env make {time.time() - t0:.1f}s")

    summary = []
    for seed in args.seeds:
        t0 = time.time()
        res = solve(env, seed)
        data = env.stacked()
        u = env.unwrapped
        id_map, peg_id, box_id = actor_ids(u)
        out = os.path.join(args.out, f"ep{seed:03d}.npz")
        np.savez_compressed(out, **data, peg_id=peg_id, box_id=box_id, seed=seed,
                            peg_half_sizes=np0(u.peg_half_sizes), box_hole_radii=np0(u.box_hole_radii),
                            env_id=ENV_ID, control_mode=CONTROL_MODE)
        with open(os.path.join(args.out, "segmentation_id_map.json"), "w") as fp:
            json.dump(id_map, fp, indent=1)
        planner_ok = not (isinstance(res, int) and res == -1)
        s = dict(seed=seed, file=out, frames=len(data["frame_idx"]), planner_ok=planner_ok,
                 success_final=bool(data["success"][-1]), success_any=bool(data["success"].any()),
                 first_success_frame=int(np.argmax(data["success"])) if data["success"].any() else None,
                 peg_id=peg_id, box_id=box_id, seconds=round(time.time() - t0, 1),
                 mb=round(os.path.getsize(out) / 1e6, 1))
        print(s)
        summary.append(s)
    env.close()
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=1)


if __name__ == "__main__":
    main()

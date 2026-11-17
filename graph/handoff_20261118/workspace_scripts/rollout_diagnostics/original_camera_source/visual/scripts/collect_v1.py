"""E0: v1 데이터셋 수집 (dev/eval seed 분리, 정책 다양화, 시뮬레이션 교란 세트).

collect.py(Step 2)의 waypoint 스크립트 정책 / Recorder / npz 포맷을 그대로 쓰고 아래를 추가한다.

정책 (seed % 10 으로 결정, ep npz 의 `policy` key + summary.json 에 기록)
  expert (0-3) : Step 2 와 동일 (reach → grasp → lift → pre-insert ×3 refine → insert)
  noisy  (4-6) : 중간 waypoint(reach 접근점, lift, 첫 pre-insert)에 ±1–2 cm / ±5° 노이즈. refine 은 GT 로 보정
  retry  (7-8) : 첫 삽입을 hole 에서 1.2–2 cm 어긋나게 시도 → 막힘 → 후퇴 → refine → 재삽입
  fail   (9)   : 노이즈 pre-insert 를 refine 없이(1회만) 삽입 시도 + 1–2 cm 어긋남 → 재시도 없음 (실패 궤적)

시뮬레이션 교란 (--perturb, 같은 seed 를 별도 디렉터리에 저장)
  box_move : 삽입 직전(마지막 refine 직후) box 를 xy 평면에서 2–5 cm 순간이동 (kinematic actor set_pose).
             expert 는 GT 로 다시 refine 해서 따라간다 (`box_move_frame`, `box_move_offset` 기록)
  lighting : ambient / directional light 세기와 색을 seed 별로 무작위화 (`lighting` 기록)
이미지 교란 patch / dropout 은 시뮬레이션이 필요 없으므로 v1data.py 가 spec(json)으로 로드 시 적용한다.

Usage:
  source env.sh
  python scripts/collect_v1.py --split dev  --seeds $(seq 100 109) --out data/v1/dev
  python scripts/collect_v1.py --split eval --seeds $(seq 200 239) --out data/v1/eval
  python scripts/collect_v1.py --split eval --seeds $(seq 200 239) --perturb box_move --out data/v1/eval_box_move
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import sapien

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import CONTROL_MODE, ENV_ID, Recorder, ScriptedPolicy, actor_ids, np0, sp  # noqa: E402
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv  # noqa: E402
from mani_skill.examples.motionplanning.base_motionplanner.utils import (  # noqa: E402
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.registration import register_env  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

POLICY_BY_MOD = ["expert"] * 4 + ["noisy"] * 3 + ["retry"] * 2 + ["fail"]


@register_env("PegInsertionSideLit-v1", max_episode_steps=100)
class PegInsertionSideLit(PegInsertionSideEnv):
    """조명을 episode seed 로 무작위화 (reconfigure 마다 _load_lighting 이 다시 불림)."""
    light_cfg = None

    def _load_lighting(self, options):
        rng = np.random.default_rng(int(self._episode_seed[0]) + 7919)
        amb = rng.uniform(0.05, 0.6) * rng.uniform(0.7, 1.0, 3)
        d1 = rng.uniform(0.3, 2.0) * rng.uniform(0.6, 1.0, 3)
        d2 = rng.uniform(0.0, 1.5) * rng.uniform(0.6, 1.0, 3)
        dir1 = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), -1.0])
        self.scene.set_ambient_light(amb.tolist())
        self.scene.add_directional_light(dir1.tolist(), d1.tolist(), shadow=self.enable_shadow,
                                         shadow_scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([0, 0, -1], d2.tolist())
        self.light_cfg = dict(ambient=amb.round(3).tolist(), dir1=dir1.round(3).tolist(), dir1_color=d1.round(3).tolist(),
                              dir2_color=d2.round(3).tolist())


def noise_pose(rng, pos_cm, rot_deg):
    """랜덤 방향, 크기 pos_cm∈[lo,hi] cm / 회전 rot_deg 이하 의 world-frame 교란 Pose."""
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    p = v * rng.uniform(*pos_cm) / 100
    r = Rotation.from_rotvec(rng.normal(size=3) / np.sqrt(3) * np.deg2rad(rot_deg)).as_quat()
    return sapien.Pose(p, [r[3], r[0], r[1], r[2]])


class V1Policy(ScriptedPolicy):
    def __init__(self, env, kind, rng, box_move=False):
        super().__init__(env)
        self.kind, self.rng, self.box_move = kind, rng, box_move
        self.log = dict(policy=kind)

    def perturb(self, pose, lo=1.0, hi=2.0, rot=5.0):
        n = noise_pose(self.rng, (lo, hi), rot)
        return sapien.Pose(pose.p + n.p, (n * sapien.Pose(q=pose.q)).q)

    def refine(self, pre, goal, offset, n=3):
        for _ in range(n):
            delta = goal * offset * sp(self.u.peg.pose).inv()
            pre = delta * pre
            self.move(pre)
        return pre

    def run(self):
        u, k = self.u, self.kind
        L = u.peg_half_sizes[0, 0].item()
        obb = get_actor_obb(u.peg)
        target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        peg_init_pose = sp(u.peg.pose)
        info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1]), target_closing=target_closing, depth=0.025)
        grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
        grasp_pose = grasp_pose * sapien.Pose([-max(0.05, L / 2 + 0.01), 0, 0])
        noisy = k in ("noisy", "fail")
        approach = grasp_pose * sapien.Pose([0, 0, -0.05])
        self.move(self.perturb(approach) if noisy else approach)
        self.move(grasp_pose)
        self.gripper(-1.0)
        lift = sapien.Pose([0, 0, 0.05]) * sp(u.agent.tcp.pose)
        self.move(self.perturb(lift) if noisy else lift)
        goal = sp(u.goal_pose)
        insert_pose = goal * peg_init_pose.inv() * grasp_pose
        offset = sapien.Pose([-0.01 - L, 0, 0])
        pre = insert_pose * offset
        self.move(self.perturb(pre) if noisy else pre)
        push = sapien.Pose([0.01 + L + 0.05, 0, 0])
        if k == "fail":
            pre = self.refine(pre, goal, offset, n=1)
            miss = self.lateral_miss(2.5, 3.5)
            self.move(pre * miss * push)
            return 0
        pre = self.refine(pre, goal, offset, n=3)
        if self.box_move:
            pre, goal = self.do_box_move(pre, offset)
        if k == "retry":
            miss = self.lateral_miss(1.2, 2.0)
            self.move(pre * miss * sapien.Pose([0.01 + 0.03, 0, 0]))  # 입구까지 밀고 막힘
            self.move(pre * miss)  # 후퇴
            pre = self.refine(pre, goal, offset, n=2)
        self.move(pre * push)
        return 0

    def lateral_miss(self, lo, hi):
        """peg 축(x)에 수직인 y/z 방향으로 lo–hi cm 어긋남 (pre-insert frame)."""
        a = self.rng.uniform(0, 2 * np.pi)
        r = self.rng.uniform(lo, hi) / 100
        self.log["miss_cm"] = round(r * 100, 2)
        return sapien.Pose([0, r * np.cos(a), r * np.sin(a)])

    def do_box_move(self, pre, offset):
        u = self.u
        a = self.rng.uniform(0, 2 * np.pi)
        r = self.rng.uniform(0.02, 0.05)
        bp = sp(u.box.pose)
        new = sapien.Pose(bp.p + np.array([r * np.cos(a), r * np.sin(a), 0.0]), bp.q)
        u.box.set_pose(new)
        self.log["box_move_frame"] = len(self.env.frames)  # 다음 기록 프레임부터 새 위치
        self.log["box_move_offset"] = [round(r * np.cos(a), 4), round(r * np.sin(a), 4)]
        goal = sp(u.goal_pose)
        pre = self.refine(pre, goal, offset, n=3)
        return pre, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb", default="none", choices=["none", "box_move", "lighting"])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cams", nargs="+", default=["base_camera", "hand_camera"])
    ap.add_argument("--shader", default="minimal", help="sensor shader_pack (E8: default)")
    ap.add_argument("--policy", default=None, help="모든 ep 에 이 정책 강제 (기본: seed%%10 규칙)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env_id = "PegInsertionSideLit-v1" if args.perturb == "lighting" else ENV_ID
    env = gym.make(env_id, obs_mode="rgb+depth+segmentation", control_mode=CONTROL_MODE,
                   sensor_configs=dict(width=args.res, height=args.res, shader_pack=args.shader),
                   max_episode_steps=1000, num_envs=1)
    env = Recorder(env, args.cams)
    summary = []
    for seed in args.seeds:
        t0 = time.time()
        kind = args.policy or POLICY_BY_MOD[seed % 10]
        if args.perturb == "box_move" and kind == "fail":
            kind = "expert"  # box_move 는 이동 후 따라가는지가 관심사 → fail 정책 제외
        env.reset(seed=seed)
        rng = np.random.default_rng(seed * 31 + 1)
        pol = V1Policy(env, kind, rng, box_move=args.perturb == "box_move")
        pol.run()
        data = env.stacked()
        u = env.unwrapped
        id_map, peg_id, box_id = actor_ids(u)
        extra = dict(pol.log)
        if args.perturb == "lighting":
            extra["lighting"] = u.light_cfg
        out = os.path.join(args.out, f"ep{seed:03d}.npz")
        np.savez_compressed(out, **data, peg_id=peg_id, box_id=box_id, seed=seed, policy=kind, split=args.split,
                            perturb=args.perturb, meta=json.dumps(dict(pol.log, **extra)),
                            peg_half_sizes=np0(u.peg_half_sizes), box_hole_radii=np0(u.box_hole_radii),
                            env_id=env_id, control_mode=CONTROL_MODE)
        with open(os.path.join(args.out, "segmentation_id_map.json"), "w") as fp:
            json.dump(id_map, fp, indent=1)
        succ = data["success"]
        extra.pop("policy", None)
        s = dict(seed=seed, policy=kind, frames=len(succ), success_final=bool(succ[-1]), success_any=bool(succ.any()),
                 first_success_frame=int(np.argmax(succ)) if succ.any() else None, peg_id=peg_id, box_id=box_id,
                 seconds=round(time.time() - t0, 1), mb=round(os.path.getsize(out) / 1e6, 1), **extra)
        print(json.dumps(s), flush=True)
        summary.append(s)
    env.close()
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=1)


if __name__ == "__main__":
    main()

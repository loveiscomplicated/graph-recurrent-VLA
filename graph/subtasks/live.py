"""Reusable one-CPU-environment GT adapter; reset once per new episode."""
import numpy as np


class LiveSubtaskAccessor:
    def reset(self,env):
        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb,compute_grasp_info_by_obb
        task=env.unwrapped
        info=compute_grasp_info_by_obb(get_actor_obb(task.peg),approaching=np.array([0,0,-1]),
            target_closing=task.agent.tcp.pose.to_transformation_matrix()[0,:3,1].cpu().numpy(),depth=.025)
        grasp=task.agent.build_grasp_pose([0,0,-1],info['closing'],info['center'])
        grasp=grasp*sapien.Pose([-max(.05,task.peg_half_sizes[0,0].item()/2+.01),0,0])
        reach=grasp*sapien.Pose([0,0,-.05])
        initial=task.peg.pose.sp
        lr=initial.inv()*reach;lg=initial.inv()*grasp
        self.reach_local=np.r_[lr.p,lr.q];self.grasp_local=np.r_[lg.p,lg.q]
        self.initial_peg_pose=task.peg.pose.raw_pose[0].cpu().numpy().copy()
        return self

    def __call__(self,env,*,contact_valid):
        """contact_valid=False after snapshot restore/reset; True only after step."""
        from envs.maniskill import get_gt_state
        if not hasattr(self,'reach_local'):raise ValueError('call reset after episode initialization')
        task=env.unwrapped;s=get_gt_state(env)
        if len(s.peg_pose)!=1:raise ValueError('this validated accessor supports one CPU environment')
        row={k:getattr(s,k)[0].cpu().numpy().copy() for k in (
            'ee_pose','peg_pose','hole_pose','box_pose','peg_half_sizes','hole_radius','gripper_qpos','qpos','qvel','success')}
        row.update(grasp=bool(task.agent.is_grasping(task.peg).item()) if contact_valid else False,
            contact_valid=bool(contact_valid),
            finger_forces=np.stack([task.scene.get_pairwise_contact_forces(link,task.peg)[0].cpu().numpy()
                for link in (task.agent.finger1_link,task.agent.finger2_link)]) if contact_valid else np.zeros((2,3)),
            peg_box_force=task.scene.get_pairwise_contact_forces(task.peg,task.box)[0].cpu().numpy() if contact_valid else np.zeros(3),
            reach_local=self.reach_local.copy(),grasp_local=self.grasp_local.copy(),initial_peg_pose=self.initial_peg_pose.copy())
        return row

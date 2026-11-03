"""Independent seeded scene-identity audit; no model forward, action or new data selection."""
from common import *
import numpy as np
import torch
from capture import physical,arr


def main():
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.utils.structs.pose import Pose
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    import sapien.physx as physx
    torch.set_num_threads(1)
    protocol=assert_source();entries=read(ROOT/'capture_manifest.json')['entries'];checks=[]
    groups={}
    for s in protocol['scenes']:groups.setdefault(s['parent'],[]).append(s)
    with logger('independent_seeded_geometry_identity') as log:
        for parent,specs in groups.items():
            seed=98202 if parent=='seed98202' else read(Path(specs[0]['raw']).parent.parent/'attempt_identity.json')['seed']
            for size in SIZES:
                env=PegInsertionSideEnv(num_envs=1,sim_backend='physx_cpu',render_backend='cpu',
                    obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose',
                    sensor_configs={c:dict(width=size,height=size) for c in CAMERAS})
                try:
                    env.reset(seed=seed);task=env.unwrapped
                    geometry={}
                    for oid,actor in enumerate((task.peg,task.box)):
                        meshes=[m for comp in actor._objs[0].components if isinstance(comp,physx.PhysxRigidBaseComponent) for m in get_component_meshes(comp)]
                        for mi,m in enumerate(meshes):geometry[f'object{oid}_vertices{mi}']=np.asarray(m.vertices);geometry[f'object{oid}_faces{mi}']=np.asarray(m.faces)
                    for s in specs:
                        t=s['timestep'];state=read(ROOT/'GT'/s['id']/'state.json')
                        if parent!='seed98202':
                            with np.load(s['gt'],allow_pickle=False) as gt:
                                task.agent.robot.set_qpos(torch.tensor(gt['physics_qpos'][t][None]));task.agent.robot.set_qvel(torch.tensor(gt['physics_qvel'][t][None]))
                                task.peg.set_pose(Pose.create(gt['physics_peg_pose'][t][None]));task.box.set_pose(Pose.create(gt['physics_box_pose'][t][None]));task._elapsed_steps[:]=t
                                key='physics_hole_pose' if 'physics_hole_pose' in gt else 'raw_hole_pose'
                                np.testing.assert_allclose(arr(task.box_hole_pose.raw_pose)[0],gt[key][t],atol=1e-6,rtol=0)
                                np.testing.assert_array_equal(arr(task.peg_half_sizes)[0],gt['raw_peg_half'][t])
                                np.testing.assert_array_equal(arr(task.box_hole_radii).ravel(),gt['raw_hole_radius'][t].ravel())
                        np.testing.assert_allclose(arr(task.box_hole_pose.raw_pose)[0],state['hole_pose'],atol=1e-6,rtol=0)
                        np.testing.assert_array_equal(arr(task.peg_half_sizes)[0],state['peg_half'])
                        assert float(arr(task.box_hole_radii).ravel()[0])==state['hole_radius']
                        with np.load(ROOT/'GT'/s['id']/'meshes.npz',allow_pickle=False) as saved:
                            assert set(saved.files)==set(geometry)
                            for k,value in geometry.items():np.testing.assert_array_equal(value,saved[k])
                        reference=next(e for e in entries if e['scene']==s['id'] and e['size']==size)
                        assert digest(physical(task))==reference['physical_sha256'],(s['id'],size,'physical hash')
                        checks.append(dict(scene=s['id'],size=size,hole_pose_checked=True,geometry_checked=True,physical_state_exact=True))
                    print('IDENTITY',parent,size,flush=True)
                finally:env.close()
        assert len(checks)==460
        write(ROOT/'geometry_identity_verification.json',dict(status='PASS',checks=checks,states=115,resolutions=4,
            independent_environment_resets=len(groups)*4,explicit_physics_steps=0,model_forwards=0,
            source_sha256=sha(__file__),source_inputs_changed=False,role='input identity only; no perception performance scoring'))
        log.log(dict(condition_checks=len(checks)));log.link_output(ROOT/'geometry_identity_verification.json')


if __name__=='__main__':main()

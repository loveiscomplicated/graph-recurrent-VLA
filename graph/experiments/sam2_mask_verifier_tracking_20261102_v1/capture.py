"""Restore every recorded state for native512 frames; no physical stepping."""
from common import *
from dataclasses import replace
import numpy as np,torch,h5py
from PIL import Image
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs.pose import Pose
from utils.experiment_logger import ExperimentLogger
class NativeEnv(PegInsertionSideEnv):
    def _setup_sensors(self,options):
        super()._setup_sensors(options)
        for uid in ['base_camera','hand_camera']:
            conf=replace(self._sensor_configs[uid],uid=uid+'_512',width=512,height=512)
            self._sensor_configs[conf.uid]=conf;self._sensors[conf.uid]=Camera(conf,self.scene,articulation=self.agent.robot if uid=='hand_camera' else None)
def arr(x):return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()
def physical(task):
    return dict(qpos=arr(task.agent.robot.get_qpos()),qvel=arr(task.agent.robot.get_qvel()),root=arr(task.agent.robot.pose.raw_pose),peg=arr(task.peg.pose.raw_pose),box=arr(task.box.pose.raw_pose),clock=arr(task.elapsed_steps))
def ids(actor):return [int(x.per_scene_id) for x in actor._objs]
v=rd(R/'verification.json');assert v['stage']=='STATIC_PREDICTIONS_FROZEN'
manifest=rd(R/'tracking_manifest.json');sequences=[];audits=[];torch.set_num_threads(4);env=None
try:
    with ExperimentLogger(team='integration',model='Task10I_native512_sequence',seed=42,job_type='prepare',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
        for parent in manifest['parents']:
            pid=parent['parent_id'];seed=rd(Path(parent['raw_path']).parent.parent/'attempt_identity.json')['seed']
            assert sha(parent['raw_path'])==parent['raw_sha256'] and sha(parent['gt_path'])==parent['gt_sha256']
            env=NativeEnv(num_envs=1,sim_backend='physx_cpu',render_backend='cpu',obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose');task=env.unwrapped;env.reset(seed=seed)
            actors=dict(peg=ids(task.peg),box=ids(task.box),robot=[int(x.entity.per_scene_id) for x in task.agent.robot._objs[0].links])
            percam={cam:[] for cam in manifest['cameras']};mismatch=[]
            with np.load(parent['gt_path'],allow_pickle=False) as gt,h5py.File(parent['raw_path'],'r') as h:
                np.testing.assert_array_equal(arr(task.peg_half_sizes)[0],gt['raw_peg_half'][0]);np.testing.assert_array_equal(arr(task.box_hole_radii).reshape(-1),gt['raw_hole_radius'][0])
                for t in manifest['timesteps']:
                    task.agent.robot.set_qpos(torch.tensor(gt['physics_qpos'][t][None]));task.agent.robot.set_qvel(torch.tensor(gt['physics_qvel'][t][None]))
                    task.peg.set_pose(Pose.create(gt['physics_peg_pose'][t][None]));task.box.set_pose(Pose.create(gt['physics_box_pose'][t][None]));task._elapsed_steps[:]=t
                    before=physical(task);obs=task.get_obs();after=physical(task);assert all(np.array_equal(before[k],after[k]) for k in before)
                    np.testing.assert_allclose(arr(task.agent.tcp.pose.raw_pose)[0],gt['physics_ee_pose'][t],atol=1e-6,rtol=0);np.testing.assert_allclose(arr(task.box_hole_pose.raw_pose)[0],gt['physics_hole_pose'][t],atol=1e-6,rtol=0)
                    for cam in manifest['cameras']:
                        uid=cam+'_512';rgb=arr(obs['sensor_data'][uid]['rgb'])[0];seg=arr(obs['sensor_data'][uid]['segmentation'])[0,...,0];depth=arr(obs['sensor_data'][uid]['depth'])[0,...,0]
                        params={k:arr(z)[0] for k,z in obs['sensor_param'][uid].items()};assert rgb.shape==(512,512,3)
                        np.testing.assert_allclose(params['extrinsic_cv'],h[cam+'/extrinsic_cv'][t],atol=2e-6,rtol=0);np.testing.assert_allclose(params['intrinsic_cv'][:2]/4,h[cam+'/intrinsic'][t][:2],atol=1e-5,rtol=0)
                        folder=R/'sequences'/pid/cam;folder.mkdir(parents=True,exist_ok=True);png=folder/f'{t:05}.png';Image.fromarray(rgb).save(png)
                        jpgfolder=folder/'jpeg';jpgfolder.mkdir(exist_ok=True);jpg=jpgfolder/f'{t:05}.jpg';Image.fromarray(rgb).save(jpg,quality=100,subsampling=0)
                        decoded=np.asarray(Image.open(jpg).convert('RGB'));delta=np.abs(decoded.astype(float)-rgb.astype(float))
                        side=R/'.execution/GT_sequences'/pid/cam/f'{t:05}.npz';side.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(side,actor_segmentation=seg,depth_raw_mm=depth,**params)
                        if t in [0,80]:
                            prior=next(c for c in rd(R/'input_manifest.json')['cases'] if c['parent_id']==pid and c['camera']==cam and c['timestep']==t)
                            assert np.array_equal(rgb,np.asarray(Image.open(prior['image_path']).convert('RGB'))),'Native512 endpoint changed'
                        percam[cam].append(dict(timestep=t,PNG_path=str(png),PNG_sha256=sha(png),JPEG_path=str(jpg),JPEG_sha256=sha(jpg),GT_path=str(side),GT_sha256=sha(side),JPEG_mean_abs_error=float(delta.mean()),JPEG_max_abs_error=float(delta.max())))
                    if (t+1)%20==0:print('CAPTURE_PROGRESS',pid,t+1,'/81',flush=True)
            for cam,frames in percam.items():sequences.append(dict(id=pid+'_'+cam,parent_id=pid,camera=cam,object_ids=actors,jpeg_folder=str(R/'sequences'/pid/cam/'jpeg'),frames=frames,seed=seed))
            audits.append(dict(parent_id=pid,frames=81,physics_steps=0,state_unchanged=True,calibration_verified=True,endpoint_RGB_exact=True,backend=dict(sim=str(task.backend.sim_backend),render=str(task.backend.render_backend),device=str(task.device))))
            env.close();env=None;wr('sequence_manifest.json',dict(sequences=sequences));wr('capture_verification.json',dict(parents=audits,environment_constructors=len(audits),explicit_seeded_resets=len(audits),physics_steps=0))
        assert len(sequences)==16 and sum(len(s['frames']) for s in sequences)==1296
        update(stage='SEQUENCES_FROZEN',sequence_manifest_sha256=sha(R/'sequence_manifest.json'));logger.log(dict(native_frames=1296,physics_steps=0));logger.link_output(R/'sequence_manifest.json');print('SEQUENCES_FROZEN',flush=True)
finally:
    if env is not None:env.close()

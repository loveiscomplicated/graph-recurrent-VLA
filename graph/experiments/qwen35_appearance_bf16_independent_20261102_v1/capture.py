"""Static, paired camera capture. No expert/action generation or simulation steps.

Privileged recorded state is used ONLY to reconstruct the scene, never a VLM input.
Fresh actor segmentation is written in sealed sidecars for post-inference evaluation.
"""
from common import *
import copy,time,traceback
from dataclasses import replace
import numpy as np,torch,h5py
from PIL import Image
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs.pose import Pose
from utils.experiment_logger import ExperimentLogger
class PairedCamerasEnv(PegInsertionSideEnv):
 def _setup_sensors(self,options):
  super()._setup_sensors(options)
  for uid in ['base_camera','hand_camera']:
   conf=replace(self._sensor_configs[uid],uid=uid+'_512',width=512,height=512)
   self._sensor_configs[conf.uid]=conf
   self._sensors[conf.uid]=Camera(conf,self.scene,articulation=self.agent.robot if uid=='hand_camera' else None)
def arr(x):return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()
def physical(task):
 return dict(qpos=arr(task.agent.robot.get_qpos()),qvel=arr(task.agent.robot.get_qvel()),root=arr(task.agent.robot.pose.raw_pose),peg=arr(task.peg.pose.raw_pose),box=arr(task.box.pose.raw_pose),clock=arr(task.elapsed_steps))
def ids(actor):return [int(x.per_scene_id) for x in actor._objs]
v=rd(R/'verification.json');assert v['stage']=='PROTOCOL_FROZEN'
sel=rd(R/'selection_manifest.json');cases=[];audits=[];env=None
start=time.perf_counter()
try:
 torch.set_num_threads(4)
 with ExperimentLogger(team='integration',model='Task10I_native_resolution_capture',seed=42,job_type='prepare',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
  for parent in sel['parents']:
   pid=parent['parent_id'];raw=Path(parent['raw_path']);seed=rd(raw.parent.parent/'attempt_identity.json')['seed']
   env=PairedCamerasEnv(num_envs=1,sim_backend='physx_cpu',render_backend='cpu',obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose')
   task=env.unwrapped
   # Constructor reset is ManiSkill's own initialization. One seeded reset creates exact historical geometry.
   env.reset(seed=seed)
   with h5py.File(raw,'r') as h,np.load(parent['gt_path'],allow_pickle=False) as gt:
    object_ids=dict(peg=ids(task.peg),box=ids(task.box),robot=[int(x.entity.per_scene_id) for x in task.agent.robot._objs[0].links])
    assert not(set(object_ids['peg'])&set(object_ids['box']))
    np.testing.assert_array_equal(arr(task.peg_half_sizes)[0],gt['raw_peg_half'][0])
    np.testing.assert_array_equal(arr(task.box_hole_radii).reshape(-1),gt['raw_hole_radius'][0])
    for t in sorted({c['timestep'] for c in sel['cases'] if c['parent_id']==pid}):
     # Static render restoration only; not claimed as a dynamically replayable solver snapshot.
     task.agent.robot.set_qpos(torch.tensor(gt['physics_qpos'][t][None]))
     task.agent.robot.set_qvel(torch.tensor(gt['physics_qvel'][t][None]))
     task.peg.set_pose(Pose.create(gt['physics_peg_pose'][t][None]))
     task.box.set_pose(Pose.create(gt['physics_box_pose'][t][None]))
     task._elapsed_steps[:]=t
     before=physical(task);clock=int(task.elapsed_steps.item())
     obs=task.get_obs();after=physical(task)
     assert all(np.array_equal(before[k],after[k]) for k in before)
     np.testing.assert_allclose(arr(task.agent.tcp.pose.raw_pose)[0],gt['physics_ee_pose'][t],atol=1e-6,rtol=0)
     np.testing.assert_allclose(arr(task.box_hole_pose.raw_pose)[0],gt['physics_hole_pose'][t],atol=1e-6,rtol=0)
     for original in [c for c in sel['cases'] if c['parent_id']==pid and c['timestep']==t]:
      cam=original['camera'];pair=dict(id=original['id'],parent_id=pid,timestep=t,camera=cam,frame_kind=original['frame_kind'],seed=seed,clock_before=clock,clock_after=int(task.elapsed_steps.item()),physical_state_unchanged=True,physics_steps=0,backend=dict(sim=str(task.backend.sim_backend),render=str(task.backend.render_backend),device=str(task.device)),object_ids=object_ids,original128_rgb_exact=False,camera_params={})
      for size,uid in [(128,cam),(512,cam+'_512')]:
       rgb=arr(obs['sensor_data'][uid]['rgb'])[0];seg=arr(obs['sensor_data'][uid]['segmentation'])[0,...,0]
       depth=arr(obs['sensor_data'][uid]['depth'])[0,...,0]
       assert rgb.dtype==np.uint8 and rgb.shape==(size,size,3) and seg.shape==(size,size)
       params={k:arr(z)[0] for k,z in obs['sensor_param'][uid].items()}
       # Native intrinsics and unchanged camera extrinsics; complete original view.
       np.testing.assert_allclose(params['extrinsic_cv'],h[cam+'/extrinsic_cv'][t],atol=2e-6,rtol=0)
       np.testing.assert_allclose(params['intrinsic_cv'][:2]/(size/128),h[cam+'/intrinsic'][t][:2],atol=1e-5,rtol=0)
       if size==128: pair['original128_rgb_exact']=bool(np.array_equal(rgb,h[cam+'/rgb'][t]));pair['original128_mean_abs_rgb_difference']=float(abs(rgb.astype(float)-h[cam+'/rgb'][t].astype(float)).mean())
       im=R/'images'/f'{original["id"]}_native{size}.png';im.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(im)
       side=R/'.execution/GT'/f'{original["id"]}_native{size}.npz';side.parent.mkdir(exist_ok=True);np.savez_compressed(side,actor_segmentation=seg,depth_raw_mm=depth,**params)
       if size==512:cases.append(dict(id=original['id']+f'_native{size}',pair_id=original['id'],parent_id=pid,frame_kind=original['frame_kind'],timestep=t,camera=cam,native_size=size,width=size,height=size,image_path=str(im),image_sha256=sha(im),RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),GT_path=str(side),GT_sha256=sha(side),object_ids=object_ids,source_raw_path=str(raw),source_raw_sha256=parent['raw_sha256']))
       pair['camera_params'][str(size)]={k:z.tolist() for k,z in params.items()}
      np.testing.assert_array_equal(pair['camera_params']['128']['extrinsic_cv'],pair['camera_params']['512']['extrinsic_cv'])
      audits.append(pair)
     print('CAPTURED',pid,'t',t,flush=True)
   env.close();env=None
  assert len(cases)==60 and len(audits)==60
  # No semantic visibility/IoU metric is computed or selected during capture.
  wr('input_manifest.json',dict(parents=sel['parents'],cases=cases,protocol_sha256=sha(R/'protocol.json'),selection_sha256=sha(R/'selection_manifest.json')))
  wr('capture_verification.json',dict(status='PASS',pairs=audits,physics_steps=0,environment_constructors=15,explicit_seeded_resets=15,restoration='recorded robot qpos/qvel + peg/box poses; seed-matched geometry, TCP/hole and camera calibration verified; no solver replay claim',wall_seconds=time.perf_counter()-start))
  v.update(stage='CAPTURE_COMPLETE',input_manifest_sha256=sha(R/'input_manifest.json'),capture_verification_sha256=sha(R/'capture_verification.json'));wr('verification.json',v)
  logger.log(dict(camera_frames=60,native_images=60,physics_steps=0));logger.link_output(R/'input_manifest.json')
except Exception as ex:
 v.update(stage='CAPTURE_BLOCKED',error_type=type(ex).__name__,error=str(ex),traceback=traceback.format_exc());wr('verification.json',v);print(traceback.format_exc(),flush=True);raise
finally:
 if env is not None:env.close()

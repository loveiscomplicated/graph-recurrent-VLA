"""Restore recorded physical states and capture aligned native RGB-D, without steps."""
from common import *
import argparse
import time
import numpy as np
import torch
import h5py
from PIL import Image


def arr(x):
    return x.detach().cpu().numpy().copy()


def physical(task):
    return {k: arr(v) for k, v in dict(qpos=task.agent.robot.get_qpos(),
            qvel=task.agent.robot.get_qvel(), peg=task.peg.pose.raw_pose,
            box=task.box.pose.raw_pose, root=task.agent.robot.pose.raw_pose,
            tcp=task.agent.tcp.pose.raw_pose, clock=task.elapsed_steps).items()}


def prepare():
    assert os.environ.get('CONTAINER_ID') == '58576054'
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'protocol.json').exists()
    appearance = Path('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1')
    timing = Path('/workspace/task10i_stage4a2_qwen35_native_resolution_prompt_20261102_v1')
    a = read(appearance/'selection_manifest.json')
    b = read(timing/'selection_manifest.json')
    scenes = []
    protected = {}
    for parent in a['parents']:
        cases = [c for c in a['cases'] if c['parent_id'] == parent['parent_id']]
        for t in sorted({c['timestep'] for c in cases}):
            scenes.append(dict(id=parent['parent_id']+f'_t{t:03d}', parent=parent['parent_id'],
                               timestep=t, cohort='static', raw=parent['raw_path'], gt=parent['gt_path']))
        protected[parent['raw_path']] = parent['raw_sha256']
        protected[parent['gt_path']] = parent['gt_sha256']
    assert len(scenes) == 30
    scenes.append(dict(id='seed98202_t000', parent='seed98202', timestep=0, cohort='seed98202',
        snapshot='/workspace/task10g_f1_abc_20261031/full/rollouts/final/C/step011487/seed98202/initial_snapshot.pt'))
    for parent in b['parents']:
        for t in range(21):
            scenes.append(dict(id=parent['parent_id']+f'_track_t{t:03d}', parent=parent['parent_id'],
                timestep=t, cohort='tracking', raw=parent['raw_path'], gt=parent['gt_path']))
        protected[parent['raw_path']] = parent['raw_sha256']
        protected[parent['gt_path']] = parent['gt_sha256']
    prompt = (ROOT/'reference/grounding_prompt.txt').read_text()
    (ROOT/'grounding_prompt.txt').write_text(prompt)
    protected[CHECKPOINT] = CHECKPOINT_SHA
    protected[scenes[30]['snapshot']] = sha(scenes[30]['snapshot'])
    for rel in ('graph/plan_a_bc/geometry.py','graph/plan_a_geometry/core.py',
                'graph/plan_a_geometry/protocol.json','graph/persistent_geometry/graph.py',
                'graph/integration_baseline/vision.py','visual/runtime/sampling.py'):
        protected[str(REPO/rel)] = sha(REPO/rel)
    for path, expected in protected.items():
        assert sha(path) == expected, path
    protocol = dict(task=NAME, created_utc=now(), sizes=SIZES, cameras=CAMERAS,
        scenes=scenes, api_repeats=3, api_requests_max=840, api_budget_new_usd=BUDGET,
        model=MODEL, reasoning_effort='low', service_tier='default', detail='original',
        max_output_tokens=MAX_OUTPUT, prompt_sha256=sha(ROOT/'grounding_prompt.txt'),
        input_price_per_million=1.0, output_price_per_million=5.0,
        request_reservation_usd=3.0, reservation_reason='Conservative full 1.05M-context Standard bound including 2x long-context input, 30-minute cache-write markup and 5120 output at 1.5x; one outstanding request only',
        api_retry=0, source_sha256={p.name:sha(p) for p in SOURCE.glob('*.py')},
        checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA, sam2_config=CONFIG,
        precision='FP32 weights; BF16 autocast; TF32 enabled', threads=4,
        static_timing_repeats=8, tracking_timing_repeats=4,
        static_warmups_per_size=3, tracking_warmups_per_size=1,
        graph_cap_per_object=128, graph_dedup_m=.003,
        graph_case_timeout_seconds=180, graph_process_memory_limit_gib=24,
        graph='CurrentGeometry C3/SF; Graph21/Edge4; no temporal features; Box main only',
        coverage_reference='same-state native1024 actor visible depth; 4mm; Peg both cameras / Box base only; all-camera ceiling separate',
        visibility='native1024 actor area >=16*(1024/128)^2; smaller visible and absent separate; denominators fixed',
        boundary_tolerance='1 pixel at128, scaled by width/128',
        causal_tracking='num_frames=t+1; one-frame forward; GT boxes only at t0; no reanchor',
        jpeg=dict(quality=100,subsampling=0),
        GT_barrier='actual API, masks, graphs and estimates frozen before oracle construction/scoring',
        protected_before=protected, shared_factory_changed=False, physics_steps=0,
        publishing=False, training=False, policy_rollout=False)
    write(ROOT/'protocol.json', protocol)
    print('PROTOCOL_FROZEN',len(scenes),'states',flush=True)


def capture(smoke=False):
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.utils.structs.pose import Pose
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    import sapien.physx as physx
    from graph.plan_a_bc.geometry import RobotFK
    from graph.integration_baseline.robot import robot_observation,flatten_state
    from visual.runtime.vision_packet import pose_matrix
    protocol = assert_source()
    assert os.environ.get('CONTAINER_ID') == '58576054'
    torch.set_num_threads(4)
    schema = read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
    fk = RobotFK('/workspace/task10g_f1_abc_20261031/full/inputs',schema)
    scenes = protocol['scenes'][:1] if smoke else sorted(protocol['scenes'],key=lambda s:(s['cohort']=='tracking' and s['timestep']>0,protocol['scenes'].index(s)))
    manifest_path = ROOT/('smoke_capture_manifest.json' if smoke else 'capture_manifest.json')
    entries = read(manifest_path)['entries'] if manifest_path.exists() else []
    done = {(e['scene'],e['size'],e['camera']) for e in entries}
    audits = []
    with logger('capture_smoke' if smoke else 'capture') as log:
        for si,s in enumerate(scenes):
            seed = 98202 if s['cohort']=='seed98202' else read(Path(s['raw']).parent.parent/'attempt_identity.json')['seed']
            for size in SIZES:
                if all((s['id'],size,c) in done for c in CAMERAS):
                    continue
                # Override the two native cameras; no duplicate cameras or enlarged images.
                env=PegInsertionSideEnv(num_envs=1, sim_backend='physx_cpu',render_backend='cpu',
                    obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose',
                    sensor_configs={c:dict(width=size,height=size) for c in CAMERAS})
                try:
                    env.reset(seed=seed);task=env.unwrapped;t=s['timestep']
                    if s['cohort']!='seed98202':
                        with np.load(s['gt'],allow_pickle=False) as g:
                            for attr,key in (('qpos','physics_qpos'),('qvel','physics_qvel')):
                                getattr(task.agent.robot,'set_'+attr)(torch.tensor(g[key][t][None]))
                            task.peg.set_pose(Pose.create(g['physics_peg_pose'][t][None]))
                            task.box.set_pose(Pose.create(g['physics_box_pose'][t][None]))
                            np.testing.assert_array_equal(arr(task.peg_half_sizes)[0],g['raw_peg_half'][t])
                            np.testing.assert_allclose(arr(task.agent.tcp.pose.raw_pose)[0],g['physics_ee_pose'][t],atol=1e-6,rtol=0)
                        task._elapsed_steps[:]=t
                    before=physical(task);start=time.perf_counter();obs=task.get_obs();capture_ms=(time.perf_counter()-start)*1000
                    after=physical(task);assert all(np.array_equal(before[k],after[k]) for k in before)
                    if s['cohort']=='seed98202':
                        snap=torch.load(s['snapshot'],map_location='cpu',weights_only=False)
                        live=task.get_state_dict()
                        for kind in ('actors','articulations'):
                            for name,value in snap['sim'][kind].items():
                                np.testing.assert_array_equal(arr(live[kind][name]),np.asarray(value))
                        np.testing.assert_array_equal(arr(task._elapsed_steps),np.asarray(snap['clock']))
                        write(ROOT/'seed98202_snapshot_verification.json',dict(sim_exact=True,clock_exact=True,snapshot_sha256=sha(s['snapshot'])))
                    ids=dict(peg=[int(x.per_scene_id) for x in task.peg._objs],box=[int(x.per_scene_id) for x in task.box._objs],
                        robot=[int(x.entity.per_scene_id) for x in task.agent.robot._objs[0].links])
                    tcp=arr(task.agent.tcp.pose.raw_pose)[0]
                    if s['cohort']=='seed98202':
                        fields,pads=robot_observation(env);robot=flatten_state(fields,schema)
                    else:
                        with h5py.File(s['raw'],'r') as h:
                            robot=h['robot_state'][t];pads=h['robot_world_geometry'][t]
                    fk.verify_robot(robot,pads,tcp)
                    public=ROOT/'sensors'/s['id']/str(size);public.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(public/'robot.npz',robot76=robot,robot_world_geometry=pads,tcp_pose=tcp)
                    gtstate=dict(peg_pose=arr(task.peg.pose.raw_pose)[0],box_pose=arr(task.box.pose.raw_pose)[0],
                        hole_pose=arr(task.box_hole_pose.raw_pose)[0],peg_half=arr(task.peg_half_sizes)[0],
                        hole_radius=float(arr(task.box_hole_radii).ravel()[0]))
                    gtbase=ROOT/'GT'/s['id'];gtbase.mkdir(parents=True,exist_ok=True)
                    write(gtbase/'state.json',gtstate)
                    meshdata={}
                    for oid,actor in enumerate((task.peg,task.box)):
                        meshes=[m for comp in actor._objs[0].components if isinstance(comp,physx.PhysxRigidBaseComponent) for m in get_component_meshes(comp)]
                        for mi,m in enumerate(meshes):
                            meshdata[f'object{oid}_vertices{mi}']=np.asarray(m.vertices)
                            meshdata[f'object{oid}_faces{mi}']=np.asarray(m.faces)
                    np.savez_compressed(gtbase/'meshes.npz',**meshdata)
                    for c in CAMERAS:
                        data={k:arr(v)[0] for k,v in obs['sensor_param'][c].items()}
                        rgb=arr(obs['sensor_data'][c]['rgb'])[0]
                        depth=arr(obs['sensor_data'][c]['depth'])[0,...,0]
                        seg=arr(obs['sensor_data'][c]['segmentation'])[0,...,0]
                        assert rgb.shape==(size,size,3) and depth.dtype==np.int16
                        if s['cohort']!='seed98202':
                            with h5py.File(s['raw'],'r') as h:
                                np.testing.assert_allclose(data['extrinsic_cv'],h[c+'/extrinsic_cv'][t],atol=2e-6,rtol=0)
                                np.testing.assert_allclose(data['intrinsic_cv'][:2]/(size/128),h[c+'/intrinsic'][t][:2],atol=1e-5,rtol=0)
                        png=public/(c+'.png');Image.fromarray(rgb).save(png)
                        np.savez_compressed(public/(c+'.npz'),rgb=rgb,depth_raw_mm=depth,**data)
                        gtfile=gtbase/str(size)/(c+'.npz');gtfile.parent.mkdir(exist_ok=True)
                        np.savez_compressed(gtfile,actor_segmentation=seg)
                        if s['cohort']=='tracking':
                            jpg=ROOT/'jpeg'/s['parent']/str(size)/c/f'{t:05d}.jpg';jpg.parent.mkdir(parents=True,exist_ok=True)
                            Image.fromarray(rgb).save(jpg,quality=100,subsampling=0)
                        e=dict(scene=s['id'],parent=s['parent'],cohort=s['cohort'],timestep=t,size=size,camera=c,
                            id=image_id(s['id'],size,c),image_file=str(png.relative_to(ROOT)),image_sha256=sha(png),
                            sensor_file=str((public/(c+'.npz')).relative_to(ROOT)),sensor_sha256=sha(public/(c+'.npz')),
                            gt_file=str(gtfile.relative_to(ROOT)),gt_sha256=sha(gtfile),object_ids=ids,
                            capture_pair_ms=capture_ms,physical_sha256=digest(before),
                            backend=dict(sim=str(task.backend.sim_backend),render=str(task.backend.render_backend),device=str(task.device)))
                        entries.append(e);done.add((s['id'],size,c))
                    audits.append(dict(scene=s['id'],size=size,state_unchanged=True,physics_steps=0))
                    write(manifest_path,dict(entries=entries,audits=audits,status='PARTIAL'))
                    print('CAPTURE',si+1,len(scenes),s['id'],size,flush=True)
                finally:env.close()
        assert len(entries)==len(scenes)*8
        for s in scenes:
            assert len({e['physical_sha256'] for e in entries if e['scene']==s['id']})==1
        write(manifest_path,dict(entries=entries,audits=audits,status='COMPLETE',physics_steps=0))
        log.log(dict(native_camera_images=len(entries),physical_states=len(scenes)))
        log.link_output(manifest_path)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','capture']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:capture(a.stage=='smoke')

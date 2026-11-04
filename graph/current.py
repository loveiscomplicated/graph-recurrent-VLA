"""Current sensor-only adapter to byte-pinned09-D geometry. No object GT API."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from graph.persistent_geometry.contracts import CameraObservation, synchronized
from graph.persistent_geometry.self_filter import LinkSurface
from graph.plan_a_geometry.core import PlanA, budget
from visual.runtime.vision_packet import pose_matrix
from visual.runtime.coordinate_frames import check_frame, FRAME_NAMES
from vla.data.files import read, sha


def origin(element):
    m = np.eye(4)
    if element is not None:
        m[:3,3] = np.fromstring(element.get('xyz','0 0 0'), sep=' ')
        m[:3,:3] = Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return m


class RobotFK:
    """URDF joint FK only, verified against recorded SAPIEN links and Robot76.

    Meshes are each original SAPIEN collision shape, with its local transform
    already included in vertices. No convex hull enclosing the whole robot.
    """
    def __init__(self, assets, schema):
        self.assets = Path(assets); self.schema = schema
        ident = read(self.assets/'geometry_identity.json')
        if sha(self.assets/'robot.urdf') != ident['robot_urdf_sha256']:
            raise ValueError('09D robot URDF changed')
        urdf = ET.parse(self.assets/'robot.urdf').getroot()
        self.joints = list(urdf.findall('joint'))
        parents = {j.find('parent').get('link') for j in self.joints}
        children = {j.find('child').get('link') for j in self.joints}
        roots = parents-children
        if roots != {'panda_link0'}: raise ValueError('unreviewed URDF root')
        self.joint_names = [f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
        with np.load(self.assets/'robot_meshes.npz', allow_pickle=False) as a:
            self.names = a['links'].tolist()
            self.vertices = [a[f'vertices_{i}'].copy() for i in range(len(self.names))]
        self.fields = {f['name']:slice(f['offset'],f['offset']+f['dimension']) for f in schema['fields']}

    def transforms(self, robot):
        q = dict(zip(self.joint_names, robot[self.fields['qpos']]))
        transforms = {'panda_link0':pose_matrix(robot[self.fields['robot_base_pose']])}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                parent = joint.find('parent').get('link')
                if parent not in transforms: continue
                child = joint.find('child').get('link'); m = origin(joint.find('origin'))
                typ = joint.get('type')
                if typ != 'fixed':
                    name = joint.get('name')
                    if name not in q: raise ValueError('unknown active robot joint '+name)
                    axis_element = joint.find('axis')
                    axis = np.fromstring(axis_element.get('xyz','1 0 0'),sep=' ')
                    motion = np.eye(4)
                    if typ in ('revolute','continuous'):
                        motion[:3,:3] = Rotation.from_rotvec(axis*float(q[name])).as_matrix()
                    elif typ == 'prismatic': motion[:3,3] = axis*float(q[name])
                    else: raise ValueError('unknown URDF joint type')
                    m = m@motion
                transforms[child] = transforms[parent]@m
                remaining.remove(joint); progressed = True
            if not progressed: raise ValueError('unresolved robot FK tree')
        return transforms

    def surfaces(self, robot):
        transforms = self.transforms(robot)
        return [LinkSurface(n,v,transforms[n]) for n,v in zip(self.names,self.vertices)]

    def verify_robot(self, robot, pads, tcp):
        transforms = self.transforms(robot)
        errors = []
        for name, link in [('tcp_pose','panda_hand_tcp'),('left_finger_pose','panda_leftfinger'),
                           ('right_finger_pose','panda_rightfinger')]:
            errors.append(np.max(abs(pose_matrix(robot[self.fields[name]])-transforms[link])))
        np.testing.assert_allclose(transforms['panda_hand_tcp'],pose_matrix(tcp),atol=3e-6,rtol=0)
        actual = [transforms['panda_hand_tcp'][:3,3]]
        for link,offset in [('panda_leftfinger',[0,.00758,.04525]),('panda_rightfinger',[0,-.00758,.04525])]:
            mat = transforms[link]; actual.append(mat[:3,3]+mat[:3,:3]@offset)
        np.testing.assert_allclose(actual,pads,atol=3e-6,rtol=0)
        if max(errors)>3e-6: raise ValueError('recorded robot/FK mismatch')
        return float(max(errors))


class CurrentGeometry:
    def __init__(self, assets, schema, *, perception_backend='triton_fused', coordinate_frame='ee',robot_filter_backend=None):
        from functools import partial
        import torch
        from graph.persistent_geometry.wrist_gate_dispatch import default_wrist_gate_backend
        from visual.runtime.graph_perception import make_packet,make_packet_cuda
        if perception_backend not in ('triton_fused','cpu','cuda_resident'):
            raise ValueError('unknown Graph perception backend')
        # The default geometry must remain usable when CUDA is absent. An
        # available CUDA runtime is never replaced after an execution error.
        if perception_backend=='triton_fused' and not torch.cuda.is_available():perception_backend='cpu'
        self.perception_backend=perception_backend
        self.wrist_gate_backend=default_wrist_gate_backend()
        self.robot_filter_backend=('cpu' if perception_backend=='cpu' else 'cuda') if robot_filter_backend is None else robot_filter_backend
        if self.robot_filter_backend not in ('cpu','cuda'):raise ValueError('unknown robot filter backend')
        if perception_backend=='cuda_resident' and self.robot_filter_backend!='cuda':raise ValueError('cuda_resident requires CUDA robot filtering')
        self.robot_preparation=None;self.last_robot_schedule=None;self.last_raw_masks=None;self.last_wrist_gate_backend=None
        self.coordinate_frame=check_frame(coordinate_frame)
        self.fk = RobotFK(assets,schema)
        frozen = read(Path(__file__).resolve().parent/'plan_a_geometry/protocol.json')
        self.system = PlanA('base_camera','hand_camera',frozen,
            packet_builder=partial(make_packet,backend='triton_fused' if perception_backend=='cuda_resident' else perception_backend,
                                   coordinate_frame=self.coordinate_frame),coordinate_frame=self.coordinate_frame,
            tensor_packet_builder=partial(make_packet_cuda,coordinate_frame=self.coordinate_frame) if perception_backend=='triton_fused' else None,
            wrist_gate_backend='auto')
        self.tensor_geometry=None
        if perception_backend=='cuda_resident':
            from graph.experiments.cuda_ingress_10i_20261103_v1.pipeline import TensorGeometry
            self.tensor_geometry=TensorGeometry(self.system)
        self.mount = None

    def reset(self):
        self.mount = None

    def from_h5(self, f, t, episode, *, tensor_images=True):
        robot = f['robot_state'][t]; tcp = f['tcp_pose'][t]; pads = f['robot_world_geometry'][t]
        self.fk.verify_robot(robot,pads,tcp)
        cameras = [CameraObservation(str(episode),int(t),name,float(f['timestamp'][t]),
            f[name+'/rgb'][t],f[name+'/depth_raw_mm'][t],f[name+'/intrinsic'][t],
            f[name+'/extrinsic_cv'][t],f[name+'/cam2world_gl'][t],tcp.copy())
            for name in ['base_camera','hand_camera']]
        surfaces = self.fk.surfaces(robot)
        data = dict(episode=np.array(str(episode)),timestep=np.int64(t),timestamp=np.float64(t/20),
            robot76=robot,robot_world_geometry=pads,tcp_pose=tcp,qpos=robot[:9],
            robot_link_world=np.stack([s.world_from_link for s in surfaces]))
        if tensor_images and self.system.tensor_packet_builder is not None:
            from graph.persistent_geometry.tensor_camera import tensor_camera
            cameras=[tensor_camera(c) for c in cameras]
        return cameras,data,surfaces

    def _check_mount(self,cameras,surfaces):
        synchronized(cameras)
        wrist = next(c for c in cameras if c.camera=='hand_camera')
        # Fixed link-to-camera mount; current link FK is recomputed at every t.
        link = next(s for s in surfaces if s.link=='camera_link')
        mount = np.linalg.inv(link.world_from_link)@wrist.world_from_camera
        if self.mount is None: self.mount = mount
        np.testing.assert_allclose(mount,self.mount,atol=2e-5,rtol=0)

    def stage_inputs(self,cameras,data,surfaces):
        """Opt-in early upload, before a caller supplies current CUDA masks."""
        if self.tensor_geometry is None:raise ValueError('stage_inputs requires cuda_resident backend')
        self._check_mount(cameras,surfaces)
        return self.tensor_geometry.stage(cameras,data,surfaces)

    def build_staged(self,frame,masks):
        """Finish the exact snapshot returned by stage_inputs; no re-upload."""
        if self.tensor_geometry is None:raise ValueError('build_staged requires cuda_resident backend')
        graph,records,gate=self.tensor_geometry.build(frame,masks)
        self.last_wrist_gate_backend='torch_cuda'
        graph.packet['wrist_gate_backend']=self.last_wrist_gate_backend
        return self._finish(graph,records,gate,frame.data)

    def close(self):
        if self.robot_preparation is not None:self.robot_preparation.close();self.robot_preparation=None
        if self.tensor_geometry is not None:self.tensor_geometry.ingress.close()

    def build(self,cameras,data,surfaces,masks,*,mask_transform=None):
        """Build from raw masks or a zero-argument current-frame mask producer.

        Default CUDA filtering starts CPU preparation before invoking a mask
        producer. Cache/rollout pass their existing SAM2 calls in that producer.
        Already computed masks are supported, with no retroactive overlap claim.
        Optional mask_transform(raw_cuda, robot_frame) shares one full current
        robot classification with correction and the final clean masks.
        """
        self.last_raw_masks=None;self.last_robot_schedule=None;self.last_wrist_gate_backend=None
        if self.tensor_geometry is not None:
            if mask_transform is not None:raise ValueError('mask correction requires the default geometry backend')
            frame=self.stage_inputs(cameras,data,surfaces)
            return self.build_staged(frame,masks() if callable(masks) else masks)
        else:
            self._check_mount(cameras,surfaces)
            if self.robot_filter_backend=='cuda':
                if self.robot_preparation is None:
                    import torch
                    if not torch.cuda.is_available():raise RuntimeError('default robot filter requires CUDA; select explicit CPU diagnostics')
                    from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
                    from graph.plan_a_bc.tensor_ingress import TensorRobotPreparation
                    self.robot_preparation=TensorRobotPreparation(CudaRobotFilter())
                snapshot={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in data.items()}
                job=self.robot_preparation.submit(cameras,surfaces)
                self.last_robot_schedule='before_vision' if callable(masks) else 'masks_already_available'
                try:
                    raw=masks() if callable(masks) else masks
                    import torch
                    gpu={name:(value if isinstance(value,torch.Tensor) and value.is_cuda
                        else torch.as_tensor(value,device='cuda')) for name,value in raw.items()}
                    if mask_transform is None:
                        clean=self.robot_preparation.filter(job,gpu)
                    else:
                        from graph.plan_a_bc.robot_frame import RobotFilterFrame
                        # Correction can propose pixels outside the original masks.
                        # Classify all pixels once, preserving its full-screen evidence.
                        coverage={name:torch.stack((torch.ones_like(value[0]),torch.zeros_like(value[1]))) for name,value in gpu.items()}
                        diagnostics=self.robot_preparation.filter(job,coverage,diagnostics=True)
                        frame=RobotFilterFrame(self.system,job.cameras,diagnostics)
                        try:
                            raw=mask_transform(gpu,frame)
                            gpu={name:(value if isinstance(value,torch.Tensor) and value.is_cuda
                                else torch.as_tensor(value,device='cuda')) for name,value in raw.items()}
                            clean=frame.clean(gpu)
                        finally:frame.close()
                    self.last_raw_masks=gpu
                    return self._build_prepared(job.cameras,snapshot,job.surfaces,gpu,clean,depths=self.robot_preparation.depths)
                finally:self.robot_preparation.discard(job)
            raw=masks() if callable(masks) else masks
            if mask_transform is not None:raw=mask_transform(raw,None)
            self.last_robot_schedule='cpu_serial';self.last_raw_masks=raw
            return self._build_prepared(cameras,data,surfaces,raw)

    def _build_prepared(self,cameras,data,surfaces,masks,clean_masks=None,*,depths=None):
        resident = self.system.tensor_packet_builder is not None
        views=None
        if resident and clean_masks is not None:
            from graph.persistent_geometry.tensor_gate import prepare_tensor_gate
            prepared,views=prepare_tensor_gate(cameras,data,masks,clean_masks,self.system.protocol,
                coordinate_frame=self.coordinate_frame,depths=depths)
        else:
            prepared=(self.system.prepare(cameras,data,masks,surfaces) if clean_masks is None else
                self.system.prepare(cameras,data,masks,surfaces,clean_masks=clean_masks))
            if resident:
                from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
                views=prepare_tensor_views(cameras,prepared['conditions']['C3'],coordinate_frame=self.coordinate_frame)
        pools,_,_,_ = self.system.represent(cameras,prepared,'C3',('SF',),tensor_records=resident,views=views)
        records = pools['SF']
        if resident:
            from graph.persistent_geometry.tensor_records import budget_tensor
            records = budget_tensor(records,128)
        else:
            records = budget(records,128)
        graph = (self.system.graph(records,data,cameras,views=views) if resident else self.system.graph(records,data,cameras))
        self.last_wrist_gate_backend=prepared['wrist_gate_backend']
        graph.packet['wrist_gate_backend']=self.last_wrist_gate_backend
        return self._finish(graph,records,prepared['gate'],data)

    def _finish(self,graph,records,gate,data):
        t=int(data['timestep']);episode=str(data['episode'])
        from graph.persistent_geometry.tensor_records import TensorRecords
        if isinstance(records,TensorRecords):
            import torch
            if records.episode!=episode or records.timestep!=t:raise ValueError('noncurrent tensor records')
            if records.coordinate_frame!=self.coordinate_frame:raise ValueError('record/config coordinate frame mismatch')
            if self.coordinate_frame=='ee' and records.reference_ee_pose!=tuple(data['tcp_pose']):raise ValueError('record EE reference')
            owners=torch.repeat_interleave(records.object_id,records.source_offsets[1:]-records.source_offsets[:-1])
            if bool(((records.source_camera<0)|(records.source_camera>=len(records.camera_names))).any()):raise ValueError('foreign source camera')
            allowed=torch.tensor([name=='base_camera' for name in records.camera_names],device=records.device)
            if bool(((owners==1)&~allowed[records.source_camera]).any()):raise ValueError('Wrist Box source')
            if 'hand_camera' in records.camera_names and not gate['accepted'] and bool((records.source_camera==records.camera_names.index('hand_camera')).any()):
                raise ValueError('rejected Wrist source')
        else:
            for r in records:
                if not r['currently_observed'] or r['last_seen_timestep']!=t:
                    raise ValueError('noncurrent point')
                for e,ts,c,p in r['source_observation_ids']:
                    if e!=episode or ts!=t: raise ValueError('future/history/foreign source')
                    if r['object_id']==1 and c!='base_camera': raise ValueError('Wrist Box source')
                    if c=='hand_camera' and not gate['accepted']: raise ValueError('rejected Wrist source')
        graph.packet['perception_backend']=self.perception_backend
        graph.packet['view_consistency_backend']=('torch_cuda' if self.perception_backend=='cuda_resident'
            or self.system.tensor_packet_builder is not None else 'numpy_cpu')
        graph.packet['records_dedup_backend']=('torch_cuda' if self.system.tensor_packet_builder is not None
            else 'existing_cpu')
        graph.packet['final_fps_backend']=('torch_cuda' if self.system.tensor_packet_builder is not None
            else 'existing_cpu')
        graph.packet.setdefault('graph_backend','existing_cpu')
        if graph.packet.get('coordinate_frame') != FRAME_NAMES[self.coordinate_frame]:
            raise ValueError('Graph/config coordinate frame mismatch')
        validate_graph(graph)
        return graph,records,gate


def validate_graph(g):
    import torch
    n = len(g.node_type); e = g.edge_index.shape[1]
    if g.packet.get('coordinate_frame', 'EE_metres') not in FRAME_NAMES.values():
        raise ValueError('Graph coordinate frame')
    if g.node_input.shape!=(n,21) or g.edge_attr.shape!=(e,4) or g.edge_index.shape!=(2,e):
        raise ValueError('Graph21/Edge4 shape')
    if not torch.equal(g.node_type[:3],torch.arange(3,device=g.node_type.device)): raise ValueError('named robot nodes')
    if (g.node_input[:,14:]!=0).any(): raise ValueError('temporal extras')
    if g.edge_type.shape!=(e,) or (e and (g.edge_index.min()<0 or g.edge_index.max()>=n)):
        raise ValueError('invalid edge endpoints')
    if not all(torch.isfinite(a).all() for a in [g.node_input,g.edge_attr]): raise ValueError('NaN/Inf')
    if any(int((g.node_type==k).sum())>128 for k in (3,4)): raise ValueError('point cap')

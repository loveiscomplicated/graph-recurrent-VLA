"""Current sensor-only adapter to byte-pinned09-D geometry. No object GT API."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from graph.persistent_geometry.contracts import CameraObservation, synchronized
from graph.persistent_geometry.self_filter import LinkSurface
from graph.plan_a_geometry.core import PlanA, budget
from visual.runtime.vision_packet import pose_matrix
from .common import read, sha


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
    def __init__(self, assets, schema):
        self.fk = RobotFK(assets,schema)
        frozen = read(Path(__file__).resolve().parents[1]/'plan_a_geometry/protocol.json')
        self.system = PlanA('base_camera','hand_camera',frozen)
        self.mount = None

    def reset(self):
        self.mount = None

    def from_h5(self, f, t, episode):
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
        return cameras,data,surfaces

    def build(self,cameras,data,surfaces,masks):
        synchronized(cameras)
        t = int(data['timestep']); episode = str(data['episode'])
        wrist = next(c for c in cameras if c.camera=='hand_camera')
        # Fixed link-to-camera mount; current link FK is recomputed at every t.
        link = next(s for s in surfaces if s.link=='camera_link')
        mount = np.linalg.inv(link.world_from_link)@wrist.world_from_camera
        if self.mount is None: self.mount = mount
        np.testing.assert_allclose(mount,self.mount,atol=2e-5,rtol=0)
        prepared = self.system.prepare(cameras,data,masks,surfaces)
        pools,_,_,_ = self.system.represent(cameras,prepared,'C3',('SF',))
        records = budget(pools['SF'],128)
        for r in records:
            if not r['currently_observed'] or r['last_seen_timestep']!=t:
                raise ValueError('noncurrent point')
            for e,ts,c,p in r['source_observation_ids']:
                if e!=episode or ts!=t: raise ValueError('future/history/foreign source')
                if r['object_id']==1 and c!='base_camera': raise ValueError('Wrist Box source')
                if c=='hand_camera' and not prepared['gate']['accepted']: raise ValueError('rejected Wrist source')
        graph = self.system.graph(records,data,cameras)
        validate_graph(graph)
        return graph,records,prepared['gate']


def validate_graph(g):
    import torch
    n = len(g.node_type); e = g.edge_index.shape[1]
    if g.node_input.shape!=(n,21) or g.edge_attr.shape!=(e,4) or g.edge_index.shape!=(2,e):
        raise ValueError('Graph21/Edge4 shape')
    if not torch.equal(g.node_type[:3],torch.arange(3)): raise ValueError('named robot nodes')
    if (g.node_input[:,14:]!=0).any(): raise ValueError('temporal extras')
    if g.edge_type.shape!=(e,) or (e and (g.edge_index.min()<0 or g.edge_index.max()>=n)):
        raise ValueError('invalid edge endpoints')
    if not all(torch.isfinite(a).all() for a in [g.node_input,g.edge_attr]): raise ValueError('NaN/Inf')
    if any(int((g.node_type==k).sum())>128 for k in (3,4)): raise ValueError('point cap')

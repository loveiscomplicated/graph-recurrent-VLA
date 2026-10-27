"""Independent URDF forward kinematics for auditing observed robot link poses."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def urdf_fk(path, joint_names, qpos, root_pose):
    robot = ET.parse(path).getroot()
    values = dict(zip(joint_names, qpos))
    joints = []
    children = set()
    for joint in robot.findall('joint'):
        parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
        children.add(child)
        origin = joint.find('origin')
        matrix = np.eye(4)
        if origin is not None:
            matrix[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            matrix[:3,:3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        axis_tag = joint.find('axis')
        axis = np.fromstring(axis_tag.get('xyz','1 0 0') if axis_tag is not None else '1 0 0',sep=' ')
        kind = joint.get('type')
        motion = np.eye(4)
        if kind in ('revolute','continuous','prismatic'):
            if joint.get('name') not in values:
                raise ValueError('missing observed active joint '+joint.get('name'))
            value = values[joint.get('name')]
            if kind=='prismatic':
                motion[:3,3] = axis*value
            else:
                motion[:3,:3] = Rotation.from_rotvec(axis*value).as_matrix()
        elif kind!='fixed':
            raise ValueError('unsupported URDF joint '+str(kind))
        joints.append((parent,child,matrix@motion))
    roots = {link.get('name') for link in robot.findall('link')}-children
    if len(roots)!=1:
        raise ValueError('URDF must have one root')
    result = {next(iter(roots)):np.asarray(root_pose,float)}
    pending = list(joints)
    while pending:
        progressed = False
        for parent,child,matrix in pending[:]:
            if parent in result:
                result[child] = result[parent]@matrix
                pending.remove((parent,child,matrix))
                progressed=True
        if not progressed:
            raise ValueError('URDF disconnected/cyclic joints')
    return result

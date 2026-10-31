"""Nominal-validated physical checks; no TCP-height surrogate for collision."""
import xml.etree.ElementTree as ET
import itertools
import numpy as np
from graph.integration_baseline.geometry09a import physical_state,collision_meshes,convex_gap

class PhysicalSafety:
    def __init__(self,env,protocol):
        self.env=env;self.protocol=protocol;self.groups={};self.names=[]
        t=env.unwrapped
        for l in t.agent.robot.get_links():
            self.names.append(l.name)
            shapes=l._objs[0].get_collision_shapes()
            self.groups[l.name]=[s.get_collision_groups() for s in shapes]
        # PhysX excludes parent/child joints; fixed connected bodies form a
        # compound. Collision-group filtering additionally disables URDF pairs.
        parent={n:n for n in self.names}
        def find(n):
            while parent[n]!=n:n=parent[n]
            return n
        urdf=ET.parse(t.agent.controller.controllers['arm'].config.urdf_path)
        joints=[]
        for j in urdf.findall('joint'):
            a=j.find('parent').get('link');b=j.find('child').get('link')
            if a not in parent or b not in parent:continue
            joints.append((a,b))
            if j.get('type')=='fixed':parent[find(a)]=find(b)
        self.excluded=set()
        adjacent={frozenset((find(a),find(b))) for a,b in joints}
        for a,b in itertools.combinations(self.names,2):
            enabled=any(((ga[0]&gb[1]) or (gb[0]&ga[1])) and not (ga[2]&gb[2] and ga[3]==gb[3])
                for ga in self.groups[a] for gb in self.groups[b])
            if not enabled or find(a)==find(b) or frozenset((find(a),find(b))) in adjacent:self.excluded.add(frozenset((a,b)))
        self.limits=t.agent.robot.get_qlimits()[0].cpu().numpy()
    def evaluate(self,initial,substeps):
        s=physical_state(self.env);meshes=collision_meshes(self.env);tol=self.protocol['penetration_tolerance_m'];margin=self.protocol['moving_external_clearance_m']
        gaps={};violations=[]
        robot=[(k,v) for k,v in meshes.items() if k.startswith('robot:') and v]
        for name,values in robot:
            for obj in ['table-workspace','peg_0','box_with_hole_0']:
                gap=min((convex_gap(a,b) for a in values for b in meshes[obj]),default=10.)
                gaps[name+'|'+obj]=gap
                minimum=-tol if name=='robot:panda_link0' and obj=='table-workspace' else margin
                if gap<minimum:violations.append(dict(pair=[name,obj],gap_m=gap,minimum_m=minimum))
        self_gaps={}
        for (na,va),(nb,vb) in itertools.combinations(robot,2):
            if frozenset((na[6:],nb[6:])) in self.excluded:continue
            gap=min((convex_gap(a,b) for a in va for b in vb),default=10.)
            self_gaps[na+'|'+nb]=gap
            if gap < -tol:violations.append(dict(pair=[na,nb],gap_m=gap,minimum_m=-tol))
        contact_bad=[c for ss in substeps for c in ss if c['min_separation_m'] < -tol or c['max_impulse']>self.protocol['contact_impulse_threshold']]
        object_translation={k:float(np.linalg.norm(s[k][:3]-initial[k][:3])) for k in ['peg_pose','box_pose','hole_pose']}
        from recurrent.expert import rotvec_between
        object_rotation={k:float(np.linalg.norm(rotvec_between(initial[k][3:],s[k][3:]))) for k in object_translation}
        limit_bad=bool(np.any(s['qpos']<self.limits[:,0]-1e-5) or np.any(s['qpos']>self.limits[:,1]+1e-5))
        valid=bool(not violations and not contact_bad and not limit_bad and np.isfinite(s['qpos']).all() and np.isfinite(s['qvel']).all()
                   and abs(s['qvel']).max()<self.protocol['max_joint_velocity_rad_s'] and max(object_translation.values())<.002 and max(object_rotation.values())<.01)
        return dict(valid=valid,shape_violations=violations,substep_contact_violations=contact_bad,joint_limit_violation=limit_bad,
            robot_external_signed_gaps_m=gaps,enabled_self_signed_gaps_m=self_gaps,object_translation_m=object_translation,object_rotation_rad=object_rotation,
            tcp_z_m=float(s['ee_pose'][2]),max_qvel=float(abs(s['qvel']).max()))

"""Read-only secondary summaries from the completed traces, with parent as unit."""
import csv,json,math
from pathlib import Path
import numpy as np


def run(root):
    rows=list(csv.DictReader((root/'final_rollouts.csv').open()));out=[]
    for row in rows:
        tr=json.loads((root/row['folder']/'trace.json').read_text());controlled=[t for t in tr if t['control']=='B']
        if not controlled:continue
        first=controlled[0];ten=controlled[min(9,len(controlled)-1)];last=controlled[-1]
        lateral=lambda t:float(np.linalg.norm(t['peg_head_pos_at_hole'][1:]))
        pre=first['pre'];prelat=float(np.linalg.norm(pre['peg_head_pos_at_hole'][1:]))
        a=np.asarray(first['policy_action'][:3]);b=np.asarray(first['expert_action'][:3]);den=float(np.linalg.norm(a)*np.linalg.norm(b))
        find=lambda predicate:next((t['timestep'] for t in controlled if predicate(t)),None)
        radius=float(json.loads(row['hole_radius'])[0])
        out.append(dict(seed=int(row['seed']),condition=row['condition'],success=row['insertion']=='True',handoff=int(row['handoff_timestep']) if row['handoff_timestep'] else None,control_actions=len(controlled),
            pre_tip_error_m=float(np.linalg.norm(pre['peg_head_pos_at_hole'])),pre_lateral_error_m=prelat,pre_axis_angle_deg=pre['alignment_angle_deg'],hole_radius_m=radius,
            first_expert_policy_translation_cosine=float(a@b/den) if den>1e-12 else None,first_action_mae=first['action_difference_mae'],
            first_lateral_change_m=lateral(first)-prelat,first10_lateral_change_m=lateral(ten)-prelat,last_lateral_error_m=lateral(last),last_axis_angle_deg=last['alignment_angle_deg'],
            first_grasp_loss=find(lambda t:t['pre']['grasp'] and not t['grasp']),first_open_while_grasped=find(lambda t:t['pre']['grasp'] and t['action'][6]>0),
            first_geometry_missing=find(lambda t:any(t['missing_objects'])),first_arm_saturation=find(lambda t:t['arm_saturation']),
            first10_geometry_missing=any(any(t['missing_objects']) for t in controlled[:10]),first10_arm_saturation=any(t['arm_saturation'] for t in controlled[:10]),
            first10_lateral_outside_clearance=any(max(abs(v) for v in t['peg_head_pos_at_hole'][1:])>radius for t in controlled[:10]),last_lateral_outside_clearance=max(abs(v) for v in last['peg_head_pos_at_hole'][1:])>radius,
            missing_control_frames=sum(any(t['missing_objects']) for t in controlled)))
    path=root.parent/'independent_parent_diagnostics.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(out[0]));w.writeheader();w.writerows(out)
    for c in ('B0','H1','H2'):
        rr=[r for r in out if r['condition']==c]
        print(c,json.dumps(dict(parents=len(rr),success=sum(r['success'] for r in rr),mean_pre_lateral_mm=np.mean([r['pre_lateral_error_m'] for r in rr])*1000,first10_lateral_increase=sum(r['first10_lateral_change_m']>0 for r in rr),first10_outside_clearance=sum(r['first10_lateral_outside_clearance'] for r in rr),first10_missing=sum(r['first10_geometry_missing'] for r in rr),arm_saturation=sum(r['first_arm_saturation'] is not None for r in rr),grasp_loss=sum(r['first_grasp_loss'] is not None for r in rr),open_grasped=sum(r['first_open_while_grasped'] is not None for r in rr))))
    return out

if __name__=='__main__':
    run(Path(__file__).resolve().parents[2]/'outputs/task10h/final_results')

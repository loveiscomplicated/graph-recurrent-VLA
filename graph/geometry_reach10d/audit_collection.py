"""Full raw collection conditional signal and original-reset audits, no policy."""
import argparse
import collections
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,append_csv
from graph.integration_baseline.bc08_codec import controller_commands
from graph.integration_baseline.rollout_stop_clock import reset_diagnostics
from graph.geometry_reach10d.runtime import environment


def run(root):
    candidates=read(root/'collection_candidates.json');records={}
    for entry in candidates:
        p=root/'raw'/str(entry['spec']['episode_id'])/'receipt.json'
        if p.exists():records[entry['spec']['episode_id']]=dict(entry=entry,receipt=read(p))
    originals={e['receipt']['spec']['seed']:e for e in records.values() if e['entry']['data_kind']=='original'}
    comparisons=[];initial_groups=collections.defaultdict(list)
    for seed,value in originals.items():
        r=value['receipt']
        if r['status']!='ACCEPTED':continue
        with h5py.File(r['raw_path']) as f:robot=f['robot_state'][0]
        before=r['audit']['original']
        key=robot.tobytes()+np.asarray(before['box_pose'],np.float32).tobytes()+np.asarray(before['hole_pose'],np.float32).tobytes()
        initial_groups[hashlib.sha256(key).hexdigest()].append(seed)
    for value in records.values():
        if value['entry']['data_kind']!='perturbed':continue
        r=value['receipt'];seed=r['spec']['seed'];base=originals[seed]['receipt']
        if r['status']!='ACCEPTED' or base['status']!='ACCEPTED':continue
        with h5py.File(r['raw_path']) as a,h5py.File(base['raw_path']) as b:
            exact=bool(np.array_equal(a['robot_state'][0],b['robot_state'][0]))
            calibration=all(np.array_equal(a[c+'/'+k][0],b[c+'/'+k][0]) for c in ['base_camera','hand_camera']
                for k in ['intrinsic','extrinsic_cv','cam2world_gl'])
            aa,bb=a['actions'][:],b['actions'][:];n=min(len(aa),len(bb))
            delta=controller_commands(aa[:1])[0,:3]-controller_commands(bb[:1])[0,:3]
            target_delta=np.array(r['audit']['grasp_pose'])[:3]-np.array(base['audit']['grasp_pose'])[:3]
            row=dict(parent_seed=seed,episode_id=r['spec']['episode_id'],variant=r['spec']['variant'],
                robot_initial_exact=exact,camera_initial_exact=calibration,
                box_hole_exact=all(np.array_equal(r['audit']['original'][k],base['audit']['original'][k]) for k in ['box_pose','hole_pose']),
                actual_offset_mm=r['spec']['offset_mm'],target_shift_mm=(target_delta*1000).tolist(),
                expert_first_delta_mm=(delta*1000).tolist(),expert_first_delta_norm_mm=float(np.linalg.norm(delta)*1000),
                common_prefix_action_rms_mm=float(np.sqrt(np.mean(((aa[:n,:3]-bb[:n,:3])*100)**2))),
                nominal_length=len(bb),perturbed_length=len(aa),
                full_prefix_note='Robot trajectories diverge after t0; RMS is not a same-state estimate')
            comparisons.append(row)
    with (root/'conditional_expert_signal.csv').open('w',newline='') as f:
        import csv
        if comparisons:
            w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    write(root/'conditional_signal_audit.json',dict(cases=comparisons,
        exact_new_initial_state_cases=sum(r['robot_initial_exact'] and r['camera_initial_exact'] and r['box_hole_exact'] for r in comparisons),
        unique_perturbed_parents=len({r['parent_seed'] for r in comparisons}),
        current_original96_exact_robot_box_hole_pairs=sum(len(v)*(len(v)-1)//2 for v in initial_groups.values()),
        old_Train96_GT_audit='not inferred from current recollection; compare Task10C2 controlled observational matching separately'),immutable=True)
    torch.set_num_threads(1);env=environment(root);reset_rows=[]
    old={e['seed']:e for e in read(Path(read(root/'run_identity.json')['baseline'])/'dataset_audit.json')['episodes']}
    try:
        for seed in sorted(originals):
            env.reset(seed=seed);actual=reset_diagnostics(env)
            reset_rows.append(dict(seed=seed,actual=actual,original=old[seed]['reset_group'],exact=actual==old[seed]['reset_group']))
    finally:env.close()
    write(root/'original_reset_identity.json',dict(status='PASS' if all(r['exact'] for r in reset_rows) else 'FAIL',
        rows=reset_rows,sim_backend='physx_cpu',renderer='sapien_cuda' if read(root/'renderer_gpu_gate.json')['adopted'] else 'sapien_cpu',
        scope='fresh Train96 resets compared to original qpos/Peg/Box/Hole/halfsize reset hash; no Final seeds or student actions'),immutable=True)
    print('CONDITIONAL AUDIT',len(comparisons),'RESET',sum(r['exact'] for r in reset_rows),'/',len(reset_rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

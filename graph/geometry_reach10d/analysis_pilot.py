"""Policy-free pilot audit and current-runtime/original compatibility evidence."""
from pathlib import Path
import h5py
import numpy as np
from graph.plan_a_bc.common import read,write,sha,append_csv


def run(root):
    all_rows=read(root/'capture_manifest.json')['episodes'];comparisons=[]
    by_seed={}
    for r in all_rows:by_seed.setdefault(r['spec']['seed'],[]).append(r)
    for seed,rows in by_seed.items():
        nominal=next(r for r in rows if r['spec']['variant']=='P0')
        with h5py.File(nominal['raw_path']) as f:
            base=f['actions'][:];base_robot=f['robot_state'][:]
            base_rgb={c:f[c+'/rgb'][0] for c in ['base_camera','hand_camera']}
        for r in rows:
            with h5py.File(r['raw_path']) as f:
                action=f['actions'][:];robot=f['robot_state'][:];n=min(len(base),len(action))
                assert np.array_equal(base_robot[0],robot[0])
                comparisons.append(dict(seed=seed,variant=r['spec']['variant'],actions=len(action),
                    offset_mm=r['spec']['offset_mm'],target_delta_mm=(1000*(np.array(r['audit']['grasp_pose'])[:3]-np.array(nominal['audit']['grasp_pose'])[:3])).tolist(),
                    first_action_delta=(action[0]-base[0]).tolist(),
                    first_translation_delta_mm=float(np.linalg.norm(action[0,:3]-base[0,:3])*100),
                    prefix_translation_delta_rms_mm=float(np.sqrt(np.mean(((action[:n,:3]-base[:n,:3])*100)**2))),
                    independent_trajectory_comparison=True,exact_same_robot_t0=True,
                    raw_rgb_changed={c:bool(not np.array_equal(base_rgb[c],f[c+'/rgb'][0])) for c in base_rgb}))
    baseline=Path(read(root/'run_identity.json')['baseline']);old_entries=read(baseline/'dataset_audit.json')['episodes']
    compat=[]
    for seed,rows in by_seed.items():
        nominal=next(r for r in rows if r['spec']['variant']=='P0')
        old=next(e for e in old_entries if e['seed']==seed)
        with h5py.File(nominal['raw_path']) as a,h5py.File(baseline/'dataset'/old['path']) as b:
            keys=['robot_state','tcp_pose','robot_world_geometry']+[c+'/'+k for c in ['base_camera','hand_camera']
                for k in ['rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl']]
            for k in keys:
                aa,bb=a[k][0],b[k][0]
                compat.append(dict(seed=seed,key=k,exact=bool(np.array_equal(aa,bb)),max_abs=float(np.max(abs(aa.astype(float)-bb.astype(float))))))
    write(root/'pilot_data_verification.json',dict(status='PASS' if all(r['status']=='ACCEPTED' for r in all_rows) else 'FAIL',
        attempts=len(all_rows),accepted=sum(r['status']=='ACCEPTED' for r in all_rows),conditional_expert=comparisons,
        historical_compatibility=compat,
        original_reuse_decision='Recollect all original Train96 Reach in current runtime, for both A/B; preserve original cache.',
        no_policy_result_used=True,no_GT_student_inputs=True))
    return comparisons

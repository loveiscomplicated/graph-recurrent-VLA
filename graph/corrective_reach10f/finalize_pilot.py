"""Aggregate all physical/raw/cache/history contracts before full collection."""
import argparse,csv
from pathlib import Path
import numpy as np
import sapien
from types import SimpleNamespace
from graph.plan_a_bc.common import read,write,append_csv
from graph.corrective_reach10f.recovery_contracts import verify
from recurrent.expert import rotvec_between,ScriptedExpert
from graph.integration_baseline.bc08_codec import controller_commands

def run(root,pilot,v4=False):
    rows=read(pilot/'recovery_pilot_manifest.json');assert len(rows)==40
    assert not (pilot/'failure.json').exists();assert all(r['approved'] for r in rows)
    raw_contracts=root/('matched_hold_v4_raw_contracts.json' if v4 else 'pilot_raw_contracts.json')
    if not raw_contracts.exists():verify(pilot,raw_contracts)
    assert read(raw_contracts)['status']=='PASS' and read(raw_contracts)['pairs']==40
    if v4:
        hold=read(root/'matched_hold_v4_exact_audit.json')
        assert hold['branches']==40 and hold['p_all_array_exact'] and hold['n_hold_all_under_1mm'] and hold['n_mask_all']
    write_metrics=not (root/'feedback_disturbance_metrics.csv').exists()
    for r in rows:
        folder=pilot/'raw'/r['branch'];assert read(folder/'N/receipt.json')['approved']
        trace=read(folder/'P/trace.json');ds=[s for s in trace if s['segment']==1]
        initial=np.array(ds[0]['tcp_before']);target=initial[:3]+np.array(r['offset_mm'])*.001
        for s in ds[-3:]:
            assert np.linalg.norm(np.asarray(s['tcp_after'][:3])-target)<.001
            assert np.linalg.norm(rotvec_between(initial[3:],np.asarray(s['tcp_after'][3:])))<.01
            assert np.max(abs(np.asarray(s['qvel_after'])))<.03
        for tag in ['N','P']:
            fulltrace=read(folder/tag/'trace.json')
            receipt=read(folder/tag/'receipt.json');grasp=receipt['target_refresh']['grasp']
            final=sapien.Pose(grasp[:3],grasp[3:]);phase='reach'
            for s in [t for t in fulltrace if t['segment']==2]:
                pose=sapien.Pose(s['tcp_before'][:3],s['tcp_before'][3:]);goal=final*sapien.Pose([0,0,-.05]) if phase=='reach' else final
                source=SimpleNamespace(_tcp=lambda:pose,gain=3.,max_rot=.15)
                expected_action=np.r_[ScriptedExpert._servo(source,goal,.02 if phase=='reach' else .01),1].astype(np.float32)
                np.testing.assert_allclose(expected_action,s['action'],atol=1e-6,rtol=1e-6)
                dp=np.linalg.norm(goal.p-pose.p);dr=np.linalg.norm(rotvec_between(pose.q,goal.q))
                if phase=='reach' and dp<.01 and dr<.05:phase='descend'
                elif phase=='descend' and dp<.004 and dr<.03:phase='close'
            assert phase=='close','ordered approach/descend contract'
            expected=controller_commands(np.array([s['action'] for s in fulltrace],np.float32))
            np.testing.assert_allclose(expected[:,:6],np.array([s['scaled_action'] for s in fulltrace]),atol=2e-8,rtol=1e-6)
            for s in fulltrace:
                assert len(s['substep_contacts'])==5 and s['physical_safety']['valid']
                xyz=np.array(s['tcp_after'][:3]);assert abs(xyz[:2]).max()<.8 and xyz[2]<1
        if write_metrics:append_csv(root/'feedback_disturbance_metrics.csv',dict(branch=r['branch'],parent=r['seed'],anchor=r['anchor'],offset_mm=r['offset_mm'],actual_offset_mm=r['actual_offset_mm'],steps=r['disturbance_steps'],residual_m=float(np.linalg.norm(np.array(r['actual_offset_mm'])-r['offset_mm'])*.001),physical_safety=True,pose_recovery=True))
    for p in pilot.glob('replay_*.json'):
        if p.name.endswith('.tracking.json'):continue
        a=read(p);assert a['physical_max']<=1e-6 and a['action_max']<=1e-6
    assert read(root/'cache_gate/corrective_cache_gate.json')['status']=='PASS'
    assert read(root/'cache_gate/causal_prefix_identity.json')['status']=='PASS'
    hs=read(root/'history_smoke02/verification.json');assert all(v=='PASS' for k,v in hs['checks'].items() if k!='formal_AB_budget')
    assert read(root/'initial_checkpoint_identity.json')['strict_readback']=='PASS'
    status=dict(status='PASS',scope='40P+40matchedN physical pilot;2episode CUDA causal graph gate;actualCUDA history/masked-loss/save-resume smoke. Formal A/B not executed.',
        source_pilot_uuid=read(pilot/'identity.json')['run_id'],checks={k:'PASS' for k in ['parent_snapshot_restore','same_anchor_replay','requested_actual_offset','controller_scaling','physical_collision_joint_limit','nominal_safety_validation','expert_target_refresh','recovery_action_alignment','ordered_reach_phases','supervision_mask','observation_action_timestep','causal_vision_prefix','gru_history','cache_hash_readback','initial_checkpoint_readback']},
        formal_AB='NOT_TESTED',closed_loop='NOT_TESTED',original_data_source_protection=read(root/'original_raw_protection.json')['status'])
    destination=root/('matched_hold_v4_completion_gate.json' if v4 else 'pilot_completion_gate_extended.json' if (root/'pilot_completion_gate.json').exists() else 'pilot_completion_gate.json')
    write(destination,status,immutable=True);print('ALL PILOT REQUIRED GATES PASS40P/40N',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--v4',action='store_true');a=p.parse_args();run(a.root,a.pilot,a.v4)

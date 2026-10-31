"""Readback contracts on actual completed v2 N/P pairs; no fabricated fixtures."""
import argparse
from pathlib import Path
import h5py,numpy as np
from graph.plan_a_bc.common import read,write,sha

def verify(root,output):
    rows=read(root/'recovery_pilot_manifest.json');checks=[]
    for r in rows:
        folder=root/'raw'/r['branch'];a=r['anchor'];d=r['disturbance_steps']
        with h5py.File(folder/'P/episode.h5') as p,h5py.File(folder/'N/episode.h5') as n:
            assert sha(folder/'P/episode.h5')==r['sha256']
            for f in [p,n]:
                T=len(f['actions']);assert len(f['robot_state'])==T+1 and f['robot_state'].shape[1]==76
                np.testing.assert_array_equal(f['timestep'][:],np.arange(T+1));np.testing.assert_array_equal(f['timestamp'][:],np.arange(T+1)/20)
                expected=np.arange(T)>=a+d;np.testing.assert_array_equal(f['supervision_mask'][:],expected)
                assert not f['actions'][a:a+d].any(),'disturbance leaked to expert labels'
                assert (f['executed_action'][:,6]==1).all();assert (f['robot_state'][:,22]>=.065).all()
                keys=[];f.visititems(lambda k,v:keys.append(k) if isinstance(v,h5py.Dataset) else None)
                assert not any(t in k.lower() for k in keys for t in ['peg_pose','target_pose','expert_phase','hidden','offset'])
                for k in keys:assert np.isfinite(f[k][:]).all()
            for k in ['robot_state','actions','executed_action','base_camera/rgb','base_camera/depth_raw_mm','hand_camera/rgb','hand_camera/depth_raw_mm']:
                np.testing.assert_array_equal(p[k][:a],n[k][:a],err_msg=k)
            pp=p['tcp_pose'][a+d];np_=n['tcp_pose'][a+d];delta=pp[:3]-np_[:3]
            nr=read(folder/'N/receipt.json');assert sha(folder/'N/episode.h5')==nr['sha256']
            pt=[t for t in read(folder/'P/trace.json') if t['segment']==2];nt=[t for t in read(folder/'N/trace.json') if t['segment']==2]
            command_delta=np.array(pt[0]['scaled_action'][:3])-np.array(nt[0]['scaled_action'][:3]);reference=-delta
            mag=np.linalg.norm(command_delta);ref=np.linalg.norm(reference)
            # Signed gain reference is measured P-N robot displacement at the
            # matched handoff, not nominal Peg geometry or requested offset.
            checks.append(dict(branch=r['branch'],raw_sha_readback='PASS',same_prefix='PASS',time_mask_alignment='PASS',no_gt_future_hidden='PASS',
                P_minus_N_handoff_translation_m=delta.tolist(),first_root_frame_command_delta_m=command_delta.tolist(),
                direction_cosine=float(command_delta@reference/(mag*ref)) if mag>1e-8 and ref>.0001 else None,
                signed_gain=float(command_delta@reference/(reference@reference)) if ref>.0001 else None,
                causal_vision_gru='NOT_TESTED by raw contracts'))
    write(output,dict(status='PASS',pairs=len(checks),checks=checks),immutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.root,a.output)

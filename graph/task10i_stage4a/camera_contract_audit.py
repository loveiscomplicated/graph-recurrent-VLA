"""Recorded timestamp/calibration/fixed Wrist-to-TCP mount audit, zero forwards."""
from pathlib import Path
import argparse
import time
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from .run_pipeline import CAMERAS,STAGE3,read,write


def pose_matrices(pose):
    matrices=np.broadcast_to(np.eye(4),(len(pose),4,4)).copy()
    matrices[:,:3,3]=pose[:,:3]
    matrices[:,:3,:3]=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    return matrices


def run(output,final):
    from utils.experiment_logger import ExperimentLogger
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_recorded_camera_mount_audit',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(model_forwards=0,read_only_recorded_inputs=True)) as log:
        write(output/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_INDEPENDENT_EVALUATION',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        protocol=read(Path('/workspace/task10i_stage4a_completion_20261101_v1/run/baseline/protocol.json'))
        entries=read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()
        rows=[];gl_from_cv=np.diag([1.,-1.,-1.,1.])
        for partition in ('development','internal_test','independent_validation'):
            for e in sorted((e for e in entries if e['parent_id'] in protocol[partition]),key=lambda e:str(e['episode_id'])):
                with h5py.File(e['raw_path'],'r') as f:
                    timestamps=f['timestamp'][:];n=len(timestamps)
                    tcp=pose_matrices(f['tcp_pose'][:]);world={};intrinsic_error={};inverse_error={}
                    for camera in CAMERAS:
                        k=f[camera+'/intrinsic'][:];c2w=f[camera+'/cam2world_gl'][:]
                        extrinsic=f[camera+'/extrinsic_cv'][:]
                        extrinsic4=np.broadcast_to(np.eye(4),(n,4,4)).copy();extrinsic4[:,:3,:]=extrinsic[:,:3,:]
                        world[camera]=c2w@gl_from_cv
                        intrinsic_error[camera]=float(np.max(abs(k-k[:1])))
                        inverse_error[camera]=float(np.max(abs(extrinsic4@world[camera]-np.eye(4))))
                    mount=np.linalg.inv(tcp)@world['hand_camera']
                    main_drift=float(np.max(abs(world['base_camera']-world['base_camera'][:1])))
                    mount_drift=float(np.max(abs(mount-mount[:1])))
                    error=float(np.max(abs(timestamps-np.arange(n)/20)))
                    row=dict(partition=partition,episode=str(e['episode_id']),parent_id=e['parent_id'],frames=n,
                        timestamp_error_to_registered_20hz_s=error,strictly_increasing=bool(np.all(np.diff(timestamps)>0)),
                        intrinsic_drift=intrinsic_error,inverse_calibration_error=inverse_error,
                        main_camera_transform_drift=main_drift,wrist_to_observed_tcp_mount_drift=mount_drift)
                    assert row['strictly_increasing'] and error<1e-5,row
                    assert max(intrinsic_error.values())<1e-6 and max(inverse_error.values())<2e-5,row
                    assert main_drift<2e-5 and mount_drift<2e-5,row
                    rows.append(row)
        write(output/'episode_contracts.json',rows)
        write(output/'scope.json',dict(status='RECORDED_CAMERA_CONTRACT_PASS',run_id=log.run_id,episodes=len(rows),frames=sum(r['frames'] for r in rows),
            timestamps_are_simulated_registered_clock_not_wall_latency=True,not_real_hardware_sync_validation=True,
            observed_TCP_reference='fixed hand-camera and TCP frames share the hand rigid link; no object GT needed',
            wrist_mount_max_drift=max(r['wrist_to_observed_tcp_mount_drift'] for r in rows),
            inverse_calibration_max_error=max(max(r['inverse_calibration_error'].values()) for r in rows),
            student_inputs_unchanged=True))
        log.link_output(output/'scope.json',kind='metrics')
        write(output/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='RECORDED_CAMERA_CONTRACT_PASS',run_id=log.run_id))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('final',type=Path)
    a=p.parse_args();run(a.output,a.final)

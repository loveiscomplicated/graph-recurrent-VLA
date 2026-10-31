"""Read-only initial RGB-D replay with original Stage1 factory; no actions/models."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(WANDB_MODE='disabled',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import h5py
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(x):return x.detach().cpu().numpy().copy()

def main(root,output):
    original=Path('/workspace/task10i_stage1_20261031')
    spec=importlib.util.spec_from_file_location('original_stage1_capture',original/'code/pipeline.py')
    stage1=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage1)
    from utils.experiment_logger import ExperimentLogger
    from recurrent.common import versions
    required=set()
    for n in ('dataset_R_train_manifest.json','dataset_D_train_manifest.json','shared_validation_manifest.json'):
        required.update(p for p in read(root/n)['parents'] if p.startswith('10i_s1_'))
    parents=sorted((root/'parents').glob('*/parent_result.json'))
    with ExperimentLogger(team='graph',model='Task10I_Stage3_independent_initial_RGBD_replay',seed=42,job_type='prepare',
        root=output.parent/'native_replay_experiments',config=dict(collection_root=str(root),actions_executed=0,training=False)) as log:
        log.record(source_run_id=read(root/'run_identity.json')['run_id'])
        env=stage1.init_env();rows=[]
        try:
            for path in parents:
                result=read(path);cid=result['candidate_id'];seed=result['seed'];folder=path.parent
                actual,_,_=stage1.capture(env,seed)
                with np.load(original/'verified_pool_v2/candidates'/cid/'initial.npz',allow_pickle=False) as expected:
                    for k in expected.files:np.testing.assert_array_equal(actual[k],expected[k],err_msg=cid+'/'+k)
                obs=env.unwrapped.get_obs();raw=folder/'full_task/episode.h5'
                if not raw.exists():raw=folder/'full_task.partial/episode.h5'
                assert raw.exists() or cid not in required
                cameras=[]
                if raw.exists():
                    with h5py.File(raw) as f:
                        for camera in ('base_camera','hand_camera'):
                            values=dict(rgb=array(obs['sensor_data'][camera]['rgb'])[0],
                                depth_raw_mm=array(obs['sensor_data'][camera]['depth'])[0,...,0],
                                intrinsic=array(obs['sensor_param'][camera]['intrinsic_cv'])[0],
                                extrinsic_cv=array(obs['sensor_param'][camera]['extrinsic_cv'])[0],
                                cam2world_gl=array(obs['sensor_param'][camera]['cam2world_gl'])[0])
                            for k,v in values.items():np.testing.assert_array_equal(v,f[camera+'/'+k][0],err_msg=cid+'/'+camera+'/'+k)
                            cameras.append(camera)
                assert int(env.unwrapped.elapsed_steps.item())==0
                rows.append(dict(parent_id=cid,seed=seed,final_dataset=cid in required,canonical_Stage1_saved_fields_exact=True,
                    stage3_initial_camera_replay_exact=cameras,actions_executed=0))
                log.log(dict(parents_checked=len(rows)),step=len(rows))
        finally:env.close()
        assert required<={r['parent_id'] for r in rows if len(r['stage3_initial_camera_replay_exact'])==2}
        data=dict(status='PASS',run_id=log.run_id,source_run_id=read(root/'run_identity.json')['run_id'],
            checked_parents=len(rows),final_new_parents=len(required),all_final_new_main_wrist_RGBD_calibration_exact=True,
            canonical_Stage1_saved_fields_exact=True,stage1_historical_wrist_reference='NOT_STORED; comparison is independent native factory replay against Stage3 T0, not a historical Wrist file comparison',
            factory_source=str(original/'code/pipeline.py'),factory_source_sha256=sha(original/'code/pipeline.py'),
            runtime=versions(),actions_executed=0,policy_perception_graph_executed=False,rows=rows,source_sha256=sha(Path(__file__)))
        output.write_text(json.dumps(data,indent=2)+'\n');log.link_output(output,kind='reports')
        print(json.dumps({k:v for k,v in data.items() if k!='rows'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);main(a.root,a.output)

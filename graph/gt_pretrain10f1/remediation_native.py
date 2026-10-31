"""Native F1 recaching of sealed corrective observations, overlapping transfer.
GT enters only the isolated label audit, never the inherited graph writer.
"""
import argparse,gc,inspect,shutil,time
from pathlib import Path
import h5py,numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,now
from graph.plan_a_bc.cache import make_vision,input_identity
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode
from graph.geometry_reach10d.optimizations import DetectionMemo
from recurrent.common import versions
from recurrent.features import build_features
from envs.maniskill import get_gt_state
from utils.experiment_logger import ExperimentLogger
from graph.gt_pretrain10f1.contract import measured_robot,object_target,split_fields,assemble_gt38


def audit_label(root,ep,schema):
    with h5py.File(root/ep['raw_path']) as f,np.load(root/ep['gt_path']) as gt,np.load(root/ep['mask_path']) as mask:
        n=ep['actions'];robot=measured_robot(f['robot_state'][:],schema)
        assert f['robot_state'].shape==(n+1,76)
        for k in ['timestep','timestamp']:np.testing.assert_array_equal(gt[k],f[k][:])
        raw={k:gt['raw_'+k] for k in ['tcp_pose','left_tip','right_tip','gripper','peg_pose','hole_pose','peg_half','hole_radius']}
        for k,v in robot.items():np.testing.assert_allclose(v,raw[k],atol=1e-7,rtol=0)
        assembled=assemble_gt38(split_fields(object_target(raw)),robot)
        official=build_features(raw,np.ones(n+1,'f'),np.zeros(n+1,'f'))
        np.testing.assert_allclose(gt['gt38'],official,atol=1e-7,rtol=0)
        np.testing.assert_allclose(assembled,official,atol=1e-7,rtol=0)
        assert mask['gnn_supervision_mask'].shape==(n,) and mask['physical_valid'].all() and mask['gnn_supervision_mask'].all()
        return dict(capture_id=ep['capture_id'],status='PASS',frames=n,oracle_max_abs=float(np.max(abs(assembled-official))),source_gt38_max_abs=float(np.max(abs(gt['gt38']-official))))


def run(root,native,probe,dino,sam):
    torch.set_num_threads(1);torch.manual_seed(42)
    while not (root/'native_input_manifest.json').exists():time.sleep(5)
    manifest=read(root/'native_input_manifest.json');assert len(manifest['episodes'])==212
    actual=versions();assert actual==read(native/'environment_worker0.json')['versions']
    schema=read(root/'normalizer.json')['schema'];label_rows=[]
    deps=['graph/plan_a_geometry/core.py','graph/plan_a_geometry/protocol.json','graph/plan_a_bc/geometry.py','graph/persistent_geometry/graph.py',
        'graph/persistent_geometry/self_filter.py','graph/integration_baseline/online_vision_adapter.py','visual/runtime/tracking.py','visual/runtime/grounding.py','recurrent/features.py']
    runtime=dict(status='PASS',actual_versions=actual,backend_role='CUDA vision + CPU geometry; no simulator executed',
        native_environment_sha256=sha(native/'environment_worker0.json'),graph_config=read(root/'config.json')['geometry'],
        geometry_protocol_sha256=sha(native/'source/graph/plan_a_geometry/protocol.json'),robot_schema=schema,config_sha256=sha(root/'config.json'),
        source_sha256={k:sha(native/'source'/k) for k in deps},gt_accessor_source_sha256=sha(Path(inspect.getfile(get_gt_state))),
        gt38_encoder_sha256=sha(native/'source/recurrent/features.py'),geometry_workers=8,
        vision_weight_sha256=dict(dino_files={p.name:sha(p) for p in dino.iterdir() if p.is_file()},sam_sha256=sha(sam)),policy_training=False,old576_graphs_included=False)
    assert runtime['gt38_encoder_sha256']=='80de6fd742be4e7c9a3477cde0419e803b72253c09d1227fc0fec23b965b0761'
    write(root/'native_runtime.json',runtime,immutable=True)
    filemap={x['path']:x for x in manifest['files']};pending=list(manifest['episodes']);done=[];detector=tracker=memo=None
    with ExperimentLogger(team='graph',model='Task10F1_all_corrective_graphs_native_runtime',root=root/'experiments',job_type='prepare',
        config=dict(input_manifest_sha256=sha(root/'native_input_manifest.json'),native_runtime_sha256=sha(root/'native_runtime.json'),prior_parity='FAIL4/6; no threshold change')) as log,make_pool(root,8) as pool:
        while pending:
            progress=False
            for ep in list(pending):
                names=[ep['raw_path'],ep['gt_path'],ep['mask_path'],f'raw/{ep["capture_id"]}/receipt.json']
                if not all((root/n).is_file() and (root/n).stat().st_size==filemap[n]['bytes'] for n in names):continue
                for n in names:assert sha(root/n)==filemap[n]['sha256'],n
                label_rows.append(audit_label(root,ep,schema));eid=ep['sampling_episode_id'];folder=root/'cache'/f'ep{eid}';raw=root/ep['raw_path'];prior=probe/'cache'/f'ep{eid}'
                if not (folder/'receipt.json').exists() and (prior/'receipt.json').exists():
                    r=read(prior/'receipt.json');assert r['raw_episode_sha256']==ep['raw_sha256'] and sha(prior/'graphs.npz')==r['sha256']
                    folder.mkdir(parents=True,exist_ok=False)
                    for name in ['graphs.npz','gate_trace.json']:shutil.copy2(prior/name,folder/name);assert sha(prior/name)==sha(folder/name)
                    r.update(identity=input_identity(root),path=str((folder/'graphs.npz').relative_to(root)),native_probe_reuse=str(prior))
                    write(folder/'receipt.json',r,immutable=True)
                if detector is None:
                    detector,tracker=make_vision('cuda',dino,sam);memo=DetectionMemo(detector,'F1_native_unified_corrective',runtime['vision_weight_sha256'])
                entry=dict(episode_id=eid,seed=ep['parent'],split=ep['split'],actions=ep['actions'],path=str(raw),sha256=ep['raw_sha256'])
                r=cache_episode(root,Path('/'),entry,memo,tracker,log,pool,8)
                done.append(dict(capture_id=ep['capture_id'],sampling_episode_id=eid,raw_sha256=ep['raw_sha256'],graph_path=r['path'],graph_sha256=r['sha256'],
                    receipt_path=str((folder/'receipt.json').relative_to(root)),receipt_sha256=sha(folder/'receipt.json'),gate_path=str((folder/'gate_trace.json').relative_to(root)),
                    gate_sha256=sha(folder/'gate_trace.json'),actions=ep['actions'],readback='PASS',native_probe_reused='native_probe_reuse' in r))
                pending.remove(ep);progress=True
                write(root/'native_progress.json',dict(completed=len(done),total=212,remaining_transfer_or_cache=len(pending),at=now()))
                memo.cache.clear();gc.collect();torch.cuda.empty_cache();print('NATIVE_COMPLETE',len(done),'/212',ep['capture_id'],flush=True)
            if not progress:time.sleep(5)
        for row in manifest['files']:
            p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        write(root/'native_label_contract.json',dict(status='PASS',episodes=label_rows),immutable=True)
        write(root/'native_graph_manifest.json',dict(status='PASS',episodes=done,native_runtime_sha256=sha(root/'native_runtime.json'),
            native_label_contract_sha256=sha(root/'native_label_contract.json'),input_manifest_sha256=sha(root/'native_input_manifest.json'),
            graph_generation_runtime='F1 native runtime for every212 graph; all source576 graph tensors excluded',at=now()),immutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    a=p.parse_args();run(a.root,a.native,a.probe,a.dino,a.sam)

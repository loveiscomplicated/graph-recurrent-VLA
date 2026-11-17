"""Recompute Object21 for the original Mamba's frame-aligned Expert data."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from frontend import ROOT, GNN, TRACKER
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train import predict_coordinates
from train_finetune import prepare_tensors
from train_gt20 import predict
from tracker import CornerTransformer
from utils.experiment_logger import ExperimentLogger, write

OLD = Path('/workspace/recurrent_new_frontend')
POS_TOL_MM = .05
ANGLE_TOL_DEG = .05


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:
        return dict(f)


def source_data():
    shards, sources = [], []
    for folder in sorted(ROOT.glob('shard_*')):
        if not folder.is_dir():
            continue
        path=folder/'data.npz'
        meta=json.loads((folder/'summary.json').read_text())
        digest=file_sha256(path)
        assert digest==meta['sha256']
        take=read_npz(path)
        active=(take['seed']<3100144)&(take['phase']!='success')
        if active.any():
            shards.append({k:a[active] for k,a in take.items()})
            sources.append(dict(run_id=meta['run_id'],sha256=digest))
    return {key:np.concatenate([part[key] for part in shards]) for key in shards[0]},sources


def angle(a,b):
    a,b=a.astype(float),b.astype(float)
    cos=np.abs((a*b).sum(axis=1))/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    return float(np.rad2deg(2*np.arccos(np.clip(cos,0,1))).max())


def align(demos, data, out):
    gt=demos['state36_gt']
    graph=gt.copy()
    drift=np.zeros(2)
    for seed,a,b in zip(demos['reset_seed'],demos['episode_offsets'][:-1],
                        demos['episode_offsets'][1:],strict=True):
        mask=data['seed']==seed
        assert mask.sum()==b-a
        np.testing.assert_array_equal(data['phase'][mask],demos['phase'][a:b])
        np.testing.assert_array_equal(data['step'][mask],np.arange(b-a))
        a_gt,b_gt=gt[a:b,15:],data['target'][mask]
        pos=max(np.linalg.norm(gt[a:b,:3]-data['tcp'][mask],axis=1).max(),
                *(np.linalg.norm(a_gt[:,sl]-b_gt[:,sl],axis=1).max()
                  for sl in (slice(0,3),slice(7,10),slice(10,13))))*1000
        orientation=max(angle(a_gt[:,3:7],b_gt[:,3:7]),angle(a_gt[:,13:17],b_gt[:,13:17]))
        drift=np.maximum(drift,[pos,orientation])
        np.testing.assert_allclose(a_gt[:,17:],b_gt[:,17:],atol=1e-6,rtol=0)
        graph[a:b,15:]=out[mask]
    assert drift[0]<=POS_TOL_MM and drift[1]<=ANGLE_TOL_DEG
    return graph,drift


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    data,sources=source_data()
    args.output.mkdir(parents=True,exist_ok=False)
    train_state=torch.load(TRACKER,map_location='cpu',weights_only=True)
    gnn_state=torch.load(GNN,map_location='cpu',weights_only=True)
    assert gnn_state['config']['tracker_sha256']==file_sha256(TRACKER)
    tracker=CornerTransformer(spatial=True).to(device).eval()
    tracker.load_state_dict(train_state['model'],strict=True)
    gnn=VertexEdgeGatedMPNN(**gnn_state['config']['gnn']).to(device).eval()
    gnn.load_state_dict(gnn_state['model'],strict=True)
    _,points=predict_coordinates(tracker,data,device,train_state['fallback'].numpy())
    tensors=prepare_tensors(dict(data,xyz=points),device)
    predictions=predict(gnn,tensors,torch.arange(len(points),device=device),96).numpy()
    for folder in sorted(OLD.glob('shard_*')):
        if not folder.is_dir():
            continue
        source=folder/'demos.npz'
        summary=json.loads((folder/'summary.json').read_text())
        check=file_sha256(source)
        assert check==summary['sha256']
        demos=read_npz(source)
        estimated,drift=align(demos,data,predictions)
        demos['state36_transformer']=estimated
        dest=args.output/folder.name
        dest.mkdir()
        config=dict(original_sha256=check,original_run_id=summary['run_id'],point_sources=sources,
                    tracker_sha256=file_sha256(TRACKER),gnn_sha256=file_sha256(GNN),
                    script_sha256=file_sha256(__file__),max_motion_drift_mm=float(drift[0]),
                    max_pose_drift_deg=float(drift[1]))
        with ExperimentLogger(team='recurrent',model='full_insertion_mamba_demos',
                              job_type='preprocess',seed=int(demos['reset_seed'][0]),config=config,
                              root=dest/'experiments') as log:
            out=dest/'demos.npz'
            np.savez_compressed(out,**demos)
            info=dest/'summary.json'
            write(info,dict(run_id=log.run_id,config=config,sha256=file_sha256(out),frames=len(estimated)))
            log.link_output(out)
            log.link_output(info)
            log.log({'frames':len(estimated),'episodes':len(demos['reset_seed'])})
        print(folder.name,len(estimated),drift.tolist(),flush=True)
    print('Completed synchronized Expert preprocessing',flush=True)


if __name__=='__main__':
    main()

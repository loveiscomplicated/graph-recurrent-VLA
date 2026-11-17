"""Fine-tune the Transformer and GNN on the existing GT-free corner-ID observations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from frontend import TRACKER, GNN
from graph.gated_mpnn import VertexEdgeGatedMPNN
from match import CornerMatcher
import match
import box_matcher
from train import predict_coordinates
from train_finetune import prepare_tensors, train as fit_gnn
from train_gt20 import error_metrics, describe, predict
from train_tracker import CornerTransformer
import train_until_plateau as tracker_train
from utils.experiment_logger import ExperimentLogger, write

VAL=3100120
TEST=3100208
RAGGED=('raw_xyz','raw_family')
MODEL_KEYS=('gnn','loss','training')
REGIONS={'peg':slice(0,8),'box':slice(8,16),'hole':slice(16,20),'all':slice(0,20)}


def read(path):
    return json.loads(path.read_text())


def match_frames(data):
    points,labels,tcp=(data[k] for k in ('raw_xyz','raw_family','tcp'))
    counts=data['raw_count']
    indices=np.r_[0,np.cumsum(counts)]
    assert indices[-1]==len(points)==len(labels)
    xyz=np.zeros((len(counts),20,3),dtype=np.float32)
    seen=np.zeros((len(counts),20),dtype=bool)
    for row,step in enumerate(data['step']):
        if step==0:
            matcher=CornerMatcher()
        lo,hi=indices[row:row+2]
        raw=points[lo:hi]
        assignments=matcher.step(raw,labels[lo:hi],tcp[row])
        found=assignments>=0
        seen[row]=found
        xyz[row,found]=raw[assignments[found]]
    return xyz,seen


def load(root):
    sources,parts,episodes=[],[],[]
    for folder in sorted(root.glob('shard_*')):
        if not folder.is_dir():
            continue
        summary=read(folder/'summary.json')
        path=folder/'raw.npz'
        assert file_sha256(path)==summary['sha256']
        meta=read(folder/'experiments'/summary['run_id']/'metadata.json')
        assert meta['status']=='finished' and not meta['git_dirty']
        assert summary['config']['script_sha256']==file_sha256(Path(__file__).with_name('collect_raw.py'))
        with np.load(path,allow_pickle=False) as p:
            captured=dict(p)
        detected,observed=match_frames(captured)
        captured.update(detected_xyz=detected,observed=observed)
        selected=np.zeros(len(captured['seed']),dtype=bool)
        for trial in summary['episodes']:
            seed=trial['seed']
            rows=captured['seed']==seed
            np.testing.assert_array_equal(captured['step'][rows],np.arange(trial['frames']))
            if trial['success']:
                assert captured['phase'][rows][-1]=='success'
                assert (captured['depth_mm'][rows]>20).any()
            if trial['success'] or seed>=TEST:
                selected|=rows
            episodes.append(trial)
        parts.append({k:v[selected] for k,v in captured.items() if k not in RAGGED})
        sources.append(dict(run_id=summary['run_id'],sha256=summary['sha256']))
    assert len(sources)==10 and len(episodes)==168
    data={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    ids=np.unique(data['seed'])
    groups=(ids[ids<VAL],ids[(ids>=VAL)&(ids<TEST)],ids[ids>=TEST])
    assert len(groups[0])>=100 and len(groups[1])>=15 and len(groups[2])==24
    return data,groups,episodes,sources


def heldout(out,data,old,new,old_gnn,new_gnn,device):
    tmask=data['seed']>=TEST
    keys=('seed','step','target','xyz','detected_xyz','observed','depth_mm','phase','tcp')
    held={k:data[k][tmask] for k in keys}
    gt=held['xyz']
    observed=held['observed']
    old_raw,old_corrected=(x[tmask] for x in old)
    new_raw,new_corrected=(x[tmask] for x in new)
    for coords in (old_raw,new_raw,old_corrected,new_corrected):
        np.testing.assert_array_equal(coords[observed],held['detected_xyz'][observed])
    old_tensor,new_tensor=[prepare_tensors(dict(held,xyz=p),device) for p in (old_corrected,new_corrected)]
    index=torch.arange(tmask.sum(),device=device)
    pairs=dict(baseline=(old_gnn,old_tensor),new_tracker=(old_gnn,new_tensor),new_front=(new_gnn,new_tensor))
    preds={name:predict(model,arr,index,96).numpy() for name,(model,arr) in pairs.items()}
    depth=held['depth_mm']
    groups=dict(all=np.ones(len(depth),bool),pre=depth<=0,entry=(depth>0)&(depth<=20),deep=depth>20)
    geometry,objects={},{}
    for name,stage in groups.items():
        geometry[name]={'frames':int(stage.sum())}
        for body,cols in REGIONS.items():
            miss=~observed[stage,cols]
            entries={}
            for tag,xyz in (('old',old_raw),('new',new_raw)):
                dif=np.linalg.norm(xyz[stage,cols]-gt[stage,cols],axis=-1)[miss]*1000
                if dif.size:
                    entries[tag]=dict(mean=float(dif.mean()),p90=float(np.percentile(dif,90)))
            geometry[name][body]=dict(n_missing=int(miss.sum()),**entries)
        objects[name]={'frames':int(stage.sum())}
        for tag,pred in preds.items():
            metric=error_metrics(torch.as_tensor(pred),torch.as_tensor(held['target']))
            objects[name][tag]=describe(metric,stage)
    path=out/'heldout.npz'
    np.savez_compressed(path,**held,old_raw=old_raw,new_raw=new_raw,old_corrected=old_corrected,
                        shared_new_gnn_input=new_corrected,**preds)
    return geometry,objects,path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    data,splits,episodes,sources=load(args.root)
    args.output.mkdir(parents=True,exist_ok=False)
    summary=args.output/'dataset.json'
    write(summary,dict(episodes=episodes,splits=[
        dict(seeds=ids.tolist(),frames=int(np.isin(data['seed'],ids).sum())) for ids in splits],
        matcher_sha256=file_sha256(match.__file__),box_matcher_sha256=file_sha256(box_matcher.__file__)))
    print('Dataset prepared',[(len(ids),int(np.isin(data['seed'],ids).sum())) for ids in splits],flush=True)

    # Reuse the existing causal/missing-only trainer, pointing its warm start
    # at the selected full-insertion checkpoint.
    tracker_options=SimpleNamespace(input=args.root,output=args.output)
    _,tracker,fallback=tracker_train.fit_tracker(data,splits,tracker_options,sources,device)
    baseline_state=torch.load(TRACKER,map_location='cpu',weights_only=True)
    assert baseline_state['epoch']==430
    baseline_tracker=CornerTransformer(spatial=True).to(device).eval()
    baseline_tracker.load_state_dict(baseline_state['model'],strict=True)
    tracker.eval()
    old=predict_coordinates(baseline_tracker,data,device,fallback)
    new=predict_coordinates(tracker,data,device,fallback)
    initial=torch.load(GNN,map_location='cpu',weights_only=True)
    assert file_sha256(TRACKER)==initial['config']['tracker_sha256']
    cfg={k:initial['config'][k] for k in MODEL_KEYS}
    cfg.update(warm_start_sha256=file_sha256(GNN),baseline_tracker_sha256=file_sha256(TRACKER),
               selected_tracker_sha256=file_sha256(args.output/'tracker/best.pt'),
               training_data=sources,matcher_sha256=file_sha256(match.__file__),
               script_sha256=file_sha256(__file__),validation_seed=VAL,test_seed=TEST)
    options=SimpleNamespace(output=args.output/'gnn')
    options.output.mkdir()
    fit_gnn.__globals__['VAL'],fit_gnn.__globals__['TEST']=VAL,TEST
    tensors=prepare_tensors(dict(data,xyz=new[1]),device)
    with ExperimentLogger(team='graph',model='gtfree_object21',job_type='train',
                          seed=0,config=cfg,root=options.output/'experiments') as log:
        log.record(runtime=dict(gpu=torch.cuda.get_device_name()))
        final,stopping=fit_gnn(options,data,tensors,initial,cfg,log)
        baseline=VertexEdgeGatedMPNN(**cfg['gnn']).to(device).eval()
        baseline.load_state_dict(initial['model'],strict=True)
        geometry,objects,out=heldout(options.output,data,old,new,baseline,final.eval(),device)
        report=options.output/'summary.json'
        write(report,dict(run_id=log.run_id,config=cfg,stopping=stopping,geometry=geometry,object21=objects,
                          heldout_sha256=file_sha256(out),checkpoint_sha256=file_sha256(options.output/'best.pt')))
        for path in (report,out,options.output/'best.pt',options.output/'resume.pt'):
            log.link_output(path,checkpoint=path.suffix=='.pt')
    print('Completed GT-free Transformer and GNN comparison',flush=True)


if __name__=='__main__':
    main()

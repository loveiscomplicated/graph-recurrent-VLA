"""Reuse preserved full-budget v2 baselines after exact pipeline checks."""
from pathlib import Path
import torch
from ..common import seed_all, write_json
from ..data import load_dataset, split_indices, sha256
from ..v2.diagnostics import load_predictor, qvel_dataset
from .history_models import HistoryPolicy
from .history_dataset import HistoryDataset


def main():
    dataset=Path('outputs/graph/dataset.pt');data,g=load_dataset(dataset);q=qvel_dataset(data)
    windows=HistoryDataset(data);results={}
    for name in ['flat','surface']:
        source=Path(f'outputs/graph/v2/training/{name}_goal/best.pt')
        predict,ck,old,norm,builder,sn=load_predictor(source,data,dataset)
        seed_all(0,ck['config']['threads']);model=HistoryPolicy(name,ck['config'],'current');model.load_state_dict(old.state_dict());model.eval()
        maximum=0.
        with torch.inference_mode():
            for split in ['validation','test']:
                for ids in split_indices(data,split).split(ck['config']['batch_size']):
                    expected=predict(g.select(ids),q[ids]);actual=windows.forward(model,ids,g,q,norm,sn,builder)
                    maximum=max(maximum,float((actual-expected).abs().max()))
        if maximum != 0:raise AssertionError(f'current pipeline differs from v2: {maximum}')
        dest=Path(f'outputs/graph/v4/training/{name}_current/best.pt')
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        ck.update(schema_version=4,history_mode='current',history_length=4,baseline_source=str(source),baseline_source_sha256=sha256(source),current_pipeline_max_difference=maximum)
        torch.save(ck,dest)
        results[name]=dict(source=str(source),source_sha256=sha256(source),checked_splits=['validation','test'],max_prediction_difference=maximum,normalizers='unchanged train-only v2 checkpoint statistics',training='reused completed 30 epoch seed 0 baseline')
    write_json('outputs/graph/v4/baseline_equivalence.json',results)


if __name__=='__main__':main()

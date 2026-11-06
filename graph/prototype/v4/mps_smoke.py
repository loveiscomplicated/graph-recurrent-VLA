"""Execute real K4/batch32 MPS forward/backward and restored AdamW steps."""
import argparse
from pathlib import Path
from time import perf_counter
import torch
from ..common import seed_all,write_json
from ..data import load_dataset,split_indices
from ..graph_builder import GraphBuilder
from ..normalization import FeatureNormalizer
from ..v2.state import StateNormalizer
from ..v2.diagnostics import qvel_dataset
from .history_models import HistoryPolicy
from .history_dataset import HistoryDataset
from .devices import cpu_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['state','both'],default=['state','both'])
    p.add_argument('--source-suffix',default='')
    p.add_argument('--memory-fraction',type=float,default=.4)
    args=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    seed_all(0,4);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    data,g=load_dataset('outputs/graph/dataset.pt');q=qvel_dataset(data);train=split_indices(data,'train');ids=train[torch.randperm(len(train))[:32]];results={}
    for mode in args.modes:
        source=Path(f'outputs/graph/v4/training/surface_{mode}{args.source_suffix}/best.pt');ck=torch.load(source,map_location='cpu',weights_only=True)
        cfg=ck['config'];norm=FeatureNormalizer(**ck['normalizer']);sn=StateNormalizer(**ck['state_normalizer']);builder=GraphBuilder(cfg['surface_k'],cfg['interaction_k']);w=HistoryDataset(data,ck['history_length'])
        cpu=HistoryPolicy('surface',cfg,mode);cpu.load_state_dict(ck['model_state']);cpu.eval()
        with torch.inference_mode():reference=w.forward(cpu,ids,g,q,norm,sn,builder)
        model=HistoryPolicy('surface',cfg,mode).to('mps');model.load_state_dict(ck['model_state']);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);optimizer.load_state_dict(ck['optimizer_state'])
        before=next(model.parameters()).detach().clone();start=perf_counter();losses=[];difference=None
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);raw=w.forward(model,ids,g,q,norm,sn,builder)
            if step==0:difference=float((raw.detach().cpu()-reference).abs().max())
            loss=(raw-data['actions'][ids].to('mps')).square().mean();loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise AssertionError('nonfinite MPS gradient/loss')
            optimizer.step();losses.append(float(loss.detach()))
        torch.mps.synchronize()
        if torch.equal(before,next(model.parameters()).detach()):raise AssertionError('MPS optimizer did not update weights')
        state=cpu_tree(model.state_dict());loaded=HistoryPolicy('surface',cfg,mode);loaded.load_state_dict(state);loaded.eval();model.eval()
        with torch.inference_mode():roundtrip=float((w.forward(loaded,ids,g,q,norm,sn,builder)-w.forward(model,ids,g,q,norm,sn,builder).cpu()).abs().max())
        if difference>2e-4 or roundtrip>2e-4:raise AssertionError('CPU/MPS parity exceeds tolerance')
        results[mode]=dict(device='mps',memory_fraction=args.memory_fraction,driver_allocated_bytes=torch.mps.driver_allocated_memory(),batch_size=32,history_length=4,resume_epoch=ck['epoch'],optimizer_restored=True,optimizer_buffer_device=str(next(iter(optimizer.state.values()))['exp_avg'].device),finite_gradients=True,weights_updated=True,losses=losses,seconds=perf_counter()-start,cpu_mps_max_difference=difference,portable_checkpoint_max_difference=roundtrip,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),determinism_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        del model,optimizer,cpu,loaded;torch.mps.empty_cache()
    write_json(args.output,results)
    print(results,flush=True)


if __name__=='__main__':main()

"""Verify the complete source alignment, split windows and exact dimensions."""
from pathlib import Path
import h5py
import numpy as np
import torch
from ..common import config, seed_all, write_json, versions
from ..data import load_dataset, split_indices, validate_trajectory, sha256
from ..graph_builder import node_features, GraphBuilder
from ..normalization import FeatureNormalizer
from ..models import parameter_count
from ..v2.diagnostics import qvel_dataset
from ..v2.state import StateNormalizer
from .history_models import HistoryPolicy, MODES
from .history_dataset import HistoryDataset


def main():
    path=Path('outputs/graph/dataset.pt');data,g=load_dataset(path);q=qvel_dataset(data);w=HistoryDataset(data)
    counts={};episodes=[]
    with h5py.File(data['manifest']['source'],'r') as source:
        for ep in data['manifest']['episodes']:
            eid=ep['episode_id'];actions,obs=validate_trajectory(source[f'traj_{eid}'])
            ids=torch.where(data['episode_id']==eid)[0];t=data['timestep'][ids].numpy()
            if len(ids)!=len(actions) or not np.array_equal(data['actions'][ids].numpy(),actions[t]):raise AssertionError('source actions not aligned')
            if not np.array_equal(q[ids].numpy(),obs[t,9:18]):raise AssertionError('qvel must come from s_t')
            if not (data['timestep'][w.states[ids]]<=data['timestep'][ids,None]).all():raise AssertionError('future state leakage')
            if not (data['timestep'][w.past[ids]][w.valid[ids]]<data['timestep'][ids,None].expand_as(w.past[ids])[w.valid[ids]]).all():raise AssertionError('current/future action leakage')
            episodes.append(dict(episode_id=eid,action_count=len(actions),state_count=len(obs),maximum_timestep=int(t.max())))
    for split in ['train','validation','test']:
        ids=split_indices(data,split)
        if not torch.isin(data['episode_id'][w.states[ids]],torch.tensor(data['manifest']['splits'][split])).all():raise AssertionError('split crossed by state window')
        if not torch.isin(data['episode_id'][w.past[ids]],torch.tensor(data['manifest']['splits'][split])).all():raise AssertionError('split crossed by action window')
        counts[split]=dict(episodes=len(data['manifest']['splits'][split]),samples=len(ids))
    cfg=config(Path(__file__).with_name('config.toml'));seed_all(0,4);train=split_indices(data,'train')
    norm=FeatureNormalizer.fit(g.select(train));sn=StateNormalizer.fit(g.select(train),q[train],'goal');small=g.select(torch.tensor([0]));x,_,_,context=node_features(small)
    models={}
    with torch.inference_mode():
        for name in ['flat','surface']:
            for mode in MODES:
                model=HistoryPolicy(name,cfg,mode);model.eval()
                encoded=model.encode(norm.apply(name,small,GraphBuilder()),sn.apply(small,q[:1],'goal'))
                models[f'{name}_{mode}']=dict(embedding_dim=128,timestep_representation_dim=encoded.shape[-1],head_input=model.input_dim,head_output=7,parameters=parameter_count(model))
    write_json('outputs/graph/v4/data_audit.json',dict(dataset_sha256=sha256(path),source=data['manifest']['source'],splits=counts,total_samples=len(data['actions']),episodes=episodes,
        node_features=list(x.shape),context_features=list(context.shape),flat_input_dim=x[0].numel()+context.shape[-1],shared_dim=31,
        target='a_t',state_inputs='s_[t-K:t+1]',action_inputs='a_[t-K:t]',history_length=4,
        exact_source_action_alignment=True,qvel_from_current_obs=True,no_future_states=True,no_current_or_future_actions=True,split_windows_episode_local=True,
        raw_vs_component_clipped_expert_max_difference=float((data['actions']-data['actions'].clamp(-1,1)).abs().max()),models=models,versions=versions()))


if __name__=='__main__':main()

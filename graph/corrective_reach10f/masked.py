"""Loss-mask adapter; graph packing, recurrent forward and Method B stay inherited."""
import numpy as np
import torch

class SupervisedDataset:
    def __init__(self,dataset,masks):
        self.dataset=dataset;self.masks=masks
        self.episodes=dataset.episodes;self.config=dataset.config;self.arrays=dataset.arrays
        for eid,mask in masks.items():
            if mask.dtype!=np.bool_ or len(mask)!=self.episodes[eid]['actions']:raise ValueError('mask/episode alignment')
    def batch(self,ids,start,length,device):
        graph,robot,target,valid=self.dataset.batch(ids,start,length,device)
        supervision=torch.zeros_like(valid)
        for b,eid in enumerate(ids):
            selected=self.masks[eid][start:start+valid.shape[1]]
            supervision[b,:len(selected)]=torch.as_tensor(selected,device=device)
        return graph,robot,target,valid&supervision


def burn_in(model,data,norm,eid,stop,sequence_length,device):
    """Current weights generate branch history. No cross-branch hidden reuse.

    No optimizer/loss on all-masked chunks. Caller decides TBPTT detach at an
    optimizer boundary; every elapsed observation remains in the recurrent scan.
    """
    hidden=None
    with torch.no_grad():
        for start in range(0,stop,sequence_length):
            g,s,_,_=data.batch([eid],start,min(sequence_length,stop-start),device)
            _,hidden=model(g,s,norm,hidden)
    return None if hidden is None else hidden.detach()

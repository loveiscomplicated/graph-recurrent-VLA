"""One fresh E0 GNN + preserved readout and spatial head. No recurrent modules."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graph.integration_baseline.model import GraphEncoder
from graph.prototype.models import mlp
from graph.plan_a_bc.common import tensor_identity
from .contract import split_fields, assemble_gt38, measured_robot


class SpatialHead(nn.Module):
    def __init__(self,input_dim,target_mean):
        super().__init__()
        self.net=mlp([input_dim,256,128,18])
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32))
        # Fresh, stable identity rotation; no BC/pretrained weights are loaded.
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[3]=1;self.net[-1].bias[10]=1

    def forward(self,x):
        y=self.net(x)
        sizes=.02*F.softplus(y[:,14:]+torch.log(torch.expm1(self.target_mean[14:]/.02)))
        q1=F.normalize(y[:,3:7],dim=-1,eps=1e-8)
        q2=F.normalize(y[:,10:14],dim=-1,eps=1e-8)
        q1=torch.where(q1[:,:1]<0,-q1,q1);q2=torch.where(q2[:,:1]<0,-q2,q2)
        return torch.cat((y[:,:3]*.1+self.target_mean[:3],q1,
            y[:,7:10]*.1+self.target_mean[7:10],q2,sizes),-1)


class StateEncoder(nn.Module):
    def __init__(self,robot_mean,robot_std,target_mean):
        super().__init__()
        # Exact Plan A CachedGraphPolicy E0 modification of the shared encoder.
        self.graph_encoder=GraphEncoder(d=64,L=2)
        self.graph_encoder.node_encoder=mlp([21+4,64,64],final_activation=True)
        self.state_head=SpatialHead(320+2+76,target_mean)
        self.register_buffer('robot_mean',torch.as_tensor(robot_mean,dtype=torch.float32))
        self.register_buffer('robot_std',torch.as_tensor(robot_std,dtype=torch.float32))

    def encode_graph(self,graph):
        """Existing five64D slots: TCP, Left, Right, mean Peg, mean Box."""
        return self.graph_encoder(graph)[0]

    def forward(self,graph,robot76):
        readout,presence=self.graph_encoder(graph)
        if robot76.shape!=(graph.count,76):raise ValueError('one measured Robot76 per graph')
        prediction=self.state_head(torch.cat((readout,presence,(robot76-self.robot_mean)/self.robot_std),-1))
        return prediction,readout,presence

    def predict_state(self,graph,robot_obs):
        return split_fields(self(graph,robot_obs)[0])

    def freeze(self):
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        self.eval().requires_grad_(False)
        for p in self.parameters():p.grad=None
        return self

    @torch.inference_mode()
    def extract(self,graph,robot76,schema):
        """Same checkpoint A and B; outputs retain exact model/schema identity."""
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('freeze model before extraction')
        pred,z,pres=self(graph,robot76)
        predicted=split_fields(pred.detach().cpu().numpy())
        measured=measured_robot(robot76.detach().cpu().numpy(),schema)
        # Metadata only: do not silently advertise ambiguous estimates as a
        # universally observable GT state, or append this mask to the38 inputs.
        identifiable=np.zeros((graph.count,38),bool)
        identifiable[:,:15]=True;identifiable[:,36:]=True
        # Presence is separate. Partial surface support alone does not certify
        # a unique actor/hole center, orientation, or hidden shape extent.
        return dict(predicted_fields=predicted,state38=assemble_gt38(predicted,measured),
            readout320=z,presence2=pres,robot76=robot76,weight_sha256=tensor_identity(self.state_dict()),
            state38_identifiability_mask=identifiable,
            identifiability_mask_rule='False means NOT_CERTIFIED from current partial geometry, not proof that estimation is impossible',
            unsupported_universal_dimensions=list(range(18,25))+list(range(28,36)),
            slot_order=['TCP','Left','Right','Peg','Box'],
            uncertainty='estimated38; geometric symmetry/missing-object limitations apply')


class RobotOnly(nn.Module):
    def __init__(self,robot_mean,robot_std,target_mean):
        super().__init__()
        self.state_head=SpatialHead(76,target_mean)
        self.register_buffer('robot_mean',torch.as_tensor(robot_mean,dtype=torch.float32))
        self.register_buffer('robot_std',torch.as_tensor(robot_std,dtype=torch.float32))

    def forward(self,graph,robot76):
        return self.state_head((robot76-self.robot_mean)/self.robot_std),None,None


def absolute_loss(prediction,target,reduction=True):
    positions=torch.cat((prediction[:,:3]-target[:,:3],prediction[:,7:10]-target[:,7:10]),-1)
    pos=(positions/.1).square().mean(-1)
    rotation=torch.stack([1-(prediction[:,s]*target[:,s]).sum(-1).square().clamp(0,1)
        for s in (slice(3,7),slice(10,14))],-1).mean(-1)
    size=((prediction[:,14:]-target[:,14:])/.02).square().mean(-1)
    loss=pos+rotation+size
    return (loss.mean() if reduction else loss),dict(position=pos.mean(),rotation=rotation.mean(),size=size.mean())


def displacement_loss(a,b,gt_a,gt_b):
    cols=[0,1,2,7,8,9]
    return (((b[:,cols]-a[:,cols])-(gt_b[:,cols]-gt_a[:,cols]))/.01).square().mean()


def restore_checkpoint(path,device='cpu',freeze=True):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False)
    stats=ck['normalization']
    model=StateEncoder(stats['robot_mean'],stats['robot_std'],stats['target_mean'])
    model.load_state_dict(ck['model']);model.to(device)
    if freeze:model.freeze()
    return model,ck

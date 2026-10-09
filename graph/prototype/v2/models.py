"""Unmodified v1 encoders and a common state bypass to the same-size head."""
import torch
from torch import nn
from ..models import make_policy, mlp
from .state import STATE_DIMS


class CommonPolicy(nn.Module):
    def __init__(self,name,cfg,state_mode='goal',gripper_mode='regression'):
        super().__init__()
        original=make_policy(name,cfg)
        self.encoder=original.encoder
        self.state_mode,self.gripper_mode=state_mode,gripper_mode
        if gripper_mode not in ('regression','binary'):raise ValueError(gripper_mode)
        self.action_head=original.action_head if state_mode=='none' else mlp([128+STATE_DIMS[state_mode],cfg.get('head_width',128),7])

    def raw_action(self,inputs,shared):
        embedding=self.encoder(inputs)
        if embedding.shape!=(len(embedding),128):raise ValueError('embedding interface changed')
        if shared.shape!=(len(embedding),STATE_DIMS[self.state_mode]):raise ValueError('shared state shape mismatch')
        return self.action_head(torch.cat((embedding,shared),-1))

    def forward(self,inputs,shared):
        raw=self.raw_action(inputs,shared)
        if self.gripper_mode=='binary':
            return torch.cat((raw[:,:6],torch.where(raw[:,6:]>0,1.,-1.)),-1)
        return raw


def bc_loss(raw,target,gripper_mode='regression'):
    if gripper_mode=='regression':return (raw-target).square().mean()
    # Each motion component has 1/7 weight; binary BCE has the remaining 1/7.
    return ((raw[:,:6]-target[:,:6]).square().sum(-1).mean()+nn.functional.binary_cross_entropy_with_logits(raw[:,6],(target[:,6]+1)/2))/7

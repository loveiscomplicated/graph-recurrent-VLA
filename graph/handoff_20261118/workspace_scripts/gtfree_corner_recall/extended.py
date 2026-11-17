"""Test additional same-slot recall-preserving gates for any Hole candidate."""
from pathlib import Path

import numpy as np

import compare
from data_collection.episode_dataset import file_sha256
from match import CornerMatcher
from match_recall import EvidenceHoleMatcher


class ConfirmedHoleMatcher(EvidenceHoleMatcher):
    def __init__(self, lateral_m, normal_m):
        super().__init__(hint_penalty_m=.003,normal_weight=1.)
        self.gate=lateral_m
        self.normal_m=normal_m

    def step(self, pose, points, hints):
        if pose is None or not len(points):
            return np.full(4,-1,dtype=int)
        if self.shape is None:
            # Always initialize the rim geometry with the previously tested rule.
            super().step(pose,points,hints)
        if self.shape is None:
            return np.full(4,-1,dtype=int)
        centered=points[:,0]+1j*points[:,1]-pose.center_xy
        normal=np.abs((centered*np.exp(-1j*pose.yaw)).real+pose.side_m/2)
        idx=np.flatnonzero(normal<=self.normal_m)
        local=super().step(pose,points[idx],hints[idx])
        result=np.full(4,-1,dtype=int)
        hit=local>=0
        result[hit]=idx[local[hit]]
        return result


def build(params):
    matcher=CornerMatcher()
    if params is not None:
        matcher.hole=ConfirmedHoleMatcher(*params)
    return matcher


def main():
    default_logger=compare.ExperimentLogger

    def tagged(*,config,**kwargs):
        config['extended_sha256']=file_sha256(Path(__file__))
        return default_logger(config=config,**kwargs)

    compare.ExperimentLogger=tagged
    compare.CONFIGS={'baseline':None}
    compare.CONFIGS.update({
        f'r{lateral}_n{normal}':(.001*lateral,.001*normal)
        for lateral in (4,5,6,8)
        for normal in (3,5,8,12)
    })
    compare.build=build
    compare.main()


if __name__=='__main__':
    main()

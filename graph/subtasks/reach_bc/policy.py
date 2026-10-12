"""Reuse v2 model, geometry/state normalizers, and live GT sampling verbatim."""
from pathlib import Path
import torch

from graph.prototype.common import config
from graph.prototype.data import load_dataset, split_indices, sha256
from graph.prototype.maniskill_gt import GTAdapter
from graph.prototype.normalization import FeatureNormalizer
from graph.prototype.v2.models import CommonPolicy
from graph.prototype.v2.state import StateNormalizer, shared_features

ROOT = Path('outputs/subtasks/reach_bc_v1')
DATASET = ROOT/'data/reach_first_success.pt'
V2_CONFIG = Path(__file__).resolve().parents[2]/'prototype/v2/config.toml'


def fit_inputs(data, geometry):
    ids = split_indices(data, 'train')
    norm = FeatureNormalizer.fit(geometry.select(ids))
    sn = StateNormalizer.fit(geometry.select(ids), data['qvel'][ids], 'goal')
    return norm, sn


def inputs(geometry, qvel, norm, sn):
    return norm.apply('flat', geometry, None), sn.apply(geometry, qvel, 'goal')


def raw_features(geometry, qvel):
    from graph.prototype.graph_builder import flat_features
    return flat_features(geometry), shared_features(geometry, qvel, 'goal')


def decode(action):
    if action.shape[-1] != 7 or not torch.isfinite(action).all():
        raise ValueError('expected finite normalized 7D EE action')
    return action.clamp(-1, 1)


class LivePolicy:
    def __init__(self, checkpoint, dataset=DATASET):
        self.ck = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if self.ck['dataset_sha256'] != sha256(dataset):
            raise ValueError('checkpoint/dataset hash mismatch')
        data, _ = load_dataset(dataset)
        if self.ck['splits'] != data['manifest']['splits']:
            raise ValueError('checkpoint/dataset split mismatch')
        self.model = CommonPolicy('flat', self.ck['config'], 'goal', 'regression')
        self.model.load_state_dict(self.ck['model_state'])
        self.model.eval()
        self.norm = FeatureNormalizer(**self.ck['normalizer'])
        self.sn = StateNormalizer(**self.ck['state_normalizer'])
        cfg = self.ck['config']
        self.adapter = GTAdapter(cfg['peg_points'], cfg['hole_points'], cfg['sampling_seed'])

    def features(self, env):
        # Fresh extraction every control step; native observation is not a cache.
        return inputs(self.adapter(env), env.unwrapped.agent.robot.get_qvel().detach(), self.norm, self.sn)

    @torch.inference_mode()
    def __call__(self, env):
        return self.model(*self.features(env))

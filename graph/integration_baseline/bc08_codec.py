"""Task08 opt-in action encodings; the architecture/controller stay unchanged."""
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EncodedAction:
    values: torch.Tensor
    codec_id: str


class ActionCodec:
    VERSION = 'task08_action_v1'

    def __init__(self, state):
        self.state = json.loads(json.dumps(state, allow_nan=False))
        if state['version'] != self.VERSION or state['method'] not in ('A', 'B', 'C'):
            raise ValueError('unsupported ActionCodec')
        self.method = state['method']
        self.mean = np.asarray(state['mean6'], np.float32)
        self.std = np.asarray(state['std6'], np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('invalid motion statistics')
        if state['gripper_mode'] != ('binary_logit' if self.method == 'C' else 'continuous'):
            raise ValueError('codec/gripper mode mismatch')
        if state['fitted_episode_ids'] != sorted(set(state['fitted_episode_ids'])):
            raise ValueError('duplicate/unsorted action statistics episode IDs')
        self.identity = digest(self.state)

    @classmethod
    def fit(cls, method, actions_by_episode, train_ids):
        ids = sorted(actions_by_episode)
        if not ids or not set(ids) <= set(train_ids):
            raise ValueError('action statistics must use actual train subset only')
        arrays = [np.asarray(actions_by_episode[e], np.float32) for e in ids]
        if any(a.ndim != 2 or a.shape[1] != 7 or not len(a) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('finite native Action7 required')
        a = np.concatenate(arrays)
        values, counts = np.unique(a[:, 6], return_counts=True)
        if method == 'C' and not np.isin(values, [-1., 1.]).all():
            raise ValueError('non-binary expert gripper: inspect distribution before mapping')
        raw = a[:, :6].std(0)
        # Same action convention as recurrent.train.make_stats: additive epsilon,
        # not its unrelated observation STD_FLOOR=.01.
        return cls(dict(version=cls.VERSION, method=method, mean6=a[:, :6].mean(0).tolist(),
            raw_std6=raw.tolist(), std6=(raw+np.float32(1e-6)).tolist(),
            stability_rule='float32 population std(ddof=0) + 1e-6; recurrent action convention',
            epsilon=1e-6, floor_used=False, small_std_components=np.flatnonzero(raw < 1e-6).tolist(),
            fitted_episode_ids=ids, count=len(a),
            gripper_mode='binary_logit' if method == 'C' else 'continuous',
            gripper_distribution={str(float(v)):int(n) for v,n in zip(values,counts)},
            label_mapping='open +1 -> 1; close -1 -> 0; logit > 0 -> +1 else -1'))

    def _stats(self, x):
        return (torch.as_tensor(self.mean, device=x.device, dtype=x.dtype),
                torch.as_tensor(self.std, device=x.device, dtype=x.dtype))

    def tag(self, values):
        if values.shape[-1] != 7 or not torch.isfinite(values).all():
            raise ValueError('finite encoded Action7 required')
        return EncodedAction(values, self.identity)

    def decode(self, encoded):
        if not isinstance(encoded, EncodedAction) or encoded.codec_id != self.identity:
            raise ValueError('explicit matching ActionCodec tag required; logits are not native commands')
        x = encoded.values
        if self.method == 'A':
            return x
        mean, std = self._stats(x)
        grip = torch.where(x[..., 6:] > 0, 1., -1.) if self.method == 'C' else x[..., 6:]
        result = torch.cat((x[..., :6]*std+mean, grip), -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite decoded native action')
        return result

    def loss(self, pred, target, mask):
        if mask.dtype != torch.bool or pred.shape != target.shape or mask.shape != pred.shape[:-1] or not mask.any():
            raise ValueError('nonempty masked timestep contract')
        p, y = pred[mask], target[mask]
        if not torch.isfinite(p).all() or not torch.isfinite(y).all():
            raise FloatingPointError('nonfinite selected targets/predictions')
        if self.method == 'A':
            return F.mse_loss(p, y)
        mean, std = self._stats(y)
        motion = F.mse_loss(p[:, :6], (y[:, :6]-mean)/std)
        if self.method == 'C':
            if not ((y[:, 6] == -1) | (y[:, 6] == 1)).all():
                raise ValueError('non-binary gripper target')
            grip = F.binary_cross_entropy_with_logits(p[:, 6], (y[:, 6] > 0).to(y.dtype))
        else:
            grip = F.mse_loss(p[:, 6], y[:, 6])
        return (6*motion+grip)/7


class DecodedPolicy(torch.nn.Module):
    """Only rollout adapter exposes native step(); train uses the original model."""
    def __init__(self, model, codec):
        super().__init__()
        self.model, self.codec = model, codec
        self.raw_outputs = []

    @property
    def graph_encoder(self):
        return self.model.graph_encoder

    @property
    def policy(self):
        return self.model.policy

    def step(self, *args, **kwargs):
        out, hidden = self.model.step(*args, **kwargs)
        self.raw_outputs.append(out.detach().cpu().numpy().copy())
        return self.codec.decode(self.codec.tag(out)), hidden


def controller_commands(actions):
    """Installed ManiSkill preprocessing; this function never applies an action."""
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    from mani_skill.utils.gym_utils import clip_and_scale_action
    a = torch.as_tensor(np.asarray(actions, np.float32))
    stub = SimpleNamespace(action_space_low=torch.full((6,), -.1),
        action_space_high=torch.full((6,), .1), config=SimpleNamespace(rot_lower=-.1))
    arm = PDEEPoseController._clip_and_scale_action(stub, a[:, :6])
    grip = clip_and_scale_action(a[:, 6:], torch.tensor([-.01]), torch.tensor([.04]))
    return torch.cat((arm, grip), 1).numpy()


def native_metrics(pred, target):
    from scipy.spatial.transform import Rotation
    p, y = np.asarray(pred, np.float32), np.asarray(target, np.float32)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 7 or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('native metrics require finite paired [T,7]')
    e = p.astype(np.float64)-y.astype(np.float64)
    cp, cy = controller_commands(p), controller_commands(y)
    angles = (Rotation.from_euler('XYZ', cp[:, 3:6]).inv()*Rotation.from_euler('XYZ', cy[:, 3:6])).magnitude()
    def summary(v):
        return dict(mean=float(np.mean(v)), p95=float(np.quantile(v,.95)), max=float(np.max(v)))
    pt, yt = np.flatnonzero(np.diff(p[:,6] > 0))+1, np.flatnonzero(np.diff(y[:,6] > 0))+1
    # Event timing and missing/extra transitions are separate; do not invent a
    # finite timing error when a prediction has no matching transition.
    nearest = [int(np.min(abs(pt-t))) for t in yt] if len(pt) else []
    return dict(native_action7_mse=float(np.mean(e**2)), native_action7_mae=float(np.mean(abs(e))),
        native_motion6_mse=float(np.mean(e[:,:6]**2)), native_motion6_mae=float(np.mean(abs(e[:,:6]))),
        translation_mse=float(np.mean(e[:,:3]**2)), rotation_mse=float(np.mean(e[:,3:6]**2)),
        gripper_mse=float(np.mean(e[:,6]**2)), gripper_accuracy=float(np.mean((p[:,6]>0)==(y[:,6]>0))),
        native_translation_vector_error=summary(np.linalg.norm(e[:,:3],axis=1)),
        controller_translation_error_m=summary(np.linalg.norm(cp[:,:3]-cy[:,:3],axis=1)),
        controller_rotation_geodesic_rad=summary(angles),
        per_dimension_mse=np.mean(e**2,axis=0).tolist(), count=len(p),
        gripper_transition=dict(predicted=pt.tolist(), target=yt.tolist(),
            nearest_prediction_error_steps=nearest, missing_prediction=bool(len(yt) and not len(pt)),
            count_error=int(len(pt)-len(yt)),
            boundary_error_rate=float(np.mean(np.diff(p[:,6]>0)!=np.diff(y[:,6]>0))) if len(p)>1 else 0.),
        units='native commands; controller translation m, intrinsic XYZ relative command geodesic rad; not achieved TCP motion')

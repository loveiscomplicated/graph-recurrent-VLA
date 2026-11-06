import torch

from graph.prototype.geometry import GeometryState, transform, rotation
from graph.prototype.surface_sampling import peg_surface, hole_surface


def geometry(batch=4):
    """Analytic geometry for unit tests only, never demonstration/training data."""
    torch.manual_seed(12)
    identity = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).expand(batch, -1).clone()
    ee = identity.clone()
    ee[:, :3] = torch.rand(batch, 3)*.2
    peg = identity.clone()
    peg[:, :3] = torch.rand(batch, 3)*.3
    box = identity.clone()
    box[:, :3] = torch.rand(batch, 3)*.3
    half = torch.rand(batch, 3)*.002+torch.tensor([.1, .02, .02])
    box_half = half[:, :1].expand(-1, 3).clone()
    offset = torch.tensor([[0., .01, -.02]]).expand(batch, -1).clone()
    radius = half[:, 1:2]+.003
    hole = box.clone()
    hole[:, :3] += offset
    pl, pn = peg_surface(half)
    hl, hn = hole_surface(box_half, radius, offset)
    return GeometryState(ee, ee[:, :3]+torch.tensor([0., .04, 0.]),
        ee[:, :3]-torch.tensor([0., .04, 0.]), torch.full((batch, 2), .04),
        torch.zeros(batch, 9), identity, peg, half, pl, transform(pl, peg), pn,
        box, box_half, hole, radius, offset, hl, transform(hl, box), hn)

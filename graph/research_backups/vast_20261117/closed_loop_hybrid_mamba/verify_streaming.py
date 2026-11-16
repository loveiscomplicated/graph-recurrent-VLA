"""Check the causal per-frame bridge against the original batched inference."""
import argparse
from pathlib import Path

import numpy as np
import torch

from hybrid_frontend import StreamingTracker
from tracker import CornerTransformer
from train_tracker import evaluate, prepare, read_npz


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    data = read_npz(a.inputs)
    seeds = np.unique(data['seed'])
    sample = seeds[[0, len(seeds)//2, -1]]
    fallback = ck['fallback'].numpy()
    packed = prepare(data, fallback, sample, device)
    with torch.inference_mode():
        reference = evaluate(model, packed, 6)
    rows = []
    for seed in sample:
        stream = StreamingTracker(model, fallback, device)
        for i in np.flatnonzero(data['seed'] == seed):
            rows.append(stream.step(data['detected_xyz'][i], data['observed'][i], data['tcp'][i]))
    output = np.stack(rows)
    np.testing.assert_allclose(output, reference, atol=3e-6, rtol=0)
    print(f'Verified {len(rows)} streaming frames; maximum absolute difference {abs(output-reference).max():.3g} m')


if __name__ == '__main__':
    main()

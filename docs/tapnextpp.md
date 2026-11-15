# TAPNext++ point tracking

The Visual prototype uses the upstream PyTorch tracker in the `visual/third_party/tapnet`
submodule. The submodule pins the same upstream revision used by the original
local benchmark. Run the commands below from the team repository root.

```bash
git submodule update --init visual/third_party/tapnet
python -m pip install numpy torchvision einops
mkdir -p visual/checkpoints/tapnextpp
curl --fail --location --retry 3 \
  https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
  -o visual/checkpoints/tapnextpp/tapnextpp_256_official.pt
```

Use a Python environment with a compatible PyTorch and torchvision pair.
The checkpoint is about 2.5 GB and lives in a Git-ignored directory.
Its SHA-256 is
`cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da`.

Run two frames to check loading, query initialization, and recurrent state.
The upstream wrapper takes BGR images and returns pixel coordinates in the
original frame as `(x, y)`, along with visibility and the next state.

```bash
PYTHONPATH="visual/third_party/tapnet${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import numpy as np
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

tracker = TAPNextPP.from_checkpoint(
    "visual/checkpoints/tapnextpp/tapnextpp_256_official.pt", device="cpu"
)
frame = np.zeros((256, 256, 3), dtype=np.uint8)
queries = np.array([[128, 128]], dtype=np.float32)
xy, visible, state = tracker.track_frame(frame, queries, autocast=False)
xy, visible, state = tracker.track_frame(frame, state=state, autocast=False)
assert xy.shape == (1, 2) and visible.shape == (1,) and np.isfinite(xy).all()
print(xy, visible)
PY
```

For the [512×512 checkpoint](https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt),
download `tapnextpp_512.ckpt` into the same directory and pass its path and
`input_resolution=512` to `from_checkpoint`.
See the [upstream tracker guide](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp/votsp2026/README.md)
for additional usage.

Other checkouts can add this repository's `visual/third_party/tapnet` directory to their
`PYTHONPATH` and pass an absolute path to the checkpoint in
`visual/checkpoints/tapnextpp/`.

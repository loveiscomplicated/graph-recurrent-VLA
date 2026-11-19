#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 SHARD_INDEX SHARD_COUNT BOOTSTRAP_DIRECTORY" >&2
  exit 2
fi

SHARD_INDEX="$1"
SHARD_COUNT="$2"
BOOTSTRAP_DIRECTORY="$3"
REPOSITORY_ROOT="/workspace/graph-recurrent-VLA"
SURFACE_ROOT="/workspace/surface_mamba"

export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null
apt-get install -y -q curl git tmux >/dev/null

if [ ! -d "$REPOSITORY_ROOT/.git" ]; then
  git clone -q "$BOOTSTRAP_DIRECTORY/surface_mamba.bundle" "$REPOSITORY_ROOT"
fi
cd "$REPOSITORY_ROOT"
git submodule update --init --recursive

mkdir -p \
  "$SURFACE_ROOT/checkpoints" \
  "$REPOSITORY_ROOT/visual/checkpoints/tapnextpp" \
  /root/.secrets
install -m 644 "$BOOTSTRAP_DIRECTORY/checkpoints/surface_tracker_best.pt" \
  "$SURFACE_ROOT/checkpoints/surface_tracker_best.pt"
install -m 644 "$BOOTSTRAP_DIRECTORY/checkpoints/surface_gnn_best.pt" \
  "$SURFACE_ROOT/checkpoints/surface_gnn_best.pt"
install -m 644 "$BOOTSTRAP_DIRECTORY/excluded_and_evaluation_seeds.json" \
  "$SURFACE_ROOT/excluded_and_evaluation_seeds.json"
install -m 644 "$BOOTSTRAP_DIRECTORY/evaluation_seeds.json" \
  "$SURFACE_ROOT/evaluation_seeds.json"
install -m 600 "$BOOTSTRAP_DIRECTORY/hf.env" /root/.secrets/hf.env

TAP_CHECKPOINT="$REPOSITORY_ROOT/visual/checkpoints/tapnextpp/tapnextpp_256_official.pt"
TAP_SHA256="cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"
if [ ! -f "$TAP_CHECKPOINT" ] || \
   [ "$(sha256sum "$TAP_CHECKPOINT" | awk '{print $1}')" != "$TAP_SHA256" ]; then
  curl --fail --location --retry 5 --retry-all-errors \
    https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
    -o "$TAP_CHECKPOINT.partial"
  if [ "$(sha256sum "$TAP_CHECKPOINT.partial" | awk '{print $1}')" != "$TAP_SHA256" ]; then
    echo "downloaded TAPNext++ checkpoint hash differs" >&2
    exit 1
  fi
  mv "$TAP_CHECKPOINT.partial" "$TAP_CHECKPOINT"
fi

bash recurrent/setup_surface_mamba_server.sh
exec bash recurrent/run_surface_mamba_shard.sh "$SHARD_INDEX" "$SHARD_COUNT"

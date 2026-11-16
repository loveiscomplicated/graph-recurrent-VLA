"""Compare the first TAPNext++ block on identical saved-episode tokens."""

import json
from functools import partial
from pathlib import Path
from time import perf_counter

import cv2
import mlx.core as mx
import numpy as np
import torch
from tapnet.tapnextpp.votsp2026 import utils as tap_utils
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

from data_collection.episode_dataset import file_sha256
from evaluation.tapnextpp_surface_tracking import read_frames
from utils.experiment_logger import ExperimentLogger
from visual.tapnextpp_mlx import OnlineBlock, RecurrentState, compile_continuation
from visual.tapnextpp_mps import (
    compile_mps_recurrent_blocks,
    crop_native_patches,
    enable_mps_fp16_vit,
    run_block_in_fp16,
)


def make_inputs(tracker, frames, queries):
    """Prepare shared first-block inputs through the upstream MPS embedding path."""
    model = tracker._model
    height, width = frames[0][0].shape[:2]
    query_xy = tap_utils.display_to_model(queries, height, width, tracker.MODEL_SIZE)
    query_tensor = tap_utils.make_query_tensor(query_xy, tracker.device)
    query_tokens = model.embed_queries(1, query_tensor).reshape(len(queries), model.width)
    mask_tokens = model.mask_token.reshape(1, model.width).expand_as(query_tokens)
    inputs = []
    for index, (rgb, *_rest) in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        video = tap_utils.preprocess_frame(bgr, tracker.device, tracker.input_resolution)
        patches = model.lin_proj(video[0].permute(0, 3, 1, 2))
        image = patches.flatten(2).transpose(1, 2)
        image = (image + model._video_pos_emb(*patches.shape[-2:])).reshape(-1, model.width)
        points = query_tokens if index == 0 else mask_tokens
        inputs.append(torch.cat((image, points)))
    return inputs


def run_mps(block, tokens, state):
    output, state = block(tokens[None, None], cache=state)
    torch.mps.synchronize()
    return output[0, 0], state


def make_mlx_runner(block):
    compiled = compile_continuation(block)

    def run(tokens, state):
        if state is None:
            output, state = block(tokens)
        else:
            output, history, hidden = compiled(tokens, state.history, state.hidden)
            state = RecurrentState(history, hidden)
        mx.eval(output, state.history, state.hidden)
        return output, state

    return run


def replay(step, inputs):
    """Measure synchronous latency with independent warmup and measured state."""
    _, warm_state = step(inputs[0], None)
    _, warm_state = step(inputs[1], warm_state)
    del warm_state
    state = None
    outputs = []
    timings = []
    for tokens in inputs:
        start = perf_counter()
        output, state = step(tokens, state)
        timings.append((perf_counter() - start) * 1000)
        outputs.append(output)
    return outputs, state, np.asarray(timings)


def timing(samples):
    samples = np.concatenate([trial[1:] for trial in samples])
    return {"mean_ms": float(samples.mean()), "p95_ms": float(np.percentile(samples, 95))}


def errors(reference, candidate):
    delta = reference.astype(np.float64) - candidate.astype(np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("First-block parity check found nonfinite values")
    rmse = np.sqrt(np.mean(delta ** 2))
    ref_rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2))
    return {"max_abs": float(np.max(np.abs(delta))), "rmse": float(rmse),
            "relative_rmse": float(rmse / max(ref_rms, 1e-12))}


@torch.no_grad()
def evaluate(config):
    """Benchmark continuous frame-by-frame block inference on MPS and MLX GPU."""
    if not torch.backends.mps.is_available() or not mx.metal.is_available():
        raise RuntimeError("The block comparison requires Apple Silicon with MPS and MLX Metal")
    mx.set_default_device(mx.gpu)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    receipt = Path(config["source"]) / "episode.json"
    episode = json.loads(receipt.read_text())
    frames = read_frames(config, episode)
    with np.load(config["reference_tracks"]) as reference:
        queries = reference["query_xy"]
        np.testing.assert_array_equal(reference["frame_indices"], np.arange(len(frames)))
    tracker = TAPNextPP.from_checkpoint(config["checkpoint"], device="mps")
    enable_mps_fp16_vit(tracker)
    for block in tracker._model.blocks:
        run_block_in_fp16(block.ssm_block.mlp_block)
    crop_native_patches(tracker, config["tap_native_patch_crop_xyxy"])
    block = tracker._model.blocks[0]
    mlx_block = OnlineBlock(block)
    torch_inputs = make_inputs(tracker, frames, queries)
    mlx_inputs = [mx.array(tokens.cpu().numpy()) for tokens in torch_inputs]
    mx.eval(mlx_inputs)
    compile_mps_recurrent_blocks(tracker)

    mps_step = partial(run_mps, block)
    mlx_step = make_mlx_runner(mlx_block)
    with ExperimentLogger(team="visual", model="tapnextpp_mlx_block", config=config,
                          seed=episode["seed"], job_type="evaluate") as logger:
        logger.record(episode_sha256=file_sha256(receipt),
                      checkpoint_sha256=file_sha256(Path(config["checkpoint"])),
                      reference_tracks_sha256=file_sha256(Path(config["reference_tracks"])),
                      torch_version=str(torch.__version__), mlx_version=mx.__version__)
        mps_first = replay(mps_step, torch_inputs)
        mlx_first = replay(mlx_step, mlx_inputs)
        mlx_second = replay(mlx_step, mlx_inputs)
        mps_second = replay(mps_step, torch_inputs)
        mps_output, mps_state, _ = mps_second
        mlx_output, mlx_state, _ = mlx_second
        reference = torch.stack(mps_output).cpu().numpy()
        candidate = np.stack([np.asarray(output) for output in mlx_output])
        result = {
            "frames": len(frames),
            "tokens": torch_inputs[0].shape[0],
            "queries": len(queries),
            "timing": {"mps": timing((mps_first[2], mps_second[2])),
                       "mlx": timing((mlx_first[2], mlx_second[2]))},
            "first_output": errors(reference[0], candidate[0]),
            "continuation_output": errors(reference[1:], candidate[1:]),
            "query_output": errors(reference[:, -len(queries):], candidate[:, -len(queries):]),
            "final_convolution_history": errors(mps_state.conv1d_state.cpu().numpy(),
                                                np.asarray(mlx_state.history)),
            "final_rg_lru_state": errors(mps_state.rg_lru_state.cpu().numpy(), np.asarray(mlx_state.hidden)),
        }
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        summary = output / "summary.json"
        summary.write_text(json.dumps(result, indent=2) + "\n")
        logger.link_output(summary)
        trace = output / "timings.npz"
        np.savez_compressed(trace, mps=np.stack((mps_first[2], mps_second[2])),
                            mlx=np.stack((mlx_first[2], mlx_second[2])))
        logger.link_output(trace)
        logger.log(result, prefix="eval")
        print(f"Output directory: {output}", flush=True)
        print(json.dumps(result, indent=2), flush=True)

"""Paired native 128/512/768 SAM2 timing; isolated from production runners."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(WANDB_MODE="disabled", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
SOURCE = Path("/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source")
sys.path.insert(0, str(SOURCE))
PREVIOUS = Path("/workspace/task10i_stage4a2_qwen35_native_resolution_prompt_20261102_v1")
CHECKPOINT = Path("/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SIZES = (128, 512, 768)
SEED = 42


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array(value):
    return value.detach().cpu().numpy().copy()


def physical(task):
    return {k: array(v) for k, v in dict(
        qpos=task.agent.robot.get_qpos(), qvel=task.agent.robot.get_qvel(),
        peg=task.peg.pose.raw_pose, box=task.box.pose.raw_pose,
        tcp=task.agent.tcp.pose.raw_pose, clock=task.elapsed_steps).items()}


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
        "--format=csv,noheader"], text=True).strip()


def capture(root):
    import numpy as np
    import torch
    from dataclasses import replace
    from PIL import Image
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.sensors.camera import Camera
    from mani_skill.utils.structs.pose import Pose
    from utils.experiment_logger import ExperimentLogger

    class TripleEnv(PegInsertionSideEnv):
        def _setup_sensors(self, options):
            super()._setup_sensors(options)
            for uid in ("base_camera", "hand_camera"):
                for size in (512, 768):
                    config = replace(self._sensor_configs[uid], uid=f"{uid}_{size}", width=size, height=size)
                    self._sensor_configs[config.uid] = config
                    self._sensors[config.uid] = Camera(config, self.scene, articulation=self.agent.robot)

    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(task="10i_sam2_resolution_latency_20261103_v1", sizes=list(SIZES), seed=SEED,
        parents=[p["parent_id"] for p in read(PREVIOUS / "selection_manifest.json")["parents"]],
        cameras=["base_camera", "hand_camera"], static_frames=[0, 80], tracking_frames=list(range(21)),
        static_repeats=6, tracking_repeats=3, static_order="all six permutations, scene-dependent rotation",
        tracking_order="three cyclic permutations, sequence-dependent rotation", warmup_static_rounds=3,
        warmup_tracking_sequences_per_resolution=1, fixed_internal_image_size=1024, objects=2,
        box_source="768 actor segmentation bbox at each static frame; tracking only t0; normalized once and scaled identically",
        absent_object_box="full image [0,0,1,1], still run both prompts; no skipped calls",
        precision="FP32 weights, CUDA BF16 autocast, TF32 allowed", model_config=CONFIG,
        checkpoint_sha256=sha(CHECKPOINT), benchmark_source_sha256=sha(__file__),
        timing="synchronized wall clock; static RGB preloaded; video init JPEG load separate; steady frame t1..20",
        no_quality_claim=True, physics_steps=0, publishing=False)
    assert protocol["checkpoint_sha256"] == "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
    write(root / "protocol.json", protocol)
    torch.set_num_threads(4)
    parents = read(PREVIOUS / "selection_manifest.json")["parents"]
    manifest, audits = [], []
    with ExperimentLogger(team="visual", model="SAM2_resolution_native_capture", seed=SEED,
                          job_type="prepare", root=root / "experiments", config=protocol) as logger:
        for parent in parents:
            raw, gtpath = Path(parent["raw_path"]), Path(parent["gt_path"])
            assert sha(raw) == parent["raw_sha256"] and sha(gtpath) == parent["gt_sha256"]
            seed = read(raw.parent.parent / "attempt_identity.json")["seed"]
            env = TripleEnv(num_envs=1, sim_backend="physx_cpu", render_backend="cpu",
                           obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose")
            try:
                env.reset(seed=seed)
                task = env.unwrapped
                ids = {name: [int(x.per_scene_id) for x in getattr(task, name)._objs] for name in ("peg", "box")}
                with np.load(gtpath, allow_pickle=False) as gt:
                    np.testing.assert_array_equal(array(task.peg_half_sizes)[0], gt["raw_peg_half"][0])
                    np.testing.assert_array_equal(array(task.box_hole_radii).reshape(-1), gt["raw_hole_radius"][0])
                    for t in list(range(21)) + [80]:
                        task.agent.robot.set_qpos(torch.tensor(gt["physics_qpos"][t][None]))
                        task.agent.robot.set_qvel(torch.tensor(gt["physics_qvel"][t][None]))
                        task.peg.set_pose(Pose.create(gt["physics_peg_pose"][t][None]))
                        task.box.set_pose(Pose.create(gt["physics_box_pose"][t][None]))
                        task._elapsed_steps[:] = t
                        before = physical(task)
                        obs = task.get_obs()
                        after = physical(task)
                        assert all(np.array_equal(before[k], after[k]) for k in before)
                        np.testing.assert_allclose(array(task.agent.tcp.pose.raw_pose)[0], gt["physics_ee_pose"][t], atol=1e-6, rtol=0)
                        for camera in protocol["cameras"]:
                            reference = array(obs["sensor_data"][f"{camera}_768"]["segmentation"])[0, ..., 0]
                            boxes, absent = [], []
                            for obj in ("peg", "box"):
                                v, u = np.nonzero(np.isin(reference, ids[obj]))
                                absent.append(not bool(len(u)))
                                boxes.append([float(u.min()/768), float(v.min()/768), float((u.max()+1)/768), float((v.max()+1)/768)] if len(u) else [0., 0., 1., 1.])
                            reference_ext = array(obs["sensor_param"][camera]["extrinsic_cv"])[0]
                            reference_k = array(obs["sensor_param"][camera]["intrinsic_cv"])[0]
                            for size in SIZES:
                                uid = camera if size == 128 else f"{camera}_{size}"
                                rgb = array(obs["sensor_data"][uid]["rgb"])[0]
                                assert rgb.dtype == np.uint8 and rgb.shape == (size, size, 3)
                                params = {k: array(v)[0] for k, v in obs["sensor_param"][uid].items()}
                                np.testing.assert_allclose(params["extrinsic_cv"], reference_ext, atol=2e-6, rtol=0)
                                np.testing.assert_allclose(params["intrinsic_cv"][:2]/(size/128), reference_k[:2], atol=1e-5, rtol=0)
                                folder = root / "inputs" / parent["parent_id"] / camera / str(size)
                                folder.mkdir(parents=True, exist_ok=True)
                                png = folder / f"{t:05}.png"
                                Image.fromarray(rgb).save(png)
                                jpeg = folder / "jpeg" / f"{t:05}.jpg"
                                if t <= 20:
                                    jpeg.parent.mkdir(exist_ok=True)
                                    Image.fromarray(rgb).save(jpeg, quality=100, subsampling=0)
                                manifest.append(dict(parent_id=parent["parent_id"], camera=camera, timestep=t,
                                    size=size, png=str(png), png_sha256=sha(png), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                    jpeg=str(jpeg) if t <= 20 else None, jpeg_sha256=sha(jpeg) if t <= 20 else None,
                                    boxes_normalized=boxes, reference_absent=absent))
                            audits.append(dict(parent_id=parent["parent_id"], camera=camera, timestep=t,
                                physical_state_unchanged=True, calibration_agrees=True,
                                extrinsic_cv=reference_ext.tolist(), intrinsic_128=reference_k.tolist(),
                                backend=dict(sim=str(task.backend.sim_backend), render=str(task.backend.render_backend), device=str(task.device))))
                        if t in (0, 20, 80):
                            print("CAPTURE", parent["parent_id"], t, flush=True)
            finally:
                env.close()
        assert len(manifest) == 528 and len(audits) == 176
        write(root / "input_manifest.json", manifest)
        write(root / "capture_verification.json", dict(status="PASS", audits=audits, physics_steps=0,
            logger_run_id=logger.run_id, source_parents=parents))
        logger.log(dict(native_images=len(manifest), physical_camera_frames=len(audits)))
        logger.link_output(root / "input_manifest.json")


def setup(root, mode):
    import numpy as np
    import torch
    from utils.experiment_logger import ExperimentLogger
    assert os.environ.get("CONTAINER_ID") == "58576054"
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert not active, "another CUDA process is active; do not contaminate timing"
    protocol = read(root / "protocol.json")
    assert sha(CHECKPOINT) == protocol["checkpoint_sha256"]
    assert sha(__file__) == protocol["benchmark_source_sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger = ExperimentLogger(team="visual", model="SAM2_resolution_" + mode, seed=SEED,
        job_type="evaluate", root=root / "experiments", config=protocol)
    identity = dict(run_id=logger.run_id, container=os.environ.get("CONTAINER_ID"), before_gpu=gpu_snapshot(),
        packages={p: importlib.metadata.version(p) for p in ("torch", "torchvision", "SAM-2", "numpy", "Pillow")},
        checkpoint_sha256=sha(CHECKPOINT), manifest_sha256=sha(root / "input_manifest.json"),
        source_sha256=sha(__file__), torch_cuda=torch.version.cuda, threads=torch.get_num_threads())
    return protocol, logger, identity


def stat(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), minimum=float(values.min()), maximum=float(values.max()))


def static(root):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    protocol, logger, identity = setup(root, "static")
    with logger:
        begin = time.perf_counter()
        model = build_sam2(CONFIG, str(CHECKPOINT), device="cuda")
        predictor = SAM2ImagePredictor(model)
        torch.cuda.synchronize()
        identity["load_ms"] = (time.perf_counter()-begin)*1000
        assert model.image_size == 1024 and all(p.dtype == torch.float32 for p in model.parameters())
        cases = [r for r in read(root / "input_manifest.json") if r["timestep"] in (0, 80)]
        images = {}
        for row in cases:
            assert sha(row["png"]) == row["png_sha256"]
            rgb = np.asarray(Image.open(row["png"]).convert("RGB"))
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"]
            images[(row["parent_id"], row["camera"], row["timestep"], row["size"])] = (row, rgb)
        pairs = sorted({k[:3] for k in images})
        rows, hashes = [], {}

        def sample(key, warmup, repeat, position):
            row, rgb = images[key]
            boxes = np.asarray(row["boxes_normalized"], dtype=np.float32)*row["size"]
            predictor.reset_predictor()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                masks = []
                for box in boxes:
                    mask, _, _ = predictor.predict(box=box, multimask_output=False)
                    masks.append(mask > 0)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
            masks = np.concatenate(masks)
            assert masks.shape == (2, row["size"], row["size"])
            digest = hashlib.sha256(np.packbits(masks).tobytes()).hexdigest()
            if not warmup:
                if key in hashes:
                    assert hashes[key] == digest, "repeat output changed"
                hashes[key] = digest
                rows.append(dict(parent_id=key[0], camera=key[1], timestep=key[2], size=key[3],
                    repeat=repeat, order_position=position, encode_wall_ms=(t1-t0)*1000,
                    two_prompt_output_wall_ms=(t2-t1)*1000, total_wall_ms=(t2-t0)*1000,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), masks_sha256=digest))

        for _ in range(protocol["warmup_static_rounds"]):
            for size in SIZES:
                sample((*pairs[0], size), True, -1, -1)
        orders = list(itertools.permutations(SIZES))
        for repeat in range(protocol["static_repeats"]):
            for scene_index, pair in enumerate(pairs):
                for position, size in enumerate(orders[(repeat+scene_index) % len(orders)]):
                    sample((*pair, size), False, repeat, position)
            print("STATIC_REPEAT", repeat+1, "samples", len(rows), flush=True)
            write(root / "static_raw.json", rows)
        assert len(rows) == 288 and len(hashes) == 48
        identity.update(after_gpu=gpu_snapshot(), repeat_masks_exact=True,
                        internal_input_shape=[1, 3, model.image_size, model.image_size])
        summary = {str(size): {metric: stat([r[metric] for r in rows if r["size"] == size]) for metric in
            ("encode_wall_ms", "two_prompt_output_wall_ms", "total_wall_ms", "peak_allocated_bytes")} for size in SIZES}
        write(root / "static_identity.json", identity)
        write(root / "static_summary.json", summary)
        write_csv(root / "static_raw.csv", rows)
        logger.log({size: {k: v["mean"] for k, v in metrics.items()} for size, metrics in summary.items()})
        logger.link_output(root / "static_summary.json")


def tracking(root):
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    protocol, logger, identity = setup(root, "tracking")
    with logger:
        start = time.perf_counter()
        model = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device="cuda")
        torch.cuda.synchronize()
        identity["load_ms"] = (time.perf_counter()-start)*1000
        assert model.image_size == 1024 and all(p.dtype == torch.float32 for p in model.parameters())
        inputs = [r for r in read(root / "input_manifest.json") if r["timestep"] <= 20]
        for row in inputs:
            assert sha(row["jpeg"]) == row["jpeg_sha256"]
        first = {(r["parent_id"], r["camera"], r["size"]): r for r in inputs if r["timestep"] == 0}
        seqs = sorted({k[:2] for k in first})
        frames, sequences, hashes = [], [], {}

        def sample(key, warmup, repeat, position):
            row = first[key]
            folder = Path(row["jpeg"]).parent
            assert len(list(folder.glob("*.jpg"))) == 21
            boxes = np.asarray(row["boxes_normalized"], np.float32)*row["size"]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            state = None
            local_frames, digests = [], []
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    state = model.init_state(video_path=str(folder), offload_video_to_cpu=True,
                        offload_state_to_cpu=False, async_loading_frames=False)
                    assert tuple(state["images"].shape) == (21, 3, 1024, 1024)
                    for object_id, box in enumerate(boxes, 1):
                        model.add_new_points_or_box(state, frame_idx=0, obj_id=object_id, box=box)
                    torch.cuda.synchronize()
                    initialized = time.perf_counter()
                    iterator = model.propagate_in_video(state, start_frame_idx=0)
                    for expected in range(21):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        t, object_ids, logits = next(iterator)
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        masks = (logits[:, 0] > 0).cpu().numpy()
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        assert int(t) == expected and list(object_ids) == [1, 2]
                        assert masks.shape == (2, row["size"], row["size"])
                        digests.append(hashlib.sha256(np.packbits(masks).tobytes()).hexdigest())
                        local_frames.append(dict(parent_id=key[0], camera=key[1], size=key[2], repeat=repeat,
                            order_position=position, timestep=expected, propagate_wall_ms=(t1-t0)*1000,
                            bool_cpu_wall_ms=(t2-t1)*1000, total_wall_ms=(t2-t0)*1000))
                    iterator.close()
                    finished = time.perf_counter()
                if not warmup:
                    digest = hashlib.sha256("".join(digests).encode()).hexdigest()
                    if key in hashes:
                        assert hashes[key] == digest, "tracking repeat output changed"
                    hashes[key] = digest
                    frames.extend(local_frames)
                    sequences.append(dict(parent_id=key[0], camera=key[1], size=key[2], repeat=repeat,
                        order_position=position, initialization_wall_ms=(initialized-begin)*1000,
                        processing_wall_ms=(finished-initialized)*1000,
                        steady_mean_wall_ms=float(np.mean([r["total_wall_ms"] for r in local_frames[1:]])),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), masks_sha256=digest))
            finally:
                if state is not None:
                    model.reset_state(state)
                    del state

        for size in SIZES:
            sample((*seqs[0], size), True, -1, -1)
        orders = (SIZES, SIZES[1:]+SIZES[:1], SIZES[2:]+SIZES[:2])
        for repeat in range(protocol["tracking_repeats"]):
            for seq_index, seq in enumerate(seqs):
                for position, size in enumerate(orders[(repeat+seq_index) % 3]):
                    sample((*seq, size), False, repeat, position)
                print("TRACKING", repeat+1, seq, "sequences", len(sequences), flush=True)
                write(root / "tracking_frames.json", frames)
                write(root / "tracking_sequences.json", sequences)
        assert len(frames) == 1512 and len(sequences) == 72 and len(hashes) == 24
        summary = {}
        for size in SIZES:
            steady = [r for r in frames if r["size"] == size and r["timestep"] > 0]
            seq = [r for r in sequences if r["size"] == size]
            summary[str(size)] = dict(steady={k: stat([r[k] for r in steady]) for k in
                ("propagate_wall_ms", "bool_cpu_wall_ms", "total_wall_ms")},
                sequence={k: stat([r[k] for r in seq]) for k in
                ("initialization_wall_ms", "steady_mean_wall_ms", "peak_allocated_bytes")})
        identity.update(after_gpu=gpu_snapshot(), repeat_masks_exact=True,
                        internal_input_shape=[21, 3, 1024, 1024])
        write(root / "tracking_identity.json", identity)
        write(root / "tracking_summary.json", summary)
        write_csv(root / "tracking_frames.csv", frames)
        write_csv(root / "tracking_sequences.csv", sequences)
        logger.log({size: metrics["steady"]["total_wall_ms"]["mean"] for size, metrics in summary.items()})
        logger.link_output(root / "tracking_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "static", "tracking"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        globals()[args.stage](args.root)
    except BaseException as exc:
        if args.root.exists():
            write(args.root / f"{args.stage}_failure.json", dict(error_type=type(exc).__name__, message=str(exc)))
        raise

"""Collect, validate, archive and explicitly publish independent episode shards.

Run python -m data_collection.collect_dataset --config
data_collection/configs/visible_peg.yaml. Existing output directories are refused.
The YAML's publish setting is an explicit publishing workflow; it uses the shared
publisher, and a failed upload leaves all local data and experiment records intact.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

import numpy as np
import yaml

from utils.experiment_artifacts import connect, digests, remote_matches, plan_upload
from utils.experiment_logger import ExperimentLogger
from .storage import write_json


def load_config(path):
    """Read and validate collection settings before starting simulator workers."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("collection config must be a YAML mapping")
    for key in (
        "num_episodes", "episodes_per_shard", "workers", "attempts_per_shard",
        "resolution", "steps", "max_episode_steps",
    ):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["attempts_per_shard"] < config["episodes_per_shard"]:
        raise ValueError("attempts_per_shard must cover episodes_per_shard")
    if not isinstance(config.get("start_seed"), int) or config["start_seed"] < 0:
        raise ValueError("start_seed must be a nonnegative integer")
    if not config.get("require_success") or not config.get("save_step_gt"):
        raise ValueError("this dataset workflow requires success and per-step GT")
    for key in ("output", "policy", "control_mode", "visibility_mode", "device",
                "render_backend", "observation_device"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string")
    return config


def shard_command(config, output, count, seed):
    """Build an argument list for the existing raw episode collector."""
    command = [sys.executable, "-m", "data_collection.collect_episodes"]
    settings = {
        "output": str(output), "num-episodes": count,
        "max-attempts": config["attempts_per_shard"], "start-seed": seed,
    }
    for key in ("resolution", "steps", "max_episode_steps", "visibility_mode",
                "policy", "control_mode", "device", "render_backend", "observation_device"):
        settings[key.replace("_", "-")] = config[key]
    for key, value in settings.items():
        command.extend([f"--{key}", str(value)])
    command.extend(["--require-success", "--save-step-gt"])
    return command


def array_headers(path):
    """Inspect raw NPZ shapes/dtypes without decoding full image payloads again.

    save_frame already read back every value before exposing each source frame.
    This independent pass checks every frame's schema and episode alignment.
    """
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError(f"unexpected NPZ member in {path.name}: {name}")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NumPy header version: {version}")
            if dtype.hasobject:
                raise ValueError(f"object dtype is forbidden in {path.name}: {name}")
            headers[name[:-4]] = (shape, dtype)
    return headers


def verify_initial_visibility(path, mode):
    """Recompute the first Main frame gate from saved raw RGB-D, calibration and GT."""
    from .visibility import CameraObservation, evaluate

    with np.load(path, allow_pickle=False) as frame:
        image = "observation/sensor_data/base_camera/"
        calibration = "observation/sensor_param/base_camera/"
        tcp = frame["ground_truth/ee_pose"][0]
        observation = CameraObservation(
            "verification", 0, "base_camera", 0.0,
            frame[image + "rgb"][0], frame[image + "depth"][0, ..., 0],
            frame[calibration + "intrinsic_cv"][0],
            frame[calibration + "extrinsic_cv"][0],
            frame[calibration + "cam2world_gl"][0], tcp,
        )
        result, _ = evaluate(
            observation, frame["ground_truth/peg_pose"][0],
            frame["ground_truth/peg_half_sizes"][0], tcp,
        )
    key = "requested_pass" if mode == "ends_and_grasp" else "strict_100_pass"
    if not result[key]:
        raise ValueError(f"saved initial frame failed independent visibility check: {path}")


def verify_shard(root, config, expected_count):
    """Check counts, unique seeds, raw image schemas, T+1/T alignment and success."""
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = manifest["episodes"]
    if manifest["status"] != "COMPLETE" or len(episodes) != expected_count:
        raise ValueError(f"incomplete collection shard: {root}")
    seeds = [episode["seed"] for episode in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seed in shard: {root}")
    resolution = config["resolution"]
    frames_checked = 0
    for episode in episodes:
        directory = root / episode["path"]
        if not episode["success"] or episode["observations"] != episode["steps"] + 1:
            raise ValueError(f"unsuccessful or misaligned episode: {directory}")
        paths = sorted(directory.glob("frame_*.npz"))
        if len(paths) != episode["observations"]:
            raise ValueError(f"frame count mismatch: {directory}")
        verify_initial_visibility(paths[0], config["visibility_mode"])
        for index, path in enumerate(paths):
            if path.name != f"frame_{index:06d}.npz":
                raise ValueError(f"nonconsecutive frame sequence: {directory}")
            headers = array_headers(path)
            for camera in ("base_camera", "hand_camera"):
                prefix = f"observation/sensor_data/{camera}/"
                expected = {
                    "rgb": ((1, resolution, resolution, 3), np.dtype("uint8")),
                    "depth": ((1, resolution, resolution, 1), np.dtype("int16")),
                }
                for channel, schema in expected.items():
                    if headers.get(prefix + channel) != schema:
                        raise ValueError(f"raw {camera}/{channel} shape/dtype mismatch: {path}")
            for key in ("ee_pose", "peg_pose", "hole_pose", "qpos", "success"):
                if f"ground_truth/{key}" not in headers:
                    raise ValueError(f"missing per-frame GT {key}: {path}")
            if ("action" in headers) != (index > 0):
                raise ValueError(f"action/observation alignment mismatch: {path}")
            if index > 0:
                with np.load(path, allow_pickle=False) as frame:
                    action = frame["action"]
                    if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                        raise ValueError(f"invalid normalized native action: {path}")
            frames_checked += 1
        with np.load(paths[-1], allow_pickle=False) as frame:
            if not bool(frame["info/success"].any()):
                raise ValueError(f"final observation does not confirm task success: {directory}")
    return {
        "status": "PASS", "episodes": len(episodes), "frames": frames_checked,
        "attempted": manifest["attempted"], "seeds": seeds,
        "visibility_passed": manifest["visibility_passed"],
        "expert_attempted": manifest["expert_attempted"],
        "expert_failed": manifest["expert_failed"],
    }


def collect_shard(config, root, index, count):
    """Collect and independently verify one bounded, disjoint seed range."""
    name = f"shard_{index:06d}"
    directory = root / "shards" / name
    seed = config["start_seed"] + index * config["attempts_per_shard"]
    command = shard_command(config, directory, count, seed)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (root / "logs" / f"{name}.log").open("w") as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"collector {name} failed; inspect logs/{name}.log")
    verification = verify_shard(directory, config, count)
    write_json(directory / "verification.json", verification)
    archive = root / "archives" / f"{name}.tar"
    manifest = json.loads((directory / "manifest.json").read_text())
    # NPZ frames are already compressed. Archive only approved episodes and their
    # audit/config files; failed expert rollouts remain local diagnostic evidence.
    with tarfile.open(archive.with_suffix(".partial"), "w") as bundle:
        for name_in_shard in ("config.json", "manifest.json", "attempts.jsonl", "verification.json"):
            bundle.add(directory / name_in_shard, arcname=f"{name}/{name_in_shard}")
        for episode in manifest["episodes"]:
            bundle.add(directory / episode["path"], arcname=f"{name}/{episode['path']}")
    archive.with_suffix(".partial").rename(archive)
    return {"path": str(archive), "shard": name, "verification": verification,
            "episodes": manifest["episodes"]}


def write_dataset_readme(root, config):
    """Document the actual selection, native arrays, action alignment and limits."""
    text = f"""# Visible peg expert dataset

{config['num_episodes']} independent successful GT-guided scripted expert episodes
from ManiSkill PegInsertionSide-v1. Both Main (base_camera) and Wrist
(hand_camera) sensors render natively at {config['resolution']} x {config['resolution']}.
Physics: CPU PhysX. Rendering: NVIDIA CUDA, Vast.ai instance 59288396.

Selection uses only the initial Main frame: the peg's eight GT corners must lie
in view, and both 17.5 mm end strips plus the expert grasp band must have valid
raw depth matching the GT front surface within 1.5 mm. Later visibility is not
guaranteed. Only trajectories reaching ManiSkill's official success condition
are included. This release has no automatic train/validation/test assignment.

Extract the TAR shards into one directory. dataset_manifest.json lists the
approved episode references and disjoint reset seeds; each shard also has
its own config, attempts, episode receipts and independent verification report.
Frames are NPZ, loadable using numpy.load(path, allow_pickle=False).

- observation/sensor_data/<camera>/rgb: [1, H, W, 3], uint8 RGB.
- observation/sensor_data/<camera>/depth: [1, H, W, 1], signed int16 millimetres.
  Raw nonpositive/overflow depth values are preserved, not repaired.
- observation/sensor_param/<camera>/: native calibration matrices.
- ground_truth/: raw simulator GT at every frame, world poses in metres and
  quaternions ordered qw,qx,qy,qz, including peg/box/hole, fingers, robot and success.
- Frame 0 is the accepted initial observation. Frame t+1 contains observation
  o[t+1], action a[t], reward, flags, info and GT. There are T+1 frames and T actions.
- Actions use ManiSkill 3.0.1 pd_ee_delta_pose: normalized root-frame xyz,
  rotation-vector components, gripper; bounds [-1,1]. Its installed rotational
  scaling has a negative sign. Gripper -1 closes; +1 opens.

The policy uses privileged simulator GT; it is not a learned visual policy.
The initial gate is also GT-assisted. No physical-state teleportation is applied
between selection and actions. Success filtering changes the reset distribution.
The dataset is not a held-out policy evaluation and no model training is included.
The horizon is {config['max_episode_steps']} control steps. Source snapshot and
package/runtime records accompany the publication. Existing historical datasets
and their partitions were not modified. Authentication is needed for the private
Hugging Face destination.
"""
    (root / "README.md").write_text(text)


def verify_publication(directory, include):
    """Confirm every remote size/hash at the committed publication revision."""
    api, _, reason = connect()
    if reason:
        raise RuntimeError("Hugging Face authentication unavailable for verification")
    receipt = json.loads((directory / "huggingface_uploads.json").read_text())
    checks = []
    for plan in plan_upload(directory, include).values():
        repository = plan["repository"]
        matches = [publication for publication in receipt["publications"].values()
                   if publication["repo_id"] == repository.repo_id
                   and publication["manifest_path"] == plan["manifest_path"]]
        if len(matches) != 1:
            raise ValueError("missing unique publication receipt")
        publication = matches[0]
        remote = {item.path: item for item in api.list_repo_tree(
            repository.repo_id, repo_type=repository.repo_type,
            path_in_repo=plan["prefix"], revision=publication["revision"], recursive=True,
        ) if hasattr(item, "blob_id")}
        for item in plan["files"]:
            if item.path not in remote or not remote_matches(item, remote[item.path]):
                raise ValueError(f"remote file hash/size verification failed: {item.path}")
        checks.append({"url": publication["url"], "revision": publication["revision"],
                       "files": len(plan["files"]), "status": "PASS"})
    return {"status": "PASS", "publications": checks}


def run(config):
    """Complete collection before publishing; persist reviewable progress throughout."""
    root = Path(config["output"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("shards", "logs", "archives"):
        (root / name).mkdir()
    write_json(root / "effective_config.json", config)
    write_dataset_readme(root, config)
    started = time.monotonic()
    progress = {"status": "COLLECTING", "episodes": 0, "completed_shards": 0}
    write_json(root / "progress.json", progress)
    with ExperimentLogger(
        team="integration", config=config, seed=config["start_seed"],
        model="scripted_peg_policy", job_type="collection",
    ) as logger:
        write_json(root / "run_identity.json", {"run_id": logger.run_id,
                                                "directory": str(logger.directory)})
        logger.record(environment={"env_id": "PegInsertionSide-v1", "vast_instance": 59288396})
        counts = []
        remaining = config["num_episodes"]
        while remaining:
            count = min(config["episodes_per_shard"], remaining)
            counts.append(count)
            remaining -= count
        completed = []
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [executor.submit(collect_shard, config, root, index, count)
                       for index, count in enumerate(counts)]
            for future in as_completed(futures):
                completed.append(future.result())
                progress.update(
                    episodes=sum(item["verification"]["episodes"] for item in completed),
                    completed_shards=len(completed), seconds=time.monotonic() - started,
                )
                write_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        completed.sort(key=lambda item: item["shard"])
        episodes = []
        archives = []
        for shard in completed:
            size, digest, _ = digests(Path(shard["path"]))
            archives.append({"file": Path(shard["path"]).name, "size": size,
                             "sha256": digest, "shard": shard["shard"]})
            for episode in shard["episodes"]:
                episodes.append(dict(episode, path=f"{shard['shard']}/{episode['path']}"))
        seeds = [episode["seed"] for episode in episodes]
        if len(episodes) != config["num_episodes"] or len(set(seeds)) != len(seeds):
            raise ValueError("dataset count or cross-shard seed uniqueness failed")
        manifest = {"status": "COMPLETE", "run_id": logger.run_id, "config": config,
                    "episode_count": len(episodes), "episodes": episodes, "archives": archives,
                    "frames": sum(item["verification"]["frames"] for item in completed),
                    "verification": "all frames checked; every initial visibility recomputed"}
        manifest["selection_counts"] = {
            key: sum(item["verification"][key] for item in completed)
            for key in ("attempted", "visibility_passed", "expert_attempted", "expert_failed")
        }
        write_json(root / "dataset_manifest.json", manifest)
        metadata_archive = root / "archives" / "dataset_metadata.tar.gz"
        with tarfile.open(metadata_archive, "w:gz") as bundle:
            for name in ("README.md", "effective_config.json", "dataset_manifest.json", "run_identity.json"):
                bundle.add(root / name, arcname=name)
            source = Path("collection_source.tar.gz")
            if source.is_file():
                bundle.add(source, arcname="collection_source.tar.gz")
        for path in [metadata_archive, *[Path(item["path"]) for item in completed]]:
            logger.link_output(path, kind="trajectory")
        logger.link_output(root / "dataset_manifest.json", kind="episode_metadata")
        logger.log({"episodes": len(episodes), "frames": manifest["frames"]})
        logger.record(task="visible_peg_expert_collection", collection_seconds=time.monotonic() - started)
    progress.update(status="COLLECTED", seconds=time.monotonic() - started)
    write_json(root / "progress.json", progress)
    if config.get("publish"):
        progress["status"] = "PUBLISHING"
        write_json(root / "progress.json", progress)
        command = [sys.executable, "-m", "tools.publish_experiment", "--run-id", logger.run_id,
                   "--include", "trajectory", "metrics"]
        if config.get("private"):
            command.append("--private")
        with (root / "publication.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verification = verify_publication(logger.directory, ["trajectory", "metrics"])
        write_json(root / "publication_verification.json", verification)
        progress.update(status="COMPLETE", publication=verification, seconds=time.monotonic() - started)
        write_json(root / "progress.json", progress)
    print(json.dumps(progress), flush=True)


def main():
    """Read the requested config and run the durable collection workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

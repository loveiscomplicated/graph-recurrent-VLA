"""Task 01: inspect saved tracking/RGB-D geometry on CPU, without model inference.

Writes a fresh ExperimentLogger run with per-frame metrics, selected dense NPZs,
nine-panel PNGs, and an HTML index. RGB is used only in diagnostic PNGs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import html
from itertools import chain
import json
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.experiment_artifacts import digests  # noqa: E402
from utils.experiment_logger import ExperimentLogger  # noqa: E402
from visual.runtime.geometry import GeometryParameters, process_geometry  # noqa: E402
from visual.runtime.rgbd import BOX, PEG, iter_saved_rgbd  # noqa: E402
from visual.scripts.pointcloud import to_world  # noqa: E402

OBJECTS = {"peg": PEG, "box": BOX}
COLORS = {PEG: np.array([255, 85, 65]), BOX: np.array([60, 160, 255])}


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), mean=float(values.mean()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95))) if values.size else dict(count=0)


def summarize_frame(frame, result, segmentation, gt_ids, variation_degrees):
    """GT only diagnoses mask support here; it never enters runtime geometry."""
    record = dict(source_frame=frame.source_frame, frame_idx=frame.frame_idx,
                  timestamp=frame.timestamp, valid_depth_pixels=int(result.geometry.valid_depth.sum()),
                  overlap_pixels=int(result.geometry.overlap_mask.sum()), objects={})
    for name, object_id in OBJECTS.items():
        f = result.objects[object_id]
        high_variation = f.normal_variation_valid & (np.rad2deg(f.normal_variation_rad) > variation_degrees)
        row = {key: int(getattr(f, key).sum()) for key in
               ("object_mask", "valid_points", "normal_valid", "depth_pair_valid", "depth_discontinuity",
                "mask_boundary", "depth_validity_boundary", "image_boundary", "normal_variation_valid")}
        row.update(high_variation_pixels=int(high_variation.sum()),
                   normal_variation_degrees=statistics(np.rad2deg(f.normal_variation_rad)),
                   depth_jump_mm=statistics(f.depth_jump_m * 1000),
                   jump_on_mask_boundary=int((f.depth_discontinuity & f.mask_boundary).sum()),
                   jump_away_from_mask_boundary=int((f.depth_discontinuity & ~f.mask_boundary).sum()),
                   observation_state=("empty_mask" if not f.object_mask.any() else
                                      "no_valid_depth" if not (f.object_mask & result.geometry.valid_depth).any() else
                                      "ambiguous_only" if not f.valid_points.any() else "observed"))
        points = result.geometry.xyz[f.valid_points]
        row["xyz_cv_min_m"] = points.min(axis=0).tolist() if len(points) else None
        row["xyz_cv_max_m"] = points.max(axis=0).tolist() if len(points) else None
        if segmentation is not None:
            gt = segmentation[frame.source_frame] == gt_ids[name]
            union = f.object_mask | gt
            row.update(gt_pixels=int(gt.sum()),
                       mask_iou=float((f.object_mask & gt).sum() / union.sum()) if union.any() else None,
                       supported_pixels_off_gt=int((f.valid_points & ~gt).sum()),
                       jump_pixels_off_gt=int((f.depth_discontinuity & ~gt).sum()),
                       high_variation_pixels_off_gt=int((high_variation & ~gt).sum()))
        record["objects"][name] = row
    return record


def save_geometry(path, frame, result):
    """Dense debug artifact, with the same C-order pixel index for every field."""
    arrays = dict(xyz=result.geometry.xyz, valid_depth=result.geometry.valid_depth,
                  overlap_mask=result.geometry.overlap_mask, source_frame=frame.source_frame,
                  intrinsic=frame.intrinsic, cam2world=frame.cam2world, camera=frame.camera)
    if frame.frame_idx is not None:
        arrays["frame_idx"] = frame.frame_idx
    if frame.timestamp is not None:
        arrays["timestamp"] = frame.timestamp
    for name, object_id in OBJECTS.items():
        for field in fields(result.objects[object_id]):
            arrays[f"{name}/{field.name}"] = getattr(result.objects[object_id], field.name)
    np.savez_compressed(path, **arrays)


def render_geometry(path, frame, result, record, variation_degrees):
    """All supported object points are plotted; no geometry sampling is applied."""
    shape = frame.depth.shape
    mask_rgb = frame.rgb.astype(float).copy()
    normal_rgb = np.zeros((*shape, 3), np.float32)
    jump_mm = np.full(shape, np.nan, np.float32)
    variation_deg = np.full(shape, np.nan, np.float32)
    boundary_rgb = np.zeros((*shape, 3), np.float32)
    support_rgb = np.zeros((*shape, 3), np.float32)
    overlay = frame.rgb.astype(float) * 0.55
    union = np.zeros(shape, bool)
    for object_id, f in result.objects.items():
        union |= f.object_mask
        mask_rgb[f.object_mask] = mask_rgb[f.object_mask] * 0.5 + COLORS[object_id] * 0.5
        normal_rgb[f.normal_valid] = (f.normals[f.normal_valid] + 1) / 2
        jump_mm[f.depth_pair_valid] = f.depth_jump_m[f.depth_pair_valid] * 1000
        variation_deg[f.normal_variation_valid] = np.rad2deg(f.normal_variation_rad[f.normal_variation_valid])
        # Additive RGB channels explicitly show coinciding boundary causes.
        boundary_rgb[..., 0] += f.depth_discontinuity
        boundary_rgb[..., 1] += f.mask_boundary
        boundary_rgb[..., 2] += f.depth_validity_boundary
        support_rgb[f.valid_points] = [0.45, 0.45, 0.45]
        support_rgb[f.normal_valid] = [0.15, 0.85, 0.3]
        overlay[f.depth_discontinuity] = [255, 220, 0]
        high_variation = f.normal_variation_valid & (np.rad2deg(f.normal_variation_rad) > variation_degrees)
        overlay[high_variation] = [255, 65, 220]
        overlay[high_variation & f.depth_discontinuity] = [255, 255, 255]
    overlap = result.geometry.overlap_mask
    mask_rgb[overlap] = [255, 155, 0]
    support_rgb[overlap] = [1, 0.6, 0]

    fig = plt.figure(figsize=(17, 15))
    fig.subplots_adjust(left=0.025, right=0.97, bottom=0.035, top=0.91, wspace=0.22, hspace=0.25)
    axes = [fig.add_subplot(3, 3, i + 1, projection="3d" if i == 2 else None) for i in range(9)]
    axes[0].imshow(frame.rgb)
    axes[0].set_title("Original RGB (full frame)")
    axes[1].imshow(mask_rgb.astype(np.uint8))
    axes[1].set_title("Tracking masks: Peg red / Box blue / overlap orange")
    ax = axes[2]
    clouds = []
    for object_id, f in result.objects.items():
        cloud = to_world(result.geometry.xyz[f.valid_points], frame.cam2world)
        if len(cloud):
            clouds.append(cloud)
            ax.scatter(*cloud.T, s=0.5, color=COLORS[object_id] / 255, depthshade=False)
    if clouds:
        cloud = np.concatenate(clouds)
        # A few finite far-depth mask contaminants otherwise hide the entire
        # object. Only the plotted viewport is cropped; ALL points are plotted,
        # analysed and saved, and the count outside the viewport is disclosed.
        low, high = np.percentile(cloud, [1, 99], axis=0)
        centre = (low + high) / 2
        radius = max(float((high - low).max()) * 0.55, 0.01)
        outside_view = int((np.abs(cloud - centre) > radius).any(axis=1).sum())
        ax.set_xlim(centre[0] - radius, centre[0] + radius)
        ax.set_ylim(centre[1] - radius, centre[1] + radius)
        ax.set_zlim(centre[2] - radius, centre[2] + radius)
        ax.text2D(0.02, 0.93, f"View crop only: {outside_view}/{len(cloud)} points outside\nFull XYZ retained in NPZ",
                  transform=ax.transAxes, fontsize=7, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-125)
    ax.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)",
           title="Organized XYZ: world view (1-99% viewport)")
    axes[3].imshow(normal_rgb)
    axes[3].set_title("CV normal RGB=(n+1)/2; unsupported black")
    heat = axes[4].imshow(jump_mm, cmap="magma", vmin=0, vmax=50)
    fig.colorbar(heat, ax=axes[4], label="max adjacent z jump (mm); clip 50", shrink=0.7)
    axes[4].set_title("Depth change (all finite neighbour labels)")
    axes[5].imshow(np.clip(boundary_rgb, 0, 1))
    axes[5].set_title("Boundary causes: additive channels, no edge labels")
    axes[5].legend(handles=[Patch(color="red", label="z jump"), Patch(color="lime", label="mask ends"),
                            Patch(color="blue", label="invalid depth")], fontsize=7,
                   loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    heat = axes[6].imshow(variation_deg, cmap="viridis", vmin=0, vmax=45)
    fig.colorbar(heat, ax=axes[6], label="max normal angle (deg); clip 45", shrink=0.7)
    axes[6].set_title("Normal variation; unsupported blank")
    axes[7].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[7].set_title(f"RGB debug: jump yellow / angle > {variation_degrees:g} deg magenta")
    axes[8].imshow(support_rgb)
    axes[8].set_title("Normal valid green / unsupported gray / overlap orange")
    if union.any():
        v, u = np.nonzero(union)
        crop = (max(0, int(u.min()) - 25), min(shape[1], int(u.max()) + 26),
                max(0, int(v.min()) - 25), min(shape[0], int(v.max()) + 26))
    else:
        crop = (0, shape[1], 0, shape[0])
    for i, ax in enumerate(axes):
        ax.title.set_fontsize(10)
        if i != 2:
            ax.axis("off")
            if i != 0:
                ax.set_xlim(crop[0] - 0.5, crop[1] - 0.5)
                ax.set_ylim(crop[3] - 0.5, crop[2] - 0.5)
    counts = " | ".join(f"{name}: mask={r['object_mask']}, points={r['valid_points']}, normals={r['normal_valid']}"
                        for name, r in record["objects"].items())
    fig.suptitle(f"source={frame.source_frame} step={frame.frame_idx} t={frame.timestamp} s\n{counts}", fontsize=12, y=0.98)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(path, data):
    figures = "\n".join(
        f'<h2>Source frame {source}</h2><a href="f{source:04d}.npz">Pixel-aligned geometry NPZ (no RGB)</a>'
        f'<br><a href="f{source:04d}.png"><img src="f{source:04d}.png" alt="Geometry diagnostics frame {source}"></a>'
        for source in data["selected_source_frames"])
    rows = "".join(f"<tr><td>{key}</td><td>{row['mean']:.3f}</td><td>{row['median']:.3f}</td>"
                   f"<td>{row['p95']:.3f}</td></tr>" for key, row in data["timing_ms"].items())
    path.write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Task 01 organized geometry</title><style>body{{font:15px system-ui;max-width:1500px;margin:24px auto;padding:0 16px}}
img{{width:100%}}td,th{{padding:6px 18px;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>Task 01: organized geometry diagnostics</h1>
<p>Run {data['run_id']} · {data['evaluated_frames']} frames · CPU NumPy · GPU NOT RUN</p>
<p>CV-camera XYZ retains source pixels; 3D panel uses the recorded CV-to-world transform.
Its viewport uses 1-99 percentiles plus margin so finite far-depth contaminants do not hide the objects;
the outside-view count is shown. All points remain plotted, analysed and saved in the NPZ.
No erosion, SOR, pre-selection, FPS, structural classification, graph construction or cache.
Empty masks produce no object points. Overlapping masks retain observations but are excluded from both object supports.</p>
<p>Red/green/blue boundary channels mean z jump / mask boundary / invalid-depth boundary;
mixed colors indicate simultaneous causes. A mask contour is not a structural edge.
Magenta overlay is an angle diagnostic; curved surfaces and tracking contamination also produce it.
The {data['config']['display_variation_degrees']:g}-degree overlay threshold is display-only. Normals need a complete same-object 5-pixel stencil;
unsupported pixels remain NaN. All heatmaps use fixed scales across frames.</p>
<table><tr><th>CPU stage (ms)</th><th>Mean</th><th>Median</th><th>p95</th></tr>{rows}</table>
<p>Geometry includes full-image backprojection, masks and validation; normals/features include both objects.
Timings exclude loading, warmup, rendering, serialization and logging. Runs are sequential, with the repeat count in settings.
GT is read only by this diagnostic runner to report mask support; it provides no structural edge truth.</p>
<p><a href="geometry.json">Full frame metrics, timings, settings and input hashes</a></p>
{figures}<details><summary>Settings</summary><pre>{html.escape(json.dumps(data['config'], indent=2))}</pre></details>
</html>''', encoding="utf-8")


def evaluate(episode_path, masks_path, *, camera="base_camera", output_root,
             selected_sources=(0, 15, 18, 45, 80, 120), repeats=3, parameters=None,
             variation_degrees=20.0):
    """Inspect all saved mask frames; save dense geometry for selected source rows."""
    parameters = parameters or GeometryParameters()
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not np.isfinite(variation_degrees) or not 0 <= variation_degrees <= 180:
        raise ValueError("overlay normal angle must be in [0,180] degrees")
    episode_path, masks_path = Path(episode_path).resolve(), Path(masks_path).resolve()
    with np.load(masks_path, allow_pickle=False) as archive:
        source = archive["source_frame"]
    if source.ndim != 1 or not len(source) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("source_frame must be nonempty one-dimensional integers")
    if len(np.unique(source)) != len(source):
        raise ValueError("source_frame must be unique for diagnostic artifacts")
    selected = sorted(set(selected_sources))
    if set(selected) - set(source.tolist()):
        raise ValueError("selected sources must occur in masks.npz source_frame")
    input_hash, masks_hash = digests(episode_path)[1], digests(masks_path)[1]
    tracking_path = masks_path.with_name("tracking.json")
    if tracking_path.is_file():
        tracking = json.loads(tracking_path.read_text())
        if tracking.get("camera") not in (None, camera):
            raise ValueError("camera differs from tracking.json")
        original_path = Path(tracking.get("input") or episode_path)
        if original_path.is_file() and original_path.resolve() != episode_path:
            if digests(original_path)[1] != input_hash:
                raise ValueError("episode differs from tracking.json")
    with np.load(episode_path, allow_pickle=False) as archive:
        has_gt = all(key in archive for key in (f"{camera}/seg", "peg_id", "box_id"))
        segmentation = archive[f"{camera}/seg"] if has_gt else None
        gt_ids = {name: int(archive[f"{name}_id"]) for name in OBJECTS} if has_gt else {}
        seed = int(archive["seed"]) if "seed" in archive else None
    iterator = iter_saved_rgbd(episode_path, masks_path, camera=camera)
    first = next(iterator)
    if segmentation is not None and segmentation.shape != (len(segmentation), *first.depth.shape):
        raise ValueError("GT segmentation resolution differs from depth")
    for _ in range(3):
        process_geometry(first, parameters)  # Warmup excluded from timing statistics.
    config = dict(input=str(episode_path), masks=str(masks_path), camera=camera,
                  input_sha256=input_hash, masks_sha256=masks_hash, parameters=asdict(parameters),
                  selected_sources=selected, timing_repeats=repeats, warmup_calls=3,
                  display_variation_degrees=variation_degrees,
                  device="cpu", gpu_executed=False, numpy=np.__version__,
                  python=platform.python_version(), platform=platform.platform(),
                  machine=platform.machine(), gt_segmentation_ids=gt_ids)
    with ExperimentLogger(team="visual", config=config, seed=seed,
                          model="organized_geometry_numpy_baseline", job_type="evaluate") as experiment:
        output = Path(output_root).resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        records, timings = [], []
        for frame in chain((first,), iterator):
            per_frame_timings = []
            for _ in range(repeats):
                result = process_geometry(frame, parameters)
                per_frame_timings.append(result.timings_ms)
            timings.extend(per_frame_timings)
            record = summarize_frame(frame, result, segmentation, gt_ids, variation_degrees)
            record["timings_ms"] = per_frame_timings
            records.append(record)
            if frame.source_frame in selected:
                npz = output / f"f{frame.source_frame:04d}.npz"
                png = npz.with_suffix(".png")
                save_geometry(npz, frame, result)
                render_geometry(png, frame, result, record, variation_degrees)
                experiment.link_output(npz)
                experiment.link_output(png)
            if len(records) == 1 or len(records) % 20 == 0:
                print(f"geometry evaluated {len(records)}/{len(source)} frames", flush=True)
        timing_summary = {key: statistics([row[key] for row in timings]) for key in timings[0]}
        summary = {name: dict(empty_mask_frames=sum(r["objects"][name]["object_mask"] == 0 for r in records),
                              no_points_frames=sum(r["objects"][name]["valid_points"] == 0 for r in records),
                              no_normals_frames=sum(r["objects"][name]["normal_valid"] == 0 for r in records))
                   for name in OBJECTS}
        data = dict(run_id=experiment.run_id, config=config, evaluated_frames=len(records),
                    selected_source_frames=selected, timing_ms=timing_summary, summary=summary, frames=records,
                    git_commit=experiment.metadata["git_commit"], git_dirty=experiment.metadata["git_dirty"])
        json_path = output / "geometry.json"
        json_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        write_report(output / "geometry.html", data)
        experiment.link_output(json_path, kind="metrics")
        experiment.link_output(output / "geometry.html")
        experiment.log(timing_summary, prefix="eval/geometry_cpu_ms")
        experiment.log(summary, prefix="eval/geometry_support")
        experiment.record(dataset_sha256=input_hash, masks_sha256=masks_hash,
                          evaluated_frames=len(records), evaluation_device="cpu", gpu_executed=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--out", type=Path, default=REPO / "visual/results/runtime_geometry")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 15, 18, 45, 80, 120], help="Source rows")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--depth-jump-abs-m", type=float, default=0.01)
    parser.add_argument("--depth-jump-rel", type=float, default=0.02)
    parser.add_argument("--overlay-angle-deg", type=float, default=20)
    args = parser.parse_args()
    try:
        output = evaluate(args.input, args.masks, camera=args.camera, output_root=args.out,
                          selected_sources=args.frames, repeats=args.repeats,
                          parameters=GeometryParameters(args.depth_jump_abs_m, args.depth_jump_rel),
                          variation_degrees=args.overlay_angle_deg)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"saved geometry report: {output / 'geometry.html'}")


if __name__ == "__main__":
    main()

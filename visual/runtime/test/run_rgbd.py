"""Evaluate saved tracker masks against GT and write a report plus depth video.

Run from the repository root::

    WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_rgbd.py \
        --input visual/data/step2/ep000.npz \
        --masks visual/results/runtime_tracking/ep000/masks.npz

GT is read only here, after Visual output has been produced by rgbd.py.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
from itertools import chain
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.experiment_logger import ExperimentLogger  # noqa: E402
from visual.runtime.rgbd import BOX, PEG, TrackedRGBDFrame, iter_saved_rgbd  # noqa: E402
from visual.scripts.pointcloud import backproject, to_world  # noqa: E402


OBJECTS = {"peg": PEG, "box": BOX}
COLORS = {"peg": (255, 100, 90), "box": (60, 205, 255)}  # RGB


def compare_object(frame: TrackedRGBDFrame, object_id: int, gt_mask: np.ndarray) -> dict:
    """Compare masks and unfiltered visible surfaces; distances are in mm.

    Geometry uses every pixel with finite positive depth, with no erosion,
    sampling, or pose fitting. Missing surfaces yield None, never zero error.
    """
    predicted = frame.masks[object_id]
    if gt_mask.shape != frame.depth.shape or gt_mask.dtype != np.bool_:
        raise ValueError("GT mask must be boolean and match the depth resolution")
    predicted_pixels, gt_pixels = int(predicted.sum()), int(gt_mask.sum())
    tp = int(np.count_nonzero(predicted & gt_mask))
    union = predicted_pixels + gt_pixels - tp
    valid_depth = np.isfinite(frame.depth) & (frame.depth > 0)
    predicted_valid = predicted & valid_depth
    gt_valid = gt_mask & valid_depth
    row = dict(
        predicted_pixels=predicted_pixels, gt_pixels=gt_pixels,
        tp=tp, fp=predicted_pixels - tp, fn=gt_pixels - tp,
        iou=tp / union if union else None,
        precision=tp / predicted_pixels if predicted_pixels else None,
        recall=tp / gt_pixels if gt_pixels else None,
        predicted_depth_pixels=int(predicted_valid.sum()),
        gt_depth_pixels=int(gt_valid.sum()),
        predicted_valid_depth_fraction=float(predicted_valid.sum() / predicted_pixels)
        if predicted_pixels else None,
        median_depth_m=float(np.median(frame.depth[predicted_valid]))
        if predicted_valid.any() else None,
        centroid_error_mm=None, chamfer_mm=None,
        predicted_to_gt_mm=None, gt_to_predicted_mm=None,
    )
    if predicted_valid.any() and gt_valid.any():
        predicted_world = to_world(
            backproject(frame.depth, frame.intrinsic, predicted_valid), frame.cam2world
        )
        gt_world = to_world(
            backproject(frame.depth, frame.intrinsic, gt_valid), frame.cam2world
        )
        predicted_distance = cKDTree(gt_world).query(predicted_world)[0]
        gt_distance = cKDTree(predicted_world).query(gt_world)[0]
        row.update(
            centroid_error_mm=float(np.linalg.norm(
                predicted_world.mean(axis=0, dtype=np.float64)
                - gt_world.mean(axis=0, dtype=np.float64)
            ) * 1000),
            predicted_to_gt_mm=float(predicted_distance.mean() * 1000),
            gt_to_predicted_mm=float(gt_distance.mean() * 1000),
            chamfer_mm=float((predicted_distance.mean() + gt_distance.mean()) * 500),
        )
    return row


def statistics(values) -> dict:
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None)
    return dict(count=len(values), mean=float(values.mean()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)))


def summarize(records: list[dict]) -> dict:
    summary = {}
    for name in OBJECTS:
        rows = [record["objects"][name] for record in records]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        visible = [row for row in rows if row["gt_pixels"] > 0]
        summary[name] = dict(
            frames=len(rows), visible_frames=len(visible),
            missed_visible_frames=sum(row["predicted_pixels"] == 0 for row in visible),
            invisible_frames=len(rows) - len(visible),
            false_positive_invisible_frames=sum(
                row["gt_pixels"] == 0 and row["predicted_pixels"] > 0 for row in rows
            ),
            geometry_unavailable_visible_frames=sum(row["chamfer_mm"] is None for row in visible),
            tp=tp, fp=fp, fn=fn,
            micro_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
            micro_precision=tp / (tp + fp) if tp + fp else None,
            micro_recall=tp / (tp + fn) if tp + fn else None,
            visible_iou=statistics(row["iou"] for row in visible),
            centroid_error_mm=statistics(row["centroid_error_mm"] for row in rows),
            chamfer_mm=statistics(row["chamfer_mm"] for row in rows),
            predicted_to_gt_mm=statistics(row["predicted_to_gt_mm"] for row in rows),
            gt_to_predicted_mm=statistics(row["gt_to_predicted_mm"] for row in rows),
        )
    return summary


def depth_colors(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Fixed metre scale for the whole video; invalid depth is black."""
    if not np.isfinite([near, far]).all() or not 0 <= near < far:
        raise ValueError("depth range must be finite and satisfy 0 <= near < far")
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros(depth.shape, dtype=np.float32)
    scaled[valid] = np.clip((depth[valid] - near) / (far - near), 0, 1)
    colored = cv2.cvtColor(
        cv2.applyColorMap(np.rint(scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    colored[~valid] = 0
    return colored


def _text(image, text, xy, scale=0.55, color=(235, 240, 245)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _outline(image, mask, color, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def _number(value, decimals=2):
    return "n/a" if value is None else f"{value:.{decimals}f}"


def render_depth_frame(frame: TrackedRGBDFrame, gt_masks: dict[str, np.ndarray],
                       metrics: dict, near: float, far: float) -> np.ndarray:
    """RGB tracking, RGB with depth overlay, and isolated object depth panels."""
    colored = depth_colors(frame.depth, near, far)
    valid = np.isfinite(frame.depth) & (frame.depth > 0)
    rgb = frame.rgb.copy()
    overlay = frame.rgb.copy()
    overlay[valid] = (0.3 * overlay[valid] + 0.7 * colored[valid]).astype(np.uint8)
    union = frame.masks[PEG] | frame.masks[BOX]
    isolated = colored.copy()
    isolated[~union] = 0
    for name, object_id in OBJECTS.items():
        _outline(rgb, gt_masks[name], (255, 255, 255), 1)
        _outline(rgb, frame.masks[object_id], COLORS[name], 2)
        _outline(overlay, frame.masks[object_id], COLORS[name], 2)
        _outline(isolated, frame.masks[object_id], COLORS[name], 1)

    # Keep labels readable for lower-resolution inputs too.
    panel_width = max(384, frame.rgb.shape[1])
    panel_width += panel_width % 2  # H.264 yuv420 needs an even canvas width.
    panel_height = round(frame.rgb.shape[0] * panel_width / frame.rgb.shape[1])
    panel_height += panel_height % 2
    header, footer = 64, 144
    canvas = np.full((header + panel_height + footer, panel_width * 3, 3),
                     (18, 23, 32), dtype=np.uint8)
    titles = ("RGB | predicted masks + white GT", "RGB + depth overlay", "Tracked-object depth")
    for column, (panel, title) in enumerate(zip((rgb, overlay, isolated), titles)):
        x = column * panel_width
        canvas[header:header + panel_height, x:x + panel_width] = cv2.resize(
            panel, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST
        )
        _text(canvas, title, (x + 12, 23), 0.55)
    timing = f"source row {frame.source_frame} | step {frame.frame_idx} | t={_number(frame.timestamp, 3)}s | {frame.camera}"
    _text(canvas, timing, (12, 49), 0.5)
    top = header + panel_height
    for line, name in enumerate(OBJECTS):
        row = metrics[name]
        label = (f"{name.upper()} | IoU {_number(row['iou'], 3)}"
                 f" | Chamfer {_number(row['chamfer_mm'])} mm"
                 f" | centroid {_number(row['centroid_error_mm'])} mm"
                 f" | median z {_number(row['median_depth_m'], 3)} m")
        _text(canvas, label, (12, top + 23 + 24 * line), 0.55, COLORS[name])
    bar_width = panel_width - 32
    gradient = np.linspace(near, far, bar_width, dtype=np.float32)[None, :]
    canvas[top + 66:top + 82, 16:16 + bar_width] = depth_colors(
        np.repeat(gradient, 16, axis=0), near, far
    )
    _text(canvas, f"{near:.2f} m", (16, top + 104), 0.5)
    _text(canvas, f"{far:.2f} m", (bar_width - 38, top + 104), 0.5)
    _text(canvas, "Fixed CV camera z scale | black = invalid / unselected",
          (panel_width + 12, top + 80), 0.5)
    _text(canvas, "Colors clip at scale limits; metric calculations use all finite positive depths.",
          (panel_width + 12, top + 103), 0.46)
    _text(canvas, "Surface differences use the same recorded depth; no independent depth prediction is evaluated.",
          (12, top + 130), 0.5)
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, data: dict) -> None:
    """Write one portable HTML containing numerical tables, plots, and raw data."""
    import matplotlib.pyplot as plt

    records, summary = data["frames"], data["summary"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, layout="constrained")
    steps = [row["source_frame"] for row in records]
    for name in OBJECTS:
        color = np.asarray(COLORS[name]) / 255
        for axis, key in zip(axes, ("iou", "chamfer_mm")):
            values = [row["objects"][name][key] for row in records]
            axis.plot(steps, [np.nan if value is None else value for value in values],
                      label=name, color=color, linewidth=1.5)
            axis.grid(alpha=0.2)
    axes[0].set(ylabel="Mask IoU", ylim=(-0.03, 1.03))
    axes[1].set(ylabel="Visible-surface Chamfer (mm)", xlabel="Source episode row")
    axes[0].legend()
    chart = io.BytesIO()
    figure.savefig(chart, format="png", dpi=140)
    plt.close(figure)
    encoded_chart = base64.b64encode(chart.getvalue()).decode("ascii")

    rows = []
    for name, values in summary.items():
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            name, f"{values['visible_frames']} / {values['frames']}",
            _number(values["visible_iou"]["mean"], 4),
            _number(values["micro_iou"], 4),
            _number(values["micro_precision"], 4), _number(values["micro_recall"], 4),
            _number(values["centroid_error_mm"]["mean"]),
            _number(values["chamfer_mm"]["mean"]), _number(values["chamfer_mm"]["p95"]),
            values["chamfer_mm"]["count"], values["geometry_unavailable_visible_frames"],
            values["missed_visible_frames"], values["false_positive_invisible_frames"],
        )) + "</tr>")

    frame_rows = []
    for record in records:
        for name, values in record["objects"].items():
            columns = (record["source_frame"], record["frame_idx"], name,
                       values["gt_pixels"], values["predicted_pixels"],
                       _number(values["iou"], 4), _number(values["precision"], 4),
                       _number(values["recall"], 4), values["predicted_depth_pixels"],
                       values["gt_depth_pixels"], _number(values["predicted_valid_depth_fraction"], 4),
                       _number(values["median_depth_m"], 4), _number(values["centroid_error_mm"]),
                       _number(values["chamfer_mm"]), _number(values["predicted_to_gt_mm"]),
                       _number(values["gt_to_predicted_mm"]))
            frame_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in columns) + "</tr>")

    def table(headers, content):
        return ("<div class='scroll'><table><thead><tr>"
                + "".join(f"<th>{header}</th>" for header in headers)
                + "</tr></thead><tbody>" + "".join(content) + "</tbody></table></div>")

    totals = table(("Object", "GT visible / total", "Mean visible IoU", "Micro IoU",
                    "Precision", "Recall", "Mean centroid mm", "Mean Chamfer mm",
                    "p95 Chamfer mm", "3D frames", "Visible, no 3D", "Missed visible",
                    "False positive invisible"), rows)
    per_frame = table(("Source row", "Step", "Object", "GT px", "Pred px", "IoU",
                       "Precision", "Recall", "Pred depth px", "GT depth px",
                       "Pred valid depth fraction", "Median z m", "Centroid mm", "Chamfer mm",
                       "Pred→GT mm", "GT→pred mm"), frame_rows)
    # Escaping < also makes the embedded JSON safe when paths contain markup.
    raw_data = json.dumps(data, allow_nan=False, indent=2).replace("<", "\\u003c")
    details = html.escape(json.dumps({key: value for key, value in data.items()
                                    if key not in ("summary", "frames")}, indent=2))
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Visual RGB-D · GT 비교</title>
<style>
body{{font:15px/1.6 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1400px;color:#202a38;background:#f6f8fb}}
h1,h2{{line-height:1.25}} .scroll{{overflow:auto;background:white;border:1px solid #dde3ed;border-radius:8px}}
table{{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}}
td,th{{padding:9px 12px;text-align:right;border-bottom:1px solid #e5e9ef}}th{{background:#eaf0f8}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;background:white;border-radius:8px}}
code,pre{{background:#eaf0f8;padding:2px 5px;border-radius:4px}}pre{{overflow:auto;padding:16px}}details{{margin:20px 0}}
.note{{padding:14px 18px;background:#e8eff9;border-left:4px solid #466eb9}}
</style>
<h1>Visual RGB-D · GT 비교</h1>
<p>{len(records)} frames · {html.escape(data['camera'])} · run <code>{data['run_id']}</code></p>
<p><a href="depth.mp4">Depth 시각화 영상 (MP4)</a> · 원본 row {steps[0]}–{steps[-1]} · {data['video_fps']:.3f} fps</p>
<p class="note">예측 mask와 simulator GT segmentation을 비교합니다. 3D 오차는 각 mask를
<strong>같은 기록 depth</strong>로 역투영한 가시 표면 간 거리이며, depth 센서 정확도나
물체 pose 추정 오차를 의미하지 않습니다. 카메라 calibration과 TCP pose는 기록된 입력입니다.</p>
<h2>물체별 수치</h2>{totals}
<h2>프레임별 변화</h2><img alt="Mask IoU and surface distance by source frame" src="data:image/png;base64,{encoded_chart}">
<h2>지표 정의와 분모</h2>
<ul>
<li>GT visible: GT mask가 1 pixel 이상인 프레임. Mean visible IoU는 모든 visible 프레임의 평균이며 미검출은 0입니다.</li>
<li>Micro IoU / Precision / Recall: 전체 프레임의 TP·FP·FN을 합산해 계산합니다. GT가 없는 프레임의 오검출도 FP에 포함합니다.</li>
<li>GT와 예측이 모두 빈 mask이면 프레임 IoU는 n/a입니다. Precision·Recall도 해당 분모가 0이면 n/a입니다.</li>
<li>역투영: CV pixel centre (u+0.5, v+0.5), finite depth &gt; 0, metre 단위. 기존 Visual backproject / to_world 함수를 재사용합니다.</li>
<li>Centroid: 예측·GT 가시 표면 centroid 간 Euclidean 거리(mm). Actor 중심에 대한 pose 오차가 아닙니다.</li>
<li>Chamfer: 0.5 × (예측→GT 평균 최근접 거리 + GT→예측 평균 최근접 거리), mm. 제곱 거리나 두 방향의 합이 아닙니다.</li>
<li>3D frames: 양쪽 모두 유효 depth point가 있는 프레임 수. 없는 경우 거리 오차는 n/a이며 평균에 포함하지 않습니다. Visible, no 3D와 Missed visible을 함께 확인해야 합니다.</li>
<li>3D 거리는 모든 유효 관측점을 사용하며 sampling·erosion·필터링을 하지 않습니다. p95는 프레임별 거리의 95 percentile입니다.</li>
<li>영상의 depth 범위는 {data['depth_range_m'][0]:.2f}–{data['depth_range_m'][1]:.2f} m로 고정합니다. 범위 밖 색은 clip하지만 수치 평가는 모든 finite positive depth를 사용합니다.</li>
<li>이 실행은 저장된 관측·mask의 CPU 평가입니다. Simulator와 SAM2 추론, CUDA backend는 실행하지 않습니다.</li>
</ul>
<details><summary>프레임별 전체 수치</summary>{per_frame}</details>
<details><summary>입력 출처·재현 설정</summary><pre>{details}</pre></details>
<script type="application/json" id="evaluation-data">{raw_data}</script>
</html>"""
    path.write_text(document, encoding="utf-8")


def evaluate(episode_path: Path, masks_path: Path, *, camera="base_camera",
             output_root: Path, near=0.05, far=1.0, fps=None) -> Path:
    """Produce exactly two primary artifacts in a fresh experiment run directory."""
    episode_path, masks_path = Path(episode_path).resolve(), Path(masks_path).resolve()
    depth_colors(np.ones((1, 1), dtype=np.float32), near, far)
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and positive")
    with np.load(episode_path, allow_pickle=False) as archive:
        required = [f"{camera}/seg", "peg_id", "box_id"]
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"GT evaluation requires: {', '.join(missing)}")
        segmentation = archive[f"{camera}/seg"]
        gt_ids = {name: int(archive[f"{name}_id"].item()) for name in OBJECTS}
        if gt_ids["peg"] == gt_ids["box"]:
            raise ValueError("GT peg and box segmentation IDs must be distinct")
        seed = int(archive["seed"].item()) if "seed" in archive else None
        timestamps = archive["timestamp"] if "timestamp" in archive else None
        expected_shape = archive[f"{camera}/depth"].shape
    if segmentation.shape != expected_shape:
        raise ValueError("GT segmentation shape must match the episode depth array")
    with np.load(masks_path, allow_pickle=False) as archive:
        source = archive["source_frame"]
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer) or not len(source):
        raise ValueError("source_frame must be a nonempty one-dimensional integer array")
    if np.any(source < 0) or np.any(source >= len(segmentation)) or np.any(np.diff(source.astype(np.int64)) <= 0):
        raise ValueError("video source_frame must be in range and strictly increasing")
    if timestamps is not None:
        if timestamps.shape != (len(segmentation),):
            raise ValueError("timestamp shape must match episode frame count")
        tracked_times = timestamps[source]
        if not np.isfinite(tracked_times).all() or np.any(np.diff(tracked_times) <= 0):
            raise ValueError("tracked timestamps must be finite and strictly increasing")
    else:
        tracked_times = None
    if fps is None:
        if tracked_times is None or len(tracked_times) < 2:
            raise ValueError("pass --fps when fewer than two tracked timestamps are available")
        fps = float(1 / np.median(np.diff(tracked_times)))
        fps_source = "inverse median tracked timestamp gap"
    else:
        fps_source = "explicit override"

    metadata_path = masks_path.with_name("tracking.json")
    provenance = {}
    if metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if recorded.get("camera") not in (None, camera):
            raise ValueError("camera differs from the mask runner's tracking.json")
        recorded_input = recorded.get("input")
        if recorded_input and Path(recorded_input).is_file() and Path(recorded_input).resolve() != episode_path:
            if _sha256(Path(recorded_input)) != _sha256(episode_path):
                raise ValueError("episode differs from the mask runner's tracking.json")
        provenance = {key: recorded.get(key) for key in
                      ("input", "camera", "device", "config", "checkpoint", "stride")}
    iterator = iter_saved_rgbd(episode_path, masks_path, camera=camera)
    first = next(iterator)  # Validate the Visual observation boundary before creating outputs.
    config = dict(input=str(episode_path), masks=str(masks_path), camera=camera,
                  depth_range_m=[near, far], video_fps=fps, fps_source=fps_source,
                  input_sha256=_sha256(episode_path), masks_sha256=_sha256(masks_path),
                  output_root=str(Path(output_root).resolve()), gt_segmentation_ids=gt_ids,
                  output_object_ids=OBJECTS, tracker=provenance,
                  geometry="all finite-positive-depth pixels, no sampling or filters")
    with ExperimentLogger(team="visual", config=config, seed=seed,
                          model="saved_sam2_rgbd", job_type="evaluate") as experiment:
        output = Path(output_root).resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        video_path = output / "depth.mp4"
        records = []
        with imageio.get_writer(video_path, fps=fps, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "18"]) as writer:
            for frame in chain((first,), iterator):
                gt_masks = {name: segmentation[frame.source_frame] == gt_ids[name] for name in OBJECTS}
                metrics = {name: compare_object(frame, object_id, gt_masks[name])
                           for name, object_id in OBJECTS.items()}
                record = dict(source_frame=frame.source_frame, frame_idx=frame.frame_idx,
                              timestamp=frame.timestamp, objects=metrics)
                records.append(record)
                writer.append_data(render_depth_frame(frame, gt_masks, metrics, near, far))
                if len(records) == 1 or len(records) % 20 == 0:
                    print(f"evaluated {len(records)}/{len(source)} frames", flush=True)
        experiment.link_output(video_path, kind="rollouts")
        summary = summarize(records)
        report_path = output / "gt_report.html"
        data = dict(run_id=experiment.run_id, seed=seed, evaluation_device="cpu",
                    git_commit=experiment.metadata.get("git_commit"),
                    git_dirty=experiment.metadata.get("git_dirty"),
                    **config, summary=summary, frames=records)
        write_report(report_path, data)
        experiment.link_output(report_path, kind="reports")
        experiment.log(summary, prefix="eval/rgbd")
        experiment.record(dataset_sha256=config["input_sha256"],
                          masks_sha256=config["masks_sha256"], evaluated_frames=len(records),
                          evaluation_device="cpu")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original recorded episode NPZ")
    parser.add_argument("--masks", type=Path, required=True, help="Runtime masks.npz")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--out", type=Path, default=REPO / "visual/results/runtime_rgbd",
                        help="Output root; creates a fresh <run_id> subdirectory")
    parser.add_argument("--depth-min", type=float, default=0.05, help="Fixed video depth scale, metres")
    parser.add_argument("--depth-max", type=float, default=1.0, help="Fixed video depth scale, metres")
    parser.add_argument("--fps", type=float, default=None, help="Override median timestamp-gap playback fps")
    args = parser.parse_args()
    try:
        output = evaluate(args.input, args.masks, camera=args.camera, output_root=args.out,
                          near=args.depth_min, far=args.depth_max, fps=args.fps)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"saved GT report: {output / 'gt_report.html'}")
    print(f"saved depth video: {output / 'depth.mp4'}")


if __name__ == "__main__":
    main()

"""Grounding DINO initialization rules from the Visual v1 prototype.

Only RGB detections are used. A missing object stays missing until two scans
agree; no simulator segmentation or pose is consulted.
"""

from __future__ import annotations

from itertools import product

PROMPTS = {"peg": "red and white block", "box": "cardboard box"}
MAX_BOX_FRAC = 0.25
MISSING_SCORE = 0.3
EXCLUSION_IOU = 0.3
SCAN_EVERY = 5


def box_iou(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def candidates(detector, rgb, prompt, limit=5):
    height, width = rgb.shape[:2]
    boxes, scores = detector(rgb, prompt)
    return [(box, float(score)) for box, score in zip(boxes, scores)
            if (box[2] - box[0]) * (box[3] - box[1]) < MAX_BOX_FRAC * height * width][:limit]


def assign(peg_candidates, box_candidates):
    """Pick distinct objects jointly, with an explicit missing choice."""
    best = (None, None)
    best_score = 2 * MISSING_SCORE
    for peg, box in product([*peg_candidates, None], [*box_candidates, None]):
        if peg is not None and box is not None and box_iou(peg[0], box[0]) >= EXCLUSION_IOU:
            continue
        score = (peg[1] if peg else MISSING_SCORE) + (box[1] if box else MISSING_SCORE)
        if score > best_score:
            best, best_score = (peg, box), score
    return best


def detect_initial(frames, detector, prompts=None):
    """Return each object's selected box, score and source-frame index."""
    prompts = PROMPTS if prompts is None else prompts
    first = {name: candidates(detector, frames[0], prompts[name]) for name in PROMPTS}
    peg, box = assign(first["peg"], first["box"])
    result = {
        "peg": {"box": peg[0] if peg else None, "score": peg[1] if peg else None,
                "frame": 0 if peg else None, "prompt": prompts["peg"], "scans": 1},
        "box": {"box": box[0] if box else None, "score": box[1] if box else None,
                "frame": 0 if box else None, "prompt": prompts["box"], "scans": 1},
    }
    for name in PROMPTS:
        if result[name]["box"] is not None:
            continue
        other = result["box" if name == "peg" else "peg"]["box"]
        pending = None
        for frame in range(SCAN_EVERY, len(frames), SCAN_EVERY):
            result[name]["scans"] += 1
            choices = [candidate for candidate in candidates(detector, frames[frame], prompts[name])
                       if candidate[1] >= MISSING_SCORE
                       and (other is None or box_iou(candidate[0], other) < EXCLUSION_IOU)]
            if not choices:
                pending = None
                continue
            choice = choices[0]
            if pending is not None and box_iou(choice[0], pending[0]) >= 0.5:
                result[name].update(box=choice[0], score=choice[1], frame=frame)
                break
            pending = choice
    return result

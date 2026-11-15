"""Common output helpers for the RoArm diagnostic scripts."""

import json


def write_report(output, report, *, display=False):
    """Save the diagnostic report and optionally print the same JSON."""
    contents = json.dumps(report, indent=2)
    (output / "report.json").write_text(contents + "\n")
    if display:
        print(contents)


def save_frames(output, images, *, duration):
    """Save an animation and a representative frame when frames were captured."""
    if not images:
        return
    images[0].save(
        output / "motion.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    images[len(images) // 2].save(output / "robot.png")

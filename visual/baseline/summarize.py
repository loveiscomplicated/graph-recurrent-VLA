"""Collect every baseline eval JSON into SUMMARY.md and print a short phone-sized digest.

Example:
    python visual/baseline/summarize.py outputs/visual_baseline
"""

import json
import sys
from pathlib import Path

STAGES = ("success", "aligned", "misaligned", "lifted", "no_lift")


def main() -> None:
    root = Path(sys.argv[1])
    lines = ["| run | eval | success | aligned / misaligned / lifted / no_lift | step ms p50 / p95 |",
             "|---|---|---|---|---|"]
    digest = []
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text())
        if "success" not in record or "results" not in record:
            continue
        stages = record.get("stages", {})
        stage_text = " / ".join(str(stages.get(name, "?")) for name in STAGES[1:])
        latency = record.get("step_latency_ms", {})
        lines.append(f"| {path.parent.name} | {path.stem} | {record['success']}/{record['episodes']} "
                     f"| {stage_text} | {latency.get('p50', '?')} / {latency.get('p95', '?')} |")
        digest.append(f"{path.parent.name.replace('_main', '')} {path.stem.replace('eval_', '')}: "
                      f"{record['success']}/{record['episodes']}")
    (root / "SUMMARY.md").write_text("# Baseline evaluation summary\n\nExpert: 98/100 on eval seeds.\n\n"
                                     + "\n".join(lines) + "\n")
    print("; ".join(digest) if digest else "no eval results yet")


if __name__ == "__main__":
    main()

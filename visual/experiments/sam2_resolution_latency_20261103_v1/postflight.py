"""Read back artifacts after all timed GPU work has ended."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


protocol = read(root / "run/protocol.json")
manifest = read(root / "run/input_manifest.json")
capture = read(root / "run/capture_verification.json")
for row in manifest:
    assert sha(row["png"]) == row["png_sha256"]
    if row["jpeg"]:
        assert sha(row["jpeg"]) == row["jpeg_sha256"]
for parent in capture["source_parents"]:
    assert sha(parent["raw_path"]) == parent["raw_sha256"]
    assert sha(parent["gt_path"]) == parent["gt_sha256"]
checkpoint = Path("/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt")
assert sha(checkpoint) == protocol["checkpoint_sha256"]
assert sha(root / "benchmark.py") == protocol["benchmark_source_sha256"]
assert sha(root / "benchmark_capture_revision.py") == read(root / "run/protocol_capture_revision.json")["benchmark_source_sha256"]
assert len(read(root / "run/static_raw.json")) == 288
assert len(read(root / "run/tracking_frames.json")) == 1512
assert len(read(root / "run/tracking_sequences.json")) == 72
statuses = {p.parent.name: read(p)["status"] for p in (root / "run/experiments").glob("*/metadata.json")}
assert list(statuses.values()).count("finished") == 3
assert list(statuses.values()).count("failed") == 1
active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
assert not active, "a CUDA process remains; inspect before completing"
import sam2
source = Path(sam2.__file__).parent
sam_sources = {str(p): sha(p) for p in source.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml")}
completion = dict(status="COMPLETE", input_file_hashes_preserved=True, original_raw_and_GT_hashes_preserved=True,
    checkpoint_preserved=True, benchmark_source_verified=True, experiments=statuses,
    retained_setup_failure="PackageNotFoundError before model load; SAM-2 distribution name fixed; log and original source/protocol preserved",
    CUDA_compute_processes_after=[], packages={p: importlib.metadata.version(p) for p in ("mani_skill", "sapien", "torch", "torchvision", "SAM-2", "numpy", "Pillow")},
    GPU_after=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], text=True).strip(),
    sam2_source_directory=str(source), sam2_source_hashes=sam_sources,
    input_artifact_files=sum(bool(r["jpeg"]) for r in manifest)+len(manifest), publishing=False)
(root / "completion.json").write_text(json.dumps(completion, indent=2)+"\n")
index = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"}
(root / "artifact_sha256.json").write_text(json.dumps(index, indent=2)+"\n")
print(json.dumps({k: v for k, v in completion.items() if k != "sam2_source_hashes"}, indent=2))

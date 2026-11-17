"""Download the revision-pinned handoff using the current user's HF credentials."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(Path(__file__).with_name("published_artifacts.json").read_text())
    for kind, entry in inventory.items():
        folder = args.output / ("models" if kind == "checkpoint" else "data")
        folder.mkdir(parents=True, exist_ok=True)
        for item in entry["files"]:
            remote = item["path"]
            cached = Path(hf_hub_download(entry["repo_id"], remote,
                          repo_type=entry["repo_type"], revision=entry["revision"]))
            destination = folder / Path(remote).name
            assert cached.stat().st_size == item["size"]
            if "/artifacts/" in remote:
                with cached.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                assert digest == remote.split("/")[-2]
            shutil.copy2(cached, destination)
        print(kind, "downloaded and verified", folder)

if __name__ == "__main__":
    main()

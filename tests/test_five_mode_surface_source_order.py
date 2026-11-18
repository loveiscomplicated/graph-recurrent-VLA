from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from data_collection import five_mode_surface_source as source
from tests.test_five_mode_box_surface import make_fixture


@pytest.mark.parametrize("index_last", [False, True])
@pytest.mark.parametrize("receipt_last", [False, True])
def test_index_and_episode_receipt_can_follow_the_frames(tmp_path, index_last, receipt_last):
    config, records = make_fixture(tmp_path)
    record = records[0]
    original = next((Path(config["raw_root"]) / record["batch"]).glob("*.tar.zst"))
    with source.open_raw_stream(original) as archive:
        members = [(member.name, archive.extractfile(member).read()) for member in archive if member.isfile()]
    index = [row for row in members if row[0].endswith("episode_index.json")]
    receipt = [row for row in members if row[0].endswith("episode.json")]
    frames = [row for row in members if row[0].endswith(".npz")]
    ordered = ([] if index_last else index) + ([] if receipt_last else receipt) + frames
    ordered += receipt if receipt_last else []
    ordered += index if index_last else []
    tar = tmp_path / "reordered.tar"
    with tarfile.open(tar, "w") as archive:
        for name, contents in ordered:
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, BytesIO(contents))
    compressed = tmp_path / "reordered.tar.zst"
    subprocess.run(["zstd", "-q", str(tar), "-o", str(compressed)], check=True)
    scratch = tmp_path / "staging"
    scratch.mkdir()
    seen = []
    with closing(source.raw_episodes(compressed, [record], scratch)) as episodes:
        for actual, directory in episodes:
            assert actual == record
            assert sorted(path.name for path in directory.glob("*.npz")) == ["frame_000000.npz", "frame_000001.npz"]
            seen.append(actual)
    assert seen == [record]

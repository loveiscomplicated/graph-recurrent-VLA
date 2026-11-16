"""Check collection overlap and compatibility with the raw episode readers."""

import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.episode_staging import persist_episodes, stage_batches
from data_collection.storage import save_frame


class EpisodeStagingTests(unittest.TestCase):
    def test_next_batch_collects_before_previous_batch_is_saved(self):
        writer_started = Event()
        second_collected = Event()
        with TemporaryDirectory() as temporary:
            def collect_batch(index, staging):
                if index == 1:
                    self.assertTrue(writer_started.wait(5))
                    second_collected.set()
                return index

            def persist_batch(staging, result):
                if result == 0:
                    writer_started.set()
                    self.assertTrue(second_collected.wait(5))
                return result

            self.assertEqual(list(stage_batches([0, 1], collect_batch, persist_batch,
                                                {'ram_root': temporary})), [0, 1])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_recovery_prefix_keeps_its_hash_and_new_frames_get_final_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            data = np.array([[[[-32768], [0], [1]]]], dtype=np.int16)

            def collect_batch(index, staging):
                directory = staging / 'episode'
                directory.mkdir()
                frame_hashes = []
                for index in range(2):
                    frame = directory / f'frame_{index:06d}.npz'
                    save_frame(frame, {'depth': data}, compress=(index == 0))
                    frame_hashes.append(file_sha256(frame))
                initial_hashes.extend(frame_hashes)
                return [{'path': directory.name, 'frame_sha256': frame_hashes}]

            initial_hashes = []
            batches = list(stage_batches([0], collect_batch, partial(persist_episodes, output=output),
                                         {'ram_root': root}))
            receipt = batches[0][0]
            self.assertEqual(receipt['frame_sha256'][0], initial_hashes[0])
            self.assertNotEqual(receipt['frame_sha256'][1], initial_hashes[1])
            self.assertEqual(receipt, json.loads((output / 'episode/episode.json').read_text()))
            for index, digest in enumerate(receipt['frame_sha256']):
                path = output / 'episode' / f'frame_{index:06d}.npz'
                self.assertEqual(file_sha256(path), digest)
                with ZipFile(path) as frame:
                    self.assertEqual(frame.infolist()[0].compress_type, ZIP_DEFLATED)
                with np.load(path, allow_pickle=False) as frame:
                    np.testing.assert_array_equal(frame['observation/depth'], data)

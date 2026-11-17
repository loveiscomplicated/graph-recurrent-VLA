"""Checks for the selected-frame Hugging Face archive restoration."""

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from data_collection.temporal import restore_hf_temporal


class TemporalRestoreTests(unittest.TestCase):
    def make_archive(self, root, entries):
        path = root / 'shard.tar'
        with tarfile.open(path, 'w') as archive:
            for name, content in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return path

    def test_selected_frames_are_hash_checked_and_other_frames_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = 'shard_000000/episode_000001/frame_000000.npz'
            ignored = 'shard_000000/episode_000002/frame_000000.npz'
            archive = self.make_archive(root, {selected: b'selected', ignored: b'not selected'})
            expected = {selected: hashlib.sha256(b'selected').hexdigest()}
            with patch.object(restore_hf_temporal, 'ROOT', root), \
                 patch.object(restore_hf_temporal, 'download', return_value=archive):
                self.assertEqual(restore_hf_temporal.restore_shard(('remote.tar', expected)), 1)
            self.assertEqual((root / 'data/episodes' / selected).read_bytes(), b'selected')
            self.assertFalse((root / 'data/episodes' / ignored).exists())
            self.assertFalse(archive.exists())

    def test_checksum_mismatch_never_commits_a_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = 'shard_000000/episode_000001/frame_000000.npz'
            archive = self.make_archive(root, {sample: b'incorrect'})
            with patch.object(restore_hf_temporal, 'ROOT', root), \
                 patch.object(restore_hf_temporal, 'download', return_value=archive):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    restore_hf_temporal.restore_shard(('remote.tar', {sample: '0' * 64}))
            self.assertFalse((root / 'data/episodes' / sample).exists())


if __name__ == '__main__':
    unittest.main()

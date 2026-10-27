"""Capture identity, signed depth and HTTP isolation regression checks."""

import base64
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from visualization.captures import capture_frame, list_captures
from visualization.server import make_handler


class CaptureViewerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'run' / 'collection' / 'seed12'
        self.folder.mkdir(parents=True)
        self.episode_id = 'run/collection/seed12'
        roles = {'main': 'base_camera', 'wrist': 'hand_camera'}
        (self.folder / 'sensor_contract.json').write_text(json.dumps({'roles': roles}))
        sensor = dict(timestep=np.int64(3), timestamp=np.float64(.15), episode=np.array('run:seed12'))
        masks = {}
        gt = dict(actor_ids=np.array([8, 9]))
        for name in roles.values():
            sensor[name + '/rgb'] = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            sensor[name + '/depth_raw_mm'] = np.array([[-32768, 0], [500, 1200]], np.int16)
            masks[name] = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 0]]], bool)
            gt[name + '/segmentation'] = np.array([[0, 0], [8, 0]], np.int16)
        self.record = dict(timestep=3, stem='f0003', phase='reach')
        for kind, data, key in [('input', sensor, 'input_sha256'),
                                ('masks', masks, 'mask_sha256'), ('gt', gt, 'gt_sha256')]:
            path = self.folder / f'f0003_{kind}.npz'
            np.savez_compressed(path, **data)
            self.record[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = self.folder.parent.parent / 'capture_manifest.json'
        self.value = dict(episodes=[dict(seed=12, split='train', episode='run:seed12',
                                         complete=True, frames=[self.record])])
        self.manifest.write_text(json.dumps(self.value))

    def test_original_pixels_signed_depth_gt_and_empty_iou(self):
        self.assertEqual(list_captures(self.root)[0]['frames'], 1)
        frame = capture_frame(self.root, self.episode_id, 3)
        self.assertTrue(frame['hashes_verified'])
        self.assertEqual(frame['timestamp'], .15)
        self.assertEqual(len(frame['cameras']), 2)
        for camera in frame['cameras']:
            np.testing.assert_array_equal(np.frombuffer(base64.b64decode(camera['rgb']), np.uint8),
                                          np.arange(12, dtype=np.uint8))
            np.testing.assert_array_equal(np.frombuffer(base64.b64decode(camera['depth_mm']), '<i2'),
                                          [-32768, 0, 500, 1200])
            self.assertEqual(camera['valid_depth_fraction'], .5)
            self.assertEqual(camera['objects'][0]['iou'], 1.)
            self.assertIsNone(camera['objects'][1]['iou'])
            self.assertEqual(camera['objects'][0]['gt_pixels'], 1)

    def test_tampering_identity_and_timeline_rejected(self):
        path = self.folder / 'f0003_input.npz'
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            capture_frame(self.root, self.episode_id, 3)
        path.write_bytes(original)
        self.value['episodes'][0]['episode'] = 'different-episode'
        self.manifest.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            capture_frame(self.root, self.episode_id, 3)
        self.record['stem'] = '../outside'
        self.manifest.write_text(json.dumps(self.value))
        self.assertEqual(list_captures(self.root), [])

    def test_http_frames_and_root_boundary(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/captures') as response:
                self.assertIn(b'id="camera-views"', response.read())
            with urlopen(base + '/api/captures') as response:
                self.assertEqual(len(json.load(response)['episodes']), 1)
            with urlopen(base + f'/api/capture-frame?id={self.episode_id}&t=3') as response:
                self.assertEqual(json.load(response)['t'], 3)
            for query in ('id=..%2F..&t=3', f'id={self.episode_id}&t=-1'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/api/capture-frame?' + query)
                self.assertEqual(error.exception.code, 400)
            external = self.root.parent / 'not-a-capture.npz'
            path = self.folder / 'f0003_gt.npz'
            path.unlink()
            path.symlink_to(external)
            with self.assertRaises(HTTPError) as error:
                urlopen(base + f'/api/capture-frame?id={self.episode_id}&t=3')
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_static_export_keeps_verified_pixels_and_relative_paths(self):
        from visualization.export_captures import export_captures
        output = self.root / 'static-site'
        result = export_captures(self.root, 'run', output)
        self.assertEqual((result['episodes'], result['frames'], result['videos']), (1, 1, 0))
        html = (output / 'index.html').read_text()
        self.assertIn('data-capture-data="./data"', html)
        self.assertNotIn('id="interactive-graph"', html)
        self.assertNotIn('href="/', html)
        exported = json.loads((output / 'data/run/collection/seed12/frames/3.json').read_text())
        self.assertEqual(exported, capture_frame(self.root, self.episode_id, 3))
        with self.assertRaises(ValueError):
            export_captures(self.root, 'run', output)


if __name__ == '__main__':
    unittest.main()

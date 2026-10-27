"""Saved graph/capture joins, world coordinates and video seeking contracts."""

import base64
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from visualization.data import list_episodes
from visualization.persistent import persistent_arrays, persistent_frame
from visualization.server import make_handler


class PersistentViewerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        capture = self.root / 'capture' / 'collection' / 'seed7'
        capture.mkdir(parents=True)
        self.capture = capture
        self.id = 'geometry_run/geometry/seed7'
        self.graph = self.root / self.id / 'f0003'
        self.graph.mkdir(parents=True)
        pose = np.array([1, 2, 3, 1, 0, 0, 0], np.float32)
        self.nodes = np.zeros((5, 21), np.float32)
        self.nodes[:, :3] = np.arange(15).reshape(5, 3)/100
        self.nodes[3:, 14] = 1
        world = self.nodes[:3, :3] + pose[:3]
        sensor = capture / 'f0003_input.npz'
        np.savez(sensor, timestep=np.int64(3), episode=np.array('capture:seed7'),
                 timestamp=np.float64(.15), tcp_pose=pose, robot_world_geometry=world)
        digest = hashlib.sha256(sensor.read_bytes()).hexdigest()
        (capture/'sensor_contract.json').write_text(json.dumps(dict(roles=dict(main='base_camera', wrist='hand_camera'))))
        (capture.parent.parent/'capture_manifest.json').write_text(json.dumps(dict(episodes=[dict(
            seed=7, split='validation', episode='capture:seed7', complete=True,
            frames=[dict(timestep=3, stem='f0003', phase='reach', input_sha256=digest)])])))
        self.manifest = self.graph.parents[2] / 'geometry_manifest.json'
        self.value = dict(config=dict(capture_run_id='capture', variants=['G0', 'G4'], geometry_revision='fixture'),
                          frames=[dict(seed=7, timestep=3, folder='geometry/seed7/f0003',
                                       phase='reach', capture_input_sha256=digest)])
        self.manifest.write_text(json.dumps(self.value))
        self.arrays = dict(node_input=self.nodes, node_type=np.array([0, 1, 2, 3, 4], np.int64),
                           edge_index=np.array([[0, 3], [3, 0]], np.int64),
                           edge_attr=np.zeros((2, 4), np.float32), edge_type=np.array([1, 1], np.int64))
        for variant in ('G0', 'G4'):
            np.savez(self.graph/f'{variant}_graph.npz', **self.arrays)
            metadata = dict(schema_id='task09c_persistent_surface_v1')
            if variant == 'G4':
                metadata.update(node_ids=['persistent-a', 'persistent-b'], timestep=3)
            else:
                metadata['selected_original_rows'] = [0, 1, 2, 7, 9]
            (self.graph/f'{variant}_graph.json').write_text(json.dumps(metadata))

    def test_catalog_preserves_unknown_outcome_and_exact_saved_features(self):
        episodes = list_episodes(self.root)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]['result'], 'unknown')
        self.assertIsNone(episodes[0]['success'])
        frame = persistent_frame(self.root, self.id, 3, 'G4')
        nodes = np.frombuffer(base64.b64decode(frame['node_input']), '<f4').reshape(5, 21)
        world = np.frombuffer(base64.b64decode(frame['world_position']), '<f4').reshape(5, 3)
        np.testing.assert_array_equal(nodes, self.nodes)
        np.testing.assert_allclose(world, self.nodes[:, :3] + [1, 2, 3], atol=1e-6)
        self.assertEqual(frame['node_ids'], ['persistent-a', 'persistent-b'])
        self.assertEqual(frame['feature_dimension'], 21)
        baseline = persistent_frame(self.root, self.id, 3, 'G0')
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(baseline['source_index']), '<i4'), [4, 6])

    def test_wrong_capture_bad_edges_and_wrong_robot_are_rejected(self):
        self.value['frames'][0]['capture_input_sha256'] = 'wrong'
        self.manifest.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError, 'different capture'):
            persistent_frame(self.root, self.id, 3, 'G4')
        self.value['frames'][0]['capture_input_sha256'] = hashlib.sha256((self.capture/'f0003_input.npz').read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps(self.value))
        self.arrays['edge_index'][0, 0] = 5
        np.savez(self.graph/'G4_graph.npz', **self.arrays)
        with self.assertRaisesRegex(ValueError, 'graph arrays'):
            persistent_arrays(self.root, self.id, 3, 'G4')
        self.arrays['edge_index'][0, 0] = 0
        self.nodes[0, 0] += .1
        np.savez(self.graph/'G4_graph.npz', **self.arrays)
        with self.assertRaisesRegex(ValueError, 'robot geometry disagree'):
            persistent_arrays(self.root, self.id, 3, 'G4')

    def test_http_variant_selection_and_video_ranges(self):
        media = self.manifest.parent / 'viewer_media'
        media.mkdir()
        path = media/'clip.mp4'
        path.write_bytes(b'0123456789')
        record = dict(path='viewer_media/clip.mp4', frames=1, fps=20)
        (media/'index.json').write_text(json.dumps(dict(geometry_run_id='geometry_run', episodes=dict(seed7=dict(
            geometry_id=self.id, capture_id='capture/collection/seed7', variants=dict(
                G4=dict(graph_rgb=record, graph_only=record), FPS=dict(graph_rgb=record, graph_only=record)))))))
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+f'/api/frame?id={self.id}&t=3&variant=G0') as response:
                self.assertEqual(json.load(response)['variant'], 'G0')
            with urlopen(base+'/api/capture-media?id=capture/collection/seed7') as response:
                variants = json.load(response)['runs'][0]['variants']
                video = variants[0]['videos']['graph_only']['url']
                self.assertEqual(variants[1]['label'], 'point cloud prototype(FPS)')
                self.assertEqual(variants[1]['kind'], 'pointcloud')
                self.assertIn('1,024', variants[1]['description'])
            with urlopen(Request(base+variants[1]['videos']['graph_only']['url'], headers={'Range': 'bytes=2-5'})) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b'2345')
            for requested, body in [('bytes=2-5', b'2345'), ('bytes=-3', b'789'), ('bytes=8-', b'89')]:
                with urlopen(Request(base+video, headers={'Range': requested})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), body)
                    self.assertEqual(response.headers['Accept-Ranges'], 'bytes')
            for url, headers, code in [(video, {'Range': 'bytes=99-'}, 416),
                                       (video, {'Range': 'bytes=0-1,4-5'}, 416),
                                       (f'/api/frame?id={self.id}&t=3&variant=../G4', {}, 400),
                                       (f'/api/frame?id={self.id}&t=3&variant=FPS', {}, 400),
                                       ('/media?run=..&seed=seed7&variant=G4&kind=graph_only', {}, 400)]:
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base+url, headers=headers))
                self.assertEqual(error.exception.code, code)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()

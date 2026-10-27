import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import numpy as np

from visualization.data import episode_info, frame_for_step, list_episodes
from visualization.server import make_handler
from visual.runtime.vision_packet import pose_matrix


class GraphRolloutViewerTest(unittest.TestCase):
    @staticmethod
    def observations(episode, timestep, nodes):
        pose = np.array([1, 2, 3, np.sqrt(.5), 0, 0, np.sqrt(.5)], np.float32)
        transform = pose_matrix(pose)
        world = nodes[:3, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
        np.savez(episode/'observations.npz', timestep=np.array([timestep], np.int64),
                 tcp_pose=pose[None], robot_ee_geometry=nodes[None, :3, :3],
                 robot_world_geometry=world.astype(np.float32)[None])
        return world

    def test_saved_graph_round_trips_through_frame_api(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root)/'episodes'/'sample'
            episode.mkdir(parents=True)
            nodes = np.arange(56, dtype=np.float32).reshape(4, 14) / 100
            world = self.observations(episode, 7, nodes)
            edges = np.array([[0, 3], [3, 0]], np.int64)
            attrs = np.array([[.1, 1, 0, 0], [.1, -1, 0, 0]], np.float32)
            np.savez(episode/'graph_0007.npz', node_input=nodes,
                     node_type=np.array([0, 1, 2, 3], np.int64),
                     edge_index=edges, edge_attr=attrs,
                     edge_type=np.array([1, 1], np.int64),
                     source_index=np.array([42], np.int64))
            (episode/'episode.json').write_text(json.dumps({'status':'COMPLETE','success':True,
                                                               'first_success_step':8}))
            frame = frame_for_step(episode, 7)
            info = episode_info(episode, root)
            self.assertEqual((info['frames'], info['first_timestep'], info['result']),
                             (1, 7, 'success'))
            def decoded(name, dtype):
                return np.frombuffer(base64.b64decode(frame[name]), dtype=dtype)
            np.testing.assert_array_equal(decoded('node_input', '<f4').reshape(4, 14), nodes)
            np.testing.assert_array_equal(decoded('edge_index', '<i4').reshape(2, 2), edges)
            np.testing.assert_array_equal(decoded('edge_attr', '<f4').reshape(2, 4), attrs)
            np.testing.assert_allclose(decoded('world_position', '<f4').reshape(4, 3)[:3],
                                       world, atol=1e-6)
            self.assertEqual(decoded('source_index', '<i4').tolist(), [42])
            self.assertEqual(list_episodes(root)[0]['id'], 'episodes/sample')

    def test_bad_edge_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root)
            self.observations(episode, 0, np.zeros((3, 14), np.float32))
            np.savez(episode/'graph_0000.npz', node_input=np.zeros((3, 14), np.float32),
                     node_type=np.arange(3, dtype=np.int64),
                     edge_index=np.array([[0], [3]], np.int64),
                     edge_attr=np.zeros((1, 4), np.float32),
                     edge_type=np.zeros(1, np.int64), source_index=np.empty(0, np.int64))
            with self.assertRaisesRegex(ValueError, 'invalid graph values'):
                frame_for_step(episode, 0)

    def test_mismatched_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root)
            nodes = np.zeros((3, 14), np.float32)
            self.observations(episode, 1, nodes)
            np.savez(episode/'graph_0000.npz', node_input=nodes,
                     node_type=np.arange(3, dtype=np.int64),
                     edge_index=np.empty((2, 0), np.int64),
                     edge_attr=np.empty((0, 4), np.float32),
                     edge_type=np.empty(0, np.int64), source_index=np.empty(0, np.int64))
            with self.assertRaisesRegex(ValueError, 'matching observation timestep is missing'):
                frame_for_step(episode, 0)

    def test_list_excludes_incompatible_legacy_observations(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root)/'legacy'
            episode.mkdir()
            np.savez(episode/'observations.npz', timestep=np.array([0]))
            np.savez(episode/'graph_0000.npz', node_input=np.zeros((3, 14), np.float32))
            self.assertEqual(list_episodes(root), [])

    def test_result_categories_and_http_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name, status, success in [('a','COMPLETE',True), ('b','COMPLETE',False),
                                          ('c','CONTRACT_SMOKE',False)]:
                episode = root/name
                episode.mkdir()
                nodes = np.zeros((3, 14), np.float32)
                self.observations(episode, 0, nodes)
                np.savez(episode/'graph_0000.npz', node_input=nodes,
                         node_type=np.arange(3, dtype=np.int64),
                         edge_index=np.empty((2, 0), np.int64),
                         edge_attr=np.empty((0, 4), np.float32),
                         edge_type=np.empty(0, np.int64), source_index=np.empty(0, np.int64))
                (episode/'episode.json').write_text(json.dumps({'status':status,'success':success}))
            self.assertEqual([x['result'] for x in list_episodes(root)],
                             ['success', 'failure', 'unknown'])
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/') as response:
                    self.assertIn(b'id="episode-cards"', response.read())
                with urlopen(base+'/viewer') as response:
                    self.assertIn(b'id="scene"', response.read())
                with urlopen(base+'/api/episodes') as response:
                    self.assertEqual(len(json.load(response)['episodes']), 3)
                with urlopen(base+'/api/frame?id=a&t=0') as response:
                    self.assertEqual(json.load(response)['n'], 3)
                with self.assertRaises(HTTPError) as error:
                    urlopen(base+'/api/frame?id=..%2F..&t=0')
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()

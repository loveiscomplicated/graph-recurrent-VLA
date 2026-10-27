"""CPU contract fixtures for the reused streaming implementation, without weights."""
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from visual.scripts.stream_v1 import CausalFrames, FutureFrameAccess, StreamTracker
from graph.integration_baseline.streaming_vision_adapter import (
    PrefixInitialization, StreamVisionAdapter, make_online_vision,
)


def load_video_frames(**kwargs):
    raise AssertionError('the directory loader must not be called')


class Predictor:
    device = torch.device('cpu')
    image_size = 4

    def __init__(self):
        self.encodes, self.propagations, self.prompts = [], [], []
        self.fail_init = False

    def feature(self, state, t):
        if t not in state['cached_features']:
            self.encodes.append(t)
            state['cached_features'] = {t: state['images'][t]}

    def init_state(self, video_path):
        images, height, width = load_video_frames(video_path=video_path)
        if self.fail_init:
            raise RuntimeError('init failure fixture')
        state = dict(images=images, num_frames=len(images), obj_ids=[], cached_features={},
                     output_dict_per_obj={}, frames_tracked_per_obj={}, height=height, width=width)
        self.feature(state, 0)
        return state

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self.feature(state, frame_idx)
        if obj_id not in state['obj_ids']:
            state['obj_ids'].append(obj_id)
            idx = state['obj_ids'].index(obj_id)
            state['output_dict_per_obj'][idx] = dict(cond_frame_outputs={}, non_cond_frame_outputs={})
            state['frames_tracked_per_obj'][idx] = {}
        idx = state['obj_ids'].index(obj_id)
        state['output_dict_per_obj'][idx]['cond_frame_outputs'][frame_idx] = object()
        self.prompts.append((frame_idx, obj_id))
        return frame_idx, list(state['obj_ids']), torch.ones(len(state['obj_ids']), 1, 4, 4)

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
        t = start_frame_idx
        if not state['obj_ids']:
            raise RuntimeError('no objects')
        if state['images'].observed_horizon:
            assert state['num_frames'] == len(state['images']) == t + 1
        assert max_frame_num_to_track == 0
        with self_assert_raises(FutureFrameAccess):
            state['images'][t + 1]
        self.feature(state, t)
        self.propagations.append(t)
        logits = torch.full((len(state['obj_ids']), 1, 4, 4), -1.)
        for idx, oid in enumerate(state['obj_ids']):
            logits[idx, 0, (t + oid) % 4, oid % 4] = 1
            state['output_dict_per_obj'][idx]['non_cond_frame_outputs'][t] = object()
            state['frames_tracked_per_obj'][idx][t] = {'reverse': False}
        yield t, list(state['obj_ids']), logits

    def reset_state(self, state):
        state['output_dict_per_obj'].clear()


def self_assert_raises(error):
    return unittest.TestCase().assertRaises(error)


def image(t):
    return np.full((4, 4, 3), t, dtype=np.uint8)


def detector(rgb, prompt):
    if 'red' in prompt:
        return [[0, 0, 1, 1]], [.9]
    return ([[2, 2, 3, 3]], [.9]) if rgb[0, 0, 0] >= 5 else ([], [])


class StreamTrackerTests(unittest.TestCase):
    def tracker(self, **kwargs):
        return StreamTracker(Predictor(), observed_horizon=True, **kwargs)

    def test_prefix_bounds_legacy_default_and_buffer_release(self):
        legacy = CausalFrames(4, device='cpu')
        self.assertEqual(len(legacy), 100000)
        stream = self.tracker()
        self.assertEqual(stream.start(image(0), {}), {})
        frames = stream.frames
        for t in range(1, 7):
            self.assertEqual(stream.step(image(t)), (t, {}))
        self.assertEqual(len(frames), 7)
        self.assertEqual(set(frames.buf), {5, 6})
        for t in (-1, 0, 7):
            with self.assertRaises(FutureFrameAccess):
                frames[t]
        stream.reset()
        self.assertFalse(frames.buf)
        self.assertIsNone(stream.state)
        with self.assertRaises(RuntimeError):
            stream.step(image(0))

    def test_late_object_one_state_one_feature_and_pruned_memory(self):
        stream = self.tracker()
        stream.start(image(0), {1: [0, 0, 1, 1]})
        state = stream.state
        memory = state['output_dict_per_obj'][0]
        cond = memory['cond_frame_outputs'][0]
        for t in range(1, 40):
            boxes = {2: [2, 2, 3, 3]} if t == 10 else None
            _, masks = stream.step(image(t), boxes=boxes)
            self.assertIs(stream.state, state)
            self.assertIs(state['output_dict_per_obj'][0], memory)
            self.assertIs(memory['cond_frame_outputs'][0], cond)
            self.assertEqual(set(masks), {1, 2} if t >= 10 else {1})
        self.assertEqual(stream.p.encodes, list(range(40)))
        self.assertEqual(stream.p.propagations, list(range(40)))
        self.assertEqual(stream.p.prompts, [(0, 1), (10, 2)])
        self.assertLessEqual(len(memory['non_cond_frame_outputs']), 16)
        self.assertEqual(state['num_frames'], 40)

    def test_loader_restored_even_when_init_fails(self):
        stream = self.tracker()
        stream.p.fail_init = True
        original = load_video_frames
        with self.assertRaises(RuntimeError):
            stream.start(image(0), {})
        self.assertIs(load_video_frames, original)

    def test_legacy_stream_still_uses_fixed_horizon(self):
        stream = StreamTracker(Predictor())
        stream.start(image(0), {1: [0, 0, 1, 1]})
        stream.step(image(1))
        self.assertEqual(stream.state['num_frames'], 100000)


class StreamAdapterTests(unittest.TestCase):
    def adapter(self, detect=detector):
        tracker = StreamTracker(Predictor(), observed_horizon=True)
        return StreamVisionAdapter(detect, tracker, bf16=False)

    def test_dino_late_registration_unchanged_prefix_decisions_and_cache_reset(self):
        calls = []
        def detect(rgb, prompt):
            calls.append((int(rgb[0, 0, 0]), prompt))
            return detector(rgb, prompt)
        adapter = self.adapter(detect)
        adapter.reset('episode')
        for t in range(12):
            masks, record = adapter.process(image(t), 'episode', t)
            self.assertEqual(masks.dtype, bool)
            self.assertEqual(masks.shape, (2, 4, 4))
            self.assertEqual(masks[1].any(), t >= 10)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(calls), 4)
        self.assertGreater(record['detector_requests'], record['detector_calls'])
        self.assertEqual(adapter.tracker.p.prompts, [(0, 1), (10, 2)])
        self.assertEqual(adapter.found['box']['decision_observation'], 10)
        adapter.reset('another')
        adapter.process(image(0), 'another', 0)
        self.assertEqual(len(calls), 6)
        adapter.close()

    def test_no_detection_and_suffix_causality(self):
        empty = self.adapter(lambda *args: ([], []))
        empty.reset('empty')
        for t in range(20):
            self.assertFalse(empty.process(image(t), 'empty', t)[0].any())
        self.assertEqual(empty.tracker.p.propagations, [])
        empty.close()
        adapter = self.adapter()
        def sequence(changed):
            adapter.reset('ep')
            return np.stack([adapter.process(image(200 if changed and t >= 11 else t), 'ep', t)[0]
                             for t in range(15)])
        first, second = sequence(False), sequence(True)
        np.testing.assert_array_equal(first[:11], second[:11])
        adapter.close()

    def test_bad_input_is_rejected_without_consuming_frame_and_failed_run_needs_reset(self):
        adapter = self.adapter()
        adapter.reset('ep')
        for rgb, identity, t in [(image(0), 'wrong', 0), (image(0), 'ep', 1),
                                  (image(0).astype(np.float32), 'ep', 0)]:
            with self.assertRaises(ValueError):
                adapter.process(rgb, identity, t)
        adapter.process(image(0), 'ep', 0)
        with self.assertRaises(ValueError):
            adapter.process(np.zeros((5, 4, 3), np.uint8), 'ep', 1)
        with patch.object(adapter.tracker, 'step', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                adapter.process(image(1), 'ep', 1)
        with self.assertRaises(RuntimeError):
            adapter.process(image(2), 'ep', 2)
        adapter.reset('ep')
        adapter.process(image(0), 'ep', 0)
        adapter.close()
        with self.assertRaises(RuntimeError):
            adapter.process(image(1), 'ep', 1)

    def test_default_factory_preserves_legacy_and_opt_in_requires_cuda(self):
        from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
        tracker = SimpleNamespace(reset=lambda state: None)
        adapter = make_online_vision({'vision_device': 'cpu'}, detector=detector, tracker=tracker)
        self.assertIsInstance(adapter, OnlineVisionAdapter)
        adapter.close()
        for backend in ('visual_stream', 'unknown'):
            with self.assertRaises(ValueError):
                make_online_vision({'vision_device': 'cpu', 'vision': {'backend': backend}}, detector=detector)

    def test_checkpoint_sources_only_expand_for_opt_in_and_rollout_config_stays_strict(self):
        import json
        from pathlib import Path
        from graph.integration_baseline.rollout_checkpoint import required_sources
        from graph.integration_baseline.rollout_stop_clock import validate_config
        config = json.loads((Path(__file__).parents[1] / 'rollout_config.json').read_text())
        validate_config(config)
        config['vision']['backend'] = 'visual_stream'
        config['vision_device'] = 'cuda'
        validate_config(config)
        self.assertNotIn('visual/scripts/stream_v1.py', required_sources())
        self.assertIn('visual/scripts/stream_v1.py', required_sources('visual_stream'))
        config['vision']['keep_ratio'] = .2
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_existing_manifest_cannot_be_relabelled_as_streaming(self):
        from graph.integration_baseline.direct_dataset import process_vision
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'dataset_manifest.json').write_text('{"episodes": []}')
            (root/'vision_manifest.json').write_text('{"episodes": []}')
            config = {'vision_device': 'cuda', 'vision': {'camera': 'base_camera', 'backend': 'visual_stream'}}
            with self.assertRaisesRegex(ValueError, 'new run UUID'):
                process_vision(root, config, None)


class BenchmarkGateTests(unittest.TestCase):
    def test_recorded_dino_pairs_roundtrip_through_atomic_json(self):
        from graph.integration_baseline.streaming_vision_benchmark import DinoMeter, FrozenDetector
        from graph.integration_baseline.direct_runtime import atomic, read
        class Detector:
            last_ms = {'forward': 1.}
            def __call__(self, rgb, prompt):
                return [[0.,0.,1.,1.]], [.8]
        meter = DinoMeter(Detector())
        boxes, scores = meter(image(0), 'peg')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'detections.json'
            atomic(path, meter.detections)
            restored_boxes, restored_scores = FrozenDetector(read(path))(image(0), 'peg')
        self.assertEqual(boxes, restored_boxes)
        self.assertEqual(scores, restored_scores)

    def test_resource_guard_parses_host_pids_and_terminates_only_our_worker_group(self):
        from graph.integration_baseline.streaming_vision_benchmark import gpu_process_ids, monitor_worker
        from unittest.mock import Mock
        from graph.integration_baseline.direct_runtime import atomic
        self.assertEqual(gpu_process_ids({'processes': '100, python, 123 MiB\n200, python, 5 MiB'}), {100,200})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            atomic(folder/'gpu_owner.json', {'nvml_host_pid': 100})
            process = Mock(pid=1234)
            process.poll.return_value = None
            process.wait.return_value = -15
            with patch('graph.integration_baseline.streaming_vision_benchmark.gpu_snapshot',
                       return_value={'processes':'100, python, 123 MiB\n200, other, 5 MiB'}), \
                 patch('graph.integration_baseline.streaming_vision_benchmark.os.killpg') as kill:
                code, reason = monitor_worker(process, folder)
            self.assertEqual(code, -15)
            self.assertEqual(reason['foreign_nvml_host_pids'], [200])
            self.assertEqual(kill.call_args.args[0], 1234)
            with patch('graph.integration_baseline.streaming_vision_benchmark.shutil.disk_usage',
                       return_value=SimpleNamespace(free=1)), \
                 patch('graph.integration_baseline.streaming_vision_benchmark.os.killpg') as kill:
                code, reason = monitor_worker(process, folder, minimum_free_bytes=2)
            self.assertEqual(reason['status'], 'ABORTED_FOR_DISK_RESERVE')
            self.assertEqual(kill.call_args.args[0], 1234)

    def test_adoption_requires_latency_masks_repeatability_and_downstream_equality(self):
        from graph.integration_baseline.streaming_vision_benchmark import summarize
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'variants': {'direct': {}, 'adapter': {}}, 'episodes': [30002]}
            def write(name, p95, pixels=0, action=0, repeatable=True):
                p = root/name; p.mkdir(exist_ok=True)
                masks = np.zeros((2,2,4,4), bool); masks.ravel()[:pixels] = True
                np.savez_compressed(p/'ep30002_masks.npz', masks=masks)
                np.savez_compressed(p/'actions.npz', actions=np.full((2,7), action, np.float32))
                (p/'pipeline_frames.json').write_text(json.dumps([{'packet_digest':'same','graph_digest':'same'}]*2))
                result = dict(initializations={}, repeats_exact=repeatable, checks={'30002': {'status':'PASS'}},
                              groups={'sam2_tracking_only': {'median':40., 'p95':p95}, 'dino_scan':{'n':0}})
                (p/'result.json').write_text(json.dumps(result))
            write('direct', 45)
            write('adapter', 46)
            result = summarize(root, plan)
            self.assertEqual(result['gates']['status'], 'RESTORED')
            self.assertTrue(result['gates']['tracking_20hz_pass'])
            for kwargs in ({'p95':60}, {'p95':46,'pixels':1}, {'p95':46,'action':.01},
                           {'p95':46,'repeatable':False}):
                write('adapter', **kwargs)
                self.assertEqual(summarize(root, plan)['gates']['status'], 'GATE_FAILED')


if __name__ == '__main__':
    unittest.main()

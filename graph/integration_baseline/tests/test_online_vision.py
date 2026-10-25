"""Synthetic streaming contracts. Real weights/simulator are exercised by CLI smoke."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter, ObservedFrames


class FakeTracker:
    def __init__(self):
        self.predictor = SimpleNamespace(image_size=4, device=torch.device('cpu'))
        self.all_states = []
        self.reset_states = []

    def start(self, directory, new, frame_idx=0):
        assert len(list(Path(directory).glob('*.jpg'))) == frame_idx+1
        state = dict(images=None, num_frames=frame_idx+1, cached_features={'sentinel':object()},
                     output_dict_per_obj={}, frames_tracked_per_obj={}, obj_ids=list(new), calls=[])
        self.all_states.append(state)
        return state

    def track(self, state, start_frame=0):
        assert len(state['images']) == state['num_frames'] == start_frame+1
        # Prove future/negative access is impossible at the actual tracker boundary.
        for index in (start_frame+1, -1):
            try:
                state['images'][index]
            except IndexError:
                pass
            else:
                raise AssertionError('future/negative access allowed')
        state['calls'].append(start_frame)
        state['output_dict_per_obj'][start_frame] = state['images'][start_frame]
        mask = torch.zeros(4,4,dtype=torch.bool)
        mask[start_frame % 4, start_frame % 4] = True
        yield start_frame, {oid:mask.clone() for oid in state['obj_ids']}

    def reset(self, state):
        self.reset_states.append(state)


def append_fixture(self, path, predictor):
    self.frames.append(torch.zeros(3,4,4))
    return 4,4


def initializer(frames, detector):
    t = len(frames)-1
    return dict(peg=dict(box=[0,0,1,1],frame=0),
                box=dict(box=[2,2,3,3] if t >= 5 else None,frame=5 if t >= 5 else None))


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(ObservedFrames,'append_jpeg',append_fixture),
                        patch('graph.integration_baseline.online_vision_adapter.detect_initial', initializer)]
        for p in self.patches:
            p.start()
        self.tracker = FakeTracker()
        self.online = OnlineVisionAdapter(None, self.tracker)
        self.online.reset('ep')

    def tearDown(self):
        self.online.close()
        for p in reversed(self.patches):
            p.stop()

    def process(self, t, value=0):
        return self.online.process(np.full((4,4,3),value,np.uint8), 'ep', t)[0]

    def test_current_only_late_object_preserves_old_memory_and_identity(self):
        outputs = [self.process(t) for t in range(7)]
        self.assertFalse(outputs[4][1].any())
        self.assertTrue(outputs[5][1].any())
        first, second = self.tracker.all_states
        self.assertEqual(first['calls'], list(range(7)))
        self.assertEqual(second['calls'], [5,6])
        self.assertIs(first['images'], second['images'])
        self.assertEqual(len(first['output_dict_per_obj']), 7)
        self.assertFalse(self.tracker.reset_states)
        self.assertEqual(outputs[6].dtype, bool)
        self.assertEqual(outputs[6].shape, (2,4,4))

    def test_duplicate_gap_identity_and_shape_are_errors(self):
        self.process(0)
        for t in (0,2):
            with self.assertRaises(ValueError):
                self.process(t)
        with self.assertRaises(ValueError):
            self.online.process(np.zeros((4,4,3),np.uint8),'other',1)
        with self.assertRaises(ValueError):
            self.online.process(np.zeros((5,4,3),np.uint8),'ep',1)

    def test_episode_reset_no_state_leak_and_prefix_suffix_causality(self):
        before = np.stack([self.process(t) for t in range(7)])
        self.online.reset('ep')
        after = np.stack([self.process(t, 200 if t >= 5 else 0) for t in range(7)])
        np.testing.assert_array_equal(before[:5],after[:5])
        self.assertEqual(len(self.tracker.reset_states),2)
        self.assertEqual(len(self.online.images),7)

    def test_empty_object_is_valid(self):
        with patch('graph.integration_baseline.online_vision_adapter.detect_initial',
                   return_value={name:dict(box=None,frame=None) for name in ('peg','box')}):
            self.assertFalse(self.process(0).any())
            self.assertEqual(self.online.states,{})

    def test_wrong_frame_and_object_identity_fail(self):
        def invalid(state,start_frame):
            yield start_frame+1,{1:torch.zeros(4,4,dtype=torch.bool)}
        with patch.object(self.tracker,'track',invalid):
            with self.assertRaises(ValueError):
                self.process(0)


if __name__ == '__main__':
    unittest.main()

"""Prefix-invariance tests without foundation-model downloads."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.integration_baseline.vision import causal_masks


class FakeTracker:
    def start(self, path, boxes, frame_idx=0):
        return dict(n=len(list(Path(path).glob('*.jpg'))),ids=list(boxes),start=frame_idx)

    def track(self, state, start_frame=0):
        for t in range(start_frame,state['num_frames']):
            assert state['num_frames']==t+1
            yield t,{i:torch.full((4,4),bool((t+i)%2)) for i in state['ids']}

    def reset(self,state):
        pass


class CausalMaskTests(unittest.TestCase):
    def test_prefix_only_late_init_no_backfill_and_suffix_invariance(self):
        frames=np.zeros((12,4,4,3),np.uint8)
        frames[:,0,0,0]=np.arange(12)
        calls=[]
        def initialize(prefix,detector):
            calls.append(prefix[:,0,0,0].copy())
            n=len(prefix)
            return {'peg':dict(box=[0,0,1,1],frame=0),
                    'box':dict(box=[2,2,3,3] if n>=6 else None,frame=5 if n>=6 else None)}
        with patch('graph.integration_baseline.vision.detect_initial',side_effect=initialize), \
             patch('graph.integration_baseline.vision.sha256',return_value='fixture-checkpoint'):
            full,meta=causal_masks(frames,None,FakeTracker())
            changed=frames.copy();changed[8:]=255
            other,_=causal_masks(changed,None,FakeTracker())
            prefix,_=causal_masks(frames[:8],None,FakeTracker())
        np.testing.assert_array_equal(full[:8],other[:8])
        np.testing.assert_array_equal(full[:8],prefix)
        self.assertFalse(full[:5,1].any())
        self.assertEqual(meta['initialization']['box']['decision_observation'],5)
        for values in calls:
            np.testing.assert_array_equal(values,np.arange(len(values)))


if __name__=='__main__':
    unittest.main()

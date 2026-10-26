"""Task08-F action-source and matched-prefix contracts; no simulator or training."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from graph.integration_baseline.bc08f_branch import BRANCHES, FIELDS, choose_action, gate_pair


class BranchContract(unittest.TestCase):
    def test_action_source_changes_only_at_frozen_branch(self):
        expert=np.array([.1,.2,.3,.4,.5,.6,-1],np.float32)
        model=np.array([-.1,-.2,-.3,-.4,-.5,-.6,1],np.float32)
        self.assertEqual(BRANCHES,(0,10,25,49))
        for arm in ('EM','ME','MM'):
            for t, expected in ((9,expert),(10,expert if arm[0]=='E' else model),
                                (11,expert if arm[1]=='E' else model)):
                action,_=choose_action('A',arm,t,10,model,expert)
                np.testing.assert_array_equal(action,expected)
                self.assertFalse(np.shares_memory(action,expected))
        with self.assertRaises(ValueError):choose_action('A','ME',0,11,model,expert)

    def test_prefix_gate_catches_observation_or_hidden_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            a.mkdir();b.mkdir()
            gt={name:np.zeros(9 if name in ('qpos','qvel') else 7,dtype=np.float32) for name in FIELDS}
            data=dict(rgb=np.zeros((2,2,3),np.uint8),depth_raw_mm=np.ones((2,2),np.int16),
                      hidden=np.zeros((1,1,144),np.float32),
                      graph_edge_index=np.zeros((2,1),np.int64),
                      predicted_action=np.zeros(7,np.float32),
                      **{'gt_'+name:value for name,value in gt.items()})
            np.savez_compressed(a/'branch_input.npz',**data)
            np.savez_compressed(b/'branch_input.npz',**data)
            ref={name:value.tolist() for name,value in gt.items()}
            self.assertTrue(gate_pair(a,b,ref)['pass_gate'])
            bad=dict(data,hidden=np.ones((1,1,144),np.float32))
            np.savez_compressed(b/'branch_input.npz',**bad)
            self.assertFalse(gate_pair(a,b,ref)['pass_gate'])


if __name__=='__main__':unittest.main()

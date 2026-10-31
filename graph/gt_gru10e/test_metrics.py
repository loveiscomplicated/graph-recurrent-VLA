"""Validate stored rollout semantics, without reconstructing model behavior."""
import json
import unittest
from pathlib import Path

class StoredMetrics(unittest.TestCase):
    def test_closed_loop_alignment_and_reach_latch(self):
        root=Path('outputs/task10e/benchmark')
        episodes=list(root.glob('rollouts/s*/*/episode.json'))
        self.assertEqual(len(episodes),54)
        for ep in episodes:
            data=json.loads(ep.read_text());m=data['metrics'];trace=json.loads(ep.with_name('trace.json').read_text())
            self.assertEqual([r['timestep'] for r in trace],list(range(len(trace))))
            self.assertEqual([r['post_step'] for r in trace],list(range(1,len(trace)+1)))
            self.assertLessEqual(len(trace),250)
            expected=any(r['reach_candidate'] for r in trace[:130])
            self.assertEqual(m['reach_success'],expected)
            stop=next((r['post_step'] for r in trace[:130] if r['reach_candidate']),min(130,len(trace)))
            self.assertEqual(m['primary_actions'],stop)
            self.assertEqual(m['final_target_position_error_m'],trace[stop-1]['target_position_error_m'])
            for r in trace:
                self.assertEqual(r['action'],r['controller']['controller_input'])
                self.assertIn(r['action'][6],[-1.0,1.0])
            if expected:
                r=trace[stop-1]
                self.assertFalse(r['early_close']);self.assertTrue(r['final']);self.assertGreaterEqual(r['opening_m'],.065)
                self.assertLess(r['peg_motion_m'],.002)

if __name__=='__main__':unittest.main()

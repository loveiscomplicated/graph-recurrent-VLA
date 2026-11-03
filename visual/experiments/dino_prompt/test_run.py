"""CPU tests for causal selection and GT evaluation, without model downloads."""
import hashlib
import unittest
import numpy as np
from visual.experiments.dino_prompt.run import (
    BASELINE, CachedDetector, choose, evaluate, gt_bbox, temporal, transitions,
)


class PromptEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 100, 3), np.uint8)
        self.seg = np.zeros((100, 100), np.uint8)
        self.seg[10:30, 10:30] = 18
        self.seg[60:80, 60:80] = 19
        self.frame = dict(id='0355_base_camera_000', parent='0355', split='search', control=False,
                          object_ids={'peg': [18], 'box': [19]}, gt_pixels={'peg': 400, 'box': 400})
        self.cache = {}
        self.digest = hashlib.sha256(self.rgb.tobytes()).hexdigest()

    def detector(self, peg, box):
        for prompt, data in zip(BASELINE, [peg, box]):
            self.cache[self.digest, prompt] = dict(boxes=[v[0] for v in data], scores=[v[1] for v in data])
        return CachedDetector(self.cache)

    def test_null_drop_and_wrong_owner_are_distinct(self):
        d = self.detector([([10, 10, 30, 30], .29)], [([60, 60, 80, 80], .8)])
        rows = evaluate(self.frame, self.rgb, self.seg, d, BASELINE)
        self.assertEqual(rows[0]['drop'], 'NULL_SCORE')
        self.assertTrue(rows[0]['proposal_hit'])
        d = self.detector([([60, 60, 80, 80], .9)], [])
        rows = evaluate(self.frame, self.rgb, self.seg, d, BASELINE)
        self.assertTrue(rows[0]['wrong_object'])
        self.assertFalse(rows[0]['correct'])

    def test_area_filter_and_small_visibility(self):
        seg = np.zeros_like(self.seg)
        seg[:60, :60] = 18
        self.frame['gt_pixels']['peg'] = 3600
        d = self.detector([([0, 0, 60, 60], .9)], [])
        row = evaluate(self.frame, self.rgb, seg, d, BASELINE)[0]
        self.assertEqual(row['drop'], 'AREA')
        self.frame['gt_pixels']['peg'] = 7
        row = evaluate(self.frame, self.rgb, seg, d, BASELINE)[0]
        self.assertFalse(row['supported'])
        self.assertEqual(row['drop'], 'SMALL_OR_INVISIBLE')

    def test_gt_cannot_change_consumer_choice(self):
        d = self.detector([([10, 10, 30, 30], .8)], [([60, 60, 80, 80], .7)])
        a = evaluate(self.frame, self.rgb, self.seg, d, BASELINE)
        b = evaluate(self.frame, self.rgb, np.zeros_like(self.seg), d, BASELINE)
        self.assertEqual([r['selected_box'] for r in a], [r['selected_box'] for r in b])
        self.assertEqual(gt_bbox(self.seg == 18), [10, 10, 30, 30])

    def test_control_preservation_precedes_more_hits(self):
        base = [dict(frame='normal', object='peg', control=True, correct=True, supported=True,
                     selected_iou=.8, wrong_object=False, proposal_hit=True)]
        alternative = [dict(base[0], correct=False, selected_iou=0)] + [
            dict(base[0], frame='fail'+str(i), control=False) for i in range(3)]
        pair = ('alternative', BASELINE[1])
        self.assertEqual(choose({BASELINE: base, pair: alternative}, [BASELINE, pair],
                                {p: 3 for p in (*BASELINE, 'alternative')}), BASELINE)
        change = transitions([dict(base[0], drop='SUCCESS')],
                             [dict(base[0], correct=False, drop='NULL_SCORE')])
        self.assertTrue(change[0]['regressed'])
        self.assertFalse(change[0]['recovered'])

    def test_temporal_requires_two_scans(self):
        frames, images, cache, rows = [], {}, {}, []
        for t in range(0, 31, 5):
            rgb = self.rgb.copy()
            rgb[0, 0, 0] = t
            fid = f'0355_base_camera_{t:03d}'
            frames.append(dict(self.frame, id=fid, camera='base_camera', timestep=t))
            images[fid] = rgb
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            for obj, prompt in zip(('peg', 'box'), BASELINE):
                box = [10, 10, 30, 30] if obj == 'peg' else [60, 60, 80, 80]
                cache[digest, prompt] = dict(boxes=[box], scores=[.2 if obj == 'peg' and t < 25 else .8])
                rows.append(dict(frame=fid, object=obj, gt_bbox=box, supported=True))
        result = temporal(frames, images, CachedDetector(cache), BASELINE, rows)
        self.assertEqual(result['peg']['frame'], 30)
        self.assertTrue(result['peg']['correct_at_commit'])
        self.assertEqual(result['box']['frame'], 0)


if __name__ == '__main__':
    unittest.main()

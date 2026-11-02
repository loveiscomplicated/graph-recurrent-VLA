"""Parent balance and unsupported-reference denominators, independent of model."""
import json
from pathlib import Path
import tempfile
import unittest
from .aggregate_development import run

class Aggregation(unittest.TestCase):
    def episode(self,root,eid,parent,precision,n=1,supported=True):
        p=root/'perception'/eid;p.mkdir(parents=True)
        m=dict(input_gt_support=10,output_support=10,true_positive=int(10*precision),precision=precision,recall=precision,iou=precision,robot_pixels=0,wrong_object_pixels=0,background_pixels=int(10*(1-precision))) if supported else None
        row=dict(episode=eid,parent_id=parent,camera='base_camera',object_id=0,stage='final_graph_sources',timestep=0,metrics=m)
        (p/'stage_rows.json').write_text(json.dumps([row]*n));(p/'representative_points.json').write_text('[]')
    def test_branch_and_frame_count_do_not_reweight_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.episode(root,'a','p1',1,10);self.episode(root,'b','p1',0);self.episode(root,'c','p2',1)
            run(root);rows=json.loads((root/'development_parent_macro.json').read_text())
            row=next(r for r in rows if r['support_stratum']=='all_supported')
            self.assertEqual(row['precision'],.75);self.assertEqual(row['parents'],2)
    def test_unsupported_frames_remain_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.episode(root,'a','p1',0,n=3,supported=False);run(root)
            rows=json.loads((root/'development_episode_macro.json').read_text())
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['frames'],3);self.assertEqual(rows[0]['supported_frames'],0);self.assertIsNone(rows[0]['precision'])
if __name__=='__main__':unittest.main()

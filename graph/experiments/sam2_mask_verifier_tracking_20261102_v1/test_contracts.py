import unittest
import numpy as np
from common import selected,mask_metrics
class Contracts(unittest.TestCase):
    def test_matching_query_and_deterministic_rank(self):
        rows=[dict(id='a',image_id='i',query_object='peg',DINO_score=.8,rank=1),dict(id='b',image_id='i',query_object='peg',DINO_score=.8,rank=0),dict(id='c',image_id='i',query_object='box',DINO_score=.99,rank=0)]
        self.assertEqual(selected(rows,{'a':'PEG','b':'PEG','c':'PEG'},'i','peg')['id'],'b')
        self.assertIsNone(selected(rows,{'a':'ROBOT','b':'UNCERTAIN'},'i','peg'))
    def test_wrong_actor_separate_from_background_and_missing(self):
        target=np.array([[1,0],[0,0]],bool);robot=np.array([[0,1],[0,0]],bool);bg=~(target|robot)
        a=mask_metrics(robot,target,dict(robot=robot,background=bg));self.assertTrue(a['wrong_object']);self.assertEqual(a['IoU'],0)
        b=mask_metrics(bg,target,dict(robot=robot,background=bg));self.assertFalse(b['wrong_object']);self.assertEqual(b['IoU'],0)
        c=mask_metrics(np.zeros((2,2),bool),target,dict(robot=robot,background=bg));self.assertEqual(c['IoU'],0);self.assertIsNone(c['pixel_precision'])
    def test_perfect_mask(self):
        t=np.eye(3,dtype=bool);m=mask_metrics(t,t,dict(robot=np.zeros((3,3),bool)));self.assertEqual(m['IoU'],1);self.assertEqual(m['pixel_recall'],1)
if __name__=='__main__':unittest.main()

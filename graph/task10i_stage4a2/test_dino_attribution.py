import unittest
import numpy as np
from .dino_semantic_attribution import actor_rectangles, counterfactual_assign, first_drop
from .reference_pair_audit import rectangle_identity
from visual.runtime.initialization import assign


class SameOutputAttributionContracts(unittest.TestCase):
    def test_vectorized_actor_support_matches_raster_crop_including_border(self):
        rng=np.random.default_rng(42)
        rr=dict(object_ids=dict(peg=[1],box=[2]),robot_ids=[3])
        seg=np.zeros((128,128),int);seg[20:25,10:80]=1;seg[30:60,30:60]=2;seg[10:90,100:120]=3
        low=rng.uniform(-30,130,(100,2));boxes=np.column_stack((low,low+rng.uniform(0,90,(100,2))))
        labels,_=actor_rectangles(boxes,seg,rr)
        for box,label in zip(boxes,labels):
            lo=np.clip(np.floor(box[:2]).astype(int),0,128);hi=np.clip(np.ceil(box[2:]).astype(int),0,128)
            mask=np.zeros_like(seg,bool);mask[lo[1]:hi[1],lo[0]:hi[0]]=True
            self.assertEqual(label,rectangle_identity(mask,seg,rr)[0])

    def test_same_output_null_and_peer_interventions_reproduce_original(self):
        peg=[([1,2,3,4],.2),([8,7,9,10],.6)];box=[([1,2,3,4],.4)]
        self.assertEqual(assign(peg,box),counterfactual_assign(peg,box))
        self.assertEqual(counterfactual_assign([([1,2,3,4],.2)],[]),(None,None))
        self.assertIsNotNone(counterfactual_assign([([1,2,3,4],.2)],[],missing=0)[0])
        peg=[([1,2,3,4],.6)];box=[([1,2,3,4],.7)]
        self.assertIsNone(counterfactual_assign(peg,box)[0])
        self.assertIsNotNone(counterfactual_assign(peg,box,exclusion=float('inf'))[0])

    def test_null_is_joint_drop_not_an_earlier_score_filter(self):
        one=np.array([True]);zero=np.array([False])
        self.assertEqual(first_drop(one,one,one,one,False,True),'JOINT_NULL_SELECTION')
        self.assertEqual(first_drop(one,one,one,zero,False,True),'TOP5')
        self.assertEqual(first_drop(one,zero,one,one,False,True),'SCORE_THRESHOLD')


if __name__=='__main__':unittest.main()

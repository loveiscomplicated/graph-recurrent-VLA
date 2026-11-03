"""Tracker's optional filter executes before the existing CPU adapter."""
import unittest
import torch
from visual.runtime.tracking import SAM2Tracker


class TrackingFilter(unittest.TestCase):
    def tracker(self):
        class Predictor:
            def propagate_in_video(self,state,start_frame_idx):
                yield start_frame_idx,[2,1],torch.tensor([[[[1.,-1.]]],[[[-1.,1.]]]])
        return SAM2Tracker(predictor=Predictor())

    def test_default_and_callback_frame_ids_mask_values(self):
        t=self.tracker();index,raw=next(t.track({},start_frame=3))
        self.assertEqual(index,3);seen=[]
        def callback(frame,masks):
            seen.append((frame,set(masks)))
            return {key:torch.zeros_like(mask) for key,mask in masks.items()}
        frame,clean=next(t.track({},start_frame=3,mask_filter=callback))
        self.assertEqual(seen,[(3,{1,2})]);self.assertEqual(frame,3)
        self.assertFalse(any(mask.any() for mask in clean.values()))
        self.assertTrue(all(mask.any() for mask in raw.values()))

    def test_callback_must_preserve_ids_shape_dtype_and_device(self):
        for fn in (lambda f,m:{},lambda f,m:{k:v.float() for k,v in m.items()},
                   lambda f,m:{k:v[:,0] for k,v in m.items()}):
            with self.subTest(fn=fn),self.assertRaisesRegex(ValueError,'mask_filter'):
                next(self.tracker().track({},mask_filter=fn))


if __name__=='__main__':unittest.main()

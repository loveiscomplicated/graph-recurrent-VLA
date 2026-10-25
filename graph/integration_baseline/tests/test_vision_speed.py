import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from graph.integration_baseline.vision_speed import MemoizedDetector, guard_preflight
from graph.integration_baseline.vision import causal_masks
from graph.integration_baseline.tests.test_causality import FakeTracker


class VisionSpeedTests(unittest.TestCase):
    def test_execution_revision_preserves_original_and_rejects_tampering(self):
        from graph.integration_baseline.vision_speed import SETTINGS,verify_revision_sources
        from graph.integration_baseline.direct_runtime import atomic,sha
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);revision=run/'execution_revisions/r1';revision.mkdir(parents=True)
            original={'graph/integration_baseline/direct_dataset.py':'old'}
            changed={'graph/integration_baseline/direct_dataset.py':'new'}
            atomic(run/'source_hashes.json',original);atomic(run/'config.yaml',{'frozen':True})
            original_bytes=(run/'source_hashes.json').read_bytes()
            atomic(revision/'verification.json',dict(status='PASS',compiler_backend=SETTINGS['compiler_backend'],exact={'masks':True,'packets':True,'actions':True,'existing_cache':True},speedup=2))
            receipt=dict(status='VALIDATED',settings=SETTINGS,revision_id='r1',
                original_sources_sha256=sha(run/'source_hashes.json'),config_sha256=sha(run/'config.yaml'),
                verification_sha256=sha(revision/'verification.json'),contract_tests_returncode=0,source_hashes=changed)
            atomic(revision/'revision.json',receipt)
            atomic(run/'execution_revision.json',dict(path='execution_revisions/r1/revision.json',sha256=sha(revision/'revision.json')))
            self.assertTrue(verify_revision_sources(run,original,changed))
            self.assertEqual(original_bytes,(run/'source_hashes.json').read_bytes())
            with self.assertRaises(ValueError):verify_revision_sources(run,original,{**changed,'recurrent/expert.py':'modified'})
            atomic(run/'config.yaml',{'frozen':False})
            with self.assertRaises(ValueError):verify_revision_sources(run,original,changed)

    def test_resume_preserves_existing_discord_tmux_session(self):
        from graph.integration_baseline import run_control
        from graph.integration_baseline.direct_runtime import atomic
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/('a'*32);run.mkdir()
            atomic(run/'config.yaml',dict(auto_callback_required=False,minimum_free_bytes=0))
            atomic(run/'preflight/verification.json',dict(status='PASS'))
            atomic(run/'status.json',dict(status='STOPPED'))
            atomic(run/'STOP',dict(reason='fixture'))
            with patch.object(run_control,'source_gate'),patch.object(run_control,'resume_gate'), \
                 patch.object(run_control.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as execute:
                run_control.launch(run,resume=True)
                commands=[call.args[0] for call in execute.call_args_list]
                self.assertTrue(any('new-window' in cmd for cmd in commands))
                self.assertFalse(any('new-session' in cmd for cmd in commands))
                self.assertFalse((run/'STOP').exists())
    def test_detector_key_and_result_isolation(self):
        calls=[]
        def detect(rgb,prompt):
            calls.append((rgb.copy(),prompt))
            return [[0.,0.,1.,1.]], [.8]
        d=MemoizedDetector(detect);image=np.zeros((4,4,3),np.uint8)
        a=d(image,'peg');a[0][0][0]=999
        self.assertEqual(d(image.copy(),'peg')[0][0][0],0.)
        d(image,'box');changed=image.copy();changed[0,0,0]=1;d(changed,'peg')
        self.assertEqual(d.statistics(),dict(requests=4,actual_inferences=3,cache_hits=1))
        d.reset_episode();self.assertFalse(d.cache)

    def test_preflight_skip_prompt_and_reset_invalidation(self):
        class Predictor:
            def propagate_in_video_preflight(self,s):
                for i,outputs in s['temp_output_dict_per_obj'].items():
                    for kind,rows in outputs.items():
                        s['output_dict_per_obj'][i][kind].update(rows);rows.clear()
        p=Predictor();counts=guard_preflight(p)
        def state():
            return dict(obj_ids=[1],output_dict_per_obj={0:{'cond_frame_outputs':{},'non_cond_frame_outputs':{}}},
                temp_output_dict_per_obj={0:{'cond_frame_outputs':{0:'prompt'},'non_cond_frame_outputs':{}}})
        s=state();p.propagate_in_video_preflight(s);p.propagate_in_video_preflight(s)
        self.assertEqual(counts,dict(executed=1,skipped=1))
        s['temp_output_dict_per_obj'][0]['cond_frame_outputs'][5]='new prompt'
        p.propagate_in_video_preflight(s);p.propagate_in_video_preflight(s)
        self.assertEqual(counts,dict(executed=2,skipped=2))
        p.propagate_in_video_preflight(state())
        self.assertEqual(counts['executed'],3)
        s['output_dict_per_obj'][0]['cond_frame_outputs'].clear()
        p.propagate_in_video_preflight(s)
        self.assertEqual(counts['executed'],4)

    def test_cached_prefix_initializer_remains_causal_and_identical(self):
        frames=np.zeros((12,4,4,3),np.uint8);frames[:,0,0,0]=np.arange(12)
        calls=[]
        def detect(image,prompt):
            calls.append((int(image[0,0,0]),prompt))
            return [[0.,0.,1.,1.]], [.9]
        def initialize(prefix,detector):
            for i in range(0,len(prefix),5):detector(prefix[i],'peg')
            return {'peg':dict(box=[0,0,1,1],frame=0),
                    'box':dict(box=[2,2,3,3] if len(prefix)>5 else None,frame=5 if len(prefix)>5 else None)}
        def run(rgb,d):
            with patch('graph.integration_baseline.vision.detect_initial',side_effect=initialize), \
                 patch('graph.integration_baseline.vision.sha256',return_value='fixture'):
                return causal_masks(rgb,d,FakeTracker())[0]
        baseline=run(frames,detect);before=len(calls)
        optimized=run(frames,MemoizedDetector(detect));used=len(calls)-before
        np.testing.assert_array_equal(baseline,optimized)
        self.assertEqual(used,2)
        np.testing.assert_array_equal(optimized[:8],run(frames[:8],MemoizedDetector(detect)))
        changed=frames.copy();changed[8:]=255
        np.testing.assert_array_equal(optimized[:8],run(changed,MemoizedDetector(detect))[:8])


if __name__=='__main__':unittest.main()

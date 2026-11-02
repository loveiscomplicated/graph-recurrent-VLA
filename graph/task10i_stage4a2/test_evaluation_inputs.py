import tempfile
import unittest
from pathlib import Path
import numpy as np
from .evaluation_inputs import combine_masks,synchronous_references
from .final_capture import freeze_barrier
from graph.task10i_stage4a.run_pipeline import write,read,sha


class FrozenEvaluationInputContracts(unittest.TestCase):
    def test_post_inference_GT_barrier_precedes_GT_path_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);baseline=root/'baseline';challenger=root/'challenger'
            write(baseline/'pipeline_state.json',dict(status='RUNNING'))
            write(challenger/'pipeline_state.json',dict(status='FROZEN_LIVE_CONDITION_COMPLETE'))
            with self.assertRaises(AssertionError):
                synchronous_references(root/'output',root/'nonexistent_GT_manifest',baseline,challenger)
            self.assertFalse((root/'output').exists())

    def test_mask_bridge_requires_complete_source_hash_and_keeps_conditions_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected=root/'expected.json'
            write(expected,dict(entries=[dict(episode_id='e',parent_id='p',actions=1,raw_sha256='raw')],
                                parents=1,episodes=1,frames=2))
            for condition in ('B0','P'):
                folder=root/condition/'perception/e';folder.mkdir(parents=True)
                masks=np.zeros((2,8192),np.uint8);masks[0,0]=condition=='P'
                np.savez_compressed(folder/(condition+'.npz'),mask_bits=masks)
                write(folder/'graph_fingerprints.json',[])
                write(folder/'receipt.json',dict(condition=condition,raw_sha256='raw',frames=2,
                    files={p.name:sha(p) for p in folder.iterdir()}))
                write(root/condition/'pipeline_state.json',dict(status='FROZEN_LIVE_CONDITION_COMPLETE',condition=condition))
            combine_masks(root/'paired',expected,root/'B0',root/'P','P')
            with np.load(root/'paired/perception/e/B0.npz') as b,np.load(root/'paired/perception/e/P.npz') as c:
                self.assertNotEqual(b['mask_bits'][0,0],c['mask_bits'][0,0])
            receipt=read(root/'paired/perception/e/receipt.json')
            self.assertTrue(all(sha(root/'paired/perception/e'/p)==h for p,h in receipt['files'].items()))
            with self.assertRaises(AssertionError):combine_masks(root/'missing_isolation',expected,root/'B0',root/'P','P',True)
            internal=root/'P/perception/e/internal_B0.npz'
            internal.write_bytes((root/'B0/perception/e/B0.npz').read_bytes())
            r=read(root/'P/perception/e/receipt.json');r['files']['internal_B0.npz']=sha(internal);write(root/'P/perception/e/receipt.json',r)
            combine_masks(root/'verified_isolation',expected,root/'B0',root/'P','P',True)
            np.savez_compressed(internal,mask_bits=np.ones((2,8192),np.uint8))
            r['files']['internal_B0.npz']=sha(internal);write(root/'P/perception/e/receipt.json',r)
            with self.assertRaises(AssertionError):combine_masks(root/'contaminated_baseline',expected,root/'B0',root/'P','P',True)
            (root/'P/perception/e/P.npz').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):combine_masks(root/'corrupt',expected,root/'B0',root/'P','P')

    def test_internal_barrier_does_not_imply_final_capture_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.py';source.write_text('frozen')
            write(root/'candidate_source_frozen.json',dict(status='FROZEN',files={str(source):sha(source)}))
            write(root/'adoption_criteria_frozen.json',dict(status='FROZEN'))
            write(root/'candidate_config_frozen.json',dict(status='FROZEN',allow_internal_evaluation=True,
                allow_independent_final_capture=False,adoption_criteria_sha256=sha(root/'adoption_criteria_frozen.json')))
            self.assertEqual(len(freeze_barrier(root,None,None,scope='internal')),3)
            with self.assertRaises(AssertionError):freeze_barrier(root,None,None,scope='final')


if __name__=='__main__':unittest.main()

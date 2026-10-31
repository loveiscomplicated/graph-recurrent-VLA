"""State machine negative controls, without simulator/GPU requirements."""
import tempfile
from pathlib import Path
import unittest
from graph.plan_a_bc.common import write,sha
from .orchestrator import verify_stage,verify_all_stages


class EvidenceTests(unittest.TestCase):
    def test_exit_zero_cannot_replace_artifact_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verify_stage(Path(d),dict(stage_id='test',status='COMPLETED',exit_code=0,outputs={}))

    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'x.json';write(p,dict(value=1))
            row=dict(stage_id='test',status='COMPLETED',exit_code=0,outputs={'x.json':sha(p)})
            self.assertTrue(verify_stage(root,row));write(p,dict(value=2))
            with self.assertRaises(ValueError):verify_stage(root,row)

    def test_partial_pipeline_cannot_claim_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'pipeline_state.json',dict(status='COMPLETED',stages={}))
            with self.assertRaises(ValueError):verify_all_stages(root)


if __name__=='__main__':unittest.main()

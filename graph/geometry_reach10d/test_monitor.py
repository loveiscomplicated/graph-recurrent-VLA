"""The actual inherited incident Detector is exercised without transport."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


class MonitorContracts(unittest.TestCase):
    def test_error_stall_completion_and_permanent_deduplication(self):
        remote=Path('/workspace/plan_a_ops_3788798a719f4ec4a66c224ca42f5d63/event_ops.py')
        local=Path('outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/event_ops.py')
        path=remote if remote.exists() else local
        if not path.exists():self.skipTest('inherited operational module unavailable')
        spec=importlib.util.spec_from_file_location('inherited_task10d_detector_contract',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=dict(comparison_uuid='task10d_contract',timeouts_seconds=dict(cache=10,preparation=10))
        base=dict(pipeline=dict(status='RUNNING'),phase='data',signature=dict(frames=1),
            supervisor_running=True,substage='cache',recent_log='')
        with tempfile.TemporaryDirectory() as d:
            detector=module.Detector(d,cfg)
            self.assertEqual(detector.tick(base,clock=100),[])
            self.assertEqual(detector.tick(base,clock=110)[0]['kind'],'STALL')
            self.assertEqual(detector.tick(base,clock=120),[])
            failed=dict(base,pipeline=dict(status='FAILED',exception='fixture_error'))
            self.assertEqual(detector.tick(failed,clock=121)[0]['kind'],'ERROR')
            self.assertEqual(detector.tick(failed,clock=122),[])
            final=dict(base,pipeline=dict(status='COMPLETE'),completion_checks=dict(missing=[]))
            self.assertEqual(detector.tick(final,clock=123)[0]['kind'],'COMPLETION')
            self.assertEqual(detector.tick(final,clock=124),[])


if __name__=='__main__':unittest.main()

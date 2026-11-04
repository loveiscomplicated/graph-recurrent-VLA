"""Current runtime works without importing experiment or legacy helper packages."""
import ast
import importlib
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ('graph.experiments', 'graph.plan_a_geometry', 'graph.plan_a_bc',
             'graph.persistent_geometry', 'graph.integration_baseline',
             'graph.prototype', 'visual.runtime', 'visual.scripts', 'legacy_fps')


class HelperBoundaries(unittest.TestCase):
    def test_current_sources_do_not_import_historical_helpers(self):
        paths = [ROOT/'graph/current.py', ROOT/'graph/vision_input.py']
        for folder in ('graph/geometry', 'graph/perception', 'graph/backends', 'vla', 'visualization'):
            paths.extend(path for path in (ROOT/folder).rglob('*.py')
                         if not path.name.startswith('test_'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                for module in imports:
                    self.assertFalse(any(module == prefix or module.startswith(prefix+'.')
                                         for prefix in FORBIDDEN), (path, node.lineno, module))

    def test_entrypoints_import_with_historical_packages_blocked(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = %r
        if any(fullname == prefix or fullname.startswith(prefix+'.') for prefix in blocked):
            raise AssertionError('historical dependency: '+fullname)
sys.meta_path.insert(0, Block())
import tools.train, tools.rollout, tools.evaluate
import vla.data.graph_cache, visualization.server
from visualization.render_pointcloud import legacy_module
assert legacy_module()[0].__name__ == 'graph.perception.pointcloud'
from graph.current import CurrentGeometry
from unittest.mock import patch
with patch('graph.current.RobotFK'):
    builder = CurrentGeometry('unused', {}, perception_backend='cpu')
    assert builder.coordinate_frame == 'ee'
    builder.close()
print('historical packages blocked; entrypoints and default CPU builder loaded')
''' % (FORBIDDEN,)
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_legacy_aliases_share_current_objects_and_monkeypatch_namespace(self):
        for old, new in (
            ('graph.plan_a_geometry.core', 'graph.geometry.core'),
            ('graph.persistent_geometry.fusion', 'graph.geometry.fusion'),
            ('graph.plan_a_bc.geometry', 'graph.current'),
            ('visual.runtime.graph_perception', 'graph.perception.graph_perception'),
            ('visual.runtime.vision_packet', 'graph.perception.vision_packet'),
            ('visual.runtime.tracking', 'vla.perception.tracking'),
            ('graph.integration_baseline.tensor_online_vision', 'vla.perception.tensor_online'),
        ):
            self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_legacy_state_normalizer_keeps_feature_api(self):
        import torch
        from graph.prototype.v2.state import StateNormalizer
        from vla.data.normalization import StateStatistics
        values = torch.tensor([[1.,2.],[3.,4.]])
        from unittest.mock import patch
        with patch('graph.prototype.v2.state.shared_features', return_value=values):
            normalizer = StateNormalizer.fit(None, None, 'goal')
            self.assertIsInstance(normalizer, StateStatistics)
            torch.testing.assert_close(normalizer.apply(None,None,'goal'), torch.tensor([[-1.,-1.],[1.,1.]]))
            self.assertEqual(set(normalizer.as_dict()), {'mean','std'})


if __name__ == '__main__':
    unittest.main()

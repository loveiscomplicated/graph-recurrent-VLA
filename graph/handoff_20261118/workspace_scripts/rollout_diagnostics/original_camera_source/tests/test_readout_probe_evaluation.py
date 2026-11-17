"""A controlled comparison must reject unequal feature or update identities."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation.gnn.readout_probe import compare
from evaluation.gnn.vertex_diagnostics import MODELS
from tests import test_vertex_diagnostics


class ReadoutProbeEvaluationTests(unittest.TestCase):
    def test_comparison_keeps_pretrained_reference_separate_and_checks_protocol(self):
        fixture = test_vertex_diagnostics.VertexDiagnosticsTests().fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'original'
            original.mkdir()
            for name in MODELS:
                np.savez(original/f'{name}_predictions.npz',**fixture)
            parents = root/'parents.json'
            parents.write_text(json.dumps({'parent_by_episode':{'episode_a':'parent_a','episode_b':'parent_b'}}))
            for name in ('attention','multiseed'):
                directory = root/'runs'/name
                directory.mkdir(parents=True)
                data = {key:value.copy() for key,value in fixture.items()}
                if name=='multiseed':
                    data['peg_half_size_x_error_mm']+=1
                np.savez(directory/'predictions.npz',**data)
                (directory/'summary.json').write_text(json.dumps(dict(
                    variant=name,seed=0,trainable_parameters=100,updates=2,
                    feature_sha256='shared',training_order_sha256='same')))
            config=dict(runs=str(root/'runs'),original_predictions=str(original),
                        parent_manifest=str(parents),bootstrap_repetitions=100,
                        output={'root':str(root/'analysis')})
            path=compare(config)
            result=json.loads((path/'summary.json').read_text())
            value=result['paired_comparisons']['multiseed']['averaged']['peg_half_size_x_error_mm']
            self.assertEqual(value['seed_mean_delta'],1)
            self.assertEqual(result['pretrained_reference']['peg_half_size_x_error_mm']['mean'],0)
            summary=root/'runs/multiseed/summary.json'
            changed=json.loads(summary.read_text())
            changed['feature_sha256']='different'
            summary.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'feature_sha256'):
                compare(config)


if __name__=='__main__':
    unittest.main()

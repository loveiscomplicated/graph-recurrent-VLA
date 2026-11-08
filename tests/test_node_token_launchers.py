"""Exercise shell orchestration against a temporary, model-free main.py."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
JOINT_RECIPES = {
    'run_joint_node_token_seed.sh': 'train_joint_node_token_mamba',
    'run_joint_node_continuation_seed.sh': 'continue_joint_node_token_mamba',
    'run_joint_early_node_token_seed.sh': 'train_joint_early_node_token_mamba',
}
BATCH_LAUNCHERS = ['run_node_token_mamba.sh', 'run_node_token_early_stopping.sh']

# Each child records its own events. The optional barrier fails if a launcher
# advertised as parallel actually waits for one seed before starting the others.
CHILD_MAIN = '''
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
options = parser.parse_args()
assert Path(options.config).is_file()
recipe = Path(options.config).stem
seed = recipe.rsplit('_', 1)[-1]
is_seed = seed in ('0', '1', '2')
root = Path(os.environ['NODE_TOKEN_TEST_EVENTS'])
record = {
    'recipe': recipe, 'time': time.monotonic_ns(), 'cwd': str(Path.cwd()),
    'pythonpath': os.environ['PYTHONPATH'], 'experiments': os.environ['EXPERIMENTS_DIR'],
    'wandb': os.environ['WANDB_MODE'], 'threads': os.environ.get('OMP_NUM_THREADS'),
}
(root / (recipe + '_start.json')).write_text(json.dumps(record))
if is_seed and os.environ.get('NODE_TOKEN_TEST_BARRIER') == '1':
    deadline = time.monotonic() + 5
    while len(list(root.glob('*_start.json'))) < 3:
        if time.monotonic() > deadline:
            raise RuntimeError('Three seeds did not start concurrently')
        time.sleep(.01)
if not is_seed:
    assert len(list(root.glob('*_finish.json'))) == 3
print('stdout ' + recipe, flush=True)
print('stderr ' + recipe, file=sys.stderr, flush=True)
record['time'] = time.monotonic_ns()
(root / (recipe + '_finish.json')).write_text(json.dumps(record))
if recipe == os.environ.get('NODE_TOKEN_TEST_FAIL'):
    sys.exit(9)
'''


class NodeTokenLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source checkout'
        self.source.mkdir()
        (self.source / 'main.py').write_text(CHILD_MAIN)
        configs = self.source / 'graph/configs/temporal/insertion'
        configs.mkdir(parents=True)
        prefixes = list(JOINT_RECIPES.values()) + ['train_node_token_mamba', 'train_node_token_early_stopping']
        for prefix in prefixes:
            for seed in range(3):
                (configs / f'{prefix}_{seed}.yaml').write_text('trainer: unused:test\n')
        for name in ['aggregate_node_token_mamba', 'compare_node_token_early_stopping']:
            (configs / f'{name}.yaml').write_text('trainer: unused:test\n')
        self.runtime = self.root / 'runtime modules'
        self.runtime.mkdir()
        self.python = self.root / 'python executable'
        self.python.symlink_to(sys.executable)
        self.invocations = 0

    def run_launcher(self, script, seed=None, failure='', barrier=False):
        self.invocations += 1
        case = self.root / f'invocation_{self.invocations}'
        events = case / 'events'
        events.mkdir(parents=True)
        run_root = case / 'run root'
        logs = case / 'custom logs'
        command = ['bash', str(REPOSITORY / 'tools' / script)]
        if seed is not None:
            command.append(str(seed))
        command.extend([
            '--experiment-root', str(run_root), '--source-root', str(self.source),
            '--python', str(self.python), '--runtime-root', str(self.runtime), '--logs', str(logs),
        ])
        environment = dict(os.environ, NODE_TOKEN_TEST_EVENTS=str(events),
                           NODE_TOKEN_TEST_FAIL=failure, NODE_TOKEN_TEST_BARRIER=str(int(barrier)),
                           OMP_NUM_THREADS='7')
        result = subprocess.run(command, cwd=self.root, env=environment,
                                capture_output=True, text=True, timeout=15)
        records = {path.stem: json.loads(path.read_text()) for path in events.glob('*.json')}
        return result, run_root, logs, records

    def test_sequential_seeds_then_aggregation_and_overridden_paths(self):
        result, root, logs, records = self.run_launcher(BATCH_LAUNCHERS[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        recipes = [f'train_node_token_mamba_{seed}' for seed in range(3)] + ['aggregate_node_token_mamba']
        for previous, following in zip(recipes, recipes[1:]):
            self.assertLessEqual(records[previous + '_finish']['time'], records[following + '_start']['time'])
        first = records[recipes[0] + '_start']
        self.assertEqual(first['cwd'], str(self.source.resolve()))
        self.assertEqual(first['pythonpath'], f'{self.runtime.resolve()}:{self.source.resolve()}')
        self.assertEqual(first['experiments'], str(root.resolve() / 'experiments'))
        self.assertEqual(first['wandb'], 'disabled')
        self.assertEqual(first['threads'], '7')
        self.assertTrue((root / 'COMPLETE').is_file())
        self.assertFalse((root / 'FAILED').exists())
        self.assertEqual({path.name for path in logs.iterdir()},
                         {'seed_0.log', 'seed_1.log', 'seed_2.log', 'aggregate.log'})

    def test_parallel_seeds_finish_before_comparison(self):
        result, root, logs, records = self.run_launcher(BATCH_LAUNCHERS[1], barrier=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        seeds = [f'train_node_token_early_stopping_{seed}' for seed in range(3)]
        starts = [records[name + '_start']['time'] for name in seeds]
        finishes = [records[name + '_finish']['time'] for name in seeds]
        self.assertLessEqual(max(starts), min(finishes))
        self.assertLessEqual(max(finishes), records['compare_node_token_early_stopping_start']['time'])
        self.assertTrue((root / 'COMPLETE').is_file())
        for seed in range(3):
            self.assertTrue((root / f'seed_{seed}_COMPLETE').is_file())
            self.assertEqual(records[seeds[seed] + '_start']['threads'], '2')
        self.assertTrue((logs / 'comparison.log').is_file())

    def test_failed_seed_prevents_batch_comparison(self):
        for script, prefix in zip(BATCH_LAUNCHERS, ['train_node_token_mamba', 'train_node_token_early_stopping']):
            with self.subTest(script=script):
                result, root, logs, records = self.run_launcher(script, failure=prefix + '_1')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(name.startswith(('aggregate_', 'compare_')) for name in records))
                self.assertTrue((root / 'FAILED').is_file())
                self.assertFalse((root / 'COMPLETE').exists())
                self.assertIn('stderr', (logs / 'seed_1.log').read_text())

    def test_failed_comparison_does_not_mark_run_complete(self):
        result, root, logs, records = self.run_launcher(
            BATCH_LAUNCHERS[1], failure='compare_node_token_early_stopping')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((root / 'FAILED').is_file())
        self.assertFalse((root / 'COMPLETE').exists())

    def test_joint_launchers_keep_one_seed_and_stream_both_output_channels(self):
        for script, prefix in JOINT_RECIPES.items():
            with self.subTest(script=script):
                result, root, logs, records = self.run_launcher(script, seed=2)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(set(records), {prefix + '_2_start', prefix + '_2_finish'})
                self.assertTrue((root / 'seed_2_COMPLETE').is_file())
                for channel in ['stdout', 'stderr']:
                    self.assertIn(channel + ' ' + prefix, result.stdout)
                    self.assertIn(channel + ' ' + prefix, (logs / 'seed_2.log').read_text())

    def test_joint_pipeline_failures_propagate_through_tee(self):
        for script, prefix in JOINT_RECIPES.items():
            with self.subTest(script=script):
                result, root, logs, records = self.run_launcher(script, seed=2, failure=prefix + '_2')
                self.assertEqual(result.returncode, 9, result.stderr)
                self.assertTrue((root / 'seed_2_FAILED').is_file())
                self.assertFalse((root / 'seed_2_COMPLETE').exists())

    def test_help_and_invalid_arguments_start_no_jobs(self):
        for script in BATCH_LAUNCHERS + list(JOINT_RECIPES):
            result = subprocess.run(['bash', str(REPOSITORY / 'tools' / script), '--help'],
                                    cwd=self.root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--source-root', result.stdout)
        script = REPOSITORY / 'tools/run_joint_node_token_seed.sh'
        for arguments in [[], ['4'], ['0', '--python'], ['0', '--unknown']]:
            result = subprocess.run(['bash', str(script), *arguments], cwd=self.root,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
        self.assertFalse(list(self.root.glob('invocation_*')))


if __name__ == '__main__':
    unittest.main()

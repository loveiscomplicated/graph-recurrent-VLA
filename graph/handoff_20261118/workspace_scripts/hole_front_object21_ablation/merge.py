"""Assemble the verified seed ranges from the interrupted and detached runs."""
import json
from pathlib import Path

import yaml

from analyze import analyze

ROOT = Path(__file__).resolve().parent
STAGES = ('initial', 'first_grasp', 'held_align')
RANGES = {'full': (200, 650), 'shard_650_900': (650, 900), 'shard_900_1150': (900, 1150),
          'shard_1150_1400': (1150, 1400)}


def merge():
    rows, provenance, common = [], [], None
    # Confirmed by the first process's stdout, before the pipe failure at seed 650.
    terminals = {str(seed): 'insert_entry' for seed in range(200, 650, 75)}
    for name, (start, stop) in RANGES.items():
        source = ROOT / 'source_runs' / name
        (logger_path,) = (source / 'experiments').iterdir()
        metadata = json.loads((logger_path / 'metadata.json').read_text())
        config = yaml.safe_load((logger_path / 'config.yaml').read_text())
        assert metadata['git_commit'] == 'bb0114977456b32e9c59bd573fa79e1899226b0b'
        assert (metadata['status'] == 'finished' if name != 'full'
                else metadata['exception_type'] == 'BrokenPipeError')
        shared_config = {key: config[key] for key in
                         ('camera', 'max_expert_steps', 'device', 'tracker_gate_m', 'detector', 'checkpoint_sha256')}
        if common is None:
            common = shared_config
        assert shared_config == common
        batch = [json.loads(line) for line in (source / 'frames.jsonl').read_text().splitlines()]
        batch = [row for row in batch if start <= row['seed'] < stop]
        assert [row['seed'] for row in batch if 'initial' in row['stages']] == list(range(start, stop))
        rows.extend(batch)
        if name != 'full':
            terminals.update(json.loads((source / 'summary.json').read_text())['expert_terminals'])
        provenance.append(dict(run_id=metadata['run_id'], source=name, accepted_seeds=[start, stop],
                               status=metadata['status'], script_sha256=config['script_sha256']))
    config = dict(common, seeds=list(range(200, 1400)), expert_seeds=list(range(200, 1400, 75)))
    assert set(terminals) == {str(seed) for seed in config['expert_seeds']}
    result = dict(config=config, expert_terminals=terminals, source_runs=provenance,
                  metrics={stage: {'frames': sum(stage in row['stages'] for row in rows)} for stage in STAGES})
    output = ROOT / 'full'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'frames.jsonl').open('w') as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + '\n')
    analyze(output)


if __name__ == '__main__':
    merge()

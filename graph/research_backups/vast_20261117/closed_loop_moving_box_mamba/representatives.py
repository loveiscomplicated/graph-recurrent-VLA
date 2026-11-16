"""Select up to two moving-Box examples from each new-frontend outcome."""
import json
from pathlib import Path


def select(records):
    selected = []
    for outcome in (True, False):
        eligible = [r for r in records if r['hybrid']['success'] == outcome
                    and r['hybrid']['box_max_shift_mm'] > 1]
        ordered = sorted(eligible, key=lambda r: (r['hybrid']['box_max_shift_mm'], r['seed']))
        if ordered:
            examples = [0, 1] if len(ordered) == 2 else list(dict.fromkeys((len(ordered) // 2, len(ordered) - 1)))
            selected.extend(ordered[i] for i in examples)
    return selected


def load_rows(root):
    result = []
    for path in sorted(root.glob('shard_*/rollouts.jsonl')):
        for line in path.read_text().splitlines():
            result.append(dict(shard=path.parent.name, **json.loads(line)))
    return result


if __name__ == '__main__':
    for row in select(load_rows(Path('.'))):
        print(f"{row['shard']}/videos/{row['hybrid']['movie']}")

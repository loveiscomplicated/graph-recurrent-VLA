"""Independent identity, frozen-input and group-split checks."""
from pathlib import Path
from graph.integration_baseline.io import sha256


def validate_split(split, groups):
    occupied_ids, occupied_groups = set(), set()
    for name, ids in split.items():
        if len(ids) != len(set(ids)) or occupied_ids & set(ids):
            raise ValueError('episode split leakage')
        current = {groups[i] for i in ids}
        if len(current) != len(ids) or occupied_groups & current:
            raise ValueError('reset/perturbation family split leakage')
        occupied_ids.update(ids)
        occupied_groups.update(current)


def protection_snapshot(paths):
    return {str(Path(p).resolve()):sha256(p) for p in paths}


def verify_protection(before):
    changed, missing = [], []
    for filename, digest in before.items():
        if not Path(filename).is_file():
            missing.append(filename)
        elif sha256(filename) != digest:
            changed.append(filename)
    return dict(status='FAIL' if changed or missing else 'PASS', checked_files=len(before),
                changed=changed, missing=missing)

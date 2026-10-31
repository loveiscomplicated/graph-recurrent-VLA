"""Read-only byte inventory and Stage 2 consumer checks. No simulator imports."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

STAGE1_ID = '098bd09a011b4c9d99ca881d6a83b760'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, p)


def resolve(root, value, source_root=None):
    root = Path(root).resolve()
    p = Path(value)
    if p.is_absolute():
        assert source_root, ('unmapped_absolute_path', value)
        p = p.relative_to(source_root)
    assert '..' not in p.parts
    out = root / p
    assert out.resolve().is_relative_to(root), value
    assert out.is_file() and not out.is_symlink(), str(out)
    return out


def inventory(root):
    root = Path(root)
    result = {}
    if not root.exists():
        return result
    assert root.is_dir() and not root.is_symlink()
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink not supported: ' + str(p))
        if p.is_file():
            rel = str(p.relative_to(root))
            try:
                result[rel] = dict(bytes=p.stat().st_size, sha256=sha(p))
            except OSError as e:
                result[rel] = dict(unreadable=type(e).__name__)
    return result


def compare(expected, actual):
    rows = []
    for p in sorted(set(expected) | set(actual)):
        e, a = expected.get(p), actual.get(p)
        status = ('MISSING' if a is None else 'UNREADABLE' if 'unreadable' in a
                  else 'EXTRA' if e is None else 'MATCHED' if a == e else 'MISMATCHED')
        rows.append(dict(path=p, classification=status,
                         expected_bytes=e.get('bytes') if e else None,
                         remote_bytes=a.get('bytes') if a else None,
                         expected_sha256=e.get('sha256') if e else None,
                         remote_sha256=a.get('sha256') if a else None))
    return dict(counts={s: sum(r['classification'] == s for r in rows)
                        for s in ['MATCHED', 'MISSING', 'MISMATCHED', 'EXTRA', 'UNREADABLE']}, rows=rows)


def digest_array(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def csvrows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def consumer(root, source_root):
    root = Path(root)
    pool = read(root / 'candidate_pool_manifest.json')
    assert pool['status'] == 'COMPLETED' and pool['run_id'] == STAGE1_ID
    assert (pool['attempted'], pool['visibility_pass'], pool['eligible_unique']) == (926, 301, 300)
    assert pool['exact_duplicates'] == 0 and pool['stage2_started'] is False
    entries = read(root / 'eligible_candidate_pool.json')
    assert len(entries) == 300 and len({e['candidate_id'] for e in entries}) == 300
    tables = {n: csvrows(root / n) for n in ['candidate_attempts.csv', 'candidate_eligibility.csv',
              'candidate_state_descriptors.csv', 'candidate_visibility_metrics.csv']}
    ids = [r['candidate_id'] for r in tables['candidate_attempts.csv']]
    assert len(ids) == len(set(ids)) == 926
    for table in tables.values():
        assert [r['candidate_id'] for r in table] == ids
    attempts = {r['candidate_id']: r for r in tables['candidate_attempts.csv']}
    eligible = {r['candidate_id'] for r in attempts.values() if r['eligible'] == 'True'}
    assert eligible == {e['candidate_id'] for e in entries}
    assert sum(r['visibility'] == 'VISIBILITY_PASS' for r in attempts.values()) == 301
    assert sum(r['physical'] == 'PHYSICAL_VALID' for r in attempts.values()) == 923
    desc = {r['candidate_id']: r for r in tables['candidate_state_descriptors.csv']}
    metrics = {r['candidate_id']: r for r in tables['candidate_visibility_metrics.csv']}
    replay = read(root / 'candidate_replay_verification.json')
    assert replay['status'] == 'PASS' and replay['count'] == 300
    assert {r['candidate_id'] for r in replay['rows']} == eligible
    source_hash = sha(root / 'source_identity.json')
    runtime_hash = sha(root / 'runtime_identity.json')
    reverify_hash = sha(root / 'reverification_source_identity.json')
    with np.load(root / 'candidate_initial_states.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert list(arrays['candidate_id']) == ids
    for k, a in arrays.items():
        assert len(a) == 926
        if a.dtype.kind in 'fiu':
            assert np.isfinite(a).all(), k
    index = {cid: i for i, cid in enumerate(ids)}
    candidate_rows, raw_rows = [], []
    for e in entries:
        cid = e['candidate_id']
        assert e['split'] == 'UNASSIGNED' and e['expert_success'] == 'NOT_TESTED'
        assert e['source_sha256'] == source_hash and e['runtime_sha256'] == runtime_hash
        assert e['reverification_source_sha256'] == reverify_hash
        assert attempts[cid]['replay'] == 'REPLAY_PASS' and attempts[cid]['physical'] == 'PHYSICAL_VALID'
        assert int(attempts[cid]['reset_seed']) == e['reset_seed']
        initial = resolve(root, e['initial_npz'], source_root)
        for key in ['snapshot', 'gt_visibility_evidence', 'annotated_rgb_preview']:
            resolve(root, e[key], source_root)
        for spec in ['raw_rgb_npz', 'raw_depth_npz']:
            assert resolve(root, e[spec]['path'], source_root) == initial
        assert e['raw_rgb_npz']['key'] == 'rgb'
        assert e['raw_depth_npz'] == dict(path=e['initial_npz'], key='depth_raw_mm', unit='mm', dtype='int16')
        with np.load(initial, allow_pickle=False) as z:
            state = {k: z[k] for k in z.files}
        for k, a in state.items():
            assert np.array_equal(a, arrays[k][index[cid]]), (cid, k)
            if a.dtype.kind in 'fiu':
                assert np.isfinite(a).all(), (cid, k)
        rgb, depth = state['rgb'], state['depth_raw_mm']
        assert rgb.dtype == np.uint8 and rgb.shape == (128, 128, 3)
        assert depth.dtype == np.int16 and depth.shape == (128, 128)
        assert (depth > 0).any()
        d = e['descriptor']
        assert d == read(root / 'candidates' / cid / 'descriptor.json')
        for k, v in d.items():
            parsed = json.loads(desc[cid][k])
            assert parsed == v, (cid, k)
            assert np.isfinite(np.asarray(v, dtype=float)).all(), (cid, k)
        assert d['peg_xyz'] == state['peg_pose'][:3].tolist()
        assert d['peg_quat_wxyz'] == state['peg_pose'][3:].tolist()
        assert d['peg_half'] == state['peg_half'].tolist()
        assert d['box_pose'] == state['box_pose'].tolist() and d['hole_pose'] == state['hole_pose'].tolist()
        assert np.array_equal(np.array(d['peg_minus_tcp_world'], dtype=np.float32), state['peg_pose'][:3] - state['tcp_pose'][:3])
        assert np.array_equal(np.array(d['peg_minus_hole_world'], dtype=np.float32), state['peg_pose'][:3] - state['hole_pose'][:3])
        assert float(metrics[cid]['boundary_margin_px']) == d['boundary_margin_px']
        assert metrics[cid]['visibility'] == 'VISIBILITY_PASS'
        visibility = read(root / 'candidates' / cid / 'visibility.json')
        assert visibility['requested_pass'] is True
        with np.load(resolve(root, e['gt_visibility_evidence'], source_root), allow_pickle=False) as z:
            evidence = {k: digest_array(z[k]) for k in z.files}
        candidate_rows.append(dict(candidate_id=cid, reset_seed=e['reset_seed'],
            descriptor_sha256=hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(),
            arrays={k: digest_array(a) for k, a in state.items()}, visibility_evidence=evidence,
            visibility='VISIBILITY_PASS', split='UNASSIGNED', initial_npz_sha256=sha(initial)))
        raw_rows.append(dict(candidate_id=cid, raw_rgb=digest_array(rgb), raw_depth=digest_array(depth),
            depth_unit='mm', nonpositive_depth_pixels=int((depth <= 0).sum()),
            raw_npz=str(initial.relative_to(root)), annotated_preview_used_as_raw=False))
    return dict(manifest=dict(status='PASS', run_id=STAGE1_ID, attempted=926, visibility_pass=301,
        eligible=300, source_sha256=source_hash, runtime_sha256=runtime_hash,
        pool_manifest_sha256=sha(root / 'candidate_pool_manifest.json'),
        visibility_protocol=read(root / 'visibility_protocol_identity.json'),
        aggregate_arrays={k: digest_array(a) for k, a in arrays.items()}),
        candidates=dict(status='PASS', count=300, rows=candidate_rows),
        rgbd=dict(status='PASS', count=300, rows=raw_rows))


def protected_usage(root):
    root = str(Path(root).resolve())
    users = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if int(p.name) == os.getpid():
                continue
            cwd = str((p / 'cwd').resolve())
            cmd = (p / 'cmdline').read_bytes()
            fds = [str(f.resolve()) for f in (p / 'fd').iterdir()]
            if cwd == root or cwd.startswith(root + '/') or root.encode() in cmd or any(f == root or f.startswith(root + '/') for f in fds):
                users.append(dict(pid=int(p.name), comm=(p / 'comm').read_text().strip()))
        except (OSError, PermissionError):
            pass
    return users


def system_info(base):
    task10h = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (p / 'cmdline').read_bytes().lower()
            if b'task10h' in cmd or b'handoff10h' in cmd:
                task10h.append(dict(pid=int(p.name), comm=(p / 'comm').read_text().strip(), cwd=str((p / 'cwd').resolve())))
        except OSError:
            pass
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    return dict(base=str(base.resolve()), disk=shutil.disk_usage(base)._asdict(), task10h_processes=task10h,
                gpu=gpu.stdout.strip(), gpu_query_exit=gpu.returncode,
                tmux_sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'], capture_output=True,text=True).stdout.splitlines())


def remote_action(action, audit):
    audit = Path(audit).resolve()
    cfg = read(audit / 'config.json')
    base, canonical, staging = map(Path, [cfg['remote_base'], cfg['remote_root'], cfg['staging_root']])
    assert base.resolve() == Path('/workspace/task10i_stage1_20261031')
    assert canonical == base / 'verified_pool_v2'
    assert staging == base / ('verified_pool_v2_staging_' + cfg['run_id'])
    expected = read(audit / 'source_inventory.json')['files']
    if action == 'inspect':
        actual = inventory(canonical)
        result = dict(root=str(canonical), actual=actual, **compare(expected, actual), system=system_info(base),
                      owner_uid=canonical.stat().st_uid if canonical.exists() else None,
                      in_use=protected_usage(canonical))
        write(audit / 'remote_partial_inventory.json', result)
        # Preserve the original BLOCKED run's entire byte inventory.
        old = inventory(base / 'run_v3')
        write(audit / 'protected_run_v3_inventory.json', old)
        assert read(base / 'run_v3' / 'progress.json')['status'] == 'BLOCKED'
        assert not result['in_use'], result['in_use']
        assert result['system']['disk']['free'] > 2 * sum(f['bytes'] for f in expected.values())
    elif action == 'seed':
        staging.mkdir(exist_ok=True)
        reused = []
        partial = read(audit / 'remote_partial_inventory.json')
        for r in partial['rows']:
            if r['classification'] == 'MATCHED':
                src, dst = canonical / r['path'], staging / r['path']
                assert sha(src) == expected[r['path']]['sha256']
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)  # no shared inode with protected partial
                assert sha(dst) == expected[r['path']]['sha256']
                reused.append(r['path'])
        write(audit / 'reuse_audit.json', dict(reused=len(reused), files=reused, source='SHA-verified prior partial', hardlinks=False))
    elif action in ['hash_staging', 'hash_final']:
        root = staging if action == 'hash_staging' else canonical
        actual = inventory(root)
        result = compare(expected, actual)
        assert result['counts'] == dict(MATCHED=len(expected), MISSING=0, MISMATCHED=0, EXTRA=0, UNREADABLE=0), result['counts']
        write(audit / (action + '.json'), dict(status='PASS', root=str(root), source_inventory_sha256=sha(audit / 'source_inventory.json'), **result))
        with (audit / ('remote_file_hash_audit.csv' if action == 'hash_final' else 'staging_file_hash_audit.csv')).open('w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(result['rows'][0])); w.writeheader(); w.writerows(result['rows'])
    elif action == 'promote':
        archive = base / ('verified_pool_v2_partial_before_' + cfg['run_id'])
        # Recovery after a crash between the two same-filesystem renames.
        if archive.exists() and staging.exists() and not canonical.exists():
            assert read(audit / 'promotion_intent.json')['archive'] == str(archive)
            assert inventory(archive) == read(audit / 'remote_partial_inventory.json')['actual']
            assert inventory(staging) == expected
            os.rename(staging, canonical)
        if canonical.exists() and archive.exists() and not staging.exists():
            assert inventory(canonical) == expected
        else:
            assert read(audit / 'hash_staging.json')['status'] == 'PASS'
            assert inventory(staging) == expected
            assert inventory(canonical) == read(audit / 'remote_partial_inventory.json')['actual']
            assert not protected_usage(canonical), 'Canonical path currently used'
            assert not archive.exists()
            assert base.stat().st_dev == staging.stat().st_dev
            write(audit / 'promotion_intent.json', dict(old=str(canonical), archive=str(archive), staging=str(staging)))
            if canonical.exists():
                os.rename(canonical, archive)
            try:
                os.rename(staging, canonical)
            except BaseException:
                if archive.exists() and not canonical.exists():
                    os.rename(archive, canonical)
                raise
        assert inventory(archive) == read(audit / 'remote_partial_inventory.json')['actual']
        assert inventory(base / 'run_v3') == read(audit / 'protected_run_v3_inventory.json')
        write(audit / 'promotion.json', dict(status='PASS', remote_root=str(canonical), partial_archive=str(archive),
            partial_preserved=True, run_v3_preserved=True, same_filesystem_atomic_renames=True))
    elif action == 'consumer':
        result = consumer(canonical, cfg['source_root'])
        baseline = read(audit / 'source_consumer.json')
        assert result == baseline, 'Consumer numeric/array identity differs from local source'
        for name, key in [('remote_manifest_verification.json','manifest'),('remote_candidate_readback.json','candidates'),('remote_raw_rgbd_readback.json','rgbd')]:
            write(audit / name, result[key])
        sources = read(canonical / 'source_identity.json')['sources']
        bad = [p for p,h in sources.items() if not Path(p).is_file() or sha(p) != h]
        assert not bad, ('external_frozen_source_mismatch', bad)
        write(audit / 'frozen_source_readback.json', dict(status='PASS', verified=len(sources)))
    else:
        raise ValueError(action)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action'); p.add_argument('audit'); a = p.parse_args()
    try:
        remote_action(a.action, a.audit)
        write(Path(a.audit) / (a.action + '.done.json'), dict(status='PASS', exit_code=0))
    except BaseException:
        write(Path(a.audit) / (a.action + '.done.json'), dict(status='BLOCKED', exit_code=1, traceback=traceback.format_exc()))
        raise

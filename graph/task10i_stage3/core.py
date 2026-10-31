"""Immutable inputs, branch ownership and crash-safe episode transactions."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

STAGE2 = Path('/workspace/task10i_stage2_20261031_v2/run')
POOL = Path('/workspace/task10i_stage1_20261031/verified_pool_v2')
RUNTIME_ROOT = Path('/workspace/task10g_f1_abc_20261031/full')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)

def freeze(p, value):
    if Path(p).exists():
        assert read(p) == value, ('immutable identity changed', str(p))
    else: write(p, value)

@contextlib.contextmanager
def lock(p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def verify_receipt(stage2=STAGE2):
    r = read(stage2/'stage3_input_receipt.json')
    assert r['stage2_complete'] and r['run_id'] == '34d10bf652b344a38d54b7672dbfe0c0'
    assert (r['R_train'],r['D_train'],r['validation'],r['new_parent_union']) == (96,96,16,90)
    for n,h in r['files'].items(): assert sha(stage2/n) == h, n
    identity = read(stage2/'selection_identity.json')
    for n,h in identity['files'].items(): assert sha(stage2/n) == h, n
    return r

def allocations(stage2=STAGE2):
    plan = read(stage2/'stage3_demonstration_plan.json')['per_parent_allocations']
    union = read(stage2/'stage3_parent_collection_union.json')['parents']
    byid = {p['id']:p for p in union}
    result = {}
    for a in plan:
        cid=a['new_parent']; p=result.setdefault(cid,dict(candidate=byid[cid], allocations=[], branches=[]))
        p['allocations'].append(a)
        for b in a['branch_templates']:
            if b['branch'] not in p['branches']: p['branches'].append(b['branch'])
    assert len(result)==90
    return result

def branch_spec(branch):
    if branch in ('full_task','nominal'): return dict(kind=branch)
    anchor,x,y = branch[1:].split('_')
    assert int(anchor) in (10,25) and (int(x),int(y)) in [(5,0),(-5,0),(0,5),(0,-5),(10,0),(-10,0),(0,10),(0,-10)]
    return dict(kind='recovery',anchor=int(anchor),offset_mm=[int(x),int(y),0])

def verified(folder):
    folder=Path(folder)
    if not (folder/'receipt.json').exists(): return None
    r=read(folder/'receipt.json')
    for n,h in r['hashes'].items(): assert sha(folder/n)==h, str(folder/n)
    return r

def commit(partial, final, receipt):
    partial=Path(partial);final=Path(final)
    assert not final.exists()
    receipt={**receipt,'hashes':{str(p.relative_to(partial)):sha(p) for p in partial.rglob('*') if p.is_file()}}
    write(partial/'receipt.json',receipt)
    verified(partial)
    os.rename(partial,final)
    fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    return receipt

def begin(folder):
    folder=Path(folder);folder.parent.mkdir(parents=True,exist_ok=True)
    partial=folder.with_name(folder.name+'.partial')
    if partial.exists():
        archive=folder.parent/'interrupted';archive.mkdir(exist_ok=True)
        os.rename(partial,archive/(folder.name+f'.{time.time_ns()}'))
    partial.mkdir()
    return partial

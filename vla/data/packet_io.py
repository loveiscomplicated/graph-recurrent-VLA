"""Lossless Vision packet serialization and content identity."""
from __future__ import annotations
import json
import numpy as np

def save_packet(path, packet):
    """Lossless arrays plus JSON skeleton; no object arrays/pickle."""
    arrays={}
    def walk(v, key='packet'):
        if isinstance(v,np.ndarray):
            arrays[key]=v
            return {'__array__':key}
        if isinstance(v,dict):return {k:walk(x,key+'/'+k) for k,x in v.items()}
        if isinstance(v,list):return [walk(x,key+'/'+str(i)) for i,x in enumerate(v)]
        return v
    skeleton=walk(packet)
    arrays['schema_json']=np.array(json.dumps(skeleton))
    np.savez_compressed(path,**arrays)
    restored=read_packet(path)
    if packet_digest(restored)!=packet_digest(packet):raise AssertionError('packet readback hash mismatch')

def read_packet(path):
    with np.load(path,allow_pickle=False) as a:
        def walk(v):
            if isinstance(v,dict):
                if set(v)=={'__array__'}:return a[v['__array__']].copy()
                return {k:walk(x) for k,x in v.items()}
            if isinstance(v,list):return [walk(x) for x in v]
            return v
        return walk(json.loads(str(a['schema_json'])))

import hashlib
import json
import numpy as np

def packet_digest(packet):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        elif isinstance(value, dict):
            for k in sorted(value):
                digest.update(k.encode()); visit(value[k])
        elif isinstance(value, (list, tuple)):
            for v in value: visit(v)
        else: digest.update(json.dumps(value, sort_keys=True).encode())
    visit(packet)
    return digest.hexdigest()

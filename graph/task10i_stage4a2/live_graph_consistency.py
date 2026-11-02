"""Prove post-inference paired evaluator Graphs equal saved live fingerprints."""
import argparse
from pathlib import Path
import numpy as np
from .frozen_inference import array_digest
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(output,expected,paired,masks,candidate):
    rows=[]
    for entry in read(expected)['entries']:
        eid=str(entry['episode_id']);folder=paired/'episodes'/eid;r=read(folder/'receipt.json')
        assert r['candidate']==candidate and r['frames']==entry['actions']+1
        assert all(sha(folder/p)==h for p,h in r['files'].items())
        upstream=read(masks/'perception'/eid/'upstream_receipts.json')
        for source in upstream:
            condition=source['condition'];path=Path(source['live_graph_fingerprints'])
            assert sha(path)==source['graph_sha256'] and sha(source['receipt'])==source['sha256']
            frames=read(path);assert len(frames)==entry['actions']+1
            for t,frame in enumerate(frames):
                assert frame['timestep']==t
                with np.load(folder/condition/f't{t:04d}'/'graph.npz') as z:
                    assert all(array_digest(z[k])==v for k,v in frame['tensors'].items()),('live/evaluation Graph mismatch',eid,condition,t)
            rows.append(dict(episode=eid,condition=condition,frames=len(frames),all_five_tensors_equal=True))
    write(output,dict(status='LIVE_AND_POSTHOC_GRAPH_BYTE_EQUAL',rows=rows,
        total_condition_frames=sum(r['frames'] for r in rows),candidate=candidate,
        source_sha256=sha(Path(__file__)),expected_sha256=sha(expected)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','expected','paired','masks'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args();run(a.output,a.expected,a.paired,a.masks,a.candidate)

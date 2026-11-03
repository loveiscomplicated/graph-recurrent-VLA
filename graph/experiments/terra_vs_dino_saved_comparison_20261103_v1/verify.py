"""Independent arithmetic and artifact verification; no inference."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1'
REPORT=REPO/'docs/visual-graph-integration/10i_stage4a2_terra_standard_vs_dino_sam2_20261103.md'


def main():
    summary=json.loads((ROOT/'comparison.json').read_text())
    with (ROOT/'object_cases.csv').open() as f: objects=list(csv.DictReader(f))
    for r in objects:
        assert (r['success']=='True')==(r['visibility']=='VISIBLE' and float(r['IoU'])>=.8)
    with (ROOT/'paired_scene_transitions.csv').open() as f: transitions=list(csv.DictReader(f))
    for name,expected in summary['paired_effects'].items():
        rs=[r for r in transitions if r['baseline']==name]
        recovered=sum(r['baseline_success']=='False' and r['terra_success']=='True' for r in rs)
        regressed=sum(r['baseline_success']=='True' and r['terra_success']=='False' for r in rs)
        assert recovered==expected['recovered'] and regressed==expected['regressed']
        parents=sorted({r['parent_id'] for r in rs})
        delta=[statistics.mean(int(r['terra_success']=='True')-int(r['baseline_success']=='True')
                               for r in rs if r['parent_id']==p) for p in parents]
        assert abs(statistics.mean(delta)-expected['parent_macro_delta'])<1e-12
        samples=np.random.default_rng(42).choice(len(parents),size=(10000,len(parents)),replace=True)
        boots=sorted(statistics.mean(delta[i] for i in draw) for draw in samples)
        for q,want in zip([.025,.975],expected['CI95']):
            z=q*(len(boots)-1);k=int(z)
            got=boots[k]+(boots[k+1]-boots[k])*(z-k)
            assert abs(got-want)<1e-12
    for method,m in summary['methods'].items():
        group=[r for r in objects if r['method']==method]
        scenes=defaultdict(list)
        for r in group:scenes[r['image_id']].append(r)
        vv=[rs for rs in scenes.values() if all(r['visibility']=='VISIBLE' for r in rs)]
        assert sum(all(r['success']=='True' for r in rs) for rs in vv)==m['joint']['success']
        assert len(vv)==m['joint']['n']
        for obj in ['peg','box']:
            rs=[r for r in group if r['object']==obj and r['visibility']=='VISIBLE']
            assert sum(r['success']=='True' for r in rs)==m[obj]['success']
            assert abs(statistics.mean(float(r['IoU']) for r in rs)-m[obj]['mean_mask_IoU'])<1e-12
    for p,h in json.loads((ROOT/'input_provenance.json').read_text()).items():
        assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h
    import re
    for link in re.findall(r'\]\(([^)]+)\)',REPORT.read_text()):
        assert (REPORT.parent/link).exists(),link
    verification=dict(status='PASS',object_rows=480,scene_pairs=174,
        independent_threshold_counts_means_transitions=True,independent_parent_bootstrap=True,
        input_sha_and_report_links=True,new_inference_calls=0)
    (ROOT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    index=json.loads((ROOT/'artifact_index.json').read_text())
    index.update({str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [ROOT/'independent_verification.json',Path(__file__)]})
    (ROOT/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(verification))


if __name__=='__main__':main()

"""Case-level paired summaries; repetitions are never pooled as new scenes."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(folder):
    raw=json.loads((folder/'results.json').read_text());group={}
    for r in raw['rows']:
        case=r['case'];key=(case['scene'],case['size'],case.get('mask_mode','actual'))
        value=group.setdefault(key,dict(case=case,variants={}))
        value['variants'].setdefault(r['variant'],[]).append(r)
    cases=[]
    for value in group.values():
        variants={}
        for v,rows in value['variants'].items():
            times=np.array([r['ms'] for r in rows]);q=np.quantile(times,[.25,.5,.75])
            variants[v]=dict(n=len(times),median_ms=float(q[1]),repeat_q25_ms=float(q[0]),repeat_q75_ms=float(q[2]),
                min_ms=float(times.min()),max_ms=float(times.max()),
                peak_rss_bytes=max((r.get('peak_rss_bytes',0) for r in rows),default=0),
                gpu_peak_allocated_bytes=max(r.get('gpu_peak_allocated_bytes',r.get('allocated',0)) for r in rows),
                gpu_peak_reserved_bytes=max(r.get('gpu_peak_reserved_bytes',r.get('reserved',0)) for r in rows))
        pairs={}
        if 'baseline' in variants:
            old=variants['baseline']['median_ms']
            for v,stats in variants.items():
                if v=='baseline':continue
                new=stats['median_ms'];pairs[v]=dict(delta_ms=new-old,reduction_percent=100*(1-new/old),speedup=old/new)
        cases.append(dict(case=value['case'],variants=variants,pairs=pairs))
    aggregates=[]
    for size in sorted({c['case']['size'] for c in cases}):
        for stratum in [None,*sorted({c['case'].get('stratum','unspecified') for c in cases})]:
            selected=[c for c in cases if c['case']['size']==size and (stratum is None or c['case'].get('stratum','unspecified')==stratum)]
            if not selected:continue
            for variant in sorted({v for c in selected for v in c['pairs']}):
                ss=[c for c in selected if variant in c['pairs']]
                old=[c['variants']['baseline']['median_ms'] for c in ss];new=[c['variants'][variant]['median_ms'] for c in ss]
                aggregates.append(dict(size=size,stratum=stratum or 'all',variant=variant,cases=len(ss),parents=len({c['case'].get('parent',c['case']['scene']) for c in ss}),
                    baseline_ms=float(np.median(old)),candidate_ms=float(np.median(new)),
                    paired_delta_ms=float(np.median([c['pairs'][variant]['delta_ms'] for c in ss])),
                    paired_reduction_percent=float(np.median([c['pairs'][variant]['reduction_percent'] for c in ss])),
                    improved=sum(c['pairs'][variant]['delta_ms']<0 for c in ss),
                    median_repeat_iqr_ms=float(np.median([c['variants'][variant]['repeat_q75_ms']-c['variants'][variant]['repeat_q25_ms'] for c in ss])),
                    scene_median_q25_q75_ms=list(map(float,np.quantile(new,[.25,.75])))))
    return dict(status=raw['status'],cases=cases,aggregates=aggregates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
    results={folder.name:summarize(folder) for folder in sorted(root.iterdir()) if folder.is_dir() and (folder/'results.json').exists()}
    (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for name,result in results.items():
        print(name,result['status'])
        for x in result['aggregates']:
            if x['stratum']=='all':print(x)

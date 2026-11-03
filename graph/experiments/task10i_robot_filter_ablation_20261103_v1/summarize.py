"""Summarize paired exploratory strata; retain case-level denominators."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def median(values):return float(statistics.median(values)) if values else None
def mean(values):return float(statistics.mean(values)) if values else None


def main(args):
    data=json.loads(Path(args.scores).read_text());results=[];paired=[]
    for size in (768,1024):
        for group in ('success','failure','ordinary'):
            rr=[x for x in data['rows'] if x['size']==size and x['stratum']==group]
            for mode in ('full','confirmed_only','off'):
                pairs=[(r,next((m for m in r['modes'] if m['mode']==mode and m['status']=='PASS'),None)) for r in rr]
                pairs=[(r,m) for r,m in pairs if m is not None]
                times=[median(m['graph_ms']) for _,m in pairs];nodes=[n for _,m in pairs for n in m['nodes']]
                acc=Counter();gate=Counter();empty=Counter();cover={0:[],1:[]};precision={0:[],1:[]};near_raw=0;near_clean=0;correct_raw=0;correct_clean=0
                for r,m in pairs:
                    gate['accepted' if m['gate_accepted'] else 'rejected']+=1
                    for n in m['nodes']:
                        for k in ('nodes','correct','robot','background','other_object','attached_sources','attached_correct','attached_robot','correct_near_finger'):
                            acc[k]+=n[k]
                        if n['precision'] is None:empty[n['object_id']]+=1
                        else:precision[n['object_id']].append(n['precision'])
                        if n['visible_coverage_4mm'] is not None:cover[n['object_id']].append(n['visible_coverage_4mm'])
                    for raw in r['raw_masks']:
                        cleaned=next(x for x in m['masks'] if x['stage']=='clean' and x['camera']==raw['camera'] and x['object_id']==raw['object_id'])
                        correct_raw+=raw['correct'];correct_clean+=cleaned['correct']
                        near_raw+=raw['correct_near_finger'];near_clean+=cleaned['correct_near_finger']
                results.append(dict(size=size,stratum=group,mode=mode,n=len(pairs),parents=len({r['parent'] for r,_ in pairs}),
                    median_case_median_ms=median(times),case_ms_min=min(times) if times else None,case_ms_max=max(times) if times else None,
                    node_totals=dict(acc),robot_nodes_per_all=acc['robot']/acc['nodes'] if acc['nodes'] else None,
                    cases_with_robot_node=sum(any(n['robot'] for n in m['nodes']) for _,m in pairs),
                    correct_node_precision_peg=mean(precision[0]),correct_node_precision_box=mean(precision[1]),
                    peg_coverage_4mm_mean=mean(cover[0]),box_coverage_4mm_mean=mean(cover[1]),
                    empty_peg_cases=empty[0],empty_box_cases=empty[1],wrist_gate=dict(gate),
                    raw_correct_pixels=correct_raw,clean_correct_pixels=correct_clean,
                    correct_pixel_retention=correct_clean/correct_raw if correct_raw else None,
                    near_raw_correct_pixels=near_raw,near_clean_correct_pixels=near_clean,
                    near_correct_pixel_retention=near_clean/near_raw if near_raw else None,
                    peak_process_rss_bytes=max((m['peak_rss_bytes'] for _,m in pairs),default=None),
                    peak_cuda_allocated_bytes=max((m['gpu_peak_allocated_bytes'] for _,m in pairs),default=None)))
            for mode in ('confirmed_only','off'):
                comparisons=[]
                for r in rr:
                    available={m['mode']:m for m in r['modes'] if m['status']=='PASS'}
                    if 'full' not in available or mode not in available:continue
                    a,b=available['full'],available[mode]
                    def total(m,key):return sum(n[key] for n in m['nodes'])
                    item=dict(scene=r['scene'],parent=r['parent'],latency_delta_ms=median(b['graph_ms'])-median(a['graph_ms']),
                        robot_node_delta=total(b,'robot')-total(a,'robot'),correct_node_delta=total(b,'correct')-total(a,'correct'),
                        correct_near_node_delta=total(b,'correct_near_finger')-total(a,'correct_near_finger'),
                        sources_changed=a['selected_sources']!=b['selected_sources'],gate_before=a['gate_accepted'],gate_after=b['gate_accepted'])
                    for oid in (0,1):
                        va=a['nodes'][oid]['visible_coverage_4mm'];vb=b['nodes'][oid]['visible_coverage_4mm']
                        item[f'coverage_delta_{oid}']=None if va is None or vb is None else vb-va
                    comparisons.append(item)
                paired.append(dict(size=size,stratum=group,mode=mode,n=len(comparisons),
                    median_latency_delta_ms=median([x['latency_delta_ms'] for x in comparisons]),
                    cases_robot_nodes_increase=sum(x['robot_node_delta']>0 for x in comparisons),
                    cases_robot_nodes_decrease=sum(x['robot_node_delta']<0 for x in comparisons),
                    cases_correct_nodes_increase=sum(x['correct_node_delta']>0 for x in comparisons),
                    cases_correct_nodes_decrease=sum(x['correct_node_delta']<0 for x in comparisons),
                    cases_near_nodes_increase=sum(x['correct_near_node_delta']>0 for x in comparisons),
                    cases_near_nodes_decrease=sum(x['correct_near_node_delta']<0 for x in comparisons),
                    cases_selected_sources_changed=sum(x['sources_changed'] for x in comparisons),
                    gate_accepted_to_rejected=sum(x['gate_before'] and not x['gate_after'] for x in comparisons),
                    gate_rejected_to_accepted=sum(not x['gate_before'] and x['gate_after'] for x in comparisons),
                    median_peg_coverage_delta=median([x['coverage_delta_0'] for x in comparisons if x['coverage_delta_0'] is not None]),
                    median_box_coverage_delta=median([x['coverage_delta_1'] for x in comparisons if x['coverage_delta_1'] is not None]),
                    cases=comparisons))
    output=dict(status=data['status'],scope='Selected-case diagnostics, not prevalence or population estimate',
        score_failures=data['failures'],groups=results,paired=paired)
    Path(args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for g in results:
        if g['n']:
            print(g['size'],g['stratum'],g['mode'],'n',g['n'],'ms',round(g['median_case_median_ms'],1),
                'robotnodes',f"{g['node_totals']['robot']}/{g['node_totals']['nodes']}",
                'nearpixels',f"{g['near_clean_correct_pixels']}/{g['near_raw_correct_pixels']}",
                'pegcoverage',None if g['peg_coverage_4mm_mean'] is None else round(g['peg_coverage_4mm_mean'],3))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--output',required=True);main(p.parse_args())

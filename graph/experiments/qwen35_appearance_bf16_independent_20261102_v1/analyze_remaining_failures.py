"""Posthoc failure accounting from frozen outputs; no model or prompt changes."""
from pathlib import Path
import json,csv,collections
ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1'
def rows(name):
 with (R/name).open() as f:return list(csv.DictReader(f))
candidates=rows('candidate_evaluation.csv');selected=rows('selected_boxes.csv');metrics=json.loads((R/'evaluation_metrics.json').read_text())
look={(r['method'],r['image_id'],r['object']):r for r in selected}
partitions={};regressions=[];per_parent={}
for obj in ['peg','box']:
 for condition,method in [('original_three','original_query_gate'),('appearance_three','appearance_query_gate')]:
  counts=collections.Counter();failures=[]
  for r in selected:
   if r['method']!=method or r['object']!=obj or r['visibility']!='VISIBLE':continue
   pool=[c for c in candidates if c['condition']==condition and c['image_id']==r['image_id'] and c['query_object']==obj];good=[c for c in pool if c['target_good']=='True'];approved=[c for c in good if c['accepted_for_query']=='True']
   reason='SUCCESS' if r['correct']=='True' else 'NO_CORRECT_DINO_PROPOSAL' if not good else 'ALL_GOOD_PROPOSALS_REJECTED' if not approved else 'CORRECT_APPROVED_BUT_OUTRANKED_BY_IOU_FAIL'
   counts[reason]+=1
   if reason!='SUCCESS':failures.append(dict(image_id=r['image_id'],parent_id=r['parent_id'],camera=r['camera'],frame_kind=r['frame_kind'],reason=reason,selected_candidate=r['candidate_id'],null=r['null']=='True',selected_GT_proxy_role=r['GT_role'],selected_IoU=float(r['bbox_IoU']),good_proposal_labels=dict(collections.Counter(c['strict_label'] for c in good)),selected_actor_pixels=json.loads(r['actor_pixels'])))
  partitions[condition+'/'+obj]=dict(counts=dict(counts),failures=failures)
for r in selected:
 if r['method']!='raw_top1' or r['visibility']!='VISIBLE' or r['correct']!='True':continue
 new=look[('appearance_query_gate',r['image_id'],r['object'])]
 if new['correct']=='True':continue
 c=next(c for c in candidates if c['id']==r['candidate_id'] and c['condition']=='appearance_three')
 regressions.append(dict(object=r['object'],image_id=r['image_id'],parent_id=r['parent_id'],camera=r['camera'],frame_kind=r['frame_kind'],old_candidate=r['candidate_id'],old_IoU=float(r['bbox_IoU']),appearance_label_on_old_correct_candidate=c['strict_label'],GT_proxy_role_on_old=c['GT_role'],new_candidate=new['candidate_id'],new_IoU=float(new['bbox_IoU']),new_null=new['null']=='True'))
for baseline,objects in metrics['paired_effects'].items():
 per_parent[baseline]={}
 for obj,z in objects.items():
  changes=[p['recall_delta'] for p in z['per_parent']];per_parent[baseline][obj]=dict(improved=sum(x>0 for x in changes),unchanged=sum(x==0 for x in changes),regressed=sum(x<0 for x in changes),parents=len(changes))
result=dict(purpose='Posthoc accounting, not new inference or causal inspection of model internals',partitions=partitions,raw_top1_regressions=regressions,parent_directions=per_parent)
(R/'remaining_failure_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(partitions={k:z['counts'] for k,z in partitions.items()},regressions=regressions,parent_directions=per_parent),indent=2))

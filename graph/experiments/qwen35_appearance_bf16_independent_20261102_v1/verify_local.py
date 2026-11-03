"""Read-only independent verification after receiving the frozen remote run."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1'
REPORT=ROOT/'docs/visual-graph-integration/10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(path):
 q=Path(path);assert q.parts[1:3]==('workspace','task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1');return R/Path(*q.parts[3:])
def csvread(p):
 with p.open() as f:return list(csv.DictReader(f))
def bbox(mask):
 ys,xs=np.nonzero(mask);return [xs.min(),ys.min(),xs.max()+1,ys.max()+1] if len(xs) else None
def iou(a,b):
 if a is None or b is None:return 0.
 overlap=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))));union=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-overlap);return overlap/union if union else 0.
idx=rd(R/'artifact_index.json')['files'];assert all(sha(R/n)==e['sha256'] for n,e in idx.items())
v=rd(R/'verification.json');assert v['stage']=='COMPLETE_VERIFIED' and v['model_loads']==v['dino_loads']==1
assert v['all_protected_SHA_unchanged'] and v['environments_unchanged'] and v['model_processor_unchanged']
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_GT']
source_checks=0
for name,expected in v['source_before'].items():
 assert sha(Path(__file__).parent/name)==expected==sha(R/'.execution'/name);source_checks+=1
prompts=rd(R/'prompt_identity.json')['conditions'];old=rd(R.parent/'qwen35_perception_diagnostic_20261102_v1/prompt_identity.json')['conditions']
assert all(prompts[k]==old[k] for k in prompts)
sel=rd(R/'selection_manifest.json');cm=rd(R/'.execution/candidate_manifest.json');images={c['id']:c for c in cm['images']};cmap={c['id']:c for c in cm['candidates']}
assert len(sel['planned_parent_ids'])==20 and len(cm['parents'])==15 and len(sel['unavailable'])==5
assert not set(cm['parents'])&set(sel['development_excluded_parent_ids'])
assert len(images)==60 and len({(c['parent_id'],c['frame_kind'],c['camera']) for c in images.values()})==60
reqs=rd(R/'request_manifest.json')['requests'];reqmap={(r['candidate_id'],r['condition']):r for r in reqs};imagechecks=0;cropchecks=0
for r in reqs:
 for p,h in zip(r['images'],r['image_sha256']):assert sha(local(p))==h;imagechecks+=1
for c in cmap.values():
 a=reqmap[(c['id'],'original_three')];b=reqmap[(c['id'],'appearance_three')];assert a['images']==b['images'] and a['image_sha256']==b['image_sha256']
 original=Image.open(local(a['images'][0])).convert('RGB');crop=Image.open(local(a['images'][-1])).convert('RGB');assert np.array_equal(np.array(crop),np.array(original.crop(c['crop_box'])));cropchecks+=1
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r for r in raw}
assert len(raw)==len(pred)==len(reqs)==len(cmap)*2
for r in raw:
 assert (r['raw_text'].strip() if r['raw_text'].strip() in prompts[r['condition']]['labels'] else 'INVALID_OUTPUT')==r['strict_label']
 assert r['timing']['initial_forward_cache']['cache_length']==0
masks={};gtboxes={};xx,yy=np.meshgrid(np.arange(512),np.arange(512));gtiou=0;gtrole=0
for im in images.values():
 assert sha(local(im['GT_path']))==im['GT_sha256']
 with np.load(local(im['GT_path'])) as z:seg=z['actor_segmentation']
 ms={k:np.isin(seg,im['object_ids'][k]) for k in ['peg','box','robot']};ms['background']=~(ms['peg']|ms['box']|ms['robot']);masks[im['id']]=ms;gtboxes[im['id']]={k:bbox(ms[k]) for k in ['peg','box']}
for r in csvread(R/'candidate_evaluation.csv'):
 box=json.loads(r['box']);region=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3]);counts={k:int(m[region].sum()) for k,m in masks[r['image_id']].items()};assert counts==json.loads(r['actor_pixels']);fg=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);role=dom.upper() if fg>=16 and counts[dom]>=16 and counts[dom]/fg>=.7 else 'OTHER' if fg<16 else 'AMBIGUOUS';assert role==r['GT_role'];gtrole+=1
 for obj in ['peg','box']:assert abs(iou(box,gtboxes[r['image_id']][obj])-float(r[obj+'_IoU']))<1e-12;gtiou+=1
selected=csvread(R/'selected_boxes.csv');replayed=0;selectediou=0
for r in selected:
 if r['method'] in ['original_query_gate','appearance_query_gate','raw_top1']:
  cond='original_three' if r['method']=='original_query_gate' else 'appearance_three';pool=[c for c in cmap.values() if c['image_id']==r['image_id'] and c['query_object']==r['object'] and (r['method']=='raw_top1' or pred[(c['id'],cond)]['strict_label']==r['object'].upper())];pool.sort(key=lambda c:(-c['DINO_score'],c['rank']));assert r['candidate_id']==(pool[0]['id'] if pool else '');replayed+=1
 if r['bbox_IoU']:
  box=json.loads(r['box']) if r['box'] else None;ov=iou(box,gtboxes[r['image_id']][r['object']]);assert abs(ov-float(r['bbox_IoU']))<1e-12;selectediou+=1
metrics=rd(R/'evaluation_metrics.json');metric_checks=0
for method in metrics['methods']:
 for obj in ['peg','box']:
  rs=[r for r in selected if r['method']==method and r['object']==obj and r['visibility']=='VISIBLE'];z=metrics['methods'][method]['image_object_level'][obj]['recall'];assert z['denominator']==len(rs) and z['numerator']==sum(r['correct']=='True' for r in rs);metric_checks+=1
result=dict(status='PASS',remote_indexed_artifacts=len(idx),source_files=source_checks,frozen_prompts_unchanged=True,planned_parents=20,evaluated_parents=15,missing_restoration_parents=5,camera_frames=60,candidates=len(cmap),requests=len(raw),request_image_hashes=imagechecks,pixel_equal_crops=cropchecks,GT_role_rows=gtrole,independent_candidate_IoUs=gtiou,independent_selected_IoUs=selectediou,GT_free_selection_replays=replayed,recall_metric_checks=metric_checks,all_initial_caches_empty=True,all_original_remote_artifacts_preserved=True)
(R/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
report=(R/'10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md').read_text()
import re
prefix='../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/'
report=re.sub(r'(!?\[[^\]]*\]\()([^):]+)(\))',lambda m:m[1]+prefix+m[2]+m[3],report)
report+='\n로컬 수신 독립 검증: Artifact '+str(len(idx))+'개 SHA, 원격/로컬 실행 Source '+str(source_checks)+'개, 후보 Crop '+str(cropchecks)+'개 pixel-equal, Actor 역할 '+str(gtrole)+'행, 후보 IoU '+str(gtiou)+'개, 선택 IoU '+str(selectediou)+'개 재계산을 통과했다. [상세 검증]('+prefix+'local_verification.json).\n'
report+='\n실행 전 준비 스크립트 requests.py가 SAPIEN의 requests 라이브러리 import를 가리는 오류가 있어 build_requests.py로 이름을 바꿨다. 첫 Capture와 모델 Load 전에 발견해 교정했고 원래 Source와 Log를 보존했다. 입력·Prompt·선정 규칙은 바뀌지 않았다.\n'
report+='\n실행 전 메타데이터 점검에서 복원 가능 Parent 수를14개로 가정한 Assertion이 중단됐다. 전체 필드 검사로15개 가능/5개 부재임을 확인해 모델 Load 전에 수만 교정했고, 이전 Source/Log를 preflight_revisions에 보존했다. 선정 기준이나 Prompt를 결과에 따라 바꾸지 않았으며 GPU 추론 재실행은 없었다.\n'
REPORT.write_text(report)
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report);assert all((REPORT.parent/l).resolve().exists() for l in links if not l.startswith('http'))
(R/'local_completion_index.json').write_text(json.dumps(dict(status='COMPLETE_VERIFIED',report_path=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),local_verification_sha256=sha(R/'local_verification.json'),verifier_source_sha256=sha(Path(__file__)),remote_index_sha256=sha(R/'artifact_index.json')),indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Add explicit posthoc interpretation without changing any frozen remote artifact."""
from pathlib import Path
import json,csv,hashlib,re,collections
ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1'
REPORT=ROOT/'docs/visual-graph-integration/10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return json.loads(p.read_text())
m=rd(R/'evaluation_metrics.json');a=rd(R/'remaining_failure_analysis.json')
with (R/'selected_boxes.csv').open() as f:rs=list(csv.DictReader(f))
cross={}
for method in ['raw_top1','original_query_gate','appearance_query_gate']:
 cross[method]={}
 for cam in ['base_camera','hand_camera']:
  for time in ['F0','F1']:
   cross[method][cam+'/'+time]={}
   for obj in ['peg','box']:
    z=[r for r in rs if r['method']==method and r['camera']==cam and r['frame_kind']==time and r['object']==obj and r['visibility']=='VISIBLE'];cross[method][cam+'/'+time][obj]=dict(numerator=sum(r['correct']=='True' for r in z),denominator=len(z))
(R/'camera_time_cross_analysis.json').write_text(json.dumps(dict(purpose='Posthoc descriptive cross-stratification; no additional inference',methods=cross),indent=2)+'\n')
prefix='../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/'
lead='''
**최종 해석: 기존 역할 Prompt 대비 개선은 재현됐지만, 일반적으로 안정적인 Peg/Box 선택기로 확인된 것은 아니다.** 외형 Prompt의 Peg Recall61.7%는 DINO 원시 Top1의65.0%보다 낮고, 후기 Main Peg는 **0/15**다. 기존 Prompt 대비 Peg Recall 차이의 Parent-cluster95% CI는0을 포함한다. BOX는 같은 비교에서 양의 CI를 보였다. 위 상태명 `APPEARANCE_PROMPT_HELDOUT_GAIN`은 사전 규칙에 따른 두 Object의 점추정 개선을 뜻하며, 통계적 보장·모든 Camera/시점 개선·Production 적합성의 판정이 아니다.
'''
text=REPORT.read_text();needle='## 1. 사전 고정 범위와 자료의 독립성';assert '**최종 해석:' not in text;text=text.replace(needle,lead+'\n'+needle,1)
text=text.replace('자료 부재 Parent: existing_30040, existing_30124, existing_30082, existing_30017, existing_30032.','자료 부재 Parent: existing_30040, existing_30124, existing_30082, existing_30017, existing_30032.\n\n제외된5개는 모두 기존 `existing_full_task` 자료이고, 평가된15개는 복원 필드가 있는 Stage3 Parent다. 이 자료 가용성 차이로 평가 범위가 제한됐다. 복원 불가능한 Source군까지 같은 성능으로 일반화하지 않는다. 실제 F1은0134만t78이고 나머지는t80이다.')
lines=['## 5.1 사후 Camera × 시점 교차 분해','',
'아래는 완료된 같은 예측의 사후 기술 통계다. 이 분해 결과로 Prompt·대상·선택 정책을 수정하지 않았다.','',
'| Camera / 시점 | 원시 Top1 PEG | 기존 Prompt PEG | 외형 Prompt PEG | 원시 Top1 BOX | 기존 Prompt BOX | 외형 Prompt BOX |','|---|---:|---:|---:|---:|---:|---:|']
for cam in ['base_camera','hand_camera']:
 for time in ['F0','F1']:
  vals=[]
  for obj in ['peg','box']:
   for method in cross:
    z=cross[method][cam+'/'+time][obj];vals.append(f"{z['numerator']}/{z['denominator']}")
  lines.append('| '+cam+' / '+time+' | '+' | '.join(vals)+' |')
lines+=['',
'외형 Prompt는 후기 Wrist PEG를0/15→8/15, BOX를0/15→7/15로 회복했지만, 후기 Main PEG는5/15→0/15로 퇴행했다. 전체 평균의 상승과 특정 조건의 집중 실패가 함께 존재한다. 초기에는 PEG29/30·BOX26/28로 상대적으로 높았지만 후기 재발견에 같은 성능을 가정할 수 없다.','',
'## 5.2 남은 실패의 사후 회계와 관측·가설','',
'각 가시 Object 실패를 같은 Query 후보 풀에서 서로 배타적으로 분해했다. 정답은Box IoU≥0.5다.','',
'| Prompt / 물체 | 성공 | 정답 DINO 후보 없음 | 정답 후보 전부 거부 | 정답도 승인했지만 IoU 실패 후보가 더 높은 점수로 선택됨 |','|---|---:|---:|---:|---:|']
for key,z in a['partitions'].items():
 c=z['counts'];lines.append('| '+key+' | '+' | '.join(str(c.get(k,0)) for k in ['SUCCESS','NO_CORRECT_DINO_PROPOSAL','ALL_GOOD_PROPOSALS_REJECTED','CORRECT_APPROVED_BUT_OUTRANKED_BY_IOU_FAIL'])+' |')
lines+=['',
'외형 Prompt의 Peg 실패23개 중15개는 정답 후보를 모두 거부했고,6개는 승인된 정답 후보보다 IoU<0.5 후보를 위에 놓은 DINO Score 순위를 그대로 따랐다. 정답 후보 자체가 없는 경우는2개였다. BOX 실패13개는 후보 없음4개, 정답 거부1개, 승인 후보 간 순위 실패8개였다. 이 숫자는 저장된 Label·좌표·Score로 확인한 선택 단계의 원인 분해이며 모델 내부 기전에 대한 인과 증명은 아니다.','',
'Peg Query의 정답 후보 유지율은50.4%→73.3%로 올랐지만, IoU<0.5 후보 승인도7.8%→27.6%로 늘었다. 실제 선택된 Peg의 IoU 실패7개는 Actor 구성상PEG6개·AMBIGUOUS1개였다. 단순 Wrong-object 판정과 Box 범위 실패가 다르다. Box Query의 오답 후보 승인은65.7%→22.4%로 줄었고, 최종 BOX Wrong-object도19/58→1/57로 줄었다.','',
'GT 역할이 명확한 후보의 분류 일치율은377/736→633/736(51.2%→86.0%), ROBOT→BOX는160/291→31/291(55.0%→10.7%)이었다. 개선됐지만 이 오류가 사라지지는 않았다. 이 후보 풀에는 OTHER 정답 사례가 없고, 두 Prompt 모두 OTHER/UNCERTAIN 출력이0이었다. PEG 미가시 이미지도0개다. BOX 미가시는2개뿐이며 외형 Prompt의 Hallucination은0/2였다. 이 작은/없는 분모로 일반적인 부재 인식·불확실성 보정을 검증했다고 주장하지 않는다.','',
'### DINO 원시1위 대비 Peg 회귀4개','',
'| Parent / 후기 Main | 원시1위 Peg IoU | 외형 Prompt가 그 후보에 붙인 Label | 최종 출력 |','|---|---:|---|---|']
for r in a['raw_top1_regressions']:lines.append(f"| {r['parent_id']} / F1 Main | {r['old_IoU']:.3f} | {r['appearance_label_on_old_correct_candidate']} | {'Null' if r['new_null'] else r['new_candidate']} |")
lines+=['',
'네 사례 모두 정답 위치의 후보를 BOX로 판단해 최종 PEG가 Null이 됐다. 0193/0327은 Gripper가 섞인 AMBIGUOUS 역할 Proxy이지만, GT Peg Box와의 IoU는0.81/0.93으로 위치 평가상 정답이다. 혼합 영역이라는 이유로 이 Detection Miss를 분모에서 제외하지 않았다.','',
'직접 확인한0767/0924의 Crop에서는 흰색·회색 Peg 부분이 두드러지고 빨간 면은 보이지 않았다. 이는 관측이다. 고정된 red-and-white 외형 설명의 일부가 가림·회전으로 보이지 않을 때 분류가 흔들렸을 가능성은 가설이며, 해당 문구만 바꾼 대응 실험이나 내부 특징 분석은 하지 않았다. 색상 설명이 모든 시점의 정체성 연결을 해결했다고 볼 수 없다.','',
'기존 Prompt 대비 Parent별 Recall 방향은 PEG 개선7/동일6/퇴행2, BOX 개선10/동일5/퇴행0이다. 대응 이미지 단위로는 PEG11개 회복·5개 회귀, BOX12개 회복·0개 회귀였다. PEG의 Parent-macro +10.0%p에 대한 탐색적95% CI는[0.0,+20.0]%p, BOX의 +20.6%p는[+12.2,+28.9]%p다. BOX의 이득은 이 코호트에서 더 일관됐고, PEG 개선의 강도는 불확실하다.','',
'[실패 분해 원자료]('+prefix+'remaining_failure_analysis.json), [Camera/시점 교차 지표]('+prefix+'camera_time_cross_analysis.json).','']
text=text.replace('## 6. 비용과 검증','\n'.join(lines)+'\n## 6. 비용과 검증',1)
REPORT.write_text(text)
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text);assert all((REPORT.parent/l).resolve().exists() for l in links if not l.startswith('http'))
interpretation=dict(status='PARTIAL_GENERALIZATION_WITH_LATE_MAIN_PEG_FAILURE',role_accuracy_improved=True,box_recall_improvement_consistent_in_this_cohort=True,peg_general_recall_gain_not_established=True,late_main_peg_success=0,late_main_peg_denominator=15,production_ready=False,SAM2_evaluated=False,no_additional_inference=True)
(R/'local_result_interpretation.json').write_text(json.dumps(interpretation,indent=2)+'\n')
idx=rd(R/'local_completion_index.json');idx['report_sha256']=sha(REPORT);idx['posthoc_artifacts']={n:sha(R/n) for n in ['remaining_failure_analysis.json','camera_time_cross_analysis.json','local_result_interpretation.json']};idx['local_source_hashes']={p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()};idx['all_remote_artifacts_still_unchanged']=all(sha(R/n)==e['sha256'] for n,e in rd(R/'artifact_index.json')['files'].items());assert idx['all_remote_artifacts_still_unchanged'];(R/'local_completion_index.json').write_text(json.dumps(idx,indent=2)+'\n')
print(json.dumps(interpretation,indent=2));print('Report links and all2279 frozen remote artifacts PASS')

"""Postprocessing only for C2. Box-uncontrolled summaries explicitly exploratory."""
import html,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .learning_signal_audit import O,R,ROOT,read,put,sha,LEVELS
from utils.experiment_artifacts import register_existing_output

def table(df):
 return df.to_markdown(index=False,floatfmt='.3f') if False else '```text\n'+df.to_string(index=False,float_format=lambda v:f'{v:.3f}')+'\n```'
def main():
 p=pd.read_csv(O/'robot_state_matching.csv');e=pd.read_csv(O/'expert_conditional_action.csv');m=pd.read_csv(O/'model_conditional_action.csv');f=pd.read_csv(O/'frame_geometry_quality.csv');cov=pd.read_csv(O/'geometry_pair_coverage.csv');run=read(O/'run_identity.json')['uuid'];exp=O/'experiments'/run
 # This relaxation is descriptive only, after empty primary result, never a replacement primary.
 # CSV parser treats missing bins as NaN; preserve them explicitly in summaries.
 p['bin']=p['bin'].fillna('missing')
 cov['bin_missing']=0
 for idx,row in cov.iterrows():
  z=p[(p.split==row.split)&(p.level==row.level)]
  if row.scope!='all':z=z[z[row.scope]]
  if row.pair_class!='all':z=z[z.pair_class==row.pair_class]
  cov.loc[idx,'bin_missing']=int((z['bin']=='missing').sum())
 cov.to_csv(O/'geometry_pair_coverage.csv',index=False)
 p['robot_only_blocks']=False
 for (split,level),q in p.groupby(['split','level']):
  blocks=set();epairs=set()
  for idx,row in q.sort_values(['score','i','j']).iterrows():
   if not row.peg_quality:continue
   a,b=(int(row.episode_a),int(row.t_a)//10),(int(row.episode_b),int(row.t_b)//10);epair=tuple(sorted((a[0],b[0])))
   if a not in blocks and b not in blocks and epair not in epairs:p.loc[idx,'robot_only_blocks']=True;blocks.update([a,b]);epairs.add(epair)
 p.to_csv(O/'exploratory_robot_only_pairs.csv',index=False)
 ids=set(p[p.robot_only_blocks].pair_id);mm=m[m.pair_id.isin(ids)].copy();ee=e[e.pair_id.isin(ids)].copy();mm.to_csv(O/'exploratory_robot_only_response.csv',index=False)
 stat=[]
 for keys,g in mm.groupby(['split','level','pair_class','model']):
  stat.append(dict(zip(['split','level','pair_class','model'],keys),n=len(g),episodes=len(set(g.episode_a)|set(g.episode_b)),small_expert=int(g.small_expert_delta.sum()),**{k:g[k].mean() for k in ['expert_delta_mm','model_delta_mm','delta_error_mm','direction_cosine','signed_gain','magnitude_ratio','zero_improvement_mm','better_than_zero','peg_action_cosine']}))
 stats=pd.DataFrame(stat);stats.to_csv(O/'exploratory_robot_only_summary.csv',index=False)
 # Episode macro descriptives: each pair contributes to each of its two episode clusters.
 expanded=pd.concat([mm.assign(cluster_episode=mm.episode_a),mm.assign(cluster_episode=mm.episode_b)])
 macro=expanded.groupby(['split','level','pair_class','model','cluster_episode'])[['expert_delta_mm','model_delta_mm','delta_error_mm','zero_improvement_mm']].mean().reset_index();macro.to_csv(O/'exploratory_episode_macro.csv',index=False)
 # Coverage bars retain missing/poor candidates as an explicit category.
 fig,ax=plt.subplots(1,3,figsize=(16,4.7));q=p[(p.split=='train')&(p.level=='primary')]
 counts=q.groupby('bin').size().reindex(['0-5','5-10','10-20','20-40','40+','missing'],fill_value=0);ax[0].bar(counts.index,counts.values,color='#597a9c');ax[0].set(title='Train96 robot-matched candidates (primary)',xlabel='Observed Peg offset bin (mm; missing=no centroid)',ylabel='Pairs (dependent)')
 qq=e[(e.split=='train')&(e.level=='primary')&(e.peg_quality)]
 ax[1].scatter(qq.peg_offset_mm,qq.expert_delta_mm,c='#d08331',label='All Peg-quality pairs, Box uncontrolled');ax[1].set(xlabel='Observed Peg offset (mm)',ylabel='Expert command difference (mm)',title='Observational action differences');ax[1].legend(fontsize=7)
 zz=stats[(stats.split=='train')&(stats.level=='primary')&(stats.pair_class=='different')]
 if len(zz):
  z=zz.iloc[0];ax[2].bar(['Expert','E0','E1'],[z.expert_delta_mm,*[zz[zz.model==v].model_delta_mm.iloc[0] for v in ['E0','E1']]],color=['#d08331','#4576a0','#68a28a'])
 ax[2].set(title='Exploratory block subset, Box uncontrolled',ylabel='Mean command difference (mm)');fig.suptitle('Primary Box-controlled eligible pairs: 0; no causal Peg effect inferred',fontsize=12);fig.tight_layout();fig.savefig(O/'audit_figures.png',dpi=160);plt.close(fig)
 # Real examples retain every primary-quality pair, not only model failures.
 examples=m[(m.split=='train')&(m.level=='primary')&(m.pair_class=='different')].merge(p[['pair_id','box_mm','q_rad','qvel','tcp_mm','angle_deg','robot_only_blocks']],on='pair_id');examples.to_csv(O/'primary_observational_examples.csv',index=False)
 def compact(split):
  rr=[]
  for level in LEVELS:
   z=p[(p.split==split)&(p.level==level)];qual=z[z.peg_quality];rr.append(dict(level=level,robot_pairs=len(z),episodes=len(set(z.episode_a)|set(z.episode_b)),peg_quality=len(qual),control=int((qual.pair_class=='control').sum()),uncertain=int((qual.pair_class=='uncertain').sum()),different=int((qual.pair_class=='different').sum()),box_controlled=int(z.box_controlled.sum()),selected=int(z.selected.sum()),exploratory_blocks=int(z.robot_only_blocks.sum())))
  return pd.DataFrame(rr)
 ct=compact('train');cv=compact('validation');quality=f.groupby('split').agg(frames=('episode','size'),episodes=('episode','nunique'),peg_quality=('peg_quality','sum'),box_quality=('box_quality','sum')).reset_index()
 desc=p[(p.split=='train')&(p.level=='primary')][['q_rad','qvel','tcp_mm','angle_deg','opening_mm','gripvel','linear','angular','box_mm','peg_offset_mm']].describe().loc[['count','mean','50%','max']].reset_index()
 show=stats[stats.split=='train'][['level','pair_class','model','n','episodes','small_expert','expert_delta_mm','model_delta_mm','delta_error_mm','direction_cosine','signed_gain','zero_improvement_mm']]
 ex=examples[['episode_a','t_a','episode_b','t_b','model','peg_offset_mm','box_mm','expert_delta_mm','model_delta_mm','delta_error_mm','direction_cosine','signed_gain','robot_only_blocks']]
 primary=stats[(stats.split=='train')&(stats.level=='primary')&(stats.pair_class=='different')]
 txt=f'''# Task 10-C2 — Geometry-conditioned Learning Signal Audit

ExperimentLogger UUID `{run}`. Parent BC `3788798a719f4ec4a66c224ca42f5d63`.

**주 결과: 고정한 Robot·과거 이력·Peg 품질·Box 조건을 모두 만족하는 cross-episode Pair는 Train96/Validation12 모두 0개다. H1과 H2의 상대적 설명력을 분리할 근거는 INCONCLUSIVE다.** Robot-only 매칭의 제한된 사례는 보존하지만 Peg의 독립적인 인과 효과로 해석하지 않는다.

## 실행 계약과 기준 동결

Train96 11,975 Graph, Validation12 1,477 Graph. Test12를 열지 않았다. 원본 HDF/캐시/체크포인트 변경, 새 BC·Probe 학습, 모델 forward, DINO/SAM2, simulator 재생성은 없다. 공통 ExperimentLogger와 기존 ActionCodec(Method B)/controller_commands를 재사용했다. 저장된 Native prediction에 codec을 다시 decode하지 않았다.

`train_matching_freeze.json`은 Action 파일과 Validation을 읽기 전에 Train matching CSV/protocol SHA를 고정했다. 기하 preflight만 읽어 support/offset 기준을 정했고 모든 threshold 결과를 보고한다. Train의 인접 중심 변화 중앙값 3.67mm, p95 14.29mm, p99 168.62mm는 실제 이동·회전·가시성 변화가 섞여 **센서 noise 추정치가 아니다**. 정확한 GT pose/관측 source purity가 없어 20mm도 실제 Peg 이동의 보증이 아니다.

관측 Graph XYZ는 이미 TCP inverse pose로 변환된 좌표다. 동일 Graph의 Peg/Box mean/min/max/bbox/span 정의는 10-C1 geometry_summary와 같다. mean은 표면 중심이며 actor origin이 아니다. 결측은 NaN, 지지점 32 미만·span 대각선 250mm 초과·mean–bbox center 거리 20mm 초과는 heuristic quality 실패로 별도 기록한다. 이 기준을 통과해도 geometry가 정확하다는 의미는 아니다. Box도 같은 품질 gate를 적용했다. 원자료에는 매우 큰 Box span이 있어 많은 쌍이 탈락한다. 이를 실제 Box 위치 차이라고 해석하지 않는다.

Robot76 전체 Euclidean distance는 사용하지 않았다. 관절7 qpos RMS(rad), qvel RMS(rad/s), TCP 위치 norm(mm)·quaternion geodesic(deg), opening(mm), opening velocity(m/s), TCP 선속도·각속도 RMS를 각각 gate했다. 현재·과거1/3/5 step에서 최악 거리를 사용하고 t<lag는 첫 frame padding이다. quaternion q/-q 동치, 중복 finger 상태를 중복 가중하지 않았다. phase는 관측 opening>=65mm/open, <=45mm/small, 사이/transition이고 open t<30만 early_open Reach proxy다. 정확한 Expert phase/grasp label은 없다. 절대 t 차이만 사용하며 episode 최종 길이/미래 state/action/GT를 matching에 쓰지 않았다.

기준표(모든 차원이 동시 통과; Box 품질/거리까지 통과해야 primary eligible):

{table(pd.DataFrame(LEVELS).T.reset_index().rename(columns={'index':'level'}))}

Control <5mm, 5–10/10–20mm 불확실, Different >=20mm(20–40/40+ 별도). 같은 Episode pair는 제외하고 모든 후보를 중복 없는 unordered frame pair로 저장했다. Selected는 robot score 오름차순으로 10-step episode block 및 unordered episode pair를 각 1회만 쓴다. 추가 episode-disjoint subset도 저장했다. 모든 쌍은 episode cluster 종속성을 유지하며 독립 표본 신뢰구간/p-value를 만들지 않는다.

## A — 조건부 다양성/관측 가능성

Train96:

{table(ct)}

Validation12 (기준 동결 후 내부 validation끼리; Train에 합산 안 함):

{table(cv)}

원 frame 품질:

{table(quality)}

Train 주 기준 Robot 유사도·geometry 분포(단위는 protocol):

{table(desc)}

`geometry_pair_coverage.csv`에 모든 threshold/offset 구간/유효·불량·Box 미통제 수/고유 frame·episode/coverage/최대 재사용을, `episode_coverage.csv`에는 **0쌍인 episode까지** 보존했다. 전체 후보 수가 유효 독립 사례 수는 아니다. Missing과 span 불량 등을 `frame_geometry_quality.csv`에서 재구분할 수 있다. t 분포와 phase는 전체 Pair CSV에 있다. 엄격 매칭의 scarcity는 데이터의 조건부 support가 제한적이라는 근거지만 “데이터에서 학습 불가능”의 증명은 아니다.

## B/C — Expert와 E0/E1 조건부 Action

Box-controlled primary에서 Expert/모델 비교 가능한 표본은 0이다. 따라서 primary의 평균·방향·gain은 정의하지 않았다. 필수 CSV에는 **모든 robot 후보의** 결과/quality/Box flags를 남겨 실패를 숨기지 않았다.

아래는 **사후 탐색적 Box-uncontrolled** 기술 통계다. 기준을 완화한 새로운 primary로 취급하지 않는다. 기존 Robot gate와 Peg quality를 유지하고 Box gate만 빼서 같은 score/block/episode-pair 제한을 적용했다. 일부 episode를 반복 사용하므로 n개 독립 실험이 아니다. `exploratory_episode_macro.csv`에 cluster별 평균, `exploratory_robot_only_pairs.csv`에 재사용 frame 정보가 있다.

{table(show)}

원시 주 기준 different 5쌍의 평균은 Expert 8.420mm, E0 3.918mm, E1 4.054mm, Δerror 6.584/7.113mm다. control 2쌍은 Expert 0.270mm라 방향 판정에서 제외했다. 이 비교는 표본이 적고 episode/구간/Box 및 관측 상태가 달라 Peg 조건부 신호의 일반적 크기를 추정하지 못한다. 완화된 sensitivity의 상대적으로 좋은 재현도 함께 보존했으며 주 기준을 대체하지 않는다.

모든 5개 Different 후보는 early_open 구간이고, 고유 frame 9개(Train의 0.0752%), 고유 Episode 7개다. 시간·episode pair 재사용을 제거한 탐색 subset은 3쌍/6 Episode이며 Expert 9.605mm 대비 E0/E1 1.943/1.335mm, Δerror 8.103/8.936mm다. 이 subset에는 control pair가 남지 않아 공정한 control 대비 추정도 불가능하다.

주 기준 Different 전체 실제 사례(반복 시점 포함; 사후 실패 사례 선택 없음):

{table(ex)}

명령은 설치된 ManiSkill PDEEPoseController의 clip/scale을 거친 Cartesian translation이다. 각 frame의 root command를 해당 robot base world rotation, TCP inverse rotation으로 바꾼 뒤 homologous TCP axes에서 b−a를 계산했다. achieved TCP 이동/IK 결과는 측정하지 않았다. Δvector error=||modelΔ−expertΔ||; signed gain=dot(modelΔ,expertΔ)/||expertΔ||²; magnitude ratio=||modelΔ||/||expertΔ||; zero improvement=||expertΔ||−error. ExpertΔ<1mm는 small로 기록하고 cosine/gain/ratio를 NaN 처리했다. 1mm는 방향 안정성용 사전 고정 분석 gate이며 controller 정확도 주장과 다르다. Zero modelΔ의 cosine도 NaN이다. Peg-action gain은 offset>=20mm와 ExpertΔ>=1mm에서만 보고한다. Rotation component norm과 gripper physical target 차이는 보조 지표이며 rotation의 finite geodesic 차이와 동일하지 않다.

이 비교는 로봇 상태가 완전히 같지 않고 이전 GRU/vision history 전체가 같은 것도 아니다. 동작 진행/Box/가시성/과거 물체 상태가 남는다. 특히 early_open Reach proxy 외 phase의 Peg-action cosine에 같은 방향을 요구하지 않는다. `expert_conditional_action.csv`의 phase별 원자료를 보존한다.

## Task 10-B/10-C1과의 연결

10-B는 동일 robot·previous hidden·vision prefix의 실제 Peg world XY ±5/10mm 개입이며 primary22에서 E0/E1 cosine 0.254/0.308, gain 0.185/0.116이었다. 이번 분석은 다른 episode·history의 관찰 비교다. 두 증거를 하나의 표본으로 합산하지 않는다. 이번 offset 5–10mm 후보는 원시 분포로 보존했지만 표면 중심 편향 때문에 실제 10-B의 5/10mm pose 개입과 동등하지 않다. 주 기준에서 유효 Box 통제 쌍 0이므로 **10-B의 exact object–robot 조합이 Train96에 충분히 있었다/없었다는 판정은 INCONCLUSIVE**다. Original HDF와 current-runtime 개입 observation도 byte-identical하지 않다. Intervention을 Probe 학습/threshold/모델 선택에 쓰지 않았다.

10-C1은 raw summary의 이동 정보와 downstream의 불안정성을 지지했으나, 이번 Train96 관찰 매칭은 H1/H2를 분리할 만큼의 controlled support를 제공하지 못했다. 그 사실로 10-B의 동일상태 반응 부족을 부정하지도, 데이터 다양성만으로 설명하지도 않는다.

## Q1–Q6 판정

- **Q1 CONFIRMED:** Train96 Robot 후보는 strict 3 / primary 77 / sensitivity 2,341쌍. primary Peg-quality 10쌍 중 control2/불확실3/different5다. Box까지 통제한 유효 pair0. 모든 offset 수·episode 수·coverage는 CSV/위 표에 있다. **H1 SUPPORTED(좁은 범위):** 이 관측 기반의 엄격 조건에서 비교 가능한 사례가 부족하다. 실제 GT 기반 데이터 다양성 부족은 INCONCLUSIVE다.
- **Q2 INCONCLUSIVE:** Robot-only 탐색에서는 Different의 ExpertΔ가 control보다 크지만 controlled pair가0이고 공변량·episode 집중이 남는다. Peg에 따른 체계적 correction이라 일반화할 수 없다.
- **Q3 INCONCLUSIVE:** 특정 관찰쌍에서 E0/E1 Δ 재현 부족은 수치로 확인되나 주 비교 표본이0이다. 작은 ExpertΔ의 control을 실패로 부르지 않는다. E0/E1 우열도 고르지 않는다.
- **Q4 SUPPORTED / INCONCLUSIVE:** 조건부 사례 scarcity와 관측 품질 문제는 제한적 설명 후보로 지지된다. 이것이 closed-loop geometry correction 실패의 주원인이라는 판단은 불가능하다. H1/H2는 함께 존재할 수 있다.
- **Q5 INCONCLUSIVE:** Box 미통제 실제 사례의 8.420mm ExpertΔ 대비 model 3.918/4.054mm 및 6.584/7.113mm Δerror는 재현 공백의 사례다. 하지만 충분한 독립 Peg 신호가 있었다는 전제를 충족하지 않아 이를 H2의 확증으로 제시하지 않는다.
- **Q6 SUPPORTED(다음 검증 우선순위):** 먼저 데이터의 조건부 support와 관측 geometry 신뢰도를 점검할 근거가 가장 강하다. BC objective와 representation 중 무엇을 먼저 변경할지는 INCONCLUSIVE다. 새 학습/구조 변경은 NOT_TESTED이고 실행하지 않았다.

최소 추가 검증은 소수의 사전 선택 robot/구간 anchor에서 **서로 다른 기존 episode**의 실제 Peg/Box 위치 또는 가시성 provenance를 수작업/저장자료로 검증하는 것이다. 5/10mm pose 변화와 centroid 변화의 대응·Box 교란을 분리하고, 동일한 고정 기준에 controlled pair가 남는지 확인한다. 그 뒤에도 pair가 거의 없다면 제한된 matched-state demonstration 다양성 보완을 별도 설계한다. 신호가 충분한 controlled 사례가 확인될 때에만 frozen 모델의 Δ 재현을 다시 검증한다. 이번에 추가 수집/학습은 실행하지 않았다.

## 검증과 한계

CONFIRMED: 108 episode cache receipt SHA, saved prediction target/action alignment, timestep, 10-C1 feature manifest/checkpoint/cache hash 연결 및 3e−5 prediction tolerance, 입력 전후 SHA 동일. 원본 Train96의 actor GT pose는 사용하지 않았다. Checkpoint weight update0, policy inference0. 저장 prediction–checkpoint 연결은 이전10-C1의 frozen forward 재현과 이번 hash/readback으로 검증했으며 새 forward를 수행하지 않았다.

C2의 3개 단위 계약 테스트: cross-episode exclusion/group gate/block reuse, controller Δ/zero response/small Δ, TCP rotation. 기존 Plan A 계약 테스트 9개도 통과했다(합성 forward/backward 계약이며 optimizer step은 없다). 원본 checkpoint의 forward/update는 실행하지 않았다. 실제 dataset 결과는 `verification.json`/CSV readback으로 별도 확인한다. CPU NumPy/SciPy + 설치 controller preprocessing만 실행했다. SAPIEN import의 Vulkan fallback warning은 physics 실행을 뜻하지 않는다. Simulator, CUDA/MPS, 새 vision, Test12, closed-loop insertion은 NOT_TESTED다. 공통 infrastructure 수정·외부 게시·git commit 없음. 실행 환경에 pandas3.1.0을 추가했으며 모델/캐시는 변경하지 않았다.

![Audit figures](audit_figures.png)

[Verification](verification.json) · [Matching](robot_state_matching.csv) · [Coverage](geometry_pair_coverage.csv) · [Expert](expert_conditional_action.csv) · [Models](model_conditional_action.csv) · [Comparison](conditional_response_comparison.csv)
'''
 (O/'10c2_learning_signal_audit.md').write_text(txt)
 # Self-contained static HTML; the report text is retained verbatim below rendered tables.
 body=f'<h1>Task 10-C2 · Learning Signal Audit</h1><p>UUID {run}</p><p class="callout">고정 기준의 Box-controlled 유효 Pair: 0. H1/H2 분리: INCONCLUSIVE.</p><img src="audit_figures.png" alt="Three audit figures"/><h2>Train96 coverage</h2>'+ct.to_html(index=False)+'<h2>Validation12 coverage</h2>'+cv.to_html(index=False)+'<h2>Exploratory only · Box uncontrolled</h2>'+show.to_html(index=False,float_format=lambda v:f'{v:.3f}')+'<h2>Full report</h2><pre>'+html.escape(txt)+'</pre>'
 links=['robot_state_matching.csv','geometry_pair_coverage.csv','expert_conditional_action.csv','model_conditional_action.csv','conditional_response_comparison.csv','verification.json','10c2_learning_signal_audit.md']
 body+='<h2>Downloads</h2>'+''.join(f'<p><a href="{v}">{v}</a></p>' for v in links)
 (O/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task 10-C2 Audit</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:40px auto;padding:20px;color:#182838;background:#f8fafc}h1,h2{color:#254a66}.callout{padding:20px;background:#fff0cd;border-left:5px solid #b87921}img{width:100%}table{border-collapse:collapse;background:white;font-size:13px;display:block;overflow-x:auto}td,th{padding:8px;border:1px solid #cbd5e1}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px;font:14px/1.7 ui-monospace,monospace}</style>'+body+'</html>')
 # Register all new outputs to the same Logger UUID; no new experiment identity.
 for path in O.iterdir():
  if path.is_file() and not path.name.endswith('.tracking.json'):register_existing_output(exp,path,kind='metrics' if path.suffix=='.json' else 'reports')
 print(stats.to_string(index=False));print('REPORT',O/'index.html')
if __name__=='__main__':main()

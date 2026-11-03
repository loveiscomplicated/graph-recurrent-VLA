"""Posthoc comparison of frozen selections and already executed SAM2 masks.

No model, simulator, API, remote host, or GT-array loading is performed.
"""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_logger import ExperimentLogger

PARENT = REPO/'outputs/task10i/stage4a2'
OLD = PARENT/'sam2_mask_verifier_tracking_20261102_v1'
SELECTION = PARENT/'qwen35_appearance_bf16_independent_20261102_v1'
TERRA = PARENT/'api_terra_luna_grounding_20261103_v1'
OUT = PARENT/'terra_vs_dino_saved_comparison_20261103_v1'
REPORT = REPO/'docs/visual-graph-integration/10i_stage4a2_terra_standard_vs_dino_sam2_20261103.md'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
def write(p, d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p, rs):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)


def summarize(rs):
    result={}
    for obj in ['peg','box']:
        vv=[r for r in rs if r['object']==obj and r['visibility']=='VISIBLE']
        em=[r for r in rs if r['object']==obj and r['nonempty'] and r['visibility']!='TINY_VISIBLE']
        result[obj]=dict(success=sum(r['success'] for r in vv),n=len(vv),
            wrong=sum(r['wrong'] for r in em),wrong_n=len(em),
            mean_mask_IoU=float(np.mean([r['IoU'] for r in vv])))
    scenes=defaultdict(list)
    for r in rs: scenes[r['image_id']].append(r)
    scene_rows=[]
    for iid,z in sorted(scenes.items()):
        assert len(z)==2 and {r['object'] for r in z}=={'peg','box'}
        if not all(r['visibility']=='VISIBLE' for r in z): continue
        scene_rows.append(dict(image_id=iid,parent_id=z[0]['parent_id'],camera=z[0]['camera'],
            frame_kind=z[0]['frame_kind'],success=all(r['success'] for r in z)))
    result['joint']=dict(success=sum(r['success'] for r in scene_rows),n=len(scene_rows))
    result['per_parent']={p:float(np.mean([r['success'] for r in scene_rows if r['parent_id']==p]))
                          for p in sorted({r['parent_id'] for r in scene_rows})}
    result['parent_macro']=float(np.mean(list(result['per_parent'].values())))
    for key,values in [('camera',['base_camera','hand_camera']),('frame_kind',['F0','F1'])]:
        result[key]={v:dict(success=sum(r['success'] for r in scene_rows if r[key]==v),
            n=sum(r[key]==v for r in scene_rows)) for v in values}
    return result,scene_rows


def paired(baseline,new,parents):
    a={r['image_id']:r for r in baseline};b={r['image_id']:r for r in new}
    assert a.keys()==b.keys()
    transitions=[dict(image_id=k,parent_id=a[k]['parent_id'],camera=a[k]['camera'],
        frame_kind=a[k]['frame_kind'],baseline_success=a[k]['success'],terra_success=b[k]['success']) for k in sorted(a)]
    delta=np.array([np.mean([int(r['terra_success'])-int(r['baseline_success']) for r in transitions if r['parent_id']==p]) for p in parents])
    draws=np.random.default_rng(42).integers(0,len(parents),size=(10000,len(parents)))
    return dict(parent_n=len(parents),parent_macro_delta=float(delta.mean()),
        CI95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),
        recovered=sum(not r['baseline_success'] and r['terra_success'] for r in transitions),
        regressed=sum(r['baseline_success'] and not r['terra_success'] for r in transitions),
        both_success=sum(r['baseline_success'] and r['terra_success'] for r in transitions),
        both_failure=sum(not r['baseline_success'] and not r['terra_success'] for r in transitions)),transitions


def main():
    assert not OUT.exists(),'Refusing to overwrite this analysis'
    sources=[OLD/'candidate_evaluation.csv',OLD/'candidate_manifest.json',OLD/'input_manifest.json',
        OLD/'sam2_initialization_cases.csv',OLD/'sam2_initialization_metrics.json',OLD/'sam2_identity.json',
        OLD/'independent_verification.json',SELECTION/'selected_boxes.csv',SELECTION/'dino_frame_selections.json',
        TERRA/'input_manifest.json',TERRA/'evaluation_rows.json',TERRA/'grounding_evaluation_metrics.json',
        TERRA/'sam2/sam2_predictions.json',TERRA/'independent_verification.json',
        REPO/'graph/task10i_stage4a2/README.md',
        REPO/'docs/visual-graph-integration/10i_stage4a2_paused_scope_review.md',Path(__file__)]
    before={str(p.relative_to(REPO)):sha(p) for p in sources}
    assert read(OLD/'independent_verification.json')['status']=='PASS'
    assert read(TERRA/'independent_verification.json')['status']=='PASS'
    om={r['id']:r for r in read(OLD/'input_manifest.json')['cases']}
    tm={r['id']:r for r in read(TERRA/'input_manifest.json')['images']}
    assert om.keys()==tm.keys() and len(tm)==60
    for iid in om:
        for k in ['parent_id','camera','frame_kind','timestep','image_sha256','RGB_sha256']:
            assert om[iid][k]==tm[iid][k],(iid,k)
    assert read(OLD/'sam2_identity.json')['sha256']==read(TERRA/'sam2/sam2_predictions.json')['checkpoint_sha256']
    candidates={r['id']:r for r in read(OLD/'candidate_manifest.json')['candidates']}
    cm={}
    for r in rows(OLD/'candidate_evaluation.csv'):
        met=(r['mask_good08'],r['wrong_mask'],r['mask_pixels'],r['mask_IoU'])
        if r['candidate_id'] in cm:
            assert met==cm[r['candidate_id']][0],'A/B/C must share identical candidate masks'
        else: cm[r['candidate_id']]=(met,r)
    selections=rows(SELECTION/'selected_boxes.csv')
    methods={}
    for method in ['legacy','raw_top1','appearance_query_gate']:
        out=[]
        for x in selections:
            if x['method']!=method: continue
            assert not x['candidate_id'] or x['candidate_id'] in cm
            c=cm[x['candidate_id']][1] if x['candidate_id'] else None
            if c:
                assert c['image_id']==x['image_id'] and c['query_object']==x['object']
                assert candidates[x['candidate_id']]['box']==json.loads(x['box'])
            out.append(dict(method=method,image_id=x['image_id'],parent_id=x['parent_id'],camera=x['camera'],
                frame_kind=x['frame_kind'],object=x['object'],candidate_id=x['candidate_id'],
                visibility=x['visibility'],success=bool(c and c['mask_good08']=='True'),
                wrong=bool(c and c['wrong_mask']=='True'),nonempty=bool(c and int(c['mask_pixels'])>0),
                IoU=float(c['mask_IoU']) if c and x['visibility']=='VISIBLE' else 0.))
        methods[method]=out
    # Replay selection alone, without GT: verify the two Qwen-free baselines.
    frames={x['id']:x for x in read(SELECTION/'dino_frame_selections.json')['frames']}
    for method in ['legacy','raw_top1']:
        for r in methods[method]:
            pool=sorted([c for c in candidates.values() if c['image_id']==r['image_id'] and c['query_object']==r['object']],
                        key=lambda c:(-c['DINO_score'],c['rank']))
            saved=frames[r['image_id']]['selected'][r['object']]
            expected=(pool[0]['id'] if pool else '') if method=='raw_top1' else next((c['id'] for c in pool if saved and c['box']==saved['box']),'')
            assert r['candidate_id']==expected
    # Reconstruct the published A result from candidate metrics as an exact control.
    ar={(r['image_id'],r['object']):r for r in rows(OLD/'sam2_initialization_cases.csv') if r['method']=='A'}
    for r in methods['appearance_query_gate']:
        a=ar[(r['image_id'],r['object'])]
        assert r['candidate_id']==a['candidate_id']
        assert r['success']==(a['success08']=='True') and r['wrong']==(a['wrong_object']=='True')
        if r['visibility']=='VISIBLE':assert abs(r['IoU']-float(a['IoU']))<1e-12
    methods['terra']=[dict(method='terra',image_id=x['image_id'],parent_id=x['parent_id'],camera=x['camera'],
        frame_kind=x['frame_kind'],object=x['object'],candidate_id='',visibility=x['visibility'],
        success=x['mask08'],wrong=x['wrong_object'],nonempty=x['mask_pixels']>0,
        IoU=x['mask_IoU'] if x['visibility']=='VISIBLE' else 0.)
        for x in read(TERRA/'evaluation_rows.json')['rows'] if x['provider']=='terra']
    visibility={(r['image_id'],r['object']):r['visibility'] for r in methods['terra']}
    for rs in methods.values():
        assert len(rs)==120
        for r in rs:assert visibility[(r['image_id'],r['object'])]==r['visibility']
    summaries={};scene_rows={}
    for method,rs in methods.items():summaries[method],scene_rows[method]=summarize(rs)
    saved_a=read(OLD/'sam2_initialization_metrics.json')['methods']['A']
    assert summaries['appearance_query_gate']['joint']['success']==saved_a['image_object_level']['scene']['both_mask08']['numerator']
    saved_t=read(TERRA/'grounding_evaluation_metrics.json')['models']['terra']
    for obj in ['peg','box']:
        assert summaries['terra'][obj]['success']==saved_t['overall'][obj]['mask08']['numerator']
    parents=sorted(summaries['terra']['per_parent']);assert len(parents)==15
    effects={};transitions=[]
    for method in ['legacy','raw_top1','appearance_query_gate']:
        effects[method],tt=paired(scene_rows[method],scene_rows['terra'],parents)
        transitions.extend(dict(baseline=method,**r) for r in tt)
    OUT.mkdir(parents=True)
    write(OUT/'input_provenance.json',before)
    config=dict(type='posthoc_saved_metric_analysis',model='Terra Standard vs saved DINO paths',
        parents=15,images=60,GT_visible_pixel_min=16,mask_IoU_min=.8,
        bootstrap=dict(unit='physical parent',replicates=10000,seed=42),new_model_calls=0)
    write(OUT/'protocol.json',config)
    write(OUT/'comparison.json',dict(methods=summaries,paired_effects=effects))
    csvwrite(OUT/'object_cases.csv',[r for rs in methods.values() for r in rs])
    csvwrite(OUT/'paired_scene_transitions.csv',transitions)
    assert all(sha(REPO/p)==h for p,h in before.items())
    verify=dict(status='PASS',matched_images=60,matching_PNG_and_RGB_hashes=60,
        matching_visibility_objects=120,legacy_and_top1_GT_free_selection_replay=240,
        reconstructed_published_A_objects=120,shared_candidate_masks=986,
        same_SAM2_checkpoint=True,protected_source_and_artifact_files=len(before),
        note='Saved metric reaggregation, not new Mask execution or independent pixel-IoU recalculation.')
    write(OUT/'verification.json',verify)
    def fmt(k,wrong=False):
        n,d=(k['wrong'],k['wrong_n']) if wrong else (k['success'],k['n'])
        return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'N/A'
    names={'legacy':'기존 DINO 선택 규칙 + SAM2','raw_top1':'DINO 원시 Top1 + SAM2',
           'appearance_query_gate':'DINO + Qwen 외형 검증 + SAM2 (기존 A)','terra':'Terra Standard + SAM2'}
    table=[]
    for method,m in summaries.items():
        table.append(f"| {names[method]} | {fmt(m['peg'])} | {fmt(m['box'])} | {fmt(m['joint'])} | {fmt(m['peg'],True)} | {fmt(m['box'],True)} |")
    groups=[]
    for key,val,title in [('camera','base_camera','Main'),('camera','hand_camera','Wrist'),
                          ('frame_kind','F0','초기 t0'),('frame_kind','F1','후기 t80/t78')]:
        groups.append('| '+title+' | '+' | '.join(fmt(summaries[m][key][val]) for m in ['legacy','appearance_query_gate','terra'])+' |')
    cis=[]
    for method,e in effects.items():
        cis.append(f"| Terra − {names[method]} | {100*e['parent_macro_delta']:+.2f}%p | {100*e['CI95'][0]:+.2f}–{100*e['CI95'][1]:+.2f}%p | {e['recovered']} | {e['regressed']} |")
    legacy=summaries['legacy']['joint'];new=summaries['terra']['joint'];a=summaries['appearance_query_gate']['joint']
    rel='../../'+str(OUT.relative_to(REPO))
    text=f'''# Task 10-I Stage 4A-2: Terra Standard와 기존 DINO–SAM2의 저장 결과 비교

**같은 native512 60장에 대한 정적 초기화에서, 기존 DINO 선택 규칙 + SAM2의 두 물체 동시 Mask 성공은 {fmt(legacy)}, Terra Standard + SAM2는 {fmt(new)}였다.** 성공률 **+{100*(new['success']/new['n']-legacy['success']/legacy['n']):.1f}%p**, 성공 이미지 **{new['success']-legacy['success']}장 증가**, 실패 **{legacy['n']-legacy['success']}→{new['n']-new['success']}장 ({100*(1-(new['n']-new['success'])/(legacy['n']-legacy['success'])):.1f}% 감소)**다.

앞서 비교에 사용한 27/58 (46.6%)는 **DINO 단독이 아니라 Qwen 외형 검증을 포함한 A 경로**였다. 그 개선된 경로와 비교해도 Terra는 **+{100*(new['success']/new['n']-a['success']/a['n']):.1f}%p, +{new['success']-a['success']}장**이다. 두 기준선을 구분한다.

## 자료와 비교 방법

- 기존 Train Parent 15개, F0=t0/F1=t80 (Parent0134만 t78), Main/Wrist 총 60장. 기존 두 실험의 Image ID·Parent·Camera·시점·PNG SHA·RGB SHA가 60/60 모두 일치했다.
- 성공은 GT Visible Pixel≥16인 대상의 **실제 SAM2 Mask IoU≥0.8**이다. Peg 60, Fixture 58, 두 물체 동시 58장이 분모다. Null·미초기화는 실패로 포함한다.
- 모든 경로의 SAM2.1 Hiera Small checkpoint SHA가 일치한다. 기존 실행은 FP32 Weight/BF16 autocast, multimask_output=False, logit>0 및 이미지 Embedding 재사용을 사용했다.
- 기존 DINO 선택 규칙은 저장된 `legacy` 선택 그대로다. 면적25% 제외·Top5·Null/Joint 점수 규칙으로 고른 후보를 사용한다. 원시 Top1은 각 Query의 원래 DINO Score 1위다. Qwen A는 같은 Query에서 외형 Label이 맞는 후보 중 DINO Score 1위다.
- **새 DINO/Qwen/Terra/SAM2 호출은 0회다.** 기존 986개 DINO 후보에 실제 실행해 둔 SAM2 Mask의 점수를 저장된 legacy/Top1 선택과 연결했다. 결과를 보고 후보를 다시 고르거나 GT로 최적 후보를 선택하지 않았다.
- legacy/Top1의 Mask 수치는 이번 사후 재집계다. 기존 보고서의 Box IoU≥0.5 수치를 Mask 성공률로 전용하지 않았다. 후보별 A/B/C Mask 지표 동일성과 A 선택 120개의 복원 결과를 기존 발표 값에 대조했다.

## 실제 Mask 초기화 결과

| 경로 | Peg 성공 | Fixture 성공 | 두 물체 동시 성공 | Peg Wrong-object | Fixture Wrong-object |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask Pixel의 50% 이상일 때다. 분모는 Tiny를 제외한 비어 있지 않은 예측 Mask이며 가시 대상 성공 분모와 다르다. Terra의 0건은 이번 표본에서 관측되지 않았다는 뜻이다.

| 평균 Mask IoU (가시 대상, 누락=0) | 기존 DINO 선택 + SAM2 | DINO+Qwen A+SAM2 | Terra Standard+SAM2 |
| --- | ---: | ---: | ---: |
| Peg | {summaries['legacy']['peg']['mean_mask_IoU']:.3f} | {summaries['appearance_query_gate']['peg']['mean_mask_IoU']:.3f} | {summaries['terra']['peg']['mean_mask_IoU']:.3f} |
| Fixture | {summaries['legacy']['box']['mean_mask_IoU']:.3f} | {summaries['appearance_query_gate']['box']['mean_mask_IoU']:.3f} | {summaries['terra']['box']['mean_mask_IoU']:.3f} |

## Camera·시점별 동시 성공

| 범위 | 기존 DINO 선택 + SAM2 | DINO+Qwen A+SAM2 | Terra Standard+SAM2 |
| --- | ---: | ---: | ---: |
{chr(10).join(groups)}

## Parent 대응 비교

같은 물리 Parent 15개를 독립 단위로 삼아 Parent별 동시 성공률 차이를 계산했다. Parent-cluster Bootstrap 10,000회, Seed42다. 아래 차이는 전체 이미지 성공률 차이와 달리 Parent-macro 차이다.

| 대응 비교 | Parent-macro 차이 | 95% CI | 기존 실패→Terra 성공 | 기존 성공→Terra 실패 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(cis)}

Camera/시점을 60개의 독립 Parent로 세지 않았다. 이는 고정된 알려진 장면에서의 탐색적 경로 비교이며 모델이나 Prompt 한 요소만의 인과 효과를 분리하지 않는다.

## 속도·비용에서 비교할 수 있는 범위

Terra Standard의 기존 HTTP 요청 평균 6.53초, 중앙값 4.59초, 60장 Usage 기반 API 비용 약 $0.147435를 재사용한다. 확정 청구액이 아니며 GPU 비용은 별도다. Qwen 후보 검증에는 조건당 약 986회×0.61초/60장, 약 10초/이미지 규모가 기록됐지만 DINO Forward·Load·SAM2·I/O를 포함한 동일 정의의 파이프라인 E2E는 확보되지 않았다. 따라서 DINO 경로 대비 정확한 속도 배수나 GPU/API 총비용 절감률은 주장하지 않는다.

## 판단과 보존

현재 자료에서는 Terra Standard로 Grounding을 바꾸면 **두 물체가 모두 올바르게 분할되는 이미지가 크게 늘고, 기존 선택 경로의 다른 Actor 오초기화가 줄었다**고 판단할 수 있다. Qwen 검증을 이미 더한 기준선보다도 개선됐다. 후기 프레임의 초기화 수치는 매 프레임 독립 재발견 조건이며 t0 초기화 뒤 SAM2로 유지한 Tracking 성공률과 섞지 않는다.

이번 결론은 **초기화/독립 재탐지 이미지의 Mask 품질**에 해당한다. 기존 DINO 런타임의 5프레임 간격 탐색·2회 일치 후 초기화·Video Predictor 전파를 새로 실행한 전체 시스템 비교가 아니다. Terra 초기화 후 Tracking과 학습/Controller 성공률은 이번 비교에서 측정하지 않았다. 이 한계 때문에 추가 장면 실험을 자동으로 시작하지 않는다.

공통 ExperimentLogger(W&B disabled)를 재사용했다. 기존 입력·평가 파일 {len(before)}개 SHA 보존, 60개 이미지의 PNG/RGB·시점 일치, GT-free 선택 재현 240개, 기존 A 결과 120개 대조를 통과했다. 기존 지표를 재집계했으며 원시 GT Pixel과 Mask IoU를 새로 계산한 검사는 아니다. 로컬 macOS CPU 파일 분석만 실행했다. 원격 접속, GPU/CUDA/MPS, Simulator Render/Rollout, Graph/학습/Controller, Production 변경, Commit, 게시를 실행하지 않았다. PAUSED_SCOPE_REVIEW·Stop Stub과 기존 산출물을 유지했다.

산출물: [비교 JSON]({rel}/comparison.json) · [물체별 재집계]({rel}/object_cases.csv) · [이미지별 성공 전환]({rel}/paired_scene_transitions.csv) · [입력 SHA]({rel}/input_provenance.json) · [검증]({rel}/verification.json).

근거 보고서: [기존 SAM2 초기화·추적](10i_stage4a2_sam2_mask_verifier_tracking.md) · [DINO 선택 규칙과 Box 평가](10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md) · [Terra Standard 및 계열 모델 비교](10i_stage4a2_api_openai_family_grounding_comparison.md). 선택 규칙의 런타임 설명은 [Visual README](../../visual/runtime/README.md)를 따른다.
'''
    REPORT.write_text(text)
    os.environ['WANDB_MODE']='disabled'
    with ExperimentLogger(team='integration',model='Terra_Standard_vs_saved_DINO_SAM2',seed=42,
            job_type='evaluate',root=OUT/'.execution/experiments',config=config) as log:
        log.log({method:dict(joint_success_rate=m['joint']['success']/m['joint']['n'],parent_macro=m['parent_macro']) for method,m in summaries.items()})
        log.link_output(OUT/'comparison.json')
        log.link_output(REPORT,kind='reports')
    write(OUT/'artifact_index.json',{str(p.relative_to(REPO)):sha(p) for p in
        sorted([p for p in OUT.rglob('*') if p.is_file() and '.execution' not in p.parts]+[REPORT,Path(__file__)])})
    print(json.dumps(dict(methods=summaries,paired_effects=effects,verification=verify),indent=2))
    print('REPORT',REPORT)


if __name__=='__main__':main()

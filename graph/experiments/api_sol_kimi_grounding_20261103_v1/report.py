"""Final paired comparison from frozen predictions and masks. No API calls."""
from common import *
from collections import Counter
from datetime import datetime
import itertools
import numpy as np
from PIL import Image,ImageDraw,ImageFont

NAMES={'astra':'GPT-6 Astra','sol':'GPT-6.1 Sol','kimi':'Kimi K3','gemma':'Gemma4 31B'}

def fmt(rate):
    n,d=rate['numerator'],rate['denominator']
    return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'N/A (0)'

def ratio(values):
    return dict(numerator=sum(bool(v) for v in values),denominator=len(values),rate=float(np.mean(values)) if values else None)

def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None)

def tables(rows,scenes):
    result={}
    for key in NAMES:
        rr=rows[key];ss=[s for s in scenes[key] if s['both_visible']]
        result[key]=dict(joint=ratio([s['both_mask08'] for s in ss]),
            main=ratio([s['both_mask08'] for s in ss if s['camera']=='base_camera']),
            wrist=ratio([s['both_mask08'] for s in ss if s['camera']=='hand_camera']),
            initial=ratio([s['both_mask08'] for s in ss if s['frame_kind']=='F0']),
            later=ratio([s['both_mask08'] for s in ss if s['frame_kind']=='F1']))
        for obj in ['peg','box']:
            visible=[r for r in rr if r['object']==obj and r['visibility']=='VISIBLE']
            allr=[r for r in rr if r['object']==obj]
            masks=[r for r in allr if r['mask_pixels'] and r['visibility']!='TINY_VISIBLE']
            result[key][obj]=dict(mask08=ratio([r['mask08'] for r in visible]),
                wrong_object=ratio([r['wrong_object'] for r in masks]),
                box03=ratio([r['box03'] for r in visible]),box05=ratio([r['box05'] for r in visible]),
                box07=ratio([r['box07'] for r in visible]),
                box_IoU=stats([r['box_IoU'] for r in visible]),
                center_error_pixels=stats([r['center_error_pixels'] for r in visible if r['center_error_pixels'] is not None]),
                mask_IoU=stats([r['mask_IoU'] for r in visible]),
                precision=stats([r['precision'] for r in visible if r['precision'] is not None]),
                recall=stats([r['recall'] for r in visible if r['recall'] is not None]),
                null=ratio([r['null'] for r in allr]),missing=ratio([r['missing'] for r in visible]),
                hallucination=ratio([r['valid_box'] for r in allr if r['visibility']=='ABSENT']),
                visibility=dict(Counter(r['visibility'] for r in allr)),
                robot_contamination=stats([r['contamination']['robot'] for r in masks]))
        perimage={iid:[r for r in rr if r['image_id']==iid] for iid in {r['image_id'] for r in rr}}
        result[key]['json_valid']=ratio([all(r['json_valid'] for r in a) for a in perimage.values()])
        result[key]['coordinate_contract']=ratio([all(r['valid_or_null'] for r in a) for a in perimage.values()])
    return result

def bootstrap(scenes):
    ids=sorted({s['parent_id'] for ss in scenes.values() for s in ss})
    draws=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)))
    macro={k:np.asarray([np.mean([s['both_mask08'] for s in ss if s['parent_id']==p and s['both_visible']]) for p in ids]) for k,ss in scenes.items()}
    absolute={k:dict(parent_n=len(ids),macro=float(a.mean()),CI95=np.quantile(a[draws].mean(1),[.025,.975]).tolist()) for k,a in macro.items()}
    paired={}
    for a,b in itertools.combinations(NAMES,2):
        delta=macro[a]-macro[b]
        paired[a+'_minus_'+b]=dict(parent_n=len(ids),macro_delta=float(delta.mean()),CI95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    return dict(absolute=absolute,paired=paired,replicates=10000,seed=42,unit='physical parent')

def figure(rows):
    # Hash-fixed selection, chosen without using success/failure values.
    cases=read(ROOT/'input_manifest.json')['images'][:4]
    ids=[c['id'] for c in cases]
    lookup={k:{(r['image_id'],r['object']):r for r in rr} for k,rr in rows.items()}
    receipts=read(ROOT/'sam2/sam2_predictions.json')['rows']
    masks={('sol' if r['provider']=='openai' else 'kimi' if r['provider']=='ollama' else 'gemma',r['image_id'],r['object']):ROOT/'sam2'/r['mask_file'] for r in receipts}
    for r in read(BASELINE/'sam2/sam2_predictions.json')['rows']:
        if r['provider']=='openai':masks['astra',r['image_id'],r['object']]=BASELINE/'sam2'/r['mask_file']
    canvas=Image.new('RGB',(5*256,len(cases)*294),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
    except OSError:font=ImageFont.load_default()
    for y,c in enumerate(cases):
        rgb=Image.open(c['image_path']).convert('RGB')
        for col,key in enumerate(['rgb']+list(NAMES)):
            im=rgb.copy().convert('RGBA')
            if key!='rgb':
                for obj,color in [('peg',(255,40,40)),('box',(40,100,255))]:
                    r=lookup[key][c['id'],obj];mp=masks.get((key,c['id'],obj))
                    if mp and r['predicted_box']:
                        overlay=Image.new('RGBA',im.size,color+(0,));alpha=Image.open(mp).convert('L').point(lambda a:70 if a else 0)
                        overlay.putalpha(alpha);im=Image.alpha_composite(im,overlay)
                    d=ImageDraw.Draw(im)
                    if r['predicted_box']:d.rectangle(r['predicted_box'],outline=color+(255,),width=3)
                    if r['GT_visible_box']:d.rectangle(r['GT_visible_box'],outline=(0,210,70,255),width=2)
            x=col*256;top=y*294
            canvas.paste(im.convert('RGB').resize((256,256)),(x,top+38))
            title='RGB '+c['id'].replace('10i_s1_','').replace('_native512','') if key=='rgb' else NAMES[key]
            draw.text((x+3,top+2),title,fill='black',font=font)
            if key!='rgb':
                a=[lookup[key][c['id'],o] for o in ['peg','box']]
                label=(a[0]['api_status']+'; no mask' if a[0]['api_status']!='OK' else
                       'FORMAT_INVALID; no mask' if not a[0]['json_valid'] else
                       'Peg %.2f / Fixture %.2f'%(a[0]['mask_IoU'] or 0,a[1]['mask_IoU'] or 0))
                draw.text((x+3,top+19),label,fill='black',font=font)
    canvas.save(ROOT/'full_image_grounding_comparison.png')
    return ids

def main():
    v=verify_frozen();assert v['stage'] in ['MASKS_FROZEN','COMPLETE']
    for group in ['api_files','raw_files','sam2_files','segmentation_inputs']:
        for n,h in v[group].items():assert sha(ROOT/n)==h,n
    prior=read(BASELINE/'evaluation_rows.json')
    amendment=read(ROOT/'scope_amendment.json') if (ROOT/'scope_amendment.json').exists() else None
    kimi_withdrawn=bool(amendment)
    summaries={};statistics={};allrows={};allscenes={}
    for mode in ['strict','fence']:
        current=read(ROOT/(('fence_' if mode=='fence' else '')+'evaluation_rows.json'))
        rows={'astra':[r for r in prior['rows'] if r['provider']=='openai']}
        scenes={'astra':[r for r in prior['scenes'] if r['provider']=='openai']}
        for key,provider in [('sol','openai'),('kimi','ollama'),('gemma','gemma_baseline')]:
            rows[key]=[r for r in current['rows'] if r['provider']==provider]
            scenes[key]=[r for r in current['scenes'] if r['provider']==provider]
        allrows[mode]=rows;allscenes[mode]=scenes
        summaries[mode]=tables(rows,scenes);statistics[mode]=bootstrap(scenes)
        if kimi_withdrawn:
            statistics[mode]['absolute'].pop('kimi',None)
            statistics[mode]['paired']={k:v for k,v in statistics[mode]['paired'].items() if 'kimi' not in k}
    # Same successful-API image subset: keep strict/fence formatting failures in denominators.
    common=set.intersection(*[{r['image_id'] for r in rr if r['api_status']=='OK'} for k,rr in allrows['fence'].items() if not (kimi_withdrawn and k=='kimi')])
    common_results={mode:tables({k:[r for r in rr if r['image_id'] in common] for k,rr in allrows[mode].items()},
                               {k:[s for s in ss if s['image_id'] in common] for k,ss in allscenes[mode].items()}) for mode in ['strict','fence']}
    jobs=list(read(ROOT/'request_ledger.json')['jobs'].values())
    costs={};latency={}
    for provider in MODELS:
        jj=[j for j in jobs if j['provider']==provider];known=[j for j in jj if j.get('usage_estimated_usd') is not None]
        accounted=sum(j['accounted_usd'] for j in jj)
        key='sol' if provider=='openai' else 'kimi';success=summaries['fence'][key]['joint']['numerator']
        costs[key]=dict(requests=len(jj),known_usage_n=len(known),usage_estimated_usd=sum(j['usage_estimated_usd'] for j in known),
            reserved_unknown_usd=sum(j['accounted_usd'] for j in jj if j.get('usage_estimated_usd') is None),
            accounted_usd=accounted,accounted_per_assigned_image_usd=accounted/60,
            accounted_per_joint_success_usd=accounted/success if success else None,
            usage={u:sum((j.get('usage') or {}).get(u) or 0 for j in known) for u in ['input_tokens','output_tokens','reasoning_tokens']},
            reasoning_tokens_separately_available=provider=='openai',invoice_confirmed=False,
            status=dict(Counter(j['status'] for j in jj)))
        ok=[j for j in jj if j['status']=='OK']
        latency[key]=dict(all_attempted=stats([j['latency_ms']/1000 for j in jj]),normal_OK=stats([j['latency_ms']/1000 for j in ok]),
            first_seconds=jj[0]['latency_ms']/1000 if jj else None,
            completed_including_token_limit=stats([j['latency_ms']/1000 for j in jj if j['status'] in ['OK','TOKEN_LIMIT']]),
            token_limit=stats([j['latency_ms']/1000 for j in jj if j['status']=='TOKEN_LIMIT']),
            subsequent_OK=stats([j['latency_ms']/1000 for j in jj[1:] if j['status']=='OK']))
        if kimi_withdrawn and key=='kimi':
            costs[key].update(accounted_per_assigned_image_usd=None,accounted_per_joint_success_usd=None,
                user_withdrawal_reason='Excessive latency; not accuracy',comparison_status='INCOMPLETE_USER_STOPPED_LATENCY')
    start=min(datetime.fromisoformat(j['started_utc']) for j in jobs);end=max(datetime.fromisoformat(j['finished_utc']) for j in jobs)
    timing=dict(models=latency,API_wall_seconds=(end-start).total_seconds(),started_utc=start.isoformat(),finished_utc=end.isoformat(),
        HTTP_duration_sum_s=sum(j['latency_ms']/1000 for j in jobs),sam2=read(ROOT/'sam2/sam2_latency_vram.json'),
        note='Client HTTP end-to-end only; server vision/prefill/decode not separately observable; first request is not proven cold.')
    write(ROOT/'grounding_latency_cost.json',dict(costs=costs,timing=timing,prior_accounted_usd=PRIOR_ACCOUNTED_USD,
        new_accounted_usd=sum(c['accounted_usd'] for c in costs.values()),
        combined_accounted_usd=PRIOR_ACCOUNTED_USD+sum(c['accounted_usd'] for c in costs.values()),budget_usd=20))
    interpretation=dict(modes=summaries,parent_cluster_bootstrap=statistics,common_OK_images=len(common),common_OK_results=common_results,
        Kimi_withdrawal=amendment,Kimi_complete_model_comparison_valid=not kimi_withdrawn,
        no_new_parent_generalization=True,production_changed=False,DINO_replacement='INSUFFICIENT_SUPPORT')
    write(ROOT/'model_comparison.json',interpretation)
    selected=figure(allrows['fence'])
    def accuracy_table(mode):
        lines=['| Model | Peg Mask >=.8 | Fixture Mask >=.8 | Both Masks >=.8 | Valid coordinates incl. null |',
               '| --- | ---: | ---: | ---: | ---: |']
        for k,n in NAMES.items():
            if kimi_withdrawn and k=='kimi':
                lines.append('| Kimi K3 | 미완료 | 미완료 | 사용자 요청으로 시간상 중단 | 미완료 |')
                continue
            a=summaries[mode][k];lines.append(f"| {n} | {fmt(a['peg']['mask08'])} | {fmt(a['box']['mask08'])} | {fmt(a['joint'])} | {fmt(a['coordinate_contract'])} |")
        return '\n'.join(lines)
    lines=['# Task 10-I Stage 4A-2: GPT-6.1 Sol · Kimi K3 Full-image Grounding 및 SAM2 비교','',
        '## 목적과 해석 범위','',
        '동일한 전체 RGB에서 Peg와 Fixture를 독립적으로 찾고, 예측 Box만으로 SAM2 초기 Mask를 생성했다. 기존 Astra와 같은 장면에서 정확도·비용·지연을 비교한다. 이미 분석한 Train Parent 15개이므로 미사용 Parent 일반화 실험은 아니다.','',
        '## 실행 설정과 입력 보존','',
        ('**범위 변경:** Kimi K3는 응답 시간이 너무 오래 걸려 사용자 요청으로 중도 탈락했다. 정확도에 따른 탈락이 아니다. 완료된 응답·시간·비용을 보존하고, 진행 중이던 Kimi 요청1개는 클라이언트에서 중단했다. 서버 처리·청구 여부는 확인되지 않아 예약 비용을 유지했다. 이후 Kimi 재호출은 없으며 Sol만 기존60장 평가를 계속했다. Kimi의 미시도 이미지를 인식 실패로 해석하거나 전체 성능 순위를 매기지 않는다.' if kimi_withdrawn else '두 모델 모두 원래 고정 범위로 실행했다.'),'',
        f"- 신규 모델: GPT-6.1 Sol `gpt-6.1-sol` / Ollama Cloud Kimi K3 `kimi-k3`. 실제 신규 요청 {len(jobs)}/120회, 자동 Retry 0회.",
        '- 동일 native512 RGB 60장: 15 Parent × F0/F1 × Main/Wrist. F0=t0, F1=t80; 0134만 t78. 누락·실패 이미지 교체 없음.',
        f"- 프롬프트 SHA256: `{sha(ROOT/'grounding_prompt.txt')}`. 이전 외형 중심 프롬프트 원문과 동일하며 위치 힌트·Few-shot·Functional-role 설명 없음.",
        '- 한 요청은 전체 RGB 하나와 프롬프트만 사용한다. DINO 후보·점수·GT·Mask·Parent ID·이전 대화·도구를 입력하지 않았다.',
        '- Sol: Responses API, detail=original, reasoning=high, 생성 한도5120. Kimi: think=true, temperature=1, top_p=.95, top_k=64, num_predict=8192. 두 API 모두 Schema 강제 없음.',
        '- 유한한 숫자4개, bool 제외, 0–1000 범위와 순서를 검사한 뒤 .512를 곱했다. Clipping·Box 확대·GT 보정·응답 재생성 없음.',
        '- 모든 신규 API 원시 응답과 두 파싱 결과를 동결한 뒤 SAM2를 실행하고, 모든 Mask 동결 후 EXACT_GT Actor Segmentation으로 채점했다.',
        '- 기존 Astra 원시 결과·Mask를 재사용했다. Gemma는 기존 동결 응답에 새로 사전 고정한 코드펜스 제거 규칙을 적용하여 SAM2를 실행했다. Gemma API 재호출은 0회다.',
        '- 서비스 모델은 alias이며 서버 내부 Weight revision·양자화·GPU는 확인할 수 없다. 반환 모델 ID·Usage·요청 ID·UTC 시각은 원시 기록에 보존했다.','',
        '공식 설정·단가 출처: [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [Kimi K3](https://ollama.com/library/kimi-k3). 입력/출력 100만 토큰당 Sol $2/$10, Kimi $3/$15 기준으로 동결했다.','',
        '## 두 출력 계약의 결과','',
        '**Strict**는 전체 응답이 정확한 JSON일 때만 인정한다. **Fence**는 정확한 JSON 또는 응답 전체를 감싼 단일 코드펜스만 제거한다. 설명에서 좌표 검색·추론 내용 추출·좌표 수정은 하지 않는다. 두 조건은 같은 원시 응답과 같은 Box를 사용하며, 같은 유효 Box의 Mask를 두 번 생성하지 않았다.','',
        '### Strict JSON','',accuracy_table('strict'),'',
        '### 사전 고정한 코드펜스 제거 허용','',accuracy_table('fence'),'',
        '주 분모는 GT Visible Pixel >=16인 물체다. Peg60개, Fixture58개, 두 물체 모두 보이는 이미지58장이다. GT 1–15pixel은 Tiny, 0pixel은 Absent로 별도 저장했다. 서비스 실패·미시도·null·무효 응답은 해당 가시 물체 초기화 실패에 포함했다. Format 실패와 물체 인식 실패를 동일 원인으로 해석하지 않는다.','',
        '## Camera·시점별 동시 Mask 성공','',
        '| Model (Fence) | Main | Wrist | F0 | F1 |','| --- | ---: | ---: | ---: | ---: |']
    for k,n in NAMES.items():
        if kimi_withdrawn and k=='kimi':continue
        a=summaries['fence'][k];lines.append('| '+n+' | '+' | '.join(fmt(a[t]) for t in ['main','wrist','initial','later'])+' |')
    lines+=['','## Box Grounding과 잘못된 물체 초기화','',
            '| Model (Fence) | Peg Box IoU>=.5 | Fixture Box IoU>=.5 | Peg Wrong-object | Fixture Wrong-object |',
            '| --- | ---: | ---: | ---: | ---: |']
    for k,n in NAMES.items():
        if kimi_withdrawn and k=='kimi':continue
        a=summaries['fence'][k];lines.append(f"| {n} | {fmt(a['peg']['box05'])} | {fmt(a['box']['box05'])} | {fmt(a['peg']['wrong_object'])} | {fmt(a['box']['wrong_object'])} |")
    lines+=['','Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask의 50% 이상인 경우다. 낮은 Mask IoU 또는 일부 Robot 오염과 구분한다. 분모는 Tiny를 제외한 실제 비어 있지 않은 예측 Mask다. Box IoU>=.3/.5/.7, 중심 오차, Pixel Precision/Recall, GT Actor 구성·배경 오염, null·누락·Absent Hallucination은 [상세 지표](./ARTIFACT/model_comparison.json)에 저장했다.','',
        '## Parent 단위 불확실성과 대응 비교','',
        'Physical Parent15개를 독립 단위로 사용했다. Parent별 동시 성공률을 평균하고, Seed42로 Parent를 10,000회 재표집했다. 같은 Parent의 Camera·시점을 독립 Parent로 세지 않는다.','',
        '| Model (Fence) | Parent-macro | 95% Cluster Bootstrap CI |','| --- | ---: | ---: |']
    for k,n in NAMES.items():
        if kimi_withdrawn and k=='kimi':continue
        a=statistics['fence']['absolute'][k];lines.append(f"| {n} | {100*a['macro']:.1f}% | {100*a['CI95'][0]:.1f}–{100*a['CI95'][1]:.1f}% |")
    lines+=['','| Difference (Fence) | Parent-macro pp | 95% CI pp |','| --- | ---: | ---: |']
    for pair in (['astra_minus_sol'] if kimi_withdrawn else ['astra_minus_sol','astra_minus_kimi','sol_minus_kimi']):
        a=statistics['fence']['paired'][pair];lines.append(f"| {pair} | {100*a['macro_delta']:+.1f} | {100*a['CI95'][0]:+.1f}–{100*a['CI95'][1]:+.1f} |")
    lines+=['',f"완료 비교 모델의 API가 정상 응답한 공통 이미지 {len(common)}장에 대한 별도 대응 결과도 저장했다. 그 집합에서 동시 Mask 성공은 "+', '.join(NAMES[k]+' '+fmt(common_results['fence'][k]['joint']) for k in NAMES if not (kimi_withdrawn and k=='kimi'))+'다.',
        'Astra 기존 실행은 Mac 절전과 겹친 전송 실패3건을 포함한다. 이번 새 모델과 가용성을 그대로 비교하지 않는다. 전체 할당 분모 결과와 공통 정상응답 집합을 함께 본다. 작은 알려진 표본의 CI가 넓다는 점과 alias·시간·서비스 설정 차이를 고려해야 한다.','',
        '## API 지연·비용 및 총 소요 시간','',
        '| New Model | API status | OK latency mean / median / p95 | First request | Accounted USD | USD / successful joint init |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for k in ['sol','kimi']:
        c=costs[k];t=latency[k];a=t['normal_OK'];f=lambda x:f'{x:.2f}' if x is not None else 'N/A'
        per_success='$%.5f'%c['accounted_per_joint_success_usd'] if c['accounted_per_joint_success_usd'] is not None else 'N/A'
        lines.append(f"| {NAMES[k]} | {c['status']} | {f(a['mean'])} / {f(a['median'])} / {f(a['p95'])} s (n={a['n']}) | {f(t['first_seconds'])} s | ${c['accounted_usd']:.5f} | {per_success} |")
    total=sum(c['accounted_usd'] for c in costs.values());sam=timing['sam2']
    if kimi_withdrawn:
        a=latency['kimi']['completed_including_token_limit'];t=latency['kimi']['token_limit']
        lines+=['',f"Kimi는 완료된 요청 {a['n']}개(정상 응답과 토큰 한도 종료 모두 포함) 기준 평균 {a['mean']:.2f}초, 중앙값 {a['median']:.2f}초, p95 {a['p95']:.2f}초였다. 그중 토큰 한도 종료 {t['n']}개는 평균 {t['mean']:.2f}초였다. 표의 OK 지연만으로 중도 탈락 당시의 대기 비용을 설명하지 않는다. 사용자 중단1건의 시간은 UTC 기준 클라이언트 취소 시점까지의 경과이며 완료 응답 시간이 아니다."]
    lines+=['',f"신규 API 경과 시간 **{timing['API_wall_seconds']/60:.2f}분**. 순차 HTTP 측정 합계 {timing['HTTP_duration_sum_s']/60:.2f}분. 시작 {timing['started_utc']}, 종료 {timing['finished_utc']} (UTC).",
        f"신규 비용 보수 집계 **${total:.5f}**, 이전 실험 및 승인된 첫 이미지 반복 **${PRIOR_ACCOUNTED_USD:.5f}**를 합쳐 **${total+PRIOR_ACCOUNTED_USD:.5f}/$20**. 알려진 Usage×공식 단가의 추정과 Usage 누락 예약액을 구분했다. 청구서 확정값이 아니며 캐시 할인은 적용하지 않았다. 기존 3090 요금은 별도다.",
        'HTTP 시간은 로컬에서 측정한 서비스 호출 E2E다. 서버 Vision Encoding·Prefill·Decode는 분리 관측할 수 없다. 첫 요청을 Cold라고 확정하지 않는다. 이번에는 최초 이미지 추가 반복을 실행하지 않았다.',
        f"SAM2는 58576054 RTX3090에서 **{sam['SAM2_loads']}회 로드**, Mask **{len(read(ROOT/'sam2/sam2_predictions.json')['rows'])}개**를 생성했다. Load {sam['load_ms']:.1f}ms, Peak allocated {sam.get('peak_allocated_bytes',0)/1024**3:.3f}GiB. FP32 Weight·BF16 autocast, multimask_output=False, logit>0, 장기 Tracking/기존 Memory Commit 없음.",
        'VLM은 초기 발견·필요 시 재Grounding 용도다. 이번 API 지연으로 Control20Hz 실행을 주장하지 않는다.','',
        '## 실패 분해','',
        '| Model (Fence) | Visible object failures | Correct Box IoU>=.7 but Mask<.8 | Wrong-object failures |',
        '| --- | ---: | ---: | ---: |']
    for k,n in NAMES.items():
        if kimi_withdrawn and k=='kimi':continue
        fail=[r for r in allrows['fence'][k] if r['visibility']=='VISIBLE' and not r['mask08']]
        lines.append(f"| {n} | {len(fail)} | {sum(r['box_IoU']>=.7 for r in fail)} | {sum(r['wrong_object'] for r in fail)} |")
    lines+=['','관측: Box가 정확해도 SAM2가 Peg 일부만 분할하거나 Robot 일부를 포함할 수 있다. 반대로 느슨한 Box에서도 정확한 Mask가 나올 수 있다. 이러한 사례를 VLM의 의미 오인식으로 일괄 분류하지 않았다. 색상 경계·가림·그리퍼 접촉이 원인일 수 있다는 설명은 가설이며 이번 실험에서 인과적으로 분리 검증하지 않았다. 모델별 모든 실패 ID와 Actor 구성은 Strict/Fence 실패 CSV에 남겼다.','',
        '## 소규모 비교 그림','',
        '![Full-image grounding comparison](./ARTIFACT/full_image_grounding_comparison.png)','',
        '빨강=예측 Peg, 파랑=예측 Fixture, 초록=사후 GT Box. 반투명 색은 실제 실행한 SAM2 Mask다. Hash 순서 첫4장 '+', '.join(selected)+'을 표시했다. 성공 사례 선택으로 그림을 구성하지 않았다.','',
        '## 검증·보존과 결론의 한계','',
        '- 공통 ExperimentLogger를 W&B disabled로 재사용했다. Provider별 별도 W&B/HF 게시기는 추가하지 않았다.',
        '- 형식·좌표·최종응답/추론 분리·단일 요청 입력·예산·Retry 차단·재개 중복 방지·Strict/Fence Mask gating 테스트12개 통과. Synthetic GT 테스트는 실제 GT를 열지 않았다.',
        '- 독립 PIL ImageChops/getbbox 구현으로 Box·Mask IoU와 Wrong-object를 재계산하고 주요 원시 응답·Mask SHA를 검증했다. Checkpoint·SAM2 소스·설치 패키지는 원격 전후 보존 검사를 수행했다.',
        '- API 키와 인증 Header는 원격으로 보내거나 로그에 저장하지 않았다. 기존 모델·데이터·이전 실험 산출물을 변경하지 않았다.',
        '- 로컬 macOS에서 CUDA를 실행하지 않았다. SAM2 CUDA는 지정된 기존 RTX3090에서만 실행했다. Simulator·Training·Video Tracking·Hole 추정·Graph/Controller·Closed-loop는 실행하지 않았다.',
        '- Production 변경, DINO 삭제, Git Commit, 외부 게시 없음. Stage4A-2 PAUSED_SCOPE_REVIEW와 기존 Stop Stub 유지.',
        '- 모델의 크기·다른 일반 벤치마크 점수로 이 태스크 성능을 대신 판정하지 않는다. 이번 관측은 같은60장·동일Prompt·고정예산과 설정에서의 서비스 경로 비교다.',
        '- DINO 제거 또는 Production 준비 판단은 **INSUFFICIENT_SUPPORT**다. 새 Parent, 희귀 오인식, 초기화 후 Tracking, Hole 위치·정밀도와 실제 제어 성공률은 별도 검증이 필요하다.','',
        '상세 산출물: [비교 지표](./ARTIFACT/model_comparison.json), [비용·지연](./ARTIFACT/grounding_latency_cost.json), [Strict 실패](./ARTIFACT/grounding_failure_cases.csv), [Fence 실패](./ARTIFACT/fence_grounding_failure_cases.csv), [독립 검증](./ARTIFACT/independent_verification.json), [원격 보존](./ARTIFACT/remote_preservation_verification.json).']
    import os
    text='\n'.join(lines)+'\n'
    REPORT.write_text(text.replace('./ARTIFACT/',os.path.relpath(ROOT,REPORT.parent)+'/'))
    (ROOT/REPORT.name).write_text(text.replace('./ARTIFACT/','./'))
    v.update(stage='COMPLETE',completed_utc=now(),report_sha256=sha(REPORT),source_after={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},
             scope_amendment_sha256=sha(ROOT/'scope_amendment.json') if amendment else None)
    write(ROOT/'verification.json',v)
    print('REPORT_COMPLETE',REPORT)

if __name__=='__main__':main()

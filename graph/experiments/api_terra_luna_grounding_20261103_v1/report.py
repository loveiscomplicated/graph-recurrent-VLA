"""Final same-frame comparison of four OpenAI models, with Gemma replay reference."""
from common import *
import importlib.util
from collections import Counter
from datetime import datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def comparison_figure(rows,names):
    lookup={k:{(r['image_id'],r['object']):r for r in rr} for k,rr in rows.items()}
    masks={}
    for root,providers in [(BASELINE,{'openai':'astra'}),(SOL_ROOT,{'openai':'sol','gemma_baseline':'gemma'}),
                           (ROOT,{'terra':'terra','luna':'luna'})]:
        for r in read(root/'sam2/sam2_predictions.json')['rows']:
            if r['provider'] in providers:masks[providers[r['provider']],r['image_id'],r['object']]=root/'sam2'/r['mask_file']
    cases=read(ROOT/'input_manifest.json')['images'][:4]
    canvas=Image.new('RGB',(256*(len(names)+1),294*len(cases)),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
    except OSError:font=ImageFont.load_default()
    for row,c in enumerate(cases):
        rgb=Image.open(c['image_path']).convert('RGB')
        for col,key in enumerate(['rgb']+list(names)):
            im=rgb.convert('RGBA')
            if key!='rgb':
                for obj,color in [('peg',(255,40,40)),('box',(40,100,255))]:
                    r=lookup[key][c['id'],obj];path=masks.get((key,c['id'],obj))
                    if path and r['predicted_box']:
                        overlay=Image.new('RGBA',im.size,color+(0,));overlay.putalpha(Image.open(path).convert('L').point(lambda a:70 if a else 0));im=Image.alpha_composite(im,overlay)
                    d=ImageDraw.Draw(im)
                    if r['predicted_box']:d.rectangle(r['predicted_box'],outline=color+(255,),width=3)
                    if r['GT_visible_box']:d.rectangle(r['GT_visible_box'],outline=(0,210,70,255),width=2)
            x=col*256;y=row*294;canvas.paste(im.convert('RGB').resize((256,256)),(x,y+38))
            title=c['id'].replace('10i_s1_','').replace('_native512','') if key=='rgb' else names[key]
            draw.text((x+3,y+2),title,fill='black',font=font)
            if key!='rgb':
                a=[lookup[key][c['id'],o] for o in ['peg','box']]
                label=(a[0]['api_status']+'; no mask' if a[0]['api_status']!='OK' else
                       'FORMAT_INVALID; no mask' if not a[0]['json_valid'] else
                       'Peg %.2f / Fixture %.2f'%(a[0]['mask_IoU'] or 0,a[1]['mask_IoU'] or 0))
                draw.text((x+3,y+19),label,fill='black',font=font)
    canvas.save(ROOT/'full_image_grounding_comparison.png')

def main():
    v=verify_frozen();assert v['stage'] in ['MASKS_FROZEN','COMPLETE']
    assert read(SOL_ROOT/'verification.json')['stage']=='COMPLETE'
    for group in ['api_files','raw_files','sam2_files']:
        for n,h in v[group].items():assert sha(ROOT/n)==h
    # Reuse already-checked pure aggregation helpers; this imports no API/GT code.
    path=REPO/'graph/experiments/api_sol_kimi_grounding_20261103_v1/report.py'
    spec=importlib.util.spec_from_file_location('sol_metric_helpers',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    names={'astra':'GPT-6 Astra','sol':'GPT-6.1 Sol','terra':'GPT-6.1 Terra','luna':'GPT-6 Luna','gemma':'Gemma4 31B'}
    helper.NAMES=names
    prior=read(BASELINE/'evaluation_rows.json');rows_by_mode={};scenes_by_mode={};summary={};clusters={}
    for mode in ['strict','fence']:
        prefix='fence_' if mode=='fence' else ''
        sol=read(SOL_ROOT/(prefix+'evaluation_rows.json'));current=read(ROOT/(prefix+'evaluation_rows.json'))
        pairs={'astra':(prior,'openai'),'sol':(sol,'openai'),'gemma':(sol,'gemma_baseline'),
               'terra':(current,'terra'),'luna':(current,'luna')}
        rr={k:[r for r in data['rows'] if r['provider']==p] for k,(data,p) in pairs.items()}
        ss={k:[s for s in data['scenes'] if s['provider']==p] for k,(data,p) in pairs.items()}
        rows_by_mode[mode]=rr;scenes_by_mode[mode]=ss
        summary[mode]=helper.tables(rr,ss);clusters[mode]=helper.bootstrap(ss)
        # Independent stage aggregation must agree with this combined-table assembly.
        stages={'astra':(BASELINE,'openai'),'sol':(SOL_ROOT,'openai'),'gemma':(SOL_ROOT,'gemma_baseline'),
                'terra':(ROOT,'terra'),'luna':(ROOT,'luna')}
        for key,(root,provider) in stages.items():
            metrics_name=('' if key=='astra' else prefix)+'grounding_evaluation_metrics.json'
            expected=read(root/metrics_name)['models'][provider]['overall']
            assert summary[mode][key]['joint']==expected['scene']['both_mask08'],(mode,key)
            for obj in ['peg','box']:
                assert summary[mode][key][obj]['mask08']==expected[obj]['mask08'],(mode,key,obj)
    common=set.intersection(*[{r['image_id'] for r in rr if r['api_status']=='OK'} for rr in rows_by_mode['fence'].values()])
    common_summary={mode:helper.tables({k:[r for r in rr if r['image_id'] in common] for k,rr in rows_by_mode[mode].items()},
        {k:[s for s in ss if s['image_id'] in common] for k,ss in scenes_by_mode[mode].items()}) for mode in ['strict','fence']}
    ledger=read(ROOT/'request_ledger.json');cost={};timing={}
    for key in MODELS:
        jobs=[j for j in ledger['jobs'].values() if j['provider']==key];known=[j for j in jobs if j.get('usage_estimated_usd') is not None]
        total=sum(j['accounted_usd'] for j in jobs);success=summary['fence'][key]['joint']['numerator']
        cost[key]=dict(requests=len(jobs),known_usage_n=len(known),usage_estimated_usd=sum(j['usage_estimated_usd'] for j in known),
            unknown_usage_reserved_usd=sum(j['accounted_usd'] for j in jobs if j.get('usage_estimated_usd') is None),
            accounted_usd=total,accounted_per_joint_success_usd=total/success if success else None,
            usage={f:sum((j.get('usage') or {}).get(f) or 0 for j in known) for f in ['input_tokens','output_tokens','reasoning_tokens']},
            status=dict(Counter(j['status'] for j in jobs)),invoice_confirmed=False)
        timing[key]=dict(all_attempted=helper.stats([j['latency_ms']/1000 for j in jobs]),
            normal_OK=helper.stats([j['latency_ms']/1000 for j in jobs if j['status']=='OK']),
            first_seconds=jobs[0]['latency_ms']/1000 if jobs else None,
            subsequent_OK=helper.stats([j['latency_ms']/1000 for j in jobs[1:] if j['status']=='OK']))
    solcost=read(SOL_ROOT/'grounding_latency_cost.json')
    astra_cost=read(BASELINE/'grounding_latency_cost.json')
    # Old run, first-request repeats, Sol/Kimi (incl. canceled reservation), and new work, once each.
    old_accounted=read(BASELINE/'final_interpretation.json')['combined_accounted_usd']
    sol_accounted=solcost['new_accounted_usd'];new_accounted=sum(c['accounted_usd'] for c in cost.values())
    total_accounted=old_accounted+sol_accounted+new_accounted
    assert total_accounted<=20
    jobs=list(ledger['jobs'].values())
    start=min(datetime.fromisoformat(j['started_utc']) for j in jobs);end=max(datetime.fromisoformat(j['finished_utc']) for j in jobs)
    sam=read(ROOT/'sam2/sam2_latency_vram.json')
    oldtiming=read(BASELINE/'observed_timing_summary.json')['no_sleep_OK_latency']
    operations={}
    for key in names:
        success=summary['fence'][key]['joint']['numerator']
        if key in ['astra','gemma']:
            provider='openai' if key=='astra' else 'ollama'
            c=astra_cost['providers'][provider];t=oldtiming[provider]
            amount=c['budget_accounted_usd']
            operations[key]=dict(accounted_usd=amount,known_usage_estimate_usd=c['usage_based_upper_estimate_usd'],
                unknown_reserved_usd=c['missing_usage_reserved_usd'],latency_OK_no_sleep_seconds=t,
                cost_per_joint_success=amount/success if success else None)
        elif key=='sol':
            c=solcost['costs']['sol'];t=solcost['timing']['models']['sol']['normal_OK']
            operations[key]=dict(accounted_usd=c['accounted_usd'],known_usage_estimate_usd=c['usage_estimated_usd'],
                unknown_reserved_usd=c['reserved_unknown_usd'],latency_OK_seconds=t,
                cost_per_joint_success=c['accounted_usd']/success if success else None)
        else:
            c=cost[key];operations[key]=dict(accounted_usd=c['accounted_usd'],known_usage_estimate_usd=c['usage_estimated_usd'],
                unknown_reserved_usd=c['unknown_usage_reserved_usd'],latency_OK_seconds=timing[key]['normal_OK'],
                cost_per_joint_success=c['accounted_usd']/success if success else None)
    write(ROOT/'grounding_latency_cost.json',dict(models=cost,timing=timing,API_wall_seconds=(end-start).total_seconds(),
        started_utc=start.isoformat(),finished_utc=end.isoformat(),SAM2=sam,
        conservative_external_reservation_ceiling=PRIOR_ACCOUNTED_USD,
        actual_accounted=dict(original_run_and_repeats=old_accounted,Sol_and_partial_Kimi=sol_accounted,new_Terra_Luna=new_accounted,total=total_accounted),
        budget_usd=20,invoice_confirmed=False))
    data=dict(models_by_mode=summary,parent_cluster_bootstrap=clusters,common_API_OK_images=len(common),common_API_OK=common_summary,operations=operations,
        Kimi='USER_STOPPED_LATENCY: user withdrew because response time was too long; not an accuracy rejection',
        no_new_parent_generalization=True,DINO_replacement='INSUFFICIENT_SUPPORT',production_changed=False)
    write(ROOT/'openai_family_comparison.json',data)
    comparison_figure(rows_by_mode['fence'],names)
    fmt=helper.fmt
    openai_keys=['astra','sol','terra','luna']
    best_rate=max(summary['fence'][k]['joint']['rate'] for k in openai_keys)
    best=[names[k] for k in openai_keys if summary['fence'][k]['joint']['rate']==best_rate]
    qualifying=[k for k in openai_keys if summary['fence'][k]['joint']['rate']>=.8]
    cheapest=min(qualifying,key=lambda k:operations[k]['accounted_usd']) if qualifying else None
    text=['# Task 10-I Stage 4A-2: Astra · Sol · Terra · Luna Grounding/SAM2 비교','',
        f"동시 Mask 성공률의 전체 할당 분모 점추정은 **{', '.join(best)} ({best_rate*100:.1f}%)**가 가장 높았다. 이는 동일한 알려진60장·고정 설정에서의 관측이며, 대응 CI와 기존 Astra 전송 실패 영향을 함께 해석해야 한다.",
        (f"탐색적 동시 성공80%를 관측한 OpenAI 모델 중 보수 집계 API 비용이 가장 낮은 모델은 **{names[cheapest]}**였다." if cheapest else '동시 Mask 성공80%를 관측한 OpenAI 모델은 없었다.'),'',
        '## 목적과 고정 조건','',
        '전체 RGB 한 장에서 Peg와 Fixture Box를 직접 출력하고, 그 Box만으로 SAM2 초기 Mask를 생성하는 경로를 비교했다. 같은 알려진 Train Parent15개·60장 비교이며 새 Parent 일반화 실험이 아니다.','',
        '- 새 모델: GPT-6.1 Terra `gpt-6.1-terra`, GPT-6 Luna `gpt-6-luna`, 각 최대60회. 기존 Astra와 이번에 완료한 Sol 결과를 재사용했다.',
        '- Kimi K3는 **응답 시간이 너무 오래 걸려 사용자 요청으로 중도 탈락**했다. 정확도 때문에 탈락한 것이 아니며, 전체 모델 정확도 순위에서 제외했다. 완료된 응답·시간·비용과 중단1건의 비용 예약은 별도 보존했다.',
        '- 동일 native512 RGB60장: 15 Parent × F0/F1 × Main/Wrist. F0=t0, F1=t80; Parent0134만 t78. 실패 이미지 대체·추가 Sampling 없음.',
        f"- 동결 Prompt SHA256 `{sha(ROOT/'grounding_prompt.txt')}`. 모든 모델에 동일한 외형 중심 영어 Prompt 원문을 사용했다.",
        '- 한 요청에는 원본 전체 RGB 하나와 Prompt만 전달했다. DINO·GT·Mask·Crop·Point·Parent ID·이전 대화·도구 없음. 요청마다 새 대화다.',
        '- 네 OpenAI 모델 모두 Responses API, detail=original, reasoning=high, 생성 한도5120, Schema 강제 없음. 모델 alias와 실제 응답 ID·Usage·UTC·원시 응답을 기록했다.',
        '- 전체 이미지 기준0–1000 좌표를 .512배로 변환했다. 유한 숫자4개·bool제외·범위·순서 검증, GT 보정·Clipping·Box 확대·응답 재생성 없음.',
        '- Strict JSON과, 응답 전체를 감싼 단일 코드펜스만 제거하는 Fence 모드를 사전에 고정했다. 설명/추론에서 좌표를 검색하거나 수정하지 않았다.',
        '- 각 실험의 예측을 모두 저장·SHA 동결한 뒤 SAM2를 실행하고, 해당 Mask 전체 동결 후에만 GT 배열로 채점했다. 병행 API 실행 때문에 지연은 서비스 부하·네트워크 영향을 포함한다.',
        '- 키는 로컬 .env에만 두었다. 신규 Instance·모델 다운로드·렌더·학습·Tracking·Graph/Controller·Production 변경·Commit·게시 없음. PAUSED_SCOPE_REVIEW 유지.','',
        '공식 문서: [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra), [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna). 입력/출력100만 토큰당 Terra $1/$5, Luna $0.10/$0.50으로 동결했다.','',
        '## 실제 Mask 초기화 성공','',
        'GT Visible Pixel>=16을 주 평가 대상으로 삼았다. Peg60개, Fixture58개, 둘 다 보이는 이미지58장이다. Tiny1–15pixel과 Absent0pixel은 별도 집계했다. 서비스 실패·무효 출력·null은 가시 대상 초기화 실패로 포함한다.','']
    for mode in ['strict','fence']:
        text+=['### '+('엄격한 JSON' if mode=='strict' else '사전 고정한 코드펜스 제거 허용'),'',
            '| Model | Peg Mask IoU>=.8 | Fixture Mask IoU>=.8 | Both Masks IoU>=.8 | Coordinate contract incl. null |',
            '| --- | ---: | ---: | ---: | ---: |']
        for key,name in names.items():
            a=summary[mode][key];text.append(f"| {name} | {fmt(a['peg']['mask08'])} | {fmt(a['box']['mask08'])} | {fmt(a['joint'])} | {fmt(a['coordinate_contract'])} |")
        text+=['']
    text+=['Gemma의 Fence 결과는 기존 응답을 재사용하여 실제 SAM2 Mask를 추가 측정한 것이다. 추가 Gemma API 호출은0회다. 형식 실패와 시각 실패를 구분하며 Strict 결과를 소급 수정하지 않았다.','',
        '## Camera·시점과 Grounding 오차','',
        '| Model (Fence) | Main joint | Wrist joint | F0 joint | F1 joint | Peg Box>=.5 | Fixture Box>=.5 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,name in names.items():
        a=summary['fence'][key];text.append('| '+name+' | '+' | '.join(fmt(a[f]) for f in ['main','wrist','initial','later'])+f" | {fmt(a['peg']['box05'])} | {fmt(a['box']['box05'])} |")
    text+=['','| Model (Fence) | Peg Wrong-object | Fixture Wrong-object | Peg mean Mask IoU | Fixture mean Mask IoU |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key,name in names.items():
        a=summary['fence'][key];text.append(f"| {name} | {fmt(a['peg']['wrong_object'])} | {fmt(a['box']['wrong_object'])} | {a['peg']['mask_IoU']['mean']:.3f} | {a['box']['mask_IoU']['mean']:.3f} |")
    text+=['','Wrong-object는 다른 Actor가 예측 Mask Pixel의50% 이상인 경우이며, 분모는 Tiny를 제외한 실제 비어 있지 않은 Mask다. Robot 일부 오염·낮은 IoU와 구분한다. 상세 Box IoU>=.3/.5/.7, 중심 Pixel 오차, Mask Precision/Recall, Actor 구성·배경 오염, null·미검출·Absent Hallucination과 실제 분모는 JSON/CSV에 저장했다.','',
        '## Parent 단위 대응 비교','',
        '| Model (Fence) | Parent-macro joint | Parent-cluster 95% CI |','| --- | ---: | ---: |']
    for key,name in names.items():
        a=clusters['fence']['absolute'][key];text.append(f"| {name} | {100*a['macro']:.1f}% | {100*a['CI95'][0]:.1f}–{100*a['CI95'][1]:.1f}% |")
    text+=['','| Paired difference | Parent-macro pp | 95% CI pp |','| --- | ---: | ---: |']
    for pair in ['astra_minus_sol','astra_minus_terra','astra_minus_luna','sol_minus_terra','sol_minus_luna','terra_minus_luna']:
        a=clusters['fence']['paired'][pair];text.append(f"| {pair} | {100*a['macro_delta']:+.1f} | {100*a['CI95'][0]:+.1f}–{100*a['CI95'][1]:+.1f} |")
    text+=['','Physical Parent15개를 독립 단위로 사용하고 Seed42·10,000회 Cluster Bootstrap을 계산했다. Camera/시점을 별도 Parent로 세지 않았다. 표본이 작고 이미 알려진 장면이라는 한계가 있다.',
        f"모든 완료 비교 모델의 API가 정상 응답한 공통 이미지 {len(common)}장의 Fence 동시 성공: "+', '.join(name+' '+fmt(common_summary['fence'][key]['joint']) for key,name in names.items())+'.',
        '기존 Astra 실행에는 Mac 절전과 겹친 전송 실패3건이 있다. 전체 분모와 공통 정상응답 집합을 함께 보며 제공자 가용성 차이로 단정하지 않는다.','',
        '## 지연·비용·소요 시간','',
        '| New model | API status | Mean / median / p95 OK latency | First request | Accounted cost | Cost / joint success |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for key in MODELS:
        c=cost[key];t=timing[key];a=t['normal_OK'];display=lambda x:f'{x:.2f}' if x is not None else 'N/A'
        successcost='$%.5f'%c['accounted_per_joint_success_usd'] if c['accounted_per_joint_success_usd'] is not None else 'N/A'
        text.append(f"| {names[key]} | {c['status']} | {display(a['mean'])} / {display(a['median'])} / {display(a['p95'])} s (n={a['n']}) | {display(t['first_seconds'])} s | ${c['accounted_usd']:.5f} | {successcost} |")
    text+=['',f"Terra/Luna 신규 API 경과 {(end-start).total_seconds()/60:.2f}분, 신규 보수 비용 ${new_accounted:.5f}. 이전 실험·첫 이미지 반복·Sol·부분 Kimi까지 중복 없이 합산한 보수 비용 **${total_accounted:.5f}/$20**다. 이 값은 Usage×공식 단가 추정 및 Usage 누락 예약이며 청구 확정값이 아니다. 캐시 할인은 적용하지 않았고 3090 요금은 별도다.",
        f"사전 예약 검증에는 동시 실행 중이던 외부 실험의 최악 예약 상한 ${PRIOR_ACCOUNTED_USD:.5f}를 사용했다. 최종 비용에 그 상한을 실제 지출로 더하지 않았다.",
        '- HTTP E2E를 측정했으며 서버 Vision/Prefill/Decode는 개별 관측하지 못했다. 첫 요청을 Cold라고 단정하지 않는다. 모델별 첫 요청과 나머지 요청 통계는 별도 보존했다.',
        f"- Terra/Luna SAM2는 기존58576054 RTX3090에서 {sam['SAM2_loads']}회 로드, {len(read(ROOT/'sam2/sam2_predictions.json')['rows'])}개 Mask를 생성했다. Load {sam['load_ms']:.1f}ms, Peak allocated {sam.get('peak_allocated_bytes',0)/1024**3:.3f}GiB.",
        '- SAM2.1 Hiera Small 기존 checkpoint SHA를 검증했다. FP32 Weight, BF16 autocast, multimask_output=False, logit>0. 임시 단일 프레임 실행이며 Memory Commit·장기 Tracking 없음.',
        '- VLM은 초기 발견/필요 시 재Grounding 용도다. 이 API 지연으로 Control20Hz 또는 실제 Closed-loop 성공을 주장하지 않는다.','',
        '### 전체 모델의 비용과 정상 응답 지연','',
        '| Model | Usage-based estimate | Unknown-usage reservation | Conservative total | Median OK latency | Total / joint success |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,name in names.items():
        a=operations[key];t=a.get('latency_OK_seconds',a.get('latency_OK_no_sleep_seconds'));median=t.get('median',t.get('median_s'))
        successcost='$%.5f'%a['cost_per_joint_success'] if a['cost_per_joint_success'] is not None else 'N/A'
        text.append(f"| {name} | ${a['known_usage_estimate_usd']:.5f} | ${a['unknown_reserved_usd']:.5f} | ${a['accounted_usd']:.5f} | {median:.2f}s (n={t['n']}) | {successcost} |")
    text+=['','위 표는 모델별 주 평가60건의 API 비용이다. 추가 첫 이미지 반복과 부분 Kimi 비용은 모델별 표에서 제외하고 전체 예산에만 포함했다. Astra/Gemma 지연은 기존 절전 중첩 요청을 제외한 정상 응답 통계이며, 병행 실행 조건 차이가 있어 순수 모델 속도 비교로 단정하지 않는다.','',
        '## 실패 분해와 판단','',
        '| Model (Fence) | Visible object failures | Box IoU>=.7 but Mask<.8 | Wrong-object failures |','| --- | ---: | ---: | ---: |']
    for key,name in names.items():
        fail=[r for r in rows_by_mode['fence'][key] if r['visibility']=='VISIBLE' and not r['mask08']]
        text.append(f"| {name} | {len(fail)} | {sum(r['box_IoU']>=.7 for r in fail)} | {sum(r['wrong_object'] for r in fail)} |")
    text+=['','관측된 Box 성공 후 Mask 실패는 SAM2의 부분 분할 또는 Robot 오염과 구분해서 해석한다. 색 경계·가림·그리퍼 접촉은 가능한 원인 가설이며 이번에 인과적으로 검증하지 않았다. 최종 모델 선택에는 동시 Mask 성공·Wrong-object·응답 안정성·지연·성공당 비용을 함께 사용해야 한다.',
        '새 Parent·희귀 오류·Hole 위치와 정밀도·초기화 후 Tracking·Graph/Controller·실제 제어는 검증하지 않았다. 작은 동일 장면 비교만으로 Production 준비나 DINO 제거를 선언하지 않는다. **DINO_REPLACEMENT: INSUFFICIENT_SUPPORT.**','',
        '## 소규모 비교 그림','',
        '![모델별 실제 Grounding 및 Mask](./ARTIFACT/full_image_grounding_comparison.png)','',
        'Hash 순서 첫4장이다. 빨강=예측 Peg Box, 파랑=예측 Fixture Box, 초록=사후 GT Box. 반투명 색은 실제 실행한 SAM2 초기 Mask이며 제목 아래 숫자는 Mask IoU다. 결과를 보고 성공 사례를 선택하지 않았다.','',
        '## 검증·산출물','',
        '- 공통 ExperimentLogger(W&B disabled)를 재사용했다. Adapter·형식·좌표·입력 격리·예산·Retry/재개 계약 테스트10개 통과. 유료 API를 테스트에서 호출하지 않았다.',
        '- 독립 PIL ImageChops/getbbox 구현으로 Box·Mask IoU와 Wrong-object를 재계산하고 SHA를 검증했다. 원격 checkpoint·SAM2 소스·패키지를 전후 비교했다.',
        '- API 키/인증 Header는 로그·Manifest·원격에 저장하지 않았다. 로컬 CUDA는 미실행이며 실제 CUDA SAM2는 지정된 RTX3090에서만 실행했다.',
        '- 기존 데이터·Checkpoint·이전 결과·PAUSED_SCOPE_REVIEW를 보존했다. Git Commit·외부 게시 없음.','',
        '[가족 모델 비교 지표](./ARTIFACT/openai_family_comparison.json) · [비용·지연](./ARTIFACT/grounding_latency_cost.json) · [Strict 실패](./ARTIFACT/grounding_failure_cases.csv) · [Fence 실패](./ARTIFACT/fence_grounding_failure_cases.csv) · [독립 검증](./ARTIFACT/independent_verification.json) · [원격 보존](./ARTIFACT/remote_preservation_verification.json)',
        '',f"Sol·Gemma 및 Kimi 중단의 전체 기록: [선행 보고서]({os.path.relpath(REPO/'docs/visual-graph-integration/10i_stage4a2_api_sol_kimi_grounding_comparison.md',REPORT.parent)})."]
    body='\n'.join(text)+'\n'
    REPORT.write_text(body.replace('./ARTIFACT/',os.path.relpath(ROOT,REPORT.parent)+'/'))
    output_body=body.replace('./ARTIFACT/','./')
    solreport=REPO/'docs/visual-graph-integration/10i_stage4a2_api_sol_kimi_grounding_comparison.md'
    output_body=output_body.replace(']('+os.path.relpath(solreport,REPORT.parent)+')',']('+os.path.relpath(solreport,ROOT)+')')
    (ROOT/REPORT.name).write_text(output_body)
    v.update(stage='COMPLETE',completed_utc=now(),report_sha256=sha(REPORT),source_after={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},
             aggregation_helper=dict(path=str(path),sha256=sha(path)))
    write(ROOT/'verification.json',v)
    print('OPENAI_FAMILY_REPORT_COMPLETE',REPORT)

if __name__=='__main__':main()

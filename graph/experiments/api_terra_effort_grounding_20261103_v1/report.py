"""Compare saved effort evaluations and frozen prior high; performs no inference."""
from common import *
import importlib.util
from collections import Counter
from datetime import datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ORDER=['none','low','medium','high']


def fmt(r):
    return f"{r['numerator']}/{r['denominator']} ({100*r['rate']:.1f}%)" if r['denominator'] else 'N/A'


def num(x):return f'{x:.2f}' if x is not None else 'N/A'


def make_figure(data):
    inputs=read(ROOT/'input_manifest.json')['images'][:4]
    lookup={p:{(x['image_id'],x['object']):x for x in rows} for p,rows in data.items()}
    masks={}
    for root,provider_map in [(ROOT,{x:x for x in MODELS}),(HIGH_ROOT,{'terra':'high'})]:
        for row in read(root/'sam2/sam2_predictions.json')['rows']:
            if row['provider'] in provider_map:
                masks[(provider_map[row['provider']],row['image_id'],row['object'])]=root/'sam2'/row['mask_file']
    canvas=Image.new('RGB',(256*5,294*len(inputs)),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
    except OSError:font=ImageFont.load_default()
    for y,c in enumerate(inputs):
        rgb=Image.open(c['image_path']).convert('RGBA')
        for col,key in enumerate(['RGB']+ORDER):
            im=rgb.copy()
            if key!='RGB':
                for obj,color in [('peg',(255,40,40)),('box',(40,100,255))]:
                    row=lookup[key][c['id'],obj];mask=masks.get((key,c['id'],obj))
                    if mask and row['predicted_box']:
                        overlay=Image.new('RGBA',im.size,color+(0,));overlay.putalpha(Image.open(mask).convert('L').point(lambda a:70 if a else 0));im=Image.alpha_composite(im,overlay)
                    dr=ImageDraw.Draw(im)
                    if row['predicted_box']:dr.rectangle(row['predicted_box'],outline=color+(255,),width=3)
                    if row['GT_visible_box']:dr.rectangle(row['GT_visible_box'],outline=(0,210,70,255),width=2)
            x=col*256;top=y*294;canvas.paste(im.convert('RGB').resize((256,256)),(x,top+38))
            title=c['id'].replace('10i_s1_','').replace('_native512','') if key=='RGB' else key
            draw.text((x+3,top+2),title,fill='black',font=font)
            if key!='RGB':
                a=[lookup[key][c['id'],o] for o in ['peg','box']]
                label=(a[0]['api_status'] if a[0]['api_status']!='OK' else
                       'INVALID' if not a[0]['json_valid'] else 'Peg %.2f / Fixture %.2f'%(a[0]['mask_IoU'] or 0,a[1]['mask_IoU'] or 0))
                draw.text((x+3,top+19),label,fill='black',font=font)
    canvas.save(ROOT/'effort_grounding_comparison.png')


def main():
    v=verify_frozen();assert v['stage']=='MASKS_FROZEN'
    for group in ['api_files','raw_files','sam2_files']:
        for n,h in v[group].items():assert sha(ROOT/n)==h,n
    helper_path=REPO/'graph/experiments/api_sol_kimi_grounding_20261103_v1/report.py'
    spec=importlib.util.spec_from_file_location('effort_aggregation_helper',helper_path)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.NAMES={x:x for x in ORDER}
    allrows={};allscenes={};summaries={};clusters={}
    for mode in ['strict','fence']:
        prefix='fence_' if mode=='fence' else ''
        current=read(ROOT/(prefix+'evaluation_rows.json'))
        prior=read(HIGH_ROOT/(prefix+'evaluation_rows.json'))
        rows={p:[r for r in current['rows'] if r['provider']==p] for p in MODELS}
        scenes={p:[r for r in current['scenes'] if r['provider']==p] for p in MODELS}
        rows['high']=[r for r in prior['rows'] if r['provider']=='terra']
        scenes['high']=[r for r in prior['scenes'] if r['provider']=='terra']
        assert all(len(rows[p])==120 and len(scenes[p])==60 for p in ORDER)
        summaries[mode]=h.tables(rows,scenes);clusters[mode]=h.bootstrap(scenes)
        for p in MODELS:
            expect=read(ROOT/(prefix+'grounding_evaluation_metrics.json'))['models'][p]['overall']
            assert summaries[mode][p]['joint']==expect['scene']['both_mask08']
        expect=read(HIGH_ROOT/(prefix+'grounding_evaluation_metrics.json'))['models']['terra']['overall']
        assert summaries[mode]['high']['joint']==expect['scene']['both_mask08']
        allrows[mode]=rows;allscenes[mode]=scenes
    ledger=read(ROOT/'request_ledger.json');highjobs=read(HIGH_ROOT/'request_ledger.json')['jobs'].values()
    jobs={p:[j for j in ledger['jobs'].values() if j['provider']==p] for p in MODELS}
    jobs['high']=[j for j in highjobs if j['provider']=='terra']
    timecost={}
    for p in ORDER:
        jj=jobs[p];known=[j for j in jj if j.get('usage_estimated_usd') is not None]
        account=sum(j.get('accounted_usd',j.get('reserved_usd',0)) for j in jj)
        ok=[j['latency_ms']/1000 for j in jj if j.get('status')=='OK' and j.get('latency_ms') is not None]
        completed=[j['latency_ms']/1000 for j in jj if j.get('status') in ['OK','TOKEN_LIMIT'] and j.get('latency_ms') is not None]
        timecost[p]=dict(requests=len(jj),status=dict(Counter(j['status'] for j in jj)),
            reported_tiers=dict(Counter(j.get('served_tier') or ('default' if p=='high' and j.get('status')=='OK' else 'unreported') for j in jj)),
            latency_ok_seconds=h.stats(ok),latency_completed_seconds=h.stats(completed),
            known_usage_estimated_usd=sum(j['usage_estimated_usd'] for j in known),
            unknown_usage_reserved_usd=sum(j.get('accounted_usd',j.get('reserved_usd',0)) for j in jj if j.get('usage_estimated_usd') is None),
            accounted_usd=account,reasoning_tokens=sum((j.get('usage') or {}).get('reasoning_tokens') or 0 for j in known),
            input_tokens=sum((j.get('usage') or {}).get('input_tokens') or 0 for j in known),
            output_tokens=sum((j.get('usage') or {}).get('output_tokens') or 0 for j in known),
            estimate_per_joint_success_usd=account/summaries['strict'][p]['joint']['numerator'] if summaries['strict'][p]['joint']['numerator'] else None)
    paired={}
    for p in ORDER[:-1]:
        z=clusters['strict']['paired'][p+'_minus_high'];paired[p]=z
    total_new=sum(timecost[p]['accounted_usd'] for p in MODELS)
    summary=dict(strict=summaries['strict'],fence=summaries['fence'],bootstrap=clusters,
        time_cost=timecost,paired_vs_high=paired,prior_accounted_usd=PRIOR_ACCOUNTED_USD,
        new_accounted_usd=total_new,aggregate_accounted_usd=PRIOR_ACCOUNTED_USD+total_new,
        statuses=ledger['stopped'],high_reused=True,no_new_parent_generalization=True)
    assert summary['aggregate_accounted_usd']<=20
    write(ROOT/'effort_comparison.json',summary)
    write(ROOT/'effort_latency_cost.json',timecost)
    make_figure(allrows['fence'])
    text=['# Task 10-I Stage 4A-2: Terra Standard reasoning effort 비교','',
          '같은 native512 60장에서 `none`, `low`, `medium`, `high`의 전체 RGB Grounding과 실제 SAM2 초기 Mask를 비교했다. 기존 `high` 60개 응답·Mask·점수는 재사용했고 새로운 `high` API 호출과 SAM2 재실행은 없었다. 표의 결과는 이번 알려진 장면과 고정 설정에서의 관측이다.','',
          '## 고정한 실험 조건','',
          '- Train Parent 15개 × 초기/후기 × Main/Wrist, 총60 native512 PNG. F0=t0, F1=t80, Parent0134만 t78. 미사용 Parent 일반화 실험이 아니다.',
          '- `gpt-6.1-terra` Responses API, Standard (`service_tier=default`), detail=original, max_output_tokens=5120, store=False. 새 요청은 `none`, `low`, `medium` 60장씩 최대180회다. 기존 high는 요청에서 tier를 생략했지만 실제 응답이 default였다.',
          '- 같은 외형 설명 Prompt SHA256 `'+sha(ROOT/'grounding_prompt.txt')+'`. 매 요청은 원본 전체 RGB 하나와 Prompt만 사용하며 GT·DINO·SAM2·Crop·Point·Parent ID·이전 대화·도구를 전달하지 않는다.',
          '- 0–1000 정규화 좌표, x/y에 .512를 곱한 Pixel 좌표. 범위·유한값·순서·bool을 검사하며 GT 보정, Clipping, Box 확대·좌표 수리, Schema 강제, 자동 Retry와 실패 이미지 대체는 없다.',
          '- 원시 응답·예측을 먼저 동결하고 동일 SAM2.1 Hiera Small을 FP32 Weight + BF16 autocast로 실행, multimask_output=False·Logit>0·이미지 Embedding 재사용·독립 Box Prompt. 전체 Mask 동결 후 GT로 평가했다. 추적은 실행하지 않았다.',
          '- 주 지표: Peg·Fixture의 SAM2 Mask IoU가 모두 ≥0.8인 이미지 비율. Visible≥16 pixel이 주 분모이며 Peg 60·Fixture 58·동시 58장이다. Tiny 1–15와 Absent 0은 별도다.','',
          '## 실제 Mask 성공: Strict JSON','',
          '| Effort | Peg | Fixture | 두 물체 동시 | Peg Wrong-object | Fixture Wrong-object | 좌표 계약 (null 허용) |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in ORDER:
        m=summaries['strict'][p]
        text.append(f"| {p} | {fmt(m['peg']['mask08'])} | {fmt(m['box']['mask08'])} | {fmt(m['joint'])} | {fmt(m['peg']['wrong_object'])} | {fmt(m['box']['wrong_object'])} | {fmt(m['coordinate_contract'])} |")
    text+=['','Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask Pixel의 50% 이상인 경우다. 분모는 Tiny를 제외한 비어 있지 않은 예측 Mask이며 Visible Recall 분모와 다르다. 서비스 오류·토큰 한도·무효 출력·null·미시도는 해당 가시 대상 초기화 실패에 포함하며 낮은 IoU를 모두 오인식으로 분류하지 않는다.','',
           '## Fence 허용 결과','',
           '| Effort | Peg | Fixture | 두 물체 동시 | 좌표 계약 (null 허용) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for p in ORDER:
        m=summaries['fence'][p];text.append(f"| {p} | {fmt(m['peg']['mask08'])} | {fmt(m['box']['mask08'])} | {fmt(m['joint'])} | {fmt(m['coordinate_contract'])} |")
    text+=['','Fence는 응답 전체를 감싼 단일 코드펜스만 제거한다. Strict의 형식 실패를 소급 성공으로 변경하지 않고, 설명이나 추론에서 좌표를 검색·수정하지 않는다. 같은 유효 Box의 SAM2 Mask는 한 번만 생성했다.','',
           '## 응답 시간과 비용','',
           '| Effort | 요청 상태 | OK 평균 / 중앙값 / p95 | 완료(OK+TOKEN_LIMIT) 중앙값 / n | Usage 추정 | 미확인 예약 | 총 계상 | 추론 토큰 합계 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in ORDER:
        c=timecost[p];a=c['latency_ok_seconds'];b=c['latency_completed_seconds']
        text.append(f"| {p} | {c['status']} | {num(a['mean'])} / {num(a['median'])} / {num(a['p95'])}초 (n={a['n']}) | {num(b['median'])}초 / {b['n']} | ${c['known_usage_estimated_usd']:.6f} | ${c['unknown_usage_reserved_usd']:.6f} | ${c['accounted_usd']:.6f} | {c['reasoning_tokens']} |")
    text+=['',f"새 요청 보수 계상 **${total_new:.6f}**, 기존 계상 **${PRIOR_ACCOUNTED_USD:.8f}**, 합계 **${summary['aggregate_accounted_usd']:.8f}/$20**. 기존 high 비용은 이전 계상에 이미 포함돼 있으므로 중복 가산하지 않았다. Usage 추정과 미확인 요청 예약은 확정 청구액이 아니며 RTX3090 요금은 별도다. 입력 $1/출력 $5 per million, 캐시 할인 미적용. [공식 Terra 가격·설정](https://developers.openai.com/api/docs/models/gpt-6.1-terra).",
           'HTTP 시간에는 네트워크와 응답 파싱이 포함되고 SAM2는 포함되지 않는다. 요청을 순차 실행했고 high는 이전 시점·Terra/Luna 교대 실행 결과이므로 시간·서비스 부하·추론 토큰 차이를 포함한다. 성공 응답 시간뿐 아니라 TOKEN_LIMIT 완료 시간을 함께 보며 순수 추론 속도나 실시간 제어 성능을 주장하지 않는다.','',
           '## Parent 단위 대응 비교','',
           '| Effort − high | Parent-macro 차이 | Parent-cluster 95% CI |',
           '| --- | ---: | ---: |']
    for p,z in paired.items():text.append(f"| {p} − high | {100*z['macro_delta']:+.2f}%p | {100*z['CI95'][0]:+.2f}~{100*z['CI95'][1]:+.2f}%p |")
    text+=['','물리 Parent 15개를 독립 단위로 사용하고 Seed42·10,000회 Parent-cluster Bootstrap을 적용했다. 차이의 CI에 0이 포함되면 우열이 확정되지 않은 것이며, 이것만으로 동등성을 입증한 것은 아니다. 여러 수준 비교는 탐색적이며 다중 비교 보정은 하지 않았다.','',
           '## 카메라·시점','',
           '| Effort | Main | Wrist | t0 | 후기 t80/t78 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for p in ORDER:
        m=summaries['strict'][p];text.append(f"| {p} | {fmt(m['main'])} | {fmt(m['wrist'])} | {fmt(m['initial'])} | {fmt(m['later'])} |")
    text+=['','## 남은 실패와 해석','',
           '| Effort | Visible 물체 실패 | Box IoU≥0.7이지만 Mask IoU<0.8 | Wrong-object 실패 |',
           '| --- | ---: | ---: | ---: |']
    for p in ORDER:
        fail=[r for r in allrows['strict'][p] if r['visibility']=='VISIBLE' and not r['mask08']]
        text.append(f"| {p} | {len(fail)} | {sum(r['box_IoU']>=.7 for r in fail)} | {sum(r['wrong_object'] for r in fail)} |")
    text+=['','정확한 Box 뒤 부분 분할·오염이 나올 수 있으므로 실패를 모두 VLM 의미 인식 오류로 해석하지 않는다. 기존에 여러 차례 본 15개 Parent의 정적 초기 Mask 비교이며, 새 Parent·초기화 이후 Tracking·Graph·Controller·Closed-loop 결과는 아니다.','',
           '## 고정 이미지 예시','',
           '![동일 이미지 effort별 실제 Mask]('+os.path.relpath(ROOT,REPORT.parent)+'/effort_grounding_comparison.png)','',
           '입력 Manifest의 고정 순서 첫 4장이다. 빨강=예측 Peg Box, 파랑=예측 Fixture Box, 초록=사후 GT Box, 반투명 영역=실제로 생성된 SAM2 Mask. 성공 여부를 보고 이미지를 고르지 않았다.','',
           '## 검증과 근거','',
           '- 공통 ExperimentLogger(W&B disabled), 기존 HTTP Adapter·좌표 Parser·SAM2 Segment·GT 평가·별도 PIL ImageChops 검증을 재사용했다. 기존 high와 새 예측·Mask를 중복 실행하지 않았다.',
           '- 기존 RTX3090 인스턴스에서만 새 SAM2 Mask를 실행했다. API 키는 로컬에서만 사용했고 원격 GPU로 전송하지 않았다. 기존 원격 checkpoint·설치 SAM2 Source·패키지를 전후 대조했다.',
           '- 신규 Simulator Render/Rollout·Tracking·Graph·학습·Controller·Production 변경·Git Commit·외부 게시 없음. PAUSED_SCOPE_REVIEW와 Stop Stub 유지. 로컬 macOS CUDA/MPS 및 Simulator는 실행하지 않았다.','',
           '[비교 JSON](./ARTIFACT/effort_comparison.json) · [시간·비용](./ARTIFACT/effort_latency_cost.json) · [요청 Ledger](./ARTIFACT/request_ledger.json) · [Strict 실패](./ARTIFACT/grounding_failure_cases.csv) · [Fence 실패](./ARTIFACT/fence_grounding_failure_cases.csv) · [독립 검증](./ARTIFACT/independent_verification.json) · [원격 보존](./ARTIFACT/remote_preservation_verification.json) · [기존 high 보고서](10i_stage4a2_api_openai_family_grounding_comparison.md).']
    body='\n'.join(text)+'\n';REPORT.write_text(body.replace('./ARTIFACT/',os.path.relpath(ROOT,REPORT.parent)+'/'))
    v.update(stage='COMPLETE',completed_utc=now(),report_sha256=sha(REPORT),
             aggregation_helper=dict(path=str(helper_path),sha256=sha(helper_path)))
    write(ROOT/'verification.json',v)
    print('EFFORT_REPORT_COMPLETE',REPORT,flush=True)


if __name__=='__main__':main()

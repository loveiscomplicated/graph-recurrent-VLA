"""Rebuild tables/Parent intervals and publication figures from saved raw results."""
from common import *
from collections import defaultdict,Counter
import numpy as np

REPORT=REPO/'docs/visual-graph-integration/10i_resolution_perception_graph_128_512_768_1024_20261103.md'


def rows(name):return read(ROOT/(name+'.json'))['rows']


def select(data,**where):return [r for r in data if all(r.get(k)==v for k,v in where.items())]


def mean(data,key):
    a=[r[key] for r in data if r.get(key) is not None]
    return float(np.mean(a)) if a else None


def rate(data,key):
    a=[r for r in data if r.get(key) is not None]
    return dict(numerator=sum(bool(r[key]) for r in a),denominator=len(a),rate=mean(a,key))


def paired(data,key,baseline,size):
    def identity(r):return tuple(r.get(k) for k in ('scene','camera','repeat','object_id'))
    a={identity(r):r for r in data if r['size']==baseline and r.get(key) is not None}
    b={identity(r):r for r in data if r['size']==size and r.get(key) is not None}
    groups=defaultdict(list)
    for k in a.keys()&b.keys():groups[a[k]['parent']].append(float(b[k][key])-float(a[k][key]))
    values=np.array([np.mean(v) for _,v in sorted(groups.items())])
    if not len(values):return dict(parents=0,paired_cases=0,difference=None,ci95=None)
    rng=np.random.default_rng(SEED);sample=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(parents=len(values),paired_cases=sum(map(len,groups.values())),difference=float(values.mean()),
                ci95=np.quantile(sample,[.025,.975]).tolist())


def number(x,digits=2):return 'N/A' if x is None else f'{x:.{digits}f}'


def percent(x):return 'N/A' if x is None else f'{100*x:.1f}%'


def fraction(v):return f"{v['numerator']}/{v['denominator']} ({percent(v['rate'])})"


def main():
    assert read(ROOT/'independent_verification.json')['status']=='PASS'
    masks=rows('mask_metrics');scenes=rows('scene_metrics');points=rows('graph_point_metrics');poses=rows('graph_pose_metrics');stages=rows('stagewise_metrics')
    jobs=read(ROOT/'api_predictions.json')['rows'];summary=dict(quality={},latency={},paired={},bootstrap=dict(seed=SEED,replicates=10000,unit='physical Parent'))
    for arm in ('actual','gt_box','gt_mask'):
        summary['quality'][arm]={}
        for size in SIZES:
            sm=select(scenes,arm=arm,size=size,cohort='static',both_visible=True)
            pp=select(points,arm=arm,size=size,cohort='static');po=select(poses,arm=arm,size=size,cohort='static')
            summary['quality'][arm][str(size)]=dict(both_mask08=rate(sm,'both_mask08'),
                peg_coverage=mean(select(pp,object_id=0),'surface_coverage_4mm'),box_coverage=mean(select(pp,object_id=1),'surface_coverage_4mm'),
                peg_contamination=mean(select(pp,object_id=0),'contamination_fraction'),box_contamination=mean(select(pp,object_id=1),'contamination_fraction'),
                peg_estimable=rate(po,'peg_estimable'),peg_below3mm=rate(po,'peg_position_below3mm'),
                peg_position_mm=stats([r.get('peg_position_error_mm') for r in po]),peg_angle_deg=stats([r.get('peg_axis_error_deg') for r in po]),
                end_pair_estimable=rate(po,'peg_end_pair_estimable'),hole_estimable=rate(po,'hole_estimable'),
                hole_position_mm=stats([r.get('hole_center_error_mm') for r in po]),
                graph_failures=dict(Counter(r['graph_status'] for r in po)))
    for size in SIZES:
        api=select(jobs,size=size,cohort='static');ok=[j for j in api if j['status']=='OK']
        st=read(ROOT/'timing_sam_static_actual.json')['rows']
        static=select(st,size=size);static=[r for r in static if r['parent']!='seed98202']
        common={}
        for r in st:common.setdefault((r['scene'],r['camera']),{})[r['size']]=r['prompt_count']
        valid={k for k,v in common.items() if len(v)==4 and all(n==2 for n in v.values())}
        matched=[r for r in static if (r['scene'],r['camera']) in valid]
        tr=read(ROOT/'timing_sam_tracking_actual.json')
        oracle_st=read(ROOT/'timing_sam_static_gt_box.json')['rows']
        steady=[r for r in tr['frames'] if r['size']==size and r['timestep']>0 and r['executed']]
        graph=select(rows('timing_graph_actual'),size=size,cohort='static')
        tracking_graph=select(rows('timing_graph_actual'),size=size,cohort='tracking')
        summary['latency'][str(size)]=dict(api_http=stats([j.get('http_ms') for j in api]),api_ok_http=stats([j.get('http_ms') for j in ok]),
            api_prepare=stats([j.get('prepare_ms') for j in api]),api_parse=stats([j.get('parse_ms',0)+j.get('decode_ms',0) for j in api]),
            api_accounted_usd=sum(j.get('accounted_usd',0) for j in select(jobs,size=size)),api_status=dict(Counter(j['status'] for j in api)),
            sam_static=stats([r['total_ms'] for r in static]),sam_static_two_common=stats([r['total_ms'] for r in matched]),
            sam_static_gt_box=stats([r['total_ms'] for r in oracle_st if r['size']==size and r['parent']!='seed98202']),
            sam_tracking=stats([r['total_ms'] for r in steady]),
            graph_static=stats([r.get('graph_ms') for r in graph]),graph_tracking=stats([r.get('graph_ms') for r in tracking_graph]),
            graph_status=dict(Counter(r['status'] for r in graph)),
            static_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in static),
            tracking_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in tr['sequences'] if r['size']==size))
        for arm in ('actual','gt_box'):
            data=select(scenes,arm=arm,cohort='static',both_visible=True)
            summary['paired'][f'{arm}_both_mask08_{size}-128']=paired(data,'both_mask08',128,size)
        for oid in (0,1):
            data=select(points,arm='actual',cohort='static',object_id=oid)
            summary['paired'][f'actual_coverage_o{oid}_{size}-128']=paired(data,'surface_coverage_4mm',128,size)
        summary['paired'][f'actual_peg_below3mm_{size}-128']=paired(select(poses,arm='actual',cohort='static'),'peg_position_below3mm',128,size)
    # Sum actual r0 request RTTs, frozen local timing means and one same-state dual-camera Graph.
    initial=[];track=[];st=read(ROOT/'timing_sam_static_actual.json')['rows'];tr=read(ROOT/'timing_sam_tracking_actual.json')['frames']
    for g in rows('timing_graph_actual'):
        if g['status']!='COMPLETE':continue
        local=g['graph_ms']+g.get('fk_ms',0)+g.get('input_join_ms',0)
        if g['cohort']!='tracking':
            aa=select(jobs,scene=g['scene'],size=g['size'],repeat=0)
            sam=select(st,scene=g['scene'],size=g['size'])
            if len(aa)!=2 or not sam:continue
            sam_ms=sum(mean(select(sam,camera=c),'total_ms') for c in CAMERAS)
            terra_ms=sum(a.get('http_ms',0)+a.get('prepare_ms',0)+a.get('decode_ms',0)+a.get('parse_ms',0) for a in aa)
            initial.append(dict(scene=g['scene'],parent=g['parent'],cohort=g['cohort'],size=g['size'],
                serial_sum_ms=terra_ms+sam_ms+local,terra_pair_ms=terra_ms,sam_pair_ms=sam_ms,graph_plus_fk_join_ms=local))
        elif g['timestep']>0:
            sam=select(tr,parent=g['parent'],size=g['size'],timestep=g['timestep'])
            sam_ms=sum(mean(select(sam,camera=c),'total_ms') for c in CAMERAS)
            track.append(dict(scene=g['scene'],parent=g['parent'],size=g['size'],serial_sum_ms=sam_ms+local))
    summary['serial_sums']=dict(static=initial,tracking=track)
    csvwrite(ROOT/'serial_processing_sums.csv',initial+track)
    write(ROOT/'summary.json',summary)
    make_figures(summary,stages)
    text=['# Task 10-I: 128·512·768·1024 해상도별 인식·Graph 정확도와 처리 시간','',
          '**실측 완료.** 아래 결과는 고정한 알려진 장면에서의 대응 비교다. 미사용 장면 일반화·삽입 성공·온라인 제어 주기를 검증한 결과가 아니다.','',
          '## 1. 실험 조건과 실행 경계','',
          '- 정적: 알려진 Train Parent15개×초기/후기×두 카메라=60장/해상도. Parent0134 후기는 t78, 나머지는 t80이다. 별도로 seed98202/t0 두 카메라를 포함했다.',
          '- 추적: 기존0531/0275/0197/0305의 두 카메라, t0..20. 첫 프레임 Grounding만 사용하고 재탐지·재앵커는 하지 않았다.',
          '- 네 해상도의 RGB-D를 같은 물리 상태·카메라 pose/FOV에서 직접 렌더링했다. CPU PhysX + CPU SAPIEN readback을 사용했다. 이전 seed98202 CUDA-rendered 미리보기와 renderer가 다르며, 이번 네 해상도끼리는 같은 backend다.',
          '- Terra low/Standard/detail=original, 같은 외형 Prompt·출력한도5120·strict JSON·좌표0..1000. 박스 확대·좌표 수리·자동 재시도·실패 이미지 대체는 없다.',
          '- SAM2.1 Hiera Small, FP32 weights/BF16 autocast/TF32, 내부1024, batch1. 원본 크기 mask를 CPU로 반환한다. 추적은 horizon=t+1, JPEG quality100/subsampling0다.',
          '- Graph는 기존 Stage4A2 CurrentGeometry C3/SF, PCA5,3mm 대표 측정점 Dedup, 물체당128점, Graph21/Edge4다. Wrist Box는 기존대로 제외한다. Pixel 단위 이웃 크기와 모든 threshold를 재조정하지 않았다.',
          '- 실제 API→SAM2→Graph와 고정 기하 추정값을 먼저 동결한 뒤, GT 박스/GT 마스크 대조와 채점을 실행했다. GT 박스는1024에서 한 번 정규화하며 추적에서는 t0만 사용한다.','',
          '정상 가시성 분모는1024 Actor Pixel≥1024(128에서16pixel 상당)로 고정했다. Tiny/Absent는 별도다. 표면 Coverage의 공통 분모는 같은 상태의1024 가시 GT Depth이며, Peg는 두 카메라·Box는 Base만 사용한다. 전체/숨겨진 Mesh 표면 Coverage가 아니다.4mm 기준과 경계 허용1px@128을 유지했다.','',
          '## 2. 실제 경로: 정확도와 시간','',
          '| 원본 크기 | 두 물체 Mask IoU≥.8 | Peg/Box Coverage | Peg 축 추정 가능 | Peg 위치<3mm | Terra HTTP 중앙값/p95 ms | SAM2 정적 평균 ms | Graph 평균 ms |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for size in SIZES:
        q=summary['quality']['actual'][str(size)];t=summary['latency'][str(size)]
        text.append(f"| {size} | {fraction(q['both_mask08'])} | {number(q['peg_coverage'],3)}/{number(q['box_coverage'],3)} | {fraction(q['peg_estimable'])} | {fraction(q['peg_below3mm'])} | {number(t['api_http']['median'])}/{number(t['api_http']['p95'])} | {number(t['sam_static']['mean'])} | {number(t['graph_static']['mean'])} |")
    text+=['', 'Terra/SAM2는 **한 카메라에서 두 물체**를 처리하는 단위다. Graph는 **동일 시점 두 카메라를 결합한 한 Graph**다. Graph 시간은 별도 단독 실행의 일반 builder 측정이며, 병렬 품질 평가·GT 채점·파일 I/O·기하 추정기 실행은 포함하지 않는다. 입력 결합·Robot FK는 별도 원자료에 있다. API HTTP에는 네트워크·서버 대기와 요청 직렬화가 포함되며 순수 모델 kernel 시간이 아니다.','',
          '![정확도와 지연 비교](../../outputs/task10i/stage4a2/'+NAME+'/resolution_quality_latency.png)','',
          '| 크기 | 정상 Peg 위치 오차 평균/p95 mm | 축 각도 평균 ° | 양 끝 추정 가능 | Hole 추정 가능 | Graph 실행 상태 |','|---|---:|---:|---:|---:|---|']
    for size in SIZES:
        q=summary['quality']['actual'][str(size)]
        text.append(f"| {size} | {number(q['peg_position_mm']['mean'])}/{number(q['peg_position_mm']['p95'])} | {number(q['peg_angle_deg']['mean'])} | {fraction(q['end_pair_estimable'])} | {fraction(q['hole_estimable'])} | {q['graph_failures']} |")
    text+=['','오차 평균은 추정 가능한 사례에 한정된다. 전체 분모의 추정 가능률·<3mm 비율을 함께 보며, 미추정은 N/A로 남겼다. 삽입 Tip 의미는 기존 Graph에서 미해결이므로 상대 정렬 오차를 만들어내지 않았다.','',
          '## 3. 정답 대조와 단계별 정보 보존','',
          '| 경로 | 크기 | 두 물체 Mask 성공 | Peg/Box Coverage | Peg 위치<3mm | Hole 추정 가능 |','|---|---:|---:|---:|---:|---:|']
    for arm in ('actual','gt_box','gt_mask'):
        for size in SIZES:
            q=summary['quality'][arm][str(size)]
            text.append(f"| {arm} | {size} | {fraction(q['both_mask08']) if arm!='gt_mask' else 'GT 입력'} | {number(q['peg_coverage'],3)}/{number(q['box_coverage'],3)} | {fraction(q['peg_below3mm'])} | {fraction(q['hole_estimable'])} |")
    text+=['', '세 경로의 차이는 원인 진단용이다. 오류를 단순 가산 분해하거나 GT 입력을 실제 인식 성능으로 해석하지 않는다. 밀집 Raw/Filtered 단계에는 표면·경계·끝/rim 보존량을 적용하고, 고정 Pose probe는 최종 capped Graph에만 적용했다.','',
          '![Graph 단계별 공통 표면 Coverage](../../outputs/task10i/stage4a2/'+NAME+'/stagewise_coverage.png)','',
          'Robot/Uncertain/Overlap 제거와 Wrist gate → PCA5/기하 근거 → Shape-First → Cross-view →3mm Dedup → FPS128 → 최종 노드/엣지 순서다. Packet 내부 component/Raw EDE 계산도 실제 호출에 포함되어 있어 별도로 계측했다. 대표점 수 감소와 병합 Source Pixel 삭제는 구분한다. 엣지는 GT Mesh에서4mm 이상 벗어나는 내부 연결을 진단하며 연결 성분 수를 정답 topology 정확도로 부르지 않는다.','',
          '## 4. 추적·반복·시간 해석','',
          '| 크기 | SAM2 t1..20 평균/중앙값/p95 ms | Graph 추적 평균/p95 ms | 정적 공통 두-prompt / 고정 GT-box 평균 ms | Torch GPU peak 정적/추적 GiB |','|---|---:|---:|---:|---:|']
    for size in SIZES:
        t=summary['latency'][str(size)];s=t['sam_tracking'];g=t['graph_tracking']
        text.append(f"| {size} | {number(s['mean'])}/{number(s['median'])}/{number(s['p95'])} | {number(g['mean'])}/{number(g['p95'])} | {number(t['sam_static_two_common']['mean'])}/{number(t['sam_static_gt_box']['mean'])} | {number(t['static_peak_allocated_bytes']/2**30,3)}/{number(t['tracking_peak_allocated_bytes']/2**30,3)} |")
    text+=['', 'SAM2 정적8회·추적4회 반복은 같은 첫 Terra 응답을 고정하고 해상도 순서를 균형 배치했다. 정적 해상도별3회, 추적 해상도별 전체 시퀀스1회를 워밍업에서 제외했다. 실제 prompt 수가 줄어든 실패를 속도 개선으로 해석하지 않도록 네 해상도 모두 두 prompt가 있는 공통 조건을 별도 제시한다.','',
          'Tracking의 파일 preload/첫 encoding·prompt 초기화·t0는 steady 시간에서 분리했다. 전체 시퀀스를 CPU에 미리 읽은 실험이므로 온라인 입력 취득·제어 deadline 검증이 아니다. 모듈 합계 CSV는 두 카메라의 API/마스크와 같은 상태 Graph를 더한 **직렬 처리 비용 합계**이며 실제 end-to-end wall time이 아니다.','',
          'Graph 세부 프로파일은 사전에 선택한 첫3 Parent의 두 시점과 seed98202(7상태×4크기)를 단독 실행했다. 일반 builder와 tensor 일치를 검사하고 계측 시간 차이를 보존했다. 전체 품질 평가의 병렬 trace 시간은 공정한 지연 결과와 섞지 않았다.','',
          '## 5. 대응 차이·검증·산출물','',
          '| 실제 경로,128 대비 | 차이 | Parent-cluster95% CI | Parent / 대응 사례 |','|---|---:|---:|---:|']
    for size in SIZES[1:]:
        for label,key,scale in [('두 물체 Mask 성공',f'actual_both_mask08_{size}-128',100),('Peg Coverage',f'actual_coverage_o0_{size}-128',1),('Box Coverage',f'actual_coverage_o1_{size}-128',1),('Peg 위치<3mm',f'actual_peg_below3mm_{size}-128',100)]:
            v=summary['paired'][key];ci=v['ci95'];unit='%p' if scale==100 else ''
            text.append(f"| {size} {label} | {number(v['difference']*scale if v['difference'] is not None else None,3)}{unit} | {number(ci[0]*scale,3) if ci else 'N/A'} ~ {number(ci[1]*scale,3) if ci else 'N/A'} | {v['parents']} / {v['paired_cases']} |")
    freeze=read(ROOT/'api_freeze.json');verify=read(ROOT/'independent_verification.json')
    text+=['', 'Parent를 독립 단위로 Seed42·10,000회 대응 bootstrap을 사용했다. CI가0을 포함하면 우열 미확정이며 동등성 증명이 아니다. 다중 비교 보정 없는 탐색적 결과다.','',
          f"API 요청 {freeze['requested']}/840, 신규 계상 **${freeze['accounted_usd']:.6f}/$20**. 실제 Usage 추정과 미확인 예약을 포함하며 확정 청구액이 아니다. 캐시 할인 미적용 계상이며 GPU 인스턴스 비용은 별도다. API 키는 로컬에만 두었다. [OpenAI 공식 모델·가격 문서](https://developers.openai.com/api/docs/models/gpt-6.1-terra).",'',
          '공통 ExperimentLogger를 W&B disabled로 재사용했다. 기존 Raw/GT·체크포인트·핵심 Graph 소스와 SAM2 설치 소스를 보존했고, 새 패키지·학습·policy rollout·외부 게시·공통 API 변경은 없다. CUDA SAM2와 CPU PhysX/SAPIEN 및 CPU Graph는58576054에서 실행했다. macOS는 API와 분석/그림·검증을 수행했으며 로컬 CUDA/MPS/Simulator 실행으로 주장하지 않는다.','',
          f"독립 검증: `{verify['status']}`. 세부 확인 범위와 미지원 항목은 verification JSON에 보존했다. 원본 프로토콜과 실행 전 코드 추가/계측 준비 변경은 protocol_history에 남겼다.",'',
          '- [요약 JSON](../../outputs/task10i/stage4a2/'+NAME+'/summary.json) · [독립 검증](../../outputs/task10i/stage4a2/'+NAME+'/independent_verification.json)',
          '- [마스크 원자료 CSV](../../outputs/task10i/stage4a2/'+NAME+'/mask_metrics.csv) · [Graph 표면 CSV](../../outputs/task10i/stage4a2/'+NAME+'/graph_point_metrics.csv) · [기하 정밀도 CSV](../../outputs/task10i/stage4a2/'+NAME+'/graph_pose_metrics.csv)',
          '- [단계별 Coverage/Source CSV](../../outputs/task10i/stage4a2/'+NAME+'/stagewise_metrics.csv) · [Graph 시간 CSV](../../outputs/task10i/stage4a2/'+NAME+'/timing_graph_actual.csv) · [직렬 비용 합계](../../outputs/task10i/stage4a2/'+NAME+'/serial_processing_sums.csv)',
          '- [실험 코드/실행 계약](../../graph/experiments/'+NAME+'/README.md)','',
          '## seed98202: 별도 사례','',
          '![seed98202 네 해상도 비교](../../outputs/task10i/stage4a2/'+NAME+'/seed98202_comparison.png)','',
          'seed98202는 이미 관찰한 한 사례다.15 Parent 주 통계에 섞거나 독립 일반화 근거로 사용하지 않는다. 각 해상도의 실제 예측·최종 점·엣지를 동일 world 좌표 범위로 표시했다.','']
    text+=['| 크기 | Base Peg/Box IoU | Hand Peg/Box IoU | Peg 위치 오차 mm / 추정 가능 |','|---|---:|---:|---:|']
    for size in SIZES:
        vals=[]
        for c in CAMERAS:
            mm=select(masks,arm='actual',cohort='seed98202',camera=c,size=size)
            vals.append('/'.join(number(mean(select(mm,object_id=oid),'mask_iou'),3) for oid in (0,1)))
        pp=select(poses,arm='actual',cohort='seed98202',size=size)
        text.append(f"| {size} | {vals[0]} | {vals[1]} | {number(mean(pp,'peg_position_error_mm'))} / {fraction(rate(pp,'peg_estimable'))} |")
    rates=[summary['quality']['actual'][str(s)]['both_mask08']['rate'] for s in SIZES]
    monotone=all(a is not None and b is not None and b>=a for a,b in zip(rates,rates[1:]))
    lead=('이번 표본의 정적 Mask 성공률 점추정은 해상도가 높아질수록 같거나 높았다.' if monotone else
          '이번 표본에서는 해상도를 올릴 때마다 정적 Mask 성공률이 좋아지지는 않았다.')
    text[4:4]=[lead+' 128→512→768→1024의 두 물체 동시 성공률은 '+ ' → '.join(percent(v) for v in rates)+'다. '
        +'Graph 표면 보존과 기하 정밀도는 아래 별도 지표로 판단한다.','']
    artifact_prefix='../../'+ROOT.relative_to(REPO).as_posix()+'/'
    body='\n'.join(text).replace('../../outputs/task10i/stage4a2/'+NAME+'/',artifact_prefix)
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(body)
    print('REPORT',REPORT,flush=True)


def make_figures(summary,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=np.arange(4)
    for arm,label in [('actual','Terra -> SAM2'),('gt_box','GT box -> SAM2')]:
        axs[0,0].plot(x,[summary['quality'][arm][str(s)]['both_mask08']['rate'] for s in SIZES],'-o',label=label)
    axs[0,0].set(title='Both visible masks IoU >=0.8',ylabel='success fraction',ylim=(0,1));axs[0,0].legend()
    for oid,label in [('peg_coverage','Peg'),('box_coverage','Box')]:
        axs[0,1].plot(x,[summary['quality']['actual'][str(s)][oid] for s in SIZES],'-o',label=label)
    axs[0,1].set(title='Final Graph: common visible1024 coverage',ylabel='fraction within4mm',ylim=(0,1));axs[0,1].legend()
    for key,label in [('api_http','Terra HTTP/view'),('sam_static','SAM2 image/view'),('graph_static','Graph/two views')]:
        axs[1,0].plot(x,[summary['latency'][str(s)][key]['mean'] for s in SIZES],'-o',label=label)
    axs[1,0].set(title='Independent module wall time',ylabel='ms (log scale)',yscale='log');axs[1,0].legend()
    for arm,label in [('actual','actual'),('gt_box','GT box'),('gt_mask','GT mask')]:
        axs[1,1].plot(x,[summary['quality'][arm][str(s)]['peg_below3mm']['rate'] for s in SIZES],'-o',label=label)
    axs[1,1].set(title='Peg axis position <3mm (all planned cases)',ylabel='fraction',ylim=(0,1));axs[1,1].legend()
    for ax in axs.flat:ax.set_xticks(x,[str(s) for s in SIZES]);ax.set_xlabel('native image size');ax.grid(alpha=.2)
    fig.savefig(ROOT/'resolution_quality_latency.png',dpi=160);plt.close(fig)
    order=('raw','filtered','shape_first','cross_view','pre_budget','final')
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for oid,ax in enumerate(axs):
        for size in SIZES:
            data=select(stages,arm='actual',cohort='static',size=size,object_id=oid)
            ax.plot(range(6),[mean(select(data,stage=s),'surface_coverage_4mm') for s in order],'-o',label=str(size))
        ax.set_xticks(range(6),order,rotation=25);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(title='native size')
        ax.set_title(('Peg','Box')[oid]+' / fixed1024 visible reference');ax.set_ylabel('correct-point coverage within4mm')
    fig.savefig(ROOT/'stagewise_coverage.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

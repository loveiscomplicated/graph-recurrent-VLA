"""Full diagnostic tables, rebuilt only from saved per-case rows."""
from common import *
from collections import Counter
import numpy as np


def append_details(text, summary, masks, points, poses):
    from report import select,mean,rate,number,percent,fraction
    base='../../'+ROOT.relative_to(REPO).as_posix()+'/'
    text += ['## 6. Grounding·마스크·관측점 상세','',
        '| 크기 | 물체 | Box IoU | 검출 IoU≥.5 | 중심 오차/영상 대각선 | NULL/무효 | 다른 물체 Box 오인식 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for size in SIZES:
        for oid in (0,1):
            data=select(masks,arm='actual',cohort='static',size=size,object_id=oid,visibility='VISIBLE')
            n=len(data);missing=sum(r['box_status']=='NULL' for r in data);invalid=sum(r['box_status'] not in ('VALID','NULL') for r in data)
            text.append(f"| {size} | {('Peg','Box')[oid]} | {number(mean(data,'box_iou'),3)} | {fraction(rate(data,'box_detected'))} | {number(mean(data,'center_error_image_diagonal'),4)} | {missing}/{n}, {invalid}/{n} | {fraction(rate(data,'box_wrong_object'))} |")
    text += ['', '정상 가시성의 고정 관측 분모다. Box 오인식은 자기 물체 IoU<.5이며 다른 물체 IoU≥.5인 경우로 한정한다. 중심 오차는 유효 예측과 GT 박스가 모두 있는 경우다. API/JSON/좌표 무효·NULL은 서로 구분하고 검출 실패에 포함한다.','',
        '| 크기 | 물체 | Mask IoU / Precision / Recall / Boundary F1 | 정상 객체 관측 수 |',
        '|---|---|---:|---:|']
    for size in SIZES:
        for oid in (0,1):
            data=select(masks,arm='actual',cohort='static',size=size,object_id=oid,visibility='VISIBLE')
            text.append(f"| {size} | {('Peg','Box')[oid]} | {' / '.join(number(mean(data,k),3) for k in ('mask_iou','precision','recall','boundary_f1'))} | {len(data)} |")
    text += ['', '빈 예측의 IoU·Recall·경계 F1은0이며 Precision은 N/A다. 경계 허용 거리는 원본 해상도/128 pixel이다.','',
        '| 크기 | 가시성 | 객체 관측 수 | 예측 Mask 존재 | 유효 예측 Box 존재 | 평균 Mask IoU |',
        '|---|---|---:|---:|---:|---:|']
    for size in SIZES:
        for visibility in ('TINY_VISIBLE','ABSENT'):
            data=select(masks,arm='actual',cohort='static',size=size,visibility=visibility);n=len(data)
            text.append(f"| {size} | {visibility} | {n} | {sum(r['mask_pixels']>0 for r in data)}/{n} | {sum(r['box_status']=='VALID' for r in data)}/{n} | {number(mean(data,'mask_iou'),3)} |")
    text += ['', '부재 객체의 Box/Mask 생성은 false positive다. 작은 객체와 정상 객체를 합쳐 분모를 바꾸지 않았다.','',
        '| 크기 | 물체 | 정상점 / 오염점 비율 | 가시 표면 경계 Coverage | 끝−/끝+ 또는 rim Coverage | 표면 거리 평균/p95 mm |',
        '|---|---|---:|---:|---:|---:|']
    for size in SIZES:
        for oid in (0,1):
            data=select(points,arm='actual',cohort='static',size=size,object_id=oid)
            contamination=mean(data,'contamination_fraction')
            region='/'.join(number(mean(data,k),3) for k in (('end_minus_coverage_4mm','end_plus_coverage_4mm') if oid==0 else ('rim_coverage_4mm',)))
            text.append(f"| {size} | {('Peg','Box')[oid]} | {number(1-contamination if contamination is not None else None,3)} / {number(contamination,3)} | {number(mean(data,'boundary_coverage_4mm'),3)} | {region} | {number(mean(data,'mesh_distance_mean_mm'))}/{number(mean(data,'mesh_distance_p95_mm'))} |")
    text += ['', '비율 평균은 점이 있는 Graph, Coverage 평균은 실행 성공 Graph에 한정한다. p95 열은 사례별 p95의 평균이며 모든 점을 합친 p95가 아니다. 같은 상태의1024 GT 가시 Depth pixel을 분모로 삼으므로 표면적 균일 가중치는 아니다. GT Mesh까지의 거리는 숨겨진 표면도 포함하므로 가시 표면까지 거리와 별도 CSV 열이다.','',
        '| 크기 | 물체 | 최종 점 수 | 내부 엣지 수 | 표면 이탈 엣지 비율 | 연결 성분 수 | 법선 오차 ° |',
        '|---|---|---:|---:|---:|---:|---:|']
    for size in SIZES:
        for oid in (0,1):
            data=select(points,arm='actual',cohort='static',size=size,object_id=oid)
            text.append(f"| {size} | {('Peg','Box')[oid]} | {number(mean(data,'point_count'))} | {number(mean(data,'intra_edges'))} | {percent(mean(data,'surface_bridge_violation_fraction'))} | {number(mean(data,'connected_components'))} | {number(mean(data,'normal_error_mean_deg'))} |")
    text += ['', '내부 엣지는 중복 방향을 제거했다. 1/4·1/2·3/4 지점 중 하나가 물체 Mesh에서4mm 이상 떨어지면 표면 이탈로 센다. 노드·엣지 수와 성분 수는 구조 진단이며 정답 topology와의 일치율이 아니다.','',
        '| 크기 | 양 끝 최대 오차 평균/p95 mm (n) | Hole 위치 평균/p95 mm (n) | Hole 축 각도 평균 ° (n) |',
        '|---|---:|---:|---:|']
    for size in SIZES:
        data=select(poses,arm='actual',cohort='static',size=size)
        cells=[]
        for key in ('peg_max_end_error_mm','hole_center_error_mm','hole_axis_error_deg'):
            s=stats([r.get(key) for r in data]);cells.append(f"{number(s['mean'])}"+(f"/{number(s['p95'])}" if key!='hole_axis_error_deg' else '')+f" ({s['n']})")
        text.append(f"| {size} | {' | '.join(cells)} |")
    text += ['', '각 오차는 해당 추정 성공 사례만 계산하며 전체 추정 가능률은 앞 표에 제시했다.','',
        '## 7. 추적 품질과 최초 보존량 감소','',
        '| 경로 | 크기 | 정상 객체-프레임 수 | 평균 IoU | IoU≥.8 비율 | t20−t0 IoU (대응 수) | 초기 성공→t20 실패 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in read(ROOT/'tracking_quality_summary.json')['rows']:
        text.append(f"| {r['arm']} | {r['size']} | {r['visible_object_frames']} | {number(r['mean_mask_iou'],3)} | {percent(r['visible_mask08'])} | {number(r['paired_endpoint_iou_delta'],3)} ({r['endpoint_pairs']}) | {r['lost_good_endpoints']}/{r['initial_good_endpoints']} ({percent(r['lost_good_endpoint_rate'])}) |")
    text += ['', '같은 시퀀스의 반복·객체·프레임은 독립 장면으로 세지 않는다. t0/t20 모두 정상 가시성인 대응만 품질 변화에 포함한다. 전체 프레임의 IoU·Robot/다른 물체 오염률은 원자료에 있으며, 소실은 초기 IoU≥.8에서 t20<.8로 정의했다.','',
        '최초 정보 손실 진단은 각 물체에서 native GT-mask Raw 대비 실제 Raw Coverage가5%p 넘게 부족한지, 이후 어느 단계에서 처음5%p 넘게 감소하는지 표시한다. 이 임계값은 설명용이며 유일한 원인이나 추정 실패의 증명이 아니다.','',
        f'- [최초 보존량 감소와 단계별 점 수]({base}first_information_loss.csv) · [Mask 실패 사례]({base}mask_failure_cases.csv) · [기하 추정 실패/5mm 이상 사례]({base}pose_failure_cases.csv)',
        f'- [추적 품질 집계]({base}tracking_quality_summary.csv) · [프레임별 전체 Mask 원자료]({base}mask_metrics.csv)','',
        '## 8. 시간 분해·메모리·사용량','',
        '| 크기 | 모듈 | 평균 / 중앙값 / p95 ms | 유효 측정 수 |',
        '|---|---|---:|---:|']
    for size in SIZES:
        t=summary['latency'][str(size)]
        for key,label in (('api_prepare','Terra 인코딩·준비'),('api_http','Terra HTTP'),('api_parse','Terra 디코딩·파싱'),('sam_static','SAM2 정적'),('graph_static','Graph 정적')):
            s=t[key];text.append(f"| {size} | {label} | {' / '.join(number(s[k]) for k in ('mean','median','p95'))} | {s['n']} |")
    text += ['', '| 크기 | Graph 입력 결합 / Robot FK / 파일 decode 평균 ms | Graph 프로세스 RSS peak 최대 GiB | 성공/전체 native Graph 측정 |',
        '|---|---:|---:|---:|']
    graph=read(ROOT/'timing_graph_actual.json')['rows']
    for size in SIZES:
        data=select(graph,size=size,cohort='static');ok=select(data,status='COMPLETE')
        peak=max((r.get('host_peak_rss_bytes',0) for r in ok),default=0)
        text.append(f"| {size} | {' / '.join(number(mean(ok,k)) for k in ('input_join_ms','fk_ms','file_read_decode_ms'))} | {number(peak/2**30,3)} | {len(ok)}/{len(data)} |")
    text += ['', 'RSS는 각 build 직전 해당 Linux 프로세스의 high-water를 초기화한 뒤 측정했다. 로드된 Python/Torch와 Graph 사본·해시·고정 기하 추정기를 포함하는 프로세스 메모리 상한 관측이며 순수 Graph 할당량이 아니다. 실패 사례는 시간 평균에서 제외하고 상태를 보존한다. SAM2 메모리는 앞의 Torch allocated peak이며 CUDA 전체 예약량·GPU 전체 VRAM과 다르다.','',
        '| 크기 | SAM2 정적 상세 단계 | 평균 ms |',
        '|---|---|---:|']
    static=read(ROOT/'timing_sam_static_actual.json')
    labels=('preprocess','image_encoder','prompt_preparation','prompt_encoder','mask_decoder','restore_original_size')
    for size in SIZES:
        ds=[r for r in static['details'] if r['size']==size and r['parent']!='seed98202']
        for label in labels:
            text.append(f"| {size} | {label} | {number(np.mean([sum(s['ms'] for s in r['detail'] if s['stage']==label) for r in ds]) if ds else None)} |")
        for label in ('transfer_feature_pack_residual_ms','cpu_output_residual_ms'):
            text.append(f"| {size} | {label} | {number(mean(ds,label))} |")
    text += ['', '상세 단계는 별도 동기화 계측이다. GPU mask prediction은 prompt/decoder/복원을 포함하므로 이를 다시 합산하지 않았다. CPU 출력은 public predict 경로의 GPU 처리·prompt 준비를 뺀 잔여 시간으로서 threshold·Python 비용도 포함한다. encoder의 잔여는 전송·feature pack 등이 함께 있다. 순수 DMA 시간으로 해석하지 않는다.','',
        '| 크기 | 추적 CPU preload+첫 encode / prompt init / t0 평균 ms | JPEG 파일 I/O / CPU preload 상세 평균 ms |',
        '|---|---:|---:|']
    tr=read(ROOT/'timing_sam_tracking_actual.json')
    for size in SIZES:
        seq=select(tr['sequences'],size=size);first=select(tr['frames'],size=size,timestep=0)
        detail=[r['timing']['file_io_profile'] for r in tr['details'] if r['size']==size]
        text.append(f"| {size} | {number(mean(seq,'video_load_first_encode_ms'))} / {number(mean(seq,'prompt_init_ms'))} / {number(mean(first,'total_ms'))} | {number(mean(detail,'file_io_ms'))} / {number(mean(detail,'cpu_preload_decode_resize_normalize_ms'))} |")
    text += ['', 'JPEG I/O는 별도 상세 실행에서 원래 open/read/seek/close 호출을 관측한 합이다. OS page cache를 비우지 않았다. CPU preload는 JPEG decode·1024 resize·normalization까지 포함하며 I/O를 포함한다. 이 둘은 중첩되므로 합산하지 않는다. 모델 load는 각 timing JSON의 identity.load_ms에 별도로 기록했다.','']
    graph_profile(text,base)
    jobs=read(ROOT/'api_predictions.json')['rows']
    text += ['', '| 크기 | 입력 / 출력 / 추론 / 캐시 토큰 합계 | Usage 계상 USD | 응답 모델·tier |','|---|---:|---:|---|']
    for size in SIZES:
        js=select(jobs,size=size)
        totals=[sum((r.get('usage') or {}).get(k,0) or 0 for r in js) for k in ('input_tokens','output_tokens')]
        totals += [sum(((r.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens',0) or 0 for r in js),
                   sum(((r.get('usage') or {}).get('input_tokens_details') or {}).get('cached_tokens',0) or 0 for r in js)]
        models=Counter((r.get('returned_model'),r.get('served_tier')) for r in js)
        text.append(f"| {size} | {' / '.join(map(str,totals))} | {sum(r.get('accounted_usd',0) for r in js):.6f} | {dict(models)} |")
    text += ['', '토큰은840개 전체 신규 응답의 합이다. 추론 토큰은 출력 토큰의 부분집합이며 다시 비용에 더하지 않는다. 캐시 할인 없이 입력$1/출력$5 per million으로 상한 계상했다.','',
        '## 9. 실행 복구와 재현성 경계','',
        '초기 Graph 실행은 어댑터가 입력 dictionary에 계측 필드 두 개를 섞어 기존 sensor-only guard가 전부 거부했다. 입력 검증을 유지하고 계측값을 별도 tuple 항목으로 옮겼다. 이 초기 시도의 Graph·대조군·시간은 최종 성능 통계에서 제외하고 원본 디렉터리에 그대로 보존했다. 수정 시도는 native 센서·840 API 응답·실제 SAM2 mask만 hash가 같은 상태로 재사용하고, 실제 Graph를 먼저 동결한 뒤 GT 대조군을 새로 실행했다.','',
        '품질 평가에서만 순수 depth 변환과 inverse extrinsic을 읽기 전용으로 memoize했다. 네 크기의 첫 사례와 별도 native 타이밍의 동일 입력 Graph 다섯 tensor hash를 대조한다. 공식 지연은 memoization 없는 기존 경로다.','',
        '품질 평가는4 worker×24GiB data limit로 시작했고, 메모리 부족 사례만 worker 종료 후 같은 서버에서 단독80GiB로 재실행했다. 원래 OOM receipt와 재실행 receipt는 모두 보존한다. 공식 native Graph 시간은 단독80GiB다. 입력·해상도·필터·알고리즘 변경이나 실패 이미지 대체는 없다.','',
        f'- [복구 경계]({base}engineering_recovery.json) · [재사용 입력 해시]({base}reused_inputs.json) · [프로토콜]({base}protocol.json) · [실행 단계 상태]({base}pipeline_status.json)',
        f'- [Graph 상세 프로파일]({base}graph_profile_summary.csv) · [상세 원자료]({base}timing_graph_profiles.json) · [런타임·패키지·체크포인트]({base}runtime_before_main.json)',
        '', 'Graph Coverage의 실패 제외 평균·CI는 성공 공통 사례에 한정된다. 대응 CI의 Parent/사례 수를 함께 보고하며 누락 사례를0점 또는 성공으로 채우지 않았다. 추정 가능률과 Peg<3mm 비율은 전체 계획된 상태×반복 분모로 계산한다. GT-box는 정규화 GT bounding box 대조이며 SAM2의 최적 prompt 성능 상한을 보장하지 않는다. GT-mask도 고정 기하 추정기의 최적 오차를 보장하지 않는다.','']


def graph_profile(text,base):
    from report import number,select,mean
    groups={
        '1 입력·깊이 유효성 잔여':{'prepare_total'},
        '2 Robot·Overlap·Wrist 필터':{'robot_filter','mask_overlap_filter','wrist_gate'},
        '3 역투영·PCA5·기하 근거':{'evidence_total','backprojection','pca5_normals','geometric_features','pair_evidence'},
        '4 Shape-First 후보·선택':{'structural_candidates','shape_prepare','shape_select','camera_packet_total'},
        '5 Packet·component·Raw EDE':{'packet_to_observations','current_records','selection_pack','feature_ee_pack','observed_components','packet_raw_ede','represent_total'},
        '6 Cross-view·3mm Dedup':{'cross_view','dedup_3mm'},
        '7 최종 FPS128·budget':{'fps_budget','final_fps_128'},
        '8 최종 특징·엣지 조립':{'final_graph_total','final_surface_edges','robot_object_edges','peg_box_edges','edge_attributes'},
        '그 외 builder·계측 잔여':{'graph_build'}}
    rows=read(ROOT/'timing_graph_profiles.json')['rows'];summary=[]
    for size in SIZES:
        ds=[r for r in rows if r['size']==size and r.get('hashes_equal')]
        for label,names in groups.items():
            values=[sum(s['exclusive_ms'] for s in r['spans'] if s['stage'] in names) for r in ds]
            summary.append(dict(size=size,stage=label,**stats(values)))
    csvwrite(ROOT/'graph_eight_stage_timing.csv',summary)
    text += ['| Graph 단계 (exclusive) | 128 ms | 512 ms | 768 ms | 1024 ms |','|---|---:|---:|---:|---:|']
    for label in groups:
        text.append('| '+label+' | '+' | '.join(number(next(r for r in summary if r['size']==size and r['stage']==label)['mean']) for size in SIZES)+' |')
    text += ['', 'Robot FK와 입력 결합은 builder 바깥의 앞 표에 있다. 이 표는 중첩 hook의 자식 시간을 뺀 exclusive 시간으로 한 번씩만 집계한다. 카메라 Packet과 represent의 이름 있는 자식 단계 외 잔여도 명시했다. 추가 callback·snapshot 비용은 계측 오버헤드에 포함된다.','',
        '| 크기 | 프로파일 성공/계획 | 일반 builder 평균 ms | 계측 builder 평균 ms | 차이 평균 ms |',
        '|---|---:|---:|---:|---:|']
    for size in SIZES:
        ds=[r for r in rows if r['size']==size and r.get('hashes_equal')]
        text.append(f"| {size} | {len(ds)}/7 | {number(mean(ds,'native_ms'))} | {number(mean(ds,'instrumented_ms'))} | {number(mean(ds,'overhead_ms'))} |")
    text += ['', f'[8단계 CSV]({base}graph_eight_stage_timing.csv). 관측 오버헤드 차이는 실행 변동도 포함하므로 음수이면0으로 잘라내지 않았다.','']

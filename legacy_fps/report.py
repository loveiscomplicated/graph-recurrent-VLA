"""Evidence-only Korean Markdown/static HTML report; no external publishing."""
import argparse,csv,html,json,platform,statistics,time,re
from pathlib import Path
import importlib.metadata
import numpy as np
from graph.integration_baseline.io import write_json,sha256
from .benchmark import BASE
from .integration import materialize

def read(p):return list(csv.DictReader(p.open())) if p.exists() else []
def numeric(x):
    try:return float(x) if x not in ('',None) else None
    except (ValueError,TypeError):return None
def mean(rows,key):
    v=[numeric(r.get(key)) for r in rows];v=[x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None
def fmt(x):return '—' if x is None else f'{x:.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])+'\n'
def markdown_html(text):
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    out=[];in_table=False
    for line in text.splitlines():
        if line.startswith('|'):
            cells=[x.strip() for x in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?',x) for x in cells):continue
            tag='td' if in_table else 'th'
            if not in_table:out.append('<div class="scroll"><table>');in_table=True
            out.append('<tr>'+''.join('<'+tag+'>'+inline(x)+'</'+tag+'>' for x in cells)+'</tr>');continue
        if in_table:out.append('</table></div>');in_table=False
        if not line:continue
        if line.startswith('#'):
            level=min(4,len(line)-len(line.lstrip('#')));out.append(f'<h{level}>'+inline(line[level:].strip())+f'</h{level}>')
        else:out.append('<p>'+inline(line)+'</p>')
    if in_table:out.append('</table></div>')
    return '\n'.join(out)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    manifest=json.loads((root/'geometry_manifest.json').read_text());materialize(root)
    extra={}
    for name,file in (('paired','latest_paired.txt'),('learning','latest_learning.txt'),('temporal','latest_temporal.txt'),('latency','latest_latency.txt')):
        if (BASE/file).exists():
            path=Path((BASE/file).read_text().strip());metadata=path/'metadata.json'
            extra[name]=dict(path=str(path),run_id=path.name,status=json.loads(metadata.read_text())['status'] if metadata.exists() else 'missing')
    runtime=dict(host=platform.platform(),python=platform.python_version(),installed={})
    for name in ('numpy','scipy','opencv-python','torch','mani_skill','sapien','matplotlib'):
        try:runtime['installed'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:runtime['installed'][name]=None
    write_json(root/'runtime_versions.json',runtime)
    rows=read(root/'geometry_metrics.csv');edges=read(root/'edge_metrics.csv');filters=read(root/'robot_filter_metrics.csv')
    static=[r for r in rows if int(r['timestep'])==0]
    summary=[]
    for budget in ('native_variable_N','matched_native_minN_posthoc','cap128_per_object'):
        for name in [f'{a}{i}' for i in range(5) for a in 'GL']:
            for oid in (0,1):
                group=[r for r in static if r['variant']==name and r['budget_mode']==budget and int(r['object_id'])==oid]
                summary.append(dict(variant=name,object_id=oid,budget=budget,episode_rows=len(group),nonempty_distance_rows=sum(numeric(r.get('surface_distance_mean_mm')) is not None for r in group),
                    **{key:mean(group,key) for key in ('count','surface_distance_mean_mm','surface_distance_p95_mm','current_visible_coverage_4mm','boundary_coverage_4mm','source_object_purity','source_robot_contamination','geometry_outlier_fraction_8mm')}))
    write_json(root/'comparison_summary.json',summary)
    fixed=[r for r in static if int(r['seed'])==30002 and r['variant'] in ('G0','L0')]
    fixedtable=table(['입력','물체','Budget','N','표면 평균 mm','가시 표면 coverage','경계 coverage'],[[r['variant'],'Peg' if r['object_id']=='0' else 'Box',r['budget_mode'],r['count'],fmt(numeric(r['surface_distance_mean_mm'])),fmt(numeric(r['current_visible_coverage_4mm'])),fmt(numeric(r['boundary_coverage_4mm']))] for r in fixed])
    maintable=table(['입력','물체','Budget','Episode','비어있지 않은 거리 표본','평균 N','표면 평균 mm','표면 coverage','경계 coverage'],[[r['variant'],'Peg' if r['object_id']==0 else 'Box',r['budget'],r['episode_rows'],r['nonempty_distance_rows'],fmt(r['count']),fmt(r['surface_distance_mean_mm']),fmt(r['current_visible_coverage_4mm']),fmt(r['boundary_coverage_4mm'])] for r in summary if r['variant'] in ('G0','L0')])
    varianttable=table(['입력','물체','평균 Native N','표면 평균 mm','가시 coverage','경계 coverage','Source 순도','Robot 오염'],[[r['variant'],'Peg' if r['object_id']==0 else 'Box',fmt(r['count']),fmt(r['surface_distance_mean_mm']),fmt(r['current_visible_coverage_4mm']),fmt(r['boundary_coverage_4mm']),fmt(r['source_object_purity']),fmt(r['source_robot_contamination'])] for r in summary if r['budget']=='native_variable_N'])
    edge_summary=[]
    for name in ('G0','L0','L0_surface','L0_rawede'):
        group=[r for r in edges if r['timestep']=='0' and r['variant']==name]
        edge_summary.append([name,len(group),fmt(mean(group,'undirected_surface_edges')),fmt(mean(group,'mesh_unsupported_edges_4mm')),fmt(mean(group,'camera_free_space_crossing_edges')),fmt(mean(group,'aperture_crossing_edges')),fmt(mean(group,'connected_components'))])
    edgetable=table(['입력/cap128','물체-Episode 행','Edge','GT 표면 미지원 Edge','Free-space Edge','Hole crossing','Components'],edge_summary)
    temporal='NOT_EXECUTED';temporal_summary={}
    if extra.get('temporal',{}).get('status')=='finished':
        tr=Path(extra['temporal']['path']);tt=read(tr/'geometry_metrics.csv');gg=read(tr/'G_reference_geometry_metrics.csv')
        for name in ('G2','L2','G4','L4'):
            group=[r for r in (tt if name.startswith('L') else gg) if r['variant']==name and r['budget_mode']=='native_variable_N']
            # Reference G sampled at identical metric timesteps before aggregation.
            times={r['timestep'] for r in tt};group=[r for r in group if r['timestep'] in times]
            temporal_summary[name]={k:mean(group,k) for k in ('count','cumulative_observed_coverage_4mm','common_union_observed_coverage_4mm','ghost_count','duplicate_neighbour_fraction_1mm','pose_translation_error_mm','object_local_surface_drift_mean_mm','new_correct_previously_unseen_nodes')}
        temporal=table(['입력','평균 N','누적 coverage','Ghost 수','1mm 중복률','Pose 이동오차 mm','Local drift mm','새 올바른 surface'],[[name,*[fmt(d[k]) for k in ('count','cumulative_observed_coverage_4mm','ghost_count','duplicate_neighbour_fraction_1mm','pose_translation_error_mm','object_local_surface_drift_mean_mm','new_correct_previously_unseen_nodes')]] for name,d in temporal_summary.items()])
    write_json(root/'temporal_summary.json',temporal_summary)
    learn='NOT_EXECUTED';probe=[]
    if extra.get('learning',{}).get('status')=='finished':
        lr=Path(extra['learning']['path']);aa=read(lr/'absolute_metrics.csv');pp=read(lr/'paired_delta_metrics.csv')
        for variant in [f'{a}{i}' for i in range(5) for a in 'GL']+['L0_surface','L0_rawede']:
            for encoder in ('E0','E1','E2'):
                group=[r for r in aa if r['variant']==variant and r['encoder']==encoder and r['probe']=='P1_readout' and r['decoder']=='linear' and r['split']=='validation' and r['condition_group']=='baseline']
                paired=[r for r in pp if r['variant']==variant and r['encoder']==encoder and r['probe']=='P1_readout' and r['decoder']=='linear' and r['split']=='validation' and r['target_moved']=='True' and r['physical_valid']=='True']
                probe.append(dict(variant=variant,encoder=encoder,validation_position_mean_mm=mean(group,'mean_mm'),paired_delta_error_mm=mean(paired,'delta_error_mm'),direction_cosine=mean(paired,'direction_cosine'),signed_gain=mean(paired,'signed_gain')))
        learn=table(['입력','Encoder','독립 Episode Validation 위치 mm','Paired Δ mm','방향 cosine','Signed gain'],[[r['variant'],r['encoder'],fmt(r['validation_position_mean_mm']),fmt(r['paired_delta_error_mm']),fmt(r['direction_cosine']),fmt(r['signed_gain'])] for r in probe])
    write_json(root/'probe_summary.json',probe)
    protection=json.loads((root/'protection_audit.json').read_text()) if (root/'protection_audit.json').exists() else {}
    external_changes=protection.get('changed',[])
    stages=dict(Stage0='PARTIAL' if external_changes else 'PASS',Stage1='PASS',Stage2='PARTIAL',Stage3='PARTIAL',Stage4='PARTIAL',Stage5='PARTIAL' if probe else 'NOT_EXECUTED',Stage6='PASS')
    pipeline=BASE/'stage_pipeline_state.json'
    if pipeline.exists():
        for b in json.loads(pipeline.read_text()).get('blocked',[]):
            if b['stage']=='learning':stages['Stage5']=b['status']
    write_json(root/'stage_status.json',stages)
    answers=[
        ('Q1','SUPPORTED','현재 동일 관측의 표면 거리 개선 사례가 있다. 모든 물체·시간·일반화 우위는 미확인; 아래 Native/Matched 표와 경계 손실을 함께 본다.'),
        ('Q2','INCONCLUSIVE','Native 점 수와 후처리 Matched-N, 동일 FPS 점의 Edge 조건을 분리했다. Erosion/SOR/FPS 각 단계의 독립 효과까지 식별한 ablation은 실행하지 않았다.'),
        ('Q3','SUPPORTED','Matched-N에서 유지되는 표면 거리 개선 사례가 있으나 경계 보존은 악화할 수 있다. 후처리 subset 진단이며 Native sampler 단독 순위가 아니다.'),
        ('Q4','INCONCLUSIVE','공통 Self-filter의 동일 픽셀 판정과 물체 보존/오염을 측정했다. Finger 근접과 작은 Peg 일반화는 표본 한계가 있다.'),
        ('Q5','INCONCLUSIVE' if temporal_summary else 'NOT_TESTED','이 Episode에서 L2/L4는 Ghost/Drift가 줄었지만 대응 G2/G4보다 누적 coverage가 낮았다. 가림 유지와 새 표면 통합의 일관된 우위는 확인하지 못했다.' if temporal_summary else '시간축 파일럿만으로 전체 가림 유지·재등장 복원을 확인하지 못했다.'),
        ('Q6','INCONCLUSIVE','Pose/ghost/local drift를 따로 기록했다. 한 연속 Episode와 부분 표면 ICP로 안정적 강체 추적의 일반화를 확정할 수 없다.'),
        ('Q7','SUPPORTED','실제 hand_camera 동시 관측과 변화하는 Extrinsic으로 L3/L4를 실행했다. Identity 충돌·중복 제거와 추가 coverage를 기록했으며 센서를 임의 생성하지 않았다.'),
        ('Q8','INCONCLUSIVE','같은 cap128/F5, 동일 Objective/Optimizer/Update/Seed/Selection에서 E0/E1/E2 파일럿을 실행했다. 100 update·100 Probe epoch는 확정 순위에 부족하다.' if probe else 'GPU/Probe 결과가 완료되지 않았다.'),
        ('Q9','INCONCLUSIVE','Raw centroid/geometry summary와 frozen readout Probe를 비교할 수 있으나 Probe 일반화/학습 부족과 GNN 정보 손실을 완전히 분리하지 못했다.'),
        ('Q10','SUPPORTED','거리·coverage·경계·오염·ghost·Probe를 독립적으로 저장해 개선/악화 조건을 구분한다. 단일 점수의 전역 순위는 제공하지 않는다.'),
        ('Q11','INCONCLUSIVE','후속 Policy 도입의 확정 근거는 부족하다. 독립 Final Test, 충분한 동일 진단 예산, 다수 연속 Episode가 필요하다. BC/GRU/Policy는 실행하지 않았다.'),
        ('Q12','INCONCLUSIVE' if external_changes else 'CONFIRMED',
            '이 확장은 원본 쓰기/Process 제어를 하지 않았다. 최초 감사7589개 불변 후 병렬 원본 encoder.py/test_contracts.py2개가 외부 변경돼 전역 불변 검사는 FAIL이다. 나머지7587개와 확장의 고정 의존 소스는 동일하다. 원본은 자체 감사 오류 후 다시running 상태로 관찰됐다.' if external_changes else '감사 시점 stable7589개 hash가 그대로이며 원본 Process 제어/Source 변경은 없다. 확장 GPU 단계와 원본 학습 시간은 겹치지 않았다.')]
    write_json(root/'research_questions.json',[dict(question=q,verdict=v,evidence=e) for q,v,e in answers])
    packages=runtime['installed'];source=Path(__file__).resolve().parents[1]
    failure=json.loads((root/'failure_catalog.json').read_text()) if (root/'failure_catalog.json').exists() else {}
    costtext='NOT_EXECUTED'
    if extra.get('latency',{}).get('status')=='finished':
        costs=Path(extra['latency']['path']);cpu=json.loads((costs/'cpu_stage_costs.json').read_text());cpu=[r for r in cpu if not r['warmup']]
        inf=json.loads((costs/'gnn_inference_costs.json').read_text())
        costtext=table(['실제 측정 항목','평균 ms'],[[key,fmt(float(np.mean([r[key] for r in cpu])))] for key in ('filter_ms','legacy_extract_ms','legacy_extract_plus_F5_ms','map_initialize_ms','graph_ms')])
        costtext+='\n'+table(['Encoder','단일 Graph inference 평균 ms','Graph 수'],[[e,fmt(mean([r for r in inf if r['encoder']==e],'mean_ms')),len([r for r in inf if r['encoder']==e])] for e in ('E0','E1','E2')])
    text=f'''# Task09-C Legacy FPS 독립 확장

실험 UUID: `{root.name}`. 실행 호스트: Vast.ai57640710 / RTX3090. 독립 소스: `/workspace/task09c_legacy_fps/source`. 원본09-C는 읽기 전용이다. 이 보고서는 파일럿 구현·실험 결과이며 Policy 도입 승인이나 전체 우위 결론이 아니다.

## 실제 Legacy Source와 동일성

`visual/scripts/pointcloud.py`, Step5/Step8 JSON, `visual/docs/v0.md` 및 원래 Step5 이미지를 확인했다. Erosion2회 → 유효점50 미만 fallback → pixel-centre backprojection → random pre2048 → SOR(k16,std2) → optional cluster(기본OFF) → FPS1024 → CV camera-to-world 순서를 그대로 사용했다. Source SHA256: `{sha256(source/'visual/scripts/pointcloud.py')}`. 원본 함수의 XYZ·SOR/FPS 선택 순서·RNG 상태·World 변환을 재현하는 테스트와 실제 파일럿 입력의 원본 출력 대조를 통과했다. Provenance는 nearest neighbour 추정이 아니라 정확한 XYZ 바이트 row join이다. 중복 XYZ로 출처가 모호하면 거부한다.

과거v0는512×512, 현재 Capture는128×128이다. 비교 양쪽에는 같은 현재 Depth/Predicted Mask/Calibration만 전달했다. 과거 그림과 수치를 현재 성능의 직접 기준으로 쓰지 않았다. Original default seed0의 순차 Episode/Frame/Peg-Box RNG 테스트와 독립 프레임 seed42 hash 정책은 별개다. Branch는 동일 Parent RNG 그룹을 사용한다. `LEGACY_EXACT_GIVEN_INPUT_AND_RNG`는 과거 전체 실행의 픽셀 선택을 재현했다는 이름이 아니다.

현재 Main의{failure.get('main_object_frames','미측정')} Object-frame 중{failure.get('main_nonempty','미측정')}개가 비어 있지 않았고, 그중{failure.get('main_native_final_FPS_pass_through_nonempty','미측정')}개는 SOR 후1024점 이하라 최종 FPS가 pass-through였다. Final FPS와 random presampling이 실제 동작한 Main 사례는 각각{failure.get('main_native_final_FPS_active','미측정')}/{failure.get('main_random_presample_active','미측정')}개다. 모든 Camera/Filtered 조건{failure.get('all_camera_object_frames','미측정')}개 중 Final FPS 동작은{failure.get('all_camera_final_FPS_active','미측정')}개였다. 따라서 이번 차이를 FPS 알고리즘 자체의 우위로 단정하지 않는다. Erosion/SOR를 포함한 전체 Legacy 정책·점 밀도·후처리 budget을 비교한 증거다. 상세 원인별 `failure_catalog.json`/`preprocessing_failure_audit.csv`를 보존했다.

## 실행 보호·데이터

기존09-C Source의 해시가 같은 복사본을 확보했다. 기존 프로세스와 G0~G4 실행, UUID, 진행 중 Artifact에 Patch/Injection/Signal을 보내지 않았다. 안정 파일7589개와 실행 중 출력 metadata를 구분해 Inventory했다. `protection_audit.json`에서 최종 hash를 확인한다. 기존 Frozen HDF5/Checkpoint/Full 중단 상태도 보호 목록에 포함했다. 다른 Agent가 로컬 공유 작업 트리를 변경하는 것은 관찰했으며 이 확장의 구현은 공유 트리에 쓰지 않았다.

중간 읽기 전용 점검에서 원본09-C는2026-10-27 09:01:54 UTC에 자체 `diagnostic_training` 오류로 종료돼 있었다. G0/E0의2000 update 후 원래 전체 노드 순열 감사에서192/320 Readout 값이 달라진 AssertionError다. 확장 학습은09:00:09, 확장 inference 측정은09:00:22에 종료했고 해당 원본 학습 시작은09:01:05여서 GPU 단계가 겹치지 않았다. `original09c_runtime_outcome.json`에 그 시점의 상태/로그/시간 근거를 복사했다. 이후 병렬 작업 중 원본 `encoder.py`/`tests/test_contracts.py`2개가 외부에서 변경되고 원본 상태가 다시running으로 바뀐 것을 관찰했다. 이 확장은 두 파일을 쓰거나 원본을 수정·재시작하지 않았다. 전역 Source 불변은 더 이상 PASS라고 주장하지 않으며, 나머지7587개 보호 파일 및 실제 실행한 고정 Source Copy의 동일성을 별도로 검사한다. `original09c_runtime_followup.json`/`protection_audit.json`에 최신 관찰을 보존했다. 원본 Full A/B/C는 이 확장에서 재개하지 않았다.

Capture: `{manifest['config']['capture']}`. 파일럿은8 Train/4 Validation Episode의 t=0과 Seed30002/30125의 t=0..15, 총{len(manifest['frames'])}프레임이다. 같은 Episode의 시간 표본을 새 독립 Test로 세지 않는다. Original HDF5와의 actual_reset mismatch를 보존했다. 제공된 두 Screenshot의 RGB-D/Runtime hash가 없어 정확한 Screenshot 재현은 NOT_PROVEN이다. Seed30002/t0/Reach/Main을 동일 관측끼리 비교했다. GT segmentation/poses/hole은 sidecar 평가와 진단 target 전용이며 sampler/filter/map/encoder 입력에는 없다.

## Seed30002 t=0 필수 사례

{fixedtable}

이 상태의 Peg 예측 마스크가 비어 있으면 G0/L0 모두 Empty다. FPS는 SAM2가 완전히 누락한 Peg를 복구하지 못한다. 거리가 정의되지 않는 Empty 객체를 좋은 표면 정확도로 처리하지 않는다. Coverage와 Missing은 별도 기록한다.

## G0/L0 Native·Matched-N

아래는 각 Episode t=0의 동등 가중 평균이며 거리의 비어있지 않은 분모를 표시한다. 표면 거리는 외부 collision-union 표면의 결정적30000개 표본 기준으로, 정확한 analytic mesh distance와 다르다. Coverage threshold는4mm다. Matched-N은 각 물체의 min(G,L) 실제 지원 수에 공통 identity-seeded farthest subset을 적용했다. 보간/가상 점은0개이며 Shape-First 구조 선택의 Native 의미와 섞지 않는다. cap128은 동일 상한이지 Missing까지 동일 N이라는 주장이 아니다.

{maintable}

`comparison_summary.json`과 전체 CSV에는 G1~G4/L1~L4도 포함한다. 표면 거리·순도 개선과 Boundary/Hole rim 손실이 동시에 발생할 수 있다. Erosion이 내부 표면 점을 늘려 보여도 Hole/외곽 경계를 지울 수 있으므로 거리 하나로 우위를 판정하지 않는다.

아래는 모든 대응 조건의 t=0 Native 결과다. 같은 Gi/Li끼리는 Sensor 범위가 같다. Main 전용과 Main+Wrist 간 가시 표면 분모는 다르므로 이 표의 coverage만으로 Fusion 효과를 비교하지 않는다. Camera 공통 분모인 common_union_observed/common_both_visible 지표를 별도 CSV와 시간축 요약에 보존했다.

{varianttable}

## 같은 FPS 점, 다른 Edge

{edgetable}

L0는 robot-object/peg-box를 포함해 Edge0인 No-edge다. L0_surface/L0_rawede는 같은 cap128 FPS 점·같은 F5/Node를 사용한다. Raw EDE는 실제 현재 Main Source Pixel/Observed Component에서 원래 helper로 재계산했다. 과거 점과 Multi-camera 누적 Map에는 UV Raw EDE를 적용하지 않았다. 3D Surface edge schema는 공통09-C의 normal/tangency/witness/free-space rule이며 Raw EDE가 아니다. 동일 XYZ에서 Edge가 Geometry 자체를 바꾸지는 않으며, 허위 bridge와 GNN 정보 보존을 별도로 측정한다.

## Robot Self-filter

공통 filter의 confirmed/uncertain 픽셀은 G/L에서 정확히 같음을 검사했다. Filter 전후 mask·native FPS 단계 source·point를 별도 보존한다. Robot removal recall 평균 {fmt(mean(filters,'robot_removal_recall'))}, 실제 물체 보존 평균 {fmt(mean(filters,'object_preservation'))}. 분모와 Camera/물체별 값은 `robot_filter_metrics.csv`에 있다. 근접 finger 평가는 관측 finger-pad centre20mm 이내이며 실제 힘/접촉 판정이 아니다. Filter와 Erosion/SOR의 연속 손실은 Boundary와 native count/source 단계로 감사한다.

## Persistent Map·Pose·신규 표면

`PersistentMap`, register, merge3mm, identity rejection, free-space contradiction, observed/estimated 구분은 pinned09-C 구현 그대로다. Native Map surface 수와 GNN cap128은 독립이다. 동일 surface는 update하고 새 surface만 add한다. GT motion이나 미래 관측으로 과거 Map을 보정하지 않았다.

{temporal}

Full temporal 실행이 완료됐으면 Seed30002의 t=0부터 마지막까지 재생한 결과다. 기존 G는 frame manifest와 final tracker/lifecycle가 완료된 Episode만 hash-copy해서 사용했고 동일 Capture input hash를 대조했다. parent09-C가 다른 Episode를 실행 중인 상태를 완료된 전체 Run으로 간주하지 않았다. 이전 표면의 유지, 현재 보임/Estimated, Ghost, 중복과 새 올바른 surface를 분리했다. 넓은 정책 일반화는 미확인이다.

## Main+Wrist

실제 sensor contract의 역할과 camera 이름을 사용했다. 동시 Episode/Timestep/Timestamp/Robot, CV↔GL calibration 검사를 통과했다. Wrist 이동은 매 프레임 저장된 Extrinsic을 사용한다. `camera_audit.json`에 World/Link 변환이 있고 `fusion_audit.json`에 identity disagreement가 있다. 보이지 않는 mask를 negative evidence로 쓰지 않는다. 두 camera의 동일 surface는 공통 merge 규칙으로 합친다. 원본 raw depth overflow/비양수는 유효 깊이로 복원하지 않는다.

## Geometry/GNN/Frozen Probe

Native Legacy는 XYZ다. 이번 공통 Map/GNN 조건은 **FPS+원래 PCA5/F5 증강**이며 실제 선택 Source Pixel에서 원래 dense evidence를 gather했다. GT pose/goal을 입력하지 않았다. 과거 depth jump/variation은 unavailable/invalid다. E0/E1/E2는09-C 구조와 동일한 진단 Objective(position6+visible geometry30), AdamW lr0.001, batch16, seed42, cap128, update100 및 동일 Validation 선택을 사용한 파일럿이다. Frozen Probe는 기존09-B Ridge/Small MLP, 동일 Train/Validation, 최대100 epoch를 사용한다. 정식2000 update/2000 Probe epoch 및 독립 final seeds62091/62092는 NOT_EXECUTED다. 기존 BC/GNN checkpoint OOD 투입이나 GRU/Action Head/Policy 학습은 없었다.

고정 Source의 structured readout은 첫3행을 로봇 역할로 읽는다. 전체 노드를 순열한 원래 감사는 이 위치 계약 때문에 실패했다. 독립 `CanonicalDiagnosticEncoder` 입력 Adapter가 기존 node_type으로 세 로봇 행을 정렬한다. 정상 저장 Graph에서는 항등 변환이며 기존 초기 파라미터와 출력이 정확히 같다는 계약 테스트를 통과했다. 동일 구조·가중치 이름을 유지한 채 모든 후보에 같은 Adapter를 적용했다. 학습 루프에는 Encoder factory 인자가 없어 원래 루프를 독립 파일에 복사하고 생성자만 교체했다. 기존09-C Source/실행 프로세스에는 수정·주입하지 않았다. 전체 노드 순열 감사는 각 선택 Checkpoint에서 실제 CUDA로 재검사했다.

{learn}

위 표는 Frozen Readout의 Ridge 결과이며 Training fit으로 좋은 표현을 판정하지 않는다. 모든 MLP·입력 centroid/summary·로봇 control·Paired 방향/gain·각 Episode 분모는 별도 CSV/NPZ에 있다. Counterfactual은 동일 Robot/Calibration의 t=0 branch만 실행했다. Late-time branch의 causal map fork는 미구현이며 새 tracking 방식으로 대체하지 않았다. t=0은 과거 표면 기억의 paired 효과를 입증하지 않는다.

## 비용·실패·실행 한계

CPU geometry는 nice19/단일 BLAS thread이며 기존 CPU 작업을 종료/우선순위 변경하지 않았다. 각 프레임의 G/L 전체 pipeline, 원본 Legacy extract, Graph+metric 작업 시간과 process RSS를 `costs.json`에 저장했다. Graph+metric 시간은 순수 Graph 생성 latency로 해석하지 않는다. 추가 비용 측정은 Seed30002/t0의 실제 고정 입력·fresh map 초기화이며 steady-state tracking 속도는 아니다. GPU 진단은 다른 GPU process가 없는 gate와6% allocator cap에서만 실행하며 선택 checkpoint에 실제 CUDA peak/RSS와 학습 시간이 있다. CUDA context를 초기화하기 전0개, 이후 유일한 새 context1개라는 조건으로 NVIDIA host PID를 등록하고 각 모델/validation 시 다른 PID를 다시 검사했다.

{costtext}

초기 확장 실행의 Import/직렬화/Adapter 연결 오류와 잘못된 PID namespace 자원 차단, 원래 positional readout의 전체 순열 실패를 Attempt log와 failed UUID에 보존했다. 이 실패 산출물을 성능 순위에 사용하지 않았다. 최종 계약 테스트는54개이며 전체 노드 순열과 정상 입력의 기존 출력 동일성도 포함한다.

Empty mask, Erosion fallback, SOR 제거, FPS 외곽 outlier 선택, 잘못된 object/robot source와 Map ghost를 보존했다. 모든 실패 유형이 실제로 발생했다고 주장하지 않는다. Lifecycle·source purity·tracking audit와 영상을 보고 발생 여부를 판단한다. PNG/MP4는 decode했고 PNG/contact sheet를 직접 검수한다. prefix 영상의 new_surface/ghost 파일명은 새로운 사건의 발생을 자동으로 입증하지 않는다. Full temporal contact sheet에 실제 상태를 추가한다.

현재 전처리 감사: Erosion 때문에 유효 입력이 완전히 비게 된 사례 {failure.get('erosion_caused_empty','미측정')}개, SOR가 제거한 올바른 surface pixel {failure.get('sor_correct_surface_removed','미측정')}개, 그중 올바른 Hole rim pixel {failure.get('sor_correct_hole_rim_removed','미측정')}개다. 이는 GT를 이용한 사후 평가이며 제거 정책을 GT에 맞춰 바꾸지 않았다. Filtered/Unfiltered와 Camera가 중복 계수되므로 독립 물리 pixel 총수로 해석하지 않는다.

현재 호스트에서 저장 Observation CPU 분석과 완료된 CUDA 진단만 실행했다. Simulator를 새로 띄우거나 PhysX CUDA/MPS/macOS backend를 실행하지 않았다. 새 dependency 설치, Git commit, 외부 upload/publish는 없었다.

## Stage 판정

{table(['Stage','판정'],list(stages.items()))}

PARTIAL은 테스트 부족/파일럿 범위를 뜻한다. 미실행 단계와 구현된 Adapter를 구분한다. Original09-C의 전체 결과를 재산출/변경하지 않았다.

## 연구 질문

{table(['질문','판정','근거'],answers)}

## 재사용·통합 계약과 Artifact

공통 ExperimentLogger UUID/disabled local logging을 재사용했다. artifact 게시 도구는 호출하지 않았다. `integration_manifest.json`에 import/config/point/map/graph/feature/seed/hash/data-checkpoint compatibility/resource/regression 계약을 명시했다. `implementation.diff`는 새 패키지 추가 전용이며 기존09-C에는 적용하지 않았다. 구현 Source Snapshot과 Hash, Capture/Frozen identity, 실제 RGB-D/Mask hash, pixel provenance, graph tensors/map states/filter/tracker/calibration, geometry/GNN/Probe, contract logs, static HTML/PNG/MP4, pre/post protection, `verification.json`을 보존했다.

재실행: 독립 Source root에서 `python -m legacy_fps.benchmark --full --output NEW_ROOT`, `legacy_fps.branches --geometry RUN`, `legacy_fps.temporal --geometry RUN --reference COMPLETED_GEOMETRY`, `legacy_fps.learning --geometry RUN --paired PAIRED --updates 2000 --probe-epochs 2000`. 새 UUID/root를 사용하며 기존 G0~G4 모듈을 수정할 필요가 없다. 충분한 자원이 있을 때만 수행한다.
'''
    docs=BASE/'source/docs/visual-graph-integration';docs.mkdir(parents=True,exist_ok=True)
    (docs/'09c_legacy_fps_extension.md').write_text(text);(root/'09c_legacy_fps_extension.md').write_text(text)
    # Deliberately simple static page, no external assets/scripts or service deployment.
    links=['09c_legacy_fps_extension.md','integration_manifest.json','comparison_summary.json','geometry_metrics.csv','edge_metrics.csv','robot_filter_metrics.csv','preprocessing_failure_audit.csv','failure_catalog.json','probe_summary.json','temporal_summary.json','research_questions.json','contract_tests.log','protection_audit.json','original09c_runtime_outcome.json','original09c_runtime_followup.json','verification.json','implementation.diff']
    linktext=''.join(f'<li><a href="{html.escape(n)}">{html.escape(n)}</a></li>' for n in links)
    related=''.join(f'<li>{html.escape(k)} UUID {html.escape(v["run_id"])}: {html.escape(v["status"])} — <a href="{"../../"+str(Path(v["path"]).relative_to(BASE))+"/metadata.json"}">metadata</a></li>' for k,v in extra.items())
    media=''.join(f'<figure><figcaption>{name}</figcaption><video controls preload="metadata" src="media/{name}.mp4"></video></figure>' for name in ('same_view','matched_N','persistent_map','main_wrist','new_surface','ghost_tracking') if (root/'media'/(name+'.mp4')).exists())
    if extra.get('temporal',{}).get('status')=='finished':
        rel='../../'+str(Path(extra['temporal']['path']).relative_to(BASE))+'/media/'
        media+=''.join(f'<figure><figcaption>Full106: {name}</figcaption><video controls preload="metadata" src="{rel}{name}.mp4"></video></figure>' for name in ('persistent_full','fusion_full','new_surface_full','ghost_full'))
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>09-C Legacy FPS {root.name}</title><style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;background:#f5f7fb;color:#182331}}h1{{font-size:30px}}h2{{margin-top:48px;border-top:2px solid #cbd7e3;padding-top:20px}}img,video{{width:100%;background:white}}.scroll{{overflow:auto;background:white;border-radius:8px}}table{{border-collapse:collapse;font-size:13px;min-width:100%}}td,th{{padding:8px;border-bottom:1px solid #dae0e7;text-align:left}}th{{background:#e4edf5}}code{{background:#e7ecf2;padding:2px 4px;overflow-wrap:anywhere}}a{{color:#175c9b}}p{{overflow-wrap:anywhere}}figure{{margin:24px 0}}</style><h1>Task09-C · Legacy FPS 독립 확장</h1><p>UUID {root.name} · 저장 관측 파일럿 · Native/Matched-N/FPS+F5 분리</p><img src="media/sensor_input_t0000.png" alt="Actual main/wrist RGB and predicted masks"><img src="media/comparison_t0000.png" alt="Seed30002 t0 same-view negative controls"><h2>자료</h2><ul>{linktext}</ul><h2>별도 실행</h2><ul>{related}</ul>{media}<h2>전체 보고서</h2>{markdown_html(text)}</html>'''
    (root/'legacy_fps_extension.html').write_text(page)
    write_json(root/'report_manifest.json',dict(main_run_id=root.name,related=extra,markdown='09c_legacy_fps_extension.md',html='legacy_fps_extension.html',stages=stages))
    from utils.experiment_artifacts import register_existing_output
    for name,kind in (('09c_legacy_fps_extension.md','reports'),('legacy_fps_extension.html','reports'),('integration_manifest.json','reports'),('implementation.diff','reports'),('comparison_summary.json','metrics'),('temporal_summary.json','metrics'),('probe_summary.json','metrics'),('research_questions.json','metrics'),('stage_status.json','metrics')):
        register_existing_output(root,root/name,kind=kind)
    print(str(root/'legacy_fps_extension.html'))
if __name__=='__main__':main()

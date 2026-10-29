"""Portable Korean research report; only completed evidence is presented."""
import argparse,json,html
from pathlib import Path
from collections import Counter
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_report import markdown_html
from utils.experiment_artifacts import register_existing_output
from .summarize import read_csv,average


def fmt(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def compose(root,prefix=''):
    read=lambda p:json.loads((root/p).read_text())
    link=lambda title,path:f'[{title}]({prefix+path})'
    stats=read('summary/geometry_macro.json');edges=read('summary/edges_macro.json');gate=read('summary/gate_decision.json');motiongt=read('summary/normal_box_motion.json')
    track=read_csv(root/'summary/tracking_per_episode.csv');motions=read_csv(root/'summary/motion_controls.csv')
    state=read('pipeline_state.json');identity=read('baseline_identity.json');cfg=read('preregistered_protocol.json')['protocol']
    oracle_motion=read_csv(root/'summary/oracle_motion_metrics.csv')
    s=lambda name,budget='cap128':next(r for r in stats if r['split']=='validation' and r['variant']==name and r['budget']==budget)
    er=lambda name:next(r for r in edges if r['split']=='validation' and r['variant']==name and r['budget']=='cap128')
    b0,b1,b2=s('B0'),s('B1'),s('B2')
    for d in gate['candidates']:
        if d['decision']!='NO-GO':raise RuntimeError('report requires completed eligible-candidate GNN path before claiming completion')
    total=sum(x['frames'] for x in identity['episodes']);validation=sum(x['frames'] for x in identity['episodes'] if x['split']=='validation')
    text=[
        '# Task 09-E — Plan B Box-only Persistent Geometry Feasibility',
        f'ExperimentLogger UUID `{root.name}`. **Plan B 판정: NO-GO (Development/Validation Geometry Gate)**. B1/B2를 구현하고 실제 정지·가림·±10mm 이동 조건에서 평가했다. 두 후보 모두 사전 Geometry Gate를 통과하지 못해 E0/E1 학습·Probe 및 새 독립 Final을 실행하지 않았다. 최종 A/B GNN 성능 우열은 NOT_TESTED다.',
        '## 1. 연구 범위·실행 환경',
        f'Vast.ai57640710 / RTX3090. 기존12 Episode(Train8/Validation4), 연속 {total}프레임(Validation {validation})를 causal 순서로 모두 갱신하고 매5번째 및 마지막 프레임에서 Geometry/Graph를 측정했다. 반복 프레임·분기를 독립 Episode로 세지 않는다. 새 이동 진단은 Development seed30002의66프레임 및45개 실제 kinematic 분기다. macOS는 계약 테스트·전달 검증에 사용했다.',
        'CPU PhysX/SAPIEN에서 새 RGB-D를 렌더하고 CUDA DINO/SAM2를 추론했다. Geometry/Graph 배열 계산은 원격 CPU다. 이번 Task에서 CUDA PhysX, macOS simulator/MPS, 실시간20Hz와 Policy closed-loop 성공률은 평가하지 않았다.',
        '## 2. 기존 결과·Plan A Source/Config Identity',
        '09-B의 readout 손실,09-C Final2EP/258프레임의 Box G1/G2 표면거리27.41/11.14mm·가시coverage31.2/16.0%·ghost0/1.1과 작은 Box pose error약0.54mm를 근거로 삼되 새 실험의 GO 근거로 대신하지 않았다. Legacy FPS 확장은100-update 개발 파일럿으로 예산과 split이 달라 순위 비교에서 제외했다.',
        '09-D 보고서·source·verification.json을 확인했다.09-D 판정은 CONDITIONAL이며 주 표현 진단 후보 C3+A-SF를 B0로 고정했다. A-L 정확도 참고 후보와 Hybrid를 이번 주 기준선으로 바꾸지 않았다. 원격09-C 원래 경로가 없어09-D 전달본에 보존된 동일 hash 원자료를 새 입력 경로로 복원했다. 이전 결과를 본 seed를 Blind Final이라 부르지 않는다.',
        link('Plan A identity·split·hash','baseline_identity.json')+' · '+link('초기 source hash','source_hashes.json')+' · '+link('최종 source·수정 diff','audits/source_identity.json'),
        'B0는 원본 PlanA.prepare/represent(C3,SF)/budget/graph를 직접 사용한다. 매 평가 프레임 cap128의 node_input/node_type/edge_index/edge_attr/edge_type를09-D 원본 저장 tensor와 exact 비교했다. Peg source/XYZ/F5, Robot76, cap128, feature21, zero temporal extras와 공통 surface/k4/M16 정책은 A/B 사이에서 동일하다. Box가 달라진 결과의 Peg–Box edge 변화는 허용한다.',
        '## 3. B0/B1/B2·Oracle 정책과 입력 계약',
        table(['조건','Box 표현','Pose update'],[['B0','원본 C3_SF 현재 Main RGB-D','없음'],['B1','World-frame surfel map','진단 registration만; node pose 운반 없음'],['B2','동일 surfel map·fusion·acceptance','09-C estimated accepted rigid update'],['O1','GT Visible Box Mask + B1','Static'],['O2','Predicted Mask + Motion Oracle','GT relative motion, 진단 전용'],['O3','GT Visible Mask + Motion Oracle','GT relative motion, 진단 전용']]),
        'Peg는 모든 조건에서 원본 Plan A 현재 관측을 그대로 사용한다. Box source는 Main만 허용하고 Wrist Box point는 거부한다. 기존 Plan A의 동시 depth/identity 및 graph free-space 검사는 그대로 유지했다. Box map 자체의 free-space contradiction은 Main depth만 사용한다. GT mask/pose/mesh/hole은 practical update가 끝난 뒤 평가와 oracle에서만 연다. GT mesh completion이나 보지 못한 Surface 생성은 없다.',
        '## 4. Persistent Node Lifecycle·Fusion·오염 차단',
        '09-C Surfel/ObjectMap과 _integrate/_connected_support/_contradictions/_age를 재사용했다. object_id=BOX만 보유한다. 초기 pose는 literal world identity이며 GT actor origin을 사용하지 않는다. persistent ID, local/world XYZ·normal, confidence, first/last seen, count, current/estimated, uncertainty와 source episode/timestep/camera/pixel을 보존한다. age는 현재 causal timestep-last_seen이다. 삭제 transition과 tracking trace를 별도 저장했다.',
        'B1/B2는 Plan A의 같은 Box point를 받는다. Memory admission은18mm 연결 성분·6점 이상·20mm 이웃3개 이상, 최초에는 가장 큰 지원 성분, 이후에는 기존 map에 연결된 성분을 사용한다. 3mm/normal-compatible duplicate fusion과4회 free-space contradiction 제거를 재사용했다. B0에는 원본 Plan A를 보존했으므로 memory admission 자체도 persistence 구현 효과에 포함된다. 이를 순수한 저장 유무만의 인과 효과라고 분리하지 않는다.',
        '현재 재측정 node와 과거 estimated node를 sidecar에 구분했다. 낮은 confidence/큰 uncertainty는 LOW_CONFIDENCE, 제거는 lifecycle event로 기록한다. 빈 관측에서는 기존 map과 uncertainty를 유지·증가시키고 처음부터 미관측이면 empty다. Episode마다 map/ID/clock를 초기화한다. confidence/uncertainty는 보정된 확률·신뢰구간이 아니다.',
        'Primary GNN feature21의 temporal/status extras7은 Plan A와 똑같이0이다. current/estimated status는 sidecar에서 감사한다. historical normal은 추정 pose로 회전하며 오래된 variation/depth-jump는 availability/validity=false, tensor0이다. 새 UV/EDE edge를 과거 node에 붙이지 않았다.',
        '## 5. 정상 Box 정지 여부·추정 Tracking',
        f'GT 평가 sidecar에서 {motiongt["episodes"]} Episode/{motiongt["frames"]}프레임의 초기 대비 최대 translation은 {fmt(motiongt["max_translation_from_initial_mm"],6)}mm, 최대 rotation은 {fmt(motiongt["max_rotation_from_initial_rad"],8)}rad였다. 이 실행 sequence에서의 정지 증거이며 모든 환경에서 Box가 정지한다는 보장은 아니다. 정상 sequence의 contact force는 원자료에 없어 NOT_TESTED다.',
        table(['조건','평균 map-center pose error mm','최대 episode pose error mm','Static violation frame 합','Registration failure 합'],[[n,fmt(average([r for r in track if r['variant']==n],'mean_pose_center_error_mm')),fmt(max([r['max_pose_center_error_mm'] for r in track if r['variant']==n and r['max_pose_center_error_mm'] is not None],default=None)),int(sum(r['static_violation_frames'] for r in track if r['variant']==n)),int(sum(r['registration_failure_frames'] for r in track if r['variant']==n))] for n in ('B1','B2','O1','O2','O3')]),
        '작은 pose error와 surface 품질은 서로 다른 지표다. B1의 world pose가0으로 유지되는 것은 정지 모델의 정의이고 fused surface와 source identity가 정확하다는 뜻은 아니다.',
        '## 6. Surface Accuracy·Purity·Ghost — Validation',
        '각 Episode에서 정의된 frame 평균을 낸 뒤 Episode macro mean을 계산했다. 거리 empty는0으로 채우지 않았다. Coverage는 지원 reference가 있고 map empty면0이다. Surface 거리는09-C와 같은 deterministic30,000개 표본에서 내부 collision face를 제거한 외부 reference까지의 거리로 exact analytic mesh distance가 아니다. 4mm support/coverage,8mm 초과 estimated node를 Ghost로 썼다.',
        table(['표현','Budget','N','Mean/Median/p90/p95 mm','Source purity','Estimated ghost 평균','Current coverage','Historical coverage'],[[n,mode,fmt(s(n,mode)['count'],1),'/'.join(fmt(s(n,mode)[k],2) for k in ('surface_distance_mean_mm','surface_distance_median_mm','surface_distance_p90_mm','surface_distance_p95_mm')),pct(s(n,mode)['source_object_purity']),fmt(s(n,mode)['ghost_nodes'],1),pct(s(n,mode)['current_visible_coverage']),pct(s(n,mode)['historical_coverage'])] for mode in ('native','cap128','matched_N') for n in ('B0','B1','B2')]),
        link('Train/Validation·분모 포함 전체 수치','summary/geometry_macro.csv')+' · '+link('원본 per-frame/episode metrics','sources/'+state['geometry_id']+'/geometry_manifest.json'),
        'Current-visible 분모는 현재 Main에서 GT Box로 실제 보인 유효 RGB-D surface다. Historical 분모는 시작부터 현재까지 관측된 GT-visible surface를 actor-local1mm voxel set으로 중복 제거한 후 현재 GT 평가 pose로 운반한 reference다. 이 GT 운반은 평가에만 쓰며 map을 보정하지 않는다. 전체 mesh coverage와 혼동하지 않는다. Matched-N은 매 frame min(B0,B1,B2,128)을 사용하고 Peg는128을 유지한다. B0 empty면 세 Box 모두0이며 허위 point를 채우지 않는다. Oracle은 같은 요청 N에서 지원 부족이 있으면 짧게 남는다.',
        '## 7. Hole/Rim·Boundary·Graph Edge',
        table(['표현','Current boundary','Historical boundary','Observed current rim','Historical rim','Aperture crossing edge','False bridge fraction','Components'],[[n,pct(s(n)['current_boundary_coverage']),pct(s(n)['historical_boundary_coverage']),pct(s(n)['current_rim_coverage']),pct(s(n)['historical_rim_coverage']),fmt(er(n)['aperture_crossing_edges'],2),pct(er(n)['false_bridge_fraction']),fmt(er(n)['connected_components'],1)] for n in ('B0','B1','B2')]),
        'Rim은09-C의 평가 전용 square collision aperture reference를 재사용하고 현재/과거 실제 관측 support와4mm 이내인 rim만 해당 coverage 분모에 넣었다. 전체 rim을 보아야 하는 정답으로 두지 않는다. Silhouette boundary와 aperture rim은 서로 다른 metric이다. Edge 길이·free-space crossing·mesh unsupported edge·components는 별도 CSV에 있다. Task goal/hole geometry는 sampler/graph feature에 들어가지 않았다.',
        link('Edge·aperture·연결 성분 감사','summary/edges_macro.csv'),
        link('관측 rim 연속 구간 유지율','diagnostics/rim_continuity.csv')+' · '+link('불확실성 대 실제 오차','diagnostics/uncertainty_bins.csv')+' · '+link('가림/누락/재관측별 올바른·잘못된 유지','diagnostics/occlusion_reappearance_strata.csv'),
        '## 8. Occlusion·Reappearance·실제 이동 Negative Control',
        '새 진단은 같은 Expert robot state와 calibration에서 Box X/Y±10mm, baseline repeat, 실제 visual-only camera-plane slab의 full/partial occlusion을 렌더했다. 충돌 SAT·새 near-contact 없음·다른 물리 상태 동일·pose 복원을 감사했다. Occluder는 collision shape가 없는 시각 개체다. 개입 후 physics step은0이며 cached contact force가 새로 계산됐다고 주장하지 않는다. Partial 설정도 GT-visible pixel이0이면 full-hidden stratum으로 해석한다.',
        table(['조건','Mode','물리 유효 row','평균 surface mm','평균 pose Δ error mm','현재 가시 point 평균','Detected/unknown 비율'],[[phase,n,len(rr),fmt(average(rr,'surface_mean_mm')),fmt(average(rr,'pose_delta_error_mm')),fmt(average(rr,'gt_visible_pixels'),1),pct(average(rr,'detected'))] for phase in ('visible','partial','hidden','reappearance') for n in ('B1','B2') for rr in [[r for r in motions if r['mode']==n and r['physical_valid'] and ((phase=='visible' and r['phase']=='intervention' and r['condition'] in ('box_x-10','box_x+10','box_y-10','box_y+10')) or (phase=='partial' and r['phase']=='intervention' and r['condition'].startswith('box_partial_')) or (phase=='hidden' and r['phase']=='intervention' and r['condition'].startswith('box_full_')) or (phase=='reappearance' and r['phase']=='reappearance_represented_snapshot'))]]]),
        'Detected에는 static_violation 또는 tracking_failure_or_motion_unknown 경보를 포함하므로 실제 움직임을 올바른 방향·크기로 추정했다는 뜻은 아니다. 채택 Gate의 visible detection은 실제 visible6pixel 이상·기존 map6node 이상·±10mm 이동 조건만 분모로 삼는다. t=0 무이력 초기화는 움직임 추정 기회가 없어 분리한다.',
        '완전 가림의 Main RGB/depth/predicted mask와 estimated map 동일성을 따로 검사했다. 입력에 변화가 없으면 실제 움직임을 알아냈다고 주장하지 않는다. Reappearance는 같은 물리 t에서 저장한 실제 visible moved RGB-D를 occluded snapshot 다음 진단 tick에 다시 제시했다. 물리 시간·접촉 동역학을 가진 장기 복구는 NOT_TESTED다. Prefix는 분기마다 deep-copy하며 분기 관측으로 정상 sequence map을 갱신하지 않았다.',
        '원래 t0은 과거 map이 없고 t25/65의 자연 가시량은0–5pixel이었다. 이를 visible-motion 탐지율의 지원 분모로 세지 않았다. 별도 Development 진단에서 실제 t0 baseline을 tick0에 먼저 처리하고 같은 물리 t0의 기존 이동 RGB-D를 tick1, reappearance를 tick2에 제시했다. 실제 sensor source의 물리 t와 알고리즘 진단 tick을 CSV에 분리했다. Map/Gate rule은 그대로이며 새 독립 Episode나 Blind Final로 세지 않는다. 위 서술 통계는 두 arm을 포함하고 control_arm 열로 분리 가능하다.',
        link('초기 map을 가진 별도 t0 진단','sources/'+state['motion_warm_id']+'/motion_manifest.json'),
        link('이동·숨김 계약/분기 provenance','sources/'+state['motion_id']+'/motion_manifest.json')+' · '+link('모든 이동 row','summary/motion_controls.csv'),
        '## 9. Oracle 원인 분리',
        table(['표현','cap128 Mean mm','p95 mm','Purity','Current coverage','Historical coverage','Current rim'],[[n,fmt(s(n)['surface_distance_mean_mm']),fmt(s(n)['surface_distance_p95_mm']),pct(s(n)['source_object_purity']),pct(s(n)['current_visible_coverage']),pct(s(n)['historical_coverage']),pct(s(n)['current_rim_coverage'])] for n in ('B1','B2','O1','O2','O3')]),
        'O1/O3는 GT-visible segmentation만 바꾸고 실제 depth·robot filter·sampling·budget을 유지했다. O2/O3는 과거 관측 node의 pose 운반만 GT relative motion으로 바꿨다. Oracle을 estimated 결과에 합산하지 않았다. Mask oracle 개선은 association/segmentation 병목의 근거이고, motion oracle만의 차이는 tracking 영향의 진단이다. 정지 sequence에서 motion oracle과 estimated가 비슷해도 실제 이동 대응이 검증됐다는 뜻은 아니다.',
        table(['Warm t0 visible ±10mm','분기 수','평균 pose Δ error mm','평균 surface mm'],[[n,len(rr),fmt(average(rr,'pose_delta_error_mm')),fmt(average(rr,'surface_mean_mm'))] for n in ('B1','B2','O1','O2','O3') for rr in [[r for r in (motions if n in ('B1','B2') else oracle_motion) if r['mode']==n and r['condition'] in ('box_x-10','box_x+10','box_y-10','box_y+10') and r['physical_valid'] and (n not in ('B1','B2') or r.get('control_arm')=='baseline_initialized_t0_same_physical_state')]]]),
        link('실제 이동의 별도 Mask/Motion Oracle','summary/oracle_motion_metrics.csv'),
        '이 동결 후보에서는 강한 memory admission 후의 source purity가 이미 높아 Mask Oracle만으로 cap128 coverage가 회복되지 않았다. GT Motion Oracle에서 coverage가 커지는 것은 pose 추정뿐 아니라 registration 성공·confidence·integration 허용 경로의 영향을 포함한다. O3의 cap128에서도 현재/경계 coverage가 원본 B0 수준에 이르지 못해 sampling/fusion/admission과 point budget 문제가 남는다. B1도 pose 운반은 없지만 공통 registration consistency gate를 사용하므로 모든 Static World Map 방법에 대한 NO-GO로 일반화하지 않는다.',
        '## 10. 사전 Gate·후보 채택',
        '절대 mean≤8mm/p95≤20mm, B0 대비 mean+2mm/p95+4mm 이내, purity≥95%/drop≤2%p, estimated ghost≤5% 및 평균4node, historical coverage gain≥5%p, current coverage drop≤3%p, current-observed rim drop≤5%p, aperture crossing0 및 false bridge≤5%를 native/cap128/matched-N에 적용했다. 이동 detection≥90%, reappearance surface≤8mm와 hidden sensor control도 검사했다. 기준은 최초 config에 고정했으며 결과를 보고 완화하지 않았다.',
        table(['후보','판정','실패 항목 수','미지원 항목 수','실패 예'],[[d['candidate'],d['decision'],sum(c['status']=='FAIL' for c in d['criteria']),sum(c['status']=='NOT_TESTED' for c in d['criteria']),'; '.join(c['criterion']+'('+str(c['budget'])+')' for c in d['criteria'] if c['status']=='FAIL')[:360]] for d in gate['candidates']]),
        link('모든 criterion·값·허용폭','summary/gate_decision.json')+' · '+link('실행 전 protocol','preregistered_protocol.json'),
        'B1/B2 모두 NO-GO다. B_SAFE는 결과를 본 뒤 Rule을 추가하지 않고 NOT_EXECUTED로 남겼다. 사후 원인 분석과 계약 bookkeeping 수정은 선택된 성능 개선으로 합산하지 않았다.',
        '## 11. Plan A/B Graph·GNN·Probe·Independent Final',
        'Paired graph 생성과 Peg/Robot76/point budget/feature/edge 계약 검증은 실행했다. Geometry Gate 실패로 Full E0/E1 2,000-update 비교, absolute Box position·paired Δ·direction cosine·signed gain·readout/object-slot/raw/robot-only probe 학습은 NOT_EXECUTED다. 따라서 Geometry 차이를 GNN 정보 보존 개선이나 Policy 성공으로 해석하지 않는다.',
        '새 독립 Final84261/84262,2 Episode를 처음에 예약하고62091/62092 및09-D Final73141/73142를 제외했다. Gate를 통과한 후보가 없어 새 Final을 수집하지 않았으며 Independent Final은 NOT_TESTED다. 기존 Final을 새 Blind Final로 재사용하지 않았다. 기존 Train/Validation parent/reset group과 perturbation split은 유지했다.',
        link('예약 Final manifest','reserved_final_manifest.json')+' · '+link('Stage 실행 상태','phase_status.json'),
        '## 12. Q1–Q15 연구 질문',
        table(['질문','증거 수준','답변'],[
            ['Q1','CONFIRMED',f'정상12 Episode 최대 translation {fmt(motiongt["max_translation_from_initial_mm"],6)}mm/rotation {fmt(motiongt["max_rotation_from_initial_rad"],8)}rad. Force는 미측정.'],
            ['Q2','SUPPORTED',f'Validation cap128 mean B0/B1 {fmt(b0["surface_distance_mean_mm"])}/{fmt(b1["surface_distance_mean_mm"])}mm. p95·purity·coverage와 함께 판단해 Gate는 실패.'],
            ['Q3','SUPPORTED',f'B1/B2 mean {fmt(b1["surface_distance_mean_mm"])}/{fmt(b2["surface_distance_mean_mm"])}mm. 작은 pose error만으로 Tracking 필요성을 지지하지 않는다.'],
            ['Q4','SUPPORTED',f'과거 관측 rim cap128 B0/B1/B2 {pct(b0["historical_rim_coverage"])}/{pct(b1["historical_rim_coverage"])}/{pct(b2["historical_rim_coverage"])}. 안전한 보존의 전체 Gate는 실패.'],
            ['Q5','CONFIRMED',f'Purity B0/B1/B2 {pct(b0["source_object_purity"])}/{pct(b1["source_object_purity"])}/{pct(b2["source_object_purity"])}; estimated ghost {fmt(b1["ghost_nodes"],1)}/{fmt(b2["ghost_nodes"],1)}node.'],
            ['Q6','CONFIRMED',f'current/historical: B0 {pct(b0["current_visible_coverage"])}/{pct(b0["historical_coverage"])}, B1 {pct(b1["current_visible_coverage"])}/{pct(b1["historical_coverage"])}, B2 {pct(b2["current_visible_coverage"])}/{pct(b2["historical_coverage"])}.'],
            ['Q7','SUPPORTED','현재/과거 관측 rim·boundary 및 aperture crossing을 분리 측정했다. 모든 boundary·오염 gate를 만족하는 후보는 없다.'],
            ['Q8','SUPPORTED','실제 visible/partial ±10mm와 registration/unknown 경보·surface 오차를 측정했다. 경보와 올바른 rigid motion 추정은 구분한다.'],
            ['Q9','INCONCLUSIVE','Actual full occlusion의 sensor/map 동일성과 UNKNOWN·uncertainty 증가를 확인했다. 실제 숨겨진 이동을 알아내지 못하며 정량 uncertainty calibration의 적합성은 미검증.'],
            ['Q10','SUPPORTED','O1/O2/O3를 별도 비교했다. Mask와 motion 제거 효과를 분리하며 practical 성능으로 합산하지 않는다.'],
            ['Q11','SUPPORTED','Native/cap128/정확한 primary matched-N을 측정했다. 사전 정확도·순도·coverage 복합 Gate는 B1/B2 모두 실패.'],
            ['Q12','NOT_TESTED','Geometry Gate 실패로 E0/E1 및 readout/Δ probe 후속 학습을 진행하지 않았다.'],
            ['Q13','NOT_TESTED','새 Final84261/84262 미수집. Development 결과로 독립 일반화를 확정하지 않는다.'],
            ['Q14','SUPPORTED','B1/B2 모두 Development/Validation Geometry NO-GO; 최종 A/B GNN 우열은 미확정.'],
            ['Q15','SUPPORTED','현재 동결 구현에서 Policy 비교에 투입할 Box persistence 후보는 없다. 원인 해결은 별도 개발·사전 고정·새 검증이 필요하다.']]),
        '## 13. Stage·검증·Artifact 보호',
        table(['Stage','상태','범위'],[[k,v['status'],v['scope']] for k,v in read('phase_status.json').items()]),
        '32개 새 synthetic 계약과 기존 Plan A/09-C를 포함한 원격86개 테스트를 실행했다. Synthetic source/schema 검사만으로 실제 센서 계약 PASS를 주장하지 않고 paired tensor/hash, 실제 branch 상태, hidden sensor 및 source/future/reset/finite 검증을 추가했다. 저장 후 JSON/CSV/NPZ readback, PNG/MP4 실제 decode와 주요 frame 시각 검수를 수행한다. 상세 완료 여부는 verification.json을 따른다.',
        '운영 기록: motion evaluator 최초 launch는 unmatched parenthesis로 sample 평가 전에 실패했고 수정 후 새 run에서 실행했다. 이후 prior-map support 수를 명시하는 bookkeeping을 추가해 동일 capture·동일 map rule로 독립 평가 UUID를 만들었다. 이전 log/결과는 보존하고 Primary는 pipeline_state에 지정했다. Map 알고리즘·Gate threshold·Final seed는 결과 후 조정하지 않았다.',
        '기존09-C/Legacy/09-D source·run, 원본 모델과 dataset/checkpoint는 read-only로 취급했다. 새 opt-in graph/box_persistence 외 Production 기본값을 변경하지 않았다. 외부 publishing/upload, dependency 설치, Git commit, GRU/Action Head/BC 학습·기존 Full A/B/C 재시작·다른 process 제어는 하지 않았다. 원격 scratch는 휘발성이므로 로컬 전달본을 hash 검증했다.',
        link('저장/좌표/동일성·보호 검증','verification.json')+' · '+link('Pre/Post protection','audits/protection_audit.json')+' · '+link('파일·원격 경로 매핑','path_remap.json')+' · '+link('PNG/MP4 frame 대응','media/media_index.json'),
        '## 14. 시각 검토',
        'Blue=current, orange=estimated, red=low confidence. GT-visible gray/rim cyan은 평가 overlay다. B0/B1/B2는 같은 World view·axis/scale을 사용했다. 전체 native extent PNG와 고정 GT-extent 확대 영상을 구분하고 확대 영역 밖 point 수를 표시했다. 영상4fps는 진단 재생 속도다.',
        '!['+'Validation geometry/budget'+']('+prefix+'media/geometry_validation.png)',
        '!['+'Boundary/Ghost temporal'+']('+prefix+'media/boundary_and_ghost.png)',
        link('Temporal current vs persistent MP4','media/temporal_box.mp4')+' · '+link('Actual motion/occlusion MP4','media/motion_controls.mp4'),
        '다음 연구에서는 Main Box의 잘못된 identity/연결 성분 누적, 현재 표면과 과거 표면의 budget 경쟁, 부분 관측의 motion observability와 재관측 충돌 처리를 분리해 개선할 필요가 있다. 이 보고서는 구현된 후보의 Geometry 적합성 판단이며 Closed-loop Manipulation 성공률의 증명이 아니다.'
    ]
    paired_path=root/'summary/paired_common_support.json'
    if paired_path.exists():
        rr=json.loads(paired_path.read_text())
        block='거리와 purity는 empty에서 미정의이므로 조건마다 정의 가능한 frame 수가 다를 수 있다. 이를 보완해 B0/B1/B2 모두 nonempty인 동일 Episode/Timestep/Budget만의 paired sensitivity도 계산했다. GT purity/가시성/error로 frame을 걸러내지 않았다. 이는 사후 서술 진단이며 Gate 재선정·threshold 변경은 하지 않았다.\n\n'
        block+=table(['Common support','Budget','EP/Frames','Mean mm','p95 mm','Purity'],[[r['variant'],r['budget'],str(r['episodes'])+'/'+str(r['frames']),fmt(r['surface_distance_mean_mm']),fmt(r['surface_distance_p95_mm']),pct(r['source_object_purity'])] for r in rr if r['split']=='validation'])
        block+='\n\n'+link('동일 지원 frame paired 거리·순도','summary/paired_common_support.csv')
        index=next(i for i,s in enumerate(text) if s.startswith('## 7.'))
        text.insert(index,block)
    extrema_path=root/'summary/ghost_extrema.json'
    if extrema_path.exists():
        rr=json.loads(extrema_path.read_text())
        block=table(['Validation','Budget','Ghost mean/max','Robot contamination','Background contamination','Wrong-object contamination'],[[r['variant'],r['budget'],fmt(s(r['variant'],r['budget'])['ghost_nodes'],2)+'/'+str(r['ghost_max']),pct(s(r['variant'],r['budget'])['source_robot_contamination']),pct(s(r['variant'],r['budget'])['source_background_contamination']),pct(s(r['variant'],r['budget'])['source_wrong_object_contamination'])] for r in rr])
        index=next(i for i,v in enumerate(text) if v.startswith('## 7.'));text.insert(index,block)
    return '\n\n'.join(text)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--doc',type=Path);a=p.parse_args()
    text=compose(a.root);md=a.root/'09e_plan_b_box_persistence.md';md.write_text(text)
    css='body{font:16px/1.65 system-ui;color:#172737;background:#f4f7fa;margin:0}main{max-width:1320px;margin:auto;background:white;padding:40px}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd7e0;padding:8px}th{background:#eaf0f6}a{color:#1662a3}img,video{max-width:100%;height:auto}code{background:#eef2f5;padding:2px 4px}'
    gallery=''.join(f'<figure><video controls preload="metadata" src="media/{p.name}"></video><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in (a.root/'media').glob('*.mp4'))
    page=a.root/'plan_b_box_persistence.html';page.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Task09-E Box Persistence</title><style>'+css+'</style><main>'+markdown_html(text)+gallery+'</main></html>')
    if a.doc:
        if a.doc.exists():raise ValueError('refuse existing report overwrite')
        a.doc.write_text(compose(a.root,'../../experiments/'+a.root.name+'/'))
    for path in (md,page):register_existing_output(a.root,path,kind='reports')
    write_json(a.root/'report_identity.json',dict(run_id=a.root.name,scope='development geometry NO-GO; independent Final/GNN NOT_TESTED'))
if __name__=='__main__':main()

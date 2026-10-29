"""Build a portable factual report from retained09-D summaries and source runs."""
import argparse,json,html,re,shutil,subprocess
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json,sha256
from graph.integration_baseline.geometry09b_report import markdown_html
from utils.experiment_logger import ExperimentLogger


def fmt(v,d=3):return '—' if v is None else f'{float(v):.{d}f}'

def tab(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def compose(root,run_id,prefix=''):
    read=lambda name:json.loads((root/name).read_text())
    final=read('summary/final.json');dev=read('summary/development.json');fp=read('summary/final_probes.json');dp=read('summary/development_probes.json');decision=read('summary/suitability.json')
    state=read('pipeline_state.json');mapping=read('path_remap.json');ids={k:Path(v).name for k,v in state['completed'].items() if k!='runtime'}
    source=lambda stage,file:prefix+'sources/'+ids[stage]+'/'+file
    link=lambda title,path:f'[{title}]({prefix}{path})'
    def geom(v,oid,budget='cap128',split='evaluation'):
        return next(r for r in final['geometry'] if r['variant']==v and r['object_id']==oid and r['budget']==budget)
    def probe(kind,v,enc,oid,name='P1_readout',decoder='linear',development=False):
        data=(dp if development else fp)[kind]
        return next((r for r in data if r['variant']==v and r['object_id']==str(oid) and r['encoder']==enc and r['probe']==name and r['decoder']==decoder and r['split']==('validation' if development else 'evaluation')),None)
    diag=read('diagnostics/summary.json');compute=read('compute/compute_counts.json')['models'];conditions=read('diagnostics/condition_counts.json')
    lines=[f'# Task09-D — Plan A Geometry Suitability & Representation Benchmark',
        f'ExperimentLogger comparison UUID `{run_id}`. 기준 source HEAD `0498a64faf0f4baac3ba4d7aa5846d1e1b84f356`, branch `visual-graph`, 시작 시 clean/ahead7. 최종 판정은 **{decision["decision"]}**이며 Geometry/GNN 연구 후보에 대한 판정이다. Policy 학습·Closed-loop 성공은 평가하지 않았다.',
        'Vast.ai57640710 / RTX3090에서 CPU PhysX/SAPIEN 진단 관측, CUDA DINO/SAM2·진단 Encoder 및 CPU 기하/Probe를 실행했다. macOS는 계약 테스트·보고서·회수 검증에 사용했다. CUDA PhysX와 macOS simulator/MPS는 이번 작업에서 실행하지 않았다.',
        '## 1. 정확한 입력·Source와 보존',
        'Peg는 현재 Main+Wrist RGB-D 중 유효 관측만 사용하고, Box는 현재 Main만 사용한다. 실제 센서는 `base_camera` / `hand_camera`다. 두 관측의 Episode/timestep/timestamp/TCP를 맞추고 각각 CV Intrinsic/Extrinsic으로 World에 역투영한다. Wrist Extrinsic은 현재 mount link pose를 반영한다. 과거 Surface/Surfel, Persistent Map, Object Pose Tracking, GT/Future 기반 Point 보충은 없다. 기존 causal DINO/SAM2 memory는 3D geometry memory와 구분한다.',
        '09-B의 위치/Δ 진단,09-C의 collision external-surface 기준·robot filter·21-feature Encoder 및 Legacy 원형을 재사용했다. 과거09-C Final UUID `e7b6f9ac66c848308e28da0716bbce0c`와 Legacy `8c41299e9d3e4989bbf32ab1b894dc49`의 수치는 이번 순위와 직접 비교하지 않는다. 새 비교는 동일 관측·분할·평가 기준 안에서 수행했다.',
        '기존 Legacy manifest는 현재 소스와 그대로 호환되지 않아 거부했다. 현재 공통 소스와 Opt-in 확장을 새 manifest로 pin하고 실제 import 경로·원형 함수/RNG·G0/G1/G3 회귀를 검사했다. Production defaults나 기존 공통 인터페이스는 변경하지 않았다.',
        link('사전 등록·source guard·runtime·계약 근거','audits/index.json')+' · '+link('고정 Protocol','frozen_protocol.json')+' · '+link('원형09-B/09-C/Legacy 참고','references/index.json'),
        '## 2. Camera·Filter·Acceptance Gate',
        'Robot filter는 현재 qpos/FK와 개별 collision shape의 depth agreement/interior를 사용한다. 전체 로봇을 감싸는 큰 sphere를 사용하지 않는다. Confirmed/uncertain은 별도로 저장하고, 기존09-C 정책대로 두 영역을 clean mask에서 제외한다. Finger 근접 지표는 pad centre20mm 이내의 기하 근접이며 실제 접촉력 판정은 아니다.',
        'Wrist Peg gate는 valid-depth fraction≥0.5, clean fraction≥0.25, clean points≥6, 12mm 연결 성분의 최대 비율≥0.6, 20mm 이웃≥3인 local-support 비율≥0.65, 최대 성분 bbox 대각≤350mm, depth-supported identity conflict≤0.1을 사용한다. 채택 시 최대 성분과 identity-consistent source만 유지한다. Main 관측 존재나 Main–Wrist 중첩은 필수가 아니다. Binary tracker API에는 현재 calibrated confidence가 없어 임의 확률을 만들지 않았다. Protocol의 overlap-distance 예약 값은 별도 distance rejection으로 사용하지 않았다.',
        tab(['조건','Peg','Box','공통 규칙'],[['C0','Main current','Main current','SF + 같은 filter/dedup/3D edge'],['C1','Main+Wrist current','Main+Wrist current','기존 depth-supported semantic association, observation gate 없음'],['C2','Main+Wrist current','Main current','Wrist Box 제외'],['C3','Gate 적용 Main+Wrist current','Main current','Plan A']]),
        'C1은 기존 G3 유형의 Camera 조건을 재현하는 control이다. 이번 공통3mm dedup 후의 node 수가 과거 G3와 같다는 뜻은 아니다. 원래 G0/G1/G3 회귀 결과는 별도 보존했다.',
        '## 3. Point Representation·Budget·Graph',
        tab(['표현','선택','Feature'],[['A-SF','원래 optimized PCA5 + outer silhouette, keep_ratio0.5 + 구조/표면 선택','원래 같은 픽셀 F5'],['A-L','원형 erosion2/fallback50 → depth backprojection → pre2048 → SOR k16/std2 → FPS≤1024/camera; 융합 native≤1024/object','원형은 XYZ; 진단 비교는 같은 source의 F5를 명시적으로 추가'],['A-H','SF structural anchor 보호 + erosion/fallback/pre2048/SOR 후보에서 anchor-initialized FPS; native≤1024/object','실제 현재 Camera source F5']]),
        'Hybrid는 Legacy 최종 FPS 이전의 SOR source를 사용한다. Erosion/SOR가 structural anchor에 적용되지 않는다. Anchor가 budget보다 많으면 stable source-ID spatial FPS로 결정적으로 줄인다. 모든 Point는 실제 valid-depth pixel에서만 온다. World3mm dedup은 structural 우선, 이후 stable source identity로 실제 representative를 유지하며 추가 source ID를 연결한다. 평균 XYZ에 다른 픽셀 F5를 붙이지 않는다.',
        '첫 구현은 Hybrid surface pool에 Legacy 최종 FPS 출력을 사용했다. 이 계약 차이를 발견해 첫 실행을 Primary에서 제외하고 소스·결과를 보존했다. Blind Final 수집 전에 수정된 v2로 전체 비교를 다시 실행했다. Gate/평가 threshold와 Final seed는 바꾸지 않았다. '+link('제외 시도 기록','audits/attempt_v1_disposition.json'),
        'Native는 원래 지원 수가 다른 전체 표현 비교다. Primary는 최대128/object이며 부족한 관측은 채우지 않는다. C3 세 표현의 정확한 Matched-N은 object/frame별 min(128, 실제 지원 수)로 정하고 GT를 사용하지 않는다. Zero/작은 N의 지원 수는 별도 표시한다.',
        'Graph는 TCP/Left/Right + 현재 object nodes, 동일21 features다. 7개 temporal/status extras는 모두0이며 estimated/persistent/age/uncertainty feature가 없다. F5 normal은 해당 현재 Camera에서 World/EE로 변환하고 jump/variation은 같은 pixel의 값만 쓴다. Edge는09-C 3D k8/radius20mm, normal/tangency·witness·free-space gate, robot–object k4, Peg–Box M16과 같은 distance+unit-direction attr다. Witness reference는 선택된 현재 support라 sparse graph fragmentation 가능성이 있다. Raw EDE는 단일 camera historical packet에만 보존한다.',
        '## 4. Data·Split·조건 발생',
        '기존09-C Train8/Validation4 연속 진단 Episode와 현재 로봇이 같은 translation/repeat 분기를 재사용했다. Geometry/GNN은 매5 timestep·마지막 프레임과 분기를 저장하고, Gate/Filter는 모든 관측을 평가했다. 새 Final73141/73142는 최초 개발 결과 접근 전에 등록하고 모든 선택/threshold/weight freeze 이후에 수집했다. 기존62091/62092와 FrozenTest12를 이번 개발/Final fitting에 사용하지 않았다. Parent Episode와 perturbation의 split은 동일하다.',
        link('실제 조건·Episode 수','diagnostics/condition_counts.csv')+' · '+link('분할·Final 시간/weight guard','audits/final_gate.json'),
        '조건 횟수는 실제 저장 프레임의 횟수이며 독립 표본 수가 아니다. Robot occlusion은 GT mesh의 front-facing sample을 실제 Camera에 투영해 robot depth/segmentation이6mm 이상 앞선 경우다. Arm만의 occlusion과 finger occlusion을 완전히 분리한 지표는 아니다. Lift/translation>5mm와 gripper opening≤30mm는 평가용 annotation이고 sampling/model input이 아니다.',
        tab(['Episode','실제 프레임','Reach','Close phase','Peg lift>5mm','Main predicted missing','Wrist-only GT 복구','Robot 가림(Main Peg)'],[[r['seed'],r['frames'],r.get('expert_phase_reach',0),r.get('expert_phase_close',0),r.get('peg_lift_gt_gt_5mm',0),r.get('main_predicted_peg_missing',0),r.get('wrist_only_peg_recovered',0),r.get('robot_occludes_projected_main_peg',0)] for r in conditions if r['stage']=='final_geometry']),
        '개발1486프레임 중 Main의 clean Peg support가 없는605프레임과 GT 지지 Wrist-only 복구387프레임을 기록했다. 독립 Final209프레임에는 Main 누락이0이어서 완전 Main 누락 복구의 독립 재현은 NOT_TESTED다. Close phase는 두 Episode에서 각8프레임 있으나 opening≤30mm는0으로 구분했다. '+link('실제 개발 누락/복구 예시','natural_recovery/natural_main_missing_wrist_recovery.png')+' · '+link('별도 실제 RGB-D mask-fault 회귀','controlled_cases/case_manifest.json'),
        '## 5. 독립 Final 입력 Geometry',
        '아래는 오염·누락도 포함하는 all-input diagnostic이며 정상 입력만의 성능표가 아니다. 깨끗한 source/geometry 구간과 오염·누락별 frozen probe는 별도로 저장했다. 아래는 baseline condition, cap128/object의 Episode macro mean이다. Coverage 분모는 두 Camera의 현재 GT-visible depth support이며, collision surface 거리는09-C의 deterministic sampled external collision boundary에 대한 거리다. Per-frame median/p95, native 및 개별 Episode 값은 원표에 있다. p95 집계는 per-frame p95의 Episode 평균이며 pooled point p95가 아니다. Main-visible 전용 coverage와 전체 rim/관측 rim은 별도 표로 구분한다.']
    rows=[]
    for oid in (0,1):
        for v in ['C0_SF','C1_SF','C2_SF','C3_SF','C3_L','C3_H']:
            r=geom(v,oid);rows.append(['Peg' if oid==0 else 'Box',v,fmt(r['count'],1),fmt(r['source_object_purity']),fmt(r['surface_distance_mean_mm']),fmt(r['surface_distance_p95_mm']),fmt(r['current_both_visible_coverage_4mm']),fmt(r['boundary_coverage_4mm']),fmt(r['missing_rate'])])
    lines += [tab(['물체','표현','평균N','Purity','거리 mm','p95 mm','Visible Cov','Boundary Cov','Visible-missing rate'],rows),link('입력 품질별 Frozen Probe','quality/input_quality_probes.csv')+' · '+link('전체 Geometry JSON','summary/final.json')+' · '+link('원시 per-frame CSV',f'sources/{ids["final_geometry"]}/metrics.csv')+' · '+link('Main-visible 전용','diagnostics/main_visible_coverage.csv')]
    lines += ['## 6. Native·Matched-N·Hybrid', '동일 cap128에서 A-L은 Peg/Box 거리0.996/17.484mm로 가장 낮았고 A-H는11.797/33.225mm였다. Peg 경계 coverage는 A-H0.764, A-SF0.607, A-L0.354로 정확도와 경계 보존의 trade-off가 남았다. Box 관측 rim coverage도 A-H0.285/A-SF0.259/A-L0.160으로 절대 coverage가 낮다. Hybrid의 Peg128개 중 평균103.36개가 structural anchor여서 surface에 남는 budget이 작았다.',tab(['물체','Budget','표현','N','거리 mm','Purity','Boundary Cov'],[[('Peg' if oid==0 else 'Box'),b,v,fmt((r:=geom(v,oid,b))['count'],1),fmt(r['surface_distance_mean_mm']),fmt(r['source_object_purity']),fmt(r['boundary_coverage_4mm'])] for oid in (0,1) for b in ('native','cap128','matched_N') for v in ('C3_SF','C3_L','C3_H')]),
        tab(['Split','물체','Matched 프레임','N≥6 프레임'],[[r['split'],r['object_id'],r['frames'],r['eligible_nonempty_frames']] for r in final['matched_support']]),
        'Native의 차이를 FPS 단독 효과로 해석하지 않는다. 별도 auxiliary 비교는 같은 C3 per-camera support에서 원 support↔erosion/pre2048/SOR support, 같은 처리 support의 SF↔FPS, 같은 N을 구분한다. 이 표는 A-SF/A-L/A-H ranking과 분리했다. '+link('Preprocessing/Sampling 원표',f'sources/{ids["final_geometry"]}/ablation.csv')+' · '+link('Structural 보존과 budget','diagnostics/structural_budget.csv'),
        '## 7. Gate·Filter·Edge 실패 분석',
        tab(['Camera','물체','Finger-near confirmed+uncertain 합산 보존','지원 Episode'],[[r['camera'],r['object_id'],fmt(r['near_total_preservation']),r['near_total_preservation_support_episodes']] for r in diag['finger_near'] if r['split']=='evaluation']),
        'Main Peg의 finger-near 합산 보존율0.814는 허용 보존율0.85보다 낮다. Confirmed-only 전 프레임 보존율0.914와 달리 uncertain 제외까지 포함한5-step 표본의 결과이며, 정상 접촉 근처 표면의 과도한 제외가 남았다. Frozen 기준표를 사후 변경하지 않고 이 추가 측정 실패를 CONDITIONAL 근거로 보존한다.',
        tab(['Final seed','프레임','Gate 채택','정상 Wrist 유지율','오점 거부율','GT 지지 Wrist-only 복구','Graph 복구(5-step)','오복구','Identity conflict'],[[r['seed'],r['frames'],r['accepted'],fmt(r['correct_retention']),fmt(r['wrong_rejection']),r['wrist_only_recovery'],r['wrist_only_graph_recovery'],r['wrist_only_false_recovery'],r['identity_conflict']] for r in final['gate_per_episode']]),
        tab(['Camera','물체','정상 표면 보존','Confirmed 오제거','Uncertain 제외','Finger-near confirmed 오제거'],[[r['camera'],r['object_id'],fmt(r['object_preservation']),fmt(r['confirmed_false_removal']),fmt(r['uncertain_object_fraction']),fmt(r['contact_near_false_removal'])] for r in final['filter']]),
        'Finger-near의 confirmed+uncertain 합산 보존율은 저장 Geometry 프레임에서 추가 측정했다. All-frame confirmed-only 지표와 분모가 다르다. '+link('Finger 근접 상세','diagnostics/finger_near_removal.csv'),
        tab(['물체','표현','표면 Edge','연결 성분','Mesh unsupported','Aperture crossing','관측 Rim Cov'],[[r['object_id'],r['variant'],fmt(r['undirected_surface_edges'],1),fmt(r['connected_components'],1),fmt(r['mesh_unsupported_edges_4mm'],1),fmt(r['aperture_crossing_edges'],1),fmt(r.get('aperture_observed_rim_coverage_4mm'))] for r in final['graph'] if r['budget']=='cap128']),
        '## 8. E0/E1와 Frozen Geometry Probe',
        tab(['입력','Encoder','전체 parameter','개발 Graph 평균 object points','평균 directed edge','optimizer updates'],[[r['variant'],r['encoder'],r['parameter_count'],fmt(r['mean_input_object_points'],1),fmt(r['mean_input_edges'],1),r['optimizer_updates']] for r in compute]),
        link('실제 입력 크기·batch RNG에 따른 논리 처리량','compute/compute_counts.csv'),
        '새 진단 Encoder를 같은 Train/Validation, seed42, AdamW lr0.001/weight_decay1e-5, batch16,2,000 optimizer updates, validation50-step 및 첫 최저 loss 규칙으로 학습했다. E0는L2/d64 edge-aware, E1은 같은 node encoder/object pooling에서 message passing을 제거한다. Parameter 수는 같다고 가정하지 않는다. 기존 BC checkpoint는 신규 표현의 우열에 사용하지 않았다.',
        link('8개 모델·parameter count·checkpoint 선택',f'sources/{ids["training"]}/model_manifest.json')+' · '+link('Frozen Probe manifest',f'sources/{ids["probes"]}/probe_manifest.json'),
        'Linear/Ridge와64-hidden Small MLP는09-B/09-C 구현과 예산·validation 선택을 유지했다. Robot76-only, Raw Centroid, Raw Geometry Summary, Object Slots128, Full Readout320을 비교한다. Target은 TCP-relative actor position과 현재 visible-surface summary이며 task goal/hole/reward/action은 사용하지 않는다. Object origin과 부분 관측 centroid의 차이는 원래 존재하므로 raw centroid를 GT pose 추정기로 해석하지 않는다.']
    rr=[]
    for v in ('C0_SF','C3_SF','C3_L','C3_H'):
        for enc in ('E0','E1'):
            for oid in (0,1):
                va=probe('position',v,enc,oid,development=True);fa=probe('position',v,enc,oid);de=probe('delta',v,enc,oid)
                rr.append([v,enc,'Peg' if oid==0 else 'Box',fmt(va['mean_mm'] if va else None),fmt(fa['mean_mm'] if fa else None),fmt(fa.get('p95_mm') if fa else None),fmt(de['delta_error_mm'] if de else None),fmt(de['direction_cosine'] if de else None),fmt(de['signed_gain'] if de else None)])
    controls=[]
    for oid in (0,1):
        for name,enc,v in [('P0_robot76','input_only','C0_SF'),('RawCentroidDirect','input_only','C3_SF'),('P4_centroid','input_only','C3_SF'),('P5_geometry_summary','input_only','C3_SF'),('P3_object_slots','E0','C3_SF'),('P3_object_slots','E1','C3_SF'),('P1_readout','E0','C3_SF'),('P1_readout','E1','C3_SF')]:
            for dec in (['unlearned'] if name=='RawCentroidDirect' else ['linear','mlp']):
                pp=probe('position',v,enc,oid,name,dec);dd=probe('delta',v,enc,oid,name,dec)
                controls.append(['Peg' if oid==0 else 'Box',name,enc,dec,fmt((pp or {}).get('mean_mm')),fmt((dd or {}).get('delta_error_mm')),fmt((dd or {}).get('direction_cosine')),fmt((dd or {}).get('signed_gain'))])
    lines += [tab(['입력','Encoder','물체','Val 위치 mm','Final 위치 mm','Final p95 mm','Final Δ mm','Cosine','Gain'],rr),
        tab(['물체','C3-SF Probe/공통 Robot control','Encoder','Decoder','Final 위치 mm','Final Δ mm','Cosine','Gain'],controls),
        '위 표는 frozen Linear P1의 Episode macro mean이다. Paired Δ는 같은 robot state에서 actor pose만 바꾸고 새 RGB-D를 렌더한 kinematic intervention에서 계산한다. 물리 step이나 새 접촉력 검증은 아니며 physically invalid branch를 제외했다. MLP·축별 MAE/R²·Episode 분포·Repeat·hidden/missing strata는 원표에 있다. '+link('Final 모든 위치 Probe','summary/final_probes_position.csv')+' · '+link('Final 모든 Δ Probe','summary/final_probes_delta.csv')+' · '+link('현재 표면 summary 오차','summary/final_probes_surface_summary.csv'),
        '## 9. 사전 고정 기준과 적합성 판정',
        tab(['기준','실측','사전 기준','판정','영역'],[[r['criterion'],fmt(r['value']),fmt(r['limit']),('PASS' if r['pass_'] is True else 'FAIL' if r['pass_'] is False else 'NOT_TESTED'),r['scope']] for r in decision['criteria']]),
        f'판정은 **{decision["decision"]}**이다. Perception은 Peg coverage와 Box Wrist 차단이 SUPPORTED이나 finger-near 제외와 Final Main-missing 미지원이 남는다. Geometry는 A-L 정확도와 A-H 경계 사이 trade-off로 CONDITIONAL이다. Representation은 E1이 대체로 낫지만 Box 위치71.76mm/Δ33.81mm 및 Peg Δ의 Raw 대비 열화가 남아 CONDITIONAL이다. 없는 support/조건을 성공으로 세지 않았다. Final은2 Episode·학습 seed1개이므로 모집단 일반화나 Closed-loop manipulation 성공을 확정하는 근거는 아니다. '+link('판정 JSON','summary/suitability.json'),
        '## 10. Q1–Q12 답변']
    c0=geom('C0_SF',0);c2=geom('C2_SF',0);c3=geom('C3_SF',0);b1=geom('C1_SF',1);b2=geom('C2_SF',1);b3=geom('C3_SF',1)
    hybrid_ok=all(geom('C3_H',o,'matched_N')['surface_distance_mean_mm']<=geom('C3_L',o,'matched_N')['surface_distance_mean_mm'] and geom('C3_H',o,'matched_N')['boundary_coverage_4mm']>=geom('C3_SF',o,'matched_N')['boundary_coverage_4mm'] for o in (0,1))
    hdetails='; '.join(('Peg' if o==0 else 'Box')+f" matched H/L 거리 {fmt(geom('C3_H',o,'matched_N')['surface_distance_mean_mm'])}/{fmt(geom('C3_L',o,'matched_N')['surface_distance_mean_mm'])}mm, H/SF 경계 {fmt(geom('C3_H',o,'matched_N')['boundary_coverage_4mm'])}/{fmt(geom('C3_SF',o,'matched_N')['boundary_coverage_4mm'])}" for o in (0,1))
    encoder_details='; '.join(('Peg' if o==0 else 'Box')+': '+', '.join(e+f" 위치{fmt((probe('position','C3_SF',e,o) or {}).get('mean_mm'))}/Δ{fmt((probe('delta','C3_SF',e,o) or {}).get('delta_error_mm'))}mm" for e in ('E0','E1')) for o in (0,1))
    raw_details='; '.join(('Peg' if o==0 else 'Box')+f" RawDirect Δ{fmt((probe('delta','C3_SF','input_only',o,'RawCentroidDirect','unlearned') or {}).get('delta_error_mm'))}mm vs E0/E1 Δ"+'/'.join(fmt((probe('delta','C3_SF',e,o) or {}).get('delta_error_mm')) for e in ('E0','E1'))+'mm' for o in (0,1))
    answers=[['Q1','SUPPORTED' if c3['current_both_visible_coverage_4mm']>c0['current_both_visible_coverage_4mm'] else 'INCONCLUSIVE',f"Peg current-visible coverage {fmt(c0['current_both_visible_coverage_4mm'])} → {fmt(c3['current_both_visible_coverage_4mm'])}; purity/누락과 함께 평가했다."],
        ['Q2','CONFIRMED',f"Box Wrist source C1 {fmt(b1['camera_wrist_sources'],1)} → C3 {fmt(b3['camera_wrist_sources'],1)}. 거리 {fmt(b1['surface_distance_mean_mm'])} → {fmt(b3['surface_distance_mean_mm'])}mm; purity {fmt(b1['source_object_purity'])} → {fmt(b3['source_object_purity'])}."],
        ['Q3','CONFIRMED',f"C1→C2는 Wrist Box 제외만, C2→C3는 Wrist Peg gate 추가만 바꾼 동일 SF/edge control. Peg coverage C2 {fmt(c2['current_both_visible_coverage_4mm'])} → C3 {fmt(c3['current_both_visible_coverage_4mm'])}."],
        ['Q4','SUPPORTED',hdetails+'; 단일 지표 우승은 정하지 않는다. Native/cap128/Matched-N과 validation probe를 함께 비교한다.'],
        ['Q5','SUPPORTED',f"Peg matched-N SF/L/H 거리 {fmt(geom('C3_SF',0,'matched_N')['surface_distance_mean_mm'])}/{fmt(geom('C3_L',0,'matched_N')['surface_distance_mean_mm'])}/{fmt(geom('C3_H',0,'matched_N')['surface_distance_mean_mm'])}mm. "+'정확한 N과 eligible support는 §6에 제시했다.'],
        ['Q6','SUPPORTED' if hybrid_ok else 'INCONCLUSIVE',hdetails+'. '+('거리/경계 두 지표에서 원형 장점을 모두 유지했다.' if hybrid_ok else '두 물체의 거리/경계 장점을 모두 유지한 것은 아니다. 구현상 anchor 보존과 실제 geometry 우위는 구분한다.')],
        ['Q7','SUPPORTED',encoder_details+'. 동일 입력 비교이며 보편적 Encoder 우위로 확대하지 않는다.'],
        ['Q8','SUPPORTED',raw_details+'. Actor-origin bias와 decoder 일반화를 포함하므로 정보이론적 보존율로 해석하지 않는다.'],
        ['Q9','SUPPORTED',f"73141/73142의2 Episode를 freeze 후 no-fit 적용. C3-SF/E1 Peg 위치 Val {fmt((probe('position','C3_SF','E1',0,development=True) or {}).get('mean_mm'))} → Final {fmt((probe('position','C3_SF','E1',0) or {}).get('mean_mm'))}mm; Box {fmt((probe('position','C3_SF','E1',1,development=True) or {}).get('mean_mm'))} → {fmt((probe('position','C3_SF','E1',1) or {}).get('mean_mm'))}mm. 일반화 범위는 제한적이다."],
        ['Q10','SUPPORTED',decision['decision']+' — 사전 고정된 threshold별 판정은 §9. Geometry/GNN 후보 판정이며 Policy 성공 판정은 아니다.'],
        ['Q11','SUPPORTED',f"Main Peg finger-near 합산 보존0.814, Box 표면거리85.195mm와 위치복원71.757mm가 남는다. C3-SF Peg/Box purity {fmt(c3['source_object_purity'])}/{fmt(b3['source_object_purity'])}, visible-missing {fmt(c3['missing_rate'])}/{fmt(b3['missing_rate'])}. "+raw_details+'. Filter·rim·연결 성분 실패도 별도로 보존하며 단일 원인으로 확정하지 않는다.'],
        ['Q12','SUPPORTED','Plan B는 NOT_TESTED. 향후 비교는 C3 current-only를 기준으로 Peg 입력·Box Main 관측·filter/sampling/budget/edge/encoder/split을 고정하고 Box memory 유무만 바꿔야 한다. 현재/전체 collision 및 현재-visible 평가 분모도 구분해야 한다.']]
    lines += [tab(['질문','증거 수준','답변'],answers),
        '## 11. 시각 검토·Artifact·실행 범위',
        '다음 정책 학습의 즉시 GO는 아니다. C3+A-SF+E1을 표현 진단의 주 기준으로, C3+A-L+E1을 표면 정확도 기준으로 유지할 근거는 있다. Hybrid를 기본 표현으로 채택할 근거는 부족하다. 추가 개발에서는 정상 finger-near surface 제외, Main Box outlier/오염, Anchor budget, Raw→Readout Δ 복원과 완전 Main 누락의 새 독립 Episode를 우선 검증해야 한다. 이번 결과에 맞춘 sampling 변경이나 추가 architecture 학습은 하지 않았다.',
        link('동일 축 확대 영상·프레임 대응','media_zoom/media_index.json')+' · '+link('고정 원자료의 unique source export','canonical_provenance/canonicalization_audit.json'),
        '전체 축 영상에는 멀리 떨어진 outlier까지 유지했다. 추가 확대 영상은 Episode의 GT collision extent+40mm로 축을 고정하고 화면 밖 object node 수를 표시한다. 이는 시각 검토용 범위이며 Point/Graph/지표를 자르거나 바꾸지 않았다.',
        'SF/SOR support 결합의 raw provenance에는 반복·공유된 auxiliary source ID가 있었다. 41,348개 중복 support entry를 별도 unique-source view에서 정규화했다(여러 budget/표현의 합계이며 독립 센서 관측 수가 아니다). Primary 대표 source 충돌은0이고, 원래 Point/Graph/가중치와 Primary 첫-source purity는 바뀌지 않았다. Raw trace와 추가 export를 모두 보존했다.',
        link('독립 Final Geometry 그림','figures/final_geometry.png')+' · '+link('Native/Matched-N 비교','figures/budget_comparison.png')+' · '+link('Validation Probe','figures/validation_probes.png')+' · '+link('Final Probe','figures/evaluation_probes.png')+' · '+link('Readout/XYZ 복원','figures/readout_reconstruction.png'),
        link('Camera/Point/Edge 영상과 프레임 대응','media/media_index.json')+' · '+link('JSON/CSV/NPZ·source·row·보호 검증','verification.json'),
        '각 영상은 실제 Episode/seed/timestep/camera/representation을 표시한다. Points-only와 동일 cap point+edge, 정확한 Matched-N, native PNG, 현재 GT-visible overlay를 같은 World axis/view로 제시한다. 영상의 재생 속도는 평가 프레임을 위한4fps이며 simulator20Hz 실시간 성능 주장과 다르다.',
        'Phase0 Source/runtime/split PASS; Phase1 Camera/filter/gate 구현 PASS, 실제 보존 기준 실패/미지원 stratum은 위 판정대로 PARTIAL; Phase2 세 표현/native/matched/auxiliary 비교 PASS; Phase3 E0/E1 8개×2,000 updates와 frozen probes PASS; Phase4 새 독립 Final/no-fit 적용 PASS; Phase5 검증·보고서 PASS. Geometry·representation의 수치 적합성은 실행 완료와 별도로 §9에서 판정한다.',
        '실행하지 않은 항목: Task09-D Persistent Geometry 조건의 구현·성능평가(Plan B 포함), GRU/Action Head/BC/DAgger/Recovery 학습, learned-policy closed-loop rollout, GT Goal/Hole sampling/input, 새 tracker/foundation 학습,20Hz 최적화, dependency 설치, Git commit, HF/W&B 외부 publishing. 기존 split/dataset/checkpoint/실험 결과와 Production 기본값을 변경하지 않았다.',
        '공통 ExperimentLogger UUID/project/group 규칙과 로컬 artifact 등록을 재사용했다. 원격 격리 source에는 .git가 없어 child metadata commit이 null일 수 있으며 초기 HEAD/배포 hash/source snapshot을 별도로 보존했다. 원격 scratch는 휘발성 영역이므로 전달본에 회수하고 SHA를 대조했다. 원래 provenance 절대 경로는 수정하지 않고 path_remap으로 연결한다.',
        link('전체 stage UUID','pipeline_state.json')+' · '+link('Source 경로 매핑','path_remap.json')+' · '+link('Source·Dependency·수정 diff','audits/index.json')]
    return '\n\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);a=p.parse_args()
    cfg=dict(task='09-D',stage='reviewed_comparison_report',source_delivery=str(a.delivery),no_publish=True)
    with ExperimentLogger(team='integration',config=cfg,seed=42,model='task09d_plan_a_suitability_report',job_type='evaluate') as log:
        root=log.directory
        for child in a.delivery.iterdir():
            target=root/child.name
            if child.is_dir():shutil.copytree(child,target)
            else:shutil.copy2(child,target)
        text=compose(root,log.run_id);(root/'09d_plan_a_geometry_suitability.md').write_text(text)
        css='body{font:15px/1.7 system-ui,sans-serif;margin:0;background:#f4f6f9;color:#18263a}main{max-width:1400px;margin:auto;background:white;padding:40px}h1,h2{line-height:1.25}h2{margin-top:46px;border-bottom:2px solid #c9d8e8;padding-bottom:10px}table{border-collapse:collapse;display:block;overflow-x:auto;width:100%;font-size:13px}th,td{padding:9px;border:1px solid #d3deea;text-align:left}th{background:#e9f1fa}code{background:#edf1f5;padding:2px 5px}a{color:#155e9d}img,video{max-width:100%;height:auto;border:1px solid #ccd7e3;margin:12px 0}'
        galleries=''.join(f'<figure><img loading="lazy" src="figures/{f.name}" alt="{html.escape(f.stem)}"><figcaption>{html.escape(f.stem)}</figcaption></figure>' for f in (root/'figures').glob('*.png'))
        galleries+=''.join(f'<figure><video controls preload="metadata" src="{f.parent.name}/{f.name}"></video><figcaption>{html.escape(f.stem)}</figcaption></figure>' for folder in ('media_zoom','media','controlled_cases') for f in (root/folder).glob('*.mp4'))
        (root/'plan_a_geometry_suitability.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Task09-D Plan A</title><style>'+css+'</style><main>'+markdown_html(text)+'<h2>검토용 그림·영상</h2>'+galleries+'</main></html>')
        doc=Path('docs/visual-graph-integration/09d_plan_a_geometry_suitability.md');assert not doc.exists();doc.write_text(compose(root,log.run_id,'../../experiments/'+log.run_id+'/'))
        write_json(root/'report_identity.json',dict(run_id=log.run_id,report=str(doc),source_heads='0498a64faf0f4baac3ba4d7aa5846d1e1b84f356',new_commit_created=False))
        for f in [root/'09d_plan_a_geometry_suitability.md',root/'plan_a_geometry_suitability.html',root/'report_identity.json',doc]:log.link_output(f,kind='reports')
        print('COMPARISON_RUN='+str(root),flush=True)
if __name__=='__main__':main()

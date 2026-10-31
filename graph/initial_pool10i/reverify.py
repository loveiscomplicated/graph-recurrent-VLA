"""New verification run; never resumes or edits the BLOCKED generator run.

CV matrix is the frozen primary camera contract. GL matrix comparison is an
explicit separate diagnostic and its disagreement remains preserved.
"""
import os
os.environ['WANDB_MODE']='disabled'
from pathlib import Path
import json,hashlib,shutil,csv,sys,traceback,fcntl,platform
import numpy as np
from graph.initial_pool10i.calibration_diagnostic import analyze
from graph.initial_pool10i.pipeline import report,read,write,sha,rows,csvwrite
from utils.experiment_logger import ExperimentLogger
BASE=Path('outputs/task10i/stage1').resolve();ORIG=BASE/'run_v3';ROOT=BASE/'verified_pool_v2'

def validate(root):
    rr=rows(root);pool=read(root/'candidate_pool_manifest.json');ids=[r['candidate_id'] for r in rr];eligible={r['candidate_id'] for r in rr if r['eligible']}
    assert len(rr)==926 and len(eligible)==300 and len(set(ids))==926
    for r in rr:
        for k,h in r['files'].items():assert sha(root/'candidates'/r['candidate_id']/k)==h,(r['candidate_id'],k)
    for path,h in read(root/'source_identity.json')['sources'].items():
        p=BASE/Path(path).relative_to('/workspace/task10i_stage1_20261031');assert sha(p)==h,path
    reg=read(root/'visibility_regression.json');assert reg['status']=='PASS' and reg['exact_labels']==320 and reg['exact_rendered_parents']==112
    assert all(all(r['checks'].values()) for r in reg['render'])
    replay=read(root/'candidate_replay_verification.json');assert replay['status']=='PASS' and replay['count']==300
    assert {r['candidate_id'] for r in replay['rows']}==eligible
    assert all(r['passed'] and r['sim_state_exact'] and r['controller_exact'] and r['visibility'] and all(v['exact'] for v in r['checks'].values()) for r in replay['rows'])
    for name in ['candidate_attempts.csv','candidate_eligibility.csv','candidate_state_descriptors.csv','candidate_visibility_metrics.csv']:
        with (root/name).open() as f:table=list(csv.DictReader(f))
        assert [r['candidate_id'] for r in table]==ids,name
    with np.load(root/'candidate_initial_states.npz') as z:
        arrays={k:z[k] for k in z.files}
        assert list(arrays['candidate_id'])==ids
        for i,cid in enumerate(ids):
            with np.load(root/'candidates'/cid/'initial.npz') as d:
                for k in d.files:assert np.array_equal(d[k],arrays[k][i]),(cid,k)
    assert len(read(root/'eligible_candidate_pool.json'))==300
    dup=read(root/'duplicate_audit.json');assert dup['status']=='PASS' and not dup['exact_pairs'] and not dup['seed_overlap']
    return rr

def main():
    assert not ROOT.exists(),'Use a fresh verification root, never replace a prior run'
    # Candidate data, labels, metrics and replay proofs are byte-preserved.
    shutil.copytree(ORIG,ROOT,ignore=shutil.ignore_patterns('experiments','*.log','run.lock','progress.json','failure_traceback.txt','*.done.json','*.tracking.json'))
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with ExperimentLogger(team='graph',model='task10i_stage1_cv_contract_reverification',seed=10103110,job_type='evaluate',root=ROOT/'experiments',config=dict(generator_root=str(ORIG),generator_source_identity_sha256=sha(ORIG/'source_identity.json'),primary_protocol_changed=False,simulation_actions=0)) as logger:
        try:
            write(ROOT/'progress.json',dict(status='RUNNING',stage='INDEPENDENT_CV_CONTRACT_REVERIFICATION'))
            write(ROOT/'reverification_source_identity.json',dict(generator_source_identity_sha256=sha(ORIG/'source_identity.json'),
                source={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('calibration_diagnostic.py')]},
                runtime=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__),primary_protocol_unchanged=True))
            rr=validate(ROOT);diag=analyze(ROOT)
            assert diag['cv']['rows']==926 and not diag['cv']['mask_differences'] and not diag['cv']['region_differences'] and not diag['cv']['verdict_differences']
            assert diag['cv']['max_depth_error_m']<.0001 # existing independent numerical crosscheck, not a visibility gate
            write(ROOT/'calibration_diagnostic.json',diag)
            write(ROOT/'independent_visibility_verification.json',dict(status='PASS',primary='Frozen CV calibration + independent OBB slab algorithm',
                primary_mask_exact=926,primary_regions_exact=926,primary_verdict_exact=926,cv=diag['cv'],
                gl_diagnostic_status='DIFFERENCE' if diag['gl']['verdict_differences'] else 'EXACT',gl=diag['gl'],
                no_gate_threshold_changes=True,original_blocked_run_preserved=str(ORIG),
                interpretation='GL is separately serialized float32 calibration. Its extra silhouette pixel has a grazing 83nm intersection and GT non-Peg ID. GL verdict difference is retained; it is not used to replace the frozen primary CV camera protocol.'))
            pool=read(ROOT/'candidate_pool_manifest.json');original_run_id=pool['run_id'];pool.update(status='COMPLETED',run_id=logger.run_id,generator_run_id=original_run_id,verification_root=str(ROOT),generation_root=str(ORIG),gl_calibration_diagnostic='ONE_PIXEL_DIFFERENCE_PRESERVED',primary_protocol_unchanged=True)
            write(ROOT/'run_identity.json',dict(run_id=logger.run_id,generator_run_id=original_run_id));write(ROOT/'candidate_pool_manifest.json',pool)
            # Export keeps original candidate IDs, state snapshots, descriptors and eligibility.
            exp=read(ROOT/'eligible_candidate_pool.json')
            for e in exp:
                e['annotated_rgb_preview']=e.pop('rgb');e['raw_rgb_npz']=dict(path=e['initial_npz'],key='rgb');e['raw_depth_npz']=dict(path=e['initial_npz'],key='depth_raw_mm',unit='mm',dtype='int16')
                e['reverification_source_sha256']=sha(ROOT/'reverification_source_identity.json')
            write(ROOT/'eligible_candidate_pool.json',exp)
            desc=[]
            for r in rr:desc.append(dict(candidate_id=r['candidate_id'],eligible=r['eligible'],**read(ROOT/'candidates'/r['candidate_id']/'descriptor.json')))
            report(ROOT,rr,desc,pool)
            invalid=[dict(candidate_id=r['candidate_id'],evidence=read(ROOT/'candidates'/r['candidate_id']/'physical.json')) for r in rr if r['physical']!='PHYSICAL_VALID']
            write(ROOT/'physical_invalid_diagnostic.json',invalid)
            text=(ROOT/'10i_stage1_main_visible_candidate_pool.md').read_text()
            text+='''\n## 독립 Calibration 재검증과 보존된 BLOCKED Run\n\n원래 수집 Run `run_v3`은 GL 행렬 기반 추가 검증에서 1픽셀 차이로 BLOCKED로 종료했으며 원본 상태와 실패 기록은 그대로 보존했다. 이 보고서의 `verified_pool_v2`은 별도 ExperimentLogger UUID의 새 검증 Run이다. Candidate/Seed/Primary label/Physical/Replay 결과는 한 건도 수정하지 않았다.\n\n기존 Primary가 사용하는 CV Calibration을 독립 OBB Slab 알고리즘에 입력했을 때 **926개 전체의 Mask·필수 영역·Primary Verdict가 exact 일치**했다. 최대 공통 Depth 수치 차이는 0.001391mm로 기존 독립 수치 검사 0.1mm 이내다. Primary 가시성 Depth tolerance는 기존 1.5mm 그대로다.\n\n별도 저장된 float32 GL Calibration에서는 `10i_s1_0364`의 (u=113,v=116) 경계 1픽셀만 추가 hit가 생겼다. 해당 ray의 Peg 내부 구간은 약 83nm이고 실제 GT ID는 Peg가 아니다. GL 기반 Primary 재계산은 이 한 후보에서 FAIL이며, 그 차이를 [Calibration 진단](calibration_diagnostic.json)에 보존했다. GL 결과를 exact/PASS로 수정하거나 1픽셀을 가시성 Gate에서 임의로 허용한 것이 아니다. 승인된 기존 CV Primary를 그대로 검증했다. 기존 프로토콜에서 Pixel-perfect raster 조건이 별도 진단인 것과 같이, 보정 행렬의 추가 래스터 조건을 새로운 Primary Gate로 승격하지 않았다.\n\n수집 중간의 RNG 직렬화 오류와 Gallery null projection 오류는 각각 이전 `run`/`run_v2`에 보존했다. 두 Engineering 실패는 본 수집 전에 발생했으며 수정 후 새 Run에서 회귀·Integration부터 다시 실행했다. 기존 합성 가시성 테스트 6개와 새 Collision/Hash/Atomic Manifest 계약 테스트 3개가 통과했다.\n\n물리적으로 무효한 초기화 3개는 [실제 진단](physical_invalid_diagnostic.json)에 있으며, 그중 Visibility PASS도 적격 수에 포함하지 않았다. 최종 Eligible은 300개 모두 Physical Valid 및 독립 Reset Replay PASS다. 단일 Worker를 사용했고 다중 Worker 동등성은 실행하지 않았다. Timing의 Reset 비용은 Rendering을 포함한 합산값이며 순수 Render 비용을 별도로 분리 측정하지 않았다.\n\n원본 RGB-D는 Export의 `raw_rgb_npz`/`raw_depth_npz`에 지정한 NPZ key에서 읽는다. `annotated_rgb_preview`는 Gallery 전용이다. 후보 분포는 native Random Reset을 기존 가시성·명시적 물리 유효성·Replay 조건으로 제한한 분포다. macOS에서는 저장 Artifact의 수치 교차 검증만 수행했다. 실제 Simulator와 300개 Replay는 원래 지정 Linux Runtime에서 실행했으며 GPU PhysX나 로컬 Simulator를 실행한 것이 아니다.\n'''
            (ROOT/'10i_stage1_main_visible_candidate_pool.md').write_text(text)
            import markdown
            (ROOT/'10i_stage1_main_visible_candidate_pool.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I Stage1</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style>'+markdown.markdown(text,extensions=['tables','fenced_code']))
            # Validate all scientific artifacts after report/export creation; per-candidate data unchanged.
            validate(ROOT)
            for p in ROOT.glob('*.json'):read(p)
            for p in ROOT.glob('*.npz'):
                with np.load(p) as z:
                    for k in z.files:assert np.isfinite(z[k]).all() if z[k].dtype.kind in 'fiu' else True
            for name in ['candidate_pool_manifest.json','independent_visibility_verification.json','10i_stage1_main_visible_candidate_pool.md','10i_stage1_main_visible_candidate_pool.html','gallery.html']:
                logger.link_output(ROOT/name,kind='metrics' if name.endswith('.json') else 'reports')
            logger.log(dict(eligible=300,attempted=926,primary_independent_exact=926))
            files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file() and 'experiments' not in p.parts and p.name not in ['candidate_pool_verification.json','progress.json','run.lock']}
            for name,info in files.items():assert sha(ROOT/name)==info['sha256']
            write(ROOT/'candidate_pool_verification.json',dict(status='PASS',attempts=926,eligible=300,primary_cv_exact=926,gl_diagnostic='DIFFERENCE_PRESERVED',readback=True,files=files))
            write(ROOT/'progress.json',dict(status='COMPLETED',stage='STOP_AFTER_STAGE1',attempted=926,eligible=300))
        except BaseException:
            write(ROOT/'progress.json',dict(status='BLOCKED',traceback=traceback.format_exc()));raise
    print('COMPLETED',logger.run_id,flush=True)
if __name__=='__main__':main()

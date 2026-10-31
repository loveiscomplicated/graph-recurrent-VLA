"""Read-only remote delivery and independent local manifest/CSV/NPZ verification."""
import csv,json,hashlib,re,subprocess,shutil
from pathlib import Path
import numpy as np
BASE=Path('outputs/task10i/stage1');ROOT=BASE/'verified_pool_v2'
REMOTE='/workspace/task10i_stage1_20261031/'
HOST='root@104.233.209.157'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    BASE.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-az','-e','ssh -p 13159','--exclude','__pycache__','--exclude','audit','--exclude','*.tmp',HOST+':'+REMOTE,str(BASE)+'/'],check=True)
    shutil.copytree('outputs/task10g/initial_visibility_audit',BASE/'audit',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.tar.gz','experiments','readback_experiments'))
    progress=read(ROOT/'progress.json');assert progress['status'] in ['COMPLETED','INSUFFICIENT_POOL'],progress
    verification=read(ROOT/'candidate_pool_verification.json');assert verification['status']=='PASS'
    for name,info in verification['files'].items():
        p=ROOT/name;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
    source=read(ROOT/'source_identity.json')
    for path,h in source['sources'].items():
        relative=Path(path).relative_to(REMOTE);assert sha(BASE/relative)==h,str(relative)
    tables={}
    for name in ['candidate_attempts.csv','candidate_eligibility.csv','candidate_state_descriptors.csv','candidate_visibility_metrics.csv']:
        with (ROOT/name).open() as f:tables[name]=list(csv.DictReader(f))
    n=progress['attempted'];ids=[r['candidate_id'] for r in tables['candidate_attempts.csv']]
    assert len(ids)==len(set(ids))==n
    for name,rows in tables.items():assert [r['candidate_id'] for r in rows]==ids,name
    export=read(ROOT/'eligible_candidate_pool.json');assert len(export)==progress['eligible']
    schema=dict(export='verified_pool_v2/eligible_candidate_pool.json',sha256=sha(ROOT/'eligible_candidate_pool.json'),raw_rgb=dict(path_field='initial_npz',npz_key='rgb',dtype='uint8',shape=[128,128,3]),raw_depth=dict(path_field='initial_npz',npz_key='depth_raw_mm',dtype='int16',unit='mm',invalid='<=0'),annotated_preview_field='annotated_rgb_preview',split='UNASSIGNED',expert_success='NOT_TESTED')
    (BASE/'eligible_candidate_pool_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    eligible={r['candidate_id'] for r in tables['candidate_eligibility.csv'] if r['eligible']=='True'}
    assert eligible=={r['candidate_id'] for r in export}
    for e in export:
        assert e['split']=='UNASSIGNED' and e['expert_success']=='NOT_TESTED'
        for key in ['initial_npz','snapshot','annotated_rgb_preview','gt_visibility_evidence']:assert (ROOT/e[key]).is_file()
    with np.load(ROOT/'candidate_initial_states.npz') as z:
        arrays={k:z[k] for k in z.files}
        assert list(arrays['candidate_id'])==ids
        for i,cid in enumerate(ids):
            with np.load(ROOT/'candidates'/cid/'initial.npz') as initial:
                for k in initial.files:assert np.array_equal(initial[k],arrays[k][i]),(cid,k)
    old=np.load('outputs/task10g/initial_visibility_audit/task10g_initial_visibility_inputs_v1/first_observations.npz')
    shared_duplicate=[]
    with np.load(ROOT/'candidate_initial_states.npz') as new:
        for i,cid in enumerate(ids):
            if cid not in eligible:continue
            matches=np.flatnonzero(np.all(old['peg_pose']==new['peg_pose'][i],axis=1)&np.all(old['peg_half']==new['peg_half'][i],axis=1))
            if len(matches):shared_duplicate.append(dict(candidate_id=cid,old_indices=matches.tolist()))
    assert not shared_duplicate,shared_duplicate
    overlap=dict(status='PASS',eligible=len(eligible),old_logical_rows=320,exact_peg_pose_shape_matches=shared_duplicate,scope='No identical peg pose+shape, so no identical full physical states with old audited parents; near-duplicate full-state comparison remains Stage2 work.')
    (BASE/'existing_vs_candidate_geometry_audit.json').write_text(json.dumps(overlap,indent=2)+'\n')
    first500=tables['candidate_eligibility.csv'][:500]
    first500_pass=sum(r['visibility']=='VISIBILITY_PASS' for r in first500)
    receipt=dict(status='PASS',verified_pool_status=progress['status'],files_sha256_verified=len(verification['files']),source_files_verified=len(source['sources']),csv_npz_rows=n,eligible=progress['eligible'],first500_visibility_pass=first500_pass,first500_visibility_rate=first500_pass/len(first500),source_runtime='Linux CPU PhysX + CUDA SAPIEN renderer on 57987772; local delivery verification uses NumPy only')
    (BASE/'delivery_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    docs=Path('docs/visual-graph-integration');text=(ROOT/'10i_stage1_main_visible_candidate_pool.md').read_text()
    prefix='../../outputs/task10i/stage1/verified_pool_v2/'
    text=re.sub(r'\]\(([^)]+)\)',lambda m:']('+m[1]+')' if m[1].startswith(('http','#','/')) else ']('+prefix+m[1]+')',text)
    text+=f'\n첫 500회 Visibility PASS는 {first500_pass}/500 ({first500_pass/500:.2%})다. [기존 Parent와 신규 Eligible의 exact Peg Geometry 중복 검사](../../outputs/task10i/stage1/existing_vs_candidate_geometry_audit.json)도 통과했다.\n'
    text+='\n후보분포는 native Random Reset 분포를 기존 Main 가시성·명시한 물리 유효성·Replay 조건으로 제한한 분포다. Export의 `annotated_rgb_preview` 경로는 주석이 있는 Gallery Preview이며, 후속 분석의 실제 원본 영상은 `initial_npz` 안의 `rgb`/`depth_raw_mm` 키를 사용한다.\n\n## 인계 검증 및 Engineering 이력\n\n첫 Engineering Run은 BatchedRNG 직접 직렬화 오류, 두 번째 Run은 카메라 뒤 투영의 null 좌표를 Gallery에서 처리하지 못해 BLOCKED로 보존했다. 두 Run 모두 본 Candidate 수집 전 중단됐으며, 수정 후 run_v3에서 회귀·Integration부터 재실행했다. run_v3은 마지막 GL 진단 1픽셀 차이로 BLOCKED를 보존했고, 별도 UUID의 verified_pool에서 기존 CV 계약을 독립 전수 검증했다. 기존 Primary Protocol/Threshold 변경은 없었다.\n\n기존 합성 가시성 테스트 6개와 신규 물리/Hash/원자적 Manifest 계약 테스트 3개가 통과했다. [독립 신규 가시성 검증]('+prefix+'independent_visibility_verification.json), [로컬 인계 SHA·CSV·NPZ 전수 Readback](../../outputs/task10i/stage1/delivery_verification.json). 로컬 검증은 macOS NumPy에서 실행했으며 로컬 Simulator/CUDA를 실행한 것은 아니다. 단일 Worker를 사용했으며 다중 Worker 동등성 및 분리된 순수 Render 비용은 별도로 측정하지 않았다. Timing의 reset_including_render_s는 Reset과 Render의 합산 시간이다.\n'
    (docs/'10i_stage1_main_visible_candidate_pool.md').write_text(text)
    import html
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    (docs/'10i_stage1_main_visible_candidate_pool.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I Stage1</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style>'+body)
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

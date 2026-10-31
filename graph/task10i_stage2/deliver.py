"""Build a local readable delivery without altering immutable remote artifacts."""
import csv
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

from graph.initial_pool10i.mirror.verify import read,sha
from .report import CSS,table

REPO=Path(__file__).resolve().parents[2]


def main():
    base=REPO/'outputs/task10i/stage2';run=base/'run';delivery=base/'delivery';delivery.mkdir(exist_ok=True)
    remote=read(base/'remote_execution/independent_completion_audit_v2.json')
    local=read(base/'local_completion_audit.json')
    assert remote['status']==local['status']=='PASS'
    assert remote['prior_failed_run_selection']=='EXACT_SHA256_MATCH'
    assert remote['raw_source_parents_checked']==340
    for rel,info in read(run/'artifact_index.json')['files'].items():assert sha(run/rel)==info['sha256']
    for name in ['selection_identity.json','coverage_selection_order.json','feature_normalization.json','candidate_pool_split_manifest.json',
                 'random_R_selected.csv','diversity_D_selected.csv']:
        assert sha(run/name)==sha(base/'failed_run_v1/run'/name)
    summary=read(run/'selection_comparison_summary.json');plan=read(run/'stage3_demonstration_plan.json')
    # Fix the original figure's overlapping subplot title in a separate delivery figure.
    with (run/'random_monte_carlo_metrics.csv').open() as f:mc=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,cohort,title in zip(axs,['selection','holdout'],['Selection Pool 220','Diagnostic Holdout 60']):
        ax.hist([float(r['mean']) for r in mc if r['cohort']==cohort],bins=22,color='#b6c6d7')
        for name,color in [('R','#3978b4'),('D','#df7a20')]:
            ax.axvline(summary['monte_carlo'][cohort]['datasets'][name]['score'],color=color,label=name)
        ax.set_title(title);ax.set_xlabel('Mean nearest-set distance (lower is better)');ax.set_ylabel('Random draws');ax.legend()
    fig.suptitle('Common44 fixed: 200 Random19 draws');fig.tight_layout(rect=[0,0,1,.94])
    fig.savefig(delivery/'random_mc_labeled.png',dpi=150);plt.close(fig)
    gallery=(run/'gallery.html').read_text().replace('src="gallery/','src="../run/gallery/')
    gallery=gallery.replace('/workspace/task10i_stage1_20261031/verified_pool_v2/','../../stage1/verified_pool_v2/')
    gallery=gallery.replace('/workspace/task10i_stage1_20261031/audit/','../../../task10g/initial_visibility_audit/')
    gallery=gallery.replace('10i_stage2_diversity_aware_dataset_selection.html','10i_stage2_diversity_aware_dataset_selection.html')
    (delivery/'gallery.html').write_text(gallery)
    source_mix=[]
    for split,s in plan['structure'].items():
        for period,countkey,totalkey in [('original','original_sources','original_logical'),('preserved','preserved_sources','preserved_logical')]:
            counts=s[countkey];n=s[totalkey]
            source_mix.append(dict(split=split,inventory=period,logical=n,full_task=counts.get('full_task',0),
                corrective=counts.get('Task10F_corrective',0),full_task_fraction=counts.get('full_task',0)/n,
                corrective_fraction=counts.get('Task10F_corrective',0)/n))
    with (run/'initial_distribution_summary.csv').open() as f:dist=list(csv.DictReader(f))
    ranges=[]
    for key in ['peg_xyz_0_m','peg_xyz_1_m','peg_yaw_rad','g4_0_m','g4_1_m','g3p_0_m','g3p_1_m']:
        row=dict(feature=key)
        for pop in ['existing33','candidate300']:
            d=next(r for r in dist if r['population']==pop and r['feature']==key)
            row[pop+'_range']=f"{float(d['minimum']):.6f} … {float(d['maximum']):.6f}"
        ranges.append(row)
    extra=f'''
## 15. 독립 완료 감사와 최종 해석

원격 실제 실행 Root는 `/workspace/task10i_stage2_20261031_v2/run`, UUID `34d10bf652b344a38d54b7672dbfe0c0`다. [독립 원격 감사](../remote_execution/independent_completion_audit_v2.json)는 원본340 Parent, 전체8,514 Canonical 파일, 764개 불변 Stage2 산출물, 별도 프로세스의 Greedy176 전체 trace exact 재현, Coverage와 Monte Carlo200회, Reserve 순서 및 Parent별 branch budget190/190/25를 확인했다. 첫 실패 Run의 선정 Identity/Order/Scale/Split/CSV는 새 Run과 SHA256 exact 일치한다.

독립 SO(3) relative Rotation 구현의 objective와 frozen 구현은 최대 **1.39×10⁻¹⁷** 차이를 보인 Reserve 후반 동률 후보가 있다(rank95/129). 같은 원격 float64 구현·Runtime에서는 전체 순서가 exact 재현된다. macOS의 수치 재추출은 10⁻¹² 이내이고, 늦은 Reserve tie의 bitwise 순서까지 cross-platform 일치를 주장하지 않는다. Stage3는 다시 계산하지 않고 저장된 Queue를 사용한다. 최종 D19에는 이 동률 차이가 없다. [로컬 전수 Readback/재계산](../local_completion_audit.json).

**Selection Pool:** D mean은 R보다 **6.40% 낮고** p90/p95/max도 낮다. **독립 Holdout:** D mean은 R보다 **1.57% 높고**, median은 동일하며 p90/p95/max도 더 높다. Conditional Random200에서 D의 Holdout mean percentile은 **83**, R은 **21**이다(낮을수록 좋음). 따라서 **독립 Holdout Coverage 개선은 INCONCLUSIVE이며 이 표본에서는 관측되지 않았다**. 이 결과를 보고 R/D나 weight·seed를 바꾸지 않았다. 실제 정책 성능·Expert 성공은 계속 **NOT_TESTED**다.

D tail19의 작은 margin 집중은 이번 기술통계에서 관측되지 않았다: margin<4px는 D tail0/19, R tail1/19이며 D tail 최소 margin5.834px다. 반면 사전 정의 위치·크기 extreme 비율은 D tail8/19(42.1%)로 R tail5/19(26.3%)보다 높다. 전체 D39/96(40.6%), R36/96(37.5%), Candidate112/300(37.3%)이다. 그러므로 경계 집중과 위치·크기 분포 변화는 구분해야 하며, 분포 보존이나 성능상의 무해성을 확정하지 않는다.

기존33과 신규300은 범위가 완전히 분리된 두 분포가 아니다. 후보가 일부 방향·크기·상대관계 범위를 확장하며 내부의 빈 영역도 보충한다. 아래는 주변 범위의 예이고, 이 표만으로 다변량 Coverage를 판정하지 않는다.

{table(ranges,['feature','existing33_range','candidate300_range'])}

기존 Full-task/Corrective 구성은 다음과 같다. Corrective source의 `nominal`도 독립 초기 상태 수로 더하지 않았다.

{table(source_mix,['split','inventory','logical','full_task','corrective','full_task_fraction','corrective_fraction'])}

수집 계획은 새 Train63마다 기존 실패63 slot의 실제 branch template를 대응하여 Full-task63+Corrective127=190개를 계획한다. Validation은 기존 실패9 slot의 Full-task6+Corrective19=25개를 계획한다. 모든 신규 Parent의 Full-task 성공 확인은 별도로 선행되어야 한다. Validation template에 Full-task logical entry가 없는 Parent도 성공 확인을 생략한다는 의미는 아니다. 자격 확인용 추가 trajectory를 학습 inventory에 자동 포함하지 않는다. 공유 tail이 다른 template를 받으면 필요한 branch union과 각 dataset inclusion을 분리한다.

로컬 최종 Markdown/Static HTML은 원격 원문 전체와 이 감사 결과를 포함한다. 원격의 최소 HTML fallback은 표/그림을 펼치지 않으므로 로컬에서 정식 Markdown renderer로 HTML을 생성했다. 원본은 변경하지 않았다. Monte Carlo 그림은 겹친 subplot 제목만 별도 전달 그림으로 수정했다. 모든 원본 결과 SHA는 그대로다. Gallery의 Raw RGB/OBB와 원본 NPZ/Evidence/Snapshot 링크는 현재 저장소의 Canonical 자료로 연결했다.

프로젝트 코드는 Git dirty 상태였고 새 Commit을 만들지 않았다. 원격 isolated source에는 `.git`가 없어 Logger의 commit이 null이므로 clean checkout 증거로 해석하지 않는다. [로컬 Git/Source provenance](../local_source_provenance.json)와 [실제 frozen source](../run/execution_source_identity.json)를 함께 사용한다. 분석은 Linux CPU, 전달 QA는 macOS 저장자료 분석으로 수행했다. Shared simulator/model/logger 코드는 변경하지 않았고 기존 ExperimentLogger를 재사용했다. 외부 Publishing·새 인스턴스·Stage3 실행은 없다.

원격 실행 전 계약 테스트 **11개 PASS**, 전달 후 계약 테스트 **12개 PASS**다. 추가 테스트는 늦은 Resume가 새 Logger UUID로 기존 결과에 연결할 때 원래 sidecar/SHA를 보존하는 경우를 검증한다. 이 운영 복구 보강과 별도 감사·전달 helper는 현재 코드에 있고 원격 연구 실행 당시 Source는 별도로 보존했다. [추가 계약 테스트 로그](../postrun_contract_tests.log). PNG 그림을 실제 열어 확인했고 모든 HTML 표/이미지 및 로컬 링크를 정적으로 검사했다. 이 세션에 사용할 브라우저가 없어 브라우저 스크린샷 검사는 실행하지 않았다.
'''
    original=(run/'10i_stage2_diversity_aware_dataset_selection.md').read_text()
    # All original relative artifact links resolve in the unchanged primary run.
    original=re.sub(r'\]\(([^)]+)\)',lambda m:'](../run/'+m.group(1)+')',original)
    original=original.replace('../run/gallery.html','gallery.html').replace('../run/figures/random_mc.png','random_mc_labeled.png')
    text=original+extra
    md=delivery/'10i_stage2_diversity_aware_dataset_selection.md';md.write_text(text)
    rendered=markdown.markdown(text,extensions=['tables','fenced_code']).replace('<img ','<img class="plot" ')
    htm=delivery/'10i_stage2_diversity_aware_dataset_selection.html'
    htm.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task10-I Stage2 — Final audited report</title><style>'+CSS+'</style>'+rendered+'</html>')
    assert rendered.count('<table>')>=7 and rendered.count('<img')==12
    docs=REPO/'docs/visual-graph-integration';docs.mkdir(exist_ok=True)
    def doclink(m):
        target=(delivery/m.group(1)).resolve().relative_to(REPO)
        return '](../../'+str(target)+')'
    doctext=re.sub(r'\]\(([^)]+)\)',doclink,text)
    docmd=docs/md.name;docmd.write_text(doctext)
    docbody=markdown.markdown(doctext,extensions=['tables','fenced_code']).replace('<img ','<img class="plot" ')
    (docs/htm.name).write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task10-I Stage2 — Final audited report</title><style>'+CSS+'</style>'+docbody+'</html>')
    # Every Markdown artifact and every gallery link/image must resolve locally.
    refs=[]
    for path in [md,docmd]:
        for ref in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            assert (path.parent/ref).is_file(),(path,ref);refs.append(str((path.parent/ref).resolve()))
    for ref in re.findall(r'(?:src|href)="([^"]+)"',gallery):
        assert (delivery/ref).is_file(),ref;refs.append(str((delivery/ref).resolve()))
    payload=dict(status='PASS',original_artifact_index_sha256=sha(run/'artifact_index.json'),
        remote_audit_sha256=sha(base/'remote_execution/independent_completion_audit_v2.json'),
        local_audit_sha256=sha(base/'local_completion_audit.json'),resolved_links=len(refs),
        files={str(p.relative_to(REPO)):dict(bytes=p.stat().st_size,sha256=sha(p))
               for p in [md,htm,delivery/'gallery.html',delivery/'random_mc_labeled.png',docmd,docs/htm.name]},
        delivery_source_sha256=sha(Path(__file__)),new_collection_union=90,stage3_started=False)
    (base/'delivery_verification.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',resolved_links=len(refs),docs=str(docmd))))


if __name__=='__main__':main()

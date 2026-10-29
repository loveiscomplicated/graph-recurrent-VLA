"""Read-only experiment watcher and final local retrieval; no Codex callback.

This watches only the explicitly requested experiment. It does not send messages,
publish artifacts, change a training process, or select models from rollout scores.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from .common import read,write,sha,now


def ssh_args(args):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15',
        '-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/plan_a_bc_watch_ssh',
        '-p',str(args.port),args.host]


def snapshot(args):
    code=r'''from pathlib import Path
import json
r=Path(ROOT)
def read(p):return json.loads(p.read_text())
es=[read(p) for p in (r/'cache').glob('ep*/receipt.json')]
result=dict(pipeline=read(r/'pipeline_state.json'),cache_episodes=len(es),cache_graphs=sum(e['graphs'] for e in es),models={},files={})
for variant in ['E0','E1']:
 p=r/(variant+'_run.json')
 if p.exists():
  run=Path(read(p)['directory']);status=run/'status.json';progress=run/'progress.json'
  result['models'][variant]=dict(run_id=run.name,status=read(status) if status.exists() else read(progress) if progress.exists() else {'status':'INITIALIZING'})
 else:result['models'][variant]=dict(status={'status':'NOT_EXECUTED','step':0})
for name in ['config.json','dataset_audit.json','cache_manifest.json','cache_pilot_verification.json','vision_contract.json','runtime_verification.json','frozen09d_source_verification.json','09d_graph_regression_target.json','pipeline_run.json','pipeline_error.json','training_smoke_verification.json','model_selection_frozen.json','final_offline_metrics.json','cpu_preparation_run.json']:
 p=r/name
 if p.exists():result['files'][name]=read(p)
print(json.dumps(result))
'''.replace('ROOT',repr(args.remote_root))
    p=subprocess.run(ssh_args(args)+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=55)
    if p.returncode:raise RuntimeError('SSH status read failed: '+str(p.returncode))
    # Connection banners are not part of the JSON payload.
    lines=p.stdout.splitlines();line=next((x for x in reversed(lines) if x.startswith('{')),None)
    if line is None:raise ValueError('remote status JSON absent')
    return json.loads(line)


def merge_csvs(root,name):
    rows=[]
    for variant in ['E0','E1']:
        marker=root/(variant+'_run.json')
        # Original marker paths are remote; resolve by immutable run UUID.
        if not marker.exists():continue
        run_id=read(marker)['run_id'];path=root/'training_runs'/run_id/name
        if not path.exists():continue
        with path.open() as f:rows.extend(dict(r,variant=variant) for r in csv.DictReader(f))
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def local_analysis(root):
    cfg=read(root/'config.json');state=read(root/'pipeline_state.json')
    for name in ['train_metrics.csv','validation_metrics.csv','rollout_metrics.csv']:merge_csvs(root,name)
    timeline=read(root/'learning_timeline.json') if (root/'learning_timeline.json').exists() else {'records':{},'paired_dev2':[]}
    final=read(root/'final_offline_metrics.json') if (root/'final_offline_metrics.json').exists() else {}
    cm=read(root/'checkpoint_manifest.json') if (root/'checkpoint_manifest.json').exists() else []
    verified=[]
    for cp in cm:
        rel=Path(cp['path']).relative_to(Path(cfg.get('remote_root','/workspace/plan_a_bc_3788798a719f4ec4a66c224ca42f5d63')))
        path=root/rel
        if not path.exists() or sha(path)!=cp['sha256']:raise ValueError('retrieved checkpoint SHA mismatch')
        verified.append(dict(path=str(rel),sha256=cp['sha256'],status='PASS'))
    write(root/'local_checkpoint_readback.json',dict(status='PASS',files=verified))
    paragraphs=['# Plan A 실행 결과 보충 분석','',f'실행 상태: **{state["status"]}**. 아래 수치는 실제 원자료에서 계산했다.','']
    for variant,r in timeline['records'].items():
        paragraphs.append(f'## {variant}')
        paragraphs.append(f'실제 optimizer updates: {r["updates"]}. 실제 성능 평가 Episode: {r["rollout_episodes"]}.')
        val=r.get('validation',[]);dev=sorted([x for x in timeline.get('paired_dev2',[]) if x['variant']==variant],key=lambda x:x['step'])
        if len(val)>=2:
            first,last=val[0],val[-1]
            paragraphs.append(f'Validation Method B loss는 step {first["step"]}의 {float(first["optimization_loss"]):.6g}에서 step {last["step"]}의 {float(last["optimization_loss"]):.6g}로 변했다.')
        if len(dev)>=2:
            first,last=dev[0],dev[-1]
            changes={k:last[k]-first[k] for k in ['reach','grasp','lift','insertion']}
            paragraphs.append('고정 Dev2의 첫→마지막 관측 성공 수는 '+', '.join(f'{k} {int(first[k]*2)}/2→{int(last[k]*2)}/2' for k in changes)+'이다.')
            if val and float(val[-1]['optimization_loss'])<float(val[0]['optimization_loss']):
                positive=any(x>0 for x in changes.values());negative=any(x<0 for x in changes.values())
                pattern='일부 단계 개선과 악화가 함께 관찰됐다' if positive and negative else '일부 단계의 개선이 관찰됐다' if positive else '일부 단계가 악화됐다' if negative else '해당 두 Episode에서는 단계 성공 수가 정체됐다'
                paragraphs.append('Offline loss 감소와 비교하면 '+pattern+'. 작은 반복 평가 분모의 기술적 결과이며 일반적인 개선·인과관계로 확정하지 않는다.')
        else:paragraphs.append('주기적 rollout의 충분한 시점이 없어 Offline–Closed-loop 관계는 NOT_TESTED다.')
        for kind,subsets in final.get(variant,{}).items():
            seed=subsets['train_seed30003']['native_action7_mse'];baseline=.0034380406404083026 if kind=='validation_best' else .003433573
            paragraphs.append(f'{kind}: seed30003 native Action7 MSE={seed:.6g}, 이전 Full B 대비 비율={seed/baseline:.3f}. Train96 평균={subsets["train"]["native_action7_mse"]:.6g}, Validation12={subsets["validation"]["native_action7_mse"]:.6g}, Test12={subsets["test"]["native_action7_mse"]:.6g}.')
        roll=r.get('rollouts',[])
        for group,kind in [('test12','last'),('test12','validation_best')]:
            rr=[x for x in roll if x['group']==group and x['final_kind']==kind and x['denominator_eligible']=='True']
            if rr:paragraphs.append(kind+' Test12: '+', '.join(f'{k} {sum(x[k]=="True" for x in rr)}/{len(rr)}' for k in ['reach','grasp','lift','alignment','insertion'])+'.')
        failed=[x for x in roll if x['denominator_eligible']=='True' and x['insertion']=='False']
        counts={k:sum(x['first_failed_stage']==k for x in failed) for k in ['reach','grasp','lift','alignment','insertion']}
        paragraphs.append('첫 미충족 단계: '+str(counts)+'. 누락·gate·action·grasp·trajectory trace가 있는 범위에서만 실패를 기술할 수 있으며, Geometry 오염이나 GRU를 단독 원인으로 확정하지 않는다.')
    paragraphs+=['','## 해석 범위','',
        '기존 Full B checkpoint의 원래 원격 경로는 이번 조회 시 없었고, 이전 검증된 Task08-E 원자료와 checkpoint hash 참조를 사용했다. 다른 Graph schema에 기존 가중치를 직접 적용하지 않았다. 동일 action/split/update budget 비교와 Geometry·feature 차원·parameter 수·runtime 차이를 구분한다.',
        '단일 training seed로 얻은 Train seed fit과 반복 Dev seed 결과는 trajectory memorization의 가능성을 배제하지 않는다. 선택을 고정한 뒤의 Test12 결과는 추가 증거이며, 12개의 독립 seed를 넘는 모집단 주장을 하지 않는다.',
        'CPU perception+live rollout 준비 probe는 SIGSEGV로 실패했다. CUDA perception/policy 경로의 별도 3-step probe는 성공했으며 성능 분모에서 제외했다. HF 게시 계약 테스트 1개는 FAIL로 보존하고 게시 기능은 호출하지 않았다.',
        '이 보충 분석은 학습 조건·모델 선택·Geometry filter를 변경하지 않는다. 모든 checkpoint/video와 실제 rollout denominator는 원래 보고서와 raw CSV/JSON을 따른다.',
        '', '[원래 연구 보고서](research_report.md) · [검증](verification.json) · [Artifact Index](artifact_index.json) · [Checkpoint readback](local_checkpoint_readback.json)']
    text='\n\n'.join(paragraphs)+'\n';(root/'completion_analysis.md').write_text(text)
    (root/'completion_analysis.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui}pre{white-space:pre-wrap}</style><p><a href="research_report.html">원래 정적 보고서</a></p><pre>'+html.escape(text)+'</pre>')


def retrieve(args):
    target=args.local_root/'remote_snapshot';target.mkdir(exist_ok=True)
    shell=shlex.join(ssh_args(args)[:-1])
    command=['rsync','-a','--partial','--safe-links','--timeout=120','-e',shell,
        '--exclude=source/','--exclude=dataset/','--exclude=inputs/','--exclude=transfer_parts/',
        '--exclude=original_dataset_published*','--exclude=*.tar.gz','--exclude=*.sock',
        args.host+':'+args.remote_root+'/',str(target)+'/']
    for attempt in range(5):
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if p.returncode==0:
            extra=args.local_root/'infrastructure_verification_addendum.json'
            if extra.exists():
                import shutil
                shutil.copy2(extra,target/extra.name)
            local_analysis(target)
            write(args.local_root/'retrieval_status.json',dict(status='PASS',completed_at=now(),directory=str(target)))
            return
        write(args.local_root/'retrieval_status.json',dict(status='RETRY',attempt=attempt+1,exit_code=p.returncode))
        time.sleep(20)
    raise RuntimeError('final retrieval failed; remote artifacts remain preserved')


def incremental_backup(args):
    """Copy finalized graphs/checkpoints/videos while computation continues."""
    target=args.local_root/'remote_snapshot';target.mkdir(exist_ok=True)
    shell=shlex.join(ssh_args(args)[:-1])
    command=['rsync','-a','--partial','--safe-links','--timeout=90','-e',shell,
        '--exclude=source/','--exclude=dataset/','--exclude=inputs/','--exclude=transfer_parts/',
        '--exclude=original_dataset_published*','--exclude=*.tar.gz','--exclude=*.sock',
        '--exclude=*.partial','--exclude=*.hf_partial','--exclude=*.log',
        args.host+':'+args.remote_root+'/',str(target)+'/']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=240)
    write(args.local_root/'incremental_backup_status.json',dict(time=now(),status='PASS' if result.returncode==0 else 'RETRY',exit_code=result.returncode))
    write(args.local_root/'path_remap.json',{args.remote_root:str(target)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--remote-root',required=True);p.add_argument('--local-root',required=True,type=Path)
    p.add_argument('--interval',type=int,default=120);a=p.parse_args();write(a.local_root/'watcher_pid.json',dict(pid=os.getpid(),started_at=now()))
    while True:
        try:
            snap=snapshot(a);write(a.local_root/'live_status.json',dict(updated_at=now(),**snap))
            folder=a.local_root/'remote_status';folder.mkdir(exist_ok=True)
            for name,value in snap['files'].items():write(folder/name,value)
            text=f'# Plan A 실행 상태\n\n{now()}\n\n상태: **{snap["pipeline"]["status"]}**, 단계: `{snap["pipeline"]["phase"]}`.\n\n캐시 {snap["cache_episodes"]}/108 Episode, {snap["cache_graphs"]}/13,452 Graph.\n\n'
            text+='\n'.join(f'- {v}: {r["status"]}' for v,r in snap['models'].items())
            text+='\n\nFull cache, E0/E1 학습, 정기 rollout, 최종 Test가 모두 끝나기 전에는 전체 완료가 아니다.\n'
            (a.local_root/'LIVE_STATUS.md').write_text(text)
            print(now(),snap['pipeline']['status'],snap['pipeline']['phase'],snap['cache_episodes'],snap['cache_graphs'],flush=True)
            if snap['pipeline']['status'] in ['COMPLETE','FAILED']:
                retrieve(a);return
            incremental_backup(a)
        except BaseException as exc:
            write(a.local_root/'watcher_last_error.json',dict(error_type=type(exc).__name__,time=now()))
            print(now(),'WATCH_ERROR',type(exc).__name__,flush=True)
        time.sleep(a.interval)


if __name__=='__main__':main()

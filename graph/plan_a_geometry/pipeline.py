"""Resumable sequential09-D execution; source inputs and old runs immutable."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from graph.integration_baseline.io import write_json,sha256
from utils.experiment_logger import ExperimentLogger

BASE=Path('/workspace/task09d_plan_a/v2');SCRATCH=Path('/dev/shm/task09d_plan_a/v2');CODE=Path(__file__).resolve().parents[2]

def main():
    protocol=json.loads((CODE/'graph/plan_a_geometry/protocol.json').read_text())
    statepath=BASE/'pipeline_state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(started_at=datetime.now(timezone.utc).isoformat(),completed={},failures=[])
    def save():write_json(statepath,state)
    def stage(name,module,args,output,uuid=True):
        if name in state['completed']:return Path(state['completed'][name])
        state['active']=name;state['status']='running';save()
        with (BASE/(name+'.log')).open('w') as stream:
            command=[sys.executable,'-u','-m',module,*map(str,args),'--output',str(output)]
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=CODE)
        paths=([p for p in output.iterdir() if p.is_dir() and (p/'metadata.json').is_file() and json.loads((p/'metadata.json').read_text())['status']=='finished'] if uuid else [output])
        if len(paths)!=1:raise RuntimeError('ambiguous completed stage '+name)
        state['completed'][name]=str(paths[0]);state['active']=None;save();return paths[0]
    try:
        runtime=stage('runtime','graph.plan_a_geometry.runtime',[],BASE/'runtime',False)
        dev=stage('development_geometry','graph.plan_a_geometry.evaluate',['--capture',protocol['development_capture'],'--workers',2],SCRATCH/'development_geometry')
        training=stage('training','graph.plan_a_geometry.train',['--geometry',dev],SCRATCH/'training')
        probes=stage('probes','graph.plan_a_geometry.probes',['--training',training],SCRATCH/'probes')
        frozen=BASE/'frozen_protocol.json'
        if not frozen.exists():
            files={str(p):sha256(p) for p in CODE.rglob('*.py') if '__pycache__' not in str(p)}
            files[str(CODE/'graph/plan_a_geometry/protocol.json')]=sha256(CODE/'graph/plan_a_geometry/protocol.json')
            for folder,pattern in [(training,'selected.pt'),(probes,'decoder.npz')]:
                files.update({str(p):sha256(p) for p in folder.rglob(pattern)})
            files[str(training/'model_manifest.json')]=sha256(training/'model_manifest.json')
            files[str(probes/'probe_manifest.json')]=sha256(probes/'probe_manifest.json')
            write_json(frozen,dict(frozen_at=datetime.now(timezone.utc).isoformat(),final_seeds=protocol['final_seeds'],protocol=protocol,frozen_files=files,development_only=True))
        capture=stage('final_capture','graph.plan_a_geometry.collect_final',['--freeze',frozen],SCRATCH/'final_capture')
        geometry=stage('final_geometry','graph.plan_a_geometry.evaluate',['--capture',capture,'--workers',2],SCRATCH/'final_geometry')
        applied=stage('final_apply','graph.plan_a_geometry.final_apply',['--training',training,'--probes',probes,'--geometry',geometry],SCRATCH/'final_apply')
        state['status']='finished';state['finished_at']=datetime.now(timezone.utc).isoformat();save()
    except BaseException as e:
        state['status']='failed';state['failures'].append(dict(stage=state.get('active'),error=type(e).__name__,at=datetime.now(timezone.utc).isoformat()));save();raise
if __name__=='__main__':main()

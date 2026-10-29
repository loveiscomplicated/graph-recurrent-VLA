"""After frozen application: verify, aggregate and render retained evidence."""
import json,subprocess,sys,time
from pathlib import Path
from graph.integration_baseline.io import write_json
BASE=Path('/workspace/task09d_plan_a/v2');SCRATCH=Path('/dev/shm/task09d_plan_a/v2')

def main():
    while True:
        state=json.loads((BASE/'pipeline_state.json').read_text())
        if state['status']=='failed':raise RuntimeError('primary pipeline failed; postprocess not executed')
        if state['status']=='finished':break
        time.sleep(10)
    completed={};statepath=BASE/'postprocess_state.json'
    tasks=[('verification','graph.plan_a_geometry.verify',['--state',BASE/'pipeline_state.json'],BASE/'verification'),
        ('summary','graph.plan_a_geometry.summarize',['--state',BASE/'pipeline_state.json'],BASE/'summary'),
        ('plots','graph.plan_a_geometry.plots',['--state',BASE/'pipeline_state.json','--summary',BASE/'summary'],BASE/'figures'),
        ('media','graph.plan_a_geometry.media',['--geometry',state['completed']['final_geometry']],SCRATCH/'final_media')]
    for name,module,args,output in tasks:
        write_json(statepath,dict(status='running',active=name,completed=completed))
        with (BASE/(name+'.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m',module,*map(str,args),'--output',str(output)],stdout=f,stderr=subprocess.STDOUT,check=True)
        completed[name]=str(output)
    write_json(statepath,dict(status='finished',active=None,completed=completed))
if __name__=='__main__':main()

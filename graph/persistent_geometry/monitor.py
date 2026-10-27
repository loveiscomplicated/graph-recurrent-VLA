"""Read-only resource sampling for the owned09-C process tree."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import csv
from graph.integration_baseline.io import write_json
from utils.experiment_logger import ExperimentLogger
from .collect import inventory_sources


def processes():
    rows={}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
            rows[int(path.parent.name)]=dict(parent=int(fields['PPid']),
                rss=int(fields.get('VmRSS','0 kB').split()[0])*1024,
                peak=int(fields.get('VmHWM','0 kB').split()[0])*1024)
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=dict(task='09-C',stage='resource_observation',interval_seconds=10,
        scope='owned benchmark process descendants; global GPU usage; sampling starts after initial pilots',
        parent_pid=args.parent_pid,process_peak='Linux VmHWM for live owned processes; includes their earlier lifetime',
        gpu_peak='sampled global usage only; not an exact allocation peak')
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_resource_monitor',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_RESOURCE_DIRECTORY='+str(root),flush=True);inventory_sources(root)
        fields=('timestamp','stage','owned_processes','owned_rss_sum_bytes','largest_owned_process_peak_bytes',
                'global_gpu_used_MiB','global_gpu_utilization_percent','container_memory_current_bytes')
        peaks={};samples=0
        with (root/'resource_samples.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
            while True:
                state=json.loads(args.state.read_text());all_processes=processes();owned={args.parent_pid}
                while True:
                    expanded=owned|{pid for pid,value in all_processes.items() if value['parent'] in owned}
                    if expanded==owned:break
                    owned=expanded
                active=[all_processes[pid] for pid in owned if pid in all_processes]
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].split(',')
                stage=state.get('active') or state['status']
                row=dict(timestamp=datetime.now(timezone.utc).isoformat(),stage=stage,owned_processes=len(active),
                    owned_rss_sum_bytes=sum(value['rss'] for value in active),
                    largest_owned_process_peak_bytes=max((value['peak'] for value in active),default=0),
                    global_gpu_used_MiB=int(gpu[0]),global_gpu_utilization_percent=int(gpu[1]),
                    container_memory_current_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))
                writer.writerow(row);out.flush();samples+=1
                entry=peaks.setdefault(stage,dict(samples=0))
                entry['samples']+=1
                for key in fields[2:]:entry[key+'_maximum']=max(entry.get(key+'_maximum',0),row[key])
                if samples%6==0:logger.log({key:value for key,value in row.items() if isinstance(value,(int,float))})
                if state['status']!='running':break
                time.sleep(10)
        write_json(root/'resource_summary.json',dict(config=config,peaks=peaks,total_samples=samples))
        logger.link_output(root/'resource_summary.json',kind='metrics');logger.link_output(root/'resource_samples.csv',kind='reports')


if __name__=='__main__':main()

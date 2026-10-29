"""Userspace, stage-labelled device/resource sampling. Never records argv/env."""
import contextlib
import csv
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil


class Resources:
    def __init__(self,root,stage):
        self.root=Path(root);self.stage=stage;self.stop=threading.Event();self.thread=None

    def __enter__(self):
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self

    def sample(self):
        process=psutil.Process();process.cpu_percent()
        path=self.root/'resource_samples.csv'
        with path.open('a',newline='') as f:
            keys=['timestamp','stage','gpu_util_pct','vram_mib','cpu_system_pct','cpu_process_pct','rss_mib','available_ram_mib']
            w=csv.DictWriter(f,fieldnames=keys)
            if f.tell()==0:w.writeheader()
            while not self.stop.is_set():
                r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                vals=r.stdout.strip().split(',') if r.returncode==0 else ['nan','nan']
                w.writerow(dict(timestamp=time.time(),stage=self.stage,gpu_util_pct=vals[0],vram_mib=vals[1],
                    cpu_system_pct=psutil.cpu_percent(),cpu_process_pct=process.cpu_percent(),
                    rss_mib=process.memory_info().rss/2**20,available_ram_mib=psutil.virtual_memory().available/2**20));f.flush()
                self.stop.wait(1)

    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=6)


def hardware():
    import torch
    quota=Path('/sys/fs/cgroup/cpu.max')
    return dict(torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        cuda_memory_bytes=torch.cuda.mem_get_info() if torch.cuda.is_available() else None,
        cpu_logical=psutil.cpu_count(),cpu_affinity=len(os.sched_getaffinity(0)),
        cpu_quota=quota.read_text().strip() if quota.exists() else None,
        ram=psutil.virtual_memory()._asdict(),disk=psutil.disk_usage('/workspace')._asdict())

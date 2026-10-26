"""Initialize CUDA before loading the Task08-F CPU PhysX / GPU Vision worker."""
import runpy
import torch

if __name__ == '__main__':
    torch.cuda.init()
    runpy.run_module('graph.integration_baseline.bc08f_branch', run_name='__main__')

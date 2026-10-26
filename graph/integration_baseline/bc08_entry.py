"""Initialize CUDA before importing simulator libraries in a Task08 child."""
import runpy
import sys


def main():
    if len(sys.argv)<2 or sys.argv[1]!='graph.integration_baseline.bc08_pipeline':
        raise SystemExit('only the Task08 worker dispatcher is allowed')
    module=sys.argv.pop(1)
    import torch
    torch.cuda.init()
    runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()

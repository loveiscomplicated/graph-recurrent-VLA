"""Run legacy contracts against their Full96 fixture, then Reach contracts.

The legacy 50-epoch test locates ../config.json relative to its source. Its
Full-task epoch length is not the Task10D Reach sampler's clock. No inherited
source/test is modified or skipped: it runs in the original read-only source.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from graph.plan_a_bc.common import read,write,now


def run(root):
    baseline=Path(read(root/'run_identity.json')['baseline'])
    suites=[('inherited_Full96',baseline/'source',['graph.plan_a_bc.tests','graph.plan_a_geometry.tests']),
            ('Task10D_Reach',root/'source',['graph.geometry_reach10d.tests','graph.geometry_reach10d.test_monitor'])]
    results=[]
    for label,cwd,modules in suites:
        print('CONTRACT DOMAIN',label,'source',cwd,flush=True)
        subprocess.run([sys.executable,'-B','-m','unittest',*modules,'-v'],cwd=cwd,check=True)
        results.append(dict(domain=label,status='PASS',source=str(cwd)))
    write(root/'contract_suite_verification.json',dict(status='PASS',created_at=now(),suites=results,
        inherited_source_modified=False,legacy_11487_test_skipped=False),immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

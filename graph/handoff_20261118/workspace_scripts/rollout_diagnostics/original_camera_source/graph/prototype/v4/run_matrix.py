"""Fixed 8-cell matrix, fail-fast training then evaluation; fresh outputs only."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','evaluate','ablate'],required=True)
    p.add_argument('--models',nargs='+',choices=['flat','surface'],default=['flat','surface'])
    p.add_argument('--histories',nargs='+',choices=['current','action','state','both'],default=['current','action','state','both'])
    p.add_argument('--root',type=Path,default=Path('outputs/graph/v4'))
    args=p.parse_args()
    for name in args.models:
        for mode in (['both'] if args.phase=='ablate' else args.histories):
            directory=args.root/'training'/f'{name}_{mode}'
            if args.phase=='train':
                if mode=='current':continue  # explicitly imported v2 baseline
                command=[sys.executable,'-m','graph.prototype.v4.train','--model',name,'--history',mode,'--epochs','30','--seed','0','--length','4','--output',str(directory)]
                log=args.root/'training'/f'{name}_{mode}.log'
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as f: subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            else:
                cases=[('ablated',['--stages','full','alignment','--ablate-history'])] if args.phase=='ablate' else [('heldout',[]),('fresh',['--stages','full','--new-seeds',*[str(s) for s in range(2000,2030)]])]
                for label,extra in cases:
                    output=args.root/'evaluation'/f'{name}_{mode}_{label}.json'
                    log=output.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-m','graph.prototype.v4.evaluate','--checkpoint',str(directory/'best.pt'),'--output',str(output),'--rollouts',*extra]
                    with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(f'completed {args.phase}: {name}/{mode}',flush=True)


if __name__=='__main__':main()

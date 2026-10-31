"""Read-only Hub recovery of the original GRU and evaluator evidence."""
import concurrent.futures, hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO='Graph-Recurrent-VLA/experiment-backups'
OUT=Path('outputs/task10e')
BASE='recurrent/reevaluation_main_20261023/1db7f7a881cc445098d952f64443ccd6'
MAIN='recurrent/main/4544b6012abc484681acfb58c57fede4'
def run():
    tree=json.loads((OUT/'hf_recurrent_tree.json').read_text())
    rev=HfApi().repo_info(REPO,repo_type='dataset').sha
    selected=[]
    for x in tree:
        p=x['path']
        if not x['size']:continue
        if (p.startswith(BASE+'/files/recurrent/') or p==BASE+'/manifest.json' or
            p.startswith(BASE+'/files/__backup_provenance__/') or
            p.startswith(BASE+'/files/outputs/') and ('gru_none_s' in p or p.endswith(('/plan.json','/initial_states.pt','/reevaluation.identity.json'))) or
            p.startswith(MAIN+'/files/outputs/recurrent/exp/main/runs/gru_none_s') or
            p==MAIN+'/manifest.json'):
            selected.append(p)
    def get(p):
        local=Path(hf_hub_download(REPO,p,repo_type='dataset',revision=rev))
        rel=p.split('/files/',1)[-1] if '/files/' in p else p.split('/')[1]+'_manifest.json'
        dest=OUT/('original' if p.startswith(BASE) else 'training')/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(local.read_bytes())
        return dict(remote=p,local=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(get,selected))
    (OUT/'download_inventory.json').write_text(json.dumps(dict(repo=REPO,revision=rev,files=rows),indent=2))
    print('downloaded',len(rows),'revision',rev,flush=True)
if __name__=='__main__':run()

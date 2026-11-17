"""Register complete captures that were saved before the reporting step failed."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root
    dest=root/'audited'
    dest.mkdir()
    for folder in sorted(p for p in root.glob('shard_*') if p.is_dir()):
        source=folder/'predictions.npz'
        original,=folder.glob('experiments/*/metadata.json')
        metadata=json.loads(original.read_text())
        assert metadata['status']=='failed'
        collector_sha=file_sha256(root/'live_compare.py')
        capture_config=original.with_name('config.yaml').read_text()
        assert collector_sha in capture_config
        episodes=[]
        with np.load(source,allow_pickle=False) as f:
            assert np.isfinite(f['free_object21']).all()
            seeds=np.unique(f['seed'])
            assert seeds.size==2
            for seed in seeds:
                mask=f['seed']==seed
                steps=f['step'][mask]
                np.testing.assert_array_equal(steps,np.arange(len(steps)))
                phase=f['phase'][mask]
                assert (phase=='success').sum()==1 and phase[-1]=='success'
                episodes.append(dict(seed=int(seed),success=True,frames=len(steps),
                                     deep=int((f['depth_mm'][mask]>20).sum())))
        out=dest/folder.name
        out.mkdir()
        prediction=out/'predictions.npz'
        shutil.copy2(source,prediction)
        cfg=dict(script_sha256=file_sha256(__file__),collector_sha256=collector_sha,
                 source_run_id=metadata['run_id'],source_status=metadata['status'],
                 original_npz_sha256=file_sha256(source))
        with ExperimentLogger(team='graph',model='gtfree_id_capture_audit',job_type='evaluate',
                              seed=int(seeds[0]),config=cfg,root=out/'experiments') as log:
            assert file_sha256(prediction)==cfg['original_npz_sha256']
            summary=out/'summary.json'
            write(summary,dict(run_id=log.run_id,config=cfg,sha256=file_sha256(prediction),episodes=episodes))
            log.link_output(summary)
            log.link_output(prediction)
            log.log({'episodes':len(episodes),'deep_frames':sum(e['deep'] for e in episodes)})
        print(folder.name,episodes,flush=True)


if __name__=='__main__':
    main()

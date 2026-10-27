"""Read-only replay of saved09-B decoders, without policy/RNN execution."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from graph.integration_baseline.geometry09b_probe import features
from graph.integration_baseline.io import save_npz,write_json,sha256
from utils.experiment_logger import ExperimentLogger
from .audit import protection_snapshot,verify_protection
from .collect import TASK09B,inventory_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    paths=[TASK09B/'frozen_readouts320.npz',*sorted((TASK09B/'probes').glob('*_checkpoint.npz')),
           *sorted((TASK09B/'probes').glob('*_predictions.npz'))]
    protected=protection_snapshot(paths)
    with ExperimentLogger(team='integration',config=dict(task='09-C',stage='09B_saved_probe_replay',source=str(TASK09B),
            fit=False,policy_or_gru_execution=False),seed=42,model='task09c_historical_probe_readback',
            job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_HISTORICAL_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'protection_before.json',protected)
        with np.load(TASK09B/'frozen_readouts320.npz',allow_pickle=False) as a:data={k:a[k] for k in a.files}
        checks=[];predictions={}
        for path in sorted((TASK09B/'probes').glob('*_checkpoint.npz')):
            stem=path.name.removesuffix('_checkpoint.npz')
            model,probe,decoder,control=stem.rsplit('_',3)
            x=features(data,model,probe).astype(np.float64)
            with np.load(path,allow_pickle=False) as a:
                standardized=(x-a['feature_mean'])/a['feature_std']
                if decoder=='linear':pred=standardized@a['weight']+a['bias']
                else:
                    network=nn.Sequential(nn.Linear(x.shape[1],a['0.weight'].shape[0]),nn.ReLU(),nn.Linear(a['2.weight'].shape[1],3))
                    network.load_state_dict({k:torch.from_numpy(a[k]) for k in ('0.weight','0.bias','2.weight','2.bias')})
                    network.eval().requires_grad_(False)
                    with torch.inference_mode():pred=network(torch.from_numpy(standardized.astype(np.float32))).numpy().astype(float)
                pred=pred*a['target_std']+a['target_mean']
            with np.load(path.with_name(stem+'_predictions.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(pred.astype(np.float32),a['prediction'])
                error=float(np.max(abs(pred-a['prediction'])))
            checks.append(dict(model=model,probe=probe,decoder=decoder,control=control,
                checkpoint_sha256=sha256(path),max_float64_vs_saved_float32_error=error,
                saved_float32_prediction_exact=True))
            predictions[stem]=pred
        save_npz(root/'replayed_predictions.npz',**predictions)
        result=verify_protection(protected)
        if result['status']!='PASS':raise ValueError('historical decoder input changed')
        write_json(root/'verification.json',dict(status='PASS',decoders=checks,number_of_decoders=len(checks),
            source_samples=len(data['sample_id']),policy_or_gru_executed=False,protection=result))
        logger.link_output(root/'verification.json',kind='metrics')


if __name__=='__main__':main()

"""Reuse09-C diagnostic training exactly; four current-only inputs × E0/E1."""
import argparse,json
from pathlib import Path
import torch
from graph.persistent_geometry.train import Dataset,train_one
from graph.persistent_geometry.collect import inventory_sources
from graph.integration_baseline.io import write_json,sha256
from utils.experiment_logger import ExperimentLogger

def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());config=dict(protocol['training'])
    config.update(task='09-D',geometry=str(a.geometry),validation_every=50,policy_training=False,input_schema='task09d_current_surface21_v1',gru_executed=False,AMP=False,TF32=False,optimizer='AdamW',optimizer_betas=[.9,.999],optimizer_eps=1e-8,gradient_clip=1.,dtype='float32',deterministic_algorithms=True,train_sample_interval=5,target_std_floor_m=.001,loss='0.5 position6 normalized MSE + 0.5 current-visible summary30 masked normalized MSE',model=dict(d=64,L=2,node_type_embedding=4,edge_type_embedding=4,diagnostic_decoder_hidden=64,output=36))
    manifest=json.loads((a.geometry/'geometry_manifest.json').read_text())
    assert json.loads((a.geometry/'metadata.json').read_text())['status']=='finished'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09d_equal_budget_encoders',job_type='train',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True);models=[]
        for variant in config['variants']:
            data=Dataset(a.geometry,variant,manifest['frames'])
            for encoder in config['encoders']:
                models.append(train_one(root,data,variant,encoder,config,log))
                write_json(root/'model_manifest.json',dict(config=config,models=models))
            del data
        inventory_sources(root);log.link_output(root/'model_manifest.json',kind='metrics')
if __name__=='__main__':main()

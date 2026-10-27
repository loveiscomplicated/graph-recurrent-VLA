"""Pilot-only paired robot-filter revision audit; source observations unchanged."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from graph.integration_baseline.io import write_json,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from utils.experiment_logger import ExperimentLogger
from .evaluate import load_cameras
from .self_filter import LinkSurface,classify_robot
from .metrics import filter_metrics
from .collect import inventory_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--original-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('graph.persistent_geometry._pilot_v1_filter',args.original_source)
    old=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=old;spec.loader.exec_module(old)
    manifest=json.loads((args.capture/'capture_manifest.json').read_text())
    config=dict(task='09-C',stage='robot_filter_revision_pilot',capture=str(args.capture),
        original_source_sha256=sha256(args.original_source),new_source_sha256=sha256(__import__('graph.persistent_geometry.self_filter',fromlist=['']).__file__),
        change='same per-link convex geometry and depth tolerance; also classify measured points strictly inside an individual known collision volume',
        calibration_tuning=False,GT_used_by_filter=False)
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_robot_filter_revision',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_FILTER_STUDY_DIRECTORY='+str(root),flush=True)
        rows=[]
        for episode in manifest['episodes']:
            folder=args.capture/'collection'/f'seed{episode["seed"]}'
            with np.load(folder/'robot_meshes.npz',allow_pickle=False) as a:
                names=a['links'].tolist();vertices=[a[f'vertices_{i}'] for i in range(len(names))]
            for frame in episode['frames']:
                stem=frame['stem'];cameras,data=load_cameras(folder/(stem+'_input.npz'))
                with np.load(folder/(stem+'_gt.npz'),allow_pickle=False) as a:gt={k:a[k] for k in a.files}
                with np.load(folder/(stem+'_masks.npz'),allow_pickle=False) as a:masks={k:a[k] for k in a.files}
                surfaces=[LinkSurface(n,v,p) for n,v,p in zip(names,vertices,data['robot_link_world'])]
                for camera in cameras:
                    for revision,fn in (('v1_depth',old.classify_robot),('v2_collision_volume',classify_robot)):
                        result=fn(camera,surfaces)
                        rows.extend({**r,'revision':revision,'seed':episode['seed'],'split':episode['split'],
                                     'timestep':frame['timestep']} for r in filter_metrics(camera,masks[camera.camera],result,gt,
                                         finger_world=data['robot_world_geometry'][1:]))
            print('filter study seed='+str(episode['seed']),flush=True)
        csv_write(root/'paired_filter_metrics.csv',rows)
        summary=[]
        metrics=('robot_removal_recall','object_preservation','confirmed_false_removal','uncertain_object_fraction',
                 'robot_contamination_before','robot_contamination_after','object_purity_before','object_purity_after',
                 'contact_near_false_removal')
        for revision in ('v1_depth','v2_collision_volume'):
            for camera in sorted({r['camera'] for r in rows}):
                for oid in (0,1):
                    selected=[r for r in rows if r['revision']==revision and r['camera']==camera and r['object_id']==oid]
                    row=dict(revision=revision,camera=camera,object_id=oid,frames=len(selected))
                    for key in metrics:
                        values=[r[key] for r in selected if r[key] is not None]
                        row[key]=float(np.mean(values)) if values else None
                    summary.append(row)
        write_json(root/'paired_filter_summary.json',summary)
        inventory_sources(root)
        logger.link_output(root/'paired_filter_summary.json',kind='metrics')
        logger.link_output(root/'paired_filter_metrics.csv',kind='reports')


if __name__=='__main__':main()

"""Evaluation-sidecar audit of actual preprocessing loss and failure cases."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.fusion import filtered_masks
from graph.persistent_geometry.metrics import sample_meshes
from graph.persistent_geometry.geometry_audit import aperture_reference
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();root=args.run
    manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture']);meshes={};rows=[];operation=[]
    for p in (root/'geometry').glob('*/source_pixel_audit.json'):operation+=json.loads(p.read_text())
    for frame in manifest['frames']:
        seed,t=frame['seed'],frame['timestep'];source=capture/'collection'/f'seed{seed}';folder=root/frame['folder']
        if seed not in meshes:meshes[seed]=sample_meshes(source/'evaluation_meshes.npz')
        cameras,_=load_cameras(source/f'f{t:04d}_input.npz')
        with np.load(source/f'f{t:04d}_masks.npz') as a:masks={k:a[k] for k in a.files}
        with np.load(source/f'f{t:04d}_gt.npz') as a:gt={k:a[k] for k in a.files}
        with np.load(folder/'robot_filter.npz') as a:filters={k:a[k] for k in a.files}
        rim=aperture_reference(meshes[seed],gt)[-1];rimtree=cKDTree(rim)
        for file in folder.glob('fps_*_object*.npz'):
            label=file.name[len('fps_'):].split('_object')[0];oid=int(file.stem.rsplit('object',1)[1]);camera_name=label.replace('_unfiltered','')
            camera=next(c for c in cameras if c.camera==camera_name);mask=masks[camera_name][oid]
            if not label.endswith('_unfiltered'):
                filt={k.split('/',1)[1]:v for k,v in filters.items() if k.startswith(camera_name+'/')}
                clean,_=filtered_masks(masks[camera_name],filt);mask=clean[oid]
            valid=np.isfinite(camera.depth);correct=(gt[camera_name+'/segmentation']==gt['actor_ids'][oid])&valid
            boundary=correct&~(cv2.erode(correct.astype('uint8'),np.ones((3,3),'uint8'))>0)
            world=camera.world_points().reshape(-1,3);object_from_world=np.linalg.inv(pose_matrix(gt['actor_pose'][oid]))
            with np.load(file) as a:
                original=np.flatnonzero(mask&valid);selected=np.flatnonzero(a['selected_mask']&valid)
                erode_removed=np.setdiff1d(original,selected);sor_removed=np.setdiff1d(a['presample_source'],a['sor_source'])
                sor_local=transform(world[sor_removed],object_from_world);dist=meshes[seed][oid]['tree'].query(sor_local)[0]
                sor_valid_surface=(dist<.004)&correct.ravel()[sor_removed]
                removed_rim=sor_valid_surface&(rimtree.query(world[sor_removed])[0]<.004) if oid==1 else np.zeros(len(sor_removed),bool)
                def outliers(indices):
                    if not len(indices):return None
                    d=meshes[seed][oid]['tree'].query(transform(world[indices],object_from_world))[0]
                    return float((d>.008).mean())
                sor_outlier=outliers(a['sor_source']);final_outlier=outliers(a['source_pixel_id'])
                rows.append(dict(seed=seed,timestep=t,camera=label,object_id=oid,original_valid_pixels=len(original),
                    input_mask_empty=not mask.any(),original_valid_empty=len(original)==0,final_empty=len(a['source_pixel_id'])==0,
                    erosion_correct_removed=int(correct.ravel()[erode_removed].sum()),erosion_correct_boundary_removed=int(boundary.ravel()[erode_removed].sum()),
                    erosion_caused_empty=bool(len(original)>0 and len(selected)==0),
                    sor_removed=len(sor_removed),sor_correct_surface_removed=int(sor_valid_surface.sum()),sor_correct_hole_rim_removed=int(removed_rim.sum()),
                    final_FPS_active=len(a['sor_source'])>1024,pre_FPS_random_active=len(a['backproject_source'])>2048,
                    FPS_outlier_fraction_increase=(final_outlier-sor_outlier) if final_outlier is not None and sor_outlier is not None else None))
    csv_write(root/'preprocessing_failure_audit.csv',rows)
    main=[r for r in operation if r['camera'].endswith('_unfiltered')]
    catalog=dict(scope='fixed42-frame pilot; GT sidecar used only for evaluation',
        main_object_frames=len(main),main_nonempty=sum(r['final_count']>0 for r in main),
        main_native_final_FPS_active=sum(r['n_sor_pts']>1024 for r in main),
        main_native_final_FPS_pass_through_nonempty=sum(0<r['n_sor_pts']<=1024 for r in main),
        main_random_presample_active=sum(r['n_raw_pts']>2048 for r in main),
        main_erosion_fallbacks=sum(r['erosion_fallback'] for r in main),all_camera_object_frames=len(operation),
        all_camera_final_FPS_active=sum(r['n_sor_pts']>1024 for r in operation),
        erosion_caused_empty=sum(r['erosion_caused_empty'] for r in rows),input_mask_empty=sum(r['input_mask_empty'] for r in rows),
        sor_correct_surface_removed=sum(r['sor_correct_surface_removed'] for r in rows),
        sor_correct_hole_rim_removed=sum(r['sor_correct_hole_rim_removed'] for r in rows),
        FPS_increased_outlier_fraction_cases=sum(r['FPS_outlier_fraction_increase'] is not None and r['FPS_outlier_fraction_increase']>0 for r in rows),
        other_failure_evidence=dict(robot_accumulation='source_robot_contamination columns in geometry_metrics.csv',
            camera_identity_conflicts='fusion_audit.json per frame; common synthetic rejection contract',
            ghost_geometry='full temporal visual_cases.json and geometry_metrics.csv; actual large G4/L4 failures retained',
            more_points_without_probe_improvement='probe_summary.json; L0/E0 validation worse than G0/E0 in this100-update pilot',
            completely_missing_SAM2_peg='fixed Seed30002t0: both G0/L0 output zero; no recovery'))
    write_json(root/'failure_catalog.json',catalog);print(json.dumps(catalog,indent=2))
if __name__=='__main__':main()

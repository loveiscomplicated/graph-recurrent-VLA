"""Materialize an opt-in manifest and an unapplied addition-only source diff."""
import difflib,json,shutil
from pathlib import Path
from graph.integration_baseline.io import write_json,sha256
from .adapter import PARAMETERS

def materialize(root):
    root=Path(root);source=Path(__file__).resolve().parents[1]
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=list((source/'legacy_fps').glob('*.py'))
    for p in files:
        target=snapshot/'legacy_fps'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    legacy=source/'visual/scripts/pointcloud.py';target=snapshot/'original_legacy_pointcloud.py';shutil.copy2(legacy,target)
    dependencies=json.loads((root/'dependency_source_hashes.json').read_text())
    write_json(root/'implementation_source_hashes.json',{str(p.relative_to(source)):sha256(p) for p in files})
    patch=''
    for p in files:
        patch+=''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile='b/legacy_fps/'+p.name))
    (root/'implementation.diff').write_text(patch)
    manifest=dict(schema='task09c_legacy_fps_opt_in_v1',default_enabled=False,applied_to_existing_09c=False,
        adapter_import='legacy_fps.adapter.frame_points',system_import='legacy_fps.pipeline.LegacySystem',
        config_key='representation.legacy_fps.enabled',parameters=PARAMETERS,
        original_legacy_source_sha256=sha256(legacy),required_dependency_source_hashes=dependencies,
        dependency_revision='copied benchmark_code; source hash pin required, not a mutable branch name',
        variant_mapping={f'L{i}':f'G{i}' for i in range(5)},
        variants=dict(L0='no robot filter/map/wrist; strict no-edge baseline',L1='same common robot filter',L2='same common filter/map/main',L3='same common filter/same-time main+wrist',L4='same common filter/map/main+wrist'),
        graph_edge_conditions=dict(L0='zero edges including robot/object interactions',L0_surface='same selected FPS+F5 points, common3D surface/robot/peg-box edges',L0_rawede='same selected FPS+F5 current main points; Raw EDE regenerated from actual pixel/components; original robot/peg-box helpers',L1_L4='common09-C3D surface/k4/M16 schema; never called Raw EDE'),
        point_schema=dict(identity=['episode_id','timestep','camera_id','object_id','source_pixel_id'],
            point_xyz_camera='original float32 CV backprojection at measured source pixel; metres',
            point_xyz_world='original float32 camera-to-world output; metres',
            source_pixel_id='v*width+u, original raster; exact byte row joins; never guessed UV',
            normal_if_valid='optional F5 augmentation measured at original source pixel',
            object_assignment_confidence='not present in stored binary masks; availability=false; no fabricated SAM2 probability',
            measurement_validity='positive finite calibrated sensor depth and predicted object mask',
            historical='world positions are estimates, not new current RGB-D measurements'),
        map_input_contract='unchanged graph.persistent_geometry.contracts.SurfaceObservation: world XYZ/normals/F5/validity/pixels/camera origin; additional camera XYZ and stage provenance remain in Points artifacts',
        map_budget='unchanged merge_distance3mm and common registration; native surfel map independent of GNN cap128/object; no forced1024 additions',
        map_lifecycle='unchanged PersistentMap merge/update/add/causal clock/pose confidence/contradiction removal/identity switch rejection',
        feature_schema=dict(graph_node_dimension=21,node_types='robot0,1,2; peg3; box4',
            fields='XYZ3,F5,availability3,validity3,status3,confidence,age,uncertainty,pose confidence',
            native_legacy='XYZ only; no claim original FPS produced F5',
            augmented='explicit FPS+F5 from same original RGB-D evidence, no resampling based on F5',
            history='depth jump and variation unavailable/invalid for historical nodes; pose-dependent normal validity follows common09-C gates'),
        seed_policy=dict(reference='original default_rng(0), shared sequential episode/frame/peg-box ordering; verified independently',
            experiment='default_rng(int.from_bytes(sha256(canonicalJSON([42,parent_episode,timestep,camera,object])).digest()[:8],little)); strip :branch_ suffix only for RNG group; full identity retained',
            claim='LEGACY_EXACT_GIVEN_INPUT_AND_RNG; independent frame seeds do not reproduce historical sequential selected pixels'),
        data_compatibility='reads finalized09-C sensor/mask/GT-sidecar captures; identical input hashes required; GT not accepted by practical process()',
        checkpoint_compatibility='existingBC/GNN14-feature checkpoint not compatible with21-feature new distribution; no policy checkpoint loaded; retrain diagnostic E0/E1/E2 on equal splits/budgets',
        readout_input_adapter=dict(import_path='legacy_fps.readout_adapter.CanonicalDiagnosticEncoder',
            reason='pinned structured_readout uses first3 rows for semantic robot slots; all-row permutation violates that positional input contract',
            behavior='canonicalize robot roles by node_type0/1/2; identity on existing serialized graphs; same initial parameter tensors and exact forward outputs verified',
            training_factory='legacy_fps.training_adapter: original loop copied with CanonicalRegressor constructor only; no original09-C module edited'),
        resources=dict(geometry='single-thread CPU nice19; measured timings in costs.json; reserve>=3GiB disk',
            learning='only no-other-GPU-process gate; process allocator capped6percent of24GiB; original pipeline takes precedence',
            external_install=False,publishing=False),
        regression_tests=['legacy_fps.tests','graph.persistent_geometry.tests.test_contracts','actual-input XYZ/RNG exactness','pixel/F5/node row joins','strict single-raster Raw EDE','map causality and source history','independent episode/reset splits','permutation audit on trained models','protection_before vs post stable hashes','all artifact readback/decode/link checks'],
        known_gaps=['nativepoint smoothing/preprocessing effects not individually isolated by posthoc matched-N','t0 paired branch runner only; late-time branch fork not implemented by this extension','no new final test evaluation; reserved62091/62092 untouched','missing SAM2 peg masks cannot be recovered by FPS'])
    write_json(root/'integration_manifest.json',manifest)
    audit=root/'protection_audit.json'
    if audit.exists():
        original=json.loads(audit.read_text())
        manifest['observed_later_original_source_changes']=original.get('changed',[])
        manifest['future_integration_status']='REVALIDATE_AGAINST_STABILIZED_FINAL_09C_SOURCE' if original.get('changed') else 'OPT_IN_PINNED_SOURCE_ONLY'
        manifest['current_running_09c_patched']=False
        write_json(root/'integration_manifest.json',manifest)
    return manifest

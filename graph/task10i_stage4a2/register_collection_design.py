"""Freeze new benchmark collection rules without opening registered seed RGB."""
import argparse
from pathlib import Path
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    protocol=Path('/workspace/task10i_stage4a2_benchmark_protocol_20261101_v1/run/new_final_benchmark_protocol.json')
    stress=Path('/workspace/task10i_stage4a2_stress_engineering_20261101_v1/run/scope.json')
    collector=Path('/workspace/task10i_stage4a2_capture_engineering_20261101_v2/run/scope.json')
    criteria=Path('/workspace/task10i_stage4a2_criteria_20261101_v1/run/adoption_criteria_frozen.json')
    assert read(stress)['status']=='STRESS_CAPTURE_ENGINEERING_COMPLETE'
    assert read(collector)['status']=='BENCHMARK_COLLECTOR_ENGINEERING_COMPLETE'
    assert read(collector)['frames']==104 and read(collector)['exact_synchronous_GT']
    plan=dict(status='FROZEN',version='task10i-stage4a2-new-final-collection-v1',
        normal_parents=32,stress_parents=12,horizon=120,occlusion_steps=list(range(8,20)),
        storage_reserve_gib=15,stress_engineering_pass=True,
        stress_engineering_scope=str(stress),stress_engineering_scope_sha256=sha(stress),
        collector_engineering_scope=str(collector),collector_engineering_scope_sha256=sha(collector),
        new_final_benchmark_protocol_sha256=sha(protocol),adoption_criteria_sha256=sha(criteria),
        seed_selection='original registered queues in order; first32/12 eligible unique physical states; retain every attempt and reject reason',
        initial_eligibility='unchanged Stage1 Main-visible requested-pass, physical-valid and exact reset replay; no expert-success or perception-quality selection',
        stress_assignment='eligible stress parent index modulo3: main/wrist/both, four parents each; never select based on achieved mask/occlusion',
        intervention=dict(collision_shapes=0,camera_cv_z_m=.1,slab_half_size_m=[.3,.3,.001],
            placement='current camera transform only, no GT actor targeting',
            off_steps='all steps outside8..19',
            actual_visibility='posthoc synchronous GT; intended camera occlusion is not an actual stratum label',
            known_limit='Wrist slab may leave near Box surfaces visible and also affect Main Box; retain and report these cases, never discard as unsuccessful stress'),
        sequence_policy='fixed121 observations/120 expert actions; keep expert failures and post-success observations; no student training data expansion',
        GT_contract='same-render RGB-D+actor segmentation readback and unchanged physical clock; GT only in sidecar; stop on nonexact sidecar rather than promote approximate reference',
        action_clock='observation t -> expert action t -> observation t+1; phase trace for evaluation only',
        comparison='same parent/frame B0 and one Development-selected, source/config-frozen challenger; same Graph builder, Graph21/Edge4/cap128/current-only/Box Main only',
        minimum_support='32 normal/12 stress is operational minimum, not statistical sufficiency; report Parent CI and counts and classify unsupported safety strata INCONCLUSIVE',
        power_review='registered Development-only review retained; no test-based sample-size increase or retuning; zero events cannot certify <1% population error',
        no_final_driven_tuning=True,Stage3_datasets_modified=False,
        capture_authorization='collector must additionally verify final candidate source/config/adoption receipt hashes and allow_independent_final_capture before any registered reset',
        candidate_selected=False,Internal_opened=False,Final_RGB_opened=False,
        source_sha256=sha(Path(__file__)),collector_source_sha256=sha(Path(__file__).with_name('final_capture.py')),
        stress_source_sha256=sha(Path(__file__).with_name('stress_capture.py')))
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_final_collection_preregistration',
            seed=1011202642,root=root/'experiments',job_type='analysis',config=plan) as logger:
        write(root/'final_collection_design.json',plan)
        write(root/'registration_receipt.json',dict(status='COLLECTION_DESIGN_REGISTERED_WITHOUT_FINAL_CAPTURE',
            run_id=logger.run_id,design_sha256=sha(root/'final_collection_design.json'),
            candidate_selected=False,Internal_and_Final_opened=False,whole_goal_complete=False))
        logger.link_output(root/'registration_receipt.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)

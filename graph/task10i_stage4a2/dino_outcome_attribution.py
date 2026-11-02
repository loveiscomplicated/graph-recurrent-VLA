"""Reconcile conservative rectangle support with actual SAM outcomes.

A mixed prompt can still produce a correct actor mask. Never count an unknown
rectangle label as an actual model failure. Retain both old diagnostic funnels.
"""
import argparse
from collections import Counter
import inspect
from pathlib import Path
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def location(function,needle):
    lines,start=inspect.getsourcelines(function)
    matches=[start+i for i,line in enumerate(lines) if needle in line]
    assert matches
    path=Path(inspect.getsourcefile(function))
    return dict(file=str(path),line=matches[0],source_sha256=sha(path))


def run(root,source):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.initialization import candidates,assign
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
    root.mkdir(parents=True,exist_ok=False)
    rows=read(source/'dino_assignment_error_cases.json')
    sites={
        'SCORE_THRESHOLD':location(GroundingDinoProcessor.post_process_grounded_object_detection,'keep = scores > threshold'),
        'AREA_FILTER':location(candidates,'MAX_BOX_FRAC'),
        'TOP5':location(candidates,'[:limit]'),
        'JOINT_NULL_SELECTION':location(assign,'score > best_score'),
        'JOINT_SELECTS_DIFFERENT_CANDIDATE':location(assign,'score > best_score'),
        'SAM_MASK_NOT_CORRECT_AFTER_LOCALIZED_PROMPT':location(OnlineVisionAdapter.process,'masks[oid-1] = mask'),
    }
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_DINO_actual_outcome_attribution',seed=42,
            root=root/'experiments',job_type='analysis',config=dict(source_sha256=sha(Path(__file__)),
            input_sha256=sha(source/'dino_assignment_error_cases.json'),new_model_forwards=0)) as logger:
        result=[];counts=Counter();unknown_good=[]
        for row in rows:
            actual=row['actual_SAM_mask_metrics'];local=row['localization_only_first_drop']
            if row['visible_pixels']<8:outcome='NOT_GT_VISIBLE'
            elif actual['correct_mask_availability']==1:outcome='NONE_T0_CORRECT_MASK'
            elif local=='ADOPTED_SUPPORTED_LOCALIZATION':outcome='SAM_MASK_NOT_CORRECT_AFTER_LOCALIZED_PROMPT'
            else:outcome=local
            r=dict(**row,operational_first_failure=outcome,
                first_drop_source_site=sites.get(outcome),
                SAM_site_is_observation_not_proven_foundation_cause=outcome=='SAM_MASK_NOT_CORRECT_AFTER_LOCALIZED_PROMPT',
                rectangle_support_is_not_operational_correctness=True)
            if outcome=='NONE_T0_CORRECT_MASK' and not row['chosen_supported_localization']:
                unknown_good.append(dict(episode=row['episode'],camera=row['camera'],object_id=row['object_id'],
                    old_actor_support_drop=row['actor_supported_first_drop'],actual_SAM=actual))
            if row['visible_pixels']>=8:counts[row['camera'],row['object_id'],outcome]+=1
            result.append(r)
        write(root/'dino_assignment_error_cases.json',result)
        csv_write(root/'dino_operational_outcome_funnel.csv',[dict(camera=c,object_id=o,operational_outcome=d,count=n,
            graph_eligible=not(c=='hand_camera' and o==1)) for (c,o,d),n in sorted(counts.items())])
        write(root/'mixed_prompt_correct_mask_cases.json',unknown_good)
        write(root/'scope.json',dict(status='HISTORICAL_DINO_ACTUAL_OUTCOME_RECONCILED',run_id=logger.run_id,
            rows=len(result),mixed_or_unconfirmed_rectangles_with_correct_SAM=len(unknown_good),
            input_sha256=sha(source/'dino_assignment_error_cases.json'),source_sha256=sha(Path(__file__)),
            previous_actor_support_funnel_preserved=True,previous_counts_not_relabeled_as_true_failures=True,
            original_localized_detection_counts_60_58_20_0_unchanged=True,
            actual_correct_mask_counts_are_separate=True,first_drop_source_sites=sites,
            new_model_forwards=0,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path)
    a=p.parse_args();run(a.root,a.source)

"""Explicit global Graph-missing controls supplement the Wrist-source prefixes.

Original four prefixes have fused correct-point presence already. Select global
Peg/Box misses from Development only before supplemental Oracle forwards, and
include the native grouped GT-box control. No practical candidate is retuned.
"""
import argparse
from pathlib import Path
import time
from .run_pipeline import read,write


def run(output,final,audits):
    from utils.experiment_logger import ExperimentLogger
    from .oracle_windows import run as oracle
    from .oracle_grouped_box import run as grouped
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_global_Graph_missing_Oracle_controls',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(condition_type='ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE',candidate_retuning=False)) as log:
        write(output/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_FROZEN_EVALUATION',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        rows=read(audits/'corrected/development_representative_points.json');choices={};selected=[]
        for oid in (0,1):
            cases=[r for r in rows if r['reference_kind']=='EXACT_GT' and r['object_id']==oid and r['visible_reference_points']>=8 and r['correct']==0]
            if not cases:continue
            r=min(cases,key=lambda r:(str(r['episode']),r['timestep']));eid=r['episode']
            end=max(choices.get(eid,{}).get('last_timestep',0),r['timestep']+5)
            choices[eid]=dict(episode=eid,last_timestep=end,reason='GT-visible global Graph has no correct point')
            selected.append(dict(episode=eid,parent_id=r['parent_id'],timestep=r['timestep'],object_id=oid,points=r['points'],correct=r['correct']))
        assert selected
        write(output/'registered_missing_cases.json',dict(selection='first lexical Development episode and earliest actual global no-correct-point frame per object',cases=selected,
            selected_before_supplemental_forward=True,not_population_estimate=True,shared_validation_not_used=True))
        try:
            write(output/'pipeline_state.json',dict(status='RUNNING',stage='GLOBAL_MISSING_O0_O4',run_id=log.run_id))
            oracle(output/'oracles',audits/'baseline',audits/'corrected',choices=choices)
            write(output/'pipeline_state.json',dict(status='RUNNING',stage='NATIVE_GROUPED_GT_BOX_CONTROL',run_id=log.run_id))
            grouped(output/'grouped',final,audits,oracle_root=output/'oracles')
            write(output/'scope.json',dict(status='GLOBAL_GRAPH_MISSING_ORACLES_FINISHED',run_id=log.run_id,
                condition_type='ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE',cases=selected,original_four_prefixes_had_fused_presence_already=True,
                separate_supplement_not_a_new_candidate=True,no_production_claim=True,no_selection_changes=True))
            log.link_output(output/'scope.json',kind='metrics')
            write(output/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='GLOBAL_GRAPH_MISSING_ORACLES_FINISHED',run_id=log.run_id))
        except BaseException as exc:
            write(output/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('final',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.final,a.audits)

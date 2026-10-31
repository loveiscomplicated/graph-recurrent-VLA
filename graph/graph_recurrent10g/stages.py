"""One stage per subprocess: CUDA memory is released at every boundary."""
import argparse
import fcntl
import signal
import shutil
from pathlib import Path
from graph.plan_a_bc.common import read,write,sha


def run(root,stage):
    from . import verification as v
    from . import data,concurrency,eval_runner,geometry_eval,analysis,report
    cfg=read(root/'config.json')
    if stage=='PREFLIGHT':return v.preflight(root)
    if stage=='DATA_VERIFICATION':return data.prepare(root)
    if stage=='MODEL_CONTRACT_TEST':return v.contracts(root)
    if stage=='LAMBDA_CALIBRATION':return v.calibration(root)
    if stage=='PRELIMINARY_PROTOCOL_FREEZE':return v.freeze(root)
    if stage=='TRAINING_SMOKE':return v.training_smoke(root)
    if stage=='RUNTIME_ROLLOUT_SMOKE':return eval_runner.run(root,'runtime_smoke')
    if stage=='CONCURRENCY_BENCHMARK':return concurrency.benchmark(root)
    if stage=='TRAINING_CONFIG_FREEZE':
        v.freeze(root)
        write(root/'full_launch_gate.json',dict(status='PASS',benchmark_sha256=sha(root/'concurrency_benchmark.json'),
            smoke_sha256=sha(root/'resume_verification.json'),runtime_sha256=sha(root/'runtime_smoke_verification.json')))
        return dict(status='PASS',files=['full_launch_gate.json'])
    if stage=='PARALLEL_TRAIN_A_B_C':return concurrency.train(root)
    if stage=='CHECKPOINT_VERIFICATION':return v.checkpoint_verification(root)
    if stage=='DEVELOPMENT_CLOSED_LOOP':return eval_runner.run(root,'development')
    if stage=='MODEL_SELECTION_FREEZE':return eval_runner.select(root)
    if stage=='FINAL_CLOSED_LOOP':return eval_runner.run(root,'final')
    if stage=='GEOMETRY_EVALUATION':return geometry_eval.run(root)
    if stage=='ACTION_CORRECTION_EVALUATION':return geometry_eval.action_correction(root)
    if stage=='ACTION_AND_FAILURE_ANALYSIS':return analysis.run(root)
    if stage=='REPORT_GENERATION':return report.generate(root)
    if stage=='ARTIFACT_HASH_VERIFICATION':return report.artifact_verify(root)
    raise ValueError(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    owner=(a.root/('stage_'+a.stage+'.lock')).open('a')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise InterruptedError('stage interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    result=run(a.root,a.stage)
    if result.get('status')!='PASS':raise ValueError('stage completion predicate')
    write(a.root/'stage_results'/(a.stage+'.json'),result)

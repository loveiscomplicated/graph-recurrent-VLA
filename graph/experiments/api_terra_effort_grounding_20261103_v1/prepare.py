"""Freeze existing native512 cohort and experiment contract; never opens GT arrays."""
from common import *
import shutil
from PIL import Image
import numpy as np


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    assert not (ROOT/'verification.json').exists(), 'Existing experiment cannot be overwritten'
    old=read(HIGH_ROOT/'verification.json')
    assert old['stage']=='COMPLETE'
    for group in ['protocol_files','api_files','raw_files','sam2_files']:
        for name,h in old[group].items(): assert sha(HIGH_ROOT/name)==h,name
    images=read(HIGH_ROOT/'input_manifest.json')['images']
    ev=read(HIGH_ROOT/'evaluation_manifest.json')['images']
    assert len(images)==60 and len(ev)==60 and len({c['parent_id'] for c in images})==15
    assert (HIGH_ROOT/'grounding_prompt.txt').read_text()==PROMPT
    for c in images:
        assert sha(c['image_path'])==c['image_sha256']
        with Image.open(c['image_path']) as im:
            assert im.mode=='RGB' and im.size==(512,512) and not im.info
            assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==c['RGB_sha256']
    for c in ev: assert sha(c['GT_path'])==c['GT_sha256']  # Opaque hash only; no GT content read.
    roots=[BASELINE,SOL_ROOT,HIGH_ROOT,REPO/'outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1']
    ledger_paths=[p/'request_ledger.json' for p in roots[:3]]+[BASELINE/'first_request_repeat/ledger.json',roots[3]/'request_ledger.json']
    prior=sum(ledger_total(read(p)['jobs']) for p in ledger_paths)
    assert abs(prior-PRIOR_ACCOUNTED_USD)<1e-8,(prior,PRIOR_ACCOUNTED_USD)
    maximum=sum(budget_reservation(m,PROMPT)*60 for m in MODELS)
    assert maximum+prior<20
    print(json.dumps(dict(network_calls=0,images=60,parents=15,new_requests_max=180,
        effort_levels=list(MODELS),high='reuse existing',prior_accounted_usd=prior,
        new_reservation_usd=maximum,combined_reservation_usd=maximum+prior,dry_run=args.dry_run)))
    if args.dry_run:return
    ROOT.mkdir(parents=True,exist_ok=False)
    for name in ['grounding_prompt.txt','input_manifest.json','evaluation_manifest.json']:
        shutil.copy2(HIGH_ROOT/name,ROOT/name)
    write(ROOT/'model_identity.json',{m:{k:v for k,v in c.items() if k!='key'} for m,c in MODELS.items()})
    write(ROOT/'protocol.json',dict(identity=ROOT.name,created_utc=now(),requests_max=180,retries=0,
        budget_total_usd=20,prior_accounted_usd=prior,new_reservation_usd=maximum,
        models=read(ROOT/'model_identity.json'),high='reused existing Terra Standard high; no new high request or masks',
        scheduling='Sequential requests, rotate effort order by image, no warmup, no retry, no replacement',
        prompt_sha256=sha(ROOT/'grounding_prompt.txt'),service_tier='default',original_image_detail='original',
        max_output_tokens=5120,store=False,input='One full native512 PNG + same frozen appearance prompt, fresh request',
        no_input=['GT','DINO','SAM2','parent ID','other frame','previous response','crop','point','tools'],
        coordinate_system='Full original RGB, normalized 0..1000; multiply .512; no repair/clipping',
        format_modes={'strict':'exact JSON object only','fence':'exact JSON or one entire-answer code fence; no text search/repair'},
        sam2='Existing SAM2.1 Hiera Small, one load, per-image embedding reused and independent object box prompts',
        sam2_checkpoint_sha256=SAM_SHA,weights='FP32',compute='BF16 autocast',multimask_output=False,logit_threshold=0,
        GT_barrier='Freeze all API predictions, then all SAM2 masks, then open GT for scoring',
        primary='Both visible objects mask IoU>=.8, strict and fence reported separately',visible_pixels=16,
        bootstrap=dict(unit='physical parent',replicates=10000,seed=42),
        prior_limitations='Same 15 known train parents; no held-out generalization. High is historical.',
        price_sources=['https://developers.openai.com/api/docs/models/gpt-6.1-terra'],
        scope=dict(production=False,render=False,training=False,tracking=False,graph=False,controller=False,
                   Stage4A2='PAUSED_SCOPE_REVIEW',commit=False,publish=False)))
    protected={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
    for c in images:protected[c['image_path']]=c['image_sha256']
    frozen=['grounding_prompt.txt','input_manifest.json','evaluation_manifest.json','model_identity.json','protocol.json']
    write(ROOT/'verification.json',dict(stage='PROTOCOL_FROZEN',prepared_utc=now(),
        protocol_files={n:sha(ROOT/n) for n in frozen},protected_before=protected,
        source_before={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix in ('.py','.md')},
        request_count=0,SAM2_loads=0))


if __name__=='__main__':main()

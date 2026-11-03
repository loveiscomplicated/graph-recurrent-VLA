"""Freeze same known RGB cohort and prompt, without opening semantic GT."""
from common import *
import argparse
import shutil
from PIL import Image
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    assert REPO == Path('/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA')
    prior=read(BASELINE/'verification.json')
    for group in ['protocol_files','api_files','raw_files','sam2_files']:
        for name,h in prior[group].items(): assert sha(BASELINE/name)==h,name
    images=read(BASELINE/'input_manifest.json')['images']
    evaluation=read(BASELINE/'evaluation_manifest.json')
    assert len(images)==60 and len({c['parent_id'] for c in images})==15
    assert (BASELINE/'grounding_prompt.txt').read_text()==PROMPT
    for c in images:
        assert sha(c['image_path'])==c['image_sha256']
        with Image.open(c['image_path']) as im:
            assert im.mode=='RGB' and im.size==(512,512) and not im.info
            assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==c['RGB_sha256']
    for c in evaluation['images']: assert sha(c['GT_path'])==c['GT_sha256']
    maximum=sum(budget_reservation(m,PROMPT)*60 for m in MODELS)
    assert maximum+PRIOR_ACCOUNTED_USD<=20
    print(json.dumps(dict(network_calls=0,requests_max=120,images=60,parents=15,
                         reserved_upper_usd=maximum,external_reserved_upper_usd=PRIOR_ACCOUNTED_USD,
                         combined_upper_usd=maximum+PRIOR_ACCOUNTED_USD,dry_run=args.dry_run)))
    if args.dry_run:return
    if (ROOT/'verification.json').exists():verify_frozen();return
    ROOT.mkdir(parents=True,exist_ok=True)
    for n in ['grounding_prompt.txt','input_manifest.json','evaluation_manifest.json']:
        shutil.copy2(BASELINE/n,ROOT/n)
    write(ROOT/'model_identity.json',{m:{k:v for k,v in c.items() if k!='key'} for m,c in MODELS.items()})
    write(ROOT/'protocol.json',dict(identity=ROOT.name,created_utc=now(),requests_max=120,retries=0,
        budget_total_usd=20,external_reserved_upper_usd=PRIOR_ACCOUNTED_USD,new_reserved_upper_usd=maximum,
        models=read(ROOT/'model_identity.json'),prompt_sha256=sha(ROOT/'grounding_prompt.txt'),
        settings={p:dict(reasoning='high',detail='original',max_output=5120) for p in MODELS},
        coordinate_system='Full original RGB, normalized 0..1000; multiply by .512; no repair/clipping',
        input='One full native512 RGB plus frozen appearance prompt only; new conversation per request',
        format_modes={'strict':'Exact JSON object only',
                      'fence':'Exact JSON or whole-answer single code fence, optional json language, whitespace allowed; no prose/search/repair'},
        sam2='One load, per-image embedding reused, independent box prompts; valid fence-mode boxes only. Strict scoring gates masks by strict validity.',
        sam2_checkpoint_sha256=SAM_SHA,weights='FP32',compute='BF16 autocast',multimask_output=False,logit_threshold=0,
        primary='Both visible objects mask IoU>=.8, reported separately for strict/fence modes',
        visible_pixels=16,bootstrap=dict(unit='physical parent',replicates=10000,seed=42),
        GT_barrier='All model outputs frozen then all masks frozen then GT scoring',
        previous_results='Astra and Sol results reused for final comparison; no new call to those models',
        new_model_mask_budget=240,
        price_sources=['https://developers.openai.com/api/docs/models/gpt-6.1-terra','https://developers.openai.com/api/docs/models/gpt-6-luna'],
        sample_claim='Known 15 Train parents, same scene model comparison; no held-out generalization claim',
        idle_sleep_guard='scoped caffeinate -i -w process, on macOS',
        scope=dict(production=False,new_DINO=False,render=False,training=False,tracking=False,
                   Stage4A2='PAUSED_SCOPE_REVIEW',commit=False,publish=False)))
    protected={str(p):sha(p) for p in BASELINE.rglob('*') if p.is_file()}
    protected.update(prior['protected_before'])
    for c in images:protected[c['image_path']]=c['image_sha256']
    frozen=['grounding_prompt.txt','input_manifest.json','evaluation_manifest.json','model_identity.json','protocol.json']
    write(ROOT/'verification.json',dict(stage='PROTOCOL_FROZEN',prepared_utc=now(),
        protocol_files={n:sha(ROOT/n) for n in frozen},protected_before=protected,
        source_before={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},request_count=0,SAM2_loads=0))

if __name__=='__main__':main()

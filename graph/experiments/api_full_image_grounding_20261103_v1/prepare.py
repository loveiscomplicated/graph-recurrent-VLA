from common import *
import argparse
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    assert REPO.name == 'graph-recurrent-VLA'
    old = read(OLD / 'input_manifest.json')
    cases, evaluation = [], []
    protected = {}
    for case in old['cases']:
        p = RGB_ROOT / 'images' / Path(case['image_path']).name
        gt = RGB_ROOT / '.execution/GT' / Path(case['GT_path']).name
        assert sha(p) == case['image_sha256']
        assert sha(gt) == case['GT_sha256']  # Opaque hash only; no semantic GT read.
        rgb = np.asarray(Image.open(p).convert('RGB'))
        assert rgb.shape == (512, 512, 3)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == case['RGB_sha256']
        c = {k: case[k] for k in ['id', 'parent_id', 'camera', 'frame_kind', 'timestep', 'width', 'height',
                                  'image_sha256', 'RGB_sha256']}
        c['image_path'] = str(p)
        cases.append(c)
        evaluation.append(dict(**c, GT_path=str(gt), GT_sha256=case['GT_sha256'], object_ids=case['object_ids']))
        protected[str(p)] = sha(p)
        protected[str(gt)] = sha(gt)
    assert len(cases) == 60 and len({c['parent_id'] for c in cases}) == 15
    cases.sort(key=lambda c: hashlib.sha256(('Task10I-API-FullImage-20261103-v1|' + c['id']).encode()).hexdigest())
    for base in [REPO / 'graph/experiments/sam2_mask_verifier_tracking_20261102_v1',
                 REPO / 'graph/experiments/qwen35_full_image_grounding_20261102_v1']:
        if base.exists():
            for p in base.glob('*'):
                if p.is_file(): protected[str(p)] = sha(p)
    for p in [OLD / 'input_manifest.json', OLD / 'sam2_initialization_metrics.json',
              OLD / 'selected_candidates.json', OLD / 'sam2_initialization_cases.csv']:
        protected[str(p)] = sha(p)
    maximum = sum(budget_reservation(m, PROMPT) * 60 for m in MODELS)
    assert maximum <= 20., maximum
    print(json.dumps({'images': 60, 'parents': 15, 'requests_max': 180,
                      'reservation_upper_usd': maximum, 'network_calls': 0,
                      'dry_run': args.dry_run}))
    if args.dry_run:
        return
    if (ROOT / 'verification.json').exists():
        verify_frozen()
        print('Already prepared; immutable protocol verified')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'grounding_prompt.txt').write_text(PROMPT)
    write(ROOT / 'input_manifest.json', {'images': cases, 'known_data': True})
    write(ROOT / 'evaluation_manifest.json', {'images': evaluation})
    identity = {m: {k: v for k, v in c.items() if k != 'key'} for m, c in MODELS.items()}
    write(ROOT / 'model_identity.json', identity)
    write(ROOT / 'protocol.json', dict(identity='20261103_v1', created_utc=now(),
        budget_usd=20., requests_max=180, retries=0, dry_run=False,
        input='One complete original native512 RGB; no DINO/GT/mask/history/tools',
        prompt_sha256=sha(ROOT / 'grounding_prompt.txt'), settings={
            'openai': {'reasoning': 'high', 'detail': 'original', 'max_output': 5120},
            'zai': {'thinking': True, 'reasoning': 'max', 'temperature': 1, 'top_p': .95, 'max_output': 8192},
            'ollama': {'thinking': True, 'temperature': 1, 'top_p': .95, 'top_k': 64, 'max_output': 8192}},
        primary='Both visible objects SAM2 mask IoU>=0.8', visible_pixels=16,
        bootstrap={'unit': 'physical parent', 'replicates': 10000, 'seed': 42},
        GT_barrier='All API predictions frozen, then all masks frozen, then semantic GT scoring',
        model_aliases_unpinned=True, no_generalization_claim=True,
        sam2={'checkpoint': SAM_CHECKPOINT, 'sha256': SAM_SHA,
              'config': 'configs/sam2.1/sam2.1_hiera_s.yaml', 'weights': 'FP32',
              'compute': 'BF16 autocast', 'multimask_output': False, 'logit_threshold': 0},
        price_sources={'openai': 'https://developers.openai.com/api/docs/pricing',
                      'zai': 'https://docs.z.ai/guides/overview/pricing',
                      'ollama': 'https://ollama.com/library/gemma4:31b-cloud'},
        scope={'production': False, 'Stage4A2': 'PAUSED_SCOPE_REVIEW', 'tracking': False,
               'new_DINO': False, 'simulator_render': False, 'training': False}))
    frozen = ['grounding_prompt.txt', 'input_manifest.json', 'evaluation_manifest.json',
              'model_identity.json', 'protocol.json']
    write(ROOT / 'verification.json', dict(stage='PROTOCOL_FROZEN', prepared_utc=now(),
        protocol_files={n: sha(ROOT / n) for n in frozen}, protected_before=protected,
        source_before={p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()},
        request_count=0, SAM2_loads=0))

if __name__ == '__main__': main()

"""Frozen, independent API grounding experiment; no production imports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
ROOT = REPO / 'outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1'
OLD = REPO / 'outputs/task10i/stage4a2/sam2_mask_verifier_tracking_20261102_v1'
RGB_ROOT = REPO / 'outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1'
REPORT = REPO / 'docs/visual-graph-integration/10i_stage4a2_api_openai_family_grounding_comparison.md'
REMOTE = '/workspace/task10i_stage4a2_api_terra_luna_grounding_20261103_v1'
SAM_CHECKPOINT = '/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt'
SAM_SHA = '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
BASELINE = REPO / 'outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1'
SOL_ROOT = REPO / 'outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1'
PRIOR_ACCOUNTED_USD = 16.32923086  # Conservative external reservation ceiling, not a billed cost.
MODELS = {
    'terra': dict(kind='openai', model='gpt-6.1-terra',endpoint='https://api.openai.com/v1/responses',
                  key='OPENAI_API_KEY',input_price=1.,output_price=5.,max_output=5120),
    'luna': dict(kind='openai', model='gpt-6-luna',endpoint='https://api.openai.com/v1/responses',
                 key='OPENAI_API_KEY',input_price=.1,output_price=.5,max_output=5120),
}
PROMPT = '''Locate the following two objects directly in this complete RGB image.
Identify them by their visible appearance:
PEG: A red-and-white elongated rectangular piece. It may be held by the gripper or partly outside the image. A separate object held by the gripper is not a robot part.
BOX: A tan or brown block-shaped fixture. An opening may be visible depending on the view. Locate the fixture itself, not only its opening.
ROBOT: Gray or white robot arm links and dark or black gripper fingers. Do not include robot parts as PEG or BOX.
Use color, shape, and visible parts together. Do not invent an object that cannot be seen. Return null if an object's location cannot be determined reliably.
For each object, tightly enclose its visible part. Do not estimate invisible parts outside the image. Return at most one PEG box and one BOX box.
Coordinates refer to the complete original image, normalized from 0 to 1000: top-left (0,0), bottom-right (1000,1000). Each box is [x1,y1,x2,y2], with x1<x2 and y1<y2.
Return exactly one JSON object with keys "peg" and "box". Each value must be a box array or null. Return only the JSON object, without Markdown, explanation, or additional text.
'''

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)

def csvwrite(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})

def validate_box(value):
    if value is None:
        return dict(status='NULL', normalized=None, pixels=None)
    valid = (isinstance(value, list) and len(value) == 4
             and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in value))
    if not valid or not (value[0] < value[2] and value[1] < value[3]):
        def diagnostic_value(x):
            if isinstance(x,float) and not math.isfinite(x): return '<nonfinite:'+repr(x)+'>'
            if isinstance(x,list): return [diagnostic_value(y) for y in x]
            if isinstance(x,dict): return {k:diagnostic_value(y) for k,y in x.items()}
            return x
        return dict(status='INVALID_BOX', normalized=diagnostic_value(value), pixels=None)
    return dict(status='VALID', normalized=value, pixels=[v * .512 for v in value])

def strict_parse(text):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate key')
            d[k] = v
        return d
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict) or set(value) != {'peg', 'box'}:
            raise ValueError('exact keys required')
        return dict(json_valid=True, **{k: validate_box(value[k]) for k in ['peg', 'box']})
    except (ValueError, TypeError):
        return dict(json_valid=False, **{k: dict(status='JSON_INVALID', normalized=None, pixels=None) for k in ['peg', 'box']})

def diagnostic_parse(text):
    """Only inspect the final answer; never promote recovered boxes to primary."""
    for match in re.finditer(r'\{[^{}]*\}', text):
        p = strict_parse(match.group())
        if p['json_valid']:
            return p
    return strict_parse(text)

def fence_parse(text):
    """Accept exact JSON or one whole-answer code fence only. No searching/repair."""
    parsed = strict_parse(text)
    if parsed['json_valid']:
        return parsed
    match = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*', text, flags=re.DOTALL | re.IGNORECASE)
    return strict_parse(match.group(1)) if match else strict_parse('')

def budget_reservation(provider, prompt):
    c = MODELS[provider]
    # ASCII byte bound for text + conservative image/protocol allowances.
    input_bound = len(prompt.encode()) + 2048
    return (input_bound * c['input_price'] + c['max_output'] * c['output_price']) / 1e6

def used_cost(provider, usage):
    if not usage or usage.get('input_tokens') is None or usage.get('output_tokens') is None:
        return None
    if any(type(usage[k]) not in (int, float) or not math.isfinite(usage[k]) or usage[k] < 0
           for k in ['input_tokens', 'output_tokens']):
        return None
    c = MODELS[provider]
    # Conservatively do not apply cached-input discounts.
    return (usage['input_tokens'] * c['input_price'] + usage['output_tokens'] * c['output_price']) / 1e6

def ledger_total(jobs):
    return sum(j.get('accounted_usd', j.get('reserved_usd', 0.)) for j in jobs.values())

def verify_frozen():
    v = read(ROOT / 'verification.json')
    for name, expected in v['protocol_files'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in v['source_before'].items():
        assert sha(SOURCE / name) == expected, name
    return v

def logger(stage):
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(REPO))
    from utils.experiment_logger import ExperimentLogger
    return ExperimentLogger(team='integration', model='Terra_Luna_full_image_grounding', seed=42,
                            job_type='evaluate', root=ROOT / '.execution/experiments',
                            config={'stage': stage, 'protocol_sha256': sha(ROOT / 'protocol.json')})

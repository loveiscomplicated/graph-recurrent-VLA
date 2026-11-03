"""Verify the completed report and index preserved research artifacts; no inference."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / 'outputs/task10i/stage4a2/qwen35_vla_research_audit_20261102_v1'
P = R.parent / 'qwen35_perception_diagnostic_20261102_v1'
REPORT = ROOT / 'docs/visual-graph-integration/10i_stage4a2_qwen35_vla_research_and_perception_diagnostic.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (R / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


text = REPORT.read_text()
prompt = json.loads((P / 'prompt_identity.json').read_text())['conditions']['appearance_three']['prompt']
assert re.search(r'```text\n(.*?)\n```', text, re.S)[1] == prompt
assert json.loads((R / 'verification.json').read_text())['status'] == 'PASS'
assert json.loads((P / 'verification.json').read_text())['all_protected_SHA_unchanged']
assert all(sha(P / name) == entry['sha256'] for name, entry in json.loads((P / 'artifact_index.json').read_text())['files'].items())
links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
local_links = [link for link in links if not link.startswith(('https://', 'http://', '#'))]
# The index is written after the report's own SHA is fixed; it does not hash itself.
index_path = R / 'artifact_index.json'
assert all((REPORT.parent / link.split('#')[0]).resolve().exists() or (REPORT.parent / link).resolve() == index_path for link in local_links)
sources = list((ROOT / 'graph/experiments/qwen35_perception_diagnostic_20261102_v1').glob('*.py')) + list(Path(__file__).parent.glob('*.py'))
for source in sources:
    ast.parse(source.read_text(), filename=str(source))
write('completion_verification.json', dict(
    status='PASS', local_report_links=len(local_links), frozen_prompt_matches_report=True,
    original_remote_artifacts_still_unchanged=True, syntax_checked_sources=len(sources),
    independent_analysis='verification.json', no_additional_inference=True,
    backend='CUDA inference executed only on Vast.ai 58576054; local analysis only',
    report_sha256=sha(REPORT), production_changed=False, published=False,
))
files = {str(path.relative_to(R)): dict(sha256=sha(path), bytes=path.stat().st_size)
         for path in sorted(R.rglob('*')) if path.is_file() and path != index_path}
source_files = {str(path.relative_to(ROOT)): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
write('artifact_index.json', dict(schema_version=1, status='COMPLETE_VERIFIED',
    purpose='Known-case perception diagnosis and primary-source VLA research; not independent accuracy',
    files=files, analysis_source_sha256=source_files,
    report=dict(path=str(REPORT.relative_to(ROOT)), sha256=sha(REPORT)),
    inference_artifact_index=dict(path=str((P / 'artifact_index.json').relative_to(ROOT)), sha256=sha(P / 'artifact_index.json')),
))
assert all((REPORT.parent / link.split('#')[0]).resolve().exists() for link in local_links)
assert all(sha(R / name) == entry['sha256'] for name, entry in files.items())
print(json.dumps(dict(status='COMPLETE_VERIFIED', indexed_files=len(files), local_report_links=len(local_links), report_sha256=sha(REPORT)), indent=2))

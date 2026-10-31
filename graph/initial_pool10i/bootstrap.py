from pathlib import Path
import shutil,hashlib,json
B=Path('/workspace/task10i_stage1_20261031');O=Path('/workspace/task10g_f1_abc_20261031')
if not (B/'source').exists():shutil.copytree(O/'source',B/'source',ignore=shutil.ignore_patterns('__pycache__','.git','outputs','experiments'))
# Copy authoritative audit files from staged local transfer. Never alter original audit.
print('source ready',B)

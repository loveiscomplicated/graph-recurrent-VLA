"""Machine-local paths; historical artifacts keep their recorded Vast.ai paths."""
import os
from pathlib import Path

BASE = Path(os.environ.get('LEGACY_FPS_ROOT', 'outputs/legacy_fps')).expanduser().resolve()
# Captures are deliberately excluded from the minimal publication.
CAPTURE = Path(os.environ.get('LEGACY_FPS_CAPTURE', 'outputs/task09c_capture')).expanduser().resolve()

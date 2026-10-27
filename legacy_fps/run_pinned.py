"""Execute the port with frozen common dependencies in a clean interpreter."""
import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] /
                        'docs/legacy_fps/integration_manifest.json')
    parser.add_argument('module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Import only the extension package before pinning common packages. A new
    # interpreter prevents graph/visual/utils modules cached by callers winning.
    bootstrap = '''
import json, runpy, sys
from pathlib import Path
checkout, source, manifest, module, *arguments = sys.argv[1:]
sys.path.insert(0, checkout)
import legacy_fps
sys.path.insert(0, source)
from legacy_fps.integration_guard import validate_source
validate_source(source, manifest, extension_root=checkout)
sys.argv = [module, *arguments]
runpy.run_module(module, run_name='__main__', alter_sys=True)
'''
    return subprocess.call([sys.executable, '-c', bootstrap,
                            str(Path(__file__).resolve().parents[1]),
                            str(args.source_root.resolve()), str(args.manifest.resolve()),
                            args.module, *args.arguments])


if __name__ == '__main__':
    raise SystemExit(main())

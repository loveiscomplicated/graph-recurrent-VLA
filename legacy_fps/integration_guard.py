"""Opt-in loader for the independently revalidated Legacy FPS candidate."""
import hashlib
import json
from pathlib import Path


def validate_source(source_root, manifest, *, extension_root=None):
    source_root = Path(source_root).resolve()
    extension_root = Path(extension_root or source_root).resolve()
    spec = json.loads(Path(manifest).read_text())
    failures = []
    checked = 0
    for group in ('required_dependency_source_hashes', 'extension_source_hashes'):
        root = source_root if group == 'required_dependency_source_hashes' else extension_root
        for relative, expected in spec[group].items():
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                failures.append(relative)
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(relative)
            checked += 1
    if failures:
        raise RuntimeError('source identity mismatch: ' + ', '.join(failures))
    return {'status': 'PASS', 'checked_files': checked}


def load_candidate(source_root, manifest, *, enabled=False, extension_root=None):
    if not enabled:
        return None
    validate_source(source_root, manifest, extension_root=extension_root)
    from legacy_fps.adapter import frame_points
    from legacy_fps.pipeline import LegacySystem
    from graph.persistent_geometry.encoder import GeometryRegressor
    # Hashing a different tree must never authorize already-imported mutable
    # modules. Use run_pinned to select the dependency tree before any imports.
    import sys
    roots = {'graph': Path(source_root).resolve(), 'visual': Path(source_root).resolve(),
             'utils': Path(source_root).resolve(),
             'legacy_fps': Path(extension_root or source_root).resolve()}
    for name, module in tuple(sys.modules.items()):
        root = roots.get(name.split('.')[0])
        filename = getattr(module, '__file__', None)
        if root is not None and filename and root not in Path(filename).resolve().parents:
            raise RuntimeError('loaded module outside validated source: ' + name)
    return {'frame_points': frame_points, 'system': LegacySystem,
            'diagnostic_regressor': GeometryRegressor}

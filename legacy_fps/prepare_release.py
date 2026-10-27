"""Prepare the explicitly selected, private publication without changing originals.

This module only stages local files. Transport is tools.publish_experiment.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

from utils.experiment_artifacts import register_existing_output

CHECKOUT = Path(__file__).resolve().parents[1]
MAIN = '8c41299e9d3e4989bbf32ab1b894dc49'
LEARNING = 'b220c3fdced649adab3edee0c7d5378b'
REVALIDATION = 'bd465b9d57ed4fd6843e87c950896edc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def owner(path, runs):
    for run in runs:
        if path == run or path.startswith(run + '/'):
            return Path(run).name
    return None


def stage(artifact_root, output):
    artifact_root, output = artifact_root.resolve(), output.resolve()
    if output.exists():
        raise ValueError('output must be a new directory; original artifacts are read only')
    if artifact_root == output or artifact_root in output.parents:
        raise ValueError('output must be outside the original artifact tree')
    selection = json.loads((CHECKOUT / 'docs/legacy_fps/release_selection.json').read_text())
    output.mkdir(parents=True)
    payload = output / 'payload'
    payload.mkdir()
    inventory = []

    def copy(source, target, run_id=None):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('selected source must be a regular file')
        sidecar = source.with_name(source.name + '.tracking.json')
        known = json.loads(sidecar.read_text()).get('run_id') if sidecar.exists() else None
        if known and run_id and known != run_id:
            raise ValueError('artifact owner differs from its selected run')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if known or run_id:
            write(target.with_name(target.name + '.tracking.json'),
                  dict(run_id=known or run_id, original_artifact=str(source.relative_to(artifact_root))))
        inventory.append(dict(source=str(source), source_sha256=sha(source),
                              path=str(target.relative_to(output)), owner=known or run_id))

    for relative in selection['dataset']:
        copy(artifact_root / relative, payload / 'task09c_legacy_fps' / relative,
             owner(relative, selection['runs']))
    for relative in selection['checkpoint']:
        copy(artifact_root / relative, output / 'models' / relative, LEARNING)

    # Archive only the hash-pinned common files, not the 254 MB working snapshot.
    historical = json.loads((CHECKOUT / 'docs/legacy_fps/historical_integration_manifest.json').read_text())
    snapshot = artifact_root / 'revalidation_runs' / REVALIDATION / 'source_snapshot'
    archive = payload / 'pinned_sources.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed, \
            tarfile.open(fileobj=compressed, mode='w') as tar:
        for group, prefix in [('required_dependency_source_hashes', 'pinned_source'),
                              ('extension_source_hashes', 'historical_extension')]:
            for relative, expected in sorted(historical[group].items()):
                source = snapshot / relative
                if sha(source) != expected:
                    raise ValueError('historical pinned source mismatch: ' + relative)
                data = source.read_bytes()
                info = tarfile.TarInfo(prefix + '/' + relative)
                info.size = len(data); info.mode = 0o644; info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
                inventory.append(dict(source=str(source), source_sha256=expected,
                                      archive_member=info.name, owner=None))
        # Git remains local, so the private download must also carry the port.
        implementation = [*sorted((CHECKOUT / 'legacy_fps').glob('*.py')),
                          CHECKOUT / 'legacy_fps/README.md',
                          CHECKOUT / 'docs/legacy_fps/integration_manifest.json',
                          CHECKOUT / 'docs/legacy_fps/historical_integration_manifest.json']
        for source in implementation:
            data = source.read_bytes()
            info = tarfile.TarInfo('implementation/' + str(source.relative_to(CHECKOUT)))
            info.size = len(data); info.mode = 0o644; info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    # Local test dependencies are byte-identical to the archive; never modify them.
    pinned = output / 'pinned_source'
    for relative in historical['required_dependency_source_hashes']:
        target = pinned / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / relative, target)

    report = payload / 'repository/docs/visual-graph-integration/09c_legacy_fps_extension.md'
    report.parent.mkdir(parents=True)
    shutil.copy2(CHECKOUT / 'docs/visual-graph-integration/09c_legacy_fps_extension.md', report)
    # Keep the full written report beside the HTML as well.
    shutil.copy2(report, payload / 'task09c_legacy_fps/experiments' / MAIN / report.name)
    for name in ('integration_manifest.json', 'historical_integration_manifest.json', 'release_selection.json'):
        target = payload / 'release' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CHECKOUT / 'docs/legacy_fps' / name, target)

    omitted = payload / 'omitted.html'
    omitted.write_text('''<!doctype html><meta charset="utf-8"><title>Excluded evidence</title>
<h1>Minimal publication</h1><p>This link refers to a full frame tensor, map state,
capture, or other file excluded from this selected release. It is not available
in the download. See the preserved aggregate metrics and verification logs.</p>
<p>No independent Final Test was executed. Historical protection remains PARTIAL;
the later source compatibility and protection window are PASS.</p>''')
    # Rewrite only document links in publication copies. Recorded runtime paths,
    # execution Git identity, source archives and checkpoints retain original bytes.
    rewrites = []
    texts = [report, *[payload / 'task09c_legacy_fps' / p for p in selection['dataset']
                      if Path(p).suffix in ('.html', '.md')]]
    texts.append(payload / 'task09c_legacy_fps/experiments' / MAIN / report.name)
    for path in set(texts):
        body = path.read_text()

        def link(value):
            if not value or value.startswith(('#', 'https:', 'http:', 'data:', 'mailto:')):
                return value
            bare = value.split('#')[0].split('?')[0]
            if bare.startswith('/workspace/task09c_legacy_fps/'):
                target = payload / 'task09c_legacy_fps' / bare.split('/workspace/task09c_legacy_fps/', 1)[1]
            else:
                target = (path.parent / bare).resolve()
            if payload.resolve() not in target.parents or not target.is_file():
                rewrites.append(dict(document=str(path.relative_to(payload)), original=value,
                                     replacement='omitted.html'))
                return os.path.relpath(omitted, path.parent)
            return os.path.relpath(target, path.parent) + ('#' + value.split('#', 1)[1] if '#' in value else '')

        if path.suffix == '.html':
            body = re.sub(r'((?:href|src)=["\'])([^"\']+)(["\'])',
                          lambda m: m[1] + link(m[2]) + m[3], body)
        else:
            body = re.sub(r'(!?\[[^\]]*\]\()([^\s)]+)(\))',
                          lambda m: m[1] + link(m[2]) + m[3], body)
        path.write_text(body)
    write(payload / 'release/path_relocations.json', dict(
        historical_artifact_prefix='/workspace/task09c_legacy_fps',
        download_artifact_prefix='task09c_legacy_fps',
        execution_metadata_paths='provenance only; not local executable paths',
        excluded_links=rewrites))
    write(output / 'original_inventory.json', inventory)

    # Completed-run publication copies keep UUID/team/git/status exactly as recorded.
    for run_id, sub in [(MAIN, 'experiments'), (LEARNING, 'learning_runs')]:
        source = artifact_root / sub / run_id
        directory = output / 'experiments' / run_id
        directory.mkdir(parents=True)
        for name in ('metadata.json', 'config.yaml', 'metrics.json'):
            shutil.copy2(source / name, directory / name)
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['status'] != 'finished' or metadata['run_id'] != run_id:
            raise ValueError('only the selected completed run may be published')
        metadata['outputs'] = []; metadata['artifact_kinds'] = {}
        write(directory / 'metadata.json', metadata)
    learning_dir = output / 'experiments' / LEARNING
    for relative in selection['checkpoint']:
        register_existing_output(learning_dir, output / 'models' / relative, kind='checkpoint')
    for name in ('model_manifest.json', 'probe_manifest.json', 'split_manifest.json'):
        register_existing_output(learning_dir, payload / 'task09c_legacy_fps/learning_runs' / LEARNING / name,
                                 kind='metrics')
    print(json.dumps(dict(status='prepared', dataset_members=len(selection['dataset']),
                          encoders=36, probes=244, output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.artifact_root, args.output)


if __name__ == '__main__':
    main()

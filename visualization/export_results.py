"""Build a graph/capture home with a separate Korean results catalogue.

The existing capture viewer stays at index.html and captures.html. No experiment is run
or automatically published; the explicit selection uses the shared publisher.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from utils.experiment_artifacts import encoded
from visualization.export_captures import export_captures
from visualization.results import FRONTEND, build_catalog, public_catalog, resource_bytes
from visualization.persistent import persistent_episode, persistent_frame
from graph.integration_baseline.graph import EDGE_TYPES, NODE_TYPES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_static_auth(html):
    """Install private Space authentication before the browser parses assets."""
    script = (FRONTEND / 'static_auth.js').read_text()
    bootstrap = '<meta name="referrer" content="no-referrer"><script>' + script + '</script>'
    match = re.search(r'<html\b[^>]*>', html, re.I)
    return html[:match.end()] + bootstrap + html[match.end():] if match else bootstrap + html


def arrange_navigation(output, files):
    """Keep the established graph/capture screen first; results are secondary."""
    output = Path(output)
    capture = (output / 'captures.html').read_text()
    capture = capture.replace('href="./index.html">실험 결과 모아보기',
                              'href="./results.html">실험 결과 모아보기')
    capture = capture.replace('<button id="video-seek">현재 프레임에 맞추기</button>', '<button id="video-seek">현재 프레임에 맞추기</button><a id="interactive-graph" class="back-link">인터랙티브 3D 열기 →</a>')
    capture = add_static_auth(capture)
    for name in ('index.html', 'captures.html'):
        (output / name).write_text(capture)
        files[name] = dict(source=str(output / name), role='result')
    results = (FRONTEND / 'results.html').read_text()
    results = results.replace('<html lang="ko">', '<html lang="ko" data-results-data="./data/results.json">')
    results = results.replace('href="/results.css"', 'href="./results.css"').replace('src="/results.js"', 'src="./results.js"')
    results = results.replace('href="/results"', 'href="./results.html"').replace('href="/captures"', 'href="./index.html"')
    (output / 'results.html').write_text(add_static_auth(results))
    files['results.html'] = dict(source=str(output / 'results.html'), role='result')


def export_results(root, output, capture_run, geometry_runs):
    output = Path(output).resolve()
    catalog = build_catalog(root)
    if not catalog['groups']:
        raise ValueError('no result reports found')
    capture = export_captures(root, capture_run, output, geometry_runs=geometry_runs)
    selection_path = Path(capture['selection'])
    selection = json.loads(selection_path.read_text())
    files = selection['files']
    # Retain a capture alias as well as the established home entrypoint.
    (output / 'index.html').rename(output / 'captures.html')
    files['captures.html'] = files.pop('index.html')
    files['captures.html']['source'] = str(output / 'captures.html')
    provenance = []

    def save(name, body, role='result'):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        files[name] = dict(source=str(path), role=role)

    for name in ('results.css', 'results.js'):
        save(name, (FRONTEND / name).read_bytes())
    save('results-unavailable.html', add_static_auth((FRONTEND / 'results_unavailable.html').read_text()).encode())
    arrange_navigation(output, files)
    save('data/results.json', encoded(public_catalog(catalog, prefix='./reports/', capture_url='./index.html', static=True)))
    viewer = (FRONTEND / 'viewer.html').read_text()
    viewer = viewer.replace('<html lang="ko">', '<html lang="ko" data-viewer-data="./data/graphs">')
    viewer = viewer.replace('href="/style.css"', 'href="./style.css"').replace('src="/app.js"', 'src="./app.js"')
    viewer = viewer.replace('href="/">← 목록', 'href="./index.html">← 수집 데이터')
    save('viewer.html', add_static_auth(viewer).encode())
    save('app.js', (FRONTEND / 'app.js').read_bytes())
    graph_count = 0
    for capture_episode in json.loads((output / 'data/captures.json').read_text())['episodes']:
        media_path = output / 'data' / capture_episode['id'] / 'media.json'
        for run in json.loads(media_path.read_text())['runs']:
            episode_id = run['geometry_id']
            _, records, info = persistent_episode(root, episode_id, check_complete=True)
            if info['capture_id'] != capture_episode['id']:
                raise ValueError('viewer geometry does not match selected capture')
            base = 'data/graphs/' + episode_id
            detail = dict(episode=info, node_types=NODE_TYPES,
                          edge_types=['Surface / Raw EDE', *EDGE_TYPES[1:]],
                          timesteps=[record['timestep'] for record in records],
                          variants=info['variants'], default_variant=info['default_variant'])
            save(base + '/episode.json', encoded(detail))
            for variant in info['variants']:
                variant_id = variant['id']
                frames = {str(record['timestep']): persistent_frame(root, episode_id, record['timestep'], variant_id)
                          for record in records}
                save(base + '/' + variant_id + '.json', encoded(frames), role='dependency')
                graph_count += len(frames)

    for key, source in sorted(catalog['resources'].items()):
        target = Path('reports') / key
        home = os.path.relpath('results.html', target.parent).replace(os.sep, '/')
        unavailable = os.path.relpath('results-unavailable.html', target.parent).replace(os.sep, '/')
        body, _ = resource_bytes(catalog, key, home=home, unavailable=unavailable)
        if target.suffix.lower() == '.html':
            body = add_static_auth(body.decode()).encode()
        save(target.as_posix(), body, role='dependency')
        source_run = source.relative_to(catalog['root']).parts[0] if source.is_relative_to(catalog['root']) else None
        provenance.append(dict(path=target.as_posix(), source_path=key, source_run_id=source_run,
                               source_sha256=digest(source), exported_sha256=hashlib.sha256(body).hexdigest(),
                               bytes=len(body)))
    save('results_export_manifest.json', encoded(dict(schema_version=1,
        hosting_run_id=capture_run, topics=len(catalog['groups']), reports=catalog['reports'],
        entrypoints=dict(home='index.html', results='results.html', capture_alias='captures.html'),
        available_reports=catalog['available_reports'], dependencies=provenance,
        unavailable_links=catalog['issues'], policy='original reports read-only; selected web resources; running runs excluded')))
    readme = ('---\ntitle: Task09-C · 수집 데이터 확인\nemoji: 🔎\ncolorFrom: blue\ncolorTo: purple\n'
              'sdk: static\napp_file: index.html\npinned: false\n---\n\n'
              '# 수집 데이터·그래프 비교\n\n첫 화면은 Task09-C 수집 데이터·FPS 비교이며, 실험 결과 모아보기는 results.html에서 엽니다.\n'
              '실험 결과를 재계산하지 않습니다. 원본 run identity는 results_export_manifest.json에 보존합니다.\n')
    save('README.md', readme.encode())
    selection['description'] = 'Graph/capture home with a linked Korean results catalogue'
    selection_path = output.parent / (output.name + '-selection.json')
    selection_path.write_bytes(encoded(selection))
    return dict(output=str(output), selection=str(selection_path), topics=len(catalog['groups']),
                reports=catalog['reports'], available_reports=catalog['available_reports'],
                report_resources=len(provenance), graph_frames=graph_count, total_files=len(files),
                total_bytes=sum(Path(v['source']).stat().st_size for v in files.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, default=Path('experiments'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-run', required=True)
    parser.add_argument('--geometry-run', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(export_results(args.experiments_root, args.output, args.capture_run, args.geometry_run), indent=2))


if __name__ == '__main__':
    main()

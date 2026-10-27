"""Export explicitly selected capture runs as a standalone static website.

Reads original captures through the existing hash-checking accessor. No model,
simulator, training run or network connection is created.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import parse_qs, urlsplit

from visualization.captures import capture_frame, list_captures, within
from visualization.media import capture_media, video_file


def export_captures(root, run_id, output, *, geometry_runs=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid capture run id')
    if output.exists():
        raise ValueError('export destination must not exist')
    episodes = [e for e in list_captures(root) if e['run_id'] == run_id]
    if not episodes:
        raise ValueError('selected capture run has no episodes')
    output.mkdir(parents=True)
    frontend = Path(__file__).parent
    files = {}

    def record(name, role='result'):
        files[name] = dict(source=str(output / name), role=role)

    def save(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        record(name)

    html = (frontend / 'captures.html').read_text()
    html = html.replace('<html lang="ko">', '<html lang="ko" data-capture-data="./data">')
    html = html.replace('href="/style.css"', 'href="./style.css"').replace('href="/captures.css"', 'href="./captures.css"')
    html = html.replace('src="/captures.js"', 'src="./captures.js"')
    html = html.replace('href="/">← 그래프 목록', 'href="./">← 처음으로')
    html = re.sub(r'<a id="interactive-graph"[^>]*>.*?</a>', '', html)
    (output / 'index.html').write_text(html, encoding='utf-8')
    record('index.html')
    for name in ('style.css', 'captures.css', 'captures.js'):
        shutil.copyfile(frontend / name, output / name)
        record(name)
    save('data/captures.json', dict(episodes=episodes))
    media_count = 0
    copied_provenance = set()
    for episode in episodes:
        for t in episode['timesteps']:
            save(f"data/{episode['id']}/frames/{t}.json", capture_frame(root, episode['id'], t))
        media = capture_media(root, episode['id']) if geometry_runs else dict(runs=[])
        media['runs'] = [r for r in media['runs'] if r['run_id'] in geometry_runs]
        for run in media['runs']:
            if any(v['id'] == 'FPS' for v in run['variants']) and run['run_id'] not in copied_provenance:
                source = within(root, root / run['run_id'] / 'viewer_media/fps_sources.json')
                media_index = json.loads(source.with_name('index.json').read_text())
                if hashlib.sha256(source.read_bytes()).hexdigest() != media_index['fps_sources_sha256']:
                    raise ValueError('FPS provenance hash mismatch')
                save(f"data/{run['run_id']}/fps_sources.json", json.loads(source.read_text()))
                copied_provenance.add(run['run_id'])
            for variant in run['variants']:
                for video in variant['videos'].values():
                    query = parse_qs(urlsplit(video['url']).query)
                    source = video_file(root, *(query[n][0] for n in ('run', 'seed', 'variant', 'kind')))
                    name = f"media/{run['run_id']}/{query['seed'][0]}/{source.name}"
                    target = output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    # Preserve the renderer's recorded video integrity at export.
                    index = json.loads(within(root, root / run['run_id'] / 'viewer_media/index.json').read_text())
                    expected = index['episodes'][query['seed'][0]]['variants'][variant['id']][query['kind'][0]]['sha256']
                    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                        raise ValueError('indexed video hash mismatch')
                    record(name, 'dependency')
                    video['url'] = './' + name
                    media_count += 1
        save(f"data/{episode['id']}/media.json", media)
    readme = '---\ntitle: Task09-C · 수집 데이터 확인\nemoji: 🔎\ncolorFrom: blue\ncolorTo: purple\nsdk: static\napp_file: index.html\npinned: false\n---\n\nSaved diagnostic captures; GT is evaluation-only. No task success claim.\n'
    (output / 'README.md').write_text(readme, encoding='utf-8')
    record('README.md')
    summary = dict(capture_run_id=run_id, geometry_run_ids=list(geometry_runs),
                   episodes=len(episodes), frames=sum(e['frames'] for e in episodes), videos=media_count)
    save('export_manifest.json', summary)
    selection = dict(entrypoint='index.html', description='Task09-C standalone capture viewer', files=files)
    selection_path = output.parent / (output.name + '-selection.json')
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(**summary, output=str(output), selection=str(selection_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, default=Path('experiments'))
    parser.add_argument('--capture-run', required=True)
    parser.add_argument('--geometry-run', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_captures(args.experiments_root, args.capture_run, args.output,
                                   geometry_runs=args.geometry_run), indent=2))


if __name__ == '__main__':
    main()

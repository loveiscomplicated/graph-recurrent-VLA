"""Curated Korean report catalogue and explicit, bounded report dependencies.

Only root-level experiment reports enter the list. Review copies and source
snapshots stay out. Files are read-only; original report bytes are never edited.
"""

from html import escape
from html.parser import HTMLParser
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from utils.experiment_logger import clean


FRONTEND = Path(__file__).parent
CATEGORIES = dict(memory='공간 기억·표현', control='제어·실패 진단',
                  learning='학습·시연 데이터', graph='그래프 연결·속도', sampling='점 선택·표면 기하')
EXCLUDED = {'browser_dom.html', 'browser_image_check.html', 'initial_automated_report.html',
            'grasp_diagnostic.raw.html', 'stop_the_clock_rollout.complete.html', 'visualizations_template.html'}
WEB_TYPES = {'.html', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.mp4', '.webm',
             '.css', '.js', '.json', '.csv', '.md', '.txt'}
DOWNLOAD_LIMIT = 2 * 1024 * 1024


class References(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.tags = []
        self.offsets = [0]
        for line in html.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        links = [(key, value) for key, value in attrs if key in ('src', 'href', 'poster', 'data') and value]
        if links:
            line, column = self.getpos()
            self.tags.append((self.offsets[line-1]+column, self.get_starttag_text(), tag, links))

    handle_startendtag = handle_starttag


def bounded(repo, path):
    path = Path(path).absolute()
    resolved = path.resolve()
    if not resolved.is_relative_to(repo):
        return None
    current = path
    while current != repo and current != current.parent:
        if current.is_symlink():
            return None
        current = current.parent
    return resolved


def local_reference(repo, page, value):
    request = urlsplit(value)
    if request.scheme or request.netloc or not request.path:
        return None
    return bounded(repo, page.parent / unquote(request.path))


def metadata(root, run):
    path = root / run / 'metadata.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def build_catalog(root, repo=None):
    root = Path(root).resolve()
    repo = Path(repo).resolve() if repo else root.parent
    if not root.is_relative_to(repo):
        raise ValueError('experiments root must be inside report repository root')
    curation = json.loads((FRONTEND / 'results_catalog.json').read_text())
    aliases = {alias: name for name, item in curation.items() for alias in item.get('aliases', [])}
    grouped, source_entries = {}, {}
    meta = {}
    for path in sorted(root.glob('*/*.html')):
        if path.name in EXCLUDED or not bounded(repo, path):
            continue
        name = aliases.get(path.name, path.name)
        run = path.parent.name
        content = curation.get(name)
        if content is not None and content.get('match_run', run) != run:
            name = run + '/' + name
            content = None
        if content is None:
            content = dict(task='추가 결과', category='learning', title='추가 실험 결과',
                subtitle='새로 생성된 보고서입니다. 내용 확인 후 한국어 제목과 설명을 보완할 수 있습니다.', notice='제목 정리 대기')
        meta.setdefault(run, metadata(root, run))
        status = meta[run].get('status', 'unknown')
        entry = dict(path=path.relative_to(repo).as_posix(), run_id=run,
                     status=status, available=status != 'running',
                     date=meta[run].get('finished_at') or meta[run].get('started_at') or '',
                     notice=content.get('notice', ''), missing_assets=0)
        if status == 'running':
            entry['notice'] = '진행 중 · 웹 게시 제외'
        for _, _, _, refs in References(path.read_text()).tags:
            for attr, value in refs:
                ref = local_reference(repo, path, value)
                if attr != 'href' and ref is not None and not ref.is_file():
                    entry['missing_assets'] += 1
        if entry['missing_assets']:
            entry['notice'] = '원본 그림 일부 없음'
        grouped.setdefault(name, dict(id=Path(name).stem, **content, versions=[]))['versions'].append(entry)
        source_entries[entry['path']] = entry

    groups = []
    for name, group in grouped.items():
        preferred = group.get('preferred_run')
        group['versions'].sort(key=lambda e: (e['available'], e['run_id'] == preferred,
            e['status'] == 'finished', not e['missing_assets'], e['date'], e['run_id']), reverse=True)
        group['primary'] = group['versions'][0]
        groups.append(group)
    # Task codes are sorting aids, never evidence of a successful experiment.
    def order(group):
        match = re.match(r'(\d+)(?:-([A-Z]+))?', group['task'])
        return (int(match[1]) if match else 6, match[2] or '' if match else '', group['title'])
    groups.sort(key=order, reverse=True)
    resources = {}
    pending = [repo / entry['path'] for entry in source_entries.values() if entry['available']]
    issues = []
    while pending:
        path = pending.pop()
        key = path.relative_to(repo).as_posix()
        if key in resources:
            continue
        if not path.is_file() or path.suffix.lower() not in WEB_TYPES or not bounded(repo, path):
            continue
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            if relative.parts:
                run = relative.parts[0]
                meta.setdefault(run, metadata(root, run))
                if meta[run].get('status') == 'running':
                    continue
        resources[key] = path
        if path.suffix.lower() != '.html':
            continue
        for _, _, tag, refs in References(path.read_text()).tags:
            for attr, value in refs:
                target = local_reference(repo, path, value)
                if target is None:
                    continue
                if (target.is_file() and target.suffix.lower() in WEB_TYPES
                        and not (attr == 'href' and target.suffix.lower() in {'.json', '.csv', '.txt'}
                                 and target.stat().st_size > DOWNLOAD_LIMIT)):
                    pending.append(target)
                else:
                    issues.append(dict(page=key, attribute=attr, tag=tag,
                                       target=target.relative_to(repo).as_posix(),
                                       reason='missing' if not target.is_file() else 'local-only'))
    return dict(groups=groups, categories=CATEGORIES, reports=len(source_entries),
                available_reports=sum(e['available'] for e in source_entries.values()),
                resources=resources, entries=source_entries, issues=issues, repo=repo, root=root)


def public_catalog(catalog, prefix='/result-files/', capture_url='/captures', static=False):
    groups = json.loads(json.dumps(catalog['groups']))
    for group in groups:
        for entry in [group['primary'], *group['versions']]:
            entry['url'] = prefix + quote(entry['path'], safe='/') if entry['available'] else None
    return dict(groups=groups, categories=catalog['categories'], reports=catalog['reports'],
                available_reports=catalog['available_reports'], capture_url=capture_url,
                static=static)


def report_html(catalog, key, home='/results', unavailable='/results-unavailable'):
    """Rewrite only real URL attributes; retain scripts, inline plots and text."""
    path = catalog['resources'][key]
    html = path.read_text()
    for start, original, tag, refs in reversed(References(html).tags):
        replacements = {}
        for attr, value in refs:
            target = local_reference(catalog['repo'], path, value)
            parsed = urlsplit(value)
            if target is None:
                if parsed.scheme == 'file' or (not parsed.scheme and parsed.path and not parsed.netloc):
                    replacements[attr] = unavailable + '?reason=outside'
                continue
            destination = target.relative_to(catalog['repo']).as_posix()
            if destination in catalog['resources']:
                # Keep relative URLs valid both under /result-files and reports/.
                relative = os.path.relpath(target, path.parent).replace(os.sep, '/')
                replacements[attr] = quote(relative, safe='/') + (('?' + parsed.query) if parsed.query else '') + (('#' + parsed.fragment) if parsed.fragment else '')
            elif attr == 'href':
                replacements[attr] = unavailable + '?file=' + quote(destination, safe='') + '&reason=' + ('missing' if not target.is_file() else 'local')
            else:
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="100"><rect width="100%" height="100%" fill="#edf1f6"/><text x="24" y="58" fill="#546278" font-size="20">Original image unavailable</text></svg>'
                replacements[attr] = 'data:image/svg+xml,' + quote(svg, safe='')
        def replace(match):
            attr = match[1].lower()
            return f'{match[1]}="{escape(replacements[attr], quote=True)}"' if attr in replacements else match[0]
        updated = re.sub(r'''\b(src|href|poster|data)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)''', replace, original, flags=re.I)
        html = html[:start] + updated + html[start+len(original):]
    group = next((g for g in catalog['groups'] if any(e['path'] == key for e in g['versions'])), None)
    title = group['title'] if group else '관련 실험 자료'
    nav = f'<nav class="results-report-nav"><a href="{escape(home, quote=True)}">← 결과 목록</a><span>{escape(title)}</span></nav><style>.results-report-nav{{display:flex;gap:24px;align-items:center;background:#101b2b;color:#e7edf7;padding:14px 20px;margin:0 0 24px;font:14px/1.5 system-ui,sans-serif;flex-wrap:wrap}}.results-report-nav a{{color:#a6d5ee;font-weight:700;text-decoration:none}}</style>'
    body = re.search(r'<body\b[^>]*>', html, re.I)
    if body:
        html = html[:body.end()] + nav + html[body.end():]
    else:
        start = re.search(r'<(?:h1|main|pre)\b', html, re.I)
        position = start.start() if start else len(html)
        html = html[:position] + nav + html[position:]
    if not re.search(r'<meta[^>]+name=["\']viewport', html, re.I):
        head = re.search(r'<head\b[^>]*>|<!doctype[^>]*>', html, re.I)
        position = head.end() if head else 0
        html = html[:position] + '<meta name="viewport" content="width=device-width,initial-scale=1">' + html[position:]
    return html


def resource_bytes(catalog, key, **kwargs):
    path = catalog['resources'].get(key)
    if path is None or not bounded(catalog['repo'], path) or not path.is_file():
        raise FileNotFoundError('selected report resource missing')
    suffix = path.suffix.lower()
    if suffix == '.html':
        return report_html(catalog, key, **kwargs).encode(), 'text/html; charset=utf-8'
    if suffix == '.json':
        return json.dumps(clean(json.loads(path.read_text())), ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8'
    return path.read_bytes(), mimetypes.guess_type(path)[0] or 'text/plain; charset=utf-8'

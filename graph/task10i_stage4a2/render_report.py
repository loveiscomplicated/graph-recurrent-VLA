"""Offline static HTML from the full documentation report, with link readback."""
import argparse
import html
import importlib.metadata
import os
from pathlib import Path
import re
from urllib.parse import quote,unquote,urlsplit
from graph.task10i_stage4a.run_pipeline import write,sha


def run(root,document,final=False):
    import markdown
    root.mkdir(parents=True,exist_ok=False)
    document=document.resolve();text=document.read_text();links=[]
    if final:
        assert '전체 작업 미완료' not in text.split('## 감사 상세 및 중간 연구 기록')[0]
        assert 'NOT_TESTED' in text and 'INCONCLUSIVE' in text and 'Stage4B' in text
    def rebase(match):
        label,target=match.groups();parts=urlsplit(target)
        if parts.scheme or target.startswith('#'):return match.group(0)
        path=Path(unquote(parts.path))
        if not path.is_absolute():path=document.parent/path
        path=path.resolve();assert path.exists(),str(path)
        links.append(dict(source_target=target,resolved_path=str(path)))
        relative=quote(os.path.relpath(path,root.resolve()),safe='/._-')
        return label+'('+relative+('#'+parts.fragment if parts.fragment else '')+')'
    rebased=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',rebase,text)
    engine=markdown.Markdown(extensions=['tables','fenced_code','toc'])
    body=engine.convert(rebased)
    style='''body{margin:0;background:#f5f6f8;color:#202632;font:16px/1.65 system-ui,sans-serif}
    nav{position:fixed;inset:0 auto 0 0;width:250px;padding:22px;overflow:auto;font-size:12px;background:#edf0f4}
    nav ul{padding-left:15px}main{margin-left:300px;max-width:1140px;padding:32px 45px;background:white}
    a{color:#1262a6;overflow-wrap:anywhere}h1,h2,h3{line-height:1.3;scroll-margin-top:20px}
    h2{border-top:1px solid #d8dee7;padding-top:24px;margin-top:42px}p,li{overflow-wrap:anywhere}
    table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:20px 0}
    td,th{border:1px solid #d4dae2;padding:9px 12px;min-width:80px}th{background:#edf2f8}
    pre{padding:18px;background:#f1f3f7;overflow:auto;font:13px/1.55 ui-monospace,monospace}
    code{font-size:.9em}img,video{max-width:100%;height:auto}.receipt{font-size:12px;color:#526174}
    @media(max-width:900px){nav{position:static;width:auto;max-height:220px}main{margin:0;padding:24px}}
    @media print{nav{display:none}main{margin:0;padding:0}body{background:white}a{color:inherit}}
    '''
    title='Task10-I Stage4A-2 — full research report'+('' if final else ' — IN PROGRESS')
    output=root/'10i_stage4a2_dino_dinov2_sam2_integration_optimization.html'
    output.write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>'+html.escape(title)+'</title><style>'+style+'</style></head><body><nav><strong>Task10-I Stage4A-2</strong>'
        +engine.toc+'</nav><main><p class="receipt">Full documentation report SHA256: '+sha(document)+'</p>'+body+'</main></body></html>')
    assert document.read_text()==text,'report changed during render'
    readback=output.read_text();assert '<!doctype html>' in readback and '<table>' in readback and '</html>' in readback
    write(root/'verification.json',dict(status='STATIC_HTML_FULL_REPORT_LINK_READBACK_PASS',scientific_completion=final,
        source_sha256=sha(Path(__file__)),report=str(document),report_sha256=sha(document),html=output.name,
        html_sha256=sha(output),markdown_package=importlib.metadata.version('Markdown'),links=links,
        no_external_scripts=True,Mermaid_blocks_preserved_as_source=True,report_not_summarized=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','document'):p.add_argument(name,type=Path)
    p.add_argument('--final',action='store_true');run(**vars(p.parse_args()))

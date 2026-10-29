"""Catalogue curation, read-only report links and resource boundary contracts."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from visualization.results import build_catalog, public_catalog, report_html, resource_bytes
from visualization.server import make_handler


class ResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.root = self.repo / 'experiments'
        self.run = self.root / 'finished'
        (self.run / 'plots').mkdir(parents=True)
        (self.run / 'plots/p.png').write_bytes(b'png image bytes')
        (self.run / 'clip.mp4').write_bytes(b'0123456789')
        (self.run / 'raw.npz').write_bytes(b'original raw arrays')
        (self.run / 'stats.json').write_text('{"HF_TOKEN":"never-publish","score":1}')
        (self.run / 'large.csv').write_bytes(b'x' * (2*1024*1024+1))
        self.html = ('<!doctype html><html><title>Task05-E Raw EDE</title><body><h1>Raw EDE vs Strict</h1>'
                     '<img src="plots/p.png"><img src="plots/missing.png">'
                     '<a href="stats.json">metrics</a><a href="raw.npz">raw data</a>'
                     '<a href="large.csv">large data</a><video src="clip.mp4"></video>'
                     '<script>const literal=\'<a href="never.npz">\';</script></body></html>')
        self.page = self.run / 'raw_ede_comparison.html'
        self.page.write_text(self.html)
        (self.run / 'metadata.json').write_text(json.dumps(dict(status='finished', finished_at='2026-10-25T02:00:00+00:00')))
        (self.run / 'browser_dom.html').write_text('review only')
        nested = self.run / 'source_snapshot'
        nested.mkdir(); (nested / 'raw_ede_comparison.html').write_text('snapshot only')
        running = self.root / 'running'
        running.mkdir(); (running / 'raw_ede_comparison.html').write_text('<h1>unfinished</h1>')
        (running / 'metadata.json').write_text('{"status":"running"}')

    def test_korean_topics_group_versions_and_keep_running_unpublished(self):
        catalog = build_catalog(self.root)
        self.assertEqual((len(catalog['groups']), catalog['reports']), (1, 2))
        group = catalog['groups'][0]
        self.assertEqual(group['title'], '그래프 연결 검증을 줄였을 때 달라지는 점')
        self.assertIn('Raw EDE와 Strict EDE', group['subtitle'])
        self.assertEqual(group['primary']['run_id'], 'finished')
        self.assertFalse(group['versions'][1]['available'])
        data = public_catalog(catalog)
        self.assertIsNone(data['groups'][0]['versions'][1]['url'])
        self.assertNotIn('experiments/running/raw_ede_comparison.html', catalog['resources'])
        self.assertFalse(any('snapshot' in k or 'browser_dom' in k for k in catalog['resources']))

    def test_graph_home_and_results_are_separate_entrypoints(self):
        from visualization.export_results import arrange_navigation
        output = self.repo / 'website'
        output.mkdir()
        (output / 'captures.html').write_text('<h1>Task09-C</h1><div id="capture-videos"></div>'
            '<a href="./index.html">실험 결과 모아보기 →</a>')
        files = {}
        arrange_navigation(output, files)
        capture = (output / 'captures.html').read_text()
        self.assertIn('id="capture-videos"', capture)
        self.assertIn('href="./results.html">실험 결과 모아보기', capture)
        self.assertNotIn('id="result-list"', capture)
        results = (output / 'results.html').read_text()
        self.assertIn('id="result-list"', results)
        self.assertIn('href="./captures.html"', results)
        self.assertEqual(set(files), {'captures.html', 'results.html'})

    def test_html_assets_navigation_redaction_and_source_preservation(self):
        catalog = build_catalog(self.root)
        key = 'experiments/finished/raw_ede_comparison.html'
        html = report_html(catalog, key)
        self.assertIn('← 결과 목록', html)
        self.assertIn('src="plots/p.png"', html)
        self.assertIn('data:image/svg+xml', html)
        self.assertIn('reason=local', html)
        self.assertIn('const literal=\'<a href="never.npz">\';', html)
        self.assertEqual(self.page.read_text(), self.html)
        self.assertNotIn('experiments/finished/raw.npz', catalog['resources'])
        self.assertNotIn('experiments/finished/large.csv', catalog['resources'])
        body, _ = resource_bytes(catalog, 'experiments/finished/stats.json')
        self.assertNotIn(b'never-publish', body)
        self.assertEqual(json.loads(body)['score'], 1)

    def test_symlinks_and_http_traversal_are_not_served(self):
        (self.run / 'plots/p.png').unlink()
        outside = self.repo / 'private.png'; outside.write_bytes(b'private')
        (self.run / 'plots/p.png').symlink_to(outside)
        catalog = build_catalog(self.root)
        self.assertNotIn('experiments/finished/plots/p.png', catalog['resources'])
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/results') as response:
                self.assertIn('실험 결과 모아보기'.encode(), response.read())
            with urlopen(base+'/api/results') as response:
                self.assertEqual(json.load(response)['reports'], 2)
            with urlopen(base+'/result-files/experiments/finished/raw_ede_comparison.html') as response:
                self.assertIn('← 결과 목록'.encode(), response.read())
            with urlopen(Request(base+'/result-files/experiments/finished/clip.mp4', headers={'Range':'bytes=2-5'})) as response:
                self.assertEqual((response.status, response.read()), (206, b'2345'))
            (self.run / 'clip.mp4').unlink()
            (self.run / 'clip.mp4').symlink_to(outside)
            for path in ('../private.png', '%2e%2e/private.png', 'experiments/finished/raw.npz',
                         'experiments/finished/plots/p.png', 'experiments/finished/clip.mp4',
                         'experiments/running/raw_ede_comparison.html'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base+'/result-files/'+path)
                self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); thread.join()

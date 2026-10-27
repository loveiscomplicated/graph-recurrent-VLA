"""Serve the graph rollout frontend and read-only experiment APIs on localhost."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from graph.integration_baseline.graph import EDGE_TYPES, NODE_TYPES
from visualization.data import episode_info, frame_for_step, graph_steps, list_episodes


STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/viewer': ('viewer.html', 'text/html; charset=utf-8'),
          '/list.js': ('list.js', 'text/javascript; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}


def make_handler(experiments_root):
    root = Path(experiments_root).resolve()
    frontend = Path(__file__).parent

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_bytes(status, body, 'application/json; charset=utf-8')

        def episode_path(self, query):
            ids = query.get('id', [])
            if len(ids) != 1 or not ids[0]:
                raise ValueError('one episode id is required')
            episode = (root/ids[0]).resolve()
            if not episode.is_relative_to(root) or not episode.is_dir():
                raise ValueError('episode outside experiments root or missing')
            if not (episode/'observations.npz').is_file():
                raise ValueError('episode observations missing')
            return episode

        def do_GET(self):
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            try:
                if request.path in STATIC:
                    name, content_type = STATIC[request.path]
                    self.send_bytes(HTTPStatus.OK, (frontend/name).read_bytes(), content_type)
                elif request.path == '/api/episodes':
                    self.send_json(HTTPStatus.OK, {'episodes': list_episodes(root)})
                elif request.path == '/api/episode':
                    episode = self.episode_path(query)
                    self.send_json(HTTPStatus.OK, {
                        'episode': episode_info(episode, root),
                        'node_types': NODE_TYPES, 'edge_types': EDGE_TYPES,
                        'timesteps': graph_steps(episode),
                    })
                elif request.path == '/api/frame':
                    episode = self.episode_path(query)
                    values = query.get('t', [])
                    if len(values) != 1 or not values[0].isdigit():
                        raise ValueError('one nonnegative timestep is required')
                    self.send_json(HTTPStatus.OK, frame_for_step(episode, int(values[0])))
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {'error': 'not found'})
            except FileNotFoundError as exc:
                self.send_json(HTTPStatus.NOT_FOUND, {'error': str(exc)})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {'error': str(exc)})

        def log_message(self, format, *args):
            # Keep experiment paths and query strings out of routine console logs.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path,
                        default=Path(__file__).resolve().parents[1]/'experiments')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not args.experiments_root.is_dir():
        parser.error(f'experiments root missing: {args.experiments_root}')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.experiments_root))
    print(f'Graph rollout viewer: http://{args.host}:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

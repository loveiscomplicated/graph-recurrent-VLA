"""Resolve only explicitly indexed viewer videos under the experiments root."""

import json
from pathlib import Path

from visualization.captures import capture_episode, within
from visualization.persistent import VARIANTS, persistent_episode


KINDS = ('graph_rgb', 'graph_only')
MEDIA_VARIANTS = {**VARIANTS, 'FPS': 'point cloud prototype(FPS)'}
FPS_DESCRIPTION = ('초기 Visual v0 추출 방식을 현재 수집 데이터에 재적용했습니다. '
                   'Main 예측 마스크·깊이에서 erosion → SOR → FPS로 물체당 최대 1,024점을 표시합니다. '
                   '프레임별 관측이며 누적·로봇 제거·그래프 엣지·GT 입력은 없습니다. '
                   '유효한 점이 없는 프레임은 빈 점군으로 표시합니다. '
                   'Wrist RGB는 같은 시점의 참고 영상입니다.')


def capture_media(root, capture_id):
    root = Path(root).resolve()
    _, _, info = capture_episode(root, capture_id)
    result = []
    for path in sorted(root.rglob('viewer_media/index.json')):
        index = json.loads(within(root, path).read_text())
        for seed, entry in index['episodes'].items():
            if entry['capture_id'] != info['id']:
                continue
            run = path.parent.parent
            folder, _, geometry = persistent_episode(root, entry['geometry_id'])
            if folder.parent.parent != run or run.name != index['geometry_run_id']:
                raise ValueError('video and geometry run identities disagree')
            variants = []
            for variant, videos in entry['variants'].items():
                if variant not in MEDIA_VARIANTS:
                    raise ValueError('unrecognized indexed video variant')
                item = dict(id=variant, label=MEDIA_VARIANTS[variant], videos={},
                            kind='pointcloud' if variant == 'FPS' else 'graph',
                            description=FPS_DESCRIPTION if variant == 'FPS' else '')
                for kind in KINDS:
                    saved = videos[kind]
                    file = within(root, run / saved['path'])
                    if not file.is_relative_to(path.parent) or not file.is_file():
                        raise ValueError('indexed video missing or outside viewer_media')
                    item['videos'][kind] = dict(
                        url=f'/media?run={run.name}&seed={seed}&variant={variant}&kind={kind}',
                        fps=saved['fps'], frames=saved['frames'])
                variants.append(item)
            result.append(dict(run_id=run.name, revision=index.get('revision'),
                               geometry_id=geometry['id'], variants=variants))
    return dict(runs=result)


def video_file(root, run_id, seed, variant, kind):
    root = Path(root).resolve()
    if (Path(run_id).name != run_id or Path(seed).name != seed or
            variant not in MEDIA_VARIANTS or kind not in KINDS):
        raise ValueError('invalid video identity')
    run = within(root, root / run_id)
    media = within(root, run / 'viewer_media')
    index = json.loads(within(root, media / 'index.json').read_text())
    if index['geometry_run_id'] != run_id:
        raise ValueError('video run identity mismatch')
    try:
        record = index['episodes'][seed]['variants'][variant][kind]
    except KeyError as exc:
        raise FileNotFoundError('indexed video unavailable') from exc
    path = within(root, run / record['path'])
    if not path.is_relative_to(media) or path.suffix != '.mp4' or not path.is_file():
        raise ValueError('video outside indexed media folder or missing')
    return path

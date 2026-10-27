"""CPU readback of a prepared release; no optimizer, simulator, or final test."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest

import cv2
import numpy as np
import torch

from graph.integration_baseline.graph import Graph, pack_graphs
from graph.persistent_geometry.encoder import GeometryRegressor
from graph.persistent_geometry.probes import decode_saved, probe_features
from graph.persistent_geometry.train import load_graph
from .adapter import original, row_join
from .integration_guard import load_candidate, validate_source
from .prepare_release import CHECKOUT, MAIN, LEARNING, sha, write


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('src', 'href') and value)


def verify_links(root):
    checked = 0
    for path in root.rglob('*'):
        if path.suffix not in ('.md', '.html') or not path.is_file():
            continue
        if path.suffix == '.html':
            parser = Links(); parser.feed(path.read_text()); links = parser.links
        else:
            links = re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', path.read_text())
        for link in links:
            if link.startswith(('http:', 'https:', 'data:', 'mailto:', '#')):
                continue
            target = (path.parent / link.split('#')[0].split('?')[0]).resolve()
            if root.resolve() not in target.parents or not target.is_file():
                raise ValueError('broken publication link: ' + str(path) + ': ' + link)
            checked += 1
    return checked


def verify(output, artifact_root):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = output / 'payload'
    report = dict(status='PASS', device='cpu', retraining=False, simulator_executed=False,
                  independent_final_test_executed=False, CUDA_executed=False, MPS_executed=False,
                  historical_protection='PARTIAL', later_revalidation='PASS')
    manifest = CHECKOUT / 'docs/legacy_fps/integration_manifest.json'
    report['source_guard'] = validate_source(output / 'pinned_source', manifest, extension_root=CHECKOUT)
    archive_spec = json.loads(manifest.read_text())
    with tarfile.open(payload / 'pinned_sources.tar.gz') as tar:
        for group, prefix in [('required_dependency_source_hashes', 'pinned_source'),
                              ('historical_extension_source_hashes', 'historical_extension'),
                              ('extension_source_hashes', 'implementation')]:
            for relative, expected in archive_spec[group].items():
                import hashlib
                assert hashlib.sha256(tar.extractfile(prefix + '/' + relative).read()).hexdigest() == expected
    report['source_archive_hashes'] = 'PASS'
    assert load_candidate('/missing', '/missing') is None
    assert load_candidate(output / 'pinned_source', manifest, enabled=True,
                          extension_root=CHECKOUT)['system'] is not None
    with tempfile.TemporaryDirectory() as folder:
        stale = json.loads(manifest.read_text())
        stale['required_dependency_source_hashes']['graph/persistent_geometry/encoder.py'] = '0' * 64
        negative = Path(folder) / 'stale.json'; write(negative, stale)
        try:
            validate_source(output / 'pinned_source', negative, extension_root=CHECKOUT)
        except RuntimeError:
            report['stale_hash_negative_control'] = 'PASS'
        else:
            raise AssertionError('stale encoder hash accepted')
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ('legacy_fps.tests', 'graph.persistent_geometry.tests.test_contracts', 'legacy_fps.test_release'))
    with (payload / 'release/contract_tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert result.testsRun == 58
    report['contract_tests'] = dict(historical=54, release=4, executed=result.testsRun,
                                    failures=len(result.failures), errors=len(result.errors))

    learning = artifact_root / 'learning_runs' / LEARNING
    models = json.loads((learning / 'model_manifest.json').read_text())['models']
    frames = json.loads((learning / 'split_manifest.json').read_text())['frames']
    graph_count = permutations = 0
    with torch.inference_mode():
        for row in models:
            checkpoint_path = output / 'models/learning_runs' / LEARNING / row['directory'] / 'selected.pt'
            assert sha(checkpoint_path) == row['checkpoint_sha256']
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = GeometryRegressor(row['encoder']).eval().requires_grad_(False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            graphs = [load_graph(artifact_root / frame['folder'].split('/workspace/task09c_legacy_fps/', 1)[1] /
                                 (row['variant'] + '_graph.npz')) for frame in frames]
            readouts = []; predictions = []
            for start in range(0, len(graphs), 16):
                readout, _ = model.encoder(pack_graphs(graphs[start:start + 16]))
                readouts.append(readout.numpy())
                predictions.append(model.decoder(readout).numpy() * checkpoint['target_std'] + checkpoint['target_mean'])
            with np.load(learning / row['directory'] / 'frozen_features.npz', allow_pickle=False) as data:
                np.testing.assert_allclose(np.concatenate(readouts), data['readout'], atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(np.concatenate(predictions), data['prediction36'], atol=2e-6, rtol=2e-5)
            graph_count += len(graphs)
            for index in (0, len(graphs) // 2, len(graphs) - 1):
                graph = graphs[index]
                order = torch.randperm(len(graph.node_type), generator=torch.Generator().manual_seed(43 + index))
                inverse = torch.empty_like(order); inverse[order] = torch.arange(len(order))
                changed = Graph(graph.node_input[order], graph.node_type[order], inverse[graph.edge_index],
                                graph.edge_attr, graph.edge_type, {}, {})
                a, _ = model.encoder(pack_graphs([graph])); b, _ = model.encoder(pack_graphs([changed]))
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                permutations += 1
    report['encoder_checkpoints'] = dict(strict_loaded=len(models), graph_readouts=graph_count,
                                        permutation_cases=permutations, atol=2e-6, rtol=2e-5)
    probes = json.loads((learning / 'probe_manifest.json').read_text())['probes']
    for row in probes:
        path = output / 'models/learning_runs' / LEARNING / row['directory'] / 'decoder.npz'
        assert sha(path) == row['decoder_sha256']
        # Probe output readback uses recorded diagnostic features, never a new fit.
        feature_row = next(m for m in models if m['variant'] == row['variant'] and
                           (row['encoder'] == 'input_only' or m['encoder'] == row['encoder']))
        with np.load(learning / feature_row['directory'] / 'frozen_features.npz', allow_pickle=False) as data:
            features = probe_features(data, row['probe'], row['object_id'])
            prediction = decode_saved(path, features)
        with np.load(learning / row['directory'] / 'predictions.npz', allow_pickle=False) as expected:
            np.testing.assert_allclose(prediction, expected['prediction'], atol=2e-6, rtol=2e-5)
    report['probe_checkpoints'] = dict(hash_verified_and_decoded=len(probes), atol=2e-6, rtol=2e-5)

    # Saved real-frame points: exact source-pixel identity and original transform.
    folder = payload / 'task09c_legacy_fps/experiments' / MAIN / 'geometry/seed30002/f0000'
    with np.load(folder / 'G0_original_packet.npz', allow_pickle=False) as packet:
        transform = packet['packet/camera_to_world']; intrinsic = packet['packet/camera_intrinsic']
        width = int(packet['packet/image_shape'][1])
    points_checked = 0; max_world_error = 0.
    for path in folder.glob('fps_base_camera*.npz'):
        with np.load(path, allow_pickle=False) as data:
            xyz = data['point_xyz_camera']; pixels = data['source_pixel_id']
            np.testing.assert_array_equal(row_join(xyz, xyz), np.arange(len(xyz)))
            world = original.to_world(xyz, transform)
            np.testing.assert_allclose(world, data['point_xyz_world'], atol=1e-7, rtol=1e-6)
            if len(world):
                max_world_error = max(max_world_error, float(np.abs(world - data['point_xyz_world']).max()))
            v, u = np.divmod(pixels, width); z = xyz[:, 2]
            back = np.c_[(u + .5 - intrinsic[0, 2]) * z / intrinsic[0, 0],
                         (v + .5 - intrinsic[1, 2]) * z / intrinsic[1, 1], z].astype('float32')
            np.testing.assert_array_equal(back, xyz)
            assert np.isin(pixels, data['sor_source']).all()
            points_checked += len(xyz)
    report['real_frame_XYZ_source_pixel_exact_rows'] = points_checked
    report['cross_host_world_transform'] = dict(max_abs_m=max_world_error, atol=1e-7, rtol=1e-6,
                                               bitwise_identity_claimed=False)
    report['raw_RGBD_recapture'] = 'NOT_EXECUTED; captures excluded; synthetic original-helper contracts also executed'
    png = mp4 = video_frames = 0
    for path in payload.rglob('*'):
        if path.suffix == '.png':
            assert cv2.imread(str(path)) is not None; png += 1
        elif path.suffix == '.mp4':
            cap = cv2.VideoCapture(str(path)); count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert frame is not None; count += 1
            cap.release(); assert count > 0
            video_frames += count; mp4 += 1
    report['decode'] = dict(png=png, mp4=mp4, decoded_video_frames=video_frames)
    report['document_links'] = verify_links(payload)
    inventory = json.loads((output / 'original_inventory.json').read_text())
    for row in inventory:
        assert sha(row['source']) == row['source_sha256']
    report['original_files_unchanged'] = len(inventory)
    write(payload / 'release/verification.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    verify(args.package_root.resolve(), args.artifact_root.resolve())


if __name__ == '__main__':
    main()

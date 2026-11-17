"""Reuse verified family archives and graph caches without changing node identity."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data_collection.episode_dataset import file_sha256
from graph.build_graph.corner_confidence import compute_confidence
from training.gnn.data import canonical_hash, load_cached_sample


def checked_path(root, relative):
    """Resolve a cache member without allowing it to escape its dataset root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Cache member escapes dataset root: {relative}")
    return path


class FamilyArchiveCache:
    """Index completed or recovered manifests whose label-generating sources match."""

    def __init__(self, entries, source_identity, detector, certification):
        self.rows = {}
        for entry in entries:
            root = Path(entry['root']).resolve()
            manifest_path = checked_path(root, entry.get('manifest', 'manifest.json'))
            manifest = json.loads(manifest_path.read_text())
            fingerprint = dict(manifest)
            expected = fingerprint.pop('dataset_sha256')
            if canonical_hash(fingerprint) != expected:
                raise ValueError(f'Family manifest hash differs: {manifest_path}')
            if manifest['source_dataset_sha256'] != source_identity['dataset_sha256']:
                raise ValueError(f"Family cache uses a different source dataset: {root}")
            if canonical_hash(manifest['certification']) != canonical_hash(asdict(certification)):
                raise ValueError(f"Family cache certification settings differ: {root}")
            if canonical_hash(manifest['detector']['parameters']) != canonical_hash(asdict(detector.parameters)):
                raise ValueError(f"Family cache detector settings differ: {root}")
            required_sources = [Path('data_collection/vertex_family_labels.py'),
                                *Path('visual/corner_detection').rglob('*.py'),
                                *Path('graph/build_graph').glob('*.py')]
            for path in required_sources:
                if manifest['sources'].get(str(path)) != file_sha256(path):
                    raise ValueError(f"Family cache source differs: {root}: {path}")
            for row in manifest['samples']:
                self.rows.setdefault(row['sample_id'], (root, row))

    def find(self, row):
        """Verify hashes and node ordering before returning a reusable archive."""
        candidate = self.rows.get(row['sample_id'])
        if candidate is None:
            return None
        root, old = candidate
        if any(row[name] != old[name] for name in ('sample_id', 'parent_id', 'source_sha256')):
            raise ValueError(f"Family cache sample identity differs: {row['sample_id']}")
        graph = checked_path(root, old['graph_path'])
        labels = checked_path(root, old['labels_path'])
        if file_sha256(graph) != old['graph_sha256'] or file_sha256(labels) != old['labels_sha256']:
            raise ValueError(f"Family cache content changed: {row['sample_id']}")
        sample = load_cached_sample(graph, old['sample_identity'], row['sample_id'])
        with np.load(labels, allow_pickle=False) as archive:
            count = int(sample.graph.node_valid.sum())
            np.testing.assert_array_equal(archive['xyz_world'], sample.graph.xyz_world[0, :count].numpy())
            np.testing.assert_array_equal(archive['object_type'], sample.graph.object_type[0, :count].numpy())
            return dict(graph=graph, labels=labels, object_ids=archive['object_type'].copy(),
                        detector_labels=archive['detector_family'].copy(),
                        oracle_labels=archive['oracle_family'].copy(),
                        reasons=json.loads(str(archive['reason_json'])))


class ExistingGraphCache:
    """Reuse graph topology only when newly detected inputs exactly match its inputs."""

    def __init__(self, root, source_identity, builder):
        self.builder = builder
        self.rows = {}
        if root is None:
            return
        self.root = Path(root).resolve()
        metadata = self.root / 'metadata'
        identity = json.loads((metadata / 'data_identity.json').read_text())
        config = yaml.safe_load((metadata / 'config.yaml').read_text())
        if identity['dataset_sha256'] != source_identity['dataset_sha256']:
            raise ValueError('Existing graph cache source dataset differs')
        if config['graph'] != asdict(builder.settings) or config['confidence'] != asdict(builder.confidence_settings):
            raise ValueError('Existing graph cache graph/confidence settings differ')
        inventory = json.loads((metadata / 'cache_inventory.json').read_text())
        self.preprocessing = inventory['preprocessing_sha256']
        self.rows = {row['sample_id']: row for row in inventory['files']}

    def find(self, row, result, tcp, target):
        """Return None on a detector mismatch so no old/new node labels get mixed."""
        entry = self.rows.get(row['sample_id'])
        if entry is None:
            return None
        path = checked_path(self.root, entry['file'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f"Existing graph cache hash mismatch: {path}")
        identity = dict(frame_sha256=row['source_sha256'], sample_id=row['sample_id'],
                        preprocessing_sha256=self.preprocessing)
        sample = load_cached_sample(path, identity, row['sample_id'])
        graph = sample.graph
        count = len(result.xyz)
        confidence, confidence_valid = compute_confidence(result.metadata, self.builder.confidence_settings)
        pairs = [(graph.xyz_world[0, :count], result.xyz.cpu()),
                 (graph.object_type[0, :count], result.object_id.cpu()),
                 (graph.tcp_xyz_world, tcp), (sample.targets['object21'], target),
                 (graph.confidence[0, :count], confidence),
                 (graph.confidence_valid[0, :count], confidence_valid)]
        if int(graph.node_valid.sum()) != count or any(not torch.equal(left, right) for left, right in pairs):
            return None
        if graph.semantic_valid.any():
            return None
        return sample

"""Prepare frozen GNN node features aligned to the existing insertion data."""

from graph.temporal.node_features import capture_pre_pool_features, ordered_node_features
from training.gnn.temporal.frozen_source import infer_insertion_data


def infer_node_data(frozen, loader, device, identity, effective, split):
    """Capture frozen tokens and Object21 in the same validated forward pass."""
    captured = {}
    current = {}

    class RecordingLoader:
        """Expose the source batch to the hook while retaining loader order."""

        def __iter__(self):
            for batch in loader:
                current['batch'] = batch
                yield batch

    def record_features(features):
        batch = current['batch']
        frames = ordered_node_features(batch.graph, features.detach())
        for sample_id, values in zip(batch.sample_ids, frames):
            if sample_id in captured:
                raise ValueError(f'Duplicate graph frame: {sample_id}')
            captured[sample_id] = values.cpu().numpy().copy()

    with capture_pre_pool_features(frozen, record_features):
        data = infer_insertion_data(frozen, RecordingLoader(), device, identity, effective, split)
    if set(captured) != set(data['sample_ids']):
        raise ValueError('Node tokens and insertion frames do not match')
    data['node_features'] = [captured[str(sample_id)] for sample_id in data['sample_ids']]
    return data

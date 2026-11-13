"""Connect independent fixed-ID corner predictions to the Object21 GNN."""

import torch
from torch import nn

from graph.build_graph import EdgeType, GraphBatch, NodeType


def build_corner_graph(xyz, object_types, observed, tcp_xyz):
    """Build one complete graph; node_valid denotes support, not visibility."""
    count = xyz.shape[0]
    slots = max(count, 1)
    if count:
        coordinates = xyz.unsqueeze(0)
        kinds = object_types.unsqueeze(0)
        valid = torch.ones((1, count), dtype=torch.bool, device=xyz.device)
    else:
        coordinates = xyz.new_zeros((1, 1, 3))
        kinds = torch.full((1, 1), NodeType.PAD, dtype=torch.long, device=xyz.device)
        valid = torch.zeros((1, 1), dtype=torch.bool, device=xyz.device)

    indices = torch.arange(count, device=xyz.device)
    sender, receiver = torch.meshgrid(indices, indices, indexing="ij")
    keep = sender != receiver
    edges = torch.stack((sender[keep], receiver[keep]))
    source, target = edges
    flat_types = kinds.reshape(-1)
    edge_types = torch.where(flat_types[source] == flat_types[target],
                             flat_types[source], EdgeType.CROSS)
    relative = coordinates[0, source] - coordinates[0, target]
    graph = GraphBatch(
        xyz_world=coordinates, object_type=kinds, node_valid=valid,
        semantic_id=torch.full((1, slots), -1, dtype=torch.long, device=xyz.device),
        semantic_valid=torch.zeros_like(valid),
        confidence=xyz.new_zeros((1, slots)), confidence_valid=torch.zeros_like(valid),
        tcp_xyz_world=tcp_xyz.unsqueeze(0), edge_index=edges, edge_type=edge_types,
        relative_xyz=relative, distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    graph.validate()
    source_feature = xyz.new_zeros((1, slots, 1))
    if count:
        source_feature[0, :, 0] = observed.to(xyz.dtype)
    return graph, source_feature


class FixedIdCornerGNN(nn.Module):
    """Predict missing corners, then produce Object21 for one synchronized frame.

    The caller supplies established identities and per-corner causal histories;
    this module neither associates detections nor maintains tracks. The temporal
    model is shared across all corners. GNN extra_node_dim must be one, carrying
    1 for a current detection and 0 for a predicted point.
    """

    def __init__(self, temporal_model, gnn):
        super().__init__()
        if gnn.extra_node_dim != 1:
            raise ValueError("GNN must use extra_node_dim=1 for detection provenance")
        self.temporal_model = temporal_model
        self.gnn = gnn

    def forward(self, inputs, context_valid, seen, observed, detection_xyz,
                last_observed_xyz, object_types, tcp_xyz):
        """Return Object21 and the exact graph inputs, with no GT in the forward path.

        inputs is [N,T,13] with right-padded histories ending at the current
        frame. Other node tensors are [N] or [N,3]; tcp_xyz is [3]. Only seen
        IDs are used. Current observations override predictions exactly.
        """
        if inputs.ndim != 3 or inputs.shape[-1] != 13 or inputs.shape[1] < 1:
            raise ValueError("inputs must have shape [N,T,13] with T >= 1")
        count, steps, _ = inputs.shape
        if context_valid.shape != (count, steps) or context_valid.dtype != torch.bool:
            raise ValueError("context_valid must be boolean [N,T]")
        for name, mask in (("seen", seen), ("observed", observed)):
            if mask.shape != (count,) or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean [N]")
        if torch.any(observed & ~seen):
            raise ValueError("observed corners must already be marked seen")
        for name, value in (("detection_xyz", detection_xyz),
                            ("last_observed_xyz", last_observed_xyz)):
            if value.shape != (count, 3) or value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32 [N,3]")
        if tcp_xyz.shape != (3,) or tcp_xyz.dtype != torch.float32:
            raise ValueError("tcp_xyz must be float32 [3]")
        if object_types.shape != (count,) or object_types.dtype != torch.long:
            raise ValueError("object_types must be int64 [N]")
        if any(value.device != inputs.device for value in
               (context_valid, seen, observed, detection_xyz, last_observed_xyz, object_types, tcp_xyz)):
            raise ValueError("all inputs must be on the same device")
        lengths = context_valid.sum(dim=1)
        expected_mask = torch.arange(steps, device=inputs.device)[None, :] < lengths[:, None]
        if not torch.equal(context_valid, expected_mask):
            raise ValueError("context_valid must represent right-padded histories")
        missing = seen & ~observed
        if torch.any(missing & (lengths == 0)):
            raise ValueError("missing corners require a nonempty causal history")
        if torch.any(seen & ((object_types < NodeType.PEG) | (object_types > NodeType.HOLE))):
            raise ValueError("seen corners require PEG, BOX or HOLE object types")
        if not torch.isfinite(inputs[missing]).all() or not torch.isfinite(last_observed_xyz[missing]).all():
            raise ValueError("missing-corner histories and last observations must be finite")
        if not torch.isfinite(detection_xyz[observed]).all() or not torch.isfinite(tcp_xyz).all():
            raise ValueError("current detections and TCP must be finite")

        selected = torch.nonzero(seen, as_tuple=True)[0]
        selected_observed = observed[selected]
        positions = detection_xyz[selected].clone()
        if missing.any():
            missing_indices = torch.nonzero(missing, as_tuple=True)[0]
            delta = self.temporal_model(inputs[missing_indices], context_valid[missing_indices])
            final_delta = delta[torch.arange(len(missing_indices), device=inputs.device),
                                lengths[missing_indices] - 1]
            positions[~selected_observed] = last_observed_xyz[missing_indices] + final_delta
        graph, provenance = build_corner_graph(positions, object_types[selected],
                                               selected_observed, tcp_xyz)
        prediction = self.gnn(graph, provenance)
        return {"object21": prediction["object21"], "graph": graph,
                "observed": selected_observed, "selected_indices": selected,
                "gnn_output": prediction}

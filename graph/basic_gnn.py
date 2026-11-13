import torch
from torch import nn
from torch.nn import functional as F

from graph.build_graph import GraphBatch


class BasicGNNLayer(nn.Module):
    def __init__(self, node_in_dim, node_out_dim, edge_dim):
        super().__init__()
        # 메시지 생성을 위한 선형 변환
        self.msg_layer = nn.Sequential(
            nn.Linear(node_in_dim + edge_dim, node_out_dim),
            nn.ReLU(),
            nn.Linear(node_out_dim, node_out_dim),
        )

        # 상태 업데이트를 위한 선형 변환
        self.update_layer = nn.Linear(node_in_dim + node_out_dim, node_out_dim)
        self.act = nn.ReLU()

    def forward(self, x, adj, edge_attr, node_valid=None):
        # x:         [batch, nodes, node_in_dim]
        # adj:       [batch, 수신 노드, 송신 노드]
        # edge_attr: [batch, 수신 노드, 송신 노드, edge_dim]

        if node_valid is not None:
            # Operate only on real edges and valid nodes, despite dense storage.
            samples, receivers, senders = adj.nonzero(as_tuple=True)
            message_inputs = torch.cat(
                (x[samples, senders], edge_attr[samples, receivers, senders]),
                dim=-1,
            )
            messages = self.msg_layer(message_inputs)
            aggregated = x.new_zeros(x.shape[0] * x.shape[1], messages.shape[-1])
            aggregated.index_add_(0, samples * x.shape[1] + receivers, messages)
            aggregated = aggregated.reshape(x.shape[0], x.shape[1], -1)
            result = torch.zeros_like(aggregated)
            combined = torch.cat((x[node_valid], aggregated[node_valid]), dim=-1)
            result[node_valid] = self.act(self.update_layer(combined))
            return result

        num_nodes = x.size(1)

        # 각 수신 노드에 대해 송신 노드의 특성을 준비
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        # 송신 노드와 해당 엣지의 특성을 함께 전달
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_layer(msg_input)

        # 연결된 이웃의 메시지를 합산
        aggr = (msg * adj.unsqueeze(-1)).sum(dim=2)

        combined = torch.cat([x, aggr], dim=-1)
        return self.act(self.update_layer(combined))


class BasicGNN(nn.Module):
    def __init__(
        self, n_layers, in_dim, h_dim, out_dim, edge_dim, dropout=0.1,
        use_confidence=False,
    ):
        super().__init__()

        if type(use_confidence) is not bool:
            raise ValueError("use_confidence must be bool")
        self.use_confidence = use_confidence
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer_in = in_dim if i == 0 else h_dim
            layer_out = out_dim if i == (self.n_layers - 1) else h_dim

            self.layers.append(BasicGNNLayer(layer_in, layer_out, edge_dim))

        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_layers - 1)])

    def _dense_graph_inputs(self, graph: GraphBatch):
        """Adapt shared geometry to dense receiver/sender tensors.

        Node width 8: TCP-relative xyz, type one-hot, quality, quality-valid.
        Edge width 8: sender-receiver xyz, distance, relation one-hot.
        Semantic IDs remain available in GraphBatch but this baseline omits them.
        """
        graph.validate()
        if self.in_dim != 8 or self.edge_dim != 8:
            raise ValueError("BasicGNN GraphBatch input requires in_dim=8 and edge_dim=8")
        batch_size, node_count, _ = graph.xyz_world.shape
        node_type = F.one_hot(graph.object_type.clamp_min(0), 3).to(graph.xyz_world.dtype)
        quality = torch.zeros_like(graph.confidence)
        quality_valid = torch.zeros_like(graph.confidence_valid)
        if self.use_confidence:
            quality = graph.confidence
            quality_valid = graph.confidence_valid
        features = torch.cat(
            (
                graph.xyz_world - graph.tcp_xyz_world[:, None, :],
                node_type,
                quality.unsqueeze(-1),
                quality_valid.unsqueeze(-1).to(graph.xyz_world.dtype),
            ),
            dim=-1,
        )
        features = features.masked_fill(~graph.node_valid.unsqueeze(-1), 0)
        adjacency = graph.xyz_world.new_zeros(batch_size, node_count, node_count)
        edge_features = graph.xyz_world.new_zeros(batch_size, node_count, node_count, 8)
        src, dst = graph.edge_index
        samples = src // node_count
        senders, receivers = src % node_count, dst % node_count
        adjacency[samples, receivers, senders] = 1
        edge_features[samples, receivers, senders] = torch.cat(
            (
                graph.relative_xyz, graph.distance,
                F.one_hot(graph.edge_type, 4).to(graph.xyz_world.dtype),
            ),
            dim=-1,
        )
        return features, adjacency, edge_features

    def forward(self, x, adj=None, edge_attr=None):
        """Accept GraphBatch, retaining the existing explicit dense-tensor call."""
        node_valid = None
        if isinstance(x, GraphBatch):
            if adj is not None or edge_attr is not None:
                raise ValueError("Pass GraphBatch alone, without adj or edge_attr")
            node_valid = x.node_valid
            x, adj, edge_attr = self._dense_graph_inputs(x)
        elif adj is None or edge_attr is None:
            raise ValueError("Dense input requires x, adj and edge_attr")
        for i, layer in enumerate(self.layers):
            x = layer(x, adj, edge_attr, node_valid=node_valid)
            if i < (self.n_layers - 1):
                if node_valid is None:
                    x = self.dropout(self.norms[i](x))
                else:
                    x = x.clone()
                    x[node_valid] = self.dropout(self.norms[i](x[node_valid]))
            if node_valid is not None:
                x = x.masked_fill(~node_valid.unsqueeze(-1), 0)
        return x


# --- 사용 예시 ---
if __name__ == "__main__":
    torch.manual_seed(0)  # 모델의 초기 가중치를 재현
    torch.set_printoptions(precision=4, sci_mode=False)

    # 노드 3개, 노드 특성 2개, 엣지 특성 1개
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    # adj[0, i, j] = 1이면 j -> i 메시지 전달
    adj = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    edge_attr = torch.tensor(
        [[[[0.0], [0.5], [0.0]], [[0.5], [0.0], [1.5]], [[0.0], [1.5], [0.0]]]]
    )

    layer = BasicGNNLayer(node_in_dim=2, node_out_dim=2, edge_dim=1)
    model = BasicGNN(n_layers=2, in_dim=2, h_dim=4, out_dim=2, edge_dim=1, dropout=0.1)
    model.eval()  # 출력 확인 시 dropout 비활성화

    with torch.no_grad():
        layer_output = layer(x, adj, edge_attr)
        model_output = model(x, adj, edge_attr)

    print("노드 특성 x:\n", x)
    print("인접 행렬 adj (행: 수신, 열: 송신):\n", adj)
    print("엣지 특성 edge_attr:\n", edge_attr)
    print("BasicGNNLayer 출력:\n", layer_output)
    print("BasicGNN 출력:\n", model_output)

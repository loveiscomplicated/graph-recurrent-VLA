"""Statistics from actual directed message-passing tensors."""
import torch


def edge_rows(graph, selected=None):
    """Exact directed edges for plotting/export, including both directions."""
    selected = set(range(graph.edge_attr.shape[1]-4)) if selected is None else set(selected)
    return [(int(a), int(b), int(t)) for (a, b), t in zip(graph.edge_index.T, graph.edge_type) if int(t) in selected]


def diagnose(graph, names):
    b, n = graph.x.shape[:2]
    src, dst = graph.edge_index
    assert bool((src//n == dst//n).all()), 'cross-batch edge'
    inc = torch.bincount(dst, minlength=b*n).reshape(b, n)
    out = torch.bincount(src, minlength=b*n).reshape(b, n)
    statistics = {}
    for t, name in enumerate(names):
        values = graph.edge_attr[graph.edge_type == t, 3]
        statistics[name] = dict(count=len(values), mean_m=float(values.mean()) if len(values) else None,
                                min_m=float(values.min()) if len(values) else None, max_m=float(values.max()) if len(values) else None)
    components = []
    for batch in range(b):
        adjacency = [set() for _ in range(n)]
        for a, z, _ in edge_rows(graph):
            if a//n == batch:
                adjacency[a % n].add(z % n)
                adjacency[z % n].add(a % n)
        remaining = set(range(n)); sizes = []
        while remaining:
            todo = [remaining.pop()]; size = 0
            while todo:
                current = todo.pop(); size += 1
                fresh = adjacency[current] & remaining
                remaining -= fresh; todo.extend(fresh)
            sizes.append(size)
        components.append(sizes)
    cross = graph.edge_type == 3
    cross_inc = torch.bincount(dst[cross], minlength=b*n).reshape(b,n)
    hole = graph.node_type == 4
    unique = torch.unique(torch.stack((src, dst, graph.edge_type)), dim=1).shape[1]
    node_degrees = {str(int(t)):dict(incoming=int(inc[:, graph.node_type == t].sum()),
                                   outgoing=int(out[:, graph.node_type == t].sum())) for t in graph.node_type.unique()}
    return dict(**graph.stats, edges=statistics, incoming_degree=inc.tolist(), outgoing_degree=out.tolist(),
                degree_by_node_type=node_degrees, components=components,
                duplicates=int(len(src)-unique), self_loops=int((src == dst).sum()),
                cross_over_10cm=int((graph.edge_attr[cross, 3] > .1).sum()),
                hole_cross_fraction_max=float((cross_inc[:,hole]/inc[:,hole].clamp_min(1)).max()))

"""Load TU datasets with a lightweight fallback graph representation."""

from __future__ import annotations

import torch

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimpleGraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor | None = None

    @property
    def num_nodes(self):
        return int(self.x.numel())

def _read_int_lines(path):
    vals = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                vals.append(int(line))
    return vals


def _read_edge_lines(path):
    edges = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            edges.append((int(a), int(b)))
    return edges


def load_tu_graph_dataset(root, name):
    """
    Load a small TU-format graph dataset from text files.

    Supports datasets with:
      NAME_A.txt
      NAME_graph_indicator.txt
      NAME_node_labels.txt
      optional NAME_edge_labels.txt
      optional NAME_graph_labels.txt

    Node and edge labels are remapped only by WL later; here we preserve their
    integer identities. Edges stay directed exactly as listed because the
    bundled TU files usually include both directions for undirected graphs.
    """
    name = str(name)
    root = Path(root)
    graph_dir = root / name
    if not graph_dir.exists():
        raise FileNotFoundError(f"Missing TU dataset directory: {graph_dir}")

    graph_indicator = _read_int_lines(graph_dir / f"{name}_graph_indicator.txt")
    node_labels = _read_int_lines(graph_dir / f"{name}_node_labels.txt")
    graph_labels_path = graph_dir / f"{name}_graph_labels.txt"
    graph_labels = _read_int_lines(graph_labels_path) if graph_labels_path.exists() else None

    edges_global = _read_edge_lines(graph_dir / f"{name}_A.txt")
    edge_labels_path = graph_dir / f"{name}_edge_labels.txt"
    if edge_labels_path.exists():
        edge_labels = _read_int_lines(edge_labels_path)
    else:
        edge_labels = [1] * len(edges_global)
    if len(edge_labels) != len(edges_global):
        raise ValueError(f"Edge-label count mismatch for {name}")

    graph_to_nodes = {}
    for global_idx, graph_id in enumerate(graph_indicator, start=1):
        graph_to_nodes.setdefault(int(graph_id), []).append(global_idx)

    graph_to_edges = {gid: [] for gid in graph_to_nodes}
    for (u, v), edge_label in zip(edges_global, edge_labels):
        gid = int(graph_indicator[u - 1])
        if gid != int(graph_indicator[v - 1]):
            continue
        graph_to_edges.setdefault(gid, []).append((u, v, int(edge_label)))

    graphs = []
    for graph_id in sorted(graph_to_nodes):
        nodes = graph_to_nodes[graph_id]
        local = {global_node: i for i, global_node in enumerate(nodes)}
        x = torch.tensor([int(node_labels[g - 1]) for g in nodes], dtype=torch.long)

        local_edges = []
        local_edge_labels = []
        for u, v, edge_label in graph_to_edges.get(graph_id, []):
            if u == v:
                continue
            local_edges.append((local[u], local[v]))
            local_edge_labels.append(edge_label)
        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(local_edge_labels, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)

        y = None
        if graph_labels is not None:
            y = torch.tensor([int(graph_labels[graph_id - 1])], dtype=torch.long)
        graphs.append(SimpleGraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


def split_graph_dataset(graphs, train_frac=0.8, val_frac=0.1, seed=0):
    n = len(graphs)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    if n == 0:
        return [], [], []
    if n == 1:
        n_train, n_val = 1, 0
    elif n == 2:
        n_train, n_val = 1, 1
    else:
        n_train = min(max(1, int(round(train_frac * n))), n - 2)
        n_val = min(max(1, int(round(val_frac * n))), n - n_train - 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return (
        [graphs[i] for i in train_idx],
        [graphs[i] for i in val_idx],
        [graphs[i] for i in test_idx],
    )

"""WL refinement, graph statistics, and generation post-processing helpers."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import torch

from warning_filters import configure_warning_filters


configure_warning_filters()

try:
    from torch_geometric.utils import degree as pyg_degree
except Exception:
    pyg_degree = None


def degree(index, num_nodes, dtype=torch.long):
    if pyg_degree is not None:
        return pyg_degree(index, num_nodes=num_nodes, dtype=dtype)

    result = torch.zeros(num_nodes, dtype=dtype, device=index.device)
    ones = torch.ones(index.size(0), dtype=dtype, device=index.device)
    result.scatter_add_(0, index, ones)
    return result


class WeisfeilerLeman:
    def __init__(self, n_iter=3):
        self.n_iter = int(n_iter)

    def _get_label(self, signature, iteration):
        if signature not in self.label_dict[iteration]:
            self.label_dict[iteration][signature] = self.label_counter[iteration]
            self.label_counter[iteration] += 1
        return self.label_dict[iteration][signature]

    def _lookup_label(self, signature, iteration):
        return self.label_dict[iteration].get(signature, 0)

    def _compute_graph_colors(self, data, *, grow):
        num_nodes = int(data.num_nodes)
        edge_index = data.edge_index
        result = torch.zeros(num_nodes, self.n_iter + 1, dtype=torch.long)

        if data.x is not None and data.x.dim() == 1:
            initial_colors = data.x.tolist()
        else:
            initial_colors = degree(
                edge_index[0],
                num_nodes=num_nodes,
                dtype=torch.long,
            ).tolist()

        label_fn = self._get_label if grow else self._lookup_label
        colors = [label_fn(color, 0) for color in initial_colors]
        result[:, 0] = torch.tensor(colors, dtype=torch.long)

        adjacency = defaultdict(list)
        has_edge_attr = data.edge_attr is not None
        for edge_pos in range(edge_index.size(1)):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            edge_attr = data.edge_attr[edge_pos].item() if has_edge_attr else None
            adjacency[source].append((target, edge_attr))

        for iteration in range(self.n_iter):
            new_colors = [0] * num_nodes
            for node_id in range(num_nodes):
                neighbors = tuple(sorted(
                    (colors[target], edge_attr)
                    for target, edge_attr in adjacency[node_id]
                ))
                signature = (colors[node_id], neighbors)
                color = label_fn(signature, iteration + 1)
                new_colors[node_id] = color
                result[node_id, iteration + 1] = color
            colors = new_colors
        return result

    def fit_transform(self, dataset):
        self.label_dict = [{} for _ in range(self.n_iter + 1)]
        self.label_counter = [1] * (self.n_iter + 1)
        self.is_fitted_ = True
        return [self._compute_graph_colors(data, grow=True) for data in dataset]

    def transform(self, dataset):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("WeisfeilerLeman.transform requires prior fit_transform.")
        return [self._compute_graph_colors(data, grow=False) for data in dataset]


def compute_stats(label_dict, wl_colors):
    max_num_nodes = max(colors.size(0) for colors in wl_colors)
    vocab_size = [len(stage) + 1 for stage in label_dict]
    n_iter = wl_colors[0].size(1) - 1

    color_counts = []
    for iteration in range(n_iter + 1):
        counts_for_iteration = torch.zeros(len(label_dict[iteration]))
        for colors in wl_colors:
            values, counts = torch.unique(colors[:, iteration], return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                counts_for_iteration[int(value) - 1] += int(count)
        color_counts.append(counts_for_iteration)
    color_probs = [counts / torch.sum(counts) for counts in color_counts]

    color_successors = {}
    for iteration in range(n_iter):
        for colors in wl_colors:
            for node_id in range(colors.size(0)):
                key = (iteration, int(colors[node_id, iteration].item()))
                color_successors.setdefault(key, set()).add(
                    int(colors[node_id, iteration + 1].item())
                )
    return max_num_nodes, vocab_size, color_probs, color_successors


def _build_inverse_label_dicts(label_dict):
    return [
        {int(label_id): signature for signature, label_id in stage.items()}
        for stage in label_dict
    ]


def build_color_id_to_iter1_descriptor(label_dict, iteration):
    """Map an iteration-N color ID to its flat iteration-1 descriptor."""
    iteration = int(iteration)
    if iteration <= 0:
        raise ValueError("Ring decoding requires a WL iteration greater than 0.")

    inverse_label_dicts = _build_inverse_label_dicts(label_dict)
    stage0_color_to_label = {
        int(color_id): int(raw_label)
        for raw_label, color_id in label_dict[0].items()
    }
    iter1_cache = {}

    def iter1_descriptor(color_id):
        color_id = int(color_id)
        if color_id in iter1_cache:
            return iter1_cache[color_id]
        signature = inverse_label_dicts[1].get(color_id)
        if signature is None:
            raise KeyError(f"Unknown color id {color_id} at iteration 1")
        previous_color, neighbor_specs = signature
        center = str(stage0_color_to_label[int(previous_color)])
        neighbors = [
            f"{int(edge_attr)}:{stage0_color_to_label[int(neighbor_color)]}"
            for neighbor_color, edge_attr in neighbor_specs
        ]
        descriptor = center + "_" + ",".join(neighbors)
        iter1_cache[color_id] = descriptor
        return descriptor

    def iter1_ancestor(color_id):
        current_iteration = iteration
        current_color = int(color_id)
        while current_iteration > 1:
            signature = inverse_label_dicts[current_iteration].get(current_color)
            if signature is None:
                raise KeyError(
                    f"Unknown color id {current_color} at iteration {current_iteration}"
                )
            current_color = int(signature[0])
            current_iteration -= 1
        return current_color

    return {
        str(int(color_id)): iter1_descriptor(iter1_ancestor(color_id))
        for color_id in inverse_label_dicts[iteration]
    }


def postprocess_graph(graph, min_nodes=2):
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        return None
    if not nx.is_connected(graph):
        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()
    if graph.number_of_nodes() < int(min_nodes):
        return None
    return graph


def deduplicate_graphs(graphs):
    unique = []
    seen_hashes = set()
    for graph in graphs:
        try:
            graph_hash = nx.weisfeiler_lehman_graph_hash(
                graph,
                node_attr="label",
                edge_attr="bond_type",
            )
        except TypeError:
            graph_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")
        if graph_hash not in seen_hashes:
            seen_hashes.add(graph_hash)
            unique.append(graph)
    return unique

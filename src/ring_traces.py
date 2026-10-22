"""Fit WL metadata and build teacher-forced ring-decoder traces."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from graph_helpers import WeisfeilerLeman, build_color_id_to_iter1_descriptor, compute_stats
from project_paths import portable_path
from wl_color_hierarchy import build_color_hierarchy
from wl_witness_ar import ColorSpec, build_color_specs


@dataclass(frozen=True)
class RingNodeTrace:
    actual_id: int
    ring: int
    prev_color: int
    final_color: int
    label: int
    stubs: tuple[tuple[int, int], ...]
    required: tuple[tuple[int, int], ...]


@dataclass
class RingGraphTrace:
    nodes: list[RingNodeTrace]
    rings: list[list[int]]
    edges: list[tuple[int, int, int]]
    sequence: list[int]
    target_nodes: int
    center_color: int
    encoder_sequence: list[int] | None = None


@dataclass
class RingTraceBuildResult:
    train_traces: list[RingGraphTrace]
    val_traces: list[RingGraphTrace]
    skipped_graphs: int
    failure_counts: dict[str, int]
    metadata: dict[str, Any]
    final_stage: int
    color_specs: list[ColorSpec]
    provenance: dict[str, Any] = field(default_factory=dict)


TRACE_SCHEMA_VERSION = 5
WL_SEMANTICS_VERSION = "requested_wl_encoder_with_frequency_safe_wl1_decoder_v2"


def fit_wl_metadata(train_graphs, n_wl_iters: int):
    wl = WeisfeilerLeman(n_iter=int(n_wl_iters))
    wl_colors = wl.fit_transform(train_graphs)
    _max_nodes, _vocab_size, color_probs, color_successors = compute_stats(wl.label_dict, wl_colors)
    return {
        "label_dict": wl.label_dict,
        "color_probs": color_probs,
        "color_successors": color_successors,
        "train_n_nodes": [int(colors.size(0)) for colors in wl_colors],
        "n_wl_iters": int(n_wl_iters),
    }, wl_colors


def validate_final_stage(metadata, final_stage: int):
    final_stage = int(final_stage)
    expected = int(metadata.get("decoder_wl_stage", metadata["n_wl_iters"]))
    label_dict = metadata.get("label_dict") or []
    if final_stage != expected:
        raise ValueError(
            f"WL final stage mismatch: final_stage={final_stage}, "
            f"decoder_wl_stage={expected}."
        )
    if final_stage < 1:
        raise ValueError("Ring VAE requires at least one WL refinement; increase --n-wl-iters.")
    if final_stage >= len(label_dict):
        raise ValueError(
            f"WL metadata has {len(label_dict)} stages, cannot select final_stage={final_stage}."
        )
    return final_stage


def transform_with_metadata(graphs, metadata):
    wl = WeisfeilerLeman(n_iter=int(metadata["n_wl_iters"]))
    wl.label_dict = metadata["label_dict"]
    wl.label_counter = [max(d.values()) + 1 if d else 1 for d in wl.label_dict]
    wl.is_fitted_ = True
    return wl.transform(graphs)


def build_specs_from_metadata(metadata, final_stage: int):
    color_to_hash = build_color_id_to_iter1_descriptor(metadata["label_dict"], int(final_stage))
    color_probs = None
    if metadata.get("color_probs") is not None:
        color_probs = torch.as_tensor(metadata["color_probs"][int(final_stage)], dtype=torch.float32).cpu()
    return build_color_specs(color_to_hash, color_probs=color_probs)


def _trace_constraint_budget(trace, dataset: str | None):
    dataset = str(dataset or "").upper()
    graph = nx.Graph()
    for node_id, node in enumerate(trace.nodes):
        graph.add_node(int(node_id), label=int(node.label))
    for u, v, edge_type in trace.edges:
        graph.add_edge(int(u), int(v), bond_type=int(edge_type))
    profile = {}
    if dataset == "SBM" and graph.number_of_edges() > 0:
        communities = [
            sorted(int(node) for node in community)
            for community in nx.algorithms.community.greedy_modularity_communities(graph)
        ]
        community_id = {
            node: block
            for block, community in enumerate(communities)
            for node in community
        }
        intra = sum(
            int(community_id.get(int(u)) == community_id.get(int(v)))
            for u, v in graph.edges()
        )
        profile.update({
            "block_sizes": [len(community) for community in communities],
            "intra_edges": int(intra),
            "inter_edges": int(graph.number_of_edges() - intra),
            "degree_sequence": sorted(
                (int(degree) for _node, degree in graph.degree()),
                reverse=True,
            ),
        })
    elif dataset == "PLANAR":
        profile["degree_sequence"] = sorted(
            (int(degree) for _node, degree in graph.degree()),
            reverse=True,
        )
    elif dataset in {"ZINC", "ZINC250K", "QM9", "QM9_H", "MOSES", "GUACAMOL"}:
        profile.update({
            "bond_order_sum": int(sum(int(data.get("bond_type", 1)) for *_uv, data in graph.edges(data=True))),
            "cycle_rank": max(0, int(graph.number_of_edges() - graph.number_of_nodes() + 1)),
        })
    return profile


def infer_ring_model_sizes(metadata, final_stage: int, color_specs: list[ColorSpec]):
    final_stage = int(final_stage)
    edge_ids = set()
    prev_ids = [0]
    hierarchy = build_color_hierarchy(metadata, final_stage)
    for cid, counts in hierarchy.neighbor_counts[final_stage].items():
        prev_ids.append(hierarchy.parent(final_stage, int(cid)) or 0)
        for edge_type, prev_color in counts:
            edge_ids.add(int(edge_type))
            prev_ids.append(int(prev_color))
    return {
        "color_vocab_size": max([0] + [int(v) for v in metadata["label_dict"][final_stage].values()]),
        "encoder_color_vocab_size": max(
            [0]
            + [
                int(v)
                for v in metadata["label_dict"][
                    int(metadata.get("requested_n_wl_iters", final_stage))
                ].values()
            ]
        ),
        "prev_color_vocab_size": max(prev_ids),
        "label_vocab_size": max([0] + [int(spec.center_label) for spec in color_specs]),
        "edge_vocab_size": max([0] + list(edge_ids) + [int(edge) for spec in color_specs for edge, _ in spec.stubs]),
        "max_ring": max(8, int(max(metadata.get("train_n_nodes") or [8]))),
    }


def _graph_fingerprint_item(graph):
    if isinstance(graph, nx.Graph):
        labels = [
            (int(node), int(data.get("label", 0)))
            for node, data in graph.nodes(data=True)
        ]
        edges = [
            (
                min(int(u), int(v)),
                max(int(u), int(v)),
                int(data.get("bond_type", data.get("edge_attr", 1))),
            )
            for u, v, data in graph.edges(data=True)
            if int(u) != int(v)
        ]
        return (len(labels), len(edges), tuple(sorted(labels)), tuple(sorted(edges)))

    num_nodes = int(getattr(graph, "num_nodes", 0))
    edge_index = getattr(graph, "edge_index", None)
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_index is None or edge_index.numel() == 0:
        edges = ()
    else:
        rows = []
        for pos in range(edge_index.size(1)):
            u = int(edge_index[0, pos].item())
            v = int(edge_index[1, pos].item())
            if u == v:
                continue
            edge_type = int(edge_attr[pos].item()) if edge_attr is not None and edge_attr.numel() > pos else 1
            rows.append((min(u, v), max(u, v), edge_type))
        edges = tuple(sorted(set(rows)))
    x = getattr(graph, "x", None)
    if x is None:
        labels = ()
    else:
        flat = x.detach().cpu().view(-1)
        labels = tuple(int(v.item()) for v in flat)
    return (num_nodes, len(edges), labels, edges)


def graph_collection_fingerprint(graphs):
    digest = hashlib.sha256()
    digest.update(str(len(graphs)).encode("utf-8"))
    for graph in graphs:
        digest.update(repr(_graph_fingerprint_item(graph)).encode("utf-8"))
    return digest.hexdigest()


def trace_cache_provenance(
    *,
    dataset: str,
    data_root,
    seed: int,
    n_wl_iters: int,
    max_train_graphs,
    max_val_graphs,
    train_graphs,
    val_graphs,
    smiles_file=None,
    moses_split: str | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    decoder_color_min_frequency: int = 1,
    decoder_rare_fraction_threshold: float = 1.0,
):
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "wl_semantics": WL_SEMANTICS_VERSION,
        "dataset": str(dataset),
        "data_root": portable_path(data_root),
        "seed": int(seed),
        "n_wl_iters": int(n_wl_iters),
        "decoder_color_min_frequency": int(decoder_color_min_frequency),
        "decoder_rare_fraction_threshold": float(decoder_rare_fraction_threshold),
        "max_train_graphs": None if max_train_graphs is None else int(max_train_graphs),
        "max_val_graphs": None if max_val_graphs is None else int(max_val_graphs),
        "train_size": len(train_graphs),
        "val_size": len(val_graphs),
        "train_fingerprint": graph_collection_fingerprint(train_graphs),
        "val_fingerprint": graph_collection_fingerprint(val_graphs),
        "smiles_file": portable_path(smiles_file),
        "moses_split": str(moses_split or "train"),
        "dataset_provenance": dict(dataset_provenance or {}),
    }


def trace_cache_matches(result, expected_provenance: dict[str, Any]):
    actual = getattr(result, "provenance", None)
    if not actual:
        return False, "missing_provenance"
    for key, expected in expected_provenance.items():
        if actual.get(key) != expected:
            return False, f"{key}: cache={actual.get(key)!r} expected={expected!r}"
    try:
        validate_final_stage(result.metadata, result.final_stage)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def graph_data_to_nx_simple(data):
    if isinstance(data, nx.Graph):
        graph = data.copy()
        for node in graph.nodes():
            graph.nodes[node].setdefault("label", 0)
        for _u, _v, edge_data in graph.edges(data=True):
            edge_data.setdefault("bond_type", 1)
        return graph

    graph = nx.Graph()
    x = data.x.view(-1) if getattr(data, "x", None) is not None else None
    for node in range(int(data.num_nodes)):
        label = int(x[node].item()) if x is not None and x.numel() > node else 0
        graph.add_node(node, label=label)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return graph
    edge_attr = getattr(data, "edge_attr", None)
    for pos in range(edge_index.size(1)):
        u = int(edge_index[0, pos].item())
        v = int(edge_index[1, pos].item())
        if u == v:
            continue
        edge_type = int(edge_attr[pos].item()) if edge_attr is not None and edge_attr.numel() > pos else 1
        if graph.has_edge(u, v):
            continue
        graph.add_edge(u, v, bond_type=edge_type)
    return graph


def deterministic_center(graph: nx.Graph, final_colors: list[int]):
    if graph.number_of_nodes() == 0:
        return None, None
    components = list(nx.connected_components(graph))
    if not components:
        return None, None
    if len(components) != 1:
        return None, None
    component = components[0]
    subgraph = graph.subgraph(component).copy()
    if subgraph.number_of_nodes() < 2 or subgraph.number_of_edges() == 0:
        return None, None
    try:
        centers = nx.center(subgraph)
    except nx.NetworkXError:
        lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
        ecc = {node: max(dist.values()) for node, dist in lengths.items()}
        best_ecc = min(ecc.values())
        centers = [node for node, value in ecc.items() if value == best_ecc]
    center = min(
        centers,
        key=lambda node: (
            int(final_colors[int(node)]) if int(node) < len(final_colors) else 0,
            -int(subgraph.degree[int(node)]),
            int(node),
        ),
    )
    return int(center), subgraph


def _required_neighbors(color_hierarchy, final_stage: int, final_color: int):
    counts = color_hierarchy.neighbor_counts[int(final_stage)].get(int(final_color), Counter())
    out = []
    for (edge_type, prev_color), count in sorted(counts.items()):
        out.extend([(int(edge_type), int(prev_color))] * int(count))
    return tuple(out)


def build_ring_trace_for_graph(
    graph,
    wl_colors,
    *,
    final_stage: int,
    encoder_stage: int | None = None,
    color_specs_by_id: dict[int, ColorSpec],
    color_hierarchy,
):
    wl_colors = torch.as_tensor(wl_colors, dtype=torch.long)
    encoder_stage = int(final_stage if encoder_stage is None else encoder_stage)
    if wl_colors.dim() != 2 or wl_colors.size(1) <= max(int(final_stage), encoder_stage):
        return None, "bad_wl_shape"

    nx_graph = graph_data_to_nx_simple(graph)
    if nx_graph.number_of_nodes() > 0 and not nx.is_connected(nx_graph):
        return None, "disconnected_graph"
    final_colors = wl_colors[:, int(final_stage)].tolist()
    prev_colors = wl_colors[:, int(final_stage) - 1].tolist()
    center, component_graph = deterministic_center(nx_graph, final_colors)
    if center is None:
        return None, "empty_or_edgeless"

    lengths = nx.single_source_shortest_path_length(component_graph, center)
    if not lengths:
        return None, "empty_component"

    actual_order = sorted(
        lengths,
        key=lambda node: (
            int(lengths[node]),
            int(final_colors[int(node)]),
            int(prev_colors[int(node)]),
            -int(component_graph.degree[int(node)]),
            int(node),
        ),
    )
    actual_to_trace = {int(actual): idx for idx, actual in enumerate(actual_order)}
    max_ring = max(int(lengths[node]) for node in actual_order)
    rings = [[] for _ in range(max_ring + 1)]
    nodes: list[RingNodeTrace] = []

    for actual in actual_order:
        actual = int(actual)
        final_color = int(final_colors[actual])
        prev_color = int(prev_colors[actual])
        if final_color <= 0 or prev_color <= 0:
            return None, "oov_wl_color"
        spec = color_specs_by_id.get(final_color)
        if spec is None:
            return None, "missing_color_spec"
        parent = color_hierarchy.parent(final_stage, final_color)
        if parent is None or int(parent) != prev_color:
            return None, "bad_parent_color"
        idx = actual_to_trace[actual]
        ring = int(lengths[actual])
        rings[ring].append(idx)
        nodes.append(RingNodeTrace(
            actual_id=actual,
            ring=ring,
            prev_color=prev_color,
            final_color=final_color,
            label=int(spec.center_label),
            stubs=tuple((int(edge), int(nbr)) for edge, nbr in spec.stubs),
            required=_required_neighbors(color_hierarchy, final_stage, final_color),
        ))

    edges = []
    for u, v, edge_data in component_graph.edges(data=True):
        u_idx = actual_to_trace[int(u)]
        v_idx = actual_to_trace[int(v)]
        if u_idx == v_idx:
            continue
        edge_type = int(edge_data.get("bond_type", 1))
        a, b = sorted((u_idx, v_idx))
        edges.append((a, b, edge_type))
    edges = sorted(set(edges))
    if not edges:
        return None, "empty_or_edgeless"

    return RingGraphTrace(
        nodes=nodes,
        rings=[ring for ring in rings if ring],
        edges=edges,
        sequence=[int(node.final_color) for node in nodes],
        target_nodes=len(nodes),
        center_color=int(nodes[0].final_color),
        encoder_sequence=[int(wl_colors[int(actual), encoder_stage].item()) for actual in actual_order],
    ), None


def build_ring_traces(
    graphs,
    wl_colors,
    *,
    final_stage: int,
    encoder_stage: int | None = None,
    color_specs: list[ColorSpec],
    metadata,
    max_graphs: int | None = None,
):
    color_specs_by_id = {int(spec.color_id): spec for spec in color_specs}
    color_hierarchy = build_color_hierarchy(metadata, final_stage)
    n_graphs = len(graphs) if max_graphs is None else min(len(graphs), int(max_graphs))
    traces = []
    failure_counts = defaultdict(int)
    skipped = 0
    for idx in range(n_graphs):
        trace, failure = build_ring_trace_for_graph(
            graphs[idx],
            wl_colors[idx],
            final_stage=final_stage,
            encoder_stage=encoder_stage,
            color_specs_by_id=color_specs_by_id,
            color_hierarchy=color_hierarchy,
        )
        if trace is None:
            skipped += 1
            failure_counts[str(failure)] += 1
            continue
        traces.append(trace)
    return traces, skipped, dict(failure_counts)


def build_training_ring_traces(
    train_graphs,
    val_graphs,
    *,
    n_wl_iters: int,
    max_train_graphs: int | None = None,
    max_val_graphs: int | None = None,
    seed: int = 0,
    decoder_color_min_frequency: int = 1,
    decoder_rare_fraction_threshold: float = 1.0,
    dataset: str | None = None,
):
    metadata, train_wl = fit_wl_metadata(train_graphs, n_wl_iters)
    requested_stage = int(metadata["n_wl_iters"])
    min_frequency = max(1, int(decoder_color_min_frequency))
    rare_fraction_threshold = min(max(float(decoder_rare_fraction_threshold), 0.0), 1.0)
    stage_stats = {}
    for stage in range(1, requested_stage + 1):
        color_ids = sorted({int(value) for value in metadata["label_dict"][stage].values() if int(value) > 0})
        counter = Counter(
            int(color)
            for colors in train_wl
            for color in torch.as_tensor(colors)[:, stage].tolist()
            if int(color) > 0
        )
        rare = sum(int(counter[color_id]) < min_frequency for color_id in color_ids)
        stage_stats[str(stage)] = {
            "vocabulary": len(color_ids),
            "rare_colors": int(rare),
            "rare_fraction": rare / max(len(color_ids), 1),
            "min_frequency": min_frequency,
        }
    final_stage = requested_stage
    if requested_stage > 1 and stage_stats[str(requested_stage)]["rare_fraction"] > rare_fraction_threshold:
        final_stage = 1
    metadata["requested_n_wl_iters"] = requested_stage
    metadata["decoder_wl_stage"] = int(final_stage)
    metadata["decoder_color_backoff"] = {
        "mode": "requested_stage_encoder_with_semantic_safe_wl1_decoder",
        "min_frequency": min_frequency,
        "rare_fraction_threshold": rare_fraction_threshold,
        "applied": bool(final_stage != requested_stage),
        "stage_stats": stage_stats,
    }
    metadata["dataset"] = None if dataset is None else str(dataset)
    final_stage = validate_final_stage(metadata, final_stage)
    color_specs = build_specs_from_metadata(metadata, final_stage)

    train_traces, train_skipped, train_failures = build_ring_traces(
        train_graphs,
        train_wl,
        final_stage=final_stage,
        encoder_stage=requested_stage,
        color_specs=color_specs,
        metadata=metadata,
        max_graphs=max_train_graphs,
    )
    metadata["train_n_nodes"] = [int(trace.target_nodes) for trace in train_traces]
    metadata["train_n_edges"] = [int(len(trace.edges)) for trace in train_traces]
    metadata["train_graph_budgets"] = []
    for trace in train_traces:
        budget = {
            "nodes": int(trace.target_nodes),
            "edges": int(len(trace.edges)),
            "mean_degree_x10": int(round(20.0 * len(trace.edges) / max(trace.target_nodes, 1))),
        }
        budget.update(_trace_constraint_budget(trace, dataset))
        metadata["train_graph_budgets"].append(budget)
    metadata["constraint_kind"] = (
        "planar" if str(dataset or "").upper() == "PLANAR"
        else "sbm" if str(dataset or "").upper() == "SBM"
        else "molecule" if str(dataset or "").upper() in {
            "ZINC", "ZINC250K", "QM9", "QM9_H", "MOSES", "GUACAMOL"
        }
        else "generic"
    )
    if metadata["constraint_kind"] == "molecule":
        allowed_bonds = set()
        max_degree_by_label = Counter()
        max_bond_order_by_label = Counter()
        for trace in train_traces:
            degree_by_node = Counter()
            bond_order_by_node = Counter()
            for u, v, edge_type in trace.edges:
                left_label = int(trace.nodes[int(u)].label)
                right_label = int(trace.nodes[int(v)].label)
                allowed_bonds.add((min(left_label, right_label), max(left_label, right_label), int(edge_type)))
                degree_by_node[int(u)] += 1
                degree_by_node[int(v)] += 1
                bond_order_by_node[int(u)] += int(edge_type)
                bond_order_by_node[int(v)] += int(edge_type)
            for node_id, degree in degree_by_node.items():
                label = int(trace.nodes[int(node_id)].label)
                max_degree_by_label[label] = max(int(max_degree_by_label[label]), int(degree))
                max_bond_order_by_label[label] = max(
                    int(max_bond_order_by_label[label]),
                    int(bond_order_by_node[node_id]),
                )
        metadata["molecule_constraint_profile"] = {
            "allowed_bonds": [list(item) for item in sorted(allowed_bonds)],
            "max_degree_by_label": {
                str(label): int(degree)
                for label, degree in sorted(max_degree_by_label.items())
            },
            "max_bond_order_by_label": {
                str(label): int(order)
                for label, order in sorted(max_bond_order_by_label.items())
            },
        }

    val_wl = transform_with_metadata(val_graphs, metadata) if val_graphs else []
    val_traces, val_skipped, val_failures = build_ring_traces(
        val_graphs,
        val_wl,
        final_stage=final_stage,
        encoder_stage=requested_stage,
        color_specs=color_specs,
        metadata=metadata,
        max_graphs=max_val_graphs,
    ) if val_graphs else ([], 0, {})

    if not val_traces and len(train_traces) > 2:
        rng = random.Random(seed)
        rng.shuffle(train_traces)
        n_val = max(1, int(round(0.1 * len(train_traces))))
        val_traces = train_traces[:n_val]
        train_traces = train_traces[n_val:]

    failure_counts = defaultdict(int)
    for key, value in train_failures.items():
        failure_counts[f"train/{key}"] += int(value)
    for key, value in val_failures.items():
        failure_counts[f"val/{key}"] += int(value)

    return RingTraceBuildResult(
        train_traces=train_traces,
        val_traces=val_traces,
        skipped_graphs=int(train_skipped + val_skipped),
        failure_counts=dict(failure_counts),
        metadata=metadata,
        final_stage=final_stage,
        color_specs=color_specs,
    )


def load_trace_cache(path: str | Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_trace_cache(result: RingTraceBuildResult, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)

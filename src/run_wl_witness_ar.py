"""Run the WL witness autoregressive baseline and shared graph metrics."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# graph-tool and other native dependencies in this environment expose
# incompatible Boost.Python converter registries.  graph-tool must be loaded
# first or BlockState construction can lose its iterator converter.  Keep the
# import guarded so graph-tool remains optional for non-SBM runs.
_GRAPH_TOOL_MODULE = None
_GRAPH_TOOL_IMPORT_ERROR = None
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Error importing cairo.*")
        warnings.filterwarnings("ignore", message="Error importing draw module.*")
        import graph_tool.all as _GRAPH_TOOL_MODULE
except Exception as exc:
    _GRAPH_TOOL_IMPORT_ERROR = exc

import networkx as nx
import numpy as np
import torch
from scipy.stats import chi2

from dataset import load_tu_graph_dataset, split_graph_dataset
from project_paths import DATASETS_ROOT, portable_path
from graph_helpers import (
    WeisfeilerLeman,
    build_color_id_to_iter1_descriptor,
    compute_stats,
    deduplicate_graphs,
    postprocess_graph,
)
from ring_traces import validate_final_stage
from wl_witness_ar import (
    PairedStubARGenerator,
    build_color_specs,
    sample_node_counts,
)
from wl_color_ar_policy import load_neural_color_policy
from zinc_utils import (
    build_atom_valence_limits,
    build_training_smiles,
    compute_mol_metrics,
    load_atom_decoder,
    load_processed_zinc_split,
)


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _prefixed(prefix, values):
    return {f"{prefix}/{key}": value for key, value in values.items()}


def make_parser():
    parser = argparse.ArgumentParser(
        description="Paired-stub autoregressive WL-witness generator."
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint with wl_metadata. If omitted, WL metadata is fit from the train split.")
    parser.add_argument("--dataset", default="ZINC250K",
                        choices=["ZINC", "ZINC250K", "MUTAG", "NCI1"])
    parser.add_argument("--zinc-subset", default=None, choices=["small", "large"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-wl-iters", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="inference_runs/wl_witness_ar/latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="auto", choices=["auto", "generic", "molecule"],
                        help="molecule computes RDKit metrics; generic computes graph metrics only.")
    parser.add_argument("--max-restarts", type=int, default=300)
    parser.add_argument("--max-steps-per-node", type=int, default=25)
    parser.add_argument("--new-node-prob", type=float, default=0.70)
    parser.add_argument("--allow-disconnected", action="store_true")
    parser.add_argument("--budget-mode", default="off",
                        choices=["off", "generic-degree", "molecule"],
                        help="Optional hard local budget during AR generation.")
    parser.add_argument("--max-node-degree", type=int, default=None,
                        help="Degree cap for --budget-mode generic-degree. If omitted, uses train max degree.")
    parser.add_argument("--edge-weight", default="unit", choices=["unit", "bond_order"],
                        help="Budget demand mode. Use bond_order for molecules.")
    parser.add_argument("--max-train-graphs", type=int, default=None,
                        help="Cap graphs used when fitting WL metadata without a checkpoint.")
    parser.add_argument("--train-smiles-cache", default=None)
    parser.add_argument("--metric-max-train-graphs", type=int, default=20000,
                        help="Cap training graphs used for distribution diagnostics.")
    parser.add_argument("--mmd-max-graphs", type=int, default=2000,
                        help="Cap train/generated graphs used for generic-graph MMD diagnostics.")
    parser.add_argument("--ar-policy-checkpoint", default=None,
                        help="Optional trained neural WL-color AR policy checkpoint.")
    parser.add_argument("--ar-policy-device", default="auto",
                        help="Device for --ar-policy-checkpoint: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--ar-policy-temperature", type=float, default=1.0,
                        help="Sampling temperature for the neural WL-color AR policy.")
    parser.add_argument("--ar-policy-empirical-bias", type=float, default=0.0,
                        help="Add this multiplier times log(empirical color frequency) to neural logits.")
    return parser


def load_checkpoint_metadata(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wl_meta = ckpt.get("wl_metadata")
    if wl_meta is None and isinstance(ckpt.get("checkpoint"), dict):
        wl_meta = ckpt["checkpoint"].get("wl_metadata")
    if wl_meta is None:
        raise RuntimeError(f"No wl_metadata found in {checkpoint_path}")
    ckpt_args = ckpt.get("args") or ckpt.get("checkpoint_args") or {}
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)
    n_wl_iters = int(ckpt_args.get("n_wl_iters") or (len(wl_meta["label_dict"]) - 1))
    return {
        "label_dict": wl_meta["label_dict"],
        "color_probs": wl_meta.get("color_probs"),
        "train_n_nodes": wl_meta.get("train_n_nodes") or [],
        "n_wl_iters": n_wl_iters,
    }


def load_train_graphs(dataset, data_root, zinc_subset, max_train_graphs=None):
    if dataset in {"ZINC", "ZINC250K"}:
        subset = zinc_subset
        if subset is None:
            subset = "large" if dataset == "ZINC250K" else "small"
        return load_processed_zinc_split(
            split="train",
            subset=subset,
            root=data_root,
            max_graphs=max_train_graphs,
        )
    graphs = load_tu_graph_dataset(Path(data_root).parent if Path(data_root).name == dataset else data_root, dataset)
    train_graphs, _val_graphs, _test_graphs = split_graph_dataset(graphs, seed=0)
    if max_train_graphs is not None:
        train_graphs = train_graphs[:int(max_train_graphs)]
    return train_graphs


def fit_metadata_from_graphs(train_graphs, n_wl_iters):
    wl = WeisfeilerLeman(n_iter=n_wl_iters)
    wl_colors = wl.fit_transform(train_graphs)
    _max_nodes, _vocab_size, color_probs, _successors = compute_stats(wl.label_dict, wl_colors)
    return {
        "label_dict": wl.label_dict,
        "color_probs": color_probs,
        "train_n_nodes": [int(c.size(0)) for c in wl_colors],
        "n_wl_iters": int(n_wl_iters),
    }


def compute_max_observed_degree(graphs):
    max_degree = 0
    for data in graphs:
        if data.edge_index is None or data.edge_index.numel() == 0:
            continue
        deg = torch.bincount(data.edge_index[0].to(torch.long), minlength=int(data.num_nodes))
        if deg.numel() > 0:
            max_degree = max(max_degree, int(deg.max().item()))
    return max_degree


def resolve_budget(args, train_graphs, data_root, mode):
    if args.budget_mode == "off":
        return None
    if args.budget_mode == "generic-degree":
        cap = args.max_node_degree
        if cap is None:
            cap = compute_max_observed_degree(train_graphs)
        return {"__default__": int(cap)}
    if args.budget_mode == "molecule":
        if mode != "molecule":
            raise ValueError("--budget-mode molecule requires molecule mode/dataset.")
        return build_atom_valence_limits(load_atom_decoder(root=data_root))
    raise ValueError(args.budget_mode)


def _dataset_key(dataset):
    if dataset is None:
        return ""
    return str(dataset).upper().replace("-", "_").replace(" ", "_")


def _sanitize_nx_graph(graph, min_nodes=2):
    nx_graph = _graph_data_to_nx(graph).copy()
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    for _node, data in nx_graph.nodes(data=True):
        data.setdefault("label", 0)
    for _u, _v, data in nx_graph.edges(data=True):
        data.setdefault("bond_type", 1)
    if nx_graph.number_of_nodes() < min_nodes:
        return None
    return nx_graph


def _topology_hash(graph):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The hashes produced for graphs without node or edge attributes changed in v3\.5.*",
                category=UserWarning,
                module=r"networkx\.algorithms\.graph_hashing",
            )
            return nx.weisfeiler_lehman_graph_hash(graph)
    except Exception:
        degrees = tuple(sorted(int(deg) for _node, deg in graph.degree()))
        return (graph.number_of_nodes(), graph.number_of_edges(), degrees)


def _iso_bucket_add(buckets, graph):
    buckets.setdefault(_topology_hash(graph), []).append(graph)


def _is_isomorphic_to_bucket(graph, buckets):
    for candidate in buckets.get(_topology_hash(graph), []):
        try:
            if nx.faster_could_be_isomorphic(graph, candidate) and nx.is_isomorphic(graph, candidate):
                return True
        except Exception:
            if nx.is_isomorphic(graph, candidate):
                return True
    return False


def _unique_topology_reps(graphs):
    buckets = {}
    unique = []
    for graph in graphs:
        if _is_isomorphic_to_bucket(graph, buckets):
            continue
        _iso_bucket_add(buckets, graph)
        unique.append(graph)
    return unique, buckets


def _is_grid_graph(graph):
    n_nodes = int(graph.number_of_nodes())
    if n_nodes < 100 or n_nodes > 361:
        return False
    for rows in range(10, 20):
        if n_nodes % rows != 0:
            continue
        cols = n_nodes // rows
        if cols < 10 or cols > 19:
            continue
        if int(graph.number_of_edges()) != rows * (cols - 1) + cols * (rows - 1):
            continue
        reference = nx.grid_2d_graph(rows, cols)
        if nx.is_isomorphic(graph, reference):
            return True
    return False


def _is_sbm_graph_heuristic(graph):
    n_nodes = int(graph.number_of_nodes())
    if n_nodes < 40 or n_nodes > 200 or graph.number_of_edges() == 0:
        return False
    try:
        communities = [
            set(c)
            for c in nx.algorithms.community.greedy_modularity_communities(graph)
        ]
    except Exception:
        return False
    if not 2 <= len(communities) <= 5:
        return False
    sizes = [len(c) for c in communities]
    if min(sizes) < 10 or max(sizes) > 55:
        return False

    community_id = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_id[node] = idx

    intra_edges = 0
    inter_edges = 0
    for u, v in graph.edges():
        if community_id.get(u) == community_id.get(v):
            intra_edges += 1
        else:
            inter_edges += 1

    intra_possible = sum(size * (size - 1) / 2 for size in sizes)
    inter_possible = n_nodes * (n_nodes - 1) / 2 - intra_possible
    if intra_possible <= 0 or inter_possible <= 0:
        return False
    intra_density = intra_edges / intra_possible
    inter_density = inter_edges / inter_possible
    return intra_density >= 0.10 and inter_density <= 0.08 and intra_density > 2.0 * inter_density


def _load_graph_tool():
    if _GRAPH_TOOL_MODULE is None:
        raise RuntimeError(
            "Exact SPECTRE SBM validity requires graph-tool in the active Python environment. "
            "Install it from conda-forge or use --sbm-validity-backend heuristic only for diagnostics."
        ) from _GRAPH_TOOL_IMPORT_ERROR
    return _GRAPH_TOOL_MODULE


def _is_sbm_graph_graph_tool(
    graph,
    *,
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 100,
):
    """Public DiGress/SPECTRE Bayesian SBM validity test using graph-tool."""
    gt = _load_graph_tool()
    adj = nx.adjacency_matrix(graph).toarray()
    idx = adj.nonzero()
    candidate = gt.Graph()
    candidate.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(candidate)
    except ValueError:
        return False if strict else 0.0
    for _ in range(int(refinement_steps)):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    blocks = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=blocks)
    edge_matrix = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = np.asarray(edge_matrix.todense()[:n_blocks, :n_blocks], dtype=float)
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1.0e-6)
    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1.0e-6)
    w_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1.0e-6)
    w_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1.0e-6)
    wald = w_inter.copy()
    np.fill_diagonal(wald, w_intra)
    probability = float((1 - chi2.cdf(abs(wald), 1)).mean())
    return probability > 0.9 if strict else probability


def _resolve_sbm_validity_backend(requested: str) -> str:
    requested = str(requested or "auto")
    if requested == "heuristic":
        return "heuristic"
    if requested == "exact":
        _load_graph_tool()
        return "exact"
    if requested != "auto":
        raise ValueError(f"Unknown SBM validity backend {requested!r}")
    try:
        _load_graph_tool()
        return "exact"
    except RuntimeError:
        return "heuristic"


def _graph_validity_kind(dataset, sbm_validity_backend="auto"):
    key = _dataset_key(dataset)
    if key == "PLANAR":
        return "planar_connected"
    if key == "GRID":
        return "grid_isomorphism"
    if key == "SBM":
        backend = _resolve_sbm_validity_backend(sbm_validity_backend)
        return "sbm_graph_tool_exact" if backend == "exact" else "sbm_heuristic"
    return "not_applicable"


def _is_valid_for_dataset(graph, dataset, *, validity_kind=None, sbm_refinement_steps=100):
    kind = validity_kind or _graph_validity_kind(dataset)
    if kind == "planar_connected":
        try:
            is_planar, _embedding = nx.check_planarity(graph)
        except Exception:
            is_planar = False
        return bool(is_planar and nx.is_connected(graph))
    if kind == "grid_isomorphism":
        return _is_grid_graph(graph)
    if kind == "sbm_heuristic":
        return _is_sbm_graph_heuristic(graph)
    if kind == "sbm_graph_tool_exact":
        return _is_sbm_graph_graph_tool(graph, refinement_steps=sbm_refinement_steps)
    return True


def graph_metrics(
    graphs,
    train_graphs=None,
    dataset=None,
    *,
    sbm_validity_backend="auto",
    sbm_refinement_steps=100,
    sbm_vun_refinement_steps=1000,
):
    raw_graphs = []
    cleaned = []
    raw_connected = 0
    clean_connected = 0
    raw_planar = 0
    clean_planar = 0
    valid = 0
    clean_valid = 0
    component_counts = []
    isolate_counts = []
    largest_component_nodes = []
    largest_component_fractions = []
    cleanup_removed_nodes = []
    cleanup_removed_edges = []
    validity_kind = _graph_validity_kind(dataset, sbm_validity_backend)
    for graph in graphs:
        graph_full = _sanitize_nx_graph(graph, min_nodes=2)
        if graph_full is None:
            continue
        raw_graphs.append(graph_full)
        components = list(nx.connected_components(graph_full))
        n_nodes = int(graph_full.number_of_nodes())
        component_counts.append(len(components))
        isolate_counts.append(len(list(nx.isolates(graph_full))))
        largest_nodes = max((len(component) for component in components), default=0)
        largest_component_nodes.append(largest_nodes)
        largest_component_fractions.append(largest_nodes / max(n_nodes, 1))
        try:
            raw_connected += int(nx.is_connected(graph_full))
        except Exception:
            pass
        try:
            is_planar, _embedding = nx.check_planarity(graph_full)
            raw_planar += int(bool(is_planar))
        except Exception:
            pass
        graph_clean = postprocess_graph(graph_full, min_nodes=2)
        if graph_clean is not None:
            cleaned.append(graph_clean)
            cleanup_removed_nodes.append(n_nodes - int(graph_clean.number_of_nodes()))
            cleanup_removed_edges.append(int(graph_full.number_of_edges()) - int(graph_clean.number_of_edges()))
            clean_valid += int(_is_valid_for_dataset(
                graph_clean,
                dataset,
                validity_kind=validity_kind,
                sbm_refinement_steps=sbm_refinement_steps,
            ))
            try:
                clean_connected += int(nx.is_connected(graph_clean))
            except Exception:
                pass
            try:
                is_planar, _embedding = nx.check_planarity(graph_clean)
                clean_planar += int(bool(is_planar))
            except Exception:
                pass
        else:
            cleanup_removed_nodes.append(n_nodes)
            cleanup_removed_edges.append(int(graph_full.number_of_edges()))
        is_valid = _is_valid_for_dataset(
            graph_full,
            dataset,
            validity_kind=validity_kind,
            sbm_refinement_steps=sbm_refinement_steps,
        )
        valid += int(is_valid)

    unique_reps, unique_buckets = _unique_topology_reps(raw_graphs)
    dedup = deduplicate_graphs(raw_graphs)

    train_buckets = {}
    if train_graphs is not None:
        for graph in train_graphs:
            graph_full = _sanitize_nx_graph(graph, min_nodes=2)
            if graph_full is not None:
                _iso_bucket_add(train_buckets, graph_full)

    novel_reps = [
        graph
        for graph in unique_reps
        if not train_buckets or not _is_isomorphic_to_bucket(graph, train_buckets)
    ]

    vun_buckets = {}
    valid_unique_novel = 0
    for graph in raw_graphs:
        if _is_isomorphic_to_bucket(graph, vun_buckets):
            continue
        _iso_bucket_add(vun_buckets, graph)
        if train_buckets and _is_isomorphic_to_bucket(graph, train_buckets):
            continue
        is_vun_valid = _is_valid_for_dataset(
            graph,
            dataset,
            validity_kind=validity_kind,
            sbm_refinement_steps=sbm_vun_refinement_steps,
        )
        if not is_vun_valid:
            continue
        valid_unique_novel += 1

    n_total = len(graphs)
    n_raw = len(raw_graphs)
    n_clean = len(cleaned)
    n_unique = len(unique_reps)
    metrics = {
        "graph/n_total": n_total,
        "graph/raw_valid_graphs": n_raw,
        "graph/after_cleanup": n_clean,
        "graph/raw_with_multiple_components": sum(value > 1 for value in component_counts),
        "graph/raw_with_isolates": sum(value > 0 for value in isolate_counts),
        "graph/raw_mean_component_count": _mean(component_counts),
        "graph/raw_max_component_count": max(component_counts, default=0),
        "graph/raw_mean_isolated_nodes": _mean(isolate_counts),
        "graph/raw_max_isolated_nodes": max(isolate_counts, default=0),
        "graph/raw_mean_largest_component_nodes": _mean(largest_component_nodes),
        "graph/raw_mean_largest_component_fraction": _mean(largest_component_fractions),
        "graph/cleanup_mean_removed_nodes": _mean(cleanup_removed_nodes),
        "graph/cleanup_mean_removed_edges": _mean(cleanup_removed_edges),
        "graph/connected": raw_connected,
        "graph/clean_connected": clean_connected,
        "graph/valid": valid,
        "graph/clean_valid": clean_valid,
        "graph/n_unique": n_unique,
        "graph/n_novel": len(novel_reps),
        "graph/n_valid_unique_novel": valid_unique_novel,
        "graph/unique_wl_hash": len(dedup),
        "graph/validity_metric": validity_kind,
        "graph/sbm_accuracy_refinement_steps": (
            int(sbm_refinement_steps) if validity_kind == "sbm_graph_tool_exact" else None
        ),
        "graph/sbm_vun_refinement_steps": (
            int(sbm_vun_refinement_steps) if validity_kind == "sbm_graph_tool_exact" else None
        ),
        "graph/cleanup_fraction": n_clean / max(n_total, 1),
        "graph/raw_multiple_components_fraction": sum(value > 1 for value in component_counts) / max(n_total, 1),
        "graph/raw_isolate_fraction": sum(value > 0 for value in isolate_counts) / max(n_total, 1),
        "graph/connected_fraction": raw_connected / max(n_total, 1),
        "graph/raw_connected_fraction": raw_connected / max(n_total, 1),
        "graph/clean_connected_fraction": clean_connected / max(n_clean, 1),
        "graph/planar_fraction": raw_planar / max(n_total, 1),
        "graph/raw_planar_fraction": raw_planar / max(n_total, 1),
        "graph/clean_planar_fraction": clean_planar / max(n_clean, 1),
        "graph/valid_fraction": valid / max(n_total, 1),
        "graph/valid_clean_fraction": clean_valid / max(n_clean, 1),
        "graph/unique_fraction": n_unique / max(n_raw, 1),
        "graph/unique_isomorphism_fraction": n_unique / max(n_raw, 1),
        "graph/novel_fraction": len(novel_reps) / max(n_unique, 1),
        "graph/valid_unique_novel_fraction": valid_unique_novel / max(n_total, 1),
    }
    return metrics


def largest_component_views(samples):
    graphs = []
    source_attempt_indices = []
    for attempt_idx, sample in enumerate(samples):
        graph = sample.graph
        if graph is None:
            continue
        components = list(nx.connected_components(graph)) if graph.number_of_nodes() > 0 else []
        n_nodes = int(graph.number_of_nodes())
        n_edges = int(graph.number_of_edges())
        largest_nodes = max((len(component) for component in components), default=0)
        cleaned = postprocess_graph(graph, min_nodes=2)
        cleaned_nodes = int(cleaned.number_of_nodes()) if cleaned is not None else 0
        cleaned_edges = int(cleaned.number_of_edges()) if cleaned is not None else 0
        sample.stats.update({
            "raw_n_components": len(components),
            "raw_n_isolates": len(list(nx.isolates(graph))),
            "raw_largest_component_nodes": largest_nodes,
            "raw_largest_component_fraction": largest_nodes / max(n_nodes, 1),
            "largest_component_available": cleaned is not None,
            "largest_component_n_nodes": cleaned_nodes,
            "largest_component_n_edges": cleaned_edges,
            "cleanup_removed_nodes": n_nodes - cleaned_nodes,
            "cleanup_removed_edges": n_edges - cleaned_edges,
        })
        if cleaned is not None:
            graphs.append(cleaned)
            source_attempt_indices.append(attempt_idx)
    return graphs, source_attempt_indices


def _mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def _std(values):
    values = [float(v) for v in values]
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def _normalised_counter(values):
    counter = Counter(int(v) for v in values)
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _distribution_l1(train_values, gen_values):
    p = _normalised_counter(train_values)
    q = _normalised_counter(gen_values)
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _count_distribution_metrics(prefix, train_values, gen_values):
    if not gen_values:
        return {
            f"{prefix}/train_mean": _mean(train_values),
            f"{prefix}/gen_mean": None,
            f"{prefix}/train_std": _std(train_values),
            f"{prefix}/gen_std": None,
            f"{prefix}/l1": None,
            f"{prefix}/tv": None,
            f"{prefix}/status": "empty_generated_distribution",
        }
    l1 = _distribution_l1(train_values, gen_values)
    return {
        f"{prefix}/train_mean": _mean(train_values),
        f"{prefix}/gen_mean": _mean(gen_values),
        f"{prefix}/train_std": _std(train_values),
        f"{prefix}/gen_std": _std(gen_values),
        f"{prefix}/l1": l1,
        f"{prefix}/tv": 0.5 * l1,
        f"{prefix}/status": "ok",
    }


def _graph_data_to_nx(data):
    if isinstance(data, nx.Graph):
        return data

    graph = nx.Graph()
    num_nodes = int(data.num_nodes)
    x = getattr(data, "x", None)
    for node in range(num_nodes):
        if x is not None and int(x.numel()) > node:
            label = int(x[node].item())
        else:
            label = 0
        graph.add_node(node, label=label)

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return graph

    edge_attr = getattr(data, "edge_attr", None)
    seen_edges = set()
    for edge_pos in range(edge_index.size(1)):
        u = int(edge_index[0, edge_pos].item())
        v = int(edge_index[1, edge_pos].item())
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) in seen_edges:
            continue
        seen_edges.add((a, b))
        if edge_attr is not None and edge_attr.numel() > edge_pos:
            edge_label = int(edge_attr[edge_pos].item())
        else:
            edge_label = 1
        graph.add_edge(a, b, bond_type=edge_label)
    return graph


def _node_labels(graph):
    return [int(data.get("label", 0)) for _node, data in graph.nodes(data=True)]


def _edge_labels(graph):
    return [int(data.get("bond_type", data.get("edge_attr", 1))) for *_uv, data in graph.edges(data=True)]


def _degrees(graph):
    return [int(deg) for _node, deg in graph.degree()]


def _aggregate(values_by_graph):
    out = []
    for values in values_by_graph:
        out.extend(values)
    return out


def _one_based_hist_from_ids(ids, size):
    vec = torch.zeros(int(size), dtype=torch.float32)
    total = 0
    for cid in ids:
        cid = int(cid)
        if 1 <= cid <= size:
            vec[cid - 1] += 1.0
            total += 1
    if total > 0:
        vec /= float(total)
    return vec


def _generated_color_ids(graphs):
    ids = []
    for graph in graphs:
        for _node, data in graph.nodes(data=True):
            cid = data.get("color_id")
            if cid is not None:
                ids.append(int(cid))
    return ids


def _stage_color_probs(metadata, final_stage):
    color_probs = metadata.get("color_probs")
    if color_probs is None or final_stage >= len(color_probs):
        return None
    stage_probs = color_probs[final_stage]
    if stage_probs is None:
        return None
    stage_probs = torch.as_tensor(stage_probs, dtype=torch.float32)
    if stage_probs.numel() == 0:
        return None
    return stage_probs


def _wl_color_distribution_metrics(graphs, metadata, final_stage):
    train_probs = _stage_color_probs(metadata, final_stage)
    gen_ids = _generated_color_ids(graphs)
    if train_probs is None or not gen_ids:
        return {}

    gen_probs = _one_based_hist_from_ids(gen_ids, int(train_probs.numel()))
    l1 = float(torch.sum(torch.abs(train_probs - gen_probs)).item())
    train_support = train_probs > 0
    gen_support = gen_probs > 0
    generated_total = max(len(gen_ids), 1)
    oov_count = sum(
        1 for cid in gen_ids
        if cid < 1 or cid > train_probs.numel() or float(train_probs[cid - 1].item()) <= 0.0
    )
    train_support_count = int(train_support.sum().item())
    gen_support_count = int(gen_support.sum().item())
    train_mass_covered = float(train_probs[gen_support].sum().item()) if gen_support.any() else 0.0
    return {
        "dist/wl_color/l1": l1,
        "dist/wl_color/tv": 0.5 * l1,
        "dist/wl_color/generated_unique": gen_support_count,
        "dist/wl_color/train_support": train_support_count,
        "dist/wl_color/generated_support_fraction": gen_support_count / max(train_support_count, 1),
        "dist/wl_color/train_mass_covered": train_mass_covered,
        "dist/wl_color/oov_fraction": oov_count / generated_total,
    }


def distribution_metrics(train_graphs, generated_graphs, metadata, final_stage, max_train_graphs):
    if max_train_graphs is not None and max_train_graphs > 0:
        train_graphs = train_graphs[:int(max_train_graphs)]
    train_nx = [_graph_data_to_nx(graph) for graph in train_graphs]
    gen_nx = [_graph_data_to_nx(graph) for graph in generated_graphs]

    train_node_counts = [graph.number_of_nodes() for graph in train_nx]
    gen_node_counts = [graph.number_of_nodes() for graph in gen_nx]
    train_edge_counts = [graph.number_of_edges() for graph in train_nx]
    gen_edge_counts = [graph.number_of_edges() for graph in gen_nx]

    train_degrees = _aggregate(_degrees(graph) for graph in train_nx)
    gen_degrees = _aggregate(_degrees(graph) for graph in gen_nx)
    train_node_labels = _aggregate(_node_labels(graph) for graph in train_nx)
    gen_node_labels = _aggregate(_node_labels(graph) for graph in gen_nx)
    train_edge_labels = _aggregate(_edge_labels(graph) for graph in train_nx)
    gen_edge_labels = _aggregate(_edge_labels(graph) for graph in gen_nx)

    metrics = {
        "dist/train_graphs_used": len(train_nx),
        "dist/generated_graphs_used": len(gen_nx),
    }
    metrics.update(_count_distribution_metrics("dist/node_count", train_node_counts, gen_node_counts))
    metrics.update(_count_distribution_metrics("dist/edge_count", train_edge_counts, gen_edge_counts))
    metrics.update(_count_distribution_metrics("dist/degree", train_degrees, gen_degrees))
    metrics.update({
        "dist/node_label/l1": _distribution_l1(train_node_labels, gen_node_labels),
        "dist/node_label/tv": 0.5 * _distribution_l1(train_node_labels, gen_node_labels),
        "dist/edge_label/l1": _distribution_l1(train_edge_labels, gen_edge_labels),
        "dist/edge_label/tv": 0.5 * _distribution_l1(train_edge_labels, gen_edge_labels),
    })
    metrics.update(_wl_color_distribution_metrics(gen_nx, metadata, final_stage))
    return metrics


def _degree_hist_feature(graph, size):
    return _one_based_hist_from_ids([d + 1 for d in _degrees(graph)], size)


def _clustering_hist_feature(graph, bins=100):
    vec = torch.zeros(int(bins), dtype=torch.float32)
    if graph.number_of_nodes() == 0:
        return vec
    values = nx.clustering(graph).values()
    total = 0
    for value in values:
        idx = min(int(bins) - 1, max(0, int(float(value) * int(bins))))
        vec[idx] += 1.0
        total += 1
    if total > 0:
        vec /= float(total)
    return vec


def _scalar_feature(value):
    return torch.tensor([float(value)], dtype=torch.float32)


def _spectral_hist_feature(graph, bins=32):
    vec = torch.zeros(int(bins), dtype=torch.float32)
    n = int(graph.number_of_nodes())
    if n <= 0:
        return vec
    try:
        import numpy as np

        nodes = list(graph.nodes())
        index = {node: idx for idx, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=np.float64)
        for u, v in graph.edges():
            i = index[u]
            j = index[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        deg = adj.sum(axis=1)
        inv_sqrt = np.zeros_like(deg)
        mask = deg > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        lap = np.eye(n, dtype=np.float64) - (inv_sqrt[:, None] * adj * inv_sqrt[None, :])
        eigvals = np.linalg.eigvalsh(lap)
        hist, _edges = np.histogram(eigvals, bins=int(bins), range=(0.0, 2.0))
        values = torch.tensor(hist, dtype=torch.float32)
        total = float(values.sum().item())
        return values / total if total > 0.0 else vec
    except Exception:
        return vec


def _four_node_orbit_counts_feature(graph, max_exact_nodes=40, max_triplets=20000, max_quadruplets=8192):
    """15-dimensional induced graphlet orbit counts for graphlets with up to 4 nodes.

    This is a pure-Python fallback for the ORCA-style orbit feature used by
    GraphRNN/GDSS/CatFlow evaluation. Ego-small and Community-small are small
    enough for exact enumeration; larger graphs use deterministic sampling for
    the 3/4-node combinations to keep benchmark runs practical.
    """
    vec = torch.zeros(15, dtype=torch.float32)
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 0:
        return vec

    adjacency = {node: set(graph.neighbors(node)) for node in nodes}

    # Orbit 0: endpoint of a 2-node edge.
    vec[0] = 2.0 * float(graph.number_of_edges())

    def has_edge(a, b):
        return b in adjacency.get(a, ())

    def combo_iter(size, max_samples):
        total = 0
        if size == 3:
            total = n * (n - 1) * (n - 2) // 6
        elif size == 4:
            total = n * (n - 1) * (n - 2) * (n - 3) // 24
        if n <= int(max_exact_nodes) or total <= int(max_samples):
            return itertools.combinations(nodes, size), 1.0
        rng = random.Random(0)
        samples = max(1, int(max_samples))
        return (tuple(rng.sample(nodes, size)) for _ in range(samples)), float(total) / float(samples)

    triples, triple_scale = combo_iter(3, max_triplets)
    for triple in triples:
        edge_count = 0
        deg = {node: 0 for node in triple}
        for u, v in itertools.combinations(triple, 2):
            if has_edge(u, v):
                edge_count += 1
                deg[u] += 1
                deg[v] += 1
        if edge_count == 2:
            for value in deg.values():
                vec[2 if value == 2 else 1] += triple_scale
        elif edge_count == 3:
            vec[3] += 3.0 * triple_scale

    if n < 4:
        return vec / float(n)

    quadruplets, quad_scale = combo_iter(4, max_quadruplets)
    for quad in quadruplets:
        edge_count = 0
        deg = {node: 0 for node in quad}
        for u, v in itertools.combinations(quad, 2):
            if has_edge(u, v):
                edge_count += 1
                deg[u] += 1
                deg[v] += 1

        degree_values = sorted(deg.values())
        if edge_count == 3:
            if degree_values == [1, 1, 1, 3]:
                vec[4] += 3.0 * quad_scale
                vec[5] += 1.0 * quad_scale
            elif degree_values == [1, 1, 2, 2]:
                vec[6] += 2.0 * quad_scale
                vec[7] += 2.0 * quad_scale
        elif edge_count == 4:
            if degree_values == [2, 2, 2, 2]:
                vec[8] += 4.0 * quad_scale
            elif degree_values == [1, 2, 2, 3]:
                vec[9] += 1.0 * quad_scale
                vec[10] += 1.0 * quad_scale
                vec[11] += 2.0 * quad_scale
        elif edge_count == 5 and degree_values == [2, 2, 3, 3]:
            vec[12] += 2.0 * quad_scale
            vec[13] += 2.0 * quad_scale
        elif edge_count == 6:
            vec[14] += 4.0 * quad_scale

    return vec / float(n)


def _gaussian_l2_mmd(x, y, sigma=1.0):
    x = x.float()
    y = y.float()
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    gamma = 1.0 / (2.0 * float(sigma) * float(sigma))
    kxx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2)).mean()
    return max(float((kxx + kyy - 2.0 * kxy).item()), 0.0)


def _gaussian_emd_mmd(x, y, sigma=1.0, distance_scaling=1.0):
    x = x.float()
    y = y.float()
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    x_cdf = torch.cumsum(x, dim=1)
    y_cdf = torch.cumsum(y, dim=1)
    gamma = 1.0 / (2.0 * float(sigma) * float(sigma))
    scale = max(float(distance_scaling), 1.0e-12)

    def kernel(a, b):
        emd = torch.cdist(a, b, p=1) / scale
        return torch.exp(-gamma * emd.pow(2)).mean()

    value = kernel(x_cdf, x_cdf) + kernel(y_cdf, y_cdf) - 2.0 * kernel(x_cdf, y_cdf)
    return max(float(value.item()), 0.0)


def _wl_hist_features_for_train(train_graphs, metadata, final_stage, vocab_size):
    wl = WeisfeilerLeman(n_iter=int(metadata["n_wl_iters"]))
    wl.label_dict = metadata["label_dict"]
    wl.label_counter = [max(d.values()) + 1 if d else 1 for d in wl.label_dict]
    wl.is_fitted_ = True
    colors = wl.transform(train_graphs)
    return [
        _one_based_hist_from_ids(color_tensor[:, final_stage].tolist(), vocab_size)
        for color_tensor in colors
    ]


def _wl_hist_features_for_generated(graphs, vocab_size):
    features = []
    for graph in graphs:
        ids = [
            int(data["color_id"])
            for _node, data in graph.nodes(data=True)
            if "color_id" in data
        ]
        features.append(_one_based_hist_from_ids(ids, vocab_size))
    return features


def _stack_features(features):
    if not features:
        return torch.zeros((0, 1), dtype=torch.float32)
    return torch.stack([f.float() for f in features], dim=0)


def _median_bandwidth(x, y, max_points=512):
    z = torch.cat([x, y], dim=0)
    if z.size(0) < 2:
        return 1.0
    if z.size(0) > max_points:
        idx = torch.linspace(0, z.size(0) - 1, steps=max_points).long()
        z = z[idx]
    d = torch.pdist(z, p=2)
    d = d[d > 0]
    if d.numel() == 0:
        return 1.0
    return max(float(torch.median(d).item()), 1e-6)


def _gaussian_mmd(x, y):
    x = x.float()
    y = y.float()
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    sigma = _median_bandwidth(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return max(float((kxx + kyy - 2.0 * kxy).item()), 0.0)


def generic_mmd_metrics(train_graphs, generated_graphs, metadata, final_stage, max_graphs):
    if max_graphs is not None and max_graphs > 0:
        train_graphs = train_graphs[:int(max_graphs)]
        generated_graphs = generated_graphs[:int(max_graphs)]
    train_nx = [_graph_data_to_nx(graph) for graph in train_graphs]
    gen_nx = [_graph_data_to_nx(graph) for graph in generated_graphs]
    if not train_nx or not gen_nx:
        return {
            "mmd/train_graphs_used": len(train_nx),
            "mmd/generated_graphs_used": len(gen_nx),
        }

    max_degree = max(
        [0]
        + [max(_degrees(graph) or [0]) for graph in train_nx]
        + [max(_degrees(graph) or [0]) for graph in gen_nx]
    )
    degree_size = int(max_degree) + 1
    train_degree = [_degree_hist_feature(graph, degree_size) for graph in train_nx]
    gen_degree = [_degree_hist_feature(graph, degree_size) for graph in gen_nx]
    train_clustering = [_clustering_hist_feature(graph) for graph in train_nx]
    gen_clustering = [_clustering_hist_feature(graph) for graph in gen_nx]
    train_spectral = [_spectral_hist_feature(graph) for graph in train_nx]
    gen_spectral = [_spectral_hist_feature(graph) for graph in gen_nx]
    train_orbit = [_four_node_orbit_counts_feature(graph) for graph in train_nx]
    gen_orbit = [_four_node_orbit_counts_feature(graph) for graph in gen_nx]
    train_nodes = [_scalar_feature(graph.number_of_nodes()) for graph in train_nx]
    gen_nodes = [_scalar_feature(graph.number_of_nodes()) for graph in gen_nx]
    train_edges = [_scalar_feature(graph.number_of_edges()) for graph in train_nx]
    gen_edges = [_scalar_feature(graph.number_of_edges()) for graph in gen_nx]

    train_probs = _stage_color_probs(metadata, final_stage)
    if train_probs is not None:
        vocab_size = int(train_probs.numel())
    else:
        vocab_size = max(metadata["label_dict"][final_stage].values())
    train_wl = _wl_hist_features_for_train(train_graphs, metadata, final_stage, vocab_size)
    gen_wl = _wl_hist_features_for_generated(gen_nx, vocab_size)

    return {
        "mmd/train_graphs_used": len(train_nx),
        "mmd/generated_graphs_used": len(gen_nx),
        "mmd/degree": _gaussian_emd_mmd(_stack_features(gen_degree), _stack_features(train_degree)),
        "mmd/clustering": _gaussian_l2_mmd(_stack_features(gen_clustering), _stack_features(train_clustering), sigma=0.1),
        "mmd/spectral": _gaussian_mmd(_stack_features(gen_spectral), _stack_features(train_spectral)),
        "mmd/orbit": _gaussian_l2_mmd(_stack_features(gen_orbit), _stack_features(train_orbit), sigma=30.0),
        "mmd/orbit_4node_proxy": _gaussian_l2_mmd(_stack_features(gen_orbit), _stack_features(train_orbit), sigma=30.0),
        "mmd/node_count": _gaussian_mmd(_stack_features(gen_nodes), _stack_features(train_nodes)),
        "mmd/edge_count": _gaussian_mmd(_stack_features(gen_edges), _stack_features(train_edges)),
        "mmd/wl_color": _gaussian_mmd(_stack_features(gen_wl), _stack_features(train_wl)),
    }


def main():
    args = make_parser().parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root is not None:
        data_root = Path(args.data_root)
    elif args.dataset in {"ZINC", "ZINC250K"}:
        data_root = DATASETS_ROOT / args.dataset
    else:
        data_root = DATASETS_ROOT / args.dataset

    mode = args.mode
    if mode == "auto":
        mode = "molecule" if args.dataset in {"ZINC", "ZINC250K"} else "generic"

    print(f"Mode: {mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")

    train_graphs = load_train_graphs(
        args.dataset,
        data_root,
        args.zinc_subset,
        max_train_graphs=args.max_train_graphs,
    )

    if args.checkpoint:
        print(f"Loading WL metadata from checkpoint: {args.checkpoint}")
        metadata = load_checkpoint_metadata(args.checkpoint)
    elif args.ar_policy_checkpoint:
        print(f"Loading WL metadata from neural AR policy checkpoint: {args.ar_policy_checkpoint}")
        policy_ckpt = torch.load(args.ar_policy_checkpoint, map_location="cpu", weights_only=False)
        metadata = policy_ckpt.get("metadata")
        if metadata is None:
            raise RuntimeError(f"No metadata found in AR policy checkpoint: {args.ar_policy_checkpoint}")
    else:
        print("Fitting WL metadata from train split...")
        metadata = fit_metadata_from_graphs(train_graphs, args.n_wl_iters)

    final_stage = validate_final_stage(metadata, int(metadata["n_wl_iters"]))
    color_to_hash = build_color_id_to_iter1_descriptor(metadata["label_dict"], final_stage)
    color_probs = metadata["color_probs"][final_stage] if metadata.get("color_probs") else None
    color_specs = build_color_specs(color_to_hash, color_probs=color_probs)
    print(f"AR color specs: {len(color_specs)} final-stage WL colors with non-empty stubs")

    budget = resolve_budget(args, train_graphs, data_root, mode)
    if budget is None:
        print("Budget: off")
    elif "__default__" in budget:
        print(f"Budget: generic-degree cap={budget['__default__']} edge_weight={args.edge_weight}")
    else:
        print(f"Budget: molecule labels={len(budget)} edge_weight={args.edge_weight}")

    color_policy = None
    if args.ar_policy_checkpoint:
        print(f"Loading neural AR color policy: {args.ar_policy_checkpoint}")
        color_policy = load_neural_color_policy(
            args.ar_policy_checkpoint,
            device=args.ar_policy_device,
            temperature=args.ar_policy_temperature,
            empirical_weight_bias=args.ar_policy_empirical_bias,
        )

    generator = PairedStubARGenerator(
        color_specs,
        max_budget_by_label=budget,
        edge_weight_mode=args.edge_weight,
        new_node_prob=args.new_node_prob,
        require_connected=not args.allow_disconnected,
        max_restarts=args.max_restarts,
        max_steps_per_node=args.max_steps_per_node,
        color_policy=color_policy,
        rng=rng,
    )

    node_counts = sample_node_counts(metadata.get("train_n_nodes") or [], args.n_samples, rng)
    t0 = time.perf_counter()
    samples = generator.sample_many(node_counts)
    elapsed = time.perf_counter() - t0

    generated_graphs = [s.graph for s in samples if s.graph is not None]
    largest_component_graphs, largest_component_source_attempt_indices = largest_component_views(samples)
    generated_samples = [s for s in samples if s.graph is not None]
    largest_component_matches_raw = (
        len(largest_component_graphs) == len(generated_graphs)
        and all(
            int(sample.stats.get("cleanup_removed_nodes", 0)) == 0
            and int(sample.stats.get("cleanup_removed_edges", 0)) == 0
            for sample in generated_samples
        )
    )
    failure_counts = {}
    restart_sum = 0
    for s in samples:
        if s.graph is None:
            failure = s.stats.get("failure", "failed")
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
            nested = s.stats.get("failure_counts") or {}
            for k, v in nested.items():
                failure_counts[f"inner/{k}"] = failure_counts.get(f"inner/{k}", 0) + int(v)
        else:
            restart_sum += int(s.stats.get("restarts_used", 0))

    metrics = {
        "config/dataset": args.dataset,
        "config/mode": mode,
        "config/primary_graph_view": "raw",
        "config/secondary_graph_view": "isolates_removed_then_largest_connected_component",
        "config/largest_component_matches_raw": largest_component_matches_raw,
        "config/checkpoint": portable_path(args.checkpoint),
        "config/n_samples": int(args.n_samples),
        "config/final_stage": final_stage,
        "config/max_restarts": int(args.max_restarts),
        "config/max_steps_per_node": int(args.max_steps_per_node),
        "config/new_node_prob": float(args.new_node_prob),
        "config/budget_mode": args.budget_mode,
        "config/edge_weight": args.edge_weight,
        "config/ar_policy_checkpoint": portable_path(args.ar_policy_checkpoint),
        "config/ar_policy_temperature": float(args.ar_policy_temperature),
        "config/ar_policy_empirical_bias": float(args.ar_policy_empirical_bias),
        "timing/sample_seconds": elapsed,
        "sampling/target_n_samples": int(args.n_samples),
        "sampling/generated_graphs": len(generated_graphs),
        "sampling/largest_component_graphs": len(largest_component_graphs),
        "sampling/generation_fraction": len(generated_graphs) / max(args.n_samples, 1),
        "sampling/largest_component_survival_fraction": len(largest_component_graphs) / max(args.n_samples, 1),
        "sampling/failure_counts": failure_counts,
        "sampling/mean_restarts_success": restart_sum / max(len(generated_graphs), 1),
        "sampling/paired_stub_fraction": 1.0 if generated_graphs else 0.0,
    }
    raw_distribution_metrics = distribution_metrics(
        train_graphs,
        generated_graphs,
        metadata,
        final_stage,
        args.metric_max_train_graphs,
    )
    raw_graph_metrics = graph_metrics(
        generated_graphs,
        train_graphs=train_graphs,
        dataset=args.dataset,
    )
    if largest_component_matches_raw:
        largest_distribution_metrics = dict(raw_distribution_metrics)
        largest_graph_metrics = dict(raw_graph_metrics)
    else:
        largest_distribution_metrics = distribution_metrics(
            train_graphs,
            largest_component_graphs,
            metadata,
            final_stage,
            args.metric_max_train_graphs,
        )
        largest_graph_metrics = graph_metrics(
            largest_component_graphs,
            train_graphs=train_graphs,
            dataset=args.dataset,
        )
    raw_metric_suite = {**raw_graph_metrics, **raw_distribution_metrics}
    largest_metric_suite = {**largest_graph_metrics, **largest_distribution_metrics}
    metrics.update(raw_metric_suite)
    metrics.update(_prefixed("raw", raw_metric_suite))
    metrics.update(_prefixed("largest_component", largest_metric_suite))

    if mode == "molecule":
        atom_decoder = load_atom_decoder(root=data_root)
        smiles_cache = args.train_smiles_cache
        if smiles_cache is None:
            subset = args.zinc_subset or ("large" if args.dataset == "ZINC250K" else "small")
            smiles_cache = data_root / "cache" / f"train_smiles_{subset}.json"
        train_smiles = build_training_smiles(train_graphs, atom_decoder, cache_path=smiles_cache)
        mol_metrics = compute_mol_metrics(generated_graphs, train_smiles, atom_decoder, requested_total=args.n_samples)
        largest_mol_metrics = mol_metrics if largest_component_matches_raw else compute_mol_metrics(
            largest_component_graphs,
            train_smiles,
            atom_decoder,
            requested_total=args.n_samples,
        )
        metrics.update({
            "mol/n_total": mol_metrics["n_total"],
            "mol/n_requested": mol_metrics["n_requested"],
            "mol/n_valid": mol_metrics["n_valid"],
            "mol/n_unique": mol_metrics["n_unique"],
            "mol/n_novel": mol_metrics["n_novel"],
            "mol/validity": mol_metrics["validity"],
            "mol/validity_per_requested_attempt": mol_metrics["validity_per_requested_attempt"],
            "mol/validity_given_graph": mol_metrics["validity_given_graph"],
            "mol/uniqueness": mol_metrics["uniqueness"],
            "mol/novelty": mol_metrics["novelty"],
        })
        for key, value in mol_metrics.items():
            if not isinstance(value, list):
                metrics[f"raw/mol/{key}"] = value
        for key, value in largest_mol_metrics.items():
            if not isinstance(value, list):
                metrics[f"largest_component/mol/{key}"] = value
        (output_dir / "valid_smiles.txt").write_text("\n".join(mol_metrics["valid_smiles"]))
        (output_dir / "unique_smiles.txt").write_text("\n".join(mol_metrics["unique_smiles"]))
        (output_dir / "novel_smiles.txt").write_text("\n".join(mol_metrics["novel_smiles"]))
        (output_dir / "largest_component_valid_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["valid_smiles"])
        )
        (output_dir / "largest_component_unique_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["unique_smiles"])
        )
        (output_dir / "largest_component_novel_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["novel_smiles"])
        )
    else:
        raw_mmd_metrics = generic_mmd_metrics(
            train_graphs,
            generated_graphs,
            metadata,
            final_stage,
            args.mmd_max_graphs,
        )
        largest_mmd_metrics = dict(raw_mmd_metrics) if largest_component_matches_raw else generic_mmd_metrics(
            train_graphs,
            largest_component_graphs,
            metadata,
            final_stage,
            args.mmd_max_graphs,
        )
        metrics.update(raw_mmd_metrics)
        metrics.update(_prefixed("raw", raw_mmd_metrics))
        metrics.update(_prefixed("largest_component", largest_mmd_metrics))

    torch.save(
        {
            "node_counts": node_counts,
            "sample_stats": [s.stats for s in samples],
            "generated_graphs": generated_graphs,
            "raw_generated_graphs": generated_graphs,
            "largest_component_graphs": largest_component_graphs,
            "largest_component_source_attempt_indices": largest_component_source_attempt_indices,
        },
        output_dir / "ar_samples.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    print("\n--- AR WL-Witness Summary ---")
    print(f"Requested samples: {args.n_samples}")
    print(f"Generated graphs:  {len(generated_graphs)} ({len(generated_graphs) / max(args.n_samples, 1):.2%})")
    print(
        f"Raw components:    {metrics.get('graph/raw_mean_component_count', 0.0):.2f} mean; "
        f"{metrics.get('graph/raw_multiple_components_fraction', 0.0):.2%} disconnected"
    )
    print(
        f"Raw isolates:      {metrics.get('graph/raw_mean_isolated_nodes', 0.0):.2f} mean; "
        f"largest component {metrics.get('graph/raw_mean_largest_component_fraction', 0.0):.2%} of nodes"
    )
    print(f"Paired stubs:      100.00% for every accepted graph")
    if mode == "molecule":
        print(f"RDKit-valid mols:  {metrics['mol/n_valid']} ({metrics['mol/validity']:.2%})")
        print(f"Unique mols:       {metrics['mol/n_unique']} ({metrics['mol/uniqueness']:.2%})")
        print(f"Novel mols:        {metrics['mol/n_novel']} ({metrics['mol/novelty']:.2%})")
    else:
        print(f"After cleanup:     {metrics['graph/after_cleanup']}")
        print(f"Connected:         {metrics['graph/connected']} ({metrics['graph/connected_fraction']:.2%})")
        print(f"MMD degree:        {metrics.get('mmd/degree', 0.0):.6f}")
        print(f"MMD clustering:    {metrics.get('mmd/clustering', 0.0):.6f}")
    print(f"Node-count TV:     {metrics.get('dist/node_count/tv', 0.0):.4f}")
    print(f"Edge-count TV:     {metrics.get('dist/edge_count/tv', 0.0):.4f}")
    print(f"Degree TV:         {metrics.get('dist/degree/tv', 0.0):.4f}")
    print(f"WL-color TV:       {metrics.get('dist/wl_color/tv', 0.0):.4f}")
    print(f"Failures:          {failure_counts}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()

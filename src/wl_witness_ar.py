"""WL witness specifications and constrained graph-sampling primitives."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch


def bond_order(edge_type: int, mode: str = "unit") -> int:
    if mode != "bond_order":
        return 1
    edge_type = int(edge_type)
    if edge_type in {1, 2, 3}:
        return edge_type
    return 1


def parse_color_descriptor(cid: int, desc: str):
    """Parse an iter-1 WL descriptor into a center label and half-edge stubs."""
    if not isinstance(desc, str) or "_" not in desc:
        return None
    head, rest = desc.split("_", 1)
    try:
        center = int(head)
    except ValueError:
        return None
    if not rest:
        return None

    stubs = []
    for item in rest.split(","):
        if ":" not in item:
            return None
        e_str, nbr_str = item.split(":", 1)
        try:
            stubs.append((int(e_str), int(nbr_str)))
        except ValueError:
            return None
    if not stubs:
        return None
    return ColorSpec(
        color_id=int(cid),
        center_label=int(center),
        stubs=tuple(stubs),
        stub_counts=Counter(stubs),
        weight=1.0,
    )


@dataclass(frozen=True)
class ColorSpec:
    color_id: int
    center_label: int
    stubs: tuple[tuple[int, int], ...]
    stub_counts: Counter
    weight: float

    @property
    def degree(self) -> int:
        return len(self.stubs)


@dataclass
class ARSample:
    graph: nx.Graph | None
    stats: dict[str, Any]


def build_color_specs(color_to_hash: dict[str, str], color_probs=None):
    """Build weighted WL-color specs from final-stage descriptor metadata."""
    specs = []
    for cid_str, desc in color_to_hash.items():
        try:
            cid = int(cid_str)
        except (TypeError, ValueError):
            continue
        spec = parse_color_descriptor(cid, desc)
        if spec is None:
            continue
        weight = 1.0
        if color_probs is not None and cid > 0 and cid - 1 < len(color_probs):
            weight = float(color_probs[cid - 1])
        if weight <= 0.0:
            continue
        specs.append(ColorSpec(
            color_id=spec.color_id,
            center_label=spec.center_label,
            stubs=spec.stubs,
            stub_counts=spec.stub_counts,
            weight=weight,
        ))
    return specs


class PairedStubARGenerator:
    """
    Autoregressive WL-witness generator.

    The generator samples final WL colors as node types, then repeatedly picks
    one unresolved half-edge stub and immediately pairs it with a reciprocal
    stub on either an existing compatible node or a newly-created compatible
    WL-color node. Accepted samples have no leftover stubs, so the realized
    witness has paired_stub_fraction = 1.0 by construction.
    """

    def __init__(
        self,
        color_specs: list[ColorSpec],
        *,
        max_budget_by_label: dict[int, int] | None = None,
        edge_weight_mode: str = "unit",
        new_node_prob: float = 0.70,
        require_connected: bool = True,
        max_restarts: int = 200,
        max_steps_per_node: int = 20,
        color_policy=None,
        rng: random.Random | None = None,
    ):
        if not color_specs:
            raise ValueError("Need at least one WL color with non-empty stubs.")
        self.color_specs = list(color_specs)
        self.max_budget_by_label = max_budget_by_label or {}
        self.edge_weight_mode = edge_weight_mode
        self.new_node_prob = float(new_node_prob)
        self.require_connected = bool(require_connected)
        self.max_restarts = int(max_restarts)
        self.max_steps_per_node = int(max_steps_per_node)
        self.color_policy = color_policy
        self.rng = rng or random.Random()

        self.seed_specs = [s for s in self.color_specs if s.degree > 0]
        self.specs_by_needed_stub = defaultdict(list)
        self.specs_by_center = defaultdict(list)
        for spec in self.color_specs:
            self.specs_by_center[int(spec.center_label)].append(spec)
            for (edge_type, nbr_label), count in spec.stub_counts.items():
                if count > 0:
                    # A source stub (a,e,b) needs a reciprocal node centered at
                    # b with local stub (e,a).
                    key = (int(spec.center_label), int(edge_type), int(nbr_label))
                    self.specs_by_needed_stub[key].append(spec)

    def _weighted_choice(self, specs: list[ColorSpec]) -> ColorSpec:
        total = sum(max(0.0, float(s.weight)) for s in specs)
        if total <= 0.0:
            return self.rng.choice(specs)
        r = self.rng.random() * total
        acc = 0.0
        for spec in specs:
            acc += max(0.0, float(spec.weight))
            if acc >= r:
                return spec
        return specs[-1]

    def _filter_policy_specs(self, specs: list[ColorSpec], context: dict[str, Any] | None):
        if self.color_policy is None or not hasattr(self.color_policy, "filter_specs"):
            return specs
        filtered = self.color_policy.filter_specs(specs, context=context)
        return filtered if filtered else specs

    def _choose_spec(
        self,
        specs: list[ColorSpec],
        prefix_color_ids: list[int],
        context: dict[str, Any] | None = None,
    ) -> ColorSpec:
        specs = self._filter_policy_specs(specs, context=context)
        if self.color_policy is not None:
            try:
                return self.color_policy.choose(specs, prefix_color_ids, self.rng, context=context)
            except TypeError:
                return self.color_policy.choose(specs, prefix_color_ids, self.rng)
        return self._weighted_choice(specs)

    def _budget_ok(self, label: int, used: int, edge_type: int) -> bool:
        cap = self.max_budget_by_label.get(
            int(label),
            self.max_budget_by_label.get("__default__"),
        )
        if cap is None:
            return True
        return used + bond_order(edge_type, self.edge_weight_mode) <= int(cap)

    def _add_node(self, G: nx.Graph, spec: ColorSpec, node_stubs, valence_used) -> int:
        node = G.number_of_nodes()
        G.add_node(
            node,
            label=int(spec.center_label),
            color_id=int(spec.color_id),
            target_stubs=list(spec.stubs),
        )
        node_stubs.append(list(spec.stubs))
        valence_used.append(0)
        return node

    def _all_unresolved(self, node_stubs):
        out = []
        for node, stubs in enumerate(node_stubs):
            for idx, (edge_type, nbr_label) in enumerate(stubs):
                out.append((node, idx, int(edge_type), int(nbr_label)))
        return out

    def _find_existing_matches(self, G, node_stubs, valence_used, src_node, edge_type, nbr_label):
        src_label = int(G.nodes[src_node]["label"])
        matches = []
        for dst_node, stubs in enumerate(node_stubs):
            if dst_node == src_node:
                continue
            if G.has_edge(src_node, dst_node):
                continue
            if int(G.nodes[dst_node]["label"]) != int(nbr_label):
                continue
            if not self._budget_ok(src_label, valence_used[src_node], edge_type):
                continue
            if not self._budget_ok(nbr_label, valence_used[dst_node], edge_type):
                continue
            for dst_idx, (dst_edge, dst_nbr) in enumerate(stubs):
                if int(dst_edge) == int(edge_type) and int(dst_nbr) == src_label:
                    matches.append((dst_node, dst_idx))
                    break
        return matches

    def _new_node_candidates(self, label, edge_type, reciprocal_nbr):
        # Need a color centered at `label` that contains stub
        # (edge_type, reciprocal_nbr).
        key = (int(label), int(edge_type), int(reciprocal_nbr))
        return self.specs_by_needed_stub.get(key, [])

    def _consume_pair(self, G, node_stubs, valence_used, src_node, src_idx, dst_node, dst_idx, edge_type):
        # Pop larger index first only when both stubs are on the same node.
        # Self-pairing is normally disallowed, but this keeps the helper safe.
        if src_node == dst_node and dst_idx > src_idx:
            dst_idx -= 1
        src_stub = node_stubs[src_node].pop(src_idx)
        dst_stub = node_stubs[dst_node].pop(dst_idx)
        src_label = int(G.nodes[src_node]["label"])
        dst_label = int(G.nodes[dst_node]["label"])
        if int(src_stub[0]) != int(edge_type) or int(dst_stub[0]) != int(edge_type):
            raise RuntimeError("internal AR stub mismatch")
        G.add_edge(src_node, dst_node, bond_type=int(edge_type))
        cost = bond_order(edge_type, self.edge_weight_mode)
        valence_used[src_node] += cost
        valence_used[dst_node] += cost
        return src_label, dst_label

    def sample_one(self, target_nodes: int) -> ARSample:
        target_nodes = max(2, int(target_nodes))
        failure_counts = Counter()

        for restart in range(self.max_restarts):
            G = nx.Graph()
            node_stubs = []
            valence_used = []
            prefix_color_ids = []

            seed_context = {
                "node_index": 0,
                "target_nodes": target_nodes,
                "choice_type": "seed",
            }
            seed_spec = self._choose_spec(self.seed_specs, prefix_color_ids, context=seed_context)
            self._add_node(G, seed_spec, node_stubs, valence_used)
            prefix_color_ids.append(int(seed_spec.color_id))

            max_steps = max(10, self.max_steps_per_node * target_nodes)
            failure = None
            paired_edges = 0
            for _step in range(max_steps):
                unresolved = self._all_unresolved(node_stubs)
                if not unresolved:
                    if G.number_of_edges() == 0:
                        failure = "no_edges"
                        break
                    if self.require_connected and not nx.is_connected(G):
                        failure = "disconnected"
                        break
                    stats = {
                        "success": True,
                        "failure": None,
                        "restarts_used": restart,
                        "n_nodes": G.number_of_nodes(),
                        "target_nodes": target_nodes,
                        "n_edges": G.number_of_edges(),
                        "paired_stub_fraction": 1.0,
                        "paired_edges": paired_edges,
                    }
                    return ARSample(G, stats)

                src_node, src_idx, edge_type, nbr_label = self.rng.choice(unresolved)
                src_label = int(G.nodes[src_node]["label"])

                existing = self._find_existing_matches(
                    G, node_stubs, valence_used, src_node, edge_type, nbr_label
                )
                can_add = G.number_of_nodes() < target_nodes
                use_new = can_add and (
                    not existing
                    or self.rng.random() < self.new_node_prob * (1.0 - G.number_of_nodes() / max(target_nodes, 1))
                )

                if use_new:
                    candidates = self._new_node_candidates(nbr_label, edge_type, src_label)
                    candidates = [
                        c for c in candidates
                        if self._budget_ok(nbr_label, 0, edge_type)
                    ]
                    if candidates:
                        new_node_index = G.number_of_nodes()
                        choice_context = {
                            "node_index": new_node_index,
                            "target_nodes": target_nodes,
                            "choice_type": "new_node",
                            "source_node": src_node,
                            "source_label": src_label,
                            "edge_type": edge_type,
                            "required_center_label": nbr_label,
                        }
                        dst_spec = self._choose_spec(candidates, prefix_color_ids, context=choice_context)
                        dst_node = self._add_node(G, dst_spec, node_stubs, valence_used)
                        prefix_color_ids.append(int(dst_spec.color_id))
                        # Find the reciprocal stub we just required.
                        dst_idx = None
                        for j, (dst_edge, dst_nbr) in enumerate(node_stubs[dst_node]):
                            if int(dst_edge) == int(edge_type) and int(dst_nbr) == src_label:
                                dst_idx = j
                                break
                        if dst_idx is None:
                            failure = "new_node_missing_reciprocal"
                            break
                        self._consume_pair(
                            G, node_stubs, valence_used,
                            src_node, src_idx, dst_node, dst_idx, edge_type,
                        )
                        paired_edges += 1
                        continue

                if existing:
                    dst_node, dst_idx = self.rng.choice(existing)
                    self._consume_pair(
                        G, node_stubs, valence_used,
                        src_node, src_idx, dst_node, dst_idx, edge_type,
                    )
                    paired_edges += 1
                    continue

                failure = "no_compatible_reciprocal"
                break

            if failure is None:
                failure = "max_steps"
            failure_counts[failure] += 1

        return ARSample(None, {
            "success": False,
            "failure": "restarts_exhausted",
            "failure_counts": dict(failure_counts),
            "target_nodes": target_nodes,
            "paired_stub_fraction": 0.0,
        })

    def sample_many(self, node_counts: list[int]):
        return [self.sample_one(n) for n in node_counts]


def sample_node_counts(train_n_nodes, n_samples: int, rng: random.Random):
    if not train_n_nodes:
        return [20] * int(n_samples)
    return [int(rng.choice(train_n_nodes)) for _ in range(int(n_samples))]


def sample_graph_budgets(metadata, n_samples: int, rng: random.Random):
    """Sample correlated node/edge budgets from the training distribution."""
    budgets = list((metadata or {}).get("train_graph_budgets") or [])
    if not budgets:
        nodes = list((metadata or {}).get("train_n_nodes") or [])
        edges = list((metadata or {}).get("train_n_edges") or [])
        if nodes and len(nodes) == len(edges):
            budgets = [
                {"nodes": int(n), "edges": int(e)}
                for n, e in zip(nodes, edges)
            ]
    if not budgets:
        return [
            {"nodes": int(n), "edges": None}
            for n in sample_node_counts(
                list((metadata or {}).get("train_n_nodes") or []),
                n_samples,
                rng,
            )
        ]
    normalized = []
    for item in budgets:
        if item.get("nodes") is None or item.get("edges") is None:
            continue
        normalized_item = dict(item)
        normalized_item["nodes"] = max(2, int(item["nodes"]))
        normalized_item["edges"] = max(1, int(item["edges"]))
        normalized.append(normalized_item)
    if not normalized:
        raise ValueError("Training graph budgets contain no usable node/edge pairs")
    return [dict(rng.choice(normalized)) for _ in range(int(n_samples))]

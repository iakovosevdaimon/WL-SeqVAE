"""Autoregressive graph generation from trained Ring WL-VAE checkpoints."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch

from ring_vae_actions import ACTION_CONNECT_EXISTING, ACTION_CREATE_NODE, ACTION_STOP
from ring_vae_model import RingWLVAE
from wl_color_hierarchy import build_color_hierarchy
from wl_witness_ar import ARSample, ColorSpec


class DecoderSamplingError(Exception):
    pass


def sample_index_from_logits(logits, rng: random.Random, *, temperature: float = 1.0, greedy: bool = False):
    logits = logits.detach().float().cpu()
    if logits.numel() == 0:
        raise ValueError("Cannot sample from empty logits.")
    if greedy:
        return int(torch.argmax(logits).item())
    temperature = max(float(temperature), 1.0e-6)
    probs = torch.softmax(logits / temperature, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        return rng.randrange(int(logits.numel()))
    r = rng.random()
    acc = 0.0
    for idx, prob in enumerate(probs.tolist()):
        acc += float(prob)
        if r <= acc:
            return idx
    return int(logits.numel()) - 1


@dataclass
class RingWLVAEGenerator:
    model: RingWLVAE
    metadata: dict[str, Any]
    final_stage: int
    color_specs: list[ColorSpec]
    temperature: float = 1.0
    edge_temperature: float = 1.0
    action_temperature: float = 1.0
    plan_temperature: float = 1.0
    plan_prior_weight: float = 0.0
    plan_greedy: bool = False
    empirical_prior_weight: float = 0.0
    adaptive_prior_weight: float = 0.0
    adaptive_prior_pseudocount: float = 1000.0
    greedy: bool = False
    use_learned_actions: bool | None = None
    use_learned_stop: bool | None = None
    use_learned_closure: bool | None = None
    require_connected: bool = True
    require_complete: bool = True
    connect_existing_first: bool = True
    use_stub_closure: bool = True
    max_restarts: int = 200
    max_rings: int = 12
    max_nodes: int | None = None
    node_budget_factor: float = 1.0
    node_budget_margin: int = 0
    max_closure_steps_per_node: int = 20
    constraint_kind: str = "generic"
    molecule_constraint_profile: dict[str, Any] | None = None
    enforce_edge_budget: bool = True
    use_budget_repair: bool = True
    rng: random.Random | None = None

    def __post_init__(self):
        self.rng = self.rng or random.Random()
        self.color_specs_by_id = {int(spec.color_id): spec for spec in self.color_specs}
        self.color_hierarchy = build_color_hierarchy(self.metadata, self.final_stage)
        self.seed_color_ids = list(self.model.seed_color_ids) or [
            int(spec.color_id) for spec in self.color_specs if int(spec.degree) > 0
        ]
        self._log_prior_cache = {}
        self._adaptive_counts = Counter()
        self._adaptive_total = 0
        self._adaptive_cache = {}
        self._active_graph = None
        self._active_target_nodes = None
        self._active_target_edges = None
        self._active_constraint_profile = {}
        self._active_block_slots = []
        self._active_block_sizes = []
        self._active_block_remaining = Counter()
        self._active_degree_targets = []
        has_action_head = bool(getattr(self.model, "use_action_head", False))
        if self.use_learned_actions is None:
            self.use_learned_actions = has_action_head
        else:
            self.use_learned_actions = bool(self.use_learned_actions) and has_action_head
        if self.use_learned_stop is None:
            self.use_learned_stop = has_action_head
        else:
            self.use_learned_stop = bool(self.use_learned_stop) and has_action_head
        if self.use_learned_closure is None:
            self.use_learned_closure = has_action_head
        else:
            self.use_learned_closure = bool(self.use_learned_closure) and has_action_head

    def _requirements(self, final_color: int):
        counts = self.color_hierarchy.neighbor_counts[int(self.final_stage)].get(int(final_color), Counter())
        return Counter({(int(edge), int(prev)): int(count) for (edge, prev), count in counts.items()})

    def _parent_prev_color(self, final_color: int):
        parent = self.color_hierarchy.parent(self.final_stage, int(final_color))
        return int(parent) if parent is not None else 0

    def _add_node(self, graph, nodes, *, final_color: int, ring: int, preferred_block: int | None = None):
        spec = self.color_specs_by_id.get(int(final_color))
        if spec is None:
            raise DecoderSamplingError(f"Missing ColorSpec for generated color {final_color}")
        node_id = len(nodes)
        prev_color = self._parent_prev_color(final_color)
        nodes.append({
            "final_color": int(final_color),
            "prev_color": int(prev_color),
            "label": int(spec.center_label),
            "ring": int(ring),
            "required": self._requirements(final_color),
            "degree": int(spec.degree),
        })
        node_attrs = {"label": int(spec.center_label), "color_id": int(final_color)}
        if self.constraint_kind == "sbm":
            available = [
                int(block)
                for block, remaining in self._active_block_remaining.items()
                if int(remaining) > 0
            ]
            if preferred_block in available:
                block = int(preferred_block)
            elif available:
                largest = max(int(self._active_block_remaining[item]) for item in available)
                block = self.rng.choice([
                    item for item in available
                    if int(self._active_block_remaining[item]) == largest
                ])
            else:
                block = int(self._active_block_slots[node_id]) if node_id < len(self._active_block_slots) else 0
            node_attrs["budget_block"] = block
            if self._active_block_remaining.get(block, 0) > 0:
                self._active_block_remaining[block] -= 1
        if node_id < len(self._active_degree_targets):
            node_attrs["budget_target_degree"] = int(self._active_degree_targets[node_id])
        graph.add_node(node_id, **node_attrs)
        return node_id

    def _constraint_edge_counts(self, graph):
        intra = 0
        inter = 0
        for u, v in graph.edges():
            if graph.nodes[int(u)].get("budget_block") == graph.nodes[int(v)].get("budget_block"):
                intra += 1
            else:
                inter += 1
        return intra, inter

    def _normalized_sbm_block_sizes(self, target_nodes: int, preferred):
        target_nodes = max(2, int(target_nodes))
        preferred = [int(value) for value in list(preferred or []) if int(value) > 0]
        if (
            sum(preferred) == target_nodes
            and 2 <= len(preferred) <= 5
            and all(20 <= value <= 40 for value in preferred)
        ):
            return preferred
        feasible_min = max(2, int(math.ceil(target_nodes / 40.0)))
        feasible_max = min(5, max(2, int(target_nodes // 20)))
        preferred_blocks = len(preferred)
        if feasible_min <= feasible_max:
            n_blocks = min(max(preferred_blocks or feasible_min, feasible_min), feasible_max)
        else:
            n_blocks = min(5, max(2, preferred_blocks or round(target_nodes / 30.0)))
        base, extra = divmod(target_nodes, n_blocks)
        return [base + int(block < extra) for block in range(n_blocks)]

    def _prepare_constraint_state(self, target_nodes: int, target_edges: int | None, profile):
        profile = dict(profile or {})
        self._active_constraint_profile = profile
        degree_sequence = [int(value) for value in profile.get("degree_sequence", [])]
        if degree_sequence:
            degree_sequence = sorted(degree_sequence, reverse=True)
            if len(degree_sequence) < int(target_nodes):
                degree_sequence.extend([degree_sequence[-1]] * (int(target_nodes) - len(degree_sequence)))
            self._active_degree_targets = degree_sequence[: int(target_nodes)]
            self.rng.shuffle(self._active_degree_targets)
        elif target_edges is not None:
            average = max(1, int(round(2.0 * int(target_edges) / max(int(target_nodes), 1))))
            self._active_degree_targets = [average] * int(target_nodes)
        else:
            self._active_degree_targets = []

        self._active_block_slots = []
        self._active_block_sizes = []
        if self.constraint_kind == "sbm":
            block_sizes = self._normalized_sbm_block_sizes(
                target_nodes,
                profile.get("block_sizes", []),
            )
            slots = [block for block, size in enumerate(block_sizes) for _ in range(int(size))]
            self._active_block_slots = slots
            self._active_block_sizes = list(block_sizes)
            total_profile_edges = int(profile.get("intra_edges", 0)) + int(profile.get("inter_edges", 0))
            if target_edges is not None and total_profile_edges > 0:
                intra_fraction = float(profile.get("intra_edges", 0)) / float(total_profile_edges)
                intra_edges = int(round(int(target_edges) * intra_fraction))
                inter_edges = int(target_edges) - intra_edges
            elif target_edges is not None:
                inter_edges = max(len(block_sizes) - 1, int(round(0.05 * int(target_edges))))
                intra_edges = int(target_edges) - inter_edges
            else:
                intra_edges = int(profile.get("intra_edges", 10**9))
                inter_edges = int(profile.get("inter_edges", 10**9))
            inter_edges = max(len(block_sizes) - 1, inter_edges)
            if target_edges is not None and intra_edges + inter_edges > int(target_edges):
                intra_edges = max(0, int(target_edges) - inter_edges)
            self._active_constraint_profile.update({
                "block_sizes": block_sizes,
                "intra_edges": int(intra_edges),
                "inter_edges": int(inter_edges),
            })

    def _reset_constraint_attempt(self):
        self._active_block_remaining = Counter({
            int(block): int(size)
            for block, size in enumerate(self._active_block_sizes)
        })

    def _preferred_sbm_child_block(self, graph, parent: int):
        parent_block = int(graph.nodes[int(parent)].get("budget_block", 0))
        intra, inter = self._constraint_edge_counts(graph)
        target_intra = int(self._active_constraint_profile.get("intra_edges", 10**9))
        target_inter = int(self._active_constraint_profile.get("inter_edges", 10**9))
        if self._active_block_remaining.get(parent_block, 0) > 0 and intra < target_intra:
            return parent_block
        alternatives = [
            int(block)
            for block, remaining in self._active_block_remaining.items()
            if int(block) != parent_block and int(remaining) > 0
        ]
        if alternatives and inter < target_inter:
            largest = max(int(self._active_block_remaining[block]) for block in alternatives)
            return self.rng.choice([
                block for block in alternatives
                if int(self._active_block_remaining[block]) == largest
            ])
        return parent_block

    def _constraint_allows_edge(self, graph, nodes, u: int, v: int, edge_type: int):
        active_target_edges = getattr(self, "_active_target_edges", None)
        if getattr(self, "enforce_edge_budget", True) and active_target_edges is not None:
            if graph.number_of_edges() >= int(active_target_edges):
                return False
        constraint_kind = getattr(self, "constraint_kind", "generic")
        if constraint_kind == "molecule":
            profile = dict(getattr(self, "molecule_constraint_profile", None) or {})
            left_label = int(nodes[int(u)]["label"])
            right_label = int(nodes[int(v)]["label"])
            allowed = {
                tuple(int(value) for value in row)
                for row in profile.get("allowed_bonds", [])
            }
            bond_key = (min(left_label, right_label), max(left_label, right_label), int(edge_type))
            if allowed and bond_key not in allowed:
                return False
            max_degree = {
                int(label): int(value)
                for label, value in dict(profile.get("max_degree_by_label") or {}).items()
            }
            if graph.degree(int(u)) >= max_degree.get(left_label, 10**9):
                return False
            if graph.degree(int(v)) >= max_degree.get(right_label, 10**9):
                return False
            max_bond_order = {
                int(label): int(value)
                for label, value in dict(profile.get("max_bond_order_by_label") or {}).items()
            }
            left_order = sum(int(data.get("bond_type", 1)) for *_uv, data in graph.edges(int(u), data=True))
            right_order = sum(int(data.get("bond_type", 1)) for *_uv, data in graph.edges(int(v), data=True))
            if left_order + int(edge_type) > max_bond_order.get(left_label, 10**9):
                return False
            if right_order + int(edge_type) > max_bond_order.get(right_label, 10**9):
                return False
        if constraint_kind == "sbm":
            left_block = graph.nodes[int(u)].get("budget_block")
            right_block = graph.nodes[int(v)].get("budget_block")
            intra, inter = self._constraint_edge_counts(graph)
            if left_block == right_block:
                if intra >= int(getattr(self, "_active_constraint_profile", {}).get("intra_edges", 10**9)):
                    return False
            elif inter >= int(getattr(self, "_active_constraint_profile", {}).get("inter_edges", 10**9)):
                return False
        return True

    def _can_connect(self, nodes, u: int, v: int, edge_type: int):
        left = nodes[int(u)]
        right = nodes[int(v)]
        return (
            left["required"].get((int(edge_type), int(right["prev_color"])), 0) > 0
            and right["required"].get((int(edge_type), int(left["prev_color"])), 0) > 0
        )

    def _possible_edge_types(self, nodes, u: int, v: int):
        left = nodes[int(u)]
        right = nodes[int(v)]
        out = []
        for edge_type, prev_color in left["required"]:
            if int(prev_color) != int(right["prev_color"]):
                continue
            if right["required"].get((int(edge_type), int(left["prev_color"])), 0) > 0:
                out.append(int(edge_type))
        return sorted(set(out))

    def _connect(self, graph, nodes, u: int, v: int, edge_type: int):
        if (
            u == v
            or graph.has_edge(u, v)
            or not self._can_connect(nodes, u, v, edge_type)
            or not self._constraint_allows_edge(graph, nodes, u, v, edge_type)
        ):
            return False
        graph.add_edge(int(u), int(v), bond_type=int(edge_type))
        if self.constraint_kind == "planar":
            is_planar, _embedding = nx.check_planarity(graph)
            if not is_planar:
                graph.remove_edge(int(u), int(v))
                return False
        left_key = (int(edge_type), int(nodes[int(v)]["prev_color"]))
        right_key = (int(edge_type), int(nodes[int(u)]["prev_color"]))
        nodes[int(u)]["required"][left_key] -= 1
        nodes[int(v)]["required"][right_key] -= 1
        if nodes[int(u)]["required"][left_key] <= 0:
            del nodes[int(u)]["required"][left_key]
        if nodes[int(v)]["required"][right_key] <= 0:
            del nodes[int(v)]["required"][right_key]
        return True

    def _force_budget_edge(self, graph, nodes, u: int, v: int, edge_type: int = 1):
        if u == v or graph.has_edge(int(u), int(v)):
            return False
        if not self._constraint_allows_edge(graph, nodes, u, v, edge_type):
            return False
        graph.add_edge(int(u), int(v), bond_type=int(edge_type), budget_repair=True)
        if self.constraint_kind == "planar":
            is_planar, _embedding = nx.check_planarity(graph)
            if not is_planar:
                graph.remove_edge(int(u), int(v))
                return False
        left_key = (int(edge_type), int(nodes[int(v)]["prev_color"]))
        right_key = (int(edge_type), int(nodes[int(u)]["prev_color"]))
        for node_id, key in ((int(u), left_key), (int(v), right_key)):
            if nodes[node_id]["required"].get(key, 0) > 0:
                nodes[node_id]["required"][key] -= 1
                if nodes[node_id]["required"][key] <= 0:
                    del nodes[node_id]["required"][key]
        return True

    def _budget_repair(self, graph, nodes):
        if (
            not self.use_budget_repair
            or self.constraint_kind not in {"planar", "sbm"}
            or self._active_target_edges is None
        ):
            return 0
        added = 0
        target = int(self._active_target_edges)
        while graph.number_of_edges() < target:
            current_intra, current_inter = self._constraint_edge_counts(graph)
            target_intra = int(self._active_constraint_profile.get("intra_edges", target))
            target_inter = int(self._active_constraint_profile.get("inter_edges", target))
            candidates = []
            for u in range(len(nodes)):
                for v in range(u + 1, len(nodes)):
                    if graph.has_edge(u, v):
                        continue
                    left_target = (
                        int(self._active_degree_targets[u])
                        if u < len(self._active_degree_targets)
                        else target
                    )
                    right_target = (
                        int(self._active_degree_targets[v])
                        if v < len(self._active_degree_targets)
                        else target
                    )
                    left_deficit = left_target - int(graph.degree(u))
                    right_deficit = right_target - int(graph.degree(v))
                    degree_improvement = (
                        abs(left_deficit) + abs(right_deficit)
                        - abs(left_deficit - 1) - abs(right_deficit - 1)
                    )
                    category_priority = 0.0
                    if self.constraint_kind == "sbm":
                        same_block = (
                            graph.nodes[u].get("budget_block")
                            == graph.nodes[v].get("budget_block")
                        )
                        if same_block and current_intra < target_intra:
                            category_priority = (target_intra - current_intra) / max(target_intra, 1)
                        elif not same_block and current_inter < target_inter:
                            category_priority = (target_inter - current_inter) / max(target_inter, 1)
                        else:
                            continue
                    candidates.append((
                        category_priority,
                        degree_improvement,
                        left_deficit + right_deficit,
                        self.rng.random(),
                        u,
                        v,
                    ))
            if not candidates:
                break
            progress = False
            for _category, _improvement, _deficit, _tie, u, v in sorted(candidates, reverse=True):
                if self._force_budget_edge(graph, nodes, u, v, edge_type=1):
                    added += 1
                    progress = True
                    break
            if not progress:
                break
        return added

    def _node_budget_repair(self, graph, nodes):
        if (
            not self.use_budget_repair
            or self.constraint_kind not in {"planar", "sbm"}
            or self._active_target_nodes is None
        ):
            return 0
        added = 0
        target_nodes = int(self._active_target_nodes)
        while len(nodes) < target_nodes:
            if (
                self._active_target_edges is not None
                and graph.number_of_edges() >= int(self._active_target_edges)
            ):
                break
            node_index = len(nodes)
            desired_degree = (
                int(self._active_degree_targets[node_index])
                if node_index < len(self._active_degree_targets)
                else 1
            )
            color_candidates = [
                int(color_id)
                for color_id in self.seed_color_ids
                if int(color_id) in self.color_specs_by_id
                and int(self.color_specs_by_id[int(color_id)].degree) > 0
            ]
            if not color_candidates:
                break
            color_id = min(
                color_candidates,
                key=lambda item: (
                    abs(int(self.color_specs_by_id[item].degree) - desired_degree),
                    self.rng.random(),
                ),
            )
            parents = list(range(len(nodes)))
            self.rng.shuffle(parents)
            parents.sort(
                key=lambda parent: (
                    (
                        int(self._active_degree_targets[parent]) - int(graph.degree(parent))
                        if parent < len(self._active_degree_targets)
                        else 0
                    ),
                    -int(graph.degree(parent)),
                ),
                reverse=True,
            )
            connected = False
            for parent in parents:
                preferred_block = (
                    self._preferred_sbm_child_block(graph, int(parent))
                    if self.constraint_kind == "sbm"
                    else None
                )
                child = self._add_node(
                    graph,
                    nodes,
                    final_color=color_id,
                    ring=int(nodes[int(parent)]["ring"]) + 1,
                    preferred_block=preferred_block,
                )
                if self._force_budget_edge(graph, nodes, int(parent), int(child), edge_type=1):
                    added += 1
                    connected = True
                    break
                block = graph.nodes[int(child)].get("budget_block")
                graph.remove_node(int(child))
                nodes.pop()
                if block is not None:
                    self._active_block_remaining[int(block)] += 1
            if not connected:
                break
        return added

    def _unresolved_for_ring(self, nodes, ring_nodes):
        out = []
        for node_id in ring_nodes:
            node = nodes[int(node_id)]
            for (edge_type, nbr_prev), count in list(node["required"].items()):
                for _ in range(max(0, int(count))):
                    out.append((int(node_id), int(edge_type), int(nbr_prev)))
        return out

    def _unresolved_entries(self, nodes):
        out = []
        for node_id in range(len(nodes)):
            node = nodes[int(node_id)]
            for (edge_type, nbr_prev), count in list(node["required"].items()):
                for _ in range(max(0, int(count))):
                    out.append((int(node_id), int(edge_type), int(nbr_prev)))
        return out

    def _all_unresolved(self, nodes):
        return sum(sum(max(0, int(count)) for count in node["required"].values()) for node in nodes)

    def _hard_node_budget(self, target_nodes: int):
        if self.max_nodes is not None:
            return max(2, int(self.max_nodes))
        scaled = int(math.ceil(float(target_nodes) * max(float(self.node_budget_factor), 0.1)))
        return max(2, scaled + int(self.node_budget_margin))

    def _budget_context(self, nodes, *, target_nodes: int, ring: int):
        remaining = max(0, int(target_nodes) - len(nodes))
        current_edges = (
            int(self._active_graph.number_of_edges())
            if self._active_graph is not None
            else 0
        )
        return self.model.budget_context(
            target_nodes=int(target_nodes),
            current_nodes=len(nodes),
            remaining_nodes=remaining,
            unresolved_stubs=self._all_unresolved(nodes),
            ring=int(ring),
            target_edges=int(self._active_target_edges or 0),
            current_edges=current_edges,
        )

    def _node_repr(self, node):
        return self.model.node_repr(
            final_color=node["final_color"],
            prev_color=node["prev_color"],
            label=node["label"],
            ring=node["ring"],
            degree=node["degree"],
        )

    def _graph_summary(self, nodes, z_ctx, *, target_nodes: int, ring: int):
        # The prediction heads condition the summary on the budget.  Keep the
        # shared graph summary raw so inference matches the training path and
        # does not add the same budget context twice.
        return self.model.graph_summary([self._node_repr(node) for node in nodes], z_ctx, budget_ctx=None)

    def _plan_context(self, plan_color_ids, node_index: int):
        if not getattr(self.model, "use_plan_context", False):
            return None
        return self.model.plan_context_from_sequence(plan_color_ids or [], int(node_index))

    def _plan_target_color(self, plan_color_ids, node_index: int):
        if plan_color_ids is None:
            return None
        node_index = int(node_index)
        if 0 <= node_index < len(plan_color_ids):
            color_id = int(plan_color_ids[node_index])
            if color_id > 0:
                return color_id
        return None

    def _plan_action(self, plan_actions, node_index: int):
        if plan_actions is None:
            return None
        node_index = int(node_index)
        if 0 <= node_index < len(plan_actions):
            return plan_actions[node_index]
        return None

    def _action_logits_for_stub(
        self,
        nodes,
        z_ctx,
        *,
        src: int,
        edge_type: int,
        nbr_prev: int,
        target_nodes: int,
        ring: int,
        plan_context=None,
    ):
        if not self.use_learned_actions:
            return None
        budget_ctx = self._budget_context(nodes, target_nodes=target_nodes, ring=ring)
        summary = self._graph_summary(nodes, z_ctx, target_nodes=target_nodes, ring=ring)
        return self.model.action_logits(
            z_ctx,
            self._node_repr(nodes[int(src)]),
            edge_type=int(edge_type),
            needed_prev_color=int(nbr_prev),
            summary=summary,
            budget_ctx=budget_ctx,
            plan_context=plan_context,
        )

    def _sample_color(self, logits, valid_ids, *, plan_color_ids=None, node_index: int | None = None):
        valid = sorted({int(v) for v in valid_ids if int(v) > 0 and int(v) <= self.model.color_vocab_size})
        if not valid:
            raise DecoderSamplingError("No valid color candidates.")
        masked = self.model._mask_logits(logits, valid)
        if plan_color_ids is not None and float(self.plan_prior_weight) != 0.0:
            idx = int(node_index if node_index is not None else 0)
            if 0 <= idx < len(plan_color_ids):
                target = int(plan_color_ids[idx])
                if target in valid:
                    masked = masked.clone()
                    masked[..., target] = masked[..., target] + float(self.plan_prior_weight)
        if float(self.empirical_prior_weight) != 0.0:
            prior = self._color_log_prior(masked.device, masked.dtype)
            masked = masked + float(self.empirical_prior_weight) * prior
        if float(self.adaptive_prior_weight) != 0.0:
            prior = self._adaptive_log_prior(masked.device, masked.dtype)
            masked = masked + float(self.adaptive_prior_weight) * prior
        idx = sample_index_from_logits(masked, self.rng, temperature=self.temperature, greedy=self.greedy)
        if idx not in valid:
            idx = self.rng.choice(valid)
        return int(idx)

    def _color_log_prior(self, device, dtype):
        key = (str(device), str(dtype))
        cached = self._log_prior_cache.get(key)
        if cached is not None:
            return cached
        values = torch.zeros((self.model.color_vocab_size + 1,), dtype=dtype, device=device)
        for color_id in range(1, self.model.color_vocab_size + 1):
            spec = self.color_specs_by_id.get(int(color_id))
            weight = max(float(spec.weight if spec is not None else 0.0), 1.0e-12)
            values[int(color_id)] = math.log(weight)
        self._log_prior_cache[key] = values
        return values

    def _train_color_probs(self, device, dtype):
        log_prior = self._color_log_prior(device, dtype)
        probs = torch.exp(log_prior).clamp_min(1.0e-12)
        probs[0] = 0.0
        total = probs.sum().clamp_min(1.0e-12)
        return probs / total

    def _adaptive_log_prior(self, device, dtype):
        key = (str(device), str(dtype), int(self._adaptive_total))
        cached = self._adaptive_cache.get(key)
        if cached is not None:
            return cached
        train_probs = self._train_color_probs(device, dtype)
        pseudo = max(float(self.adaptive_prior_pseudocount), 1.0e-6)
        counts = pseudo * train_probs
        if self._adaptive_counts:
            idx = torch.tensor(
                [int(color_id) for color_id in self._adaptive_counts.keys()],
                dtype=torch.long,
                device=device,
            )
            vals = torch.tensor(
                [float(count) for count in self._adaptive_counts.values()],
                dtype=dtype,
                device=device,
            )
            idx = torch.clamp(idx, min=0, max=self.model.color_vocab_size)
            counts.index_add_(0, idx, vals)
        current_probs = counts / max(float(self._adaptive_total) + pseudo, 1.0e-6)
        values = torch.log(train_probs.clamp_min(1.0e-12)) - torch.log(current_probs.clamp_min(1.0e-12))
        values[0] = 0.0
        self._adaptive_cache = {key: values}
        return values

    def _record_graph_colors(self, graph):
        if float(self.adaptive_prior_weight) == 0.0 or graph is None:
            return
        for _node, data in graph.nodes(data=True):
            color_id = data.get("color_id")
            if color_id is None:
                continue
            color_id = int(color_id)
            if 0 < color_id <= self.model.color_vocab_size:
                self._adaptive_counts[color_id] += 1
                self._adaptive_total += 1
        self._adaptive_cache.clear()

    def _child_candidates(self, *, prev_color: int, source_prev_color: int, edge_type: int):
        children = self.model.children_by_prev.get(int(prev_color), [])
        out = []
        for color_id in children:
            if int(color_id) not in self.color_specs_by_id:
                continue
            req = self._requirements(color_id)
            if req.get((int(edge_type), int(source_prev_color)), 0) > 0:
                out.append(int(color_id))
        return out

    def _stub_accepts_plan_color(self, nodes, src: int, edge_type: int, nbr_prev: int, color_id: int | None):
        if color_id is None or int(color_id) not in self.color_specs_by_id:
            return False
        if self._parent_prev_color(int(color_id)) != int(nbr_prev):
            return False
        req = self._requirements(int(color_id))
        return req.get((int(edge_type), int(nodes[int(src)]["prev_color"])), 0) > 0

    def _stub_matches_plan_action(self, nodes, src: int, edge_type: int, action):
        if action is None:
            return True
        parent_edge_type = int(action.get("parent_edge_type", 0))
        if parent_edge_type > 0 and int(edge_type) != parent_edge_type:
            return False
        parent_prev_color = int(action.get("parent_prev_color", 0))
        if parent_prev_color > 0 and int(nodes[int(src)]["prev_color"]) != parent_prev_color:
            return False
        planned_ring = int(action.get("ring", 0))
        if planned_ring > 0 and int(nodes[int(src)]["ring"]) + 1 != planned_ring:
            return False
        return True

    def _choose_stub_for_new_node(self, nodes, unresolved, plan_color_ids, node_index: int, plan_actions=None):
        target_color = self._plan_target_color(plan_color_ids, node_index)
        action = self._plan_action(plan_actions, node_index)
        if target_color is not None or action is not None:
            guided = [
                (int(src), int(edge_type), int(nbr_prev))
                for src, edge_type, nbr_prev in unresolved
                if (target_color is None or self._stub_accepts_plan_color(nodes, src, edge_type, nbr_prev, target_color))
                and self._stub_matches_plan_action(nodes, src, edge_type, action)
            ]
            if guided:
                return self.rng.choice(guided)
        return self.rng.choice(unresolved)

    def _edge_decision(self, z_ctx, nodes, u: int, v: int, edge_type: int, target_nodes: int, plan_context=None):
        left_repr = self._node_repr(nodes[int(u)])
        right_repr = self._node_repr(nodes[int(v)])
        budget_ctx = self._budget_context(
            nodes,
            target_nodes=target_nodes,
            ring=max(int(nodes[int(u)]["ring"]), int(nodes[int(v)]["ring"])),
        )
        logit = self.model.edge_logit(
            z_ctx,
            left_repr,
            right_repr,
            edge_type,
            budget_ctx=budget_ctx,
            plan_context=plan_context,
        )
        if self.greedy:
            return bool(float(logit.item()) >= 0.0)
        prob = torch.sigmoid(logit / max(float(self.edge_temperature), 1.0e-6)).item()
        return self.rng.random() < float(prob)

    def _existing_candidates_for_stub(self, graph, nodes, src: int, edge_type: int, nbr_prev: int):
        return [
            int(dst)
            for dst in range(len(nodes))
            if int(dst) != int(src)
            and not graph.has_edge(int(src), int(dst))
            and abs(int(nodes[int(dst)]["ring"]) - int(nodes[int(src)]["ring"])) <= 1
            and int(nodes[int(dst)]["prev_color"]) == int(nbr_prev)
            and self._can_connect(nodes, int(src), int(dst), int(edge_type))
            and self._constraint_allows_edge(graph, nodes, int(src), int(dst), int(edge_type))
        ]

    def _stubs_with_existing_candidates(self, graph, nodes, unresolved):
        return [
            (int(src), int(edge_type), int(nbr_prev))
            for src, edge_type, nbr_prev in unresolved
            if self._existing_candidates_for_stub(graph, nodes, src, edge_type, nbr_prev)
        ]

    def _connect_existing_for_stub(self, graph, nodes, z_ctx, src: int, edge_type: int, nbr_prev: int, target_nodes: int, plan_context=None):
        candidates = self._existing_candidates_for_stub(graph, nodes, src, edge_type, nbr_prev)
        if not candidates:
            return False
        logits = torch.stack([
            self.model.edge_logit(
                z_ctx,
                self._node_repr(nodes[int(src)]),
                self._node_repr(nodes[int(dst)]),
                int(edge_type),
                budget_ctx=self._budget_context(
                    nodes,
                    target_nodes=target_nodes,
                    ring=max(int(nodes[int(src)]["ring"]), int(nodes[int(dst)]["ring"])),
                ),
                plan_context=plan_context,
            )
            for dst in candidates
        ])
        scaled = logits / max(float(self.edge_temperature), 1.0e-6)
        if self.greedy:
            best = int(torch.argmax(scaled).item())
            if float(scaled[best].item()) < 0.0:
                return False
            idx = best
        else:
            probs = torch.sigmoid(scaled).detach().float().cpu().tolist()
            accepted = [idx for idx, prob in enumerate(probs) if self.rng.random() < float(prob)]
            if not accepted:
                return False
            idx = max(accepted, key=lambda pos: float(scaled[int(pos)].item()))
        return self._connect(graph, nodes, int(src), candidates[int(idx)], int(edge_type))

    def _create_node_for_stub(
        self,
        graph,
        nodes,
        z_ctx,
        *,
        src: int,
        edge_type: int,
        nbr_prev: int,
        target_nodes: int,
        ring: int,
        plan_color_ids=None,
        plan_context=None,
    ):
        candidates = self._child_candidates(
            prev_color=nbr_prev,
            source_prev_color=nodes[int(src)]["prev_color"],
            edge_type=edge_type,
        )
        if not candidates:
            return False, None, "no_child_candidate"
        summary = self._graph_summary(nodes, z_ctx, target_nodes=target_nodes, ring=ring)
        budget_ctx = self._budget_context(nodes, target_nodes=target_nodes, ring=ring)
        logits = self.model.color_logits(
            z_ctx,
            prev_color=nbr_prev,
            ring=ring,
            summary=summary,
            budget_ctx=budget_ctx,
            plan_context=plan_context,
        )
        try:
            color_id = self._sample_color(
                logits,
                candidates,
                plan_color_ids=plan_color_ids,
                node_index=len(nodes),
            )
            preferred_block = (
                self._preferred_sbm_child_block(graph, int(src))
                if self.constraint_kind == "sbm"
                else None
            )
            dst = self._add_node(
                graph,
                nodes,
                final_color=color_id,
                ring=ring,
                preferred_block=preferred_block,
            )
        except DecoderSamplingError:
            return False, None, "child_color_failed"
        if not self._connect(graph, nodes, src, dst, edge_type):
            return False, dst, "mandatory_edge_failed"
        return True, dst, None

    def _choose_learned_action(
        self,
        graph,
        nodes,
        z_ctx,
        unresolved,
        *,
        hard_max_nodes: int,
        target_nodes: int,
        ring: int,
        plan_context=None,
        planned_node_action=None,
    ):
        if not self.use_learned_actions:
            return None
        candidates = []
        logits = []
        for src, edge_type, nbr_prev in unresolved:
            src = int(src)
            edge_type = int(edge_type)
            nbr_prev = int(nbr_prev)
            action_logits = self._action_logits_for_stub(
                nodes,
                z_ctx,
                src=src,
                edge_type=edge_type,
                nbr_prev=nbr_prev,
                target_nodes=target_nodes,
                ring=ring,
                plan_context=plan_context,
            )
            if action_logits is None:
                continue
            if self._existing_candidates_for_stub(graph, nodes, src, edge_type, nbr_prev):
                candidates.append((ACTION_CONNECT_EXISTING, src, edge_type, nbr_prev))
                logits.append(action_logits[ACTION_CONNECT_EXISTING])
            if (
                len(nodes) < int(hard_max_nodes)
                and int(nodes[src]["ring"]) < int(self.max_rings)
                and self._child_candidates(
                    prev_color=nbr_prev,
                    source_prev_color=nodes[src]["prev_color"],
                    edge_type=edge_type,
                )
                and self._stub_matches_plan_action(nodes, src, edge_type, planned_node_action)
            ):
                candidates.append((ACTION_CREATE_NODE, src, edge_type, nbr_prev))
                logits.append(action_logits[ACTION_CREATE_NODE])
        if not candidates:
            return None
        idx = sample_index_from_logits(
            torch.stack(logits),
            self.rng,
            temperature=self.action_temperature,
            greedy=self.greedy,
        )
        return candidates[int(idx)]

    def _learned_stop_decision(self, nodes, z_ctx, *, target_nodes: int, ring: int, plan_context=None):
        if not self.use_learned_stop or not nodes:
            return True
        budget_ctx = self._budget_context(nodes, target_nodes=target_nodes, ring=ring)
        summary = self._graph_summary(nodes, z_ctx, target_nodes=target_nodes, ring=ring)
        logits = self.model.action_logits(
            z_ctx,
            self._node_repr(nodes[0]),
            edge_type=0,
            needed_prev_color=0,
            summary=summary,
            budget_ctx=budget_ctx,
            plan_context=plan_context,
        )
        if self.greedy:
            return int(torch.argmax(logits).item()) == ACTION_STOP
        probs = torch.softmax(logits / max(float(self.action_temperature), 1.0e-6), dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return True
        return self.rng.random() < float(probs[ACTION_STOP].item())

    def _stub_closure(
        self,
        graph,
        nodes,
        z_ctx,
        *,
        hard_max_nodes: int,
        target_nodes: int,
        action_counts=None,
        plan_color_ids=None,
        plan_actions=None,
    ):
        max_steps = max(10, int(self.max_closure_steps_per_node) * max(1, int(target_nodes)))
        closure_edges = 0
        for _ in range(max_steps):
            unresolved = self._unresolved_entries(nodes)
            if not unresolved:
                return True, closure_edges

            learned_action = None
            plan_context = self._plan_context(plan_color_ids, len(nodes))
            if self.use_learned_closure:
                ring = max((int(node["ring"]) for node in nodes), default=0)
                learned_action = self._choose_learned_action(
                    graph,
                    nodes,
                    z_ctx,
                    unresolved,
                    hard_max_nodes=hard_max_nodes,
                    target_nodes=target_nodes,
                    ring=ring,
                    plan_context=plan_context,
                    planned_node_action=self._plan_action(plan_actions, len(nodes)),
                )
            if learned_action is not None:
                action, src, edge_type, nbr_prev = learned_action
                if int(action) == ACTION_CONNECT_EXISTING:
                    if self._connect_existing_for_stub(
                        graph,
                        nodes,
                        z_ctx,
                        src,
                        edge_type,
                        nbr_prev,
                        target_nodes,
                        plan_context=plan_context,
                    ):
                        closure_edges += 1
                        if action_counts is not None:
                            action_counts["closure_connect_existing"] += 1
                        continue
                    return False, closure_edges
                if int(action) == ACTION_CREATE_NODE:
                    next_ring = int(nodes[int(src)]["ring"]) + 1
                    if next_ring > int(self.max_rings):
                        return False, closure_edges
                    ok, _dst, _failure = self._create_node_for_stub(
                        graph,
                        nodes,
                        z_ctx,
                        src=src,
                        edge_type=edge_type,
                        nbr_prev=nbr_prev,
                        target_nodes=target_nodes,
                        ring=next_ring,
                        plan_color_ids=plan_color_ids,
                        plan_context=plan_context,
                    )
                    if not ok:
                        return False, closure_edges
                    if action_counts is not None:
                        action_counts["closure_create_node"] += 1
                    continue

            with_existing = self._stubs_with_existing_candidates(graph, nodes, unresolved)
            if with_existing:
                src, edge_type, nbr_prev = self.rng.choice(with_existing)
                if self._connect_existing_for_stub(
                    graph,
                    nodes,
                    z_ctx,
                    src,
                    edge_type,
                    nbr_prev,
                    target_nodes,
                    plan_context=plan_context,
                ):
                    closure_edges += 1
                    continue

            if len(nodes) >= int(hard_max_nodes):
                return False, closure_edges

            # If no existing connection was available or accepted, choose the
            # creation stub separately so every newly created node can follow
            # its latent-plan parent/ring action.
            creatable = [
                (int(src), int(edge_type), int(nbr_prev))
                for src, edge_type, nbr_prev in unresolved
                if int(nodes[int(src)]["ring"]) < int(self.max_rings)
            ]
            if not creatable:
                return False, closure_edges
            src, edge_type, nbr_prev = self._choose_stub_for_new_node(
                nodes,
                creatable,
                plan_color_ids,
                len(nodes),
                plan_actions=plan_actions,
            )

            next_ring = int(nodes[int(src)]["ring"]) + 1
            if next_ring > int(self.max_rings):
                return False, closure_edges

            candidates = self._child_candidates(
                prev_color=nbr_prev,
                source_prev_color=nodes[int(src)]["prev_color"],
                edge_type=edge_type,
            )
            if not candidates:
                return False, closure_edges
            summary = self._graph_summary(nodes, z_ctx, target_nodes=target_nodes, ring=next_ring)
            budget_ctx = self._budget_context(nodes, target_nodes=target_nodes, ring=next_ring)
            logits = self.model.color_logits(
                z_ctx,
                prev_color=nbr_prev,
                ring=next_ring,
                summary=summary,
                budget_ctx=budget_ctx,
                plan_context=plan_context,
            )
            try:
                color_id = self._sample_color(
                    logits,
                    candidates,
                    plan_color_ids=plan_color_ids,
                    node_index=len(nodes),
                )
                preferred_block = (
                    self._preferred_sbm_child_block(graph, int(src))
                    if self.constraint_kind == "sbm"
                    else None
                )
                dst = self._add_node(
                    graph,
                    nodes,
                    final_color=color_id,
                    ring=next_ring,
                    preferred_block=preferred_block,
                )
            except DecoderSamplingError:
                return False, closure_edges
            if not self._connect(graph, nodes, src, dst, edge_type):
                return False, closure_edges
        return False, closure_edges

    def _acceptable_partial(self, graph, nodes):
        if graph.number_of_edges() == 0:
            return False
        if self.require_connected and not nx.is_connected(graph):
            return False
        if self.require_complete and self._all_unresolved(nodes) > 0:
            return False
        if self.constraint_kind in {"planar", "sbm"}:
            if self._active_target_nodes is not None and graph.number_of_nodes() != int(self._active_target_nodes):
                return False
            if self._active_target_edges is not None and graph.number_of_edges() != int(self._active_target_edges):
                return False
        return True

    def sample_one(
        self,
        target_nodes: int,
        target_edges: int | None = None,
        constraint_profile: dict[str, Any] | None = None,
        plan_color_ids: list[int] | None = None,
        plan_actions: list[dict[str, int]] | None = None,
    ) -> ARSample:
        target_nodes = max(2, int(target_nodes))
        target_edges = None if target_edges is None else max(1, int(target_edges))
        self._active_target_nodes = target_nodes
        self._active_target_edges = target_edges
        self._prepare_constraint_state(target_nodes, target_edges, constraint_profile)
        hard_max_nodes = self._hard_node_budget(target_nodes)
        failure_counts = Counter()
        self.model.eval()
        external_plan_color_ids = [int(v) for v in list(plan_color_ids or [])]
        external_plan_actions = list(plan_actions or [])

        with torch.no_grad():
            z = self.model.sample_latent(1)[0]
            z_ctx = self.model.z_context(z)[0]

        for restart in range(int(self.max_restarts)):
            with torch.no_grad():
                active_plan_colors = list(external_plan_color_ids)
                active_plan_actions = list(external_plan_actions)
                if not active_plan_colors and bool(getattr(self.model, "use_latent_plan", False)):
                    active_plan_colors, active_plan_actions = self.model.sample_latent_plan(
                        z_ctx,
                        target_nodes,
                        temperature=self.plan_temperature,
                        greedy=self.plan_greedy,
                    )
                graph = nx.Graph()
                self._active_graph = graph
                self._reset_constraint_attempt()
                nodes = []
                try:
                    center_plan_context = self._plan_context(active_plan_colors, 0)
                    center_budget = self.model.budget_context(
                        target_nodes=target_nodes,
                        current_nodes=0,
                        remaining_nodes=target_nodes,
                        unresolved_stubs=0,
                        ring=0,
                        target_edges=int(target_edges or 0),
                        current_edges=0,
                    )
                    center_color = self._sample_color(
                        self.model.center_logits(
                            z_ctx,
                            budget_ctx=center_budget,
                            plan_context=center_plan_context,
                        ),
                        self.seed_color_ids,
                        plan_color_ids=active_plan_colors,
                        node_index=0,
                    )
                    center = self._add_node(graph, nodes, final_color=center_color, ring=0)
                except DecoderSamplingError:
                    failure_counts["center_failed"] += 1
                    continue

                rings = [[center]]
                failure = None
                completed = False
                complete_before_closure = False
                closure_edges = 0
                frontier_closure_edges = 0
                final_closure_edges = 0
                action_counts = Counter()
                for ring_idx in range(1, int(self.max_rings) + 1):
                    prev_ring = rings[-1]
                    current_ring = []
                    while True:
                        unresolved = self._unresolved_for_ring(nodes, prev_ring)
                        if not unresolved:
                            break
                        plan_context = self._plan_context(active_plan_colors, len(nodes))
                        learned_action = self._choose_learned_action(
                            graph,
                            nodes,
                            z_ctx,
                            unresolved,
                            hard_max_nodes=hard_max_nodes,
                            target_nodes=target_nodes,
                            ring=ring_idx,
                            plan_context=plan_context,
                            planned_node_action=self._plan_action(active_plan_actions, len(nodes)),
                        )
                        if learned_action is not None:
                            action, src, edge_type, nbr_prev = learned_action
                            if int(action) == ACTION_CONNECT_EXISTING:
                                if self._connect_existing_for_stub(
                                    graph,
                                    nodes,
                                    z_ctx,
                                    src,
                                    edge_type,
                                    nbr_prev,
                                    target_nodes,
                                    plan_context=plan_context,
                                ):
                                    closure_edges += 1
                                    frontier_closure_edges += 1
                                    action_counts["connect_existing"] += 1
                                    continue
                                failure = "learned_connect_failed"
                                break
                            if int(action) == ACTION_CREATE_NODE:
                                ok, dst, create_failure = self._create_node_for_stub(
                                    graph,
                                    nodes,
                                    z_ctx,
                                    src=src,
                                    edge_type=edge_type,
                                    nbr_prev=nbr_prev,
                                    target_nodes=target_nodes,
                                    ring=ring_idx,
                                    plan_color_ids=active_plan_colors,
                                    plan_context=plan_context,
                                )
                                if not ok:
                                    failure = create_failure
                                    break
                                action_counts["create_node"] += 1
                                current_ring.append(int(dst))
                                edge_plan_context = self._plan_context(active_plan_colors, len(nodes))

                                for other in prev_ring:
                                    if int(other) == int(src) or graph.has_edge(other, int(dst)):
                                        continue
                                    for extra_edge in self._possible_edge_types(nodes, other, int(dst)):
                                        if self._edge_decision(
                                            z_ctx,
                                            nodes,
                                            other,
                                            int(dst),
                                            extra_edge,
                                            target_nodes,
                                            plan_context=edge_plan_context,
                                        ):
                                            self._connect(graph, nodes, other, int(dst), extra_edge)
                                            break
                                continue

                        existing_unresolved = (
                            self._stubs_with_existing_candidates(graph, nodes, unresolved)
                            if self.connect_existing_first
                            else []
                        )
                        if existing_unresolved:
                            src, edge_type, nbr_prev = self.rng.choice(existing_unresolved)
                            if self._connect_existing_for_stub(
                                graph,
                                nodes,
                                z_ctx,
                                src,
                                edge_type,
                                nbr_prev,
                                target_nodes,
                                plan_context=plan_context,
                            ):
                                closure_edges += 1
                                frontier_closure_edges += 1
                                continue
                        if (
                            not self.connect_existing_first
                            and graph.number_of_nodes() >= target_nodes
                        ):
                            src, edge_type, nbr_prev = self.rng.choice(unresolved)
                            if self._connect_existing_for_stub(
                                graph,
                                nodes,
                                z_ctx,
                                src,
                                edge_type,
                                nbr_prev,
                                target_nodes,
                                plan_context=plan_context,
                            ):
                                closure_edges += 1
                                frontier_closure_edges += 1
                                continue
                        if graph.number_of_nodes() >= hard_max_nodes:
                            break
                        src, edge_type, nbr_prev = self._choose_stub_for_new_node(
                            nodes,
                            unresolved,
                            active_plan_colors,
                            len(nodes),
                            plan_actions=active_plan_actions,
                        )
                        ok, dst, create_failure = self._create_node_for_stub(
                            graph,
                            nodes,
                            z_ctx,
                            src=src,
                            edge_type=edge_type,
                            nbr_prev=nbr_prev,
                            target_nodes=target_nodes,
                            ring=ring_idx,
                            plan_color_ids=active_plan_colors,
                            plan_context=plan_context,
                        )
                        if not ok:
                            failure = create_failure
                            break
                        current_ring.append(int(dst))
                        edge_plan_context = self._plan_context(active_plan_colors, len(nodes))

                        for other in prev_ring:
                            if int(other) == int(src) or graph.has_edge(other, int(dst)):
                                continue
                            for extra_edge in self._possible_edge_types(nodes, other, int(dst)):
                                if self._edge_decision(
                                    z_ctx,
                                    nodes,
                                    other,
                                    int(dst),
                                    extra_edge,
                                    target_nodes,
                                    plan_context=edge_plan_context,
                                ):
                                    self._connect(graph, nodes, other, int(dst), extra_edge)
                                    break

                    if failure is not None:
                        break

                    for pos, u in enumerate(current_ring):
                        for v in current_ring[pos + 1:]:
                            if graph.has_edge(u, v):
                                continue
                            for edge_type in self._possible_edge_types(nodes, u, v):
                                plan_context = self._plan_context(active_plan_colors, len(nodes))
                                if self._edge_decision(
                                    z_ctx,
                                    nodes,
                                    u,
                                    v,
                                    edge_type,
                                    target_nodes,
                                    plan_context=plan_context,
                                ):
                                    self._connect(graph, nodes, u, v, edge_type)
                                    break

                    if self._all_unresolved(nodes) == 0:
                        if self._learned_stop_decision(
                            nodes,
                            z_ctx,
                            target_nodes=target_nodes,
                            ring=ring_idx,
                            plan_context=self._plan_context(active_plan_colors, len(nodes)),
                        ):
                            action_counts["stop"] += 1
                            completed = True
                            break
                        failure = "learned_stop_rejected"
                        break
                    if not current_ring:
                        failure = "no_new_ring"
                        break
                    rings.append(current_ring)

                complete_before_closure = (
                    self._all_unresolved(nodes) == 0
                    and len(nodes) == target_nodes
                )
                node_budget_repair_edges = self._node_budget_repair(graph, nodes)
                closure_edges += int(node_budget_repair_edges)
                final_closure_edges += int(node_budget_repair_edges)
                if not complete_before_closure and self.use_stub_closure:
                    completed, added_closure_edges = self._stub_closure(
                        graph,
                        nodes,
                        z_ctx,
                        hard_max_nodes=hard_max_nodes,
                        target_nodes=target_nodes,
                        action_counts=action_counts,
                        plan_color_ids=active_plan_colors,
                        plan_actions=active_plan_actions,
                    )
                    closure_edges += int(added_closure_edges)
                    final_closure_edges += int(added_closure_edges)

                budget_repair_edges = self._budget_repair(graph, nodes)
                closure_edges += int(budget_repair_edges)
                final_closure_edges += int(budget_repair_edges)

                if failure != "learned_stop_rejected" and self._acceptable_partial(graph, nodes):
                    complete = self._all_unresolved(nodes) == 0
                    return ARSample(graph, {
                        "success": True,
                        "complete": bool(complete),
                        "complete_before_closure": bool(complete_before_closure),
                        "failure": None if complete else (failure or "limit_reached"),
                        "restarts_used": restart,
                        "latent_draws": 1,
                        "decode_attempts": restart + 1,
                        "n_nodes": graph.number_of_nodes(),
                        "target_nodes": target_nodes,
                        "target_edges": target_edges,
                        "n_edges": graph.number_of_edges(),
                        "n_rings": len({int(node["ring"]) for node in nodes}),
                        "hard_max_nodes": hard_max_nodes,
                        "closure_edges": int(closure_edges),
                        "frontier_closure_edges": int(frontier_closure_edges),
                        "final_closure_edges": int(final_closure_edges),
                        "budget_repair_edges": int(budget_repair_edges),
                        "node_budget_repair_edges": int(node_budget_repair_edges),
                        "unresolved_stubs": self._all_unresolved(nodes),
                        "learned_action_policy": bool(self.use_learned_actions),
                        "latent_plan_policy": bool(getattr(self.model, "use_latent_plan", False)),
                        "plan_length": len(active_plan_colors),
                        "plan_prefix_matches": _count_plan_prefix_matches(nodes, active_plan_colors),
                        "action_counts": dict(action_counts),
                    })
                failure_counts[str(failure or "incomplete")] += 1

        return ARSample(None, {
            "success": False,
            "failure": "restarts_exhausted",
            "failure_counts": dict(failure_counts),
            "target_nodes": target_nodes,
            "target_edges": target_edges,
            "latent_draws": 1,
            "decode_attempts": int(self.max_restarts),
        })

    def sample_many(
        self,
        node_counts: list[int],
        edge_counts: list[int | None] | None = None,
        constraint_profiles: list[dict[str, Any] | None] | None = None,
        plan_color_ids_by_sample: list[list[int]] | None = None,
        plan_actions_by_sample: list[list[dict[str, int]]] | None = None,
    ):
        plans = plan_color_ids_by_sample or [None] * len(node_counts)
        actions = plan_actions_by_sample or [None] * len(node_counts)
        edges = edge_counts or [None] * len(node_counts)
        profiles = constraint_profiles or [None] * len(node_counts)
        samples = []
        for idx, n in enumerate(node_counts):
            sample = self.sample_one(
                n,
                target_edges=edges[idx] if idx < len(edges) else None,
                constraint_profile=profiles[idx] if idx < len(profiles) else None,
                plan_color_ids=plans[idx] if idx < len(plans) else None,
                plan_actions=actions[idx] if idx < len(actions) else None,
            )
            samples.append(sample)
            if sample.graph is not None:
                self._record_graph_colors(sample.graph)
        return samples


def _count_plan_prefix_matches(nodes, plan_color_ids):
    if not plan_color_ids:
        return 0
    n = min(len(nodes), len(plan_color_ids))
    return sum(int(nodes[idx]["final_color"]) == int(plan_color_ids[idx]) for idx in range(n))


def ring_vae_model_from_checkpoint(ckpt, device):
    model = RingWLVAE(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

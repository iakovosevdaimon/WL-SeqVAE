"""Pack variable-size ring traces into decoder supervision batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ring_vae_actions import ACTION_CONNECT_EXISTING, ACTION_CREATE_NODE, ACTION_STOP


@dataclass(frozen=True)
class PackedRingTrace:
    sequence: tuple[int, ...]
    encoder_sequence: tuple[int, ...]
    center_color: int
    target_nodes: int
    target_edges: int
    final_colors: tuple[int, ...]
    prev_colors: tuple[int, ...]
    labels: tuple[int, ...]
    rings: tuple[int, ...]
    degrees: tuple[int, ...]
    color_local_indices: tuple[int, ...]
    color_targets: tuple[int, ...]
    color_prev_colors: tuple[int, ...]
    color_rings: tuple[int, ...]
    color_unresolved_stubs: tuple[int, ...]
    color_current_edges: tuple[int, ...]
    prev_edge_left: tuple[int, ...]
    prev_edge_right: tuple[int, ...]
    prev_edge_types: tuple[int, ...]
    prev_edge_prefix_counts: tuple[int, ...]
    prev_edge_rings: tuple[int, ...]
    prev_edge_unresolved_stubs: tuple[int, ...]
    prev_edge_current_edges: tuple[int, ...]
    prev_edge_targets: tuple[float, ...]
    intra_edge_left: tuple[int, ...]
    intra_edge_right: tuple[int, ...]
    intra_edge_types: tuple[int, ...]
    intra_edge_prefix_counts: tuple[int, ...]
    intra_edge_rings: tuple[int, ...]
    intra_edge_unresolved_stubs: tuple[int, ...]
    intra_edge_current_edges: tuple[int, ...]
    intra_edge_targets: tuple[float, ...]
    action_source_indices: tuple[int, ...]
    action_edge_types: tuple[int, ...]
    action_needed_prev_colors: tuple[int, ...]
    action_prefix_counts: tuple[int, ...]
    action_rings: tuple[int, ...]
    action_unresolved_stubs: tuple[int, ...]
    action_current_edges: tuple[int, ...]
    action_next_unresolved_stubs: tuple[int, ...]
    action_delta_unresolved_stubs: tuple[int, ...]
    action_targets: tuple[int, ...]
    plan_parent_edge_types: tuple[int, ...]
    plan_parent_prev_colors: tuple[int, ...]
    plan_rings: tuple[int, ...]
    n_nodes: int


@dataclass(frozen=True)
class PackedRingBatch:
    sequences: list[tuple[int, ...]]
    encoder_sequences: list[tuple[int, ...]]
    center_targets: tuple[int, ...]
    target_nodes: tuple[int, ...]
    target_edges: tuple[int, ...]
    node_final_colors: tuple[int, ...]
    node_prev_colors: tuple[int, ...]
    node_labels: tuple[int, ...]
    node_rings: tuple[int, ...]
    node_degrees: tuple[int, ...]
    color_node_indices: tuple[int, ...]
    color_graph_start_indices: tuple[int, ...]
    color_prefix_counts: tuple[int, ...]
    color_graph_indices: tuple[int, ...]
    color_targets: tuple[int, ...]
    color_prev_colors: tuple[int, ...]
    color_rings: tuple[int, ...]
    color_unresolved_stubs: tuple[int, ...]
    color_current_edges: tuple[int, ...]
    prev_edge_left_indices: tuple[int, ...]
    prev_edge_right_indices: tuple[int, ...]
    prev_edge_graph_indices: tuple[int, ...]
    prev_edge_types: tuple[int, ...]
    prev_edge_prefix_counts: tuple[int, ...]
    prev_edge_rings: tuple[int, ...]
    prev_edge_unresolved_stubs: tuple[int, ...]
    prev_edge_current_edges: tuple[int, ...]
    prev_edge_targets: tuple[float, ...]
    intra_edge_left_indices: tuple[int, ...]
    intra_edge_right_indices: tuple[int, ...]
    intra_edge_graph_indices: tuple[int, ...]
    intra_edge_types: tuple[int, ...]
    intra_edge_prefix_counts: tuple[int, ...]
    intra_edge_rings: tuple[int, ...]
    intra_edge_unresolved_stubs: tuple[int, ...]
    intra_edge_current_edges: tuple[int, ...]
    intra_edge_targets: tuple[float, ...]
    action_source_indices: tuple[int, ...]
    action_graph_start_indices: tuple[int, ...]
    action_graph_indices: tuple[int, ...]
    action_edge_types: tuple[int, ...]
    action_needed_prev_colors: tuple[int, ...]
    action_prefix_counts: tuple[int, ...]
    action_rings: tuple[int, ...]
    action_unresolved_stubs: tuple[int, ...]
    action_current_edges: tuple[int, ...]
    action_next_unresolved_stubs: tuple[int, ...]
    action_delta_unresolved_stubs: tuple[int, ...]
    action_targets: tuple[int, ...]
    plan_parent_edge_types: list[tuple[int, ...]]
    plan_parent_prev_colors: list[tuple[int, ...]]
    plan_rings: list[tuple[int, ...]]
    n_graphs: int

    @property
    def n_prev_edges(self):
        return len(self.prev_edge_targets)

    @property
    def n_intra_edges(self):
        return len(self.intra_edge_targets)

    @property
    def n_actions(self):
        return len(self.action_targets)


def _trace_degrees_static(trace):
    degrees = [0] * len(trace.nodes)
    for u, v, _edge in trace.edges:
        degrees[int(u)] += 1
        degrees[int(v)] += 1
    return tuple(int(value) for value in degrees)


def _edge_set_static(trace):
    return {(min(int(u), int(v)), max(int(u), int(v)), int(edge)) for u, v, edge in trace.edges}


def _edge_candidates_static(trace, *, adjacent_only: bool):
    candidates = []
    required_by_prev = []
    for node in trace.nodes:
        lookup = defaultdict(set)
        for edge_type, prev_color in node.required:
            lookup[int(prev_color)].add(int(edge_type))
        required_by_prev.append(lookup)

    for ring_idx, ring in enumerate(trace.rings):
        if adjacent_only:
            if ring_idx == 0:
                continue
            pairs = [(u, v) for u in ring for v in trace.rings[ring_idx - 1]]
        else:
            pairs = []
            for pos, u in enumerate(ring):
                for v in ring[pos + 1:]:
                    pairs.append((u, v))
        for u, v in pairs:
            left = trace.nodes[int(u)]
            right = trace.nodes[int(v)]
            edge_types = required_by_prev[int(u)].get(int(right.prev_color), set())
            edge_types = edge_types & required_by_prev[int(v)].get(int(left.prev_color), set())
            for edge_type in sorted(edge_types):
                a, b = sorted((int(u), int(v)))
                candidates.append((a, b, int(edge_type)))
    return tuple(candidates)


def _decrement_requirement(remaining, node_id: int, key):
    node_id = int(node_id)
    if remaining[node_id].get(key, 0) <= 0:
        return 0
    remaining[node_id][key] -= 1
    if remaining[node_id][key] <= 0:
        del remaining[node_id][key]
    return 1


def _node_requirements(node):
    return Counter((int(edge), int(prev)) for edge, prev in node.required)


def _parent_edge_for_node(nodes, edge_set, node_id: int):
    node = nodes[int(node_id)]
    previous_ring = int(node.ring) - 1
    candidates = []
    fallbacks = []
    for u, v, edge_type in edge_set:
        if int(node_id) not in {int(u), int(v)}:
            continue
        src = int(v) if int(u) == int(node_id) else int(u)
        if src >= int(node_id):
            continue
        row = (src, int(node_id), int(edge_type))
        fallbacks.append(row)
        if int(nodes[src].ring) == previous_ring:
            candidates.append(row)
    candidates = candidates or fallbacks
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(int(nodes[row[0]].ring) - previous_ring),
            int(nodes[row[0]].final_color),
            int(row[0]),
            int(row[2]),
        ),
    )


def _connect_requirements(remaining, nodes, u: int, v: int, edge_type: int):
    left_key = (int(edge_type), int(nodes[int(v)].prev_color))
    right_key = (int(edge_type), int(nodes[int(u)].prev_color))
    removed = 0
    removed += _decrement_requirement(remaining, int(u), left_key)
    removed += _decrement_requirement(remaining, int(v), right_key)
    return int(removed)


def _choose_action_source(remaining, nodes, u: int, v: int, edge_type: int):
    left_key = (int(edge_type), int(nodes[int(v)].prev_color))
    right_key = (int(edge_type), int(nodes[int(u)].prev_color))
    left_ok = remaining[int(u)].get(left_key, 0) > 0
    right_ok = remaining[int(v)].get(right_key, 0) > 0
    if left_ok:
        return int(u), int(v)
    if right_ok:
        return int(v), int(u)
    return int(u), int(v)


def _remaining_unresolved_count(remaining):
    return sum(
        sum(max(0, int(count)) for count in node_remaining.values())
        for node_remaining in remaining
    )


def _remaining_edge_types(remaining, nodes, u: int, v: int):
    left = remaining[int(u)]
    right = remaining[int(v)]
    left_prev = int(nodes[int(u)].prev_color)
    right_prev = int(nodes[int(v)].prev_color)
    return sorted({
        int(edge_type)
        for edge_type, prev_color in left
        if int(prev_color) == right_prev
        and int(left[(edge_type, prev_color)]) > 0
        and int(right.get((int(edge_type), left_prev), 0)) > 0
    })


def _decoder_timeline_static(trace):
    """Teacher-force decoder events in the same order used at inference.

    Edge contexts must describe the state *before* the candidate edge is
    decided.  In particular, they must not subtract the edge target itself or
    future edges, and same-ring candidates are considered only after the whole
    ring has been created.
    """
    nodes = tuple(trace.nodes)
    n_nodes = len(nodes)
    edge_set = _edge_set_static(trace)
    remaining = [Counter() for _node in nodes]
    connected_pairs = set()
    color_unresolved = [0] * n_nodes
    color_current_edges = [0] * n_nodes
    prev_events = []
    intra_events = []
    created_count = 0

    def add_node(node_id: int):
        nonlocal created_count
        node_id = int(node_id)
        if node_id != created_count:
            raise ValueError(
                f"Ring trace nodes are not in decoder order: expected {created_count}, received {node_id}"
            )
        remaining[node_id] = _node_requirements(nodes[node_id])
        created_count += 1

    def connect(u: int, v: int, edge_type: int):
        u = int(u)
        v = int(v)
        edge_type = int(edge_type)
        removed = _connect_requirements(remaining, nodes, u, v, edge_type)
        if removed != 2:
            raise ValueError(
                f"Ring trace edge {(u, v, edge_type)} cannot be applied to the decoder state"
            )
        connected_pairs.add((min(u, v), max(u, v)))

    def record_pair(events, u: int, v: int, *, ring: int):
        u = int(u)
        v = int(v)
        pair = (min(u, v), max(u, v))
        if pair in connected_pairs:
            return
        for edge_type in _remaining_edge_types(remaining, nodes, u, v):
            candidate = (pair[0], pair[1], int(edge_type))
            target = candidate in edge_set
            events.append((
                pair[0],
                pair[1],
                int(edge_type),
                int(created_count),
                int(ring),
                int(_remaining_unresolved_count(remaining)),
                int(len(connected_pairs)),
                float(target),
            ))
            if target:
                connect(u, v, edge_type)
                break

    if not nodes:
        return (
            tuple(color_unresolved),
            tuple(color_current_edges),
            tuple(prev_events),
            tuple(intra_events),
        )

    if not trace.rings or list(trace.rings[0]) != [0]:
        raise ValueError("Ring trace must start with node 0 as its center ring")
    add_node(0)

    for ring_idx in range(1, len(trace.rings)):
        previous_ring = [int(node_id) for node_id in trace.rings[ring_idx - 1]]
        current_ring = [int(node_id) for node_id in trace.rings[ring_idx]]
        for node_id in current_ring:
            color_unresolved[node_id] = int(_remaining_unresolved_count(remaining))
            color_current_edges[node_id] = int(len(connected_pairs))
            add_node(node_id)
            parent = _parent_edge_for_node(nodes, edge_set, node_id)
            if parent is None:
                raise ValueError(f"Ring trace node {node_id} has no decoder parent")
            parent_id, _dst, parent_edge_type = parent
            connect(parent_id, node_id, parent_edge_type)

            for other in previous_ring:
                if int(other) == int(parent_id):
                    continue
                record_pair(prev_events, other, node_id, ring=ring_idx)

        for pos, u in enumerate(current_ring):
            for v in current_ring[pos + 1:]:
                record_pair(intra_events, u, v, ring=ring_idx)

    if created_count != n_nodes:
        raise ValueError(f"Ring trace contains {n_nodes} nodes but its rings contain {created_count}")
    return (
        tuple(color_unresolved),
        tuple(color_current_edges),
        tuple(prev_events),
        tuple(intra_events),
    )


def _action_events_static(trace, edge_set):
    nodes = tuple(trace.nodes)
    events = []
    create_edges = set()
    remaining = [Counter() for _node in nodes]
    total_unresolved = 0
    current_edges = 0
    if nodes:
        remaining[0] = _node_requirements(nodes[0])
        total_unresolved = sum(max(0, int(count)) for count in remaining[0].values())

    for node_id in range(1, len(nodes)):
        node = nodes[int(node_id)]
        parent = _parent_edge_for_node(nodes, edge_set, int(node_id))
        if parent is None:
            continue
        src, dst, edge_type = parent
        create_edges.add((min(src, dst), max(src, dst), int(edge_type)))
        prefix = int(node_id)
        current = int(total_unresolved)
        remaining[int(node_id)] = _node_requirements(node)
        total_unresolved += sum(max(0, int(count)) for count in remaining[int(node_id)].values())
        total_unresolved -= _connect_requirements(remaining, nodes, int(src), int(dst), int(edge_type))
        next_unresolved = max(0, int(total_unresolved))
        events.append((
            int(src),
            int(edge_type),
            int(node.prev_color),
            prefix,
            int(node.ring),
            current,
            next_unresolved,
            int(next_unresolved - current),
            int(current_edges),
            ACTION_CREATE_NODE,
        ))
        current_edges += 1

        for u, v, closure_edge_type in sorted(edge_set):
            edge_key = (min(int(u), int(v)), max(int(u), int(v)), int(closure_edge_type))
            if edge_key in create_edges or max(int(u), int(v)) != int(node_id):
                continue
            src, dst = _choose_action_source(remaining, nodes, int(u), int(v), int(closure_edge_type))
            current = int(total_unresolved)
            total_unresolved -= _connect_requirements(remaining, nodes, int(u), int(v), int(closure_edge_type))
            next_unresolved = max(0, int(total_unresolved))
            events.append((
                int(src),
                int(closure_edge_type),
                int(nodes[dst].prev_color),
                int(node_id) + 1,
                max(int(nodes[src].ring), int(nodes[dst].ring)),
                current,
                next_unresolved,
                int(next_unresolved - current),
                int(current_edges),
                ACTION_CONNECT_EXISTING,
            ))
            current_edges += 1

    if nodes:
        prefix = len(nodes)
        current = max(0, int(total_unresolved))
        events.append((
            0,
            0,
            0,
            prefix,
            max(int(node.ring) for node in nodes),
            current,
            current,
            0,
            int(current_edges),
            ACTION_STOP,
        ))
    return tuple(events)


def _plan_parent_fields_static(trace, edge_set):
    nodes = tuple(trace.nodes)
    if not nodes:
        return (), (), ()
    parent_edges = [0] * len(nodes)
    parent_prev_colors = [0] * len(nodes)
    rings = [int(node.ring) for node in nodes]

    by_node = defaultdict(list)
    for u, v, edge_type in edge_set:
        u = int(u)
        v = int(v)
        by_node[u].append((v, int(edge_type)))
        by_node[v].append((u, int(edge_type)))

    for node_id in range(1, len(nodes)):
        node_ring = int(nodes[int(node_id)].ring)
        previous_ring = node_ring - 1
        parent_candidates = [
            (int(other), int(edge_type))
            for other, edge_type in by_node.get(int(node_id), [])
            if int(other) < int(node_id) and int(nodes[int(other)].ring) == previous_ring
        ]
        if not parent_candidates:
            parent_candidates = [
                (int(other), int(edge_type))
                for other, edge_type in by_node.get(int(node_id), [])
                if int(other) < int(node_id)
            ]
        if not parent_candidates:
            continue
        parent_idx, edge_type = min(
            parent_candidates,
            key=lambda row: (
                abs(int(nodes[row[0]].ring) - previous_ring),
                int(nodes[row[0]].final_color),
                int(row[0]),
                int(row[1]),
            ),
        )
        parent_edges[int(node_id)] = int(edge_type)
        parent_prev_colors[int(node_id)] = int(nodes[int(parent_idx)].prev_color)
    return tuple(parent_edges), tuple(parent_prev_colors), tuple(rings)


def pack_ring_trace(trace) -> PackedRingTrace:
    degrees = _trace_degrees_static(trace)
    edge_set = _edge_set_static(trace)
    nodes = tuple(trace.nodes)
    target_nodes = len(nodes)
    color_unresolved, color_current_edges, prev_events, intra_events = _decoder_timeline_static(trace)
    color_local_indices = tuple(range(1, len(nodes)))
    action_events = _action_events_static(trace, edge_set)
    plan_parent_edge_types, plan_parent_prev_colors, plan_rings = _plan_parent_fields_static(trace, edge_set)
    return PackedRingTrace(
        sequence=tuple(int(value) for value in trace.sequence),
        encoder_sequence=tuple(int(value) for value in (getattr(trace, "encoder_sequence", None) or trace.sequence)),
        center_color=int(trace.center_color),
        target_nodes=int(target_nodes),
        target_edges=int(len(edge_set)),
        final_colors=tuple(int(node.final_color) for node in nodes),
        prev_colors=tuple(int(node.prev_color) for node in nodes),
        labels=tuple(int(node.label) for node in nodes),
        rings=tuple(int(node.ring) for node in nodes),
        degrees=degrees,
        color_local_indices=color_local_indices,
        color_targets=tuple(int(node.final_color) for node in nodes[1:]),
        color_prev_colors=tuple(int(node.prev_color) for node in nodes[1:]),
        color_rings=tuple(int(node.ring) for node in nodes[1:]),
        color_unresolved_stubs=tuple(int(color_unresolved[idx]) for idx in color_local_indices),
        color_current_edges=tuple(int(color_current_edges[idx]) for idx in color_local_indices),
        prev_edge_left=tuple(int(row[0]) for row in prev_events),
        prev_edge_right=tuple(int(row[1]) for row in prev_events),
        prev_edge_types=tuple(int(row[2]) for row in prev_events),
        prev_edge_prefix_counts=tuple(int(row[3]) for row in prev_events),
        prev_edge_rings=tuple(int(row[4]) for row in prev_events),
        prev_edge_unresolved_stubs=tuple(int(row[5]) for row in prev_events),
        prev_edge_current_edges=tuple(int(row[6]) for row in prev_events),
        prev_edge_targets=tuple(float(row[7]) for row in prev_events),
        intra_edge_left=tuple(int(row[0]) for row in intra_events),
        intra_edge_right=tuple(int(row[1]) for row in intra_events),
        intra_edge_types=tuple(int(row[2]) for row in intra_events),
        intra_edge_prefix_counts=tuple(int(row[3]) for row in intra_events),
        intra_edge_rings=tuple(int(row[4]) for row in intra_events),
        intra_edge_unresolved_stubs=tuple(int(row[5]) for row in intra_events),
        intra_edge_current_edges=tuple(int(row[6]) for row in intra_events),
        intra_edge_targets=tuple(float(row[7]) for row in intra_events),
        action_source_indices=tuple(int(row[0]) for row in action_events),
        action_edge_types=tuple(int(row[1]) for row in action_events),
        action_needed_prev_colors=tuple(int(row[2]) for row in action_events),
        action_prefix_counts=tuple(int(row[3]) for row in action_events),
        action_rings=tuple(int(row[4]) for row in action_events),
        action_unresolved_stubs=tuple(int(row[5]) for row in action_events),
        action_next_unresolved_stubs=tuple(int(row[6]) for row in action_events),
        action_delta_unresolved_stubs=tuple(int(row[7]) for row in action_events),
        action_current_edges=tuple(int(row[8]) for row in action_events),
        action_targets=tuple(int(row[9]) for row in action_events),
        plan_parent_edge_types=tuple(int(v) for v in plan_parent_edge_types),
        plan_parent_prev_colors=tuple(int(v) for v in plan_parent_prev_colors),
        plan_rings=tuple(int(v) for v in plan_rings),
        n_nodes=len(nodes),
    )


def make_packed_ring_batch(traces) -> PackedRingBatch:
    if isinstance(traces, PackedRingBatch):
        return traces
    packed = [trace if isinstance(trace, PackedRingTrace) else pack_ring_trace(trace) for trace in traces]

    sequences = []
    encoder_sequences = []
    center_targets = []
    target_nodes = []
    target_edges = []
    node_final_colors = []
    node_prev_colors = []
    node_labels = []
    node_rings = []
    node_degrees = []
    color_node_indices = []
    color_graph_start_indices = []
    color_prefix_counts = []
    color_graph_indices = []
    color_targets = []
    color_prev_colors = []
    color_rings = []
    color_unresolved_stubs = []
    color_current_edges = []
    prev_edge_left_indices = []
    prev_edge_right_indices = []
    prev_edge_graph_indices = []
    prev_edge_types = []
    prev_edge_prefix_counts = []
    prev_edge_rings = []
    prev_edge_unresolved_stubs = []
    prev_edge_current_edges = []
    prev_edge_targets = []
    intra_edge_left_indices = []
    intra_edge_right_indices = []
    intra_edge_graph_indices = []
    intra_edge_types = []
    intra_edge_prefix_counts = []
    intra_edge_rings = []
    intra_edge_unresolved_stubs = []
    intra_edge_current_edges = []
    intra_edge_targets = []
    action_source_indices = []
    action_graph_start_indices = []
    action_graph_indices = []
    action_edge_types = []
    action_needed_prev_colors = []
    action_prefix_counts = []
    action_rings = []
    action_unresolved_stubs = []
    action_current_edges = []
    action_next_unresolved_stubs = []
    action_delta_unresolved_stubs = []
    action_targets = []
    plan_parent_edge_types = []
    plan_parent_prev_colors = []
    plan_rings = []

    node_offset = 0
    for graph_idx, trace in enumerate(packed):
        sequences.append(trace.sequence)
        encoder_sequences.append(trace.encoder_sequence)
        center_targets.append(trace.center_color)
        target_nodes.append(trace.target_nodes)
        target_edges.append(trace.target_edges)
        graph_start = node_offset

        node_final_colors.extend(trace.final_colors)
        node_prev_colors.extend(trace.prev_colors)
        node_labels.extend(trace.labels)
        node_rings.extend(trace.rings)
        node_degrees.extend(trace.degrees)

        for local_idx in trace.color_local_indices:
            color_node_indices.append(graph_start + int(local_idx))
            color_graph_start_indices.append(graph_start)
            color_prefix_counts.append(int(local_idx))
            color_graph_indices.append(graph_idx)
        color_targets.extend(trace.color_targets)
        color_prev_colors.extend(trace.color_prev_colors)
        color_rings.extend(trace.color_rings)
        color_unresolved_stubs.extend(trace.color_unresolved_stubs)
        color_current_edges.extend(trace.color_current_edges)

        prev_edge_left_indices.extend(graph_start + int(value) for value in trace.prev_edge_left)
        prev_edge_right_indices.extend(graph_start + int(value) for value in trace.prev_edge_right)
        prev_edge_graph_indices.extend([graph_idx] * len(trace.prev_edge_targets))
        prev_edge_types.extend(trace.prev_edge_types)
        prev_edge_prefix_counts.extend(trace.prev_edge_prefix_counts)
        prev_edge_rings.extend(trace.prev_edge_rings)
        prev_edge_unresolved_stubs.extend(trace.prev_edge_unresolved_stubs)
        prev_edge_current_edges.extend(trace.prev_edge_current_edges)
        prev_edge_targets.extend(trace.prev_edge_targets)

        intra_edge_left_indices.extend(graph_start + int(value) for value in trace.intra_edge_left)
        intra_edge_right_indices.extend(graph_start + int(value) for value in trace.intra_edge_right)
        intra_edge_graph_indices.extend([graph_idx] * len(trace.intra_edge_targets))
        intra_edge_types.extend(trace.intra_edge_types)
        intra_edge_prefix_counts.extend(trace.intra_edge_prefix_counts)
        intra_edge_rings.extend(trace.intra_edge_rings)
        intra_edge_unresolved_stubs.extend(trace.intra_edge_unresolved_stubs)
        intra_edge_current_edges.extend(trace.intra_edge_current_edges)
        intra_edge_targets.extend(trace.intra_edge_targets)

        action_source_indices.extend(graph_start + int(value) for value in trace.action_source_indices)
        action_graph_start_indices.extend([graph_start] * len(trace.action_targets))
        action_graph_indices.extend([graph_idx] * len(trace.action_targets))
        action_edge_types.extend(trace.action_edge_types)
        action_needed_prev_colors.extend(trace.action_needed_prev_colors)
        action_prefix_counts.extend(trace.action_prefix_counts)
        action_rings.extend(trace.action_rings)
        action_unresolved_stubs.extend(trace.action_unresolved_stubs)
        action_current_edges.extend(trace.action_current_edges)
        action_next_unresolved_stubs.extend(getattr(trace, "action_next_unresolved_stubs", trace.action_unresolved_stubs))
        action_delta_unresolved_stubs.extend(getattr(
            trace,
            "action_delta_unresolved_stubs",
            [0] * len(trace.action_targets),
        ))
        action_targets.extend(trace.action_targets)
        plan_parent_edge_types.append(tuple(getattr(trace, "plan_parent_edge_types", [0] * len(trace.final_colors))))
        plan_parent_prev_colors.append(tuple(getattr(trace, "plan_parent_prev_colors", [0] * len(trace.final_colors))))
        plan_rings.append(tuple(getattr(trace, "plan_rings", trace.rings)))

        node_offset += int(trace.n_nodes)

    return PackedRingBatch(
        sequences=sequences,
        encoder_sequences=encoder_sequences,
        center_targets=tuple(center_targets),
        target_nodes=tuple(target_nodes),
        target_edges=tuple(target_edges),
        node_final_colors=tuple(node_final_colors),
        node_prev_colors=tuple(node_prev_colors),
        node_labels=tuple(node_labels),
        node_rings=tuple(node_rings),
        node_degrees=tuple(node_degrees),
        color_node_indices=tuple(color_node_indices),
        color_graph_start_indices=tuple(color_graph_start_indices),
        color_prefix_counts=tuple(color_prefix_counts),
        color_graph_indices=tuple(color_graph_indices),
        color_targets=tuple(color_targets),
        color_prev_colors=tuple(color_prev_colors),
        color_rings=tuple(color_rings),
        color_unresolved_stubs=tuple(color_unresolved_stubs),
        color_current_edges=tuple(color_current_edges),
        prev_edge_left_indices=tuple(prev_edge_left_indices),
        prev_edge_right_indices=tuple(prev_edge_right_indices),
        prev_edge_graph_indices=tuple(prev_edge_graph_indices),
        prev_edge_types=tuple(prev_edge_types),
        prev_edge_prefix_counts=tuple(prev_edge_prefix_counts),
        prev_edge_rings=tuple(prev_edge_rings),
        prev_edge_unresolved_stubs=tuple(prev_edge_unresolved_stubs),
        prev_edge_current_edges=tuple(prev_edge_current_edges),
        prev_edge_targets=tuple(prev_edge_targets),
        intra_edge_left_indices=tuple(intra_edge_left_indices),
        intra_edge_right_indices=tuple(intra_edge_right_indices),
        intra_edge_graph_indices=tuple(intra_edge_graph_indices),
        intra_edge_types=tuple(intra_edge_types),
        intra_edge_prefix_counts=tuple(intra_edge_prefix_counts),
        intra_edge_rings=tuple(intra_edge_rings),
        intra_edge_unresolved_stubs=tuple(intra_edge_unresolved_stubs),
        intra_edge_current_edges=tuple(intra_edge_current_edges),
        intra_edge_targets=tuple(intra_edge_targets),
        action_source_indices=tuple(action_source_indices),
        action_graph_start_indices=tuple(action_graph_start_indices),
        action_graph_indices=tuple(action_graph_indices),
        action_edge_types=tuple(action_edge_types),
        action_needed_prev_colors=tuple(action_needed_prev_colors),
        action_prefix_counts=tuple(action_prefix_counts),
        action_rings=tuple(action_rings),
        action_unresolved_stubs=tuple(action_unresolved_stubs),
        action_current_edges=tuple(action_current_edges),
        action_next_unresolved_stubs=tuple(action_next_unresolved_stubs),
        action_delta_unresolved_stubs=tuple(action_delta_unresolved_stubs),
        action_targets=tuple(action_targets),
        plan_parent_edge_types=plan_parent_edge_types,
        plan_parent_prev_colors=plan_parent_prev_colors,
        plan_rings=plan_rings,
        n_graphs=len(packed),
    )

"""Construct and query the hierarchy between consecutive WL color stages."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class ColorHierarchy:
    """WL color-refinement hierarchy used for L1-L3 coloring masks."""

    metadata: dict
    final_stage: int | None = None
    support_only: bool = True

    def __post_init__(self):
        if self.final_stage is None:
            self.final_stage = int(self.metadata["n_wl_iters"]) - 1
        self.final_stage = int(self.final_stage)
        self.label_dict = self.metadata["label_dict"]
        self.inverse = []
        for stage_map in self.label_dict:
            inv = {}
            for signature, cid in stage_map.items():
                inv[int(cid)] = signature
            self.inverse.append(inv)

        self.parents: list[dict[int, int]] = [dict() for _ in self.inverse]
        self.children_by_parent: list[dict[int, set[int]]] = [defaultdict(set) for _ in self.inverse]
        self.neighbor_counts: list[dict[int, Counter]] = [dict() for _ in self.inverse]
        self.support: list[set[int]] = []
        color_probs = self.metadata.get("color_probs") or []

        for stage, inv in enumerate(self.inverse):
            if self.support_only and stage < len(color_probs) and color_probs[stage] is not None:
                probs = color_probs[stage]
                valid = {idx + 1 for idx, value in enumerate(probs) if float(value) > 0.0}
            else:
                valid = set(inv)
            self.support.append({int(v) for v in valid})

            if stage == 0:
                continue
            for cid, signature in inv.items():
                try:
                    parent, nbr_specs = signature
                except Exception:
                    continue
                parent = int(parent)
                cid = int(cid)
                self.parents[stage][cid] = parent
                self.children_by_parent[stage][parent].add(cid)
                counts = Counter()
                for nbr_prev_color, edge_attr in nbr_specs:
                    counts[(int(edge_attr), int(nbr_prev_color))] += 1
                self.neighbor_counts[stage][cid] = counts

    def valid_colors(self, stage: int) -> set[int]:
        stage = int(stage)
        if stage < 0 or stage >= len(self.support):
            return set()
        return set(self.support[stage])

    def is_valid_color(self, stage: int, color: int) -> bool:
        return int(color) in self.valid_colors(stage)

    def parent(self, stage: int, color: int) -> int | None:
        stage = int(stage)
        if stage <= 0 or stage >= len(self.parents):
            return None
        return self.parents[stage].get(int(color))

    def children(self, stage: int, parent_color: int) -> set[int]:
        stage = int(stage)
        if stage <= 0 or stage >= len(self.children_by_parent):
            return set()
        return set(self.children_by_parent[stage].get(int(parent_color), set()))

    def ancestors(self, stage: int, color: int) -> dict[int, int] | None:
        stage = int(stage)
        cur = int(color)
        out = {stage: cur}
        if not self.is_valid_color(stage, cur):
            return None
        for s in range(stage, 0, -1):
            parent = self.parent(s, cur)
            if parent is None or not self.is_valid_color(s - 1, parent):
                return None
            out[s - 1] = int(parent)
            cur = int(parent)
        return out

    def pair_compatible(self, color_a: int, color_b: int, edge_type: int, stage: int | None = None) -> bool:
        """Return whether final colors can be adjacent under all WL ancestors.

        L3 compatibility checks every level s>=1:
          (edge, ancestor_{s-1}(b)) in neighbors_s(ancestor_s(a))
          (edge, ancestor_{s-1}(a)) in neighbors_s(ancestor_s(b))
        """
        stage = self.final_stage if stage is None else int(stage)
        edge_type = int(edge_type)
        anc_a = self.ancestors(stage, int(color_a))
        anc_b = self.ancestors(stage, int(color_b))
        if anc_a is None or anc_b is None:
            return False
        for s in range(1, stage + 1):
            a_s = anc_a[s]
            b_s = anc_b[s]
            a_prev = anc_a[s - 1]
            b_prev = anc_b[s - 1]
            if self.neighbor_counts[s].get(a_s, Counter()).get((edge_type, b_prev), 0) <= 0:
                return False
            if self.neighbor_counts[s].get(b_s, Counter()).get((edge_type, a_prev), 0) <= 0:
                return False
        return True


def build_color_hierarchy(metadata, final_stage=None) -> ColorHierarchy:
    return ColorHierarchy(metadata=metadata, final_stage=final_stage)

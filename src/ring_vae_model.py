"""Transformer encoder and ring-structured autoregressive VAE decoder."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from ring_vae_actions import N_ACTIONS
from ring_vae_objectives import (
    color_balance_kl,
    completion_smooth_l1,
    masked_cross_entropy_batch,
    mean_masked_probs,
)
from ring_vae_tensors import (
    clamped_long_tensor as _clamped_long_tensor,
    long_tensor as _long_tensor,
    safe_ids as _safe_ids,
)
from ring_vae_trace_packing import (
    PackedRingBatch,
    PackedRingTrace,
    _edge_candidates_static,
    _edge_set_static,
    _trace_degrees_static,
    make_packed_ring_batch,
    pack_ring_trace,
)


MODEL_SCHEMA_VERSION = 5


class RingWLVAE(nn.Module):
    """Transformer VAE with latent-conditioned ring decoder heads."""

    def __init__(
        self,
        *,
        color_vocab_size: int,
        encoder_color_vocab_size: int | None = None,
        prev_color_vocab_size: int,
        label_vocab_size: int,
        edge_vocab_size: int,
        children_by_prev: dict[int, list[int]] | None = None,
        seed_color_ids: list[int] | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        max_ring: int = 128,
        count_vocab_size: int = 128,
        use_budget_context: bool = False,
        use_edge_budget_context: bool = False,
        use_action_head: bool = False,
        use_completion_head: bool = False,
        use_plan_context: bool = False,
        use_latent_plan: bool = False,
        plan_dropout: float = 0.10,
        train_color_probs: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.color_vocab_size = int(color_vocab_size)
        self.encoder_color_vocab_size = int(encoder_color_vocab_size or color_vocab_size)
        self.prev_color_vocab_size = int(prev_color_vocab_size)
        self.label_vocab_size = int(label_vocab_size)
        self.edge_vocab_size = int(edge_vocab_size)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.max_ring = int(max_ring)
        self.count_vocab_size = int(count_vocab_size)
        self.use_budget_context = bool(use_budget_context)
        self.use_edge_budget_context = bool(use_edge_budget_context and use_budget_context)
        self.use_action_head = bool(use_action_head)
        self.use_completion_head = bool(use_completion_head)
        self.use_plan_context = bool(use_plan_context)
        self.use_latent_plan = bool(use_latent_plan)
        self.plan_dropout = float(plan_dropout)
        self.cls_token_id = self.encoder_color_vocab_size + 1

        self.children_by_prev = {
            int(key): sorted({int(v) for v in values if int(v) > 0})
            for key, values in dict(children_by_prev or {}).items()
        }
        self.seed_color_ids = sorted({int(v) for v in (seed_color_ids or []) if int(v) > 0})

        h = self.hidden_dim
        self.color_emb = nn.Embedding(self.color_vocab_size + 2, h, padding_idx=0)
        self.encoder_color_emb = (
            nn.Embedding(self.encoder_color_vocab_size + 2, h, padding_idx=0)
            if self.encoder_color_vocab_size != self.color_vocab_size
            else None
        )
        self.prev_color_emb = nn.Embedding(self.prev_color_vocab_size + 1, h, padding_idx=0)
        # Molecular node labels are zero-based, so row 0 is a real atom type
        # rather than padding.  Trace padding is masked separately.
        self.label_emb = nn.Embedding(self.label_vocab_size + 1, h)
        self.edge_emb = nn.Embedding(self.edge_vocab_size + 1, h, padding_idx=0)
        self.ring_emb = nn.Embedding(self.max_ring + 1, h)
        self.count_emb = nn.Embedding(self.count_vocab_size, h)
        self.pos_emb = nn.Embedding(self.max_seq_len, h)
        if self.use_plan_context:
            self.plan_context_mlp = nn.Sequential(
                nn.Linear(4 * h, h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
            )
            self.plan_context_norm = nn.LayerNorm(h)
        else:
            self.plan_context_mlp = None
            self.plan_context_norm = None

        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(n_heads),
            dim_feedforward=4 * h,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(n_layers),
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.encoder_norm = nn.LayerNorm(h)
        self.mu_head = nn.Linear(h, self.latent_dim)
        self.logvar_head = nn.Linear(h, self.latent_dim)
        self.z_proj = nn.Sequential(nn.Linear(self.latent_dim, h), nn.SiLU(), nn.Linear(h, h))
        if self.use_budget_context:
            budget_fields = 10 if self.use_edge_budget_context else 5
            self.budget_mlp = nn.Sequential(
                nn.Linear(budget_fields * h, h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
                nn.SiLU(),
            )
        if self.use_latent_plan:
            self.plan_init = nn.Sequential(
                nn.Linear(2 * h, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
            self.plan_input_mlp = nn.Sequential(
                nn.Linear(4 * h, h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
            )
            self.plan_gru = nn.GRU(input_size=h, hidden_size=h, num_layers=1, batch_first=True)
            self.plan_norm = nn.LayerNorm(h)
            self.plan_color_head = nn.Linear(h, self.color_vocab_size + 1)
            self.plan_edge_head = nn.Linear(h, self.edge_vocab_size + 1)
            self.plan_prev_color_head = nn.Linear(h, self.prev_color_vocab_size + 1)
            self.plan_ring_head = nn.Linear(h, self.max_ring + 1)

        self.node_state_mlp = nn.Sequential(
            nn.Linear(5 * h, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.color_head = nn.Sequential(
            nn.Linear(4 * h, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, self.color_vocab_size + 1),
        )
        self.center_head = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, self.color_vocab_size + 1),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(4 * h, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        if self.use_action_head:
            self.action_head = nn.Sequential(
                nn.Linear(5 * h, h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, N_ACTIONS),
            )
        if self.use_completion_head:
            self.completion_head = nn.Sequential(
                nn.Linear(5 * h, h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        prior = torch.zeros((self.color_vocab_size + 1,), dtype=torch.float32)
        if train_color_probs is not None:
            values = torch.as_tensor(list(train_color_probs), dtype=torch.float32).view(-1)
            n = min(values.numel(), prior.numel())
            if n:
                prior[:n] = values[:n]
        if float(prior[1:].sum().item()) <= 0.0 and self.color_vocab_size > 0:
            prior[1:] = 1.0
        prior[0] = 0.0
        prior = prior / prior.sum().clamp_min(1.0e-12)
        self.register_buffer("train_color_probs", prior, persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_sequences(self, sequences: list[list[int]]):
        if not sequences:
            empty = torch.empty((0, self.latent_dim), dtype=torch.float32, device=self.device)
            return empty, empty
        max_events = max(1, self.max_seq_len - 1)
        rows = []
        lengths = []
        for seq in sequences:
            clipped = [int(v) for v in list(seq or [])[:max_events]]
            row = [self.cls_token_id] + clipped
            rows.append(row)
            lengths.append(len(row))
        batch_size = len(rows)
        seq_len = max(lengths)
        padded = []
        for row in rows:
            vals = _safe_ids(row, self.encoder_color_vocab_size + 1)
            padded.append(vals + [0] * (seq_len - len(vals)))
        tokens = torch.tensor(padded, dtype=torch.long, device=self.device)

        pos = torch.arange(seq_len, dtype=torch.long, device=self.device).clamp(0, self.max_seq_len - 1)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=self.device)
        padding_mask = torch.arange(seq_len, device=self.device).unsqueeze(0) >= lengths_t.unsqueeze(1)
        encoder_embedding = self.encoder_color_emb if self.encoder_color_emb is not None else self.color_emb
        hidden = self.encoder(encoder_embedding(tokens) + self.pos_emb(pos), src_key_padding_mask=padding_mask)
        cls = self.encoder_norm(hidden[:, 0])
        return self.mu_head(cls), self.logvar_head(cls)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _mask_logits(self, logits, valid_ids):
        valid = [int(v) for v in valid_ids if 0 < int(v) <= self.color_vocab_size]
        if not valid:
            return logits
        mask = torch.full_like(logits, -1.0e9)
        mask[..., valid] = logits[..., valid]
        return mask

    def _masked_ce(self, logits, target: int, valid_ids):
        target = int(target)
        valid = sorted({int(v) for v in valid_ids if 0 < int(v) <= self.color_vocab_size} | {target})
        logits = self._mask_logits(logits, valid)
        return F.cross_entropy(logits.view(1, -1), torch.tensor([target], dtype=torch.long, device=self.device))

    def _masked_ce_batch(self, logits, targets, valid_ids_by_row):
        return masked_cross_entropy_batch(logits, targets, valid_ids_by_row, self.color_vocab_size)

    def _masked_ce_by_prev_color(self, logits, targets, prev_colors):
        valid_ids = [self.children_by_prev.get(int(prev_color), []) for prev_color in prev_colors]
        return self._masked_ce_batch(logits, targets, valid_ids)

    def _mean_masked_probs(self, logits, valid_ids_by_row):
        return mean_masked_probs(logits, valid_ids_by_row, self.color_vocab_size)

    def z_context(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.z_proj(z)

    def _active_plan_dropout(self):
        if not self.training:
            return 0.0
        return max(0.0, min(float(self.plan_dropout), 1.0))

    def _infer_context_batch_size(self, values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    continue
                return int(value.numel())
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    def _context_ids(self, value, *, upper: int, batch_size: int):
        upper = max(0, int(upper))
        if value is None:
            return torch.zeros((int(batch_size),), dtype=torch.long, device=self.device)
        if isinstance(value, torch.Tensor):
            out = value.to(device=self.device, dtype=torch.long).view(-1)
            if out.numel() == 1 and int(batch_size) != 1:
                out = out.expand(int(batch_size))
        elif isinstance(value, (list, tuple)):
            out = _long_tensor(value, self.device).view(-1)
        else:
            out = torch.full((int(batch_size),), int(value), dtype=torch.long, device=self.device)
        if out.numel() != int(batch_size):
            if out.numel() == 1:
                out = out.expand(int(batch_size))
            else:
                raise ValueError(f"Budget context length mismatch: got {out.numel()} expected {batch_size}")
        return torch.clamp(out, min=0, max=upper)

    def budget_context(
        self,
        *,
        target_nodes=None,
        current_nodes=None,
        remaining_nodes=None,
        unresolved_stubs=None,
        ring=None,
        target_edges=None,
        current_edges=None,
        remaining_edges=None,
        batch_size: int | None = None,
    ):
        if not self.use_budget_context:
            return None
        if batch_size is None:
            batch_size = self._infer_context_batch_size(
                [
                    target_nodes,
                    current_nodes,
                    remaining_nodes,
                    unresolved_stubs,
                    ring,
                    target_edges,
                    current_edges,
                    remaining_edges,
                ]
            )
        batch_size = max(1, int(batch_size))
        target_t = self._context_ids(target_nodes, upper=self.count_vocab_size - 1, batch_size=batch_size)
        current_t = self._context_ids(current_nodes, upper=self.count_vocab_size - 1, batch_size=batch_size)
        remaining_t = self._context_ids(remaining_nodes, upper=self.count_vocab_size - 1, batch_size=batch_size)
        unresolved_t = self._context_ids(unresolved_stubs, upper=self.count_vocab_size - 1, batch_size=batch_size)
        ring_t = self._context_ids(ring, upper=self.max_ring, batch_size=batch_size)
        fields = [
            self.count_emb(target_t),
            self.count_emb(current_t),
            self.count_emb(remaining_t),
            self.count_emb(unresolved_t),
            self.ring_emb(ring_t),
        ]
        if self.use_edge_budget_context:
            target_edges_t = self._context_ids(
                target_edges,
                upper=self.count_vocab_size - 1,
                batch_size=batch_size,
            )
            current_edges_t = self._context_ids(
                current_edges,
                upper=self.count_vocab_size - 1,
                batch_size=batch_size,
            )
            if remaining_edges is None:
                remaining_edges_t = torch.clamp(
                    target_edges_t - current_edges_t,
                    min=0,
                    max=self.count_vocab_size - 1,
                )
            else:
                remaining_edges_t = self._context_ids(
                    remaining_edges,
                    upper=self.count_vocab_size - 1,
                    batch_size=batch_size,
                )
            current_degree_t = torch.round(
                20.0 * current_edges_t.to(dtype=torch.float32)
                / current_t.to(dtype=torch.float32).clamp_min(1.0)
            ).to(dtype=torch.long).clamp(0, self.count_vocab_size - 1)
            target_degree_t = torch.round(
                20.0 * target_edges_t.to(dtype=torch.float32)
                / target_t.to(dtype=torch.float32).clamp_min(1.0)
            ).to(dtype=torch.long).clamp(0, self.count_vocab_size - 1)
            fields.extend([
                self.count_emb(target_edges_t),
                self.count_emb(current_edges_t),
                self.count_emb(remaining_edges_t),
                self.count_emb(current_degree_t),
                self.count_emb(target_degree_t),
            ])
        return self.budget_mlp(torch.cat(fields, dim=-1))

    def plan_contexts_from_sequences(self, sequences, positions, graph_indices=None, *, dropout: float | None = None):
        if not self.use_plan_context:
            return None
        positions = [int(v) for v in positions]
        if graph_indices is None:
            graph_indices = list(range(len(positions)))
        else:
            graph_indices = [int(v) for v in graph_indices]
        if not positions:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32, device=self.device)

        sequences = [tuple(int(v) for v in (seq or ())) for seq in list(sequences or [])]
        if not sequences:
            return torch.zeros((len(positions), self.hidden_dim), dtype=torch.float32, device=self.device)
        max_len = max([1] + [len(seq) for seq in sequences])
        tokens = torch.zeros((len(sequences), max_len), dtype=torch.long, device=self.device)
        lengths = torch.zeros((len(sequences),), dtype=torch.long, device=self.device)
        for row, seq in enumerate(sequences):
            if not seq:
                continue
            values = torch.tensor(_safe_ids(seq[:max_len], self.color_vocab_size), dtype=torch.long, device=self.device)
            tokens[row, : values.numel()] = values
            lengths[row] = int(values.numel())

        dropout = self._active_plan_dropout() if dropout is None else float(dropout)
        if dropout > 0.0 and tokens.numel() > 0:
            valid = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            drop = torch.rand(tokens.shape, dtype=torch.float32, device=self.device) < float(dropout)
            tokens = tokens.masked_fill(valid & drop, 0)

        embeds = self.color_emb(tokens)
        prefix = torch.cumsum(embeds, dim=1)
        prefix = torch.cat([
            torch.zeros((prefix.size(0), 1, prefix.size(2)), dtype=prefix.dtype, device=self.device),
            prefix,
        ], dim=1)

        graph_t = _clamped_long_tensor(graph_indices, max(len(sequences) - 1, 0), self.device)
        pos_t = _clamped_long_tensor(positions, max_len, self.device)
        safe_pos_t = torch.minimum(pos_t, lengths.index_select(0, graph_t))
        has_prev = safe_pos_t > 0
        prev_ids = torch.zeros((len(positions),), dtype=torch.long, device=self.device)
        if has_prev.any():
            prev_ids[has_prev] = tokens[graph_t[has_prev], safe_pos_t[has_prev] - 1]
        prev_emb = self.color_emb(prev_ids)
        known_sum = prefix[graph_t, safe_pos_t]
        known_count = safe_pos_t.clamp_min(0)
        known_mean = known_sum / known_count.to(dtype=known_sum.dtype).clamp_min(1).unsqueeze(-1)
        known_count_t = known_count.clamp(0, self.count_vocab_size - 1).to(dtype=torch.long)
        pos_count_t = pos_t.clamp(0, self.count_vocab_size - 1).to(dtype=torch.long)
        plan_ctx = self.plan_context_mlp(torch.cat([
            prev_emb,
            known_mean,
            self.count_emb(known_count_t),
            self.count_emb(pos_count_t),
        ], dim=-1))
        return self.plan_context_norm(plan_ctx)

    def plan_context_from_sequence(self, sequence, position: int):
        ctx = self.plan_contexts_from_sequences([tuple(sequence or ())], [int(position)], dropout=0.0)
        return None if ctx is None else ctx[0]

    def condition_z(self, z_ctx, budget_ctx=None, plan_context=None):
        if self.use_budget_context and budget_ctx is not None:
            if z_ctx.dim() == 1 and budget_ctx.dim() == 2 and budget_ctx.size(0) == 1:
                budget_ctx = budget_ctx[0]
            elif z_ctx.dim() == 2 and budget_ctx.dim() == 1:
                budget_ctx = budget_ctx.unsqueeze(0).expand(z_ctx.size(0), -1)
            z_ctx = z_ctx + budget_ctx
        if self.use_plan_context and plan_context is not None:
            plan_context = plan_context.to(device=z_ctx.device, dtype=z_ctx.dtype)
            if z_ctx.dim() == 1 and plan_context.dim() == 2 and plan_context.size(0) == 1:
                plan_context = plan_context[0]
            elif z_ctx.dim() == 2 and plan_context.dim() == 1:
                plan_context = plan_context.unsqueeze(0).expand(z_ctx.size(0), -1)
            z_ctx = z_ctx + plan_context
        return z_ctx

    def condition_summary(self, summary, budget_ctx=None):
        if self.use_budget_context and budget_ctx is not None:
            if summary.dim() == 1 and budget_ctx.dim() == 2 and budget_ctx.size(0) == 1:
                budget_ctx = budget_ctx[0]
            elif summary.dim() == 2 and budget_ctx.dim() == 1:
                budget_ctx = budget_ctx.unsqueeze(0).expand(summary.size(0), -1)
            return summary + budget_ctx
        return summary

    def node_repr(self, *, final_color: int, prev_color: int, label: int, ring: int, degree: int):
        device = self.device
        final_t = torch.tensor([min(max(int(final_color), 0), self.color_vocab_size)], dtype=torch.long, device=device)
        prev_t = torch.tensor([min(max(int(prev_color), 0), self.prev_color_vocab_size)], dtype=torch.long, device=device)
        label_t = torch.tensor([min(max(int(label), 0), self.label_vocab_size)], dtype=torch.long, device=device)
        ring_t = torch.tensor([min(max(int(ring), 0), self.max_ring)], dtype=torch.long, device=device)
        degree_t = torch.tensor([min(max(int(degree), 0), self.count_vocab_size - 1)], dtype=torch.long, device=device)
        return self.node_state_mlp(torch.cat([
            self.color_emb(final_t),
            self.prev_color_emb(prev_t),
            self.label_emb(label_t),
            self.ring_emb(ring_t),
            self.count_emb(degree_t),
        ], dim=-1))[0]

    def node_reprs_from_values(self, final_colors, prev_colors, labels, rings, degrees):
        if not final_colors:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32, device=self.device)
        final_t = _clamped_long_tensor(final_colors, self.color_vocab_size, self.device)
        prev_t = _clamped_long_tensor(prev_colors, self.prev_color_vocab_size, self.device)
        label_t = _clamped_long_tensor(labels, self.label_vocab_size, self.device)
        ring_t = _clamped_long_tensor(rings, self.max_ring, self.device)
        degree_t = _clamped_long_tensor(degrees, self.count_vocab_size - 1, self.device)
        return self.node_state_mlp(torch.cat([
            self.color_emb(final_t),
            self.prev_color_emb(prev_t),
            self.label_emb(label_t),
            self.ring_emb(ring_t),
            self.count_emb(degree_t),
        ], dim=-1))

    def node_reprs_for_trace(self, trace, degrees):
        return self.node_reprs_from_values(
            (node.final_color for node in trace.nodes),
            (node.prev_color for node in trace.nodes),
            (node.label for node in trace.nodes),
            (node.ring for node in trace.nodes),
            degrees,
        )

    def graph_summary(self, node_reprs: list[torch.Tensor], z_ctx: torch.Tensor, budget_ctx=None):
        if node_reprs:
            summary = torch.stack(node_reprs, dim=0).mean(dim=0)
        else:
            summary = torch.zeros_like(z_ctx)
        return self.condition_summary(summary, budget_ctx)

    def center_logits(self, z_ctx, budget_ctx=None, plan_context=None):
        return self.center_head(self.condition_z(z_ctx, budget_ctx, plan_context))

    def color_logits(self, z_ctx, *, prev_color: int, ring: int, summary: torch.Tensor, budget_ctx=None, plan_context=None):
        device = self.device
        prev_t = torch.tensor([min(max(int(prev_color), 0), self.prev_color_vocab_size)], dtype=torch.long, device=device)
        ring_t = torch.tensor([min(max(int(ring), 0), self.max_ring)], dtype=torch.long, device=device)
        x = torch.cat([
            self.condition_z(z_ctx, budget_ctx, plan_context),
            self.prev_color_emb(prev_t)[0],
            self.ring_emb(ring_t)[0],
            self.condition_summary(summary, budget_ctx),
        ], dim=-1)
        return self.color_head(x)

    def edge_logit(self, z_ctx, left_repr, right_repr, edge_type: int, budget_ctx=None, plan_context=None):
        edge_t = torch.tensor([min(max(int(edge_type), 0), self.edge_vocab_size)], dtype=torch.long, device=self.device)
        x = torch.cat([self.condition_z(z_ctx, budget_ctx, plan_context), left_repr, right_repr, self.edge_emb(edge_t)[0]], dim=-1)
        return self.edge_head(x).view(())

    def action_logits(
        self,
        z_ctx,
        source_repr,
        *,
        edge_type: int,
        needed_prev_color: int,
        summary: torch.Tensor,
        budget_ctx=None,
        plan_context=None,
    ):
        if not self.use_action_head:
            raise RuntimeError("This checkpoint was not trained with an action head.")
        edge_t = torch.tensor([min(max(int(edge_type), 0), self.edge_vocab_size)], dtype=torch.long, device=self.device)
        prev_t = torch.tensor(
            [min(max(int(needed_prev_color), 0), self.prev_color_vocab_size)],
            dtype=torch.long,
            device=self.device,
        )
        x = torch.cat([
            self.condition_z(z_ctx, budget_ctx, plan_context),
            source_repr,
            self.edge_emb(edge_t)[0],
            self.prev_color_emb(prev_t)[0],
            self.condition_summary(summary, budget_ctx),
        ], dim=-1)
        return self.action_head(x)

    def edge_inputs_for_candidates(self, z_ctx, node_reprs, candidates, budget_ctx=None, plan_context=None):
        if not candidates:
            return torch.empty((0, 4 * self.hidden_dim), dtype=torch.float32, device=self.device)
        left_t = _clamped_long_tensor((u for u, _v, _edge in candidates), max(node_reprs.size(0) - 1, 0), self.device)
        right_t = _clamped_long_tensor((v for _u, v, _edge in candidates), max(node_reprs.size(0) - 1, 0), self.device)
        edge_t = _clamped_long_tensor((edge for _u, _v, edge in candidates), self.edge_vocab_size, self.device)
        return torch.cat([
            self.condition_z(z_ctx.unsqueeze(0).expand(len(candidates), -1), budget_ctx, plan_context),
            node_reprs[left_t],
            node_reprs[right_t],
            self.edge_emb(edge_t),
        ], dim=-1)

    def edge_logits_for_candidates(self, z_ctx, node_reprs, candidates, budget_ctx=None, plan_context=None):
        edge_inputs = self.edge_inputs_for_candidates(
            z_ctx,
            node_reprs,
            candidates,
            budget_ctx=budget_ctx,
            plan_context=plan_context,
        )
        if edge_inputs.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        return self.edge_head(edge_inputs).view(-1)

    def edge_logits_from_indices(
        self,
        z_ctxs,
        node_reprs,
        graph_indices,
        left_indices,
        right_indices,
        edge_types,
        *,
        target_nodes=None,
        target_edges=None,
        current_edges=None,
        prefix_counts=None,
        unresolved_stubs=None,
        rings=None,
        plan_contexts=None,
    ):
        if not edge_types:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        graph_t = _clamped_long_tensor(graph_indices, max(z_ctxs.size(0) - 1, 0), self.device)
        left_t = _clamped_long_tensor(left_indices, max(node_reprs.size(0) - 1, 0), self.device)
        right_t = _clamped_long_tensor(right_indices, max(node_reprs.size(0) - 1, 0), self.device)
        edge_t = _clamped_long_tensor(edge_types, self.edge_vocab_size, self.device)
        budget_ctx = None
        if self.use_budget_context:
            if prefix_counts is None:
                prefix_counts = [0] * len(edge_types)
            if target_nodes is None:
                target_nodes = [0] * len(edge_types)
            if target_edges is None:
                target_edges = [0] * len(edge_types)
            if current_edges is None:
                current_edges = [0] * len(edge_types)
            if unresolved_stubs is None:
                unresolved_stubs = [0] * len(edge_types)
            if rings is None:
                rings = [0] * len(edge_types)
            prefix_t = _clamped_long_tensor(prefix_counts, self.count_vocab_size - 1, self.device)
            target_t = _clamped_long_tensor(target_nodes, self.count_vocab_size - 1, self.device)
            remaining_t = torch.clamp(target_t - prefix_t, min=0, max=self.count_vocab_size - 1)
            budget_ctx = self.budget_context(
                target_nodes=target_t,
                current_nodes=prefix_t,
                remaining_nodes=remaining_t,
                unresolved_stubs=unresolved_stubs,
                ring=rings,
                target_edges=target_edges,
                current_edges=current_edges,
                batch_size=len(edge_types),
            )
        x = torch.cat([
            self.condition_z(z_ctxs[graph_t], budget_ctx, plan_contexts),
            node_reprs[left_t],
            node_reprs[right_t],
            self.edge_emb(edge_t),
        ], dim=-1)
        return self.edge_head(x).view(-1)

    def _trace_degrees(self, trace):
        cached = getattr(trace, "_degree_cache", None)
        if cached is not None:
            return cached
        degrees = list(_trace_degrees_static(trace))
        try:
            trace._degree_cache = degrees
        except Exception:
            pass
        return degrees

    def _edge_set(self, trace):
        cached = getattr(trace, "_edge_set_cache", None)
        if cached is not None:
            return cached
        edge_set = _edge_set_static(trace)
        try:
            trace._edge_set_cache = edge_set
        except Exception:
            pass
        return edge_set

    def _possible_edge_types(self, left, right):
        left_counts = Counter((int(edge), int(prev)) for edge, prev in left.required)
        right_counts = Counter((int(edge), int(prev)) for edge, prev in right.required)
        edge_types = []
        for edge_type, prev_color in left_counts:
            if int(prev_color) != int(right.prev_color):
                continue
            if right_counts.get((int(edge_type), int(left.prev_color)), 0) > 0:
                edge_types.append(int(edge_type))
        return sorted(set(edge_types))

    def _edge_candidates(self, trace, *, adjacent_only: bool):
        return list(_edge_candidates_static(trace, adjacent_only=adjacent_only))

    def _trace_color_inputs(self, trace, z_ctx, node_reprs):
        n_nodes = int(node_reprs.size(0))
        if n_nodes <= 1:
            return None, None, []
        prefix_sums = torch.cumsum(node_reprs, dim=0)
        summaries = torch.zeros_like(node_reprs)
        counts = torch.arange(1, n_nodes, dtype=node_reprs.dtype, device=self.device).unsqueeze(-1)
        summaries[1:] = prefix_sums[:-1] / counts

        color_nodes = trace.nodes[1:]
        prev_t = _clamped_long_tensor((node.prev_color for node in color_nodes), self.prev_color_vocab_size, self.device)
        ring_t = _clamped_long_tensor((node.ring for node in color_nodes), self.max_ring, self.device)
        x = torch.cat([
            z_ctx.unsqueeze(0).expand(n_nodes - 1, -1),
            self.prev_color_emb(prev_t),
            self.ring_emb(ring_t),
            summaries[1:],
        ], dim=-1)
        targets = _long_tensor((node.final_color for node in color_nodes), self.device)
        valid_ids = [self.children_by_prev.get(int(node.prev_color), []) for node in color_nodes]
        return x, targets, valid_ids

    def _plan_targets(self, batch: PackedRingBatch):
        batch_size = int(batch.n_graphs)
        max_len = max([1] + [len(seq) for seq in batch.sequences])
        colors = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        parent_edges = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        parent_prev = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        rings = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        for row, seq in enumerate(batch.sequences):
            seq = tuple(int(v) for v in seq)
            n = min(len(seq), max_len)
            lengths[row] = int(n)
            if n <= 0:
                continue
            colors[row, :n] = torch.tensor(_safe_ids(seq[:n], self.color_vocab_size), dtype=torch.long, device=self.device)
            edge_values = tuple(batch.plan_parent_edge_types[row] if row < len(batch.plan_parent_edge_types) else ())
            prev_values = tuple(batch.plan_parent_prev_colors[row] if row < len(batch.plan_parent_prev_colors) else ())
            ring_values = tuple(batch.plan_rings[row] if row < len(batch.plan_rings) else ())
            if edge_values:
                parent_edges[row, :min(n, len(edge_values))] = _clamped_long_tensor(edge_values[:n], self.edge_vocab_size, self.device)
            if prev_values:
                parent_prev[row, :min(n, len(prev_values))] = _clamped_long_tensor(prev_values[:n], self.prev_color_vocab_size, self.device)
            if ring_values:
                rings[row, :min(n, len(ring_values))] = _clamped_long_tensor(ring_values[:n], self.max_ring, self.device)
        return colors, parent_edges, parent_prev, rings, lengths

    def plan_logits(self, z_ctxs, target_nodes, *, teacher_colors=None, max_len: int | None = None):
        if not self.use_latent_plan:
            raise RuntimeError("This checkpoint was not trained with a latent plan decoder.")
        if z_ctxs.dim() == 1:
            z_ctxs = z_ctxs.unsqueeze(0)
        batch_size = int(z_ctxs.size(0))
        target_t = self._context_ids(
            target_nodes,
            upper=self.count_vocab_size - 1,
            batch_size=batch_size,
        )
        if teacher_colors is not None:
            max_len = int(teacher_colors.size(1))
        else:
            max_len = int(max_len or int(target_t.max().item() if target_t.numel() else 1))
        max_len = max(1, min(int(max_len), int(self.max_seq_len)))

        start = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        if teacher_colors is not None:
            shifted = torch.cat([
                start,
                teacher_colors[:, : max_len - 1].to(device=self.device, dtype=torch.long),
            ], dim=1)
        else:
            shifted = start.expand(batch_size, max_len)
        shifted = torch.clamp(shifted, min=0, max=self.color_vocab_size)
        pos_t = torch.arange(max_len, dtype=torch.long, device=self.device).clamp(0, self.max_seq_len - 1)
        pos_t = pos_t.unsqueeze(0).expand(batch_size, max_len)
        target_emb = self.count_emb(target_t).unsqueeze(1).expand(batch_size, max_len, -1)
        z_seq = z_ctxs.unsqueeze(1).expand(batch_size, max_len, -1)
        inputs = self.plan_input_mlp(torch.cat([
            z_seq,
            self.color_emb(shifted),
            self.pos_emb(pos_t),
            target_emb,
        ], dim=-1))
        init = self.plan_init(torch.cat([z_ctxs, self.count_emb(target_t)], dim=-1)).unsqueeze(0)
        hidden, _state = self.plan_gru(inputs, init)
        hidden = self.plan_norm(hidden)
        return {
            "color": self.plan_color_head(hidden),
            "edge": self.plan_edge_head(hidden),
            "prev_color": self.plan_prev_color_head(hidden),
            "ring": self.plan_ring_head(hidden),
        }

    def compute_plan_loss(self, z_ctxs, batch: PackedRingBatch):
        if not self.use_latent_plan:
            zero = z_ctxs.sum() * 0.0
            return zero
        colors, parent_edges, parent_prev, rings, lengths = self._plan_targets(batch)
        logits = self.plan_logits(z_ctxs, batch.target_nodes, teacher_colors=colors, max_len=colors.size(1))
        mask = torch.arange(colors.size(1), device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not mask.any():
            return z_ctxs.sum() * 0.0
        rows = mask.reshape(-1)
        return (
            F.cross_entropy(logits["color"].reshape(-1, self.color_vocab_size + 1)[rows], colors.reshape(-1)[rows])
            + F.cross_entropy(logits["edge"].reshape(-1, self.edge_vocab_size + 1)[rows], parent_edges.reshape(-1)[rows])
            + F.cross_entropy(logits["prev_color"].reshape(-1, self.prev_color_vocab_size + 1)[rows], parent_prev.reshape(-1)[rows])
            + F.cross_entropy(logits["ring"].reshape(-1, self.max_ring + 1)[rows], rings.reshape(-1)[rows])
        ) / 4.0

    @torch.no_grad()
    def sample_latent_plan(
        self,
        z_ctx,
        target_nodes: int,
        *,
        temperature: float = 1.0,
        greedy: bool = False,
    ):
        if not self.use_latent_plan:
            return [], []
        if z_ctx.dim() == 1:
            z_ctx = z_ctx.unsqueeze(0)
        n = max(1, min(int(target_nodes), int(self.max_seq_len)))
        target = torch.tensor([int(target_nodes)], dtype=torch.long, device=self.device)
        sampled = torch.zeros((1, n), dtype=torch.long, device=self.device)
        actions = []
        temperature = max(float(temperature), 1.0e-6)
        for pos in range(n):
            logits = self.plan_logits(z_ctx, target, teacher_colors=sampled, max_len=n)
            color_logits = logits["color"][0, pos]
            if greedy:
                color = int(torch.argmax(color_logits).item())
                edge = int(torch.argmax(logits["edge"][0, pos]).item())
                prev = int(torch.argmax(logits["prev_color"][0, pos]).item())
                ring = int(torch.argmax(logits["ring"][0, pos]).item())
            else:
                color = int(torch.multinomial(torch.softmax(color_logits / temperature, dim=-1), 1).item())
                edge = int(torch.multinomial(torch.softmax(logits["edge"][0, pos] / temperature, dim=-1), 1).item())
                prev = int(torch.multinomial(torch.softmax(logits["prev_color"][0, pos] / temperature, dim=-1), 1).item())
                ring = int(torch.multinomial(torch.softmax(logits["ring"][0, pos] / temperature, dim=-1), 1).item())
            if color <= 0:
                color = 1
            sampled[0, pos] = int(color)
            actions.append({
                "final_color": int(color),
                "parent_edge_type": max(0, int(edge)),
                "parent_prev_color": max(0, int(prev)),
                "ring": max(0, int(ring)),
            })
        return [int(v) for v in sampled[0].tolist() if int(v) > 0], actions

    def compute_loss(
        self,
        traces,
        *,
        beta: float = 1.0,
        completion_weight: float = 0.0,
        color_balance_weight: float = 0.0,
        plan_loss_weight: float = 0.0,
        action_class_weights=None,
    ):
        return self.compute_loss_packed(
            make_packed_ring_batch(traces),
            beta=beta,
            completion_weight=completion_weight,
            color_balance_weight=color_balance_weight,
            plan_loss_weight=plan_loss_weight,
            action_class_weights=action_class_weights,
        )

    def compute_loss_packed(
        self,
        batch: PackedRingBatch,
        *,
        beta: float = 1.0,
        completion_weight: float = 0.0,
        color_balance_weight: float = 0.0,
        plan_loss_weight: float = 0.0,
        action_class_weights=None,
    ):
        mu, logvar = self.encode_sequences(batch.encoder_sequences)
        z = self.reparameterize(mu, logvar)
        z_ctxs = self.z_context(z)
        plan_dropout = self._active_plan_dropout()
        graph_plan_ctxs = self.plan_contexts_from_sequences(
            batch.sequences,
            [0] * batch.n_graphs,
            dropout=plan_dropout,
        ) if self.use_plan_context else None

        seed_ids = self.seed_color_ids or list(range(1, self.color_vocab_size + 1))
        center_targets = _long_tensor(batch.center_targets, self.device)
        target_nodes_t = _clamped_long_tensor(batch.target_nodes, self.count_vocab_size - 1, self.device)
        target_edges_t = _clamped_long_tensor(batch.target_edges, self.count_vocab_size - 1, self.device)
        center_budget = self.budget_context(
            target_nodes=target_nodes_t,
            current_nodes=[0] * batch.n_graphs,
            remaining_nodes=target_nodes_t,
            unresolved_stubs=[0] * batch.n_graphs,
            ring=[0] * batch.n_graphs,
            target_edges=target_edges_t,
            current_edges=[0] * batch.n_graphs,
            batch_size=batch.n_graphs,
        )
        center_logits = self.center_logits(z_ctxs, budget_ctx=center_budget, plan_context=graph_plan_ctxs)
        center_loss = self._masked_ce_batch(
            center_logits,
            center_targets,
            [seed_ids] * batch.n_graphs,
        )
        color_balance_terms = []
        if float(color_balance_weight) != 0.0:
            color_balance_terms.append((
                self._mean_masked_probs(center_logits, [seed_ids] * batch.n_graphs),
                int(batch.n_graphs),
            ))

        node_reprs = self.node_reprs_from_values(
            batch.node_final_colors,
            batch.node_prev_colors,
            batch.node_labels,
            batch.node_rings,
            batch.node_degrees,
        )

        zero = mu.sum() * 0.0
        if batch.color_targets:
            color_node_t = _long_tensor(batch.color_node_indices, self.device)
            graph_start_t = _long_tensor(batch.color_graph_start_indices, self.device)
            prefix_counts_t = _long_tensor(batch.color_prefix_counts, self.device)
            color_graph_t = _clamped_long_tensor(batch.color_graph_indices, max(z_ctxs.size(0) - 1, 0), self.device)
            color_prev_t = _clamped_long_tensor(batch.color_prev_colors, self.prev_color_vocab_size, self.device)
            color_ring_t = _clamped_long_tensor(batch.color_rings, self.max_ring, self.device)
            color_target_t = target_nodes_t[color_graph_t]
            color_target_edges_t = target_edges_t[color_graph_t]
            color_remaining_t = torch.clamp(color_target_t - prefix_counts_t, min=0, max=self.count_vocab_size - 1)
            color_plan_ctxs = self.plan_contexts_from_sequences(
                batch.sequences,
                batch.color_prefix_counts,
                graph_indices=batch.color_graph_indices,
                dropout=plan_dropout,
            ) if self.use_plan_context else None
            color_budget = self.budget_context(
                target_nodes=color_target_t,
                current_nodes=prefix_counts_t,
                remaining_nodes=color_remaining_t,
                unresolved_stubs=batch.color_unresolved_stubs,
                ring=batch.color_rings,
                target_edges=color_target_edges_t,
                current_edges=batch.color_current_edges,
                batch_size=len(batch.color_targets),
            )

            cumsum = torch.cumsum(node_reprs, dim=0)
            cumsum = torch.cat([torch.zeros((1, self.hidden_dim), dtype=node_reprs.dtype, device=self.device), cumsum], dim=0)
            summaries = cumsum[color_node_t] - cumsum[graph_start_t]
            summaries = summaries / prefix_counts_t.to(dtype=node_reprs.dtype).clamp_min(1).unsqueeze(-1)

            color_logits = self.color_head(torch.cat([
                self.condition_z(z_ctxs[color_graph_t], color_budget, color_plan_ctxs),
                self.prev_color_emb(color_prev_t),
                self.ring_emb(color_ring_t),
                self.condition_summary(summaries, color_budget),
            ], dim=-1))
            color_loss = self._masked_ce_batch(
                color_logits,
                _long_tensor(batch.color_targets, self.device),
                [self.children_by_prev.get(int(prev_color), []) for prev_color in batch.color_prev_colors],
            )
            if float(color_balance_weight) != 0.0:
                color_balance_terms.append((
                    self._mean_masked_probs(
                        color_logits,
                        [self.children_by_prev.get(int(prev_color), []) for prev_color in batch.color_prev_colors],
                    ),
                    int(len(batch.color_prev_colors)),
                ))
        else:
            color_loss = zero
        if batch.prev_edge_targets:
            prev_edge_loss = F.binary_cross_entropy_with_logits(
                self.edge_logits_from_indices(
                    z_ctxs,
                    node_reprs,
                    batch.prev_edge_graph_indices,
                    batch.prev_edge_left_indices,
                    batch.prev_edge_right_indices,
                    batch.prev_edge_types,
                    target_nodes=[batch.target_nodes[int(idx)] for idx in batch.prev_edge_graph_indices],
                    target_edges=[batch.target_edges[int(idx)] for idx in batch.prev_edge_graph_indices],
                    current_edges=batch.prev_edge_current_edges,
                    prefix_counts=batch.prev_edge_prefix_counts,
                    unresolved_stubs=batch.prev_edge_unresolved_stubs,
                    rings=batch.prev_edge_rings,
                    plan_contexts=self.plan_contexts_from_sequences(
                        batch.sequences,
                        batch.prev_edge_prefix_counts,
                        graph_indices=batch.prev_edge_graph_indices,
                        dropout=plan_dropout,
                    ) if self.use_plan_context else None,
                ),
                torch.tensor(batch.prev_edge_targets, dtype=torch.float32, device=self.device),
            )
        else:
            prev_edge_loss = zero
        if batch.intra_edge_targets:
            intra_edge_loss = F.binary_cross_entropy_with_logits(
                self.edge_logits_from_indices(
                    z_ctxs,
                    node_reprs,
                    batch.intra_edge_graph_indices,
                    batch.intra_edge_left_indices,
                    batch.intra_edge_right_indices,
                    batch.intra_edge_types,
                    target_nodes=[batch.target_nodes[int(idx)] for idx in batch.intra_edge_graph_indices],
                    target_edges=[batch.target_edges[int(idx)] for idx in batch.intra_edge_graph_indices],
                    current_edges=batch.intra_edge_current_edges,
                    prefix_counts=batch.intra_edge_prefix_counts,
                    unresolved_stubs=batch.intra_edge_unresolved_stubs,
                    rings=batch.intra_edge_rings,
                    plan_contexts=self.plan_contexts_from_sequences(
                        batch.sequences,
                        batch.intra_edge_prefix_counts,
                        graph_indices=batch.intra_edge_graph_indices,
                        dropout=plan_dropout,
                    ) if self.use_plan_context else None,
                ),
                torch.tensor(batch.intra_edge_targets, dtype=torch.float32, device=self.device),
            )
        else:
            intra_edge_loss = zero

        action_loss = zero
        completion_loss = zero
        if (self.use_action_head or self.use_completion_head) and batch.action_targets:
            action_graph_t = _clamped_long_tensor(batch.action_graph_indices, max(z_ctxs.size(0) - 1, 0), self.device)
            action_source_t = _clamped_long_tensor(batch.action_source_indices, max(node_reprs.size(0) - 1, 0), self.device)
            action_start_t = _long_tensor(batch.action_graph_start_indices, self.device)
            action_prefix_t = _long_tensor(batch.action_prefix_counts, self.device)
            action_edge_t = _clamped_long_tensor(batch.action_edge_types, self.edge_vocab_size, self.device)
            action_prev_t = _clamped_long_tensor(
                batch.action_needed_prev_colors,
                self.prev_color_vocab_size,
                self.device,
            )
            action_target_nodes = target_nodes_t[action_graph_t]
            action_target_edges = target_edges_t[action_graph_t]
            action_remaining_t = torch.clamp(action_target_nodes - action_prefix_t, min=0, max=self.count_vocab_size - 1)
            action_plan_ctxs = self.plan_contexts_from_sequences(
                batch.sequences,
                batch.action_prefix_counts,
                graph_indices=batch.action_graph_indices,
                dropout=plan_dropout,
            ) if self.use_plan_context else None
            action_budget = self.budget_context(
                target_nodes=action_target_nodes,
                current_nodes=action_prefix_t,
                remaining_nodes=action_remaining_t,
                unresolved_stubs=batch.action_unresolved_stubs,
                ring=batch.action_rings,
                target_edges=action_target_edges,
                current_edges=batch.action_current_edges,
                batch_size=len(batch.action_targets),
            )

            cumsum = torch.cumsum(node_reprs, dim=0)
            cumsum = torch.cat([torch.zeros((1, self.hidden_dim), dtype=node_reprs.dtype, device=self.device), cumsum], dim=0)
            prefix_end_t = action_start_t + action_prefix_t
            prefix_end_t = torch.clamp(prefix_end_t, min=0, max=node_reprs.size(0))
            action_summaries = cumsum[prefix_end_t] - cumsum[action_start_t]
            action_summaries = action_summaries / action_prefix_t.to(dtype=node_reprs.dtype).clamp_min(1).unsqueeze(-1)

            action_inputs = torch.cat([
                self.condition_z(z_ctxs[action_graph_t], action_budget, action_plan_ctxs),
                node_reprs[action_source_t],
                self.edge_emb(action_edge_t),
                self.prev_color_emb(action_prev_t),
                self.condition_summary(action_summaries, action_budget),
            ], dim=-1)
            if self.use_action_head:
                action_weight_t = None
                if action_class_weights is not None:
                    weights = list(action_class_weights)
                    if len(weights) != N_ACTIONS:
                        raise ValueError(f"Expected {N_ACTIONS} action weights, got {len(weights)}")
                    action_weight_t = torch.tensor(weights, dtype=action_inputs.dtype, device=self.device)
                action_loss = F.cross_entropy(
                    self.action_head(action_inputs),
                    _long_tensor(batch.action_targets, self.device),
                    weight=action_weight_t,
                )
            if self.use_completion_head:
                completion_loss = completion_smooth_l1(
                    self.completion_head(action_inputs).view(-1),
                    _long_tensor(batch.action_delta_unresolved_stubs, self.device),
                )

        if color_balance_terms and float(color_balance_weight) != 0.0:
            total_weight = sum(max(0, int(weight)) for _probs, weight in color_balance_terms)
            pred_color_probs = sum(
                probs * float(max(0, int(weight)))
                for probs, weight in color_balance_terms
            ) / max(float(total_weight), 1.0)
            target_color_probs = self.train_color_probs.to(device=pred_color_probs.device, dtype=pred_color_probs.dtype)
            color_balance_loss = color_balance_kl(pred_color_probs, target_color_probs)
        else:
            color_balance_loss = zero

        plan_loss = self.compute_plan_loss(z_ctxs, batch) if self.use_latent_plan else zero

        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        recon = (
            center_loss
            + color_loss
            + prev_edge_loss
            + intra_edge_loss
            + action_loss
            + float(completion_weight) * completion_loss
            + float(color_balance_weight) * color_balance_loss
            + float(plan_loss_weight) * plan_loss
        )
        total = recon + float(beta) * kl
        return {
            "loss": total,
            "recon": recon.detach(),
            "center": center_loss.detach(),
            "color": color_loss.detach(),
            "prev_edge": prev_edge_loss.detach(),
            "intra_edge": intra_edge_loss.detach(),
            "action": action_loss.detach(),
            "completion": completion_loss.detach(),
            "color_balance": color_balance_loss.detach(),
            "plan": plan_loss.detach(),
            "kl": kl.detach(),
            "n_graphs": batch.n_graphs,
            "n_prev_edges": batch.n_prev_edges,
            "n_intra_edges": batch.n_intra_edges,
            "n_actions": batch.n_actions,
        }

    def sample_latent(self, n_samples: int):
        return torch.randn((int(n_samples), self.latent_dim), dtype=torch.float32, device=self.device)

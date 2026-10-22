"""Masked structural, completion, and color-balance training objectives."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def masked_cross_entropy_batch(logits, targets, valid_ids_by_row, color_vocab_size: int):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    targets = targets.to(dtype=torch.long, device=logits.device)
    grouped_rows = defaultdict(list)
    target_values = targets.tolist()
    for row, (target, valid_ids) in enumerate(zip(target_values, valid_ids_by_row)):
        valid = sorted({
            int(v)
            for v in list(valid_ids or []) + [int(target)]
            if 0 < int(v) <= int(color_vocab_size)
        })
        grouped_rows[tuple(valid)].append(row)

    total = logits.sum() * 0.0
    n_rows = 0
    for valid, rows in grouped_rows.items():
        rows_t = torch.tensor(rows, dtype=torch.long, device=logits.device)
        if valid:
            valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
            local_logits = logits.index_select(0, rows_t).index_select(1, valid_t)
            local_index = {int(value): idx for idx, value in enumerate(valid)}
            local_targets = torch.tensor(
                [local_index[int(target_values[row])] for row in rows],
                dtype=torch.long,
                device=logits.device,
            )
            total = total + F.cross_entropy(local_logits, local_targets, reduction="sum")
        else:
            total = total + F.cross_entropy(
                logits.index_select(0, rows_t),
                targets.index_select(0, rows_t),
                reduction="sum",
            )
        n_rows += len(rows)
    return total / max(1, n_rows)


def mean_masked_probs(logits, valid_ids_by_row, color_vocab_size: int):
    if logits.numel() == 0:
        return torch.zeros((int(color_vocab_size) + 1,), dtype=logits.dtype, device=logits.device)
    grouped_rows = defaultdict(list)
    for row, valid_ids in enumerate(valid_ids_by_row):
        valid = tuple(sorted({
            int(v)
            for v in list(valid_ids or [])
            if 0 < int(v) <= int(color_vocab_size)
        }))
        grouped_rows[valid].append(row)

    probs_sum = torch.zeros((logits.size(-1),), dtype=logits.dtype, device=logits.device)
    n_rows = 0
    for valid, rows in grouped_rows.items():
        rows_t = torch.tensor(rows, dtype=torch.long, device=logits.device)
        local = logits.index_select(0, rows_t)
        if valid:
            masked = torch.full_like(local, -1.0e9)
            valid_t = torch.tensor(valid, dtype=torch.long, device=logits.device)
            masked.index_copy_(1, valid_t, local.index_select(1, valid_t))
            local = masked
        probs_sum = probs_sum + torch.softmax(local, dim=-1).sum(dim=0)
        n_rows += len(rows)
    probs = probs_sum / max(1, n_rows)
    probs[0] = 0.0
    return probs / probs.sum().clamp_min(1.0e-12)


def color_balance_kl(pred_color_probs, target_color_probs):
    return F.kl_div(
        pred_color_probs.clamp_min(1.0e-12).log(),
        target_color_probs,
        reduction="sum",
    )


def completion_smooth_l1(predicted_delta_unresolved, delta_unresolved_stubs):
    targets = delta_unresolved_stubs.to(dtype=predicted_delta_unresolved.dtype)
    return F.smooth_l1_loss(predicted_delta_unresolved.view(-1), targets.view(-1))

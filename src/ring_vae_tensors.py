"""Small tensor-construction helpers used by the decoder."""

from __future__ import annotations

import torch


def long_tensor(values, device):
    return torch.tensor([int(v) for v in values], dtype=torch.long, device=device)


def clamped_long_tensor(values, upper, device):
    upper = max(0, int(upper))
    return torch.clamp(long_tensor(values, device), min=0, max=upper)


def safe_ids(values, upper):
    upper = max(0, int(upper))
    return [min(max(int(v), 0), upper) for v in values]

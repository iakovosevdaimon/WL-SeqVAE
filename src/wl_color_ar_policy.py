"""Optional neural policy for autoregressive WL-color selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


class WLColorGRUPolicy(nn.Module):
    """Causal next-WL-color policy used inside the constructive witness sampler."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bos_token_id = self.vocab_size + 1
        self.embedding = nn.Embedding(self.vocab_size + 2, int(embed_dim), padding_idx=0)
        self.rnn = nn.GRU(
            int(embed_dim),
            int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(int(hidden_dim), self.vocab_size + 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids.clamp(min=0, max=self.bos_token_id))
        hidden, _state = self.rnn(emb)
        logits = self.output(hidden)
        logits[..., 0] = -1e9
        return logits

    @torch.no_grad()
    def next_logits(self, prefix_color_ids: Sequence[int], device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        prefix = [self.bos_token_id] + [
            int(cid) for cid in prefix_color_ids
            if 1 <= int(cid) <= self.vocab_size
        ]
        input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        return self(input_ids)[0, -1].detach().cpu()


@dataclass
class NeuralColorPolicy:
    model: WLColorGRUPolicy
    device: torch.device
    temperature: float = 1.0
    empirical_weight_bias: float = 0.0

    def __post_init__(self):
        self.model.to(self.device)
        self.model.eval()
        self.temperature = max(float(self.temperature), 1e-6)
        self.empirical_weight_bias = float(self.empirical_weight_bias)

    def choose(self, specs, prefix_color_ids: Sequence[int], rng: random.Random):
        if not specs:
            raise ValueError("Cannot sample from an empty candidate set.")
        logits = self.model.next_logits(prefix_color_ids, device=self.device)

        scored = []
        for spec in specs:
            cid = int(spec.color_id)
            if 0 <= cid < logits.numel():
                score = float(logits[cid].item()) / self.temperature
                if self.empirical_weight_bias:
                    score += self.empirical_weight_bias * math.log(max(float(spec.weight), 1e-12))
                if math.isfinite(score):
                    scored.append((spec, score))

        if not scored:
            return _weighted_choice(specs, rng)

        max_score = max(score for _spec, score in scored)
        weights = [math.exp(score - max_score) for _spec, score in scored]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            return _weighted_choice(specs, rng)

        r = rng.random() * total
        acc = 0.0
        for (spec, _score), weight in zip(scored, weights):
            acc += weight
            if acc >= r:
                return spec
        return scored[-1][0]


def _weighted_choice(specs, rng: random.Random):
    total = sum(max(0.0, float(s.weight)) for s in specs)
    if total <= 0.0:
        return rng.choice(specs)
    r = rng.random() * total
    acc = 0.0
    for spec in specs:
        acc += max(0.0, float(spec.weight))
        if acc >= r:
            return spec
    return specs[-1]


def load_neural_color_policy(
    checkpoint_path,
    *,
    device: str = "auto",
    temperature: float = 1.0,
    empirical_weight_bias: float = 0.0,
) -> NeuralColorPolicy:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    model = WLColorGRUPolicy(**model_args)
    model.load_state_dict(ckpt["model_state"])
    return NeuralColorPolicy(
        model=model,
        device=device,
        temperature=temperature,
        empirical_weight_bias=empirical_weight_bias,
    )

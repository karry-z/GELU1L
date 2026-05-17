from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import pandas as pd
import torch

from gelu_1l.constants import HOOK_POST


@torch.no_grad()
def get_post_activations(model, tokens: torch.Tensor, hook_name: str = HOOK_POST) -> torch.Tensor:
    _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=hook_name)
    return cache[hook_name]


@dataclass
class FeatureStatsTracker:
    dict_size: int

    def __post_init__(self) -> None:
        self.activation_counts = torch.zeros(self.dict_size, dtype=torch.float64)
        self.activation_sums = torch.zeros(self.dict_size, dtype=torch.float64)
        self.max_values = torch.full((self.dict_size,), -float("inf"), dtype=torch.float64)
        self.total_tokens = 0

    def update(self, feature_acts: torch.Tensor) -> None:
        acts = feature_acts.detach().float().cpu()
        self.total_tokens += int(acts.shape[0])
        self.activation_counts += (acts > 0).sum(dim=0)
        self.activation_sums += acts.sum(dim=0)
        self.max_values = torch.maximum(self.max_values, acts.max(dim=0).values.double())

    def to_frame(self) -> pd.DataFrame:
        counts = self.activation_counts.numpy()
        means = self.activation_sums.numpy() / max(self.total_tokens, 1)
        max_values = self.max_values.numpy()
        rates = counts / max(self.total_tokens, 1)
        rank_score = max_values * np.log1p(counts)
        return pd.DataFrame(
            {
                "feature_id": np.arange(self.dict_size),
                "activation_count": counts.astype(np.int64),
                "activation_rate": rates,
                "mean_activation": means,
                "max_activation": max_values,
                "rank_score": rank_score,
                "total_tokens": self.total_tokens,
            }
        ).sort_values("rank_score", ascending=False)


class TopOccurrenceCollector:
    def __init__(self, limit: int):
        self.limit = limit
        self._counter = 0
        self.heap: list[tuple[float, int, dict[str, object]]] = []

    def add(self, activation: float, row: dict[str, object]) -> None:
        self._counter += 1
        item = (activation, self._counter, row)
        if len(self.heap) < self.limit:
            heappush(self.heap, item)
            return
        if activation > self.heap[0][0]:
            heappop(self.heap)
            heappush(self.heap, item)

    def rows(self) -> list[dict[str, object]]:
        return [row for _, _, row in sorted(self.heap, key=lambda item: item[0], reverse=True)]


def format_token_window(str_tokens: list[str], pos: int, radius: int = 8) -> dict[str, str]:
    start = max(pos - radius, 0)
    end = min(pos + radius + 1, len(str_tokens))
    left = "".join(str_tokens[start:pos])
    token = str_tokens[pos]
    right = "".join(str_tokens[pos + 1 : end])
    window = f"{left}<<{token}>>{right}"
    return {"left_context": left, "token": token, "right_context": right, "window": window}


def occurrence_row(
    model,
    tokens: torch.Tensor,
    texts: list[str],
    text_offset: int,
    batch_row: int,
    pos: int,
    activation: float,
    radius: int,
) -> dict[str, object]:
    str_tokens = model.to_str_tokens(tokens[batch_row].detach().cpu())
    window = format_token_window(str_tokens, pos=pos, radius=radius)
    text_id = text_offset + batch_row
    return {
        "text_id": text_id,
        "position": pos,
        "activation": activation,
        "source_text_preview": texts[batch_row][:240].replace("\n", "\\n"),
        **window,
    }


def feature_logit_effects(model, sae, feature_id: int, top_k: int = 20) -> pd.DataFrame:
    direction = sae.W_dec[feature_id].detach().to(model.W_U.device).float()
    resid_direction = direction @ model.blocks[0].mlp.W_out.detach().float()
    logit_effects = resid_direction @ model.W_U.detach().float()
    pos_values, pos_ids = torch.topk(logit_effects, k=top_k)
    neg_values, neg_ids = torch.topk(-logit_effects, k=top_k)

    rows = []
    for rank, (token_id, value) in enumerate(zip(pos_ids.tolist(), pos_values.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "direction": "positive",
                "token_id": token_id,
                "token": model.tokenizer.decode([token_id]),
                "logit_effect": value,
            }
        )
    for rank, (token_id, value) in enumerate(zip(neg_ids.tolist(), neg_values.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "direction": "negative",
                "token_id": token_id,
                "token": model.tokenizer.decode([token_id]),
                "logit_effect": -value,
            }
        )
    return pd.DataFrame(rows)


def neuron_alignment(sae, feature_id: int, top_k: int = 25) -> pd.DataFrame:
    weights = sae.W_dec[feature_id].detach().float().cpu()
    values, neuron_ids = torch.topk(weights.abs(), k=top_k)
    rows = []
    for rank, (neuron_id, abs_weight) in enumerate(zip(neuron_ids.tolist(), values.tolist()), start=1):
        weight = float(weights[neuron_id])
        rows.append(
            {
                "rank": rank,
                "neuron_id": neuron_id,
                "decoder_weight": weight,
                "abs_decoder_weight": abs_weight,
                "sign": "positive" if weight >= 0 else "negative",
            }
        )
    return pd.DataFrame(rows)


def top_feature_acts(feature_acts: torch.Tensor, k: int) -> tuple[list[float], list[int]]:
    flat = feature_acts.detach().flatten().float()
    values, indices = torch.topk(flat, k=min(k, flat.numel()))
    return values.cpu().tolist(), indices.cpu().tolist()

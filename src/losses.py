from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossOutput:
    total: Tensor
    bce: Tensor
    top1: Tensor


def _segment_logsumexp(values: Tensor, index: Tensor, num_segments: int) -> tuple[Tensor, Tensor]:
    """Vectorized logsumexp for variable-size graph segments."""
    maximum = values.new_full((num_segments,), -torch.inf)
    maximum.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

    finite_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    shifted = values - finite_maximum.index_select(0, index)
    exponential_sum = values.new_zeros((num_segments,))
    exponential_sum.index_add_(0, index, torch.exp(shifted))
    present = exponential_sum > 0
    result = finite_maximum + torch.log(exponential_sum.clamp_min(torch.finfo(values.dtype).tiny))
    return result, present


def multilabel_top1_loss(
    logits: Tensor,
    labels: Tensor,
    border_graph_index: Tensor,
    graph_count: int,
    *,
    bce_weight: float,
    top1_weight: float,
    pos_weight: float,
) -> LossOutput:
    """Graph-balanced BCE plus differentiable top-1 set loss.

    Each graph contributes equally even when it has a different number of
    border nodes. For a graph with positive set P, the ranking term is:

        logsumexp(all logits) - logsumexp(logits in P)

    This is ``-log(sum_{i in P} softmax(logits)_i)``.
    """
    # Keep loss arithmetic in float32 when automatic mixed precision is active.
    logits = logits.float().reshape(-1)
    labels = labels.float().reshape(-1)
    border_graph_index = border_graph_index.long().reshape(-1)
    if logits.numel() != labels.numel() or logits.numel() != border_graph_index.numel():
        raise ValueError("Logits, labels and border graph indices must have the same length.")
    if graph_count <= 0:
        raise ValueError("graph_count must be positive.")

    scalar_pos_weight = logits.new_tensor(float(pos_weight))
    node_bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=scalar_pos_weight,
        reduction="none",
    )
    bce_sum = logits.new_zeros((graph_count,))
    bce_sum.index_add_(0, border_graph_index, node_bce)
    node_count = torch.bincount(border_graph_index, minlength=graph_count).to(logits.dtype)
    if torch.any(node_count == 0):
        raise ValueError("Every graph must contain at least one border node.")
    bce = (bce_sum / node_count).mean()

    all_lse, _ = _segment_logsumexp(logits, border_graph_index, graph_count)
    positive_mask = labels > 0.5
    if torch.any(positive_mask):
        positive_lse, has_positive = _segment_logsumexp(
            logits[positive_mask],
            border_graph_index[positive_mask],
            graph_count,
        )
        top1_per_graph = torch.where(
            has_positive,
            all_lse - positive_lse,
            torch.zeros_like(all_lse),
        )
    else:
        top1_per_graph = torch.zeros_like(all_lse)
    top1 = top1_per_graph.mean()

    total = float(bce_weight) * bce + float(top1_weight) * top1
    return LossOutput(total=total, bce=bce, top1=top1)

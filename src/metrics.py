from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class MetricAccumulator:
    threshold_logit: float = 0.0
    top1_correct: Optional[Tensor] = None
    graph_count: int = 0
    confusion_counts: Optional[Tensor] = None

    # Accumulate exact metrics on the device without synchronizing per batch.
    # Parameters: logits: Border activity scores; labels: binary target labels
    #   border_ptr: Border segment offsets
    # Returns: None; updates state or writes output.
    def update(self, logits: Tensor, labels: Tensor, border_ptr: Tensor) -> None:
        """Accumulate exact metrics on the device without synchronizing per batch."""
        logits = logits.detach().float().reshape(-1)
        labels = labels.detach().reshape(-1) > 0.5
        border_ptr = border_ptr.detach().to(device=logits.device, dtype=torch.long)
        predictions = logits >= float(self.threshold_logit)

        batch_counts = torch.stack(
            (
                torch.sum(predictions & labels),
                torch.sum((~predictions) & (~labels)),
                torch.sum(predictions & (~labels)),
                torch.sum((~predictions) & labels),
            )
        ).to(dtype=torch.float64)
        if self.confusion_counts is None:
            self.confusion_counts = batch_counts
        else:
            self.confusion_counts += batch_counts

        batch_graph_count = border_ptr.numel() - 1
        lengths = border_ptr[1:] - border_ptr[:-1]
        graph_index = torch.repeat_interleave(
            torch.arange(batch_graph_count, device=logits.device), lengths
        )

        # Select the first maximum in every graph, matching torch.argmax's tie
        # behavior while avoiding the previous Python loop over all graphs.
        graph_maximum = logits.new_full((batch_graph_count,), -torch.inf)
        graph_maximum.scatter_reduce_(
            0, graph_index, logits, reduce="amax", include_self=True
        )
        positions = torch.arange(logits.numel(), device=logits.device)
        candidates = torch.where(
            logits == graph_maximum.index_select(0, graph_index),
            positions,
            torch.full_like(positions, logits.numel()),
        )
        top_indices = torch.full(
            (batch_graph_count,),
            logits.numel(),
            dtype=torch.long,
            device=logits.device,
        )
        top_indices.scatter_reduce_(
            0, graph_index, candidates, reduce="amin", include_self=True
        )
        batch_top1_correct = labels.index_select(0, top_indices).sum().to(torch.float64)
        if self.top1_correct is None:
            self.top1_correct = batch_top1_correct
        else:
            self.top1_correct += batch_top1_correct
        self.graph_count += batch_graph_count

    # Transfer the accumulated scalars to the CPU in one synchronization.
    # Parameters: none (uses instance/enclosing state).
    # Returns: aggregated metric dictionary.
    def compute(self) -> Dict[str, float]:
        """Transfer the accumulated scalars to the CPU in one synchronization."""
        eps = 1e-12
        if self.confusion_counts is None or self.top1_correct is None:
            top1_correct = tp = tn = fp = fn = 0.0
        else:
            values = torch.cat(
                (self.top1_correct.reshape(1), self.confusion_counts)
            ).cpu().tolist()
            top1_correct, tp, tn, fp, fn = (float(value) for value in values)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return {
            "top1_any_positive": top1_correct / max(self.graph_count, 1),
            "binary_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall + eps),
        }

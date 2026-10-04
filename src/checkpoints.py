from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

STATE_DICT_KEYS = ("model_state", "model_state_dict", "state_dict")
STATE_KEY_PREFIXES = ("module.", "_orig_mod.", "model.")


def load_checkpoint(path: str | Path, map_location: torch.device | str) -> Any:
    """Load a checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(checkpoint: Any) -> dict[str, Tensor]:
    """Extract and normalize a model state dictionary from common checkpoints."""

    state: Any = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in STATE_DICT_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break

    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise TypeError("The checkpoint does not contain a valid model state dictionary.")

    normalized = dict(state)
    while normalized:
        matched_prefix = next(
            (
                prefix
                for prefix in STATE_KEY_PREFIXES
                if all(key.startswith(prefix) for key in normalized)
            ),
            None,
        )
        if matched_prefix is None:
            break
        normalized = {
            key[len(matched_prefix) :]: value for key, value in normalized.items()
        }
    return normalized


def load_model_state(model: nn.Module, checkpoint: Any, *, strict: bool = True) -> None:
    """Load normalized checkpoint weights into ``model``."""

    model.load_state_dict(extract_model_state(checkpoint), strict=strict)


def model_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return portable keys for plain, compiled or data-parallel models."""

    unwrapped = model
    while True:
        candidate = getattr(unwrapped, "_orig_mod", None)
        if candidate is None:
            candidate = getattr(unwrapped, "module", None)
        if not isinstance(candidate, nn.Module):
            break
        unwrapped = candidate
    return dict(unwrapped.state_dict())

from __future__ import annotations

import math
from copy import deepcopy
from typing import Mapping, Sequence

from .graph_schema import canonicalize_graph_features


DEFAULT_TEMPORAL_LOG_REFERENCE = 100.0
DEFAULT_RESOURCE_CLIP = True
_PROXIMITY_EPS = 1e-8


def _temporal_log_scale(value: object, reference: float) -> float:
    """Compress a non-negative temporal quantity without clipping.

    The reference is a fixed model scale, not a maximum learned from training.
    Values above the reference are intentionally allowed to exceed 1.0 so that
    inference can distinguish durations/deltas outside the training range.
    """

    value = float(value)
    reference = float(reference)
    if value < 0.0:
        raise ValueError("Temporal quantities must be non-negative.")
    if reference <= 0.0:
        raise ValueError("The temporal log reference must be positive.")
    return math.log1p(value) / math.log1p(reference)


def _resource_ratio_vector(
    values: Sequence[object],
    capacities: Sequence[object],
    *,
    clip: bool,
) -> list[float]:
    """Normalize resource quantities by the capacity of the current instance."""

    if len(values) != len(capacities):
        raise ValueError(
            f"Resource vector width differs from capacity width: "
            f"{len(values)} != {len(capacities)}."
        )

    normalized: list[float] = []
    for value, capacity in zip(values, capacities):
        value = float(value)
        capacity = float(capacity)
        if capacity < 0.0:
            raise ValueError("Resource capacities must be non-negative.")
        if value < 0.0:
            raise ValueError("Resource quantities must be non-negative.")
        if capacity == 0.0:
            normalized.append(0.0)
            continue
        ratio = value / capacity
        if clip:
            ratio = max(0.0, min(1.0, ratio))
        normalized.append(ratio)
    return normalized


def _resource_capacities(graph: Mapping) -> list[float]:
    metadata = graph.get("graph", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Graph metadata must be an object.")
    capacities = metadata.get("resource_capacities")
    if not isinstance(capacities, Sequence) or isinstance(capacities, (str, bytes)):
        raise ValueError(
            "Graph metadata is missing 'resource_capacities'. "
            "Generate graphs with the instance resource capacities before normalization."
        )
    return [float(value) for value in capacities]


def _is_resource_vector(field: str) -> bool:
    return "resource" in field.lower()


def _add_local_proximity_weights(edges: list[dict]) -> None:
    """Add inverse-distance weights normalized independently for each Border.

    ``delta_task_relation_border`` edges are Other -> Border in the dynamic
    graph. For every destination Border, inverse raw deltas are normalized to
    sum to one. This keeps proximity local and independent of instance scale or
    the number of descendants. The raw delta is used here *before* log scaling.
    """

    groups: dict[object, list[tuple[dict, float]]] = {}
    for edge in edges:
        if "delta_task_relation_border" not in edge:
            continue
        delta = float(edge["delta_task_relation_border"])
        if delta < 0.0:
            raise ValueError("Delta values must be non-negative.")
        inverse = 1.0 / max(delta, _PROXIMITY_EPS)
        groups.setdefault(edge.get("target"), []).append((edge, inverse))

    for group in groups.values():
        total = sum(inverse for _, inverse in group)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("Could not normalize local proximity weights.")
        for edge, inverse in group:
            edge["proximity_weight"] = inverse / total


def normalize_graph(
    graph: Mapping,
    *,
    temporal_reference: float = DEFAULT_TEMPORAL_LOG_REFERENCE,
    resource_clip: bool = DEFAULT_RESOURCE_CLIP,
) -> dict:
    """Normalize one dynamic graph for both training and inference.

    Policy:
    - resources: quantity / instance capacity, clipped to [0, 1];
    - durations and deltas: log1p(value) / log1p(reference), with NO clipping;
    - legacy Other -> Border proximity: inverse raw delta normalized locally
      so incoming proximity weights sum to one for every Border.
    """

    graph = canonicalize_graph_features(graph)
    output = deepcopy(dict(graph))
    capacities = _resource_capacities(graph)

    normalized_nodes = []
    for raw_node in graph.get("nodes", []):
        node = deepcopy(raw_node)
        for field, value in list(node.items()):
            if field == "solution":
                continue
            if field.startswith("duration"):
                node[field] = _temporal_log_scale(value, temporal_reference)
            elif isinstance(value, list) and _is_resource_vector(field):
                node[field] = _resource_ratio_vector(
                    value,
                    capacities,
                    clip=resource_clip,
                )
        normalized_nodes.append(node)
    output["nodes"] = normalized_nodes

    edge_key = "edges" if "edges" in graph else "links"
    normalized_edges = [deepcopy(raw_edge) for raw_edge in graph.get(edge_key, [])]

    # Proximity must be computed from the raw temporal distance, not from the
    # log-scaled feature that the GNN will receive.
    _add_local_proximity_weights(normalized_edges)

    for edge in normalized_edges:
        for field, value in list(edge.items()):
            if field.startswith("delta_"):
                edge[field] = _temporal_log_scale(value, temporal_reference)
            elif field.startswith("duration_"):
                edge[field] = _temporal_log_scale(value, temporal_reference)
            elif isinstance(value, list) and _is_resource_vector(field):
                edge[field] = _resource_ratio_vector(
                    value,
                    capacities,
                    clip=resource_clip,
                )
    output[edge_key] = normalized_edges
    return output


class GraphNormalizer:
    """Reusable transform shared by graph generation and makespan rollout."""

    def __init__(
        self,
        *,
        temporal_reference: float = DEFAULT_TEMPORAL_LOG_REFERENCE,
        resource_clip: bool = DEFAULT_RESOURCE_CLIP,
    ) -> None:
        temporal_reference = float(temporal_reference)
        if temporal_reference <= 0.0:
            raise ValueError("temporal_reference must be positive.")
        self.temporal_reference = temporal_reference
        self.resource_clip = bool(resource_clip)

    def __call__(self, graph: Mapping) -> dict:
        return normalize_graph(
            graph,
            temporal_reference=self.temporal_reference,
            resource_clip=self.resource_clip,
        )

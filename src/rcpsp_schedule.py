from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Iterable, List, Mapping, MutableSequence, Sequence

import numpy as np

from .rcpsp_instance import (
    INF,
    MODEL_RESOURCE_COUNT,
    MRCPSPInstance,
    create_distance_matrix,
    build_activity_resource_data,
)

if TYPE_CHECKING:
    import torch
    from torch import nn


TEMPORAL_EDGE_SCALES = (1, 8, 64, 512)


@dataclass
class BorderPrediction:
    node_id: int | str
    logit: float
    probability: float
    scores: List[dict]


@dataclass
class ScheduleResult:
    makespan: int
    solution: List[List[int]]
    decisions: List[dict]

    # Export the schedule result as a dictionary.
    # Parameters: none (uses instance/enclosing state).
    # Returns: dict.
    def as_dict(self) -> dict:
        return asdict(self)


# Deterministic choice, not a model score (probability 1, sentinel logit 0).
# Parameters: node: selected activity ID
# Returns: BorderPrediction.
def forced_border_prediction(node: int) -> BorderPrediction:
    """Deterministic choice, not a model score (probability 1, sentinel logit 0)."""
    return BorderPrediction(int(node), 0.0, 1.0, [])


# In-place max-plus Floyd-Warshall used by the original scheduling code.
# Parameters: matrix: temporal distance matrix; inf: unreachable-distance sentinel
# Returns: None; updates state or writes output.
def longest_path_closure(matrix: MutableSequence[MutableSequence[int]], inf: int = INF) -> None:
    """In-place max-plus Floyd-Warshall used by the original scheduling code."""

    size = len(matrix)
    for middle in range(size):
        for source in range(size):
            source_middle = matrix[source][middle]
            if source_middle == inf:
                continue
            for target in range(size):
                middle_target = matrix[middle][target]
                if middle_target == inf:
                    continue
                candidate = source_middle + middle_target
                if candidate > matrix[source][target]:
                    matrix[source][target] = candidate


# Propagate changes in the source row through an already closed matrix.
# Parameters: matrix: temporal distance matrix; inf: unreachable-distance sentinel
# Returns: None; updates state or writes output.
def propagate_earliest_start_times(
    matrix: MutableSequence[MutableSequence[int]],
    inf: int = INF,
) -> None:
    """Propagate changes in the source row through an already closed matrix.

    ``initialize_schedule_state`` computes the full max-plus closure once. During
    serial schedule generation only ``matrix[0]`` changes, when resource conflicts
    delay an available activity. Because all pairwise longest-path distances are
    already present in the remaining rows, updating the source row needs O(n^2)
    work instead of another O(n^3) Floyd--Warshall pass.
    """

    size = len(matrix)
    current_starts = list(matrix[0])
    propagated = list(current_starts)

    for source in range(1, size):
        source_start = current_starts[source]
        if source_start == inf:
            continue
        source_row = matrix[source]
        for target in range(1, size):
            distance = source_row[target]
            if distance == inf:
                continue
            candidate = source_start + distance
            if candidate > propagated[target]:
                propagated[target] = candidate

    matrix[0][:] = propagated


# Extract an activity resource-demand vector.
# Parameters: nodes_dict: activity durations and demands; node_id: activity ID
#   num_resources: resource channel count
# Returns: List[int].
def _resources_for_node(
    nodes_dict: Mapping[str, Sequence[int]],
    node_id: int,
    num_resources: int = MODEL_RESOURCE_COUNT,
) -> List[int]:
    return [
        int(nodes_dict.get(f"R{resource + 1}", [0] * len(nodes_dict["nodes"]))[node_id])
        for resource in range(num_resources)
    ]


# Read residual capacities at one time instant.
# Parameters: resource_time: residual resource-time profile; time_index: time instant
#   num_resources: resource channel count
# Returns: List[int].
def _resources_at_time(
    resource_time: np.ndarray,
    time_index: int,
    num_resources: int = MODEL_RESOURCE_COUNT,
) -> List[int]:
    if not 0 <= time_index < resource_time.shape[0]:
        raise IndexError(
            f"Time index {time_index} is outside resource horizon "
            f"[0, {resource_time.shape[0] - 1}]. Increase max_time."
        )
    values = resource_time[time_index].astype(int).tolist()
    return values[:num_resources] + [0] * max(0, num_resources - len(values))


# Initialize activity sets, temporal constraints and resource availability.
# Parameters: instance: parsed RCPSP instance; max_time: schedule horizon
# Returns: features, capacities, matrices, resource profile and activity sets.
def initialize_schedule_state(
    instance: MRCPSPInstance,
    max_time: int,
) -> tuple[dict, List[int], List[List[int]], np.ndarray, List[List[int]], set[int], set[int]]:
    if max_time <= 0:
        raise ValueError("max_time must be positive.")

    nodes_dict, resource_levels = build_activity_resource_data(instance)
    matrix = create_distance_matrix(instance)
    longest_path_closure(matrix)

    solution = [[0, 0]]
    border: set[int] = set()
    others: set[int] = set()
    for node in range(1, len(matrix)):
        if matrix[0][node] == 0:
            border.add(node)
        else:
            others.add(node)

    if not border:
        raise RuntimeError("No initial border node was found after the dummy source.")

    resource_time = np.zeros((max_time, len(resource_levels)), dtype=np.int64)
    for resource_index, capacity in enumerate(resource_levels):
        resource_time[:, resource_index] = int(capacity)

    return (
        nodes_dict,
        resource_levels,
        matrix,
        resource_time,
        solution,
        border,
        others,
    )


# Schedule an activity and refresh eligibility and resource availability.
# Parameters: nodes_dict: activity durations and demands; resource_levels: resource capacities
#   matrix: temporal distance matrix; node: selected activity ID; border: eligible activity IDs
#   others: non-Border activities; resource_time: residual resource-time profile
# Returns: updated Border and Other sets.
def update_schedule_state(
    nodes_dict: Mapping[str, Sequence[int]],
    resource_levels: Sequence[int],
    matrix: MutableSequence[MutableSequence[int]],
    node: int,
    border: set[int],
    others: set[int],
    resource_time: np.ndarray,
) -> tuple[set[int], set[int]]:
    if node not in border:
        raise ValueError(f"Selected node {node} is not in the current border {sorted(border)}.")

    border = set(border)
    others = set(others)
    border.remove(node)

    newly_available: set[int] = set()
    for candidate in range(len(matrix)):
        if candidate not in others or matrix[node][candidate] == INF:
            continue
        has_unscheduled_predecessor = False
        for predecessor in range(len(matrix)):
            if predecessor in border or predecessor in others:
                if matrix[predecessor][candidate] != INF:
                    has_unscheduled_predecessor = True
                    break
        if not has_unscheduled_predecessor:
            newly_available.add(candidate)

    border |= newly_available
    others -= newly_available

    start = int(matrix[0][node])
    duration = int(nodes_dict["duration"][node])
    end = start + duration
    if start < 0 or end > resource_time.shape[0]:
        raise RuntimeError(
            f"Activity {node} requires time interval [{start}, {end}), but max_time="
            f"{resource_time.shape[0]}. Increase the prediction horizon."
        )

    for time_index in range(start, end):
        for resource_index in range(len(resource_levels)):
            resource_time[time_index, resource_index] -= int(
                nodes_dict[f"R{resource_index + 1}"][node]
            )
            if resource_time[time_index, resource_index] < 0:
                raise RuntimeError(
                    f"Negative resource availability after scheduling activity {node} "
                    f"at time {time_index}, resource {resource_index}."
                )

    # Shift every newly/currently available activity until all resources fit.
    for candidate in sorted(border):
        feasible = False
        while not feasible:
            feasible = True
            candidate_start = int(matrix[0][candidate])
            candidate_duration = int(nodes_dict["duration"][candidate])
            candidate_end = candidate_start + candidate_duration
            if candidate_start < 0 or candidate_end > resource_time.shape[0]:
                raise RuntimeError(
                    f"Activity {candidate} exceeds max_time={resource_time.shape[0]}."
                )
            for time_index in range(candidate_start, candidate_end):
                for resource_index in range(len(resource_levels)):
                    demand = int(nodes_dict[f"R{resource_index + 1}"][candidate])
                    if resource_time[time_index, resource_index] - demand < 0:
                        matrix[0][candidate] = time_index + 1
                        feasible = False
                        break
                if not feasible:
                    break

    propagate_earliest_start_times(matrix)
    return border, others


# Build bidirectional temporal edges at the configured time distances.
# Parameters: minimum_time: first time instant; maximum_time: last time instant
#   scales: temporal edge distances
# Returns: List[dict].
def build_temporal_multiscale_edges(
    minimum_time: int,
    maximum_time: int,
    scales: Sequence[int] = TEMPORAL_EDGE_SCALES,
) -> List[dict]:
    """Build bidirectional temporal edges at the configured time distances."""

    links: List[dict] = []
    for scale in scales:
        distance = int(scale)
        if distance <= 0:
            raise ValueError("Temporal edge scales must be positive integers.")
        for time_index in range(minimum_time, maximum_time - distance):
            links.append(
                {
                    "source": f"time-resources{time_index}",
                    "target": f"time-resources{time_index + distance}",
                    "way_time": 1,
                    "delta_time": distance,
                }
            )
            links.append(
                {
                    "source": f"time-resources{time_index + distance}",
                    "target": f"time-resources{time_index}",
                    "way_time": 0,
                    "delta_time": distance,
                }
            )
    return links


# Build the heterogeneous node-link graph for the current schedule state.
# Parameters: nodes_dict: activity durations and demands; matrix: temporal distance matrix
#   border: eligible activity IDs; resource_time: residual resource-time profile
#   all_nodes: original activity IDs; positive_nodes: reference-consistent activity IDs
#   resource_capacities: instance resource capacities
# Returns: dict.
def generate_dynamic_graph(
    nodes_dict: Mapping[str, Sequence[int]],
    matrix: Sequence[Sequence[int]],
    border: set[int],
    resource_time: np.ndarray,
    all_nodes: Sequence[int] | None = None,
    positive_nodes: Iterable[int] = (),
    resource_capacities: Sequence[int] | None = None,
) -> dict:
    """Build the heterogeneous node-link graph for the current schedule state."""

    if all_nodes is None:
        all_nodes = sorted(border | set(range(1, len(matrix))))
    all_nodes = list(all_nodes)
    if not all_nodes:
        raise ValueError("Cannot build a graph without unscheduled nodes.")

    all_node_set = set(all_nodes)
    positives = set(positive_nodes)
    nodes: List[dict] = []
    links: List[dict] = []

    for node in all_nodes:
        resources = _resources_for_node(nodes_dict, node)
        if node in border:
            nodes.append(
                {
                    "id": str(node),
                    "duration": int(nodes_dict["duration"][node]),
                    "entity": "border",
                    "resources_needed": resources,
                    "solution": 1 if node in positives else 0,
                }
            )
        else:
            nodes.append(
                {
                    "id": str(node),
                    "duration_other": int(nodes_dict["duration"][node]),
                    "entity": "other",
                    "resources_needed_other": resources,
                }
            )

    minimum_time = min(int(matrix[0][node]) for node in all_nodes)
    maximum_time = int(matrix[0][-1]) + 1
    if minimum_time < 0:
        raise RuntimeError(f"Invalid negative earliest start time {minimum_time}.")
    if maximum_time > resource_time.shape[0]:
        raise RuntimeError(
            f"Graph needs time-resources nodes up to {maximum_time - 1}, but max_time="
            f"{resource_time.shape[0]}."
        )

    time_span = maximum_time - minimum_time - 1
    for time_index in range(minimum_time, maximum_time):
        nodes.append(
            {
                "id": f"time-resources{time_index}",
                "entity": "time-resources",
                "resources_available": _resources_at_time(resource_time, time_index),
                # Encode position inside the current temporal window. This is
                # the fifth time-resource feature expected by the trained model.
                "relative_time": (
                    float(time_index - minimum_time) / float(time_span)
                    if time_span > 0
                    else 0.0
                ),
            }
        )

    for source in all_nodes:
        for target in all_nodes:
            delta = int(matrix[source][target])
            if delta <= INF:
                continue
            if source in border:
                links.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "way_precedence_border": 1,
                        "delta_precedence_border": delta,
                        "duration_precedence_border": int(
                            nodes_dict["duration"][source]
                        ),
                    }
                )
                links.append(
                    {
                        "source": str(target),
                        "target": str(source),
                        "way_task_relation_border": 1,
                        "delta_task_relation_border": delta,
                        "duration_task_relation_border": int(
                            nodes_dict["duration"][target]
                        ),
                    }
                )
            else:
                links.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "way_others_edges": 1,
                        "delta_others_edges": delta,
                        "duration_others_edges": int(nodes_dict["duration"][source]),
                    }
                )
                links.append(
                    {
                        "source": str(target),
                        "target": str(source),
                        "way_others_edges": 0,
                        "delta_others_edges": delta,
                        "duration_others_edges": int(nodes_dict["duration"][target]),
                    }
                )

        start_time = int(matrix[0][source])
        resources = _resources_for_node(nodes_dict, source)
        if source in border:
            links.append(
                {
                    "source": str(source),
                    "target": f"time-resources{start_time}",
                    "way_min_start_border": 1,
                    "resource_needed_min_start_border": resources,
                    "duration_min_start_border": int(nodes_dict["duration"][source]),
                }
            )
            links.append(
                {
                    "source": f"time-resources{start_time}",
                    "target": str(source),
                    "way_time_resources_link_border": 1,
                    "resource_free_time_resources_link_border": _resources_at_time(
                        resource_time, start_time
                    ),
                }
            )
        else:
            links.append(
                {
                    "source": str(source),
                    "target": f"time-resources{start_time}",
                    "way_min_start_other": 1,
                    "resource_needed_min_start_other": resources,
                    "duration_min_start_other": int(nodes_dict["duration"][source]),
                }
            )
            links.append(
                {
                    "source": f"time-resources{start_time}",
                    "target": str(source),
                    "way_time_resources_link_other": 1,
                    "resource_free_time_resources_link_other": _resources_at_time(
                        resource_time, start_time
                    ),
                }
            )

    links.extend(build_temporal_multiscale_edges(minimum_time, maximum_time))

    if resource_capacities is None:
        # Backward-compatible fallback for callers that do not pass the instance
        # capacities explicitly. The schedule horizon is initialized with the
        # full capacity, so the component-wise maximum recovers that capacity.
        if resource_time.ndim != 2:
            raise ValueError("resource_time must be a two-dimensional array.")
        inferred = resource_time.max(axis=0).astype(int).tolist()
        resource_capacities = inferred

    capacities = [int(value) for value in resource_capacities]
    capacities = capacities[:MODEL_RESOURCE_COUNT] + [0] * max(
        0, MODEL_RESOURCE_COUNT - len(capacities)
    )

    return {
        "directed": True,
        "multigraph": False,
        "graph": {
            "resource_capacities": capacities,
        },
        "nodes": nodes,
        "links": links,
    }


# Add the immutable original precedence relations used during training.
# Parameters: graph: node-link graph; original_matrix: immutable precedence matrix
#   nodes_dict: activity durations and demands; border: eligible activity IDs
#   all_nodes: original activity IDs
# Returns: dict.
def add_typed_original_relations(
    graph: Mapping,
    original_matrix: Sequence[Sequence[int]],
    nodes_dict: Mapping[str, Sequence[int]],
    border: set[int],
    all_nodes: Sequence[int],
) -> dict:
    """Add the immutable original precedence relations used during training.

    ``matrix`` in the dynamic graph changes as activities are scheduled. These
    four typed relations must instead come from the untouched instance matrix,
    so training and rollout expose the model to the same graph structure.
    """

    output = deepcopy(dict(graph))
    edge_key = "edges" if "edges" in output else "links"
    existing_edges = [deepcopy(edge) for edge in output.get(edge_key, [])]

    next_key: defaultdict[tuple[str, str], int] = defaultdict(int)
    for edge in existing_edges:
        pair = (str(edge.get("source")), str(edge.get("target")))
        edge["key"] = next_key[pair]
        next_key[pair] += 1

    # Append one typed original-precedence edge.
    # Parameters: source: source features or activity ID; target: destination activity ID
    #   relation: typed edge relation; way: relation direction marker; delta: temporal distance
    #   duration: activity duration
    # Returns: None; updates state or writes output.
    def append_relation(
        source: int,
        target: int,
        relation: str,
        way: int,
        delta: int,
        duration: int,
    ) -> None:
        pair = (str(source), str(target))
        existing_edges.append(
            {
                "source": pair[0],
                "target": pair[1],
                "key": next_key[pair],
                "relation": relation,
                "way_original_edges": int(way),
                "delta_original_edges": int(delta),
                "duration_original_edges": int(duration),
            }
        )
        next_key[pair] += 1

    border = set(border)
    for source_node in all_nodes:
        for target_node in all_nodes:
            if source_node == target_node:
                continue

            delta = int(original_matrix[source_node][target_node])
            if delta == INF:
                continue

            source_is_border = source_node in border
            target_is_border = target_node in border
            if source_is_border and target_is_border:
                raise ValueError(
                    "Unexpected original precedence between border nodes: "
                    f"{source_node} -> {target_node}."
                )
            if source_is_border:
                forward_relation = "original_border_other_forward"
                reverse_relation = "original_border_other_reverse"
            elif target_is_border:
                raise ValueError(
                    "Unexpected original precedence from other to border: "
                    f"{source_node} -> {target_node}."
                )
            else:
                forward_relation = "original_other_other_forward"
                reverse_relation = "original_other_other_reverse"

            append_relation(
                source_node,
                target_node,
                forward_relation,
                1,
                delta,
                int(nodes_dict["duration"][source_node]),
            )
            append_relation(
                target_node,
                source_node,
                reverse_relation,
                0,
                delta,
                int(nodes_dict["duration"][target_node]),
            )

    output[edge_key] = existing_edges
    output["directed"] = True
    output["multigraph"] = True
    return output


# Convert a predicted identifier to a Python scalar.
# Parameters: value: input value
# Returns: int | str.
def _python_node_id(value: object) -> int | str:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return str(value)


# Run the PyG model on one dynamically generated graph and return top-1.
# Parameters: model: priority network; graph: node-link graph; device: compute device
# Returns: BorderPrediction.
def predict_border_node(
    model: nn.Module,
    graph: Mapping,
    device: torch.device | str,
) -> BorderPrediction:
    """Run the PyG model on one dynamically generated graph and return top-1."""

    # Lazy import keeps the scheduling utilities testable without PyG installed.
    import torch

    from .data import border_metadata, parse_rcpsp_graph

    data = parse_rcpsp_graph(graph)
    node_ids = border_metadata(data)[0]
    device = torch.device(device)
    data = data.to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(data).reshape(-1)
        probabilities = torch.sigmoid(logits)

    if logits.numel() == 0:
        raise RuntimeError("The dynamic graph contains no border prediction.")
    if len(node_ids) != logits.numel():
        raise RuntimeError(
            f"Found {len(node_ids)} border IDs for {logits.numel()} logits."
        )

    order = torch.argsort(logits, descending=True).detach().cpu().tolist()
    scores: List[dict] = []
    for rank, index in enumerate(order, start=1):
        scores.append(
            {
                "rank": rank,
                "node_id": _python_node_id(node_ids[index]),
                "logit": float(logits[index].detach().cpu().item()),
                "probability": float(probabilities[index].detach().cpu().item()),
            }
        )

    best = scores[0]
    return BorderPrediction(
        node_id=best["node_id"],
        logit=best["logit"],
        probability=best["probability"],
        scores=scores,
    )


@dataclass
class _BatchedScheduleState:
    original_matrix: list
    nodes_dict: dict
    resource_levels: list
    matrix: list
    resource_time: list
    solution: List[List[int]]
    border: set[int]
    others: set[int]
    decisions: List[dict]
    real_activity_count: int
    step: int = 0


# Initialize one state for batched schedule construction.
# Parameters: instance: parsed RCPSP instance; max_time: schedule horizon
# Returns: _BatchedScheduleState.
def _initialize_batched_schedule_state(
    instance: MRCPSPInstance,
    max_time: int,
) -> _BatchedScheduleState:
    original_matrix = create_distance_matrix(instance)
    (
        nodes_dict,
        resource_levels,
        matrix,
        resource_time,
        solution,
        border,
        others,
    ) = initialize_schedule_state(instance, max_time)
    return _BatchedScheduleState(
        original_matrix=original_matrix,
        nodes_dict=nodes_dict,
        resource_levels=resource_levels,
        matrix=matrix,
        resource_time=resource_time,
        solution=solution,
        border=border,
        others=others,
        decisions=[],
        real_activity_count=len(nodes_dict["nodes"]) - 2,
    )


# Prepare a partial-state graph and its Border identifiers.
# Parameters: state: partial schedule state; graph_transform: optional graph normalization
# Returns: parsed graph and original Border IDs.
def _prepare_batched_border_graph(
    state: _BatchedScheduleState,
    graph_transform: GraphTransform | None,
) -> tuple[object, list[object]]:
    from .data import border_metadata, parse_rcpsp_graph

    all_nodes = sorted(state.border | state.others)
    graph = generate_dynamic_graph(
        state.nodes_dict,
        state.matrix,
        state.border,
        state.resource_time,
        all_nodes,
        resource_capacities=state.resource_levels,
    )
    graph = add_typed_original_relations(
        graph,
        state.original_matrix,
        state.nodes_dict,
        state.border,
        all_nodes,
    )
    if graph_transform is not None:
        graph = graph_transform(graph)
    data = parse_rcpsp_graph(graph)
    return data, border_metadata(data)[0]


# Score several independent rollout states in one PyG forward pass.
# Parameters: model: priority network; prepared_graphs: parsed graphs with activity IDs
#   device: compute device; include_scores: retain selected-node scores
# Returns: list[BorderPrediction].
def _predict_prepared_border_nodes(
    model: nn.Module,
    prepared_graphs: Sequence[tuple[object, list[object]]],
    device: torch.device | str,
    *,
    include_scores: bool,
) -> list[BorderPrediction]:
    """Score several independent rollout states in one PyG forward pass."""

    import torch
    from torch_geometric.data import Batch

    if not prepared_graphs:
        return []

    data_list = [data for data, _ in prepared_graphs]
    node_ids_per_graph = [node_ids for _, node_ids in prepared_graphs]
    border_counts = [len(node_ids) for node_ids in node_ids_per_graph]
    if any(count < 1 for count in border_counts):
        raise RuntimeError("Every rollout graph must contain at least one border node.")

    device = torch.device(device)
    batch = Batch.from_data_list(data_list).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(batch).reshape(-1)

    expected_logits = sum(border_counts)
    if logits.numel() != expected_logits:
        raise RuntimeError(
            f"Found {logits.numel()} batched logits for {expected_logits} border nodes."
        )

    # One device-to-host synchronization for the whole batch is substantially
    # cheaper than synchronizing every rollout and every score independently.
    logits_cpu = logits.detach().float().cpu()
    probabilities_cpu = torch.sigmoid(logits_cpu)
    predictions: list[BorderPrediction] = []
    offset = 0
    for node_ids, count in zip(node_ids_per_graph, border_counts):
        graph_logits = logits_cpu[offset : offset + count]
        graph_probabilities = probabilities_cpu[offset : offset + count]
        offset += count

        if include_scores:
            order = torch.argsort(graph_logits, descending=True).tolist()
            scores = [
                {
                    "rank": rank,
                    "node_id": _python_node_id(node_ids[index]),
                    "logit": float(graph_logits[index].item()),
                    "probability": float(graph_probabilities[index].item()),
                }
                for rank, index in enumerate(order, start=1)
            ]
            best = scores[0]
        else:
            best_index = int(torch.argmax(graph_logits).item())
            scores = []
            best = {
                "node_id": _python_node_id(node_ids[best_index]),
                "logit": float(graph_logits[best_index].item()),
                "probability": float(graph_probabilities[best_index].item()),
            }

        predictions.append(
            BorderPrediction(
                node_id=best["node_id"],
                logit=best["logit"],
                probability=best["probability"],
                scores=scores,
            )
        )
    return predictions


# Apply a selected activity to its schedule state.
# Parameters: state: partial schedule state; prediction: selected activity and scores
# Returns: None; updates state or writes output.
def _apply_batched_prediction(
    state: _BatchedScheduleState,
    prediction: BorderPrediction,
) -> None:
    selected_node = int(prediction.node_id)
    if selected_node not in state.border:
        raise RuntimeError(
            f"The selector returned node {selected_node}, which is not in border "
            f"{sorted(state.border)}."
        )

    start_time = int(state.matrix[0][selected_node])
    state.solution.append([selected_node, start_time])
    state.decisions.append(
        {
            "step": state.step,
            "selected_node": selected_node,
            "start_time": start_time,
            "duration": int(state.nodes_dict["duration"][selected_node]),
            "border_size": len(state.border),
            "selection_method": "forced_singleton" if len(state.border) == 1 else "gnn",
            "logit": float(prediction.logit),
            "probability": float(prediction.probability),
            "scores": prediction.scores,
        }
    )
    state.border, state.others = update_schedule_state(
        state.nodes_dict,
        state.resource_levels,
        state.matrix,
        selected_node,
        state.border,
        state.others,
        state.resource_time,
    )
    state.step += 1


# Collect the completed schedule and makespan.
# Parameters: state: partial schedule state
# Returns: ScheduleResult.
def _finalize_batched_schedule_state(state: _BatchedScheduleState) -> ScheduleResult:
    sink = len(state.matrix) - 1
    solution = list(state.solution)
    solution.append([sink, int(state.matrix[0][sink])])
    return ScheduleResult(
        makespan=int(state.matrix[0][sink]),
        solution=solution,
        decisions=state.decisions,
    )


# Run independent schedules with batched GNN inference.
# Parameters: model: priority network; instances: independent RCPSP instances
#   max_times: schedule horizon per instance; device: compute device
#   graph_transform: optional graph normalization; graph_workers: graph preparation threads
#   include_scores: retain selected-node scores
# Returns: list[ScheduleResult].
def calculate_makespans_batched(
    model: nn.Module,
    instances: Sequence[MRCPSPInstance],
    max_times: Sequence[int],
    device: torch.device | str,
    graph_transform: GraphTransform | None = None,
    *,
    graph_workers: int = 1,
    include_scores: bool = False,
) -> list[ScheduleResult]:
    """Run independent schedules with batched GNN inference.

    Each nontrivial active instance contributes one dynamic graph to a PyG ``Batch`` per
    decision step.  The model is evaluated once for the whole group, while CPU
    graph construction can use a persistent thread pool.  Result order matches
    ``instances``.
    """

    from concurrent.futures import ThreadPoolExecutor

    if len(instances) != len(max_times):
        raise ValueError("instances and max_times must have the same length.")
    if graph_workers < 1:
        raise ValueError("graph_workers must be at least 1.")
    if not instances:
        return []

    states = [
        _initialize_batched_schedule_state(instance, int(max_time))
        for instance, max_time in zip(instances, max_times)
    ]
    executor = None
    if graph_workers > 1 and len(states) > 1:
        executor = ThreadPoolExecutor(max_workers=min(graph_workers, len(states)))

    try:
        while True:
            active_states = [
                state for state in states if state.step < state.real_activity_count
            ]
            if not active_states:
                break

            # Advance forced decisions without constructing/normalizing a graph.
            branching_states = []
            for state in active_states:
                if len(state.border) == 1:
                    _apply_batched_prediction(state, forced_border_prediction(next(iter(state.border))))
                else:
                    branching_states.append(state)
            active_states = branching_states
            if not active_states:
                continue

            if executor is None:
                prepared = [
                    _prepare_batched_border_graph(state, graph_transform)
                    for state in active_states
                ]
            else:
                prepared = list(
                    executor.map(
                        lambda state: _prepare_batched_border_graph(
                            state, graph_transform
                        ),
                        active_states,
                    )
                )

            predictions = _predict_prepared_border_nodes(
                model,
                prepared,
                device,
                include_scores=include_scores,
            )
            for state, prediction in zip(active_states, predictions):
                _apply_batched_prediction(state, prediction)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return [_finalize_batched_schedule_state(state) for state in states]


Selector = Callable[[dict, set[int], int], BorderPrediction]
GraphTransform = Callable[[Mapping], dict]


# Serial schedule generation using a model/heuristic selector per border.
# Parameters: instance: parsed RCPSP instance; max_time: schedule horizon
#   selector: activity-selection callback; skip_singleton: bypass forced decisions
# Returns: ScheduleResult.
def calculate_makespan_with_selector(
    instance: MRCPSPInstance,
    max_time: int,
    selector: Selector,
    *,
    skip_singleton: bool = False,
) -> ScheduleResult:
    """Serial schedule generation using a model/heuristic selector per border."""

    original_matrix = create_distance_matrix(instance)
    (
        nodes_dict,
        resource_levels,
        matrix,
        resource_time,
        solution,
        border,
        others,
    ) = initialize_schedule_state(instance, max_time)

    decisions: List[dict] = []
    real_activity_count = len(nodes_dict["nodes"]) - 2

    for step in range(real_activity_count):
        forced = skip_singleton and len(border) == 1
        if forced:
            prediction = forced_border_prediction(next(iter(border)))
        else:
            all_nodes = sorted(border | others)
            graph = generate_dynamic_graph(
                nodes_dict, matrix, border, resource_time, all_nodes,
                resource_capacities=resource_levels,
            )
            graph = add_typed_original_relations(
                graph, original_matrix, nodes_dict, border, all_nodes,
            )
            prediction = selector(graph, set(border), step)
        selected_node = int(prediction.node_id)
        if selected_node not in border:
            raise RuntimeError(
                f"The selector returned node {selected_node}, which is not in border "
                f"{sorted(border)}."
            )

        start_time = int(matrix[0][selected_node])
        solution.append([selected_node, start_time])
        decisions.append(
            {
                "step": step,
                "selected_node": selected_node,
                "start_time": start_time,
                "duration": int(nodes_dict["duration"][selected_node]),
                "border_size": len(border),
                "selection_method": "forced_singleton" if forced else "selector",
                "logit": float(prediction.logit),
                "probability": float(prediction.probability),
                "scores": prediction.scores,
            }
        )

        border, others = update_schedule_state(
            nodes_dict,
            resource_levels,
            matrix,
            selected_node,
            border,
            others,
            resource_time,
        )

    sink = len(matrix) - 1
    solution.append([sink, int(matrix[0][sink])])
    return ScheduleResult(
        makespan=int(matrix[0][sink]),
        solution=solution,
        decisions=decisions,
    )


# Construct a feasible schedule using learned priorities.
# Parameters: model: priority network; instance: parsed RCPSP instance; max_time: schedule horizon
#   device: compute device; graph_transform: optional graph normalization
# Returns: ScheduleResult.
def calculate_makespan(
    model: nn.Module,
    instance: MRCPSPInstance,
    max_time: int,
    device: torch.device | str,
    graph_transform: GraphTransform | None = None,
) -> ScheduleResult:
    import torch

    device = torch.device(device)

    # Normalize a decision graph and choose its top-scoring activity.
    # Parameters: graph: node-link graph; border: eligible activity IDs; step: construction step
    # Returns: BorderPrediction.
    def selector(graph: dict, border: set[int], step: int) -> BorderPrediction:
        del border, step
        if graph_transform is not None:
            graph = graph_transform(graph)
        return predict_border_node(model, graph, device)

    return calculate_makespan_with_selector(instance, max_time, selector, skip_singleton=True)


# Simple deterministic selector useful as a baseline and for unit tests.
# Parameters: graph: node-link graph; border: eligible activity IDs; step: construction step
# Returns: BorderPrediction.
def earliest_start_selector(
    graph: dict,
    border: set[int],
    step: int,
) -> BorderPrediction:
    """Simple deterministic selector useful as a baseline and for unit tests."""

    del graph, step
    node = min(border)
    return BorderPrediction(node_id=node, logit=0.0, probability=0.5, scores=[])

from __future__ import annotations

import io
import json
import math
import random
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, List, Mapping, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import HeteroData

from .graph_schema import canonicalize_graph_features

ENTITIES: Tuple[str, ...] = ("border", "other", "time-resources")
EdgeType = Tuple[str, str, str]

ORIGINAL_BORDER_OTHER_FORWARD: EdgeType = (
    "border",
    "original_border_other_forward",
    "other",
)
ORIGINAL_BORDER_OTHER_REVERSE: EdgeType = (
    "other",
    "original_border_other_reverse",
    "border",
)
ORIGINAL_OTHER_OTHER_FORWARD: EdgeType = (
    "other",
    "original_other_other_forward",
    "other",
)
ORIGINAL_OTHER_OTHER_REVERSE: EdgeType = (
    "other",
    "original_other_other_reverse",
    "other",
)
TIME_RESOURCES_BORDER: EdgeType = ("time-resources", "to", "border")
TIME_RESOURCES_OTHER: EdgeType = ("time-resources", "to", "other")
BORDER_TIME_RESOURCES: EdgeType = ("border", "to", "time-resources")
OTHER_TIME_RESOURCES: EdgeType = ("other", "to", "time-resources")
TIME_RESOURCES_TIME_RESOURCES: EdgeType = ("time-resources", "to", "time-resources")
LEGACY_OTHER_BORDER_READOUT: EdgeType = (
    "other",
    "legacy_task_relation_border",
    "border",
)

RELATIONS: Tuple[EdgeType, ...] = (
    ORIGINAL_BORDER_OTHER_FORWARD,
    ORIGINAL_OTHER_OTHER_FORWARD,
    TIME_RESOURCES_BORDER,
    TIME_RESOURCES_OTHER,
    BORDER_TIME_RESOURCES,
    OTHER_TIME_RESOURCES,
    TIME_RESOURCES_TIME_RESOURCES,
    LEGACY_OTHER_BORDER_READOUT,
    ORIGINAL_OTHER_OTHER_REVERSE,
    ORIGINAL_BORDER_OTHER_REVERSE,
)

# Edge attributes consumed by the relation-specific message encoders. Widths
# are explicit so empty edge stores keep a stable shape when PyG batches them.
EDGE_FEATURES: Mapping[EdgeType, Tuple[Tuple[str, int], ...]] = {
    ORIGINAL_BORDER_OTHER_FORWARD: (
        ("delta_original_edges", 1),
        ("duration_original_edges", 1),
    ),
    ORIGINAL_BORDER_OTHER_REVERSE: (
        ("delta_original_edges", 1),
        ("duration_original_edges", 1),
    ),
    ORIGINAL_OTHER_OTHER_FORWARD: (
        ("delta_original_edges", 1),
        ("duration_original_edges", 1),
    ),
    ORIGINAL_OTHER_OTHER_REVERSE: (
        ("delta_original_edges", 1),
        ("duration_original_edges", 1),
    ),
    TIME_RESOURCES_BORDER: (
        ("resource_free_time_resources_link_border", 4),
    ),
    TIME_RESOURCES_OTHER: (
        ("resource_free_time_resources_link_other", 4),
    ),
    BORDER_TIME_RESOURCES: (
        ("duration_min_start_border", 1),
        ("resource_needed_min_start_border", 4),
    ),
    OTHER_TIME_RESOURCES: (
        ("duration_min_start_other", 1),
        ("resource_needed_min_start_other", 4),
    ),
    TIME_RESOURCES_TIME_RESOURCES: (
        ("way_time", 1),
        ("delta_time", 1),
    ),
}
EDGE_FEATURE_WIDTHS: Mapping[EdgeType, int] = {
    edge_type: sum(width for _, width in fields)
    for edge_type, fields in EDGE_FEATURES.items()
}

ORIGINAL_RELATIONS: Mapping[str, EdgeType] = {
    edge_type[1]: edge_type
    for edge_type in (
        ORIGINAL_BORDER_OTHER_FORWARD,
        ORIGINAL_BORDER_OTHER_REVERSE,
        ORIGINAL_OTHER_OTHER_FORWARD,
        ORIGINAL_OTHER_OTHER_REVERSE,
    )
}
LEGACY_RELATIONS: Mapping[Tuple[str, str], EdgeType] = {
    (edge_type[0], edge_type[2]): edge_type
    for edge_type in (TIME_RESOURCES_BORDER, TIME_RESOURCES_OTHER, BORDER_TIME_RESOURCES, OTHER_TIME_RESOURCES, TIME_RESOURCES_TIME_RESOURCES)
}
FEATURES: Mapping[str, Tuple[str, ...]] = {
    "border": ("duration", "resources_needed"),
    "other": ("duration_other", "resources_needed_other"),
    "time-resources": ("resources_available", "relative_time"),
}
FEATURE_WIDTHS: Mapping[str, int] = {
    "border": 5,
    "other": 5,
    "time-resources": 5,
}
LABEL_NAME = "solution"
DATASET_SUFFIXES = (".json", ".tar.gz")
VARIANT_POLICIES = ("all", "original")


def _hashable_node_id(value):
    if isinstance(value, list):
        return tuple(_hashable_node_id(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable_node_id(v)) for k, v in value.items()))
    return value


def _as_float_vector(value) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _legacy_proximity_weight(edge: Mapping) -> float:
    weight = float(edge.get("proximity_weight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("Legacy proximity_weight must be finite and non-negative.")
    return weight


def _legacy_delta_feature(edge: Mapping) -> float:
    delta_name = "delta_task_relation_border"
    if delta_name not in edge:
        raise ValueError(f"Legacy other -> border edge is missing '{delta_name}'.")
    delta = float(edge[delta_name])
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError(f"Edge field '{delta_name}' must be finite and non-negative.")
    return delta


def _relation_edge_features(edge: Mapping, edge_type: EdgeType) -> List[float]:
    """Extract and validate the fixed-width attributes of one typed edge."""
    if edge_type not in EDGE_FEATURES:
        raise ValueError(f"No edge feature schema is defined for relation {edge_type}.")

    row: List[float] = []
    for feature_name, expected_width in EDGE_FEATURES[edge_type]:
        if feature_name not in edge:
            raise ValueError(
                f"Relation '{edge_type[1]}' is missing edge feature "
                f"'{feature_name}'. Regenerate the graph dataset."
            )
        values = _as_float_vector(edge[feature_name])
        if len(values) != expected_width:
            raise ValueError(
                f"Edge feature '{feature_name}' in relation '{edge_type[1]}' "
                f"has width {len(values)}; expected {expected_width}."
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Edge feature '{feature_name}' in relation '{edge_type[1]}' "
                "must contain finite values."
            )
        row.extend(values)
    return row


def parse_rcpsp_graph(raw: Mapping) -> HeteroData:
    """Convert an RCPSP node-link graph into ``HeteroData``.

    The parser accepts both the legacy ``links`` key and the modern ``edges``
    key. All node and edge types are created even when one of them is empty, so
    PyG can batch samples with different graph sizes safely. Parallel edges are
    supported because PyG represents them as repeated columns in ``edge_index``.
    """
    raw = canonicalize_graph_features(raw)
    if raw.get("directed") is False:
        raise ValueError("The graph is undirected; this model expects directed graphs.")
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Each graph must contain a non-empty 'nodes' list.")

    links = raw.get("links", raw.get("edges"))
    if links is None:
        raise ValueError("The graph does not contain 'links' or 'edges'.")
    if not isinstance(links, list):
        raise ValueError("The graph edge collection must be a list.")

    feature_rows = {entity: [] for entity in ENTITIES}
    node_ids = {entity: [] for entity in ENTITIES}
    labels: List[float] = []
    id_to_local = {}

    for position, node in enumerate(nodes):
        if "entity" not in node:
            raise ValueError(f"Node at position {position} has no 'entity' attribute.")
        entity = str(node["entity"])
        if entity not in FEATURES:
            raise ValueError(f"Unsupported entity '{entity}'. Expected one of {ENTITIES}.")

        original_id = node.get("id", position)
        node_id = _hashable_node_id(original_id)
        if node_id in id_to_local:
            raise ValueError(f"Duplicated node id: {node_id!r}")

        row: List[float] = []
        for feature_name in FEATURES[entity]:
            if feature_name not in node:
                raise ValueError(
                    f"Node {node_id!r} ({entity}) is missing feature '{feature_name}'."
                )
            row.extend(_as_float_vector(node[feature_name]))

        local_index = len(feature_rows[entity])
        feature_rows[entity].append(row)
        node_ids[entity].append(original_id)
        id_to_local[node_id] = (entity, local_index)

        if entity == "border":
            if LABEL_NAME not in node:
                raise ValueError(f"Border node {node_id!r} is missing '{LABEL_NAME}'.")
            labels.append(float(node[LABEL_NAME]))

    if not labels:
        raise ValueError("Every sample must contain at least one border node.")

    data = HeteroData()
    for entity in ENTITIES:
        rows = feature_rows[entity]
        feature_dim = FEATURE_WIDTHS[entity]
        if rows:
            row_dim = len(rows[0])
            if any(len(row) != row_dim for row in rows):
                raise ValueError(f"Feature dimensions are inconsistent for '{entity}'.")
            if row_dim != feature_dim:
                raise ValueError(
                    f"Entity '{entity}' has feature width {row_dim}; "
                    f"expected {feature_dim}."
                )
            x = torch.tensor(rows, dtype=torch.float32)
        else:
            x = torch.empty((0, feature_dim), dtype=torch.float32)
        data[entity].x = x
        data[entity].num_nodes = int(x.shape[0])

    data["border"].y = torch.tensor(labels, dtype=torch.float32)

    edge_lists = {edge_type: [] for edge_type in RELATIONS}
    edge_feature_rows = {edge_type: [] for edge_type in EDGE_FEATURES}
    legacy_other_border_weights: List[float] = []
    legacy_other_border_deltas: List[float] = []
    for edge in links:
        source_id = _hashable_node_id(edge.get("source"))
        target_id = _hashable_node_id(edge.get("target"))
        if source_id not in id_to_local or target_id not in id_to_local:
            raise ValueError(f"Edge references an unknown node: {source_id!r} -> {target_id!r}")
        src_entity, src_index = id_to_local[source_id]
        dst_entity, dst_index = id_to_local[target_id]
        relation_name = edge.get("relation")
        if relation_name in ORIGINAL_RELATIONS:
            edge_type = ORIGINAL_RELATIONS[str(relation_name)]
            endpoints = (edge_type[0], edge_type[2])
            if (src_entity, dst_entity) != endpoints:
                raise ValueError(
                    f"Relation '{relation_name}' expects {endpoints[0]} -> {endpoints[1]}, "
                    f"got {src_entity} -> {dst_entity}."
                )
        elif (
            (src_entity, dst_entity) == ("other", "border")
            and "delta_task_relation_border" in edge
        ):
            edge_type = LEGACY_OTHER_BORDER_READOUT
        else:
            # Legacy structural edges remain valid only for the temporal
            # relations. Untyped border/other edges are intentionally
            # ignored in favour of the new original_* relations.
            edge_type = LEGACY_RELATIONS.get((src_entity, dst_entity))

        if edge_type is not None:
            edge_lists[edge_type].append((src_index, dst_index))
            if edge_type == LEGACY_OTHER_BORDER_READOUT:
                legacy_other_border_weights.append(_legacy_proximity_weight(edge))
                legacy_other_border_deltas.append(_legacy_delta_feature(edge))
            else:
                edge_feature_rows[edge_type].append(
                    _relation_edge_features(edge, edge_type)
                )

    for edge_type, pairs in edge_lists.items():
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data[edge_type].edge_index = edge_index
        if edge_type == LEGACY_OTHER_BORDER_READOUT:
            data[edge_type].edge_weight = torch.tensor(
                legacy_other_border_weights,
                dtype=torch.float32,
            )
            data[edge_type].edge_attr = torch.tensor(
                legacy_other_border_deltas,
                dtype=torch.float32,
            ).unsqueeze(1)
        else:
            feature_width = EDGE_FEATURE_WIDTHS[edge_type]
            rows = edge_feature_rows[edge_type]
            if rows:
                data[edge_type].edge_attr = torch.tensor(
                    rows,
                    dtype=torch.float32,
                )
            else:
                data[edge_type].edge_attr = torch.empty(
                    (0, feature_width),
                    dtype=torch.float32,
                )

    # PyG batches arbitrary strings as a list. JSON keeps arbitrary node IDs
    # intact and allows prediction to recover one list per graph.
    data.border_node_ids_json = json.dumps(node_ids["border"], ensure_ascii=False)
    return data


def _first_non_whitespace(stream: BinaryIO) -> bytes:
    position = stream.tell()
    first = b""
    while True:
        char = stream.read(1)
        if not char or not char.isspace():
            first = char
            break
    stream.seek(position)
    return first


def _iter_graph_dicts(stream: BinaryIO) -> Iterator[Mapping]:
    first = _first_non_whitespace(stream)
    if first == b"[":
        try:
            import ijson  # type: ignore

            yield from ijson.items(stream, "item")
        except ImportError:
            data = json.load(io.TextIOWrapper(stream, encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array of graphs.")
            yield from data
    elif first == b"{":
        data = json.load(io.TextIOWrapper(stream, encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected one JSON graph object.")
        yield data
    else:
        raise ValueError("JSON dataset must start with '[' or '{'.")


def iter_graphs_from_file(path: Path) -> Iterator[Mapping]:
    name = path.name.lower()
    if name.endswith(".json"):
        with path.open("rb") as stream:
            yield from _iter_graph_dicts(stream)
        return

    if name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".json")
            ]
            if not members:
                raise ValueError(f"Archive {path} contains no JSON file.")
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is not None:
                    yield from _iter_graph_dicts(extracted)
        return

    raise ValueError(f"Unsupported dataset file: {path}")


def find_dataset_files(directory: str | Path) -> List[Path]:
    """Return the supported dataset files directly inside ``directory``."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.lower().endswith(DATASET_SUFFIXES)
    )


def _buffer_shuffle(
    items: Iterable[HeteroData], buffer_size: int, rng: random.Random
) -> Iterator[HeteroData]:
    if buffer_size <= 1:
        yield from items
        return

    buffer: List[HeteroData] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item

    rng.shuffle(buffer)
    yield from buffer


def _is_original_variant(
    raw_graph: Mapping, graph_index: int, legacy_variant_group_size: int
) -> bool:
    """Return whether a graph is the unaugmented member of its variant group.

    Newly generated graphs carry ``graph.augmentation_variant``. Existing
    datasets predate that metadata but store every group in the documented
    order ``original, resource-permuted, duration-scaled``; for those files the
    first graph of every fixed-size group is the original.
    """

    metadata = raw_graph.get("graph", {})
    if isinstance(metadata, Mapping):
        explicit_variant = metadata.get("augmentation_variant")
        if explicit_variant is not None:
            return str(explicit_variant).strip().lower() == "original"
    return graph_index % legacy_variant_group_size == 0


class RcpspGraphDataset(IterableDataset):
    """Streaming dataset with deterministic optional augmentation filtering."""

    def __init__(
        self,
        directory: str | Path,
        *,
        repeat: bool,
        shuffle_files: bool,
        shuffle_buffer: int,
        seed: int,
        variant_policy: str = "all",
        legacy_variant_group_size: int = 3,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.repeat = bool(repeat)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.variant_policy = str(variant_policy).strip().lower()
        self.legacy_variant_group_size = int(legacy_variant_group_size)
        if self.variant_policy not in VARIANT_POLICIES:
            raise ValueError(
                f"variant_policy must be one of {VARIANT_POLICIES}, "
                f"got {variant_policy!r}."
            )
        if self.legacy_variant_group_size < 1:
            raise ValueError("legacy_variant_group_size must be at least 1.")

        self.files = find_dataset_files(self.directory)
        if not self.files:
            raise FileNotFoundError(
                f"No .json or .tar.gz files found directly in {self.directory}"
            )

    def _worker_files(self) -> Tuple[List[Path], int]:
        worker = get_worker_info()
        if worker is None:
            return list(self.files), 0
        files = list(self.files)[worker.id :: worker.num_workers]
        return files, worker.id

    def __iter__(self) -> Iterator[HeteroData]:
        files, worker_id = self._worker_files()
        if not files:
            return

        cycle = 0
        while True:
            rng = random.Random(self.seed + worker_id * 100003 + cycle)
            cycle_files = list(files)
            if self.shuffle_files:
                rng.shuffle(cycle_files)

            def parsed_items() -> Iterator[HeteroData]:
                for file_path in cycle_files:
                    for graph_index, raw_graph in enumerate(
                        iter_graphs_from_file(file_path)
                    ):
                        if self.variant_policy == "original" and not _is_original_variant(
                            raw_graph,
                            graph_index,
                            self.legacy_variant_group_size,
                        ):
                            continue
                        yield parse_rcpsp_graph(raw_graph)

            iterator: Iterable[HeteroData] = parsed_items()
            if self.shuffle_buffer > 1:
                iterator = _buffer_shuffle(iterator, self.shuffle_buffer, rng)
            yield from iterator

            if not self.repeat:
                break
            cycle += 1


def border_metadata(batch: HeteroData) -> List[List[object]]:
    """Recover the original border node IDs from a single graph or batch."""
    raw = batch.border_node_ids_json
    values = [raw] if isinstance(raw, str) else list(raw)
    return [json.loads(value) for value in values]


def batch_graph_count(batch: HeteroData) -> int:
    """Return the number of graphs in a ``HeteroData`` or PyG ``Batch``.

    A graph produced directly by :func:`parse_rcpsp_graph` is a plain
    ``HeteroData`` object.  In that case PyG does not create ``batch`` or
    ``ptr`` attributes on the node stores, so the graph count is one.
    ``ptr`` and ``batch`` are only guaranteed after collation by a PyG
    ``DataLoader``.
    """
    value = getattr(batch, "num_graphs", None)
    if value is not None:
        return int(value)

    border = batch["border"]

    ptr = getattr(border, "ptr", None)
    if ptr is not None:
        return max(int(ptr.numel()) - 1, 1)

    batch_index = getattr(border, "batch", None)
    if batch_index is not None and batch_index.numel() > 0:
        return int(batch_index.max().item()) + 1

    return 1

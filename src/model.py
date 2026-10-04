from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing

from .data import (
    BORDER_TIME_RESOURCES,
    EDGE_FEATURE_WIDTHS,
    ENTITIES,
    FEATURE_WIDTHS,
    LEGACY_OTHER_BORDER_READOUT,
    ORIGINAL_BORDER_OTHER_FORWARD,
    ORIGINAL_BORDER_OTHER_REVERSE,
    ORIGINAL_OTHER_OTHER_FORWARD,
    ORIGINAL_OTHER_OTHER_REVERSE,
    OTHER_TIME_RESOURCES,
    TIME_RESOURCES_BORDER,
    TIME_RESOURCES_OTHER,
    TIME_RESOURCES_TIME_RESOURCES,
    EdgeType,
    batch_graph_count,
)

Relation = EdgeType
PropagationStage = Tuple[Relation, ...]

PROPAGATION_STAGES: Tuple[PropagationStage, ...] = (
    (ORIGINAL_BORDER_OTHER_FORWARD,),
    (ORIGINAL_OTHER_OTHER_FORWARD,),
    (TIME_RESOURCES_BORDER,),
    (TIME_RESOURCES_OTHER,),
    (BORDER_TIME_RESOURCES, OTHER_TIME_RESOURCES),
    (TIME_RESOURCES_TIME_RESOURCES,),
    (TIME_RESOURCES_BORDER,),
    (TIME_RESOURCES_OTHER,),
    (ORIGINAL_OTHER_OTHER_REVERSE,),
    (ORIGINAL_BORDER_OTHER_REVERSE,),
)

ENTITY_TO_INDEX = {entity: index for index, entity in enumerate(ENTITIES)}

# Resource channels are anonymous: applying the same permutation to every
# resource vector of a graph must not change the prediction. The indices below
# separate the four resource channels from temporal/scalar features.
NODE_RESOURCE_INDICES = {
    "border": (1, 2, 3, 4),
    "other": (1, 2, 3, 4),
    "time-resources": (0, 1, 2, 3),
}
EDGE_RESOURCE_INDICES = {
    TIME_RESOURCES_BORDER: (0, 1, 2, 3),
    TIME_RESOURCES_OTHER: (0, 1, 2, 3),
    BORDER_TIME_RESOURCES: (1, 2, 3, 4),
    OTHER_TIME_RESOURCES: (1, 2, 3, 4),
}


def relation_module_key(relation: Relation) -> str:
    """Return a stable ModuleDict key for a heterogeneous relation tuple."""
    return "__".join(relation).replace("-", "_")


PROPAGATION_RELATIONS: Tuple[Relation, ...] = tuple(
    dict.fromkeys(
        relation
        for stage in PROPAGATION_STAGES
        for relation in stage
    )
)


class WeightedSumAggregation(MessagePassing):
    """Scale source states per edge and sum them at the destination."""

    def __init__(self) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

    def forward(
        self,
        source: Tensor,
        destination: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_feature: Tensor,
    ) -> Tensor:
        return self.propagate(
            edge_index,
            x=(source, destination),
            edge_weight=edge_weight,
            edge_feature=edge_feature,
            size=(source.shape[0], destination.shape[0]),
        )

    def message(
        self, x_j: Tensor, edge_weight: Tensor, edge_feature: Tensor
    ) -> Tensor:
        return (x_j + edge_feature) * edge_weight.unsqueeze(-1)


class PermutationInvariantResourceEncoder(nn.Module):
    """DeepSets encoder for scalar features plus anonymous resource channels."""

    def __init__(
        self,
        input_width: int,
        resource_indices: Tuple[int, ...],
        output_width: int,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        resource_indices = tuple(int(index) for index in resource_indices)
        if not resource_indices:
            raise ValueError("At least one resource index is required.")
        if len(set(resource_indices)) != len(resource_indices):
            raise ValueError("Resource indices must be unique.")
        if min(resource_indices) < 0 or max(resource_indices) >= self.input_width:
            raise ValueError("A resource index is outside the input feature width.")

        scalar_indices = tuple(
            index for index in range(self.input_width) if index not in resource_indices
        )
        self.register_buffer(
            "resource_indices",
            torch.tensor(resource_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long),
            persistent=False,
        )
        self.resource_item_encoder = nn.Sequential(
            nn.Linear(1, self.output_width),
            nn.ReLU(),
            nn.Linear(self.output_width, self.output_width, bias=False),
        )
        self.scalar_encoder = (
            nn.Linear(len(scalar_indices), self.output_width, bias=False)
            if scalar_indices
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.resource_item_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if self.scalar_encoder is not None:
            nn.init.xavier_uniform_(self.scalar_encoder.weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError(
                f"Expected feature shape (*, {self.input_width}), got "
                f"{tuple(features.shape)}."
            )
        resource_values = features.index_select(1, self.resource_indices).unsqueeze(-1)
        # Shared encoding plus symmetric aggregation makes the result exactly
        # invariant to every permutation of the resource channels.
        encoded = self.resource_item_encoder(resource_values).mean(dim=1)
        if self.scalar_encoder is not None:
            scalar_values = features.index_select(1, self.scalar_indices)
            encoded = encoded + self.scalar_encoder(scalar_values)
        return encoded


class PermutationInvariantResourcePairEncoder(nn.Module):
    """Encode aligned source/destination resource pairs as an anonymous set."""

    def __init__(self, output_width: int) -> None:
        super().__init__()
        self.output_width = int(output_width)
        self.item_encoder = nn.Sequential(
            nn.Linear(2, self.output_width),
            nn.ReLU(),
            nn.Linear(self.output_width, self.output_width, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.item_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        source_resources: Tensor,
        destination_resources: Tensor,
    ) -> Tensor:
        if source_resources.shape != destination_resources.shape:
            raise ValueError("Source and destination resource shapes must match.")
        if source_resources.ndim != 2:
            raise ValueError("Resource-pair tensors must be two-dimensional.")
        pairs = torch.stack((source_resources, destination_resources), dim=-1)
        return self.item_encoder(pairs).mean(dim=1)


class RcpspGNN(nn.Module):
    """Heterogeneous message-passing network for RCPSP priority prediction.

    PyG manages heterogeneous storage and mini-batch edge offsets. Each stage
    sums incoming source states and applies a GRU update to the destination.
    """

    def __init__(
        self,
        state_dim: int = 32,
        num_iterations: int = 8,
        readout_hidden: int = 32,
        readout_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_iterations = int(num_iterations)
        readout_hidden = int(readout_hidden)
        self.readout_dropout = float(readout_dropout)
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if readout_hidden <= 0:
            raise ValueError("readout_hidden must be positive.")
        if not 0.0 <= self.readout_dropout < 1.0:
            raise ValueError("readout_dropout must be in the interval [0, 1).")

        self.weighted_propagation = WeightedSumAggregation()
        self.node_encoders = nn.ModuleDict(
            {
                entity: PermutationInvariantResourceEncoder(
                    int(FEATURE_WIDTHS[entity]),
                    NODE_RESOURCE_INDICES[entity],
                    self.state_dim,
                )
                for entity in ENTITIES
            }
        )
        self.entity_embedding = nn.Embedding(len(ENTITIES), self.state_dim)
        self.resource_pair_encoder = PermutationInvariantResourcePairEncoder(
            self.state_dim
        )
        self.node_norms = nn.ModuleDict(
            {
                entity: nn.LayerNorm(self.state_dim)
                for entity in ENTITIES
            }
        )
        self.relation_source_transforms = nn.ModuleDict(
            {
                relation_module_key(relation): nn.Linear(
                    self.state_dim,
                    self.state_dim,
                    bias=False,
                )
                for relation in PROPAGATION_RELATIONS
            }
        )
        self.relation_edge_encoders = nn.ModuleDict(
            {
                relation_module_key(relation): (
                    PermutationInvariantResourceEncoder(
                        int(EDGE_FEATURE_WIDTHS[relation]),
                        EDGE_RESOURCE_INDICES[relation],
                        self.state_dim,
                    )
                    if relation in EDGE_RESOURCE_INDICES
                    else nn.Linear(
                        int(EDGE_FEATURE_WIDTHS[relation]),
                        self.state_dim,
                        bias=False,
                    )
                )
                for relation in PROPAGATION_RELATIONS
            }
        )
        self.relation_gates = nn.ModuleDict(
            {
                relation_module_key(relation): nn.Linear(
                    2 * self.state_dim,
                    1,
                )
                for relation in PROPAGATION_RELATIONS
            }
        )
        self.border_update = nn.GRUCell(self.state_dim, self.state_dim)
        self.other_update = nn.GRUCell(self.state_dim, self.state_dim)
        self.time_resources_update = nn.GRUCell(self.state_dim, self.state_dim)
        self.legacy_delta_encoder = nn.Sequential(
            nn.Linear(1, self.state_dim),
            nn.Tanh(),
        )

        self.readout = nn.Sequential(
            nn.Linear(3 * self.state_dim, readout_hidden),
            nn.ReLU(),
            nn.Dropout(self.readout_dropout),
            nn.Linear(readout_hidden, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize feed-forward weights with Xavier and recurrent weights safely."""
        for encoder in self.node_encoders.values():
            encoder.reset_parameters()
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.resource_pair_encoder.reset_parameters()
        for normalization in self.node_norms.values():
            normalization.reset_parameters()

        for transform in self.relation_source_transforms.values():
            nn.init.xavier_uniform_(transform.weight)
        for encoder in self.relation_edge_encoders.values():
            if isinstance(encoder, PermutationInvariantResourceEncoder):
                encoder.reset_parameters()
            else:
                nn.init.xavier_uniform_(encoder.weight)
        for gate in self.relation_gates.values():
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

        delta_linear = self.legacy_delta_encoder[0]
        nn.init.xavier_uniform_(delta_linear.weight)
        nn.init.zeros_(delta_linear.bias)

        for layer in self.readout:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for cell in (
            self.border_update,
            self.other_update,
            self.time_resources_update,
        ):
            for gate_weight in cell.weight_ih.chunk(3, dim=0):
                nn.init.xavier_uniform_(gate_weight)
            for gate_weight in cell.weight_hh.chunk(3, dim=0):
                nn.init.orthogonal_(gate_weight)
            nn.init.zeros_(cell.bias_ih)
            nn.init.zeros_(cell.bias_hh)

    def _build_state(self, node_type: str, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected a 2-D feature tensor, got {tuple(features.shape)}")
        if node_type not in ENTITY_TO_INDEX:
            raise ValueError(f"Unsupported node type: {node_type!r}")

        expected_width = int(FEATURE_WIDTHS[node_type])
        if features.shape[1] != expected_width:
            raise ValueError(
                f"Node type {node_type!r} has feature width {features.shape[1]}; "
                f"expected {expected_width}."
            )

        encoded = self.node_encoders[node_type](features)
        type_embedding = self.entity_embedding.weight[ENTITY_TO_INDEX[node_type]]
        return self.node_norms[node_type](encoded + type_embedding.unsqueeze(0))

    @staticmethod
    def _node_batch(data: HeteroData, node_type: str) -> Tensor:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch is not None:
            return store.batch
        return torch.zeros(store.num_nodes, dtype=torch.long, device=store.x.device)

    def _relation_active_graphs(
        self,
        data: HeteroData,
        relation: Relation,
        graph_count: int,
    ) -> Tensor:
        edge_index = data[relation].edge_index
        active = torch.zeros(graph_count, dtype=torch.bool, device=edge_index.device)
        if edge_index.shape[1] > 0:
            destination_type = relation[2]
            destination_batch = self._node_batch(data, destination_type)
            edge_graphs = destination_batch.index_select(0, edge_index[1])
            active.index_fill_(0, edge_graphs, True)
        return active

    @staticmethod
    def _resource_channels(node_type: str, features: Tensor) -> Tensor:
        if node_type == "time-resources":
            return features[:, :4]
        if node_type in {"border", "other"}:
            return features[:, 1:5]
        raise ValueError(f"Unsupported node type: {node_type!r}")

    def _sum_messages(
        self,
        data: HeteroData,
        relation: Relation,
        states: Dict[str, Tensor],
    ) -> Tensor:
        source_type, _, destination_type = relation
        store = data[relation]
        edge_index = store.edge_index
        destination_count = states[destination_type].shape[0]
        if edge_index.shape[1] == 0:
            return states[destination_type].new_zeros(
                (destination_count, self.state_dim)
            )

        expected_width = int(EDGE_FEATURE_WIDTHS[relation])
        edge_features = store.edge_attr
        if edge_features.ndim != 2 or edge_features.shape != (
            edge_index.shape[1],
            expected_width,
        ):
            raise ValueError(
                f"Relation {relation} has edge_attr shape "
                f"{tuple(edge_features.shape)}; expected "
                f"({edge_index.shape[1]}, {expected_width})."
            )

        key = relation_module_key(relation)
        source_on_edges = states[source_type].index_select(0, edge_index[0])
        source_raw = data[source_type].x.index_select(0, edge_index[0])
        destination_raw = data[destination_type].x.index_select(0, edge_index[1])
        resource_pair_features = self.resource_pair_encoder(
            self._resource_channels(source_type, source_raw),
            self._resource_channels(destination_type, destination_raw),
        )
        encoded_edge_features = self.relation_edge_encoders[key](edge_features)
        encoded_relation_context = encoded_edge_features + resource_pair_features
        message = (
            self.relation_source_transforms[key](source_on_edges)
            + encoded_relation_context
        )
        gate_input = torch.cat((source_on_edges, encoded_relation_context), dim=-1)
        gate = torch.sigmoid(self.relation_gates[key](gate_input))

        # Relation-local symmetric normalization for directed/bipartite edges:
        # 1 / sqrt(out_degree(source) * in_degree(destination)). The GRU keeps
        # the destination's old state, so explicit self-loops are unnecessary.
        source_degree = torch.bincount(
            edge_index[0],
            minlength=states[source_type].shape[0],
        ).to(dtype=torch.float32)
        destination_degree = torch.bincount(
            edge_index[1],
            minlength=destination_count,
        ).to(dtype=torch.float32)
        degree_normalization = torch.rsqrt(
            source_degree.index_select(0, edge_index[0]).clamp_min(1.0)
            * destination_degree.index_select(0, edge_index[1]).clamp_min(1.0)
        ).to(dtype=message.dtype)
        message = message * gate * degree_normalization.unsqueeze(-1)

        aggregated = message.new_zeros((destination_count, self.state_dim))
        aggregated.index_add_(0, edge_index[1], message)
        return aggregated

    def _sum_weighted_messages(
        self,
        data: HeteroData,
        relation: Relation,
        states: Dict[str, Tensor],
    ) -> Tensor:
        source_type, _, destination_type = relation
        store = data[relation]
        return self.weighted_propagation(
            states[source_type],
            states[destination_type],
            store.edge_index,
            store.edge_weight,
            self.legacy_delta_encoder(store.edge_attr),
        )

    def _apply_relation(
        self,
        data: HeteroData,
        relation: Relation,
        states: Dict[str, Tensor],
        node_batch: Dict[str, Tensor],
        graph_count: int,
    ) -> None:
        self._apply_relations(
            data,
            (relation,),
            states,
            node_batch,
            graph_count,
        )

    def _apply_relations(
        self,
        data: HeteroData,
        relations: PropagationStage,
        states: Dict[str, Tensor],
        node_batch: Dict[str, Tensor],
        graph_count: int,
    ) -> None:
        if not relations:
            raise ValueError("A propagation stage must contain at least one relation.")
        destination_types = {relation[2] for relation in relations}
        if len(destination_types) != 1:
            raise ValueError("All relations in a propagation stage must share a destination.")

        destination_type = relations[0][2]
        update_cell = getattr(self, f"{destination_type.replace('-', '_')}_update")
        message = self._sum_messages(data, relations[0], states)
        active = self._relation_active_graphs(data, relations[0], graph_count)
        for relation in relations[1:]:
            message = message + self._sum_messages(data, relation, states)
            active = active | self._relation_active_graphs(data, relation, graph_count)

        states[destination_type] = self._masked_gru_update(
            update_cell,
            message,
            states[destination_type],
            node_batch[destination_type],
            active,
        )

    @staticmethod
    def _masked_gru_update(
        cell: nn.GRUCell,
        message: Tensor,
        old_state: Tensor,
        destination_batch: Tensor,
        active_graphs: Tensor,
    ) -> Tensor:
        if old_state.shape[0] == 0:
            return old_state
        candidate = cell(message, old_state)
        active_nodes = active_graphs.index_select(0, destination_batch).unsqueeze(1)
        return torch.where(active_nodes, candidate, old_state)

    def _build_readout_features(
        self,
        data: HeteroData,
        states: Dict[str, Tensor],
    ) -> Tensor:
        other_context = self._sum_weighted_messages(
            data,
            LEGACY_OTHER_BORDER_READOUT,
            states,
        )
        time_resources_context = self._sum_messages(data, TIME_RESOURCES_BORDER, states)
        return torch.cat(
            (states["border"], other_context, time_resources_context),
            dim=-1,
        )

    def forward(self, data: HeteroData) -> Tensor:
        states: Dict[str, Tensor] = {
            "border": self._build_state("border", data["border"].x),
            "other": self._build_state("other", data["other"].x),
            "time-resources": self._build_state(
                "time-resources",
                data["time-resources"].x,
            ),
        }
        node_batch = {
            node_type: self._node_batch(data, node_type)
            for node_type in ("border", "other", "time-resources")
        }
        graph_count = batch_graph_count(data)

        for _ in range(self.num_iterations):
            for relations in PROPAGATION_STAGES:
                self._apply_relations(
                data,
                relations,
                states,
                node_batch,
                graph_count,
            )

        readout_features = self._build_readout_features(data, states)
        return self.readout(readout_features).squeeze(-1)

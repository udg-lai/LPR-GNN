"""Cache only non-differentiable batch metadata during a training forward.

Learned encodings are recomputed in their original order at every stage, so
parameter-gradient accumulation and dropout consumption are not regrouped.
"""
import torch
from .data import ENTITIES, EDGE_FEATURE_WIDTHS, batch_graph_count
from .model import PROPAGATION_RELATIONS, PROPAGATION_STAGES, relation_module_key


def forward_training(model, data):
    # Preserve the original graph for input-gradient use cases.
    if any(data[t].x.requires_grad for t in ENTITIES) or any(
        data[r].edge_attr.requires_grad for r in PROPAGATION_RELATIONS
    ):
        return model.forward_reference(data)
    states = {t: model._build_state(t, data[t].x) for t in ENTITIES}
    node_batch = {t: model._node_batch(data, t) for t in ENTITIES}
    graph_count = batch_graph_count(data)
    metadata, active_graphs = {}, {}
    for relation in PROPAGATION_RELATIONS:
        source_type, _, destination_type = relation
        edge_index = data[relation].edge_index
        active_graphs[relation] = model._relation_active_graphs(data, relation, graph_count)
        if edge_index.shape[1] == 0:
            metadata[relation] = None
            continue
        features = data[relation].edge_attr
        if features.ndim != 2 or features.shape != (edge_index.shape[1], EDGE_FEATURE_WIDTHS[relation]):
            raise ValueError(f'Invalid edge features for {relation}: {tuple(features.shape)}')
        source_raw = data[source_type].x.index_select(0, edge_index[0])
        destination_raw = data[destination_type].x.index_select(0, edge_index[1])
        source_degree = torch.bincount(edge_index[0], minlength=states[source_type].shape[0]).to(dtype=torch.float32)
        destination_degree = torch.bincount(edge_index[1], minlength=states[destination_type].shape[0]).to(dtype=torch.float32)
        normalization = torch.rsqrt(
            source_degree.index_select(0, edge_index[0]).clamp_min(1.0)
            * destination_degree.index_select(0, edge_index[1]).clamp_min(1.0))
        metadata[relation] = (edge_index, features, relation_module_key(relation),
            model._resource_values(source_type, source_raw),
            model._resource_values(destination_type, destination_raw), normalization)
    stages = []
    for relations in PROPAGATION_STAGES:
        destination = relations[0][2]
        active = active_graphs[relations[0]]
        for relation in relations[1:]:
            active = active | active_graphs[relation]
        active_nodes = active.index_select(0, node_batch[destination]).unsqueeze(1)
        stages.append((relations, destination,
            getattr(model, f"{destination.replace('-', '_')}_update"), active_nodes))

    def messages(relation):
        source_type, _, destination_type = relation
        count = states[destination_type].shape[0]
        entry = metadata[relation]
        if entry is None:
            return states[destination_type].new_zeros((count, model.state_dim))
        edge_index, features, key, source_resources, destination_resources, normalization = entry
        source = states[source_type].index_select(0, edge_index[0])
        pair = model.resource_pair_encoder(source_resources, destination_resources)
        encoded = model.relation_edge_encoders[key](features)
        context = encoded + pair
        message = model.relation_source_transforms[key](source) + context
        gate = torch.sigmoid(model.relation_gates[key](torch.cat((source, context), dim=-1)))
        message = message * gate * normalization.to(dtype=message.dtype).unsqueeze(-1)
        aggregated = message.new_zeros((count, model.state_dim))
        aggregated.index_add_(0, edge_index[1], message)
        return aggregated

    for _ in range(model.num_iterations):
        for relations, destination, cell, active_nodes in stages:
            message = messages(relations[0])
            for relation in relations[1:]:
                message = message + messages(relation)
            old = states[destination]
            if old.shape[0] != 0:
                candidate = cell(message, old)
                states[destination] = torch.where(active_nodes, candidate, old)
    # Keep the original readout, including its final message calculation.
    return model.readout(model._build_readout_features(data, states)).squeeze(-1)

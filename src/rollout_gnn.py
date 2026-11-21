"""Evaluation-only invariant hoisting, preserving the model's arithmetic order.

All caches are local to a single forward. Parameters, buffers, GRU stages and
edge reduction order are unchanged; no precision conversion or kernel fusion.
"""
import torch

from .data import ENTITIES, EDGE_FEATURE_WIDTHS, TIME_RESOURCES_BORDER, batch_graph_count
from .model import PROPAGATION_RELATIONS, PROPAGATION_STAGES, relation_module_key


def forward_rollout(model, data):
    states = {entity: model._build_state(entity, data[entity].x) for entity in ENTITIES}
    node_batch = {entity: model._node_batch(data, entity) for entity in ENTITIES}
    graph_count = batch_graph_count(data)
    contexts, active_graphs = {}, {}

    for relation in PROPAGATION_RELATIONS:
        source_type, _, destination_type = relation
        store = data[relation]
        edge_index = store.edge_index
        active_graphs[relation] = model._relation_active_graphs(data, relation, graph_count)
        if edge_index.shape[1] == 0:
            contexts[relation] = None
            continue
        edge_features = store.edge_attr
        expected_width = int(EDGE_FEATURE_WIDTHS[relation])
        if edge_features.ndim != 2 or edge_features.shape != (edge_index.shape[1], expected_width):
            raise ValueError(f"Relation {relation} has edge_attr shape {tuple(edge_features.shape)}; "
                             f"expected ({edge_index.shape[1]}, {expected_width}).")
        key = relation_module_key(relation)
        source_raw = data[source_type].x.index_select(0, edge_index[0])
        destination_raw = data[destination_type].x.index_select(0, edge_index[1])
        pair = model.resource_pair_encoder(
            model._resource_values(source_type, source_raw),
            model._resource_values(destination_type, destination_raw))
        encoded = model.relation_edge_encoders[key](edge_features)
        context = encoded + pair
        source_degree = torch.bincount(edge_index[0], minlength=states[source_type].shape[0]).to(dtype=torch.float32)
        destination_degree = torch.bincount(edge_index[1], minlength=states[destination_type].shape[0]).to(dtype=torch.float32)
        normalization = torch.rsqrt(
            source_degree.index_select(0, edge_index[0]).clamp_min(1.0)
            * destination_degree.index_select(0, edge_index[1]).clamp_min(1.0))
        contexts[relation] = (edge_index[0], edge_index[1], context, normalization,
                              model.relation_source_transforms[key], model.relation_gates[key])

    stages = []
    for relations in PROPAGATION_STAGES:
        destination = relations[0][2]
        active = active_graphs[relations[0]]
        for relation in relations[1:]:
            active = active | active_graphs[relation]
        active_nodes = active.index_select(0, node_batch[destination]).unsqueeze(1)
        cell = getattr(model, f"{destination.replace('-', '_')}_update")
        stages.append((relations, destination, cell, active_nodes))

    def messages(relation):
        source_type, _, destination_type = relation
        entry = contexts[relation]
        count = states[destination_type].shape[0]
        if entry is None:
            return states[destination_type].new_zeros((count, model.state_dim))
        source_index, target_index, context, normalization, transform, gate_layer = entry
        source = states[source_type].index_select(0, source_index)
        message = transform(source) + context
        gate = torch.sigmoid(gate_layer(torch.cat((source, context), dim=-1)))
        message = message * gate * normalization.to(dtype=message.dtype).unsqueeze(-1)
        aggregated = message.new_zeros((count, model.state_dim))
        aggregated.index_add_(0, target_index, message)
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

    # The legacy readout is already computed once in the original forward.
    from .data import LEGACY_OTHER_BORDER_READOUT
    other = model._sum_weighted_messages(data, LEGACY_OTHER_BORDER_READOUT, states)
    time = messages(TIME_RESOURCES_BORDER)
    readout = torch.cat((states['border'], other, time), dim=-1)
    return model.readout(readout).squeeze(-1)

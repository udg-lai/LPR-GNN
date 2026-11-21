"""Direct, order-preserving PyG graph construction for scheduling rollouts.

The node-link exporter remains the reference implementation for datasets and
custom transforms. This builder only implements its None/GraphNormalizer paths.
Caches belong to one schedule, never to a global instance or training dataset.
"""
from __future__ import annotations

import json
import math
import numpy as np
import torch
from torch_geometric.data import HeteroData

from . import data as schema
from .normalization import GraphNormalizer
from .rcpsp_instance import INF


class RolloutGraphBuilder:
    def __init__(self, state, transform):
        from .rcpsp_schedule import _resources_for_node

        self.transform = transform
        self.reference = None if transform is None else transform.temporal_reference
        self.clip = False if transform is None else transform.resource_clip
        if self.reference is not None and (not math.isfinite(self.reference) or self.reference <= 0):
            raise ValueError("The temporal log reference must be finite and positive.")
        self.log_denominator = None if transform is None else math.log1p(self.reference)
        self.logs = {}
        self.capacity = np.asarray((list(state.resource_levels) + [0] * 4)[:4], dtype=np.float64)
        if (self.capacity < 0).any():
            raise ValueError("Resource capacities must be non-negative.")
        self.duration = np.asarray(state.nodes_dict['duration'], dtype=np.int64)
        demands = np.asarray([_resources_for_node(state.nodes_dict, i)
                              for i in range(len(self.duration))], dtype=np.float64)
        self.activity_x = np.column_stack((self.temporal(self.duration), self.resources(demands)))
        original = np.asarray(state.original_matrix, dtype=np.int64)
        valid = original != INF
        np.fill_diagonal(valid, False)
        self.os, self.ot = np.nonzero(valid)
        self.od = self.temporal(original[self.os, self.ot])
        # Scheduling updates only row zero of the closed distance matrix.
        # Cache the immutable activity-to-activity readout distances as well.
        self.closed = np.asarray(state.matrix, dtype=np.int64)
        self.closed_normalized = np.zeros_like(self.closed, dtype=np.float64)
        reachable = self.closed > INF
        self.closed_normalized[reachable] = self.temporal(self.closed[reachable])
        self.temporal_cache = None

    def temporal(self, values):
        values = np.asarray(values)
        if self.transform is None:
            return values.astype(np.float64)
        if (values < 0).any():
            raise ValueError("Temporal quantities must be non-negative.")
        # Use the same scalar math as the reference, including float32 rounding.
        unique, inverse = np.unique(values, return_inverse=True)
        result = []
        for value in unique:
            key = int(value)
            if key not in self.logs:
                self.logs[key] = math.log1p(float(key)) / self.log_denominator
            result.append(self.logs[key])
        return np.asarray(result, dtype=np.float64)[inverse].reshape(values.shape)

    def resources(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.transform is None:
            return values
        if (values < 0).any():
            raise ValueError("Resource quantities must be non-negative.")
        out = np.divide(values, self.capacity, out=np.zeros_like(values), where=self.capacity != 0)
        return np.clip(out, 0, 1) if self.clip else out

    def time_edges(self, length):
        from .rcpsp_schedule import TEMPORAL_EDGE_SCALES

        key = (length, tuple(TEMPORAL_EDGE_SCALES))
        if self.temporal_cache is not None and self.temporal_cache[0] == key:
            return self.temporal_cache[1:]
        pairs, attrs = [], []
        for distance in TEMPORAL_EDGE_SCALES:
            distance = int(distance)
            if distance <= 0:
                raise ValueError("Temporal edge scales must be positive integers.")
            n = max(0, length - distance)
            ix = np.arange(n)
            pair = np.empty((2, 2 * n), dtype=np.int64)
            pair[:, 0::2] = np.vstack((ix, ix + distance))
            pair[:, 1::2] = np.vstack((ix + distance, ix))
            attr = np.empty((2 * n, 2), dtype=np.float64)
            attr[:, 0] = np.tile([1., 0.], n)
            attr[:, 1] = self.temporal(np.asarray([distance]))[0]
            pairs.append(pair)
            attrs.append(attr)
        index = torch.from_numpy(np.concatenate(pairs, axis=1))
        features = torch.from_numpy(np.concatenate(attrs).astype(np.float32))
        # One entry only: changing horizons cannot accumulate unbounded caches.
        self.temporal_cache = (key, index, features)
        return index, features

    def build(self, state):
        active = np.asarray(sorted(state.border | state.others), dtype=np.int64)
        if not len(active):
            raise ValueError("Cannot build a graph without unscheduled nodes.")
        is_border = np.zeros(len(self.duration), dtype=bool)
        is_border[list(state.border)] = True
        b = active[is_border[active]]
        o = active[~is_border[active]]
        if not len(b):
            raise ValueError("Every sample must contain at least one border node.")
        local = np.full(len(self.duration), -1, dtype=np.int64)
        local[b], local[o] = np.arange(len(b)), np.arange(len(o))
        present = np.zeros(len(self.duration), dtype=bool)
        present[active] = True
        starts = np.asarray(state.matrix[0], dtype=np.int64)
        minimum, maximum = int(starts[active].min()), int(starts[-1]) + 1
        if minimum < 0 or maximum > state.resource_time.shape[0] or maximum <= minimum:
            raise RuntimeError("Invalid time-resource window.")
        length = maximum - minimum
        raw = np.asarray(state.resource_time[minimum:maximum, :4], dtype=np.float64)
        if raw.shape[1] < 4:
            raw = np.pad(raw, ((0, 0), (0, 4 - raw.shape[1])))
        free = self.resources(raw)
        relative = np.arange(length, dtype=np.float64) / max(1, length - 1)
        data = HeteroData()
        for name, x in [('border', self.activity_x[b]), ('other', self.activity_x[o]),
                        ('time-resources', np.column_stack((free, relative)))]:
            data[name].x = torch.from_numpy(x.astype(np.float32))
            data[name].num_nodes = len(x)
        data['border'].y = torch.zeros(len(b), dtype=torch.float32)
        # Same store order and empty shapes as parse_rcpsp_graph.
        for relation in schema.RELATIONS:
            data[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[relation].edge_attr = torch.empty((0, schema.EDGE_FEATURE_WIDTHS.get(relation, 1)), dtype=torch.float32)
        legacy = schema.LEGACY_OTHER_BORDER_READOUT
        data[legacy].edge_weight = torch.empty(0, dtype=torch.float32)

        def edge(relation, source, target, features):
            data[relation].edge_index = torch.from_numpy(np.stack((source, target)).astype(np.int64))
            data[relation].edge_attr = torch.from_numpy(np.asarray(features, dtype=np.float32))

        for ids, forward, reverse in [(b, schema.BORDER_TIME_RESOURCES, schema.TIME_RESOURCES_BORDER),
                                      (o, schema.OTHER_TIME_RESOURCES, schema.TIME_RESOURCES_OTHER)]:
            times = starts[ids] - minimum
            edge(forward, local[ids], times, self.activity_x[ids])
            edge(reverse, times, local[ids], free[times])
        ti, ta = self.time_edges(length)
        # Do not expose cached storage to callers that might mutate a graph.
        data[schema.TIME_RESOURCES_TIME_RESOURCES].edge_index = ti.clone()
        data[schema.TIME_RESOURCES_TIME_RESOURCES].edge_attr = ta.clone()

        valid = present[self.os] & present[self.ot]
        s, t, d = self.os[valid], self.ot[valid], self.od[valid]
        if is_border[t].any():
            raise ValueError("Unexpected original precedence into a border node.")
        for mask, forward, reverse in [
            (is_border[s], schema.ORIGINAL_BORDER_OTHER_FORWARD, schema.ORIGINAL_BORDER_OTHER_REVERSE),
            (~is_border[s], schema.ORIGINAL_OTHER_OTHER_FORWARD, schema.ORIGINAL_OTHER_OTHER_REVERSE)]:
            ss, tt, dd = s[mask], t[mask], d[mask]
            edge(forward, local[ss], local[tt], np.column_stack((dd, self.activity_x[ss, 0])))
            edge(reverse, local[tt], local[ss], np.column_stack((dd, self.activity_x[tt, 0])))

        # Only dynamic Border -> Other closure edges survive the reference
        # parser (as reversed readout edges). Avoid materializing ignored edges.
        rows = self.closed[np.ix_(b, o)]
        bi, oi = np.nonzero(rows > INF)
        delta = rows[bi, oi]
        edge(legacy, local[o[oi]], local[b[bi]], self.closed_normalized[b[bi], o[oi]][:, None])
        weights = np.ones(len(delta), dtype=np.float64)
        if self.transform is not None:
            inverse = 1.0 / np.maximum(delta.astype(np.float64), 1e-8)
            # Python sum matches the reference's left-to-right accumulation.
            for i in range(len(b)):
                mask = bi == i
                values = inverse[mask]
                if len(values):
                    weights[mask] = values / sum(values.tolist())
        data[legacy].edge_weight = torch.from_numpy(weights.astype(np.float32))
        ids = [str(i) for i in b]
        data.border_node_ids_json = json.dumps(ids, ensure_ascii=False)
        return data, ids


def supports_transform(transform):
    # Subclasses can override __call__: never silently bypass custom behavior.
    return transform is None or type(transform) is GraphNormalizer

"""Canonical English feature names with read-only support for older graph schemas."""
from typing import Mapping


LEGACY_FEATURE_ALIASES = {
    "ressources_needed": "resources_needed",
    "ressources_needed_other": "resources_needed_other",
    "ressources_available": "resources_available",
    "ressource_free_time_resources_link_border": "resource_free_time_resources_link_border",
    "ressource_free_time_resources_link_other": "resource_free_time_resources_link_other",
    "ressource_needed_min_start_border": "resource_needed_min_start_border",
    "ressource_needed_min_start_other": "resource_needed_min_start_other",
}


# Translate feature keys without changing values, topology, or the input graph.
# Parameters: graph: node-link graph
# Returns: dict.
def canonicalize_graph_features(graph: Mapping) -> dict:
    """Translate feature keys without changing values, topology, or the input graph.

    Both spellings may be present only when they contain identical values.
    Newly generated graphs always use the canonical English spellings.
    """
    output = dict(graph)
    for collection in ("nodes", "links", "edges"):
        records = graph.get(collection)
        if not isinstance(records, list):
            continue
        translated = []
        for record in records:
            item = dict(record)
            for old_name, new_name in LEGACY_FEATURE_ALIASES.items():
                if old_name not in item:
                    continue
                if new_name in item and item[new_name] != item[old_name]:
                    raise ValueError(f"Conflicting graph features: {old_name!r} and {new_name!r}.")
                item[new_name] = item.pop(old_name)
            translated.append(item)
        output[collection] = translated
    return output

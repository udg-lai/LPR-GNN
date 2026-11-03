from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


# Canonicalize an instance filename for bound lookup.
# Parameters: value: input value
# Returns: str.
def normalize_instance_name(value: str | Path) -> str:
    name = Path(str(value).replace("\\", "/")).name.lower().strip()
    if name.endswith(".pat.data"):
        name = name[: -len(".pat.data")]
    for extension in (".rcp", ".sm", ".txt", ".data", ".pat"):
        if name.endswith(extension):
            name = name[: -len(extension)]
    return name.replace("-", "_")


# Infer the PSPLIB family size from a filename.
# Parameters: path: input/output path
# Returns: int | None.
def infer_problem_size(path: str | Path) -> int | None:
    match = re.search(r"j(\d+)", Path(path).name.lower())
    return int(match.group(1)) if match else None


# Build equivalent PSPLIB bound-lookup keys.
# Parameters: problem_size: PSPLIB activity count; group: PSPLIB group ID; instance: within-group instance ID
# Returns: list[str].
def _instance_keys(problem_size: int, group: int, instance: int) -> list[str]:
    return [
        f"j{problem_size}{group}_{instance}",
        f"j{problem_size}_{group}_{instance}",
    ]


# Read reference bounds into a filename-keyed mapping.
# Parameters: path: input/output path; problem_size: PSPLIB activity count
# Returns: Dict[str, int].
def load_bounds_file(path: str | Path, problem_size: int | None = None) -> Dict[str, int]:
    path = Path(path)
    problem_size = infer_problem_size(path) if problem_size is None else problem_size
    bounds: Dict[str, int] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "-" in line and ":" in line:
                instance_name, sequence = line.split("-", 1)
                last_value = None
                for pair in sequence.split(";"):
                    if ":" not in pair:
                        continue
                    _, value = pair.split(":", 1)
                    if value.strip():
                        last_value = int(float(value))
                if last_value is not None:
                    key = normalize_instance_name(instance_name)
                    bounds[key] = last_value
                    continue

            parts = (
                line.replace(";", " ")
                .replace(",", " ")
                .replace("\t", " ")
                .split()
            )
            if len(parts) < 3:
                continue
            try:
                group = int(parts[0])
                instance = int(parts[1])
                makespan = int(float(parts[2]))
            except ValueError:
                continue

            if problem_size is not None:
                for key in _instance_keys(problem_size, group, instance):
                    bounds[key] = makespan

    return bounds


# Look up an instance bound with a fallback.
# Parameters: filename: instance filename; bounds: reference makespans; default: fallback bound
# Returns: int.
def get_bound(filename: str | Path, bounds: Dict[str, int], default: int = -1) -> int:
    key = normalize_instance_name(filename)
    if key in bounds:
        return bounds[key]
    return default

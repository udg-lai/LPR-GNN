from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

INF = -999999
MODEL_RESOURCE_COUNT = 4


@dataclass
class MRCPSPInstance:
    """Single-mode RCPSP instance in the structure used by the original predictor."""

    n_activities: int
    n_resources: int
    capacities: List[int] = field(default_factory=list)
    durations: List[List[int]] = field(default_factory=list)
    demands: List[List[List[int]]] = field(default_factory=list)
    successors: List[List[int]] = field(default_factory=list)
    n_modes: List[int] = field(default_factory=list)

    # Allocate missing per-activity storage.
    # Parameters: none (uses instance/enclosing state).
    # Returns: None; updates state or writes output.
    def __post_init__(self) -> None:
        if not self.durations:
            self.durations = [[] for _ in range(self.n_activities)]
        if not self.successors:
            self.successors = [[] for _ in range(self.n_activities)]
        if not self.demands:
            self.demands = [
                [[] for _ in range(self.n_resources)]
                for _ in range(self.n_activities)
            ]

    # Set a resource capacity.
    # Parameters: resource_index: resource index; capacity: resource capacity
    # Returns: None; updates state or writes output.
    def set_capacity(self, resource_index: int, capacity: int) -> None:
        while len(self.capacities) <= resource_index:
            self.capacities.append(0)
        self.capacities[resource_index] = int(capacity)

    # Set an activity-mode duration.
    # Parameters: activity_index: activity index; mode_index: execution mode index
    #   duration: activity duration
    # Returns: None; updates state or writes output.
    def set_duration(self, activity_index: int, mode_index: int, duration: int) -> None:
        values = self.durations[activity_index]
        while len(values) <= mode_index:
            values.append(0)
        values[mode_index] = int(duration)

    # Set an activity-mode resource demand.
    # Parameters: activity_index: activity index; resource_index: resource index
    #   mode_index: execution mode index; demand: resource consumption
    # Returns: None; updates state or writes output.
    def set_demand(
        self,
        activity_index: int,
        resource_index: int,
        mode_index: int,
        demand: int,
    ) -> None:
        values = self.demands[activity_index][resource_index]
        while len(values) <= mode_index:
            values.append(0)
        values[mode_index] = int(demand)

    # Set an activity mode count.
    # Parameters: activity_index: activity index; n_modes: number of execution modes
    # Returns: None; updates state or writes output.
    def set_n_modes(self, activity_index: int, n_modes: int) -> None:
        while len(self.n_modes) <= activity_index:
            self.n_modes.append(1)
        self.n_modes[activity_index] = int(n_modes)

    # Append a precedence successor.
    # Parameters: activity_index: activity index; successor_index: successor activity index
    # Returns: None; updates state or writes output.
    def add_successor(self, activity_index: int, successor_index: int) -> None:
        self.successors[activity_index].append(int(successor_index))


# Dispatch the single-mode instance parser by format.
# Parameters: path: input/output path; file_type: input format override
# Returns: MRCPSPInstance.
def parse_mrcpsp(path: str | Path, file_type: str | None = None) -> MRCPSPInstance:
    path = Path(path)
    kind = (file_type or path.suffix.lstrip(".")).lower()
    if kind == "rcp":
        return parse_mrcpsp_from_rcp(path)
    if kind == "sm":
        return parse_mrcpsp_from_sm(path)
    raise ValueError(f"Unsupported RCPSP format '{kind}'. Expected .RCP or .sm.")


# Parse a Patterson-format instance.
# Parameters: path: input/output path
# Returns: MRCPSPInstance.
def parse_mrcpsp_from_rcp(path: str | Path) -> MRCPSPInstance:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        header = stream.readline().split()
        if len(header) < 2:
            raise ValueError(f"Invalid RCP header in {path}.")
        n_activities, original_resources = map(int, header[:2])

        capacities = list(map(int, stream.readline().split()))
        if len(capacities) < original_resources:
            raise ValueError(f"Not enough resource capacities in {path}.")

        n_resources = max(original_resources, MODEL_RESOURCE_COUNT)
        instance = MRCPSPInstance(n_activities, n_resources)
        for resource_index in range(n_resources):
            capacity = capacities[resource_index] if resource_index < original_resources else 0
            instance.set_capacity(resource_index, capacity)

        for activity in range(n_activities):
            line = stream.readline().split()
            if not line:
                raise ValueError(
                    f"Unexpected end of RCP file at activity {activity} in {path}."
                )
            required = original_resources + 2
            if len(line) < required:
                raise ValueError(f"Malformed activity line {activity + 3} in {path}.")

            instance.set_duration(activity, 0, int(line[0]))
            for resource_index in range(n_resources):
                demand = int(line[resource_index + 1]) if resource_index < original_resources else 0
                instance.set_demand(activity, resource_index, 0, demand)

            successor_count = int(line[original_resources + 1])
            successors = [int(value) - 1 for value in line[original_resources + 2 :]]
            if len(successors) != successor_count:
                raise ValueError(
                    f"Activity {activity} declares {successor_count} successors "
                    f"but contains {len(successors)} in {path}."
                )
            instance.set_n_modes(activity, 1)
            for successor in successors:
                instance.add_successor(activity, successor)

    return instance


# Read until a required input section is found.
# Parameters: stream: input stream; needle: required text marker; path: input/output path
# Returns: str.
def _read_until(stream, needle: str, path: Path) -> str:
    while True:
        line = stream.readline()
        if not line:
            raise ValueError(f"Could not find '{needle}' in {path}.")
        if needle.lower() in line.lower():
            return line


# Parse the PSPLIB single-mode .sm format used by the source project.
# Parameters: path: input/output path
# Returns: MRCPSPInstance.
def parse_mrcpsp_from_sm(path: str | Path) -> MRCPSPInstance:
    """Parse the PSPLIB single-mode ``.sm`` format used by the source project."""

    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        renewable_line = _read_until(stream, "renewable", path)
        try:
            original_resources = int(renewable_line.split(":", 1)[1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot read renewable resource count in {path}.") from exc

        _read_until(stream, "pronr.", path)
        project_line = stream.readline().split()
        if len(project_line) < 2:
            raise ValueError(f"Cannot read activity count in {path}.")
        internal_activities = int(project_line[1])
        n_activities = internal_activities + 2
        n_resources = max(original_resources, MODEL_RESOURCE_COUNT)
        instance = MRCPSPInstance(n_activities, n_resources)

        _read_until(stream, "jobnr.", path)
        successors: List[List[int]] = []
        for _ in range(internal_activities + 1):
            parts = stream.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Malformed precedence section in {path}.")
            count = int(parts[2])
            successors.append([int(value) - 1 for value in parts[3 : 3 + count]])
        successors.append([])

        # Advance to the requests/durations table header.
        _read_until(stream, "duration", path)
        # Most PSPLIB files have one additional header/separator line.
        line = stream.readline()
        while line and (not line.split() or not line.split()[0].lstrip("+-").isdigit()):
            line = stream.readline()

        durations: List[int] = []
        demands: List[List[int]] = []
        for activity in range(n_activities):
            parts = line.split() if activity == 0 else stream.readline().split()
            if len(parts) < 3 + original_resources:
                raise ValueError(f"Malformed request line for activity {activity} in {path}.")
            durations.append(int(parts[2]))
            demands.append([int(value) for value in parts[3 : 3 + original_resources]])

        availability_line = _read_until(stream, "RESOURCEAVAILABILITIES", path)
        del availability_line
        capacities: List[int] = []
        while True:
            line = stream.readline()
            if not line:
                break
            parts = line.split()
            try:
                values = [int(value) for value in parts]
            except ValueError:
                continue
            if len(values) >= original_resources:
                capacities = values[:original_resources]
                break
        if len(capacities) < original_resources:
            raise ValueError(f"Cannot read resource capacities in {path}.")

        for resource_index in range(n_resources):
            instance.set_capacity(
                resource_index,
                capacities[resource_index] if resource_index < original_resources else 0,
            )

        for activity in range(n_activities):
            instance.set_duration(activity, 0, durations[activity])
            instance.set_n_modes(activity, 1)
            for resource_index in range(n_resources):
                demand = (
                    demands[activity][resource_index]
                    if resource_index < original_resources
                    else 0
                )
                instance.set_demand(activity, resource_index, 0, demand)
            for successor in successors[activity]:
                instance.add_successor(activity, successor)

    return instance


# Extract activity features and resource capacities.
# Parameters: instance: parsed RCPSP instance
# Returns: activity feature dictionary and capacities.
def build_activity_resource_data(instance: MRCPSPInstance) -> tuple[dict, List[int]]:
    resource_levels = [int(value) for value in instance.capacities]
    nodes = list(range(instance.n_activities))
    result = {
        "nodes": nodes,
        "duration": [int(values[0]) for values in instance.durations],
    }
    for resource_index in range(len(resource_levels)):
        result[f"R{resource_index + 1}"] = [
            int(instance.demands[activity][resource_index][0])
            for activity in nodes
        ]
    return result, resource_levels


# Build the duration-weighted precedence matrix.
# Parameters: instance: parsed RCPSP instance; inf: unreachable-distance sentinel
# Returns: List[List[int]].
def create_distance_matrix(
    instance: MRCPSPInstance,
    inf: int = INF,
) -> List[List[int]]:
    matrix = [[inf] * instance.n_activities for _ in range(instance.n_activities)]
    for activity, successors in enumerate(instance.successors):
        for successor in successors:
            if not 0 <= successor < instance.n_activities:
                raise ValueError(
                    f"Successor {successor} of activity {activity} is outside the instance."
                )
            matrix[activity][successor] = int(instance.durations[activity][0])
    return matrix

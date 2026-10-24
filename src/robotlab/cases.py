from dataclasses import asdict, dataclass, field

import numpy as np

from .safety import Disturbance


@dataclass(frozen=True)
class Scenario:
    name: str = "warehouse"
    duration: float = 40.0
    dt: float = 0.002
    seed: int = 42
    start: tuple = (0.6, 0.6, 0.0)
    goal: tuple = (7.1, 5.2)
    width: float = 8.0
    height: float = 6.0
    resolution: float = 0.1
    radius: float = 0.22
    margin: float = 0.18
    cruise_speed: float = 0.6
    soc: float = 0.8
    sensor_latency: float = 0.0
    initial_temperature: float = 25.0
    shelves: tuple = ((2.0, 1.0, 2.8, 4.5), (4.2, 1.5, 5.0, 5.0))
    events: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("nonempty scenario name required")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("integer random seed required")
        if not np.all(np.isfinite([self.margin, self.sensor_latency, self.initial_temperature])):
            raise ValueError("finite scenario initial conditions required")
        values = (
            self.duration,
            self.dt,
            self.width,
            self.height,
            self.resolution,
            self.radius,
            self.cruise_speed,
        )
        if not all(np.isfinite(v) and v > 0 for v in values) or self.dt > 0.002 or self.margin < 0:
            raise ValueError("positive finite scenario geometry and timing required")
        if self.sensor_latency < 0 or not 0 < self.soc < 1 or self.seed < 0:
            raise ValueError("invalid scenario initial conditions")
        if (
            len(self.start) != 3
            or len(self.goal) != 2
            or not np.all(np.isfinite([*self.start, *self.goal, self.initial_temperature]))
        ):
            raise ValueError("finite planar start and goal required")
        for rectangle in self.shelves:
            if (
                len(rectangle) != 4
                or not np.all(np.isfinite(rectangle))
                or rectangle[0] >= rectangle[2]
                or rectangle[1] >= rectangle[3]
            ):
                raise ValueError("invalid shelf rectangle")
        for event in self.events:
            if not isinstance(event, Disturbance) or event.end > self.duration:
                raise ValueError("event outside scenario")
        for kind in {e.kind for e in self.events}:
            intervals = sorted((e.start, e.end) for e in self.events if e.kind == kind)
            if any(intervals[k + 1][0] < intervals[k][1] for k in range(len(intervals) - 1)):
                raise ValueError("overlapping disturbance intervals")

    def to_dict(self):
        return asdict(self)

    def obstacles(self):
        w, h = self.width, self.height
        return [
            *self.shelves,
            (-0.2, -0.2, w + 0.2, 0),
            (-0.2, h, w + 0.2, h + 0.2),
            (-0.2, 0, 0, h),
            (w, 0, w + 0.2, h),
        ]


def from_dict(data):
    from dataclasses import fields

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    allowed = {field.name for field in fields(Scenario)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError("unknown scenario fields: " + ", ".join(sorted(unknown)))
    values = dict(data)
    values["events"] = tuple(Disturbance(**event) for event in values.get("events", []))
    for key in ("start", "goal"):
        if key in values:
            values[key] = tuple(values[key])
    if "shelves" in values:
        values["shelves"] = tuple(tuple(r) for r in values["shelves"])
    return Scenario(**values)


def load_case(path):
    import json
    from pathlib import Path

    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def plan_route(case):
    from .mapping import OccupancyGrid, distance_field, inflated_mask, rasterize
    from .planning import astar, clearance_cost, shortcut

    grid = rasterize(
        OccupancyGrid(
            int(np.ceil(case.width / case.resolution)),
            int(np.ceil(case.height / case.resolution)),
            case.resolution,
        ),
        case.obstacles(),
    )
    blocked = inflated_mask(grid, case.radius, case.margin)
    path, cost = astar(
        blocked,
        grid.cell(case.start[:2]),
        grid.cell(case.goal),
        clearance_cost(distance_field(grid), weight=2, scale=0.4),
    )
    route = shortcut(path, blocked)
    points = np.array([grid.center(cell) for cell in route])
    # Exact endpoints are checked against the inflated map before substitution.
    from .planning import line_free

    def coordinate(point):
        cell = (np.asarray(point) - grid.origin) / grid.resolution - 0.5
        return cell[::-1]

    points[0], points[-1] = case.start[:2], case.goal
    if not all(
        line_free(blocked, coordinate(points[k]), coordinate(points[k + 1])) for k in range(len(points) - 1)
    ):
        raise ValueError("exact endpoints make the route unsafe")
    return points, grid, blocked, float(cost)


def experiment_catalog():
    from dataclasses import replace

    warehouse = Scenario()
    commissioning = Scenario(
        name="commissioning", duration=7, width=3, height=3, start=(0.6, 0.6, 0), goal=(2.3, 0.6), shelves=()
    )
    return {
        "warehouse": warehouse,
        "commissioning": commissioning,
        "grade-load": replace(commissioning, name="grade-load", events=(Disturbance(1, 4, "grade", 0.12),)),
        "traction-loss": replace(
            commissioning, name="traction-loss", events=(Disturbance(1, 2.5, "left_traction", 0.004),)
        ),
        "gyro-bias": replace(commissioning, name="gyro-bias", events=(Disturbance(1, 3, "gyro_bias", 0.15),)),
        "encoder-outage": replace(
            commissioning, name="encoder-outage", events=(Disturbance(1, 2, "encoder_dropout", 1),)
        ),
        "position-outage": replace(
            commissioning, name="position-outage", events=(Disturbance(1, 4, "position_dropout", 1),)
        ),
        "emergency-stop": replace(
            commissioning, name="emergency-stop", events=(Disturbance(2, 3, "emergency", 1),)
        ),
        "hot-windings": replace(commissioning, name="hot-windings", initial_temperature=98),
        "sensor-latency": replace(commissioning, name="sensor-latency", sensor_latency=0.05),
        "low-charge": replace(commissioning, name="low-charge", soc=0.08),
    }


def get_case(name):
    catalog = experiment_catalog()
    if name not in catalog:
        raise ValueError("unknown experiment: " + name)
    return catalog[name]

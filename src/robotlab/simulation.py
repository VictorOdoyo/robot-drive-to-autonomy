import numpy as np


class PeriodicClock:
    def __init__(self, period):
        if period <= 0:
            raise ValueError("positive clock period required")
        self.period, self.tick = period, 0

    def due(self, time):
        scheduled = self.tick * self.period
        if time + 1e-11 >= scheduled:
            self.tick += 1
            return True
        return False


def simulation_times(case):
    from .numerics import timeline

    clocks = [np.arange(0, case.duration, period) for period in (0.01, 0.02, 0.05, 0.2)]
    events = [value for event in case.events for value in (event.start, event.end)]
    candidates = timeline(case.duration, case.dt, np.r_[*clocks, events])
    merged = [candidates[0]]
    for value in candidates[1:]:
        if value - merged[-1] > 1e-11:
            merged.append(value)
    merged[-1] = case.duration
    for event in events:
        index = int(np.argmin(abs(np.asarray(merged) - event)))
        merged[index] = event
    return np.asarray(merged)


class SensorSuite:
    def __init__(self, case):
        from .sensors import DelayQueue, Encoder

        self.rng = np.random.default_rng(case.seed)
        self.encoder = Encoder()
        self.queue = DelayQueue(case.sensor_latency)
        self.counts = None
        self.last_time = None
        self.rates = np.zeros(2)
        self.age_stamp = -np.inf
        self.position_noise = 0.025
        self.gyro_noise = 0.008

    def sample_wheels(self, state, time, dropout=False):
        counts = [self.encoder.count(angle) for angle in state[14:16]]
        if self.counts is not None:
            dt = time - self.last_time
            increments = np.array(
                [self.encoder.increment(a, b) for a, b in zip(self.counts, counts, strict=True)]
            )
            if not dropout:
                self.queue.push(time, (increments / dt, dt))
        self.counts, self.last_time = counts, time
        for stamp, (rates, dt) in self.queue.ready(time):
            alpha = 1 - np.exp(-dt / 0.03)
            self.rates += alpha * (rates - self.rates)
            self.age_stamp = stamp
        return self.rates.copy()

    def gyro(self, state, bias=0.015):
        from .sensors import gyro_measurement

        return gyro_measurement(state[5], bias, self.gyro_noise, self.rng)

    def position(self, state):
        return state[6:8] + self.rng.normal(0, self.position_noise, 2)


def run(case=None, parameters=None):
    from dataclasses import asdict

    from .cases import Scenario, plan_route
    from .drive import DriveController, current_tick, set_twist, velocity_tick
    from .estimation import PoseEKF, update_gyro_bias, update_position
    from .geometry import circle_clearance, swept_clearance
    from .kinematics import DriveGeometry
    from .navigation import path_target, pure_pursuit, safe_speed
    from .plant import PlantParameters, bus_operating_point, energy_residual, initial_state, step, telemetry
    from .safety import FaultJournal, FaultLatch, disturbance_value, inhibit_reason

    case = Scenario() if case is None else case
    parameters = PlantParameters() if parameters is None else parameters
    route, grid, blocked, route_cost = plan_route(case)
    state = initial_state(parameters, case.start, case.soc)
    state[11:13] = case.initial_temperature
    initial = state.copy()
    controller = DriveController(parameters)
    sensors = SensorSuite(case)
    estimate = PoseEKF([*case.start, 0], np.diag([0.001, 0.001, 0.002, 0.001]))
    geometry = DriveGeometry(parameters.chassis.radius, parameters.chassis.radius, parameters.chassis.track)
    clocks = {
        name: PeriodicClock(period)
        for name, period in [
            ("wheel", 0.01),
            ("estimate", 0.02),
            ("position", 0.2),
            ("navigation", 0.05),
            ("log", 0.02),
        ]
    }
    latch, journal = FaultLatch(), FaultJournal()
    command = np.zeros(2)
    progress = 0.0
    last_estimate = 0.0
    last_fix = 0.0
    done = False
    termination = "duration"
    names = (
        "time",
        "state",
        "estimate",
        "covariance",
        "command",
        "duty",
        "current_reference",
        "wheel_reference",
        "bus_voltage",
        "battery_current",
        "slip_speed",
        "clearance",
        "fault",
    )
    logs = {name: [] for name in names}
    times = simulation_times(case)

    def disturbance(kind, time, default=0):
        return disturbance_value(case.events, kind, time, default)

    for index, time in enumerate(times):
        environment = {
            "grade": disturbance("grade", time),
            "traction": (disturbance("left_traction", time, 0.6), disturbance("right_traction", time, 0.6)),
            "external_force": disturbance("load", time),
        }
        wheel_due = clocks["wheel"].due(time)
        if wheel_due:
            sensors.sample_wheels(state, time, bool(disturbance("encoder_dropout", time)))
        if clocks["estimate"].due(time) and time > 0:
            speed, encoder_yaw = geometry.forward(sensors.rates)
            gyro = sensors.gyro(state, 0.015 + disturbance("gyro_bias", time))
            estimate.predict(speed, gyro, time - last_estimate)
            # Reject large disagreement; biased wheel yaw under slip is not a bias reference.
            if abs(gyro - encoder_yaw - estimate.mean[3]) < 0.08 and time - sensors.age_stamp < 0.05:
                update_gyro_bias(estimate, gyro, encoder_yaw, 0.0025)
            last_estimate = time
        if clocks["position"].due(time) and not disturbance("position_dropout", time):
            if update_position(estimate, sensors.position(state), np.eye(2) * sensors.position_noise**2):
                last_fix = time
        reason = inhibit_reason(
            emergency=bool(disturbance("emergency", time)),
            overcurrent=bool(np.max(abs(state[:2])) > 22),
            overtemperature=bool(np.max(state[11:13]) >= 100),
            undervoltage=bool(state[9] <= 0.02),
        )
        enabled = latch.update(time, reason)
        transient = (
            "encoder_timeout"
            if time > 0.2 and time - sensors.age_stamp > 0.15
            else "localization_timeout"
            if time - last_fix > 2.0
            else None
        )
        active_reason = latch.fault or transient
        journal.update(time, active_reason)
        if clocks["navigation"].due(time):
            position = estimate.mean[:2]
            remaining = np.linalg.norm(position - case.goal)
            if remaining < 0.12 and abs(state[4]) < 0.06:
                done = True
            if done or not enabled or transient:
                command = np.zeros(2)
            else:
                target, progress, _error = path_target(position, route, 0.4, progress)
                clearance = circle_clearance(position, case.radius, case.obstacles())
                speed = min(
                    case.cruise_speed,
                    1.2 * remaining,
                    safe_speed(
                        clearance, 0.6, 0.1, np.sqrt(np.max(np.linalg.eigvalsh(estimate.covariance[:2, :2])))
                    ),
                )
                command = pure_pursuit(estimate.mean[:3], target, speed)
                if swept_clearance(estimate.mean[:3], *command, 0.3, case.radius, case.obstacles()) < 0.02:
                    command[0] = 0.0
        if wheel_due:
            set_twist(controller, command, 0.01)
            velocity_tick(controller, sensors.rates, state[11:13], 0.01)
        bus, _ = bus_operating_point(parameters, state, controller.duty)
        if index < len(times) - 1:
            interval = times[index + 1] - time
            current_tick(controller, state[:2], sensors.rates, bus, interval)
        if clocks["log"].due(time) or index == len(times) - 1:
            data = telemetry(parameters, state, controller.duty, **environment)
            values = (
                time,
                state.copy(),
                estimate.mean.copy(),
                estimate.covariance.copy(),
                command.copy(),
                controller.duty.copy(),
                controller.current_reference.copy(),
                controller.reference.copy(),
                data["bus_voltage"],
                data["battery_current"],
                data["slip_speed"],
                circle_clearance(state[6:8], case.radius, case.obstacles()),
                active_reason or "",
            )
            for name, value in zip(names, values, strict=True):
                logs[name].append(value)
        if index == len(times) - 1:
            break
        state = step(parameters, state, controller.duty, times[index + 1] - time, **environment)
        if circle_clearance(state[6:8], case.radius, case.obstacles()) <= 0:
            termination = "collision"
            # Keep the violating endpoint visible in the trace.
            raise RuntimeError(f"physical footprint collision at {times[index + 1]:.6f} s")
        if not 0.001 < state[9] < 0.999:
            raise RuntimeError("battery charge-state operating boundary reached")
    result = {name: np.asarray(values) for name, values in logs.items()}
    result.update(
        {
            "case": case.to_dict(),
            "parameters": asdict(parameters),
            "route": route,
            "route_cost": route_cost,
            "map_probability": grid.probability(),
            "blocked": blocked,
            "journal": journal.events,
            "completed": done,
            "termination": termination,
            "energy_residual_j": energy_residual(parameters, initial, state),
        }
    )
    return result

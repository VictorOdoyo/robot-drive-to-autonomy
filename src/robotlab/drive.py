import numpy as np


class DriveController:
    def __init__(self, parameters=None, current_limit=8.0, wheel_limit=14.0):
        from .control import PI, current_gains
        from .plant import PlantParameters

        self.parameters = PlantParameters() if parameters is None else parameters
        if min(current_limit, wheel_limit) <= 0:
            raise ValueError("positive drive limits required")
        self.current_limit, self.wheel_limit = current_limit, wheel_limit
        kp, ki = current_gains(self.parameters.motor.resistance, self.parameters.motor.inductance, 150)
        self.current_loops = [PI(kp, ki, 100), PI(kp, ki, 100)]
        self.velocity_loops = [PI(1.0, 5.0, 15), PI(1.0, 5.0, 15)]
        self.reference = np.zeros(2)
        self.current_reference = np.zeros(2)
        self.duty = np.zeros(2)


def set_twist(controller, twist, dt, max_wheel_acceleration=12):
    from .control import slew
    from .kinematics import DriveGeometry, limit_twist

    chassis = controller.parameters.chassis
    geometry = DriveGeometry(chassis.radius, chassis.radius, chassis.track)
    target, _ = limit_twist(*twist, controller.wheel_limit, geometry)
    controller.reference = np.array(
        [
            slew(previous, next_, max_wheel_acceleration, dt)
            for previous, next_ in zip(controller.reference, target, strict=True)
        ]
    )
    return controller.reference.copy()


def velocity_tick(controller, measured_rates, temperatures, dt):
    from .power import thermal_limit

    measured_rates, temperatures = np.asarray(measured_rates), np.asarray(temperatures)
    for k, loop in enumerate(controller.velocity_loops):
        limit = thermal_limit(temperatures[k], controller.current_limit)
        controller.current_reference[k] = loop.step(
            controller.reference[k] - measured_rates[k], dt, -limit, limit
        )
    return controller.current_reference.copy()


def current_tick(controller, measured_current, wheel_rates, bus_voltage, dt):
    from .control import motor_feedforward

    if bus_voltage <= 0:
        raise ValueError("positive DC bus voltage required")
    motor = controller.parameters.motor
    feedforward = motor_feedforward(
        controller.current_reference,
        controller.parameters.ratio * np.asarray(wheel_rates),
        motor.resistance,
        motor.constant,
    )
    for k, loop in enumerate(controller.current_loops):
        voltage = loop.step(
            controller.current_reference[k] - measured_current[k],
            dt,
            -bus_voltage,
            bus_voltage,
            feedforward[k],
        )
        controller.duty[k] = voltage / bus_voltage
    return controller.duty.copy()


def regulated_drive(duration=1, speed=0.5, dt=0.002, parameters=None):
    from .numerics import timeline
    from .plant import PlantParameters, bus_operating_point, initial_state, step

    parameters = PlantParameters() if parameters is None else parameters
    controller = DriveController(parameters)
    state = initial_state(parameters)
    times = timeline(duration, dt, np.arange(0, duration, 0.01))
    states = [state.copy()]
    for k in range(len(times) - 1):
        time = times[k]
        if abs(time / 0.01 - round(time / 0.01)) < 1e-8:
            set_twist(controller, [speed, 0], 0.01)
            velocity_tick(controller, state[2:4], state[11:13], 0.01)
        voltage, _ = bus_operating_point(parameters, state, controller.duty)
        duty = current_tick(controller, state[:2], state[2:4], voltage, times[k + 1] - time)
        state = step(parameters, state, duty, times[k + 1] - time)
        states.append(state.copy())
    return times, np.array(states)

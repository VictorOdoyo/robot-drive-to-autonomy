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

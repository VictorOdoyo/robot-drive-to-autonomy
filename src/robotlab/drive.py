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

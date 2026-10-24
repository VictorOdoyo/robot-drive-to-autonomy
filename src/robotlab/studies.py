import numpy as np


def current_stability(periods, bandwidths, motor=None, delayed=False):
    from .control import current_gains, current_loop_matrix
    from .electrical import Motor

    motor = Motor() if motor is None else motor
    periods, bandwidths = np.asarray(periods), np.asarray(bandwidths)
    radius = np.empty((len(periods), len(bandwidths)))
    for row, dt in enumerate(periods):
        for col, bandwidth in enumerate(bandwidths):
            kp, ki = current_gains(motor.resistance, motor.inductance, bandwidth)
            matrix = current_loop_matrix(motor.resistance, motor.inductance, kp, ki, dt, delayed)
            radius[row, col] = np.max(abs(np.linalg.eigvals(matrix)))
    return radius

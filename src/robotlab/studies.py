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


def motor_frequency_response(frequencies, motor=None):
    from .electrical import Motor, motor_state_space

    motor = Motor() if motor is None else motor
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies < 0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("finite nonnegative angular frequencies required")
    matrix, inputs = motor_state_space(motor)
    return np.array(
        [np.linalg.solve(1j * frequency * np.eye(2) - matrix, inputs[:, 0]) for frequency in frequencies]
    )

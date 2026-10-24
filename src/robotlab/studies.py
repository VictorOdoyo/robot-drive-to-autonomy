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


def operating_point(speed=0.5, yaw_rate=0, parameters=None, **environment):
    from scipy.optimize import least_squares

    from .kinematics import DriveGeometry
    from .plant import PlantParameters, initial_state, mechanical_electrical_rhs

    parameters = PlantParameters() if parameters is None else parameters
    state = initial_state(parameters)
    state[4:6] = [speed, yaw_rate]
    geometry = DriveGeometry(parameters.chassis.radius, parameters.chassis.radius, parameters.chassis.track)
    wheels = geometry.inverse(speed, yaw_rate)
    guess = np.r_[np.ones(2) * 0.5, wheels, parameters.motor.constant * parameters.ratio * wheels / 27]

    def residual(value):
        state[:4] = value[:4]
        derivative, _, _ = mechanical_electrical_rhs(parameters, state, value[4:], **environment)
        return derivative[:6]

    solution = least_squares(
        residual,
        guess,
        bounds=(np.r_[[-20, -20], [-60, -60], [-1, -1]], np.r_[[20, 20], [60, 60], [1, 1]]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    if np.linalg.norm(solution.fun) > 1e-5:
        raise ValueError("requested chassis operating point is infeasible")
    state[:4] = solution.x[:4]
    return state, solution.x[4:]


def linearize_drive(parameters, state, duty, **environment):
    from .numerics import jacobian
    from .plant import mechanical_electrical_rhs

    state = np.asarray(state).copy()

    def dynamics(value, command):
        full = state.copy()
        full[:6] = value
        return mechanical_electrical_rhs(parameters, full, command, **environment)[0][:6]

    matrix = jacobian(lambda value: dynamics(value, duty), state[:6])
    inputs = jacobian(lambda command: dynamics(state[:6], command), duty)
    return matrix, inputs

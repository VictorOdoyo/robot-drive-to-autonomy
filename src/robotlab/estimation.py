import numpy as np


def gaussian_update(mean, covariance, innovation, observation, noise, gate=np.inf):
    from .numerics import covariance as validate

    mean, covariance, innovation, observation, noise = map(
        np.asarray, (mean, covariance, innovation, observation, noise)
    )
    validate(covariance, len(mean))
    validate(noise, len(innovation))
    residual_covariance = observation @ covariance @ observation.T + noise
    solved = np.linalg.solve(residual_covariance, innovation)
    nis = float(innovation @ solved)
    if nis > gate:
        return mean.copy(), covariance.copy(), nis, False
    gain = np.linalg.solve(residual_covariance, observation @ covariance).T
    remainder = np.eye(len(mean)) - gain @ observation
    posterior = remainder @ covariance @ remainder.T + gain @ noise @ gain.T
    return mean + gain @ innovation, (posterior + posterior.T) / 2, nis, True


def motion(mean, speed, gyro, dt):
    from .geometry import integrate

    mean = np.asarray(mean)
    pose = integrate(mean[:3], speed, gyro - mean[3], dt)
    # Keep angle unwrapped for linearization; normalize only the filter state.
    pose[2] = mean[2] + (gyro - mean[3]) * dt
    return np.r_[pose, mean[3]]


def motion_jacobians(mean, speed, gyro, dt):
    from .numerics import jacobian

    return (
        jacobian(lambda x: motion(x, speed, gyro, dt), mean),
        jacobian(lambda u: motion(mean, u[0], u[1], dt), [speed, gyro]),
    )


class PoseEKF:
    def __init__(self, mean=None, covariance=None):
        from .numerics import covariance as validate

        self.mean = np.array([0.0, 0.0, 0.0, 0.0] if mean is None else mean, dtype=float)
        if self.mean.shape != (4,) or not np.all(np.isfinite(self.mean)):
            raise ValueError("finite [x,y,yaw,bias] state required")
        self.covariance = validate(
            np.diag([0.01, 0.01, 0.01, 0.001]) if covariance is None else covariance, 4
        )
        self.last_nis = None

    def predict(self, speed, gyro, dt, speed_density=0.002, gyro_density=0.0004, bias_density=1e-7):
        from .geometry import wrap

        if dt <= 0 or min(speed_density, gyro_density, bias_density) < 0:
            raise ValueError("invalid EKF process noise or dt")
        transition, inputs = motion_jacobians(self.mean, speed, gyro, dt)
        process = inputs @ np.diag([speed_density, gyro_density]) @ inputs.T / dt
        process[3, 3] += bias_density * dt
        self.mean = motion(self.mean, speed, gyro, dt)
        self.mean[2] = wrap(self.mean[2])
        self.covariance = transition @ self.covariance @ transition.T + process
        self.covariance = (self.covariance + self.covariance.T) / 2
        return transition


def update_position(filter_, position, noise, gate=9.21):
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    result = gaussian_update(
        filter_.mean, filter_.covariance, np.asarray(position) - filter_.mean[:2], observation, noise, gate
    )
    filter_.mean, filter_.covariance, filter_.last_nis, accepted = result
    return accepted


def update_heading(filter_, heading, variance, gate=6.63):
    from .geometry import wrap

    result = gaussian_update(
        filter_.mean,
        filter_.covariance,
        [wrap(heading - filter_.mean[2])],
        [[0, 0, 1, 0]],
        [[variance]],
        gate,
    )
    filter_.mean, filter_.covariance, filter_.last_nis, accepted = result
    filter_.mean[2] = wrap(filter_.mean[2])
    return accepted


def landmark_observation(mean, landmark):
    from .geometry import wrap

    dx, dy = np.asarray(landmark) - mean[:2]
    distance2 = dx * dx + dy * dy
    if distance2 < 1e-12:
        raise ValueError("landmark coincides with robot")
    distance = np.sqrt(distance2)
    expected = np.array([distance, wrap(np.arctan2(dy, dx) - mean[2])])
    jacobian = np.array([[-dx / distance, -dy / distance, 0, 0], [dy / distance2, -dx / distance2, -1, 0]])
    return expected, jacobian


def update_landmark(filter_, landmark, measurement, noise, gate=9.21):
    from .geometry import wrap

    expected, observation = landmark_observation(filter_.mean, landmark)
    residual = np.asarray(measurement) - expected
    residual[1] = wrap(residual[1])
    result = gaussian_update(filter_.mean, filter_.covariance, residual, observation, noise, gate)
    filter_.mean, filter_.covariance, filter_.last_nis, accepted = result
    filter_.mean[2] = wrap(filter_.mean[2])
    return accepted


def update_gyro_bias(filter_, gyro, encoder_yaw, variance, gate=6.63):
    residual = gyro - encoder_yaw - filter_.mean[3]
    result = gaussian_update(filter_.mean, filter_.covariance, [residual], [[0, 0, 0, 1]], [[variance]], gate)
    filter_.mean, filter_.covariance, filter_.last_nis, accepted = result
    return accepted


def nees(estimate, truth, covariance, angle_index=2):
    from .geometry import wrap

    error = np.asarray(estimate) - truth
    if angle_index is not None:
        error[angle_index] = wrap(error[angle_index])
    return float(error @ np.linalg.solve(covariance, error))


def confidence_ellipse(covariance, probability=0.95):
    from scipy.stats import chi2

    from .numerics import covariance as validate

    matrix = validate(covariance, 2)
    if not 0 < probability < 1:
        raise ValueError("probability outside (0,1)")
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    axes = np.sqrt(np.maximum(values[order], 0) * chi2.ppf(probability, 2))
    direction = vectors[:, order[0]]
    return axes, float(np.arctan2(direction[1], direction[0]))

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

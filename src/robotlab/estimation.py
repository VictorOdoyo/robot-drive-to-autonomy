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

import numpy as np


def wrap(angle):
    """Principal angle in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

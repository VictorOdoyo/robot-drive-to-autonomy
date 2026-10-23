import numpy as np


def wrap(angle):
    """Principal angle in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    xy = np.asarray(a[:2]) + rotation(a[2]) @ b[:2]
    return np.r_[xy, wrap(a[2] + b[2])]


def inverse(pose):
    return np.r_[-rotation(pose[2]).T @ pose[:2], wrap(-pose[2])]

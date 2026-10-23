import numpy as np
import pytest

from robotlab import mechanics as m


def test_chassis_effective_inertia():
    robot = m.Chassis()
    inertia = robot.wheel_mass_matrix()
    assert np.all(np.linalg.eigvalsh(inertia) > 0)
    assert np.ones(2) @ inertia @ np.ones(2) == pytest.approx(
        robot.mass * robot.radius**2 + 2 * robot.wheel_inertia
    )

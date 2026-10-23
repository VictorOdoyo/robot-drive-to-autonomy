import numpy as np
import pytest

from robotlab import navigation as m


def test_lookahead_uses_arc_length():
    target, progress, error = m.path_target([0.8, 0.1], [[0, 0], [1, 0], [1, 1]], 0.5)
    assert np.allclose(target, [1, 0.3])
    assert progress == pytest.approx(0.8)
    assert error == pytest.approx(0.1)


def test_pursuit_steering_direction():
    assert m.pure_pursuit([0, 0, 0], [1, 1])[1] > 0
    assert np.allclose(m.pure_pursuit([0, 0, 0], [1, 0]), [0.6, 0])
    assert m.pure_pursuit([0, 0, 0], [-1, 0])[0] == 0


def test_trajectory_feedback_zero_error():
    pose = np.array([1.0, 2.0, 0.5])
    assert np.allclose(m.trajectory_control(pose, pose, [0.4, 0.2]), [0.4, 0.2])
    assert m.trajectory_control([0, 1, 0], [0, 0, 0], [0.4, 0])[1] < 0


def test_docking_turns_in_place_at_position_goal():
    twist = m.dock_control([1, 2, 0], [1, 2, 0.4])
    assert twist[0] == 0 and twist[1] > 0
    assert m.dock_control([0, 0, 0], [1, 0, 0])[0] > 0

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


def test_dynamic_window_respects_reachability():
    assert np.allclose(m.dynamic_window([0.5, 0.2], 0.1), [[0.44, 0.56], [0, 0.4]])


def test_dwa_stops_when_forward_commands_collide():
    assert np.allclose(m.dwa([0, 0, 0], [0.5, 0], [2, 0], [[0.25, -1, 0.4, 1]]), 0)
    assert m.dwa([0, 0, 0], [0, 0], [2, 0], [])[0] > 0


def test_moving_circle_collision():
    assert m.collision_time([3, 0], [-1, 0], 1) == pytest.approx(2)
    assert np.isinf(m.collision_time([3, 0], [1, 0], 1))
    assert np.isinf(m.collision_time([3, 2], [-1, 0], 1))


def test_safe_speed_exact_stopping_constraint():
    speed = m.safe_speed(1, 0.6, 0.2, 0.05)
    assert speed * 0.2 + speed**2 / (2 * 0.6) == pytest.approx(0.85)
    assert m.safe_speed(0.1, position_sigma=0.1) == 0


def test_waypoint_dwell_resets_on_departure():
    mission = m.WaypointMission([[1, 0]], dwell=0.5)
    assert mission.update([1, 0], 0) is not None
    mission.update([0, 0], 0.4)
    assert mission.update([1, 0], 0.6) is not None
    assert mission.update([1, 0], 1.2) is None and mission.done


def test_stall_requires_commanded_motion():
    time = np.linspace(0, 3, 31)
    position = np.zeros((31, 2))
    assert m.stalled(time, position, np.tile([0.5, 0], (31, 1)))
    assert not m.stalled(time, position, np.zeros((31, 2)))


def test_recovery_refuses_existing_collision():
    assert np.allclose(m.recovery_turn([0, 0, 0], [[-0.1, -0.1, 0.1, 0.1]]), 0)
    assert m.recovery_turn([0, 0, 0], [])[1] > 0


def test_energy_optimal_speed_stationarity():
    speed = m.economical_speed(2, 2, max_speed=2)
    assert -2 / speed**2 + 4 * speed == pytest.approx(0)
    assert m.economical_speed(20, 1, max_speed=0.8) == 0.8


def test_progress_floor_retains_nearby_segment_under_measurement_noise():
    path = [[0, 0], [0, 4], [4, 4], [4, 0]]
    target, progress, error = m.path_target([0.01, 0.99], path, 0.4, 1.0)
    assert progress == pytest.approx(1)
    assert np.allclose(target, [0, 1.4])
    assert error < 0.02

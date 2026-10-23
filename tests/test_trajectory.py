import numpy as np
import pytest

from robotlab import trajectory as m


def test_quintic_boundary_conditions():
    coefficients = m.quintic([1, 0.2, 0.1], [4, -0.3, 0.2], 2)
    assert np.allclose(m.polynomial_state(coefficients, 0), [1, 0.2, 0.1])
    assert m.polynomial_state(coefficients, 2) == pytest.approx([4, -0.3, 0.2])


def test_trapezoid_and_triangle():
    for distance in (0.1, 10):
        time, position, speed = m.trapezoid(distance, 1, 2)
        assert position[-1] == pytest.approx(distance)
        assert speed[0] == speed[-1] == 0
        assert np.max(speed) <= 1 + 1e-12
        assert np.max(abs(np.diff(speed) / np.diff(time))) <= 2 + 1e-10


def test_resample_duplicate_waypoints():
    points = m.resample_path([[0, 0], [0, 0], [1, 0], [1, 1]], 0.1)
    assert np.allclose(points[[0, -1]], [[0, 0], [1, 1]])
    assert m.arc_length(points)[-1] == pytest.approx(2)


def test_circular_path_curvature():
    angle = np.linspace(0, np.pi, 100)
    points = np.column_stack([2 * np.cos(angle), 2 * np.sin(angle)])
    assert np.allclose(m.path_curvature(points), 0.5, atol=1e-10)


def test_retiming_satisfies_squared_speed_constraints():
    points = m.resample_path([[0, 0], [3, 0], [3, 3]], 0.05)
    speed = m.speed_profile(points)
    ds = np.diff(m.arc_length(points))
    assert np.all(np.abs(np.diff(speed**2)) <= 2 * 0.6 * ds + 1e-12)
    assert np.all(speed**2 * np.abs(m.path_curvature(points)) <= 0.8 + 1e-12)


def test_profile_duration_constant_speed():
    points = [[0, 0], [1, 0], [2, 0]]
    assert np.allclose(m.profile_time(points, [0.5, 0.5, 0.5]), [0, 2, 4])
    with pytest.raises(ValueError):
        m.profile_time(points, [0, 0, 0])


def test_circle_reference_derivative():
    time = np.linspace(0, 1, 1001)
    pose, twist = m.circle(time, 2, 0.5)
    velocity = np.gradient(pose[:, :2], time, axis=0)
    assert np.allclose(np.linalg.norm(velocity[1:-1], axis=1), twist[1:-1, 0], atol=1e-7)


def test_figure_eight_closed_reference():
    period = 2 * np.pi / 0.2
    pose, twist = m.figure_eight(np.linspace(0, period, 1001), frequency=0.2)
    assert np.allclose(pose[0, :2], pose[-1, :2], atol=1e-12)
    assert np.all(np.isfinite(twist))
    assert np.max(twist[:, 1]) > 0 and np.min(twist[:, 1]) < 0


def test_wheel_reference_linear_acceleration():
    from robotlab.kinematics import DriveGeometry

    time = np.linspace(0, 1, 20)
    wheels, acc = m.wheel_reference(time, np.column_stack([time, np.zeros(20)]))
    assert np.allclose(acc, 1 / DriveGeometry().left_radius)
    assert np.allclose(wheels[-1], [12.5, 12.5])


def test_time_scaling_enforces_both_limits():
    time = np.linspace(0, 1, 20)
    wheels = np.column_stack([20 * time, 10 * time])
    scaled, rates, factor = m.time_scale(time, wheels, 10, 5)
    assert factor >= 2
    assert np.max(abs(rates)) <= 10
    assert np.max(abs(np.gradient(rates, scaled, axis=0))) <= 5 + 1e-12


def test_pose_interpolation_heading_continuity():
    result = m.sample_poses([0, 1], np.array([[0.0, 0, 3.1], [1, 0, -3.1]]), [0.5])
    assert result[0, 2] == pytest.approx(np.pi)

import numpy as np
import pytest

from robotlab import geometry as m


def test_rotations():
    assert m.wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert np.allclose(m.rotation(0.7).T @ m.rotation(0.7), np.eye(2))


def test_frame_inverse():
    pose = np.array([2.0, -1.0, 0.8])
    assert np.allclose(m.compose(pose, m.inverse(pose)), 0, atol=1e-14)
    assert np.allclose(m.compose([0, 0, np.pi / 2], [1, 0, 0]), [0, 1, np.pi / 2])


def test_point_cloud_roundtrip():
    points = np.array([[1, 2], [-3, 1], [0, 0]])
    pose = [4, 2, -0.5]
    assert np.allclose(m.transform_points(m.inverse(pose), m.transform_points(pose, points)), points)
    with pytest.raises(ValueError):
        m.transform_points(pose, [1, 2])


def test_exact_arc():
    assert np.allclose(m.integrate([0, 0, 0], 2, 1, np.pi), [0, 4, -np.pi], atol=1e-14)
    assert np.allclose(m.integrate([0, 0, 0], 2, 1e-14, 1), [2, 0, 0], atol=1e-12)


def test_body_error():
    assert np.allclose(m.pose_error([0, 0, np.pi / 2], [0, 2, np.pi / 2]), [2, 0, 0], atol=1e-14)


def test_interpolation_crosses_angle_branch():
    result = m.interpolate([0, 0, 3.1], [2, 4, -3.1], 0.5)
    assert np.allclose(result[:2], [1, 2])
    assert abs(result[2]) == pytest.approx(np.pi)


def test_segment_endpoints():
    point, fraction, distance = m.segment_projection([3, 1], [0, 0], [2, 0])
    assert np.allclose(point, [2, 0])
    assert fraction == 1
    assert distance == pytest.approx(np.sqrt(2))


def test_signed_path_error():
    cross, heading = m.path_error([1, 2, np.pi / 4], [0, 0], [3, 0])
    assert cross == 2
    assert heading == pytest.approx(np.pi / 4)


def test_signed_clearance():
    rect = [0, 0, 2, 2]
    assert m.rectangle_clearance([1, 1], rect) == -1
    assert m.circle_clearance([3, 1], 0.25, [rect]) == pytest.approx(0.75)
    assert m.rectangle_clearance([3, 3], rect) == pytest.approx(np.sqrt(2))


def test_swept_footprint_catches_tunneling():
    assert m.swept_clearance([0, 0, 0], 4, 0, 1, 0.2, [[1.9, -1, 2.1, 1]]) < 0
    assert m.swept_clearance([0, 2, 0], 4, 0, 1, 0.2, [[1.9, -1, 2.1, 1]]) > 0.7

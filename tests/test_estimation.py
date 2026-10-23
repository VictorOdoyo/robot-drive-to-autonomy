import numpy as np
import pytest

from robotlab import estimation as m


def test_gaussian_update_independent_scalar_solution():
    mean, cov, nis, accepted = m.gaussian_update([0.0], [[4.0]], [3.0], [[1.0]], [[5.0]])
    assert mean[0] == pytest.approx(4 / 3)
    assert cov[0, 0] == pytest.approx(20 / 9)
    assert nis == pytest.approx(1)
    assert accepted and np.all(np.linalg.eigvalsh(cov) > 0)


def test_bias_corrected_arc_and_jacobian():
    state = np.array([0.0, 0.0, 0.0, 0.1])
    result = m.motion(state, 1, 1.1, np.pi)
    assert np.allclose(result[:3], [0, 2, np.pi], atol=1e-14)
    transition, _ = m.motion_jacobians(state, 1, 1.1, 0.1)
    assert transition[2, 3] == pytest.approx(-0.1)


def test_ekf_prediction_preserves_psd():
    filt = m.PoseEKF()
    for _ in range(30):
        filt.predict(1, 0.2, 0.1)
    assert np.linalg.eigvalsh(filt.covariance).min() > 0
    assert filt.mean[0] > 2


def test_position_outlier_is_rejected():
    filt = m.PoseEKF()
    assert not m.update_position(filt, [100, 100], np.eye(2) * 0.01)
    assert np.allclose(filt.mean, 0)
    assert m.update_position(filt, [0.05, 0.05], np.eye(2) * 0.01)
    assert filt.covariance[0, 0] < 0.01


def test_heading_branch_cut():
    filt = m.PoseEKF([0, 0, 3.13, 0], np.eye(4) * 0.1)
    assert m.update_heading(filt, -3.13, 0.001)
    assert abs(filt.mean[2]) > 3.12
    assert filt.last_nis < 0.01


def test_landmark_jacobian_independent_differences():
    from robotlab.numerics import jacobian

    mean = np.array([1.0, 2.0, 0.3, 0.01])
    expected, analytic = m.landmark_observation(mean, [4, 6])
    numeric = jacobian(lambda x: m.landmark_observation(x, [4, 6])[0], mean)
    assert expected[0] == pytest.approx(5)
    assert np.allclose(analytic, numeric, atol=1e-8)


def test_landmark_reduces_uncertainty():
    filt = m.PoseEKF([0.1, 0.1, 0.05, 0], np.eye(4) * 0.1)
    before = np.trace(filt.covariance)
    assert m.update_landmark(filt, [3, 0], [3, 0], np.diag([0.01, 0.001]))
    assert np.trace(filt.covariance) < before


def test_bias_observability_from_wheels():
    filt = m.PoseEKF()
    for _ in range(50):
        assert m.update_gyro_bias(filt, 0.14, 0.1, 0.0001)
    assert filt.mean[3] == pytest.approx(0.04, abs=0.0001)


def test_nees_wraps_angles():
    value = m.nees([0, 0, 3.13], [0, 0, -3.13], np.eye(3))
    assert value < 0.001
    assert m.nees([2], [0], [[4]], None) == pytest.approx(1)


def test_uncertainty_ellipse():
    axes, _ = m.confidence_ellipse(np.diag([4.0, 1.0]))
    assert axes[0] / axes[1] == pytest.approx(2)
    assert axes[1] == pytest.approx(np.sqrt(5.991464547), rel=1e-8)


def test_heading_and_bias_observability():
    transition = np.eye(2)
    transition[0, 1] = -0.1
    rank, _ = m.observability_rank(transition, [[1, 0]])
    assert rank == 2
    assert m.observability_rank(np.eye(2), [[1, 0]])[0] == 1


def test_rts_uses_future_measurement():
    states, covs = m.rts_smooth(
        [[0.0], [1.0]], [[[1.0]], [[0.2]]], [[0.0], [0.0]], [[[1.0]], [[2.0]]], [[[1.0]]]
    )
    assert states[0, 0] == pytest.approx(0.5)
    assert covs[0, 0, 0] < 1

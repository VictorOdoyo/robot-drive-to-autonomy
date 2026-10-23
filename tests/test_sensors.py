import numpy as np
import pytest

from robotlab import sensors as m


def test_encoder_rollover():
    encoder = m.Encoder(1024, 16)
    a, b = encoder.count(-0.01), encoder.count(0.01)
    delta = encoder.increment(a, b)
    assert delta == pytest.approx(0.02, abs=2 * np.pi / 1024)
    assert np.isfinite(delta)


def test_encoder_uncertainty_rate_scaling():
    encoder = m.Encoder()
    assert m.encoder_speed_variance(encoder, 0.01) / m.encoder_speed_variance(encoder, 0.02) == pytest.approx(
        4
    )


def test_count_window_rejects_dc_count_offset():
    times = np.array([0, 0.1, 0.2, 0.3])
    counts = 100000 + 100 * times
    assert m.count_window_speed(times, counts, 100) == pytest.approx(2 * np.pi)


def test_seeded_gyro_noise():
    rng = np.random.default_rng(42)
    samples = np.array([m.gyro_measurement(0.2, 0.03, 0.01, rng) for _ in range(5000)])
    assert samples.mean() == pytest.approx(0.23, abs=0.0005)
    assert samples.std() == pytest.approx(0.01, rel=0.04)


def test_bias_noise_discretization():
    rng = np.random.default_rng(7)
    changes = np.array([m.bias_step(0, 0.02, 0.1, rng) for _ in range(10000)])
    assert changes.var() == pytest.approx(0.002, rel=0.04)


def test_adc_saturation():
    assert np.allclose(m.adc([-100, 0, 100], 20, 12), [-20, 0, 20])
    assert m.adc(1, 20, 12, 0.1) == pytest.approx(1.1, abs=20 / 2047)


def test_delayed_measurements_keep_acquisition_stamp():
    queue = m.DelayQueue(0.05)
    queue.push(0.1, 7)
    assert queue.ready(0.14) == []
    assert queue.ready(0.15) == [(0.1, 7)]
    assert queue.ready(1) == []


def test_packet_timing_checks():
    assert m.valid_stamp(0.9, 1, 0.8, 0.2)
    assert not m.valid_stamp(1.1, 1, 0.8, 0.2)
    assert not m.valid_stamp(0.7, 1, 0.8, 0.2)
    assert not m.valid_stamp(0.9, 1, 0.9, 0.2)


def test_lidar_rays():
    rectangles = [[2, -1, 3, 1]]
    assert m.ray_range([0, 0], 0, rectangles) == pytest.approx(2)
    assert m.ray_range([0, 0], np.pi / 2, rectangles) == 10
    assert m.ray_range([2.5, 0], 0, rectangles) == 0


def test_lidar_extrinsic():
    scan = m.lidar_scan([0, 0, 0], [0, np.pi], [[2, -1, 3, 1]], extrinsic=(0.2, 0, 0))
    assert np.allclose(scan, [1.8, 10])


def test_burst_dropout_boundaries():
    assert np.array_equal(m.dropout_mask([0, 0.1, 0.2, 0.3], [(0.1, 0.3)]), [True, False, False, True])


def test_stationary_bias_calibration():
    bias, variance = m.calibrate_bias([0.01, 0.02, 0.03])
    assert bias == pytest.approx(0.02)
    assert variance == pytest.approx(0.0001 / 3)

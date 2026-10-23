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

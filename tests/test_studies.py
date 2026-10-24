import numpy as np
import pytest

from robotlab import studies as m


def test_sampling_and_delay_stability_regions():
    periods = [0.0005, 0.001, 0.002]
    bandwidths = [100, 500, 1500]
    direct = m.current_stability(periods, bandwidths)
    delayed = m.current_stability(periods, bandwidths, delayed=True)
    assert np.all(direct[:, 0] < 1)
    assert np.any((direct < 1) & (delayed > 1))
    assert direct.shape == (3, 3)
    assert np.max(delayed) > pytest.approx(1).expected


def test_motor_dc_gain_matches_equilibrium():
    from robotlab.electrical import Motor, motor_equilibrium

    motor = Motor()
    response = m.motor_frequency_response([0, 1e8], motor)
    assert np.allclose(response[0].real, motor_equilibrium(motor, 1))
    assert abs(response[1, 0]) < abs(response[0, 0])


def test_loaded_operating_point():
    from robotlab.plant import PlantParameters, mechanical_electrical_rhs

    parameters = PlantParameters()
    state, duty = m.operating_point(0.5, 0.2, parameters, grade=0.03)
    derivative, _, _ = mechanical_electrical_rhs(parameters, state, duty, grade=0.03)
    assert np.allclose(derivative[:6], 0, atol=1e-5)
    assert state[4] == pytest.approx(0.5)
    assert np.all(state[:2] > 0)


def test_drive_linearization_predicts_small_perturbations():
    from robotlab.plant import PlantParameters, mechanical_electrical_rhs

    parameters = PlantParameters()
    state, duty = m.operating_point(0.4, 0, parameters)
    matrix, inputs = m.linearize_drive(parameters, state, duty)
    delta = np.array([1, -1, 2, -2, 0.1, -0.2]) * 1e-6
    perturbed = state.copy()
    perturbed[:6] += delta
    exact = mechanical_electrical_rhs(parameters, perturbed, duty)[0][:6]
    assert np.allclose(exact, matrix @ delta, atol=1e-7)
    assert inputs.shape == (6, 2)


def test_coupled_lqr_is_schur_stable():
    from robotlab.plant import PlantParameters

    design = m.drive_lqr(PlantParameters())
    assert np.max(abs(design["poles"])) < 1
    assert np.linalg.eigvalsh(design["cost"]).min() > 0


def test_nonlinear_lqr_reaches_loaded_speed():
    time, states = m.nonlinear_lqr_response(1.5, 0.4)
    assert time[-1] == pytest.approx(1.5)
    assert states[-1, 4] == pytest.approx(0.4, abs=0.015)
    assert np.max(abs(states[:, :2])) < 8.5


def test_copper_thermal_runaway_boundary():
    safe = m.copper_thermal_analysis(5)
    critical = safe["critical_current_a"]
    assert safe["stable"] and safe["equilibrium_c"] > 25
    assert not m.copper_thermal_analysis(critical * 1.01)["stable"]
    theta = safe["equilibrium_c"] - 25
    assert 5**2 * 0.35 * (1 + 0.00393 * theta) == pytest.approx(theta / 2)


def test_reconstructed_map_contains_free_and_occupied_evidence():
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.2, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    grid = m.mapping_reconstruction(result)
    assert np.any(grid.log_odds < 0) and np.any(grid.log_odds > 0)
    assert np.mean(grid.observed) > 0.1

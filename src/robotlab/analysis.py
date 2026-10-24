import numpy as np


def mission_metrics(result):
    from .geometry import wrap

    state, estimate, time = result["state"], result["estimate"], result["time"]
    position_error = np.linalg.norm(state[:, 6:8] - estimate[:, :2], axis=1)
    heading_error = wrap(state[:, 8] - estimate[:, 2])
    distance = float(np.sum(np.linalg.norm(np.diff(state[:, 6:8], axis=0), axis=1)))
    return {
        "completed": bool(result["completed"]),
        "duration_s": float(time[-1] - time[0]),
        "goal_error_m": float(np.linalg.norm(state[-1, 6:8] - result["case"]["goal"])),
        "distance_m": distance,
        "minimum_clearance_m": float(np.min(result["clearance"])),
        "peak_current_a": float(np.max(np.abs(state[:, :2]))),
        "peak_motor_temperature_c": float(np.max(state[:, 11:13])),
        "soc_consumed": float(state[0, 9] - state[-1, 9]),
        "dissipated_j": float(state[-1, 16] - state[0, 16]),
        "external_work_j": float(state[-1, 17] - state[0, 17]),
        "energy_residual_j": float(result["energy_residual_j"]),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_error**2))),
        "fault_transitions": len(result["journal"]),
    }


def consistency(result, probability=0.95):
    from scipy.stats import chi2

    from .estimation import nees

    if not 0 < probability < 1:
        raise ValueError("invalid confidence probability")
    errors = np.array(
        [
            nees(estimate[:3], truth[6:9], covariance[:3, :3])
            for estimate, truth, covariance in zip(
                result["estimate"], result["state"], result["covariance"], strict=True
            )
        ]
    )
    bound = float(chi2.ppf(probability, 3))
    return {
        "nees": errors,
        "bound": bound,
        "coverage": float(np.mean(errors <= bound)),
        "mean_nees": float(errors.mean()),
    }


def step_metrics(time, response, target, band=0.02):
    time, response = np.asarray(time), np.asarray(response)
    if response.shape != time.shape or len(time) < 2 or np.any(np.diff(time) <= 0) or band <= 0:
        raise ValueError("invalid response series")
    scale = max(abs(target - response[0]), 1e-12)
    outside = np.flatnonzero(abs(response - target) > band * scale)
    settling = (
        float(time[0] if len(outside) == 0 else time[outside[-1] + 1])
        if len(outside) == 0 or outside[-1] < len(time) - 1
        else None
    )
    direction = np.sign(target - response[0])
    overshoot = max(0, float(np.max(direction * (response - target)))) / scale
    return {
        "settling_s": settling,
        "overshoot_fraction": overshoot,
        "steady_error": float(target - np.mean(response[-max(1, len(response) // 10) :])),
    }

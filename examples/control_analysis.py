"""Run a coupled operating-point and feedback-design study."""

import json

import numpy as np

from robotlab.plant import PlantParameters
from robotlab.studies import copper_thermal_analysis, drive_lqr, linearize_drive, operating_point


def main():
    parameters = PlantParameters()
    state, duty = operating_point(0.5, 0.2, parameters, grade=0.03)
    matrix, inputs = linearize_drive(parameters, state, duty, grade=0.03)
    design = drive_lqr(parameters)
    print(
        json.dumps(
            {
                "steady_current_a": state[:2].tolist(),
                "steady_duty": duty.tolist(),
                "linear_model_shape": list(matrix.shape),
                "input_shape": list(inputs.shape),
                "lqr_max_pole_radius": float(np.max(abs(design["poles"]))),
                "thermal_design": copper_thermal_analysis(5),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

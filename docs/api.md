# Study APIs

Use the installed package from the repository root or any other directory.

## Loaded drive design

```python
import numpy as np
from robotlab.plant import PlantParameters
from robotlab.studies import operating_point, linearize_drive, drive_lqr

parameters = PlantParameters()
state, duty = operating_point(.5, .2, parameters, grade=.03)
A, B = linearize_drive(parameters, state, duty, grade=.03)
design = drive_lqr(parameters, speed=.5, dt=.002)
assert np.max(np.abs(design["poles"])) < 1
```

The operating-point solve fixes forward speed/yaw rate and solves currents,
wheel rates and duties so all six drive derivatives vanish. Battery SOC and
temperature are frozen parameters of this local linearization.

Run `uv run python examples/control_analysis.py` for a complete example.
`nonlinear_lqr_response` validates the design with nonlinear tires/battery,
full-state feedback and a current governor. Mission control uses nested PI.

## Delayed absolute fixes

```python
import numpy as np
from robotlab.estimation import PoseEKF, PositionReplay

history = PositionReplay(PoseEKF(), max_steps=100)
for _ in range(10):
    history.predict(.5, .02, .1)
history.correct(steps_ago=5, position=[.25, 0.], noise=np.eye(2)*.01)
```

The correction occurs five retained control steps before the current state.
Replay moves its anchor to the corrected epoch; fixes older than that anchor
are rejected. Acquisition-time handling belongs to the caller. The utility
is not a general factor-graph smoother.

## Known landmarks and occupancy reconstruction

`landmark_observation` returns predicted range/bearing and its analytic
Jacobian. `update_landmark` wraps bearing innovations and applies an NIS gate.

`mapping_reconstruction(result)` creates an occupancy grid from a recorded
mission's true simulated lidar rays and estimated poses. It requires the
scenario geometry retained in the archive. `icp(source, target)` registers
point clouds locally; correspondence limits and initial alignment matter.

## Other module entry points

| Module | Useful entry points |
|---|---|
| geometry | integrate, compose, inverse, swept_clearance |
| kinematics | DriveGeometry, calibrate_radii, calibrate_track |
| electrical | Motor, motor_step, motor_power, identify_winding |
| mechanics | Chassis, reflected_inertia, tire_force, backlash_torque |
| power | current_for_power, battery_current_bounds, reserve_margin |
| control | PI, discrete_lqr, current_loop_matrix |
| sensors | Encoder, DelayQueue, lidar_scan |
| estimation | gaussian_update, update_landmark, rts_smooth |
| mapping | OccupancyGrid, insert_scan, distance_field, icp |
| planning | astar, heading_path, space_time_path, energy_path |
| trajectory | quintic, speed_profile, trapezoid, figure_eight |
| navigation | pure_pursuit, trajectory_control, dwa |
| safety | FaultLatch, CommandWatchdog, Overload, FaultJournal |
| reporting | save_archive, load_archive, write_report, verify_report |

All lengths, angles, time, current and energy use SI units; temperatures are
degrees C. Array conventions and state ordering are documented in the model
guide. Inputs are numerical research interfaces rather than hardware messages.

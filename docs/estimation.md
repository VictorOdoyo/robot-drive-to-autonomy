# Estimation and sensing

The mission EKF estimates [x, y, yaw, gyro bias]. It propagates exact
constant-twist arcs using encoder-derived forward speed and a bias-corrected
gyro. The filter linearizes the unwrapped motion function before wrapping the
stored heading, avoiding a derivative discontinuity at the angle branch cut.

The mission supplies simulated independent absolute position fixes with
0.025 m standard deviation. They are not derived from the known map or lidar.
The gyro has white measurement noise and a default 0.015 rad/s bias.
Large encoder/gyro disagreements are excluded from gyro-bias correction.

The map used for mission planning is known in advance. This is localized
navigation, not a complete SLAM pipeline. Known-landmark range/bearing
updates, scan likelihoods, ICP and offline occupancy reconstruction are
separate research tools exposed by the package.

Gaussian updates use innovation gates and Joseph covariance stabilization.
The tests compare landmark Jacobians to independent finite differences,
check positive covariance eigenvalues and verify wrapped heading updates.
The replay utility handles delayed absolute position fixes within a bounded
control history and matches chronological filtering in its regression test.
The main mission's latency parameter delays encoder packets; it does not
activate delayed absolute-fix replay.

Noise densities in the mission filter are illustrative tuning values, not
identified from a physical IMU. Reports expose position/yaw RMSE, NEES and
95% chi-square coverage. A single deterministic run cannot establish
statistical consistency. Sequential filtered errors are correlated, so
coverage fractions are descriptive rather than independent confidence tests.

Rauch-Tung-Striebel smoothing, observability rank, confidence ellipses, robust
Huber noise inflation and wheel-quantization covariance propagation support
controlled estimation experiments. Their assumptions and input dimensions
are explicit in each API; they are not all enabled in the default mission.

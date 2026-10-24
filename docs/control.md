# Motor-control stack

The mission simulator runs wheel sensing and velocity control at 100 Hz,
state estimation and logging at 50 Hz, navigation at 20 Hz, and absolute
position fixes at 5 Hz. Current control runs at the plant integration rate.
The default 2 ms integration interval gives 500 Hz current updates.

Each wheel has an independent PI velocity loop and PI current loop.
The current controller is tuned to a 150 rad/s electrical bandwidth using
kp = L bandwidth and ki = R bandwidth. Resistance/back-EMF feedforward supplies
the nominal winding voltage. Bridge duties saturate at [-1, 1], and
back-calculation limits integral accumulation. Winding-temperature derating
reduces the velocity controller's current reference from its 8 A nominal limit.

Chassis commands become wheel-rate references through calibrated geometry.
Wheel-reference acceleration is limited before the velocity loops.
The commissioning experiment exercises the complete chain, including quantized
encoders, tire slip, bus sag and winding dynamics.

An encoder timeout exposed a feedback failure during development: a frozen
positive encoder rate could keep commanding negative current after the physical
wheel reversed. The simulator now falls back to an electrical speed estimate:
omega_wheel = (Vterminal - R i - L di/dt)/(k N).
It filters that estimate and uses it after encoder measurements become stale.
A separate timeout requests zero motion, so the fallback controls braking
rather than continuing the mission without wheel measurements.

The fallback assumes winding current and terminal voltage remain available.
Its discrete current derivative, winding resistance model and electrical
parameters are imperfect. It is a simulated diagnostic/control fallback,
not a hardware safety mechanism. The main plant is averaged and has no
semiconductor gate-disable, diode-commutation or contactor model.

The study API also provides exact held-input motor discretization, sampled PI
pole maps with optional one-cycle delay, motor frequency response, nonlinear
operating-point solving, and a six-state coupled-drive LQR. The LQR benchmark
uses full simulated state feedback and an approximate held-speed current
governor; its performance does not imply the mission estimator supplies all
those states or that current limits hold for arbitrary parameter mismatch.

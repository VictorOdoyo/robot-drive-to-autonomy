# Coupled electrical and mechanical model

The main plant uses two averaged permanent-magnet DC motors and H bridges.
It is a planar differential-drive robot with independent longitudinal tire
slip. It is not a switching-level inverter or a rigid-body contact solver.

The 18-state vector in `plant.py` is:

| Index | State | Unit |
|---|---|---|
| 0, 1 | left/right winding current | A |
| 2, 3 | left/right wheel angular velocity | rad/s |
| 4, 5 | body forward velocity / yaw rate | m/s, rad/s |
| 6, 7, 8 | world x / y / yaw | m, m, rad |
| 9, 10 | battery SOC / RC polarization voltage | 1, V |
| 11, 12, 13 | two winding temperatures / battery temperature | degrees C |
| 14, 15 | wheel angles | rad |
| 16, 17 | accumulated dissipation / external work | J |

For each winding, L di/dt = duty Vbus - (R(T) + Rbridge)i - k N omega.
The SI motor constant k is shared by back EMF and electromagnetic torque.
Wheel inertia includes rotor inertia multiplied by the square of gear ratio N.
The wheel equation is J_eq domega/dt = Nk i - bearing torque - radius F_tire.
The chassis receives the sum of tire forces and their differential moment.
World pose follows the body forward velocity and yaw rate.

Each tire uses a linear slip-speed stiffness with a Coulomb force ceiling.
Normal loads include quasi-static lateral load transfer. Wheel lift is exposed
in telemetry; pitch, roll, lateral sideslip, suspension and tire relaxation
are outside the model. Bearing/Coulomb drag represents drivetrain loss in the
coupled plant. The separate bidirectional fixed-efficiency gearbox and backlash
functions are component studies, not additional losses silently stacked here.

Battery SOC follows coulomb counting. The OCV curve is affine in SOC and its
integral defines chemical energy. A first-order Thevenin RC branch adds
polarization voltage and stored energy. Loaded bus voltage is the high-voltage
root of a quadratic including bridge input current and constant auxiliary power.

The energy audit includes chemical, polarization, winding, rotor/wheel and body
kinetic energy. Dissipation includes battery resistance, polarization resistance,
winding copper, bridge conduction, bearings, tire slip, rolling drag, air drag,
yaw drag and auxiliaries. Grade and external longitudinal forces enter a
separate signed work ledger. Thermal states receive selected losses; their
stored heat is deliberately outside this electrochemical/mechanical ledger.

RK4 integrates the coupled equations with a maximum 2 ms step. The test suite
checks analytical component limits, instantaneous power identities, driven
energy residuals and refinement of a physical transient. These are numerical
checks of stated equations, not validation against a real robot.
